#ifndef STORM_RDMA_COMMUNICATION_ENGINE_HPP
#define STORM_RDMA_COMMUNICATION_ENGINE_HPP
#ifdef STORM_WITH_MPI
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <sstream>
#include <boost/container/flat_set.hpp>
#include <mpi_utils/mpi_commands.hpp>
#include "CommunicationEngine.hpp"
#include "../../utils/GhostMap.hpp"
#include "../../utils/RankSync.hpp"
#include "../parallel/RankHandler2.hpp"
#include "../parallel/RegisteredSendBuffer.hpp"
#include "../parallel/ReallocationAgent.hpp"
namespace STORM
{
template<typename Grid>
std::vector<rank_t> GetNeighborList2(const Grid &tess, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &ghostsMap)
{
    size_t n = tess.GetPointNo();
    boost::container::flat_set<rank_t> ranks;

    std::vector<size_t> allNeighboringGhosts;
    for(size_t i = 0; i < n; i++)
    {
        for(size_t ghostIdx : tess.GetNeighbors(i))
        {
            if(ghostIdx >= n)
            {
                auto it = ghostsMap.find(ghostIdx);
                if(it != ghostsMap.end())
                {
                    rank_t ownerRank = (*it).second.first;
                    ranks.insert(ownerRank);
                }
            }
        }
    }

    return std::vector<rank_t>(ranks.cbegin(), ranks.cend());
}

template<class T, class Grid>
class RDMACommunicationEngine : public CommunicationEngine<T>
{

    using MCParticle = Particle<T>;
    using RankHandler_t = RankHandler2<T, Grid>;
    using RegisteredSendBuffer_t = RegisteredSendBuffer<MCParticle, RankHandler_t>;
    const Grid &grid;
    MonteCarloConfig config;
    MPI_Comm commWorld;
    rank_t rankWorld;
    rank_t sizeWorld;
    std::vector<MPI_Comm> communicators;
    std::vector<rank_t> ranksOrder, neighbors;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranksGhostMap;
    std::vector<RankHandler_t *> rankHandlers;
    std::vector<MCParticle> localParticles;
    std::shared_ptr<ReallocationAgent> reallocationAgent;
    RDMA_Type rdmaType;
    size_t transfersCounter = 0;
    std::vector<RegisteredSendBuffer_t> sendBuffers;
    std::vector<rank_t> sendBufferActiveRanks, readySendBufferRanks;
    std::vector<unsigned char> sendBufferActive, sendBufferListed, sendBufferReadyQueued;
    size_t readySendBufferCursor = 0, sendBufferPendingRanks = 0;
    size_t sendBufferPendingParticles = 0;
    size_t progressCycle = 0;

public:
    RDMACommunicationEngine(const Grid &grid, const MonteCarloConfig &config,
                            MPI_Comm comm, RDMA_Type type)
        : grid(grid), config(config), commWorld(comm), rdmaType(type)
    {
        MPI_Comm_rank(this->commWorld, &this->rankWorld);
        MPI_Comm_size(this->commWorld, &this->sizeWorld);

        // OFIContext construction exchanges endpoint addresses over commWorld and
        // must therefore happen collectively before neighbor-specific handlers are built.
        RMAFactory::Initialize(this->rdmaType, this->commWorld);

        this->ranksOrder = GetRanksOrder(this->commWorld);
        this->communicators = std::vector<MPI_Comm>(this->sizeWorld, MPI_COMM_NULL);

        this->rankHandlers = std::vector<RankHandler_t *>(this->sizeWorld, nullptr);
        this->sendBuffers.resize(this->sizeWorld);
        this->sendBufferActive.assign(this->sizeWorld, 0);
        this->sendBufferListed.assign(this->sizeWorld, 0);
        this->sendBufferReadyQueued.assign(this->sizeWorld, 0);

        auto reallocationFunction = [this](rank_t rank)
        {
            RankHandler_t *handler = this->rankHandlers[rank];
            double factor = this->config.bufferReallocationFactor;
            handler->Reallocate(factor);
        };

        auto localReallocationFunction = [this](rank_t rank, double factor) -> ReallocationMetadata
        {
            RankHandler_t *handler = this->rankHandlers[rank];
            if(handler == nullptr)
            {
                throw std::runtime_error("RDMACommunicationEngine: received async reallocation request for a missing handler");
            }
            return handler->LocalReallocate(factor);
        };

        auto metadataUpdateFunction = [this](rank_t rank, const ReallocationMetadata &metadata)
        {
            RankHandler_t *handler = this->rankHandlers[rank];
            if(handler == nullptr)
            {
                throw std::runtime_error("RDMACommunicationEngine: received async reallocation metadata for a missing handler");
            }
            handler->UpdatePeerRemoteInfo(metadata);
        };

        this->reallocationAgent = std::make_shared<ReallocationAgent>(this->commWorld, reallocationFunction,
                                                                      localReallocationFunction, metadataUpdateFunction);
        this->reallocationAgent->ConfigureAsyncPolling(
            this->config.asyncReallocationSendPollMinCycles,
            this->config.asyncReallocationIncomingPollActiveCycles,
            this->config.asyncReallocationIncomingPollIdleCycles,
            this->config.asyncReallocationMaxIncomingRequestsPerPoll);
    }

    ~RDMACommunicationEngine() override
    {
        if(!std::uncaught_exceptions())
        {
            ReleaseSendBufferRegistrations();
            FreeHandlers();
            ClearCommunicator();
        }
    }

    rank_t Rank() const override
    {
        return rankWorld;
    }

    rank_t Size() const override
    {
        return sizeWorld;
    }

    MPI_Comm Communicator() const override
    {
        return commWorld;
    }

    const std::vector<rank_t> &Neighbors() const override
    {
        return neighbors;
    }

    void Prepare() override
    {
        ranksGhostMap = GetGhostMap(grid);
        PrepareHandlers();
        ResetSendBuffers();
        transfersCounter = 0;
        progressCycle = 0;
    }

    void Progress() override
    {
        PumpRMAProgress();
        if(!UsesAsyncReallocation() || progressCycle++ % std::max<size_t>(1, this->config.asyncReallocationProgressMinCycles) == 0 ||
           reallocationAgent->HasPendingAsyncReallocations())
        {
            ProgressReallocations();
        }
        MakeRDMAProgress();
    }

    void Poll() override
    {
        PumpRMAProgress();
    }

    void Flush(bool idle) override
    {
        FlushSendBuffers(idle);
    }

    void FlushAll() override
    {
        FlushAllSendBuffers();
    }

    bool Pending() const override
    {
        if(!localParticles.empty() || !AllSendBuffersEmpty() || reallocationAgent->HasPendingAsyncReallocations())
        {
            return true;
        }
        for(RankHandler_t *handler : rankHandlers)
        {
            if(handler && !handler->LocalEmpty())
            {
                return true;
            }
        }
        return false;
    }

    void Send(rank_t destination, const MCParticle &particle) override
    {
        if(destination == this->rankWorld)
        {
            this->localParticles.push_back(particle);
            return;
        }
        RegisteredSendBuffer_t &buffer = GetSendBuffer(destination);
        const size_t previous = buffer.size();
        buffer.push_back(particle);
        NoteSendBufferGrowth(destination, previous, buffer, 1);
    }

    void AppendLocal(const std::vector<MCParticle> &particles) override
    {
        this->localParticles.insert(this->localParticles.end(), particles.begin(), particles.end());
    }

    void Detach(rank_t source, std::vector<MCParticle> &particles) override
    {
        if(source == this->rankWorld)
        {
            particles.swap(this->localParticles);
            return;
        }
        RankHandler_t *handler = this->rankHandlers[source];
        if(handler)
        {
            handler->DetachLocalParticles(particles);
        }
    }
    size_t LocalSize(rank_t source) const override
    {
        if(source == this->rankWorld)
        {
            return this->localParticles.size();
        }
        RankHandler_t *handler = this->rankHandlers[source];
        return handler ? handler->LocalSize() : 0;
    }

    void VisitLocal(const std::function<void(MCParticle &)> &visit) override
    {
        for(MCParticle &particle : this->localParticles)
        {
            visit(particle);
        }
        for(RankHandler_t *handler : this->rankHandlers)
        {
            if(handler)
            {
                handler->ForEachLocalParticle([&](MCParticle &particle, size_t)
                                              {
                                                  visit(particle);
                                              });
            }
        }
    }

    size_t MemoryBytes() const override
    {
        size_t bytes = this->localParticles.capacity() * sizeof(MCParticle);
        for(RankHandler_t *handler : rankHandlers)
        {
            if(handler)
            {
                bytes += handler->buffsize * sizeof(MCParticle);
            }
        }
        for(const RegisteredSendBuffer_t &buffer : sendBuffers)
        {
            bytes += buffer.capacity() * sizeof(MCParticle);
        }
        return bytes;
    }

    size_t Transfers() const override
    {
        return transfersCounter;
    }

    void ShrinkBuffers() override;

private:
    void ClearCommunicator();
    void FreeHandlers();
    bool UsesAsyncReallocation() const;
    void PumpRMAProgress();
    void ProgressReallocations();
    void MakeRDMAProgress();
    void PrepareHandlers();
    void RetireStaleHandlers();
    void ResetAllBuffers()
    {
        this->localParticles.clear();
        for(RankHandler_t *handler : rankHandlers)
        {
            if(handler)
            {
                handler->Reset();
            }
        }
    }
    RegisteredSendBuffer_t &GetSendBuffer(rank_t rank);
    void QueueReadySendBuffer(rank_t rank);
    void MarkSendBufferEmpty(rank_t rank);
    void ResetSendBuffers();
    void ReleaseSendBufferRegistrations();
    void NoteSendBufferGrowth(rank_t rank, size_t previousSize, const RegisteredSendBuffer_t &buffer, size_t addedParticles);
    void NoteSendBufferFlush(rank_t rank, size_t flushedParticles);
    void FlushSendBuffers(bool idle);
    void FlushAllSendBuffers();
    bool AllSendBuffersEmpty() const;
};
template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::ClearCommunicator()
{
    if(this->commWorld == MPI_COMM_NULL)
    {
        return;
    }

    if(this->communicators.size() < static_cast<size_t>(this->sizeWorld))
    {
        return;
    }

    auto clearRankComm = [this](rank_t rank)
    {
        MPI_Comm &comm = this->communicators[rank];
        if(comm == MPI_COMM_NULL)
        {
            return;
        }
        MPI_Comm_free(&comm);
    };

    ForEachRankSync(this->commWorld, this->ranksOrder, clearRankComm);

    this->commWorld = MPI_COMM_NULL;
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::FreeHandlers(void)
{
    auto freeHandler = [&](rank_t rank)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler != nullptr)
        {
            handler->Destroy();
            delete handler;
        }
        this->rankHandlers[rank] = nullptr;
    };

    ForEachRankSync(this->commWorld, this->ranksOrder, freeHandler);
}

template<typename T, typename Grid>
bool RDMACommunicationEngine<T, Grid>::UsesAsyncReallocation(void) const
{
    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler and handler->UsesAsyncReallocation())
        {
            return true;
        }
    }
    return false;
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::PumpRMAProgress(void)
{
    if(this->UsesAsyncReallocation())
    {
        RMAFactory::MakeProgress(this->rdmaType);
    }
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::ProgressReallocations(void)
{
    this->PumpRMAProgress();
    if(this->UsesAsyncReallocation())
    {
        this->reallocationAgent->ProgressAsyncReallocations();
    }
    else
    {
        this->reallocationAgent->HandleAllWaitingReallocations();
    }
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::MakeRDMAProgress(void)
{
    RMAFactory::MakeProgress(this->rdmaType);
}

#include "../parallel/RDMARankHandlerLifecycle.hpp"
#include "../parallel/RDMASendBufferProtocol.hpp"
} // namespace STORM
#endif
#endif
