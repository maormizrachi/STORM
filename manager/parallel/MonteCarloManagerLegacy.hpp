#ifndef MONTE_CARLO_MANAGER_LEGACY_HPP
#define MONTE_CARLO_MANAGER_LEGACY_HPP

#ifdef STORM_WITH_MPI

#include <cassert>
#include <mpi_utils/mpi_commands.hpp>
#include <mpi_utils/AmountManager.hpp>
#include "../../particle/Particle.hpp"
#include "../../physics/MonteCarloPhysics.hpp"
#include "../../population/PopulationControl.hpp"
#include "../../boundary/BoundaryCondition.hpp"
#include "../../utils/GhostMap.hpp"
#include "../../utils/RankSync.hpp"
#include "RankHandler.hpp"
#include "ReallocationAgent.hpp"
#ifdef MEMORY_DEBUG
#include "misc/memory_debug.hpp"
#else
#ifndef MEMORY_DEBUG_PRINT
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#endif
#endif
#include <array>
#include <boost/container/flat_set.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#include "../MonteCarloConfig.hpp"
#include "../../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

template<typename Grid>
std::vector<rank_t> GetNeighborList(const Grid &tess, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &ghostsMap)
{
    size_t N = tess.GetPointNo();
    boost::container::flat_set<rank_t> ranks;

    std::vector<size_t> allNeighboringGhosts;
    for(size_t i = 0; i < N; i++)
    {
        for(size_t ghostIdx : tess.GetNeighbors(i))
        {
            if(ghostIdx >= N)
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


template<typename T, typename Grid>
class MonteCarloManagerLegacy
{
    using MCParticle = MonteCarloParticle<T, Grid>;
    using RankHandler_t = RankHandler<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        size_t leavingCount = 0;
    };

    MonteCarloManagerLegacy(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MonteCarloConfig &config = MonteCarloConfig(),
                    const MPI_Comm &comm = MPI_COMM_WORLD, RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA);

    ~MonteCarloManagerLegacy();

    void ClearCommunicator(void);

    void TransferParticles(rank_t rankBuffer, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num);

    void TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks);

    inline size_t GetStepCounter(void) const{return this->allStepsCounter;};

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const {return this->cellsStepsCounters;}

    inline std::vector<size_t> &GetCellsStepsCounters(void) {return this->cellsStepsCounters;}

    inline const std::vector<size_t> &GetCellsParticleCounters(void) const {return this->cellsParticleCounters;}

    inline size_t GetStartParticleCount(void) const {return this->startParticleCount;}

    inline size_t GetEndParticleCount(void) const {return this->endParticleCount;}

    inline size_t GetInitialParticleCount(void) const {return this->initialParticleCount;}

    inline double GetPureComputeTime(void) const {
        return 0;
    }

    inline const std::vector<size_t> &GetBeginningParticleCount(void) const {return this->beginningParticleCount;}

    inline std::vector<size_t> &GetBeginningParticleCount(void) {return this->beginningParticleCount;}

    inline size_t GetHandlerMemoryBytes(void) const {return this->handlerMemoryBytes;}

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, dt_t fullDt);

    class Tracker
    {
    public:
        Tracker(const MPI_Comm &comm);

        void Reset(void);

        #ifdef STORM_WITH_MPI
            std::vector<MCParticle> GetLocalTrackParticleRoute(size_t id) const;
        #endif // STORM_WITH_MPI

        std::vector<MCParticle> GetTrackParticleRoute(size_t id) const;

        void ReportParticle(MCParticle &particle);

    private:
        MPI_Comm comm;
        boost::container::flat_map<size_t, std::vector<MCParticle>> track;
    };

    inline const Tracker &getTracker(void){return this->tracker;};

    inline void resetTracker(void){this->tracker.Reset();};

private:
    const Grid &grid;
    MonteCarloConfig config;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    // std::shared_ptr<ProgressCounter> progress;
    typename AmountManager::counter_t localDecrementAmount;
    std::vector<MPI_Comm> communicators;
    std::vector<rank_t> ranksOrder;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler_t*> rankHandlers;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    std::shared_ptr<ReallocationAgent> reallocationAgent;
    size_t myIDCounter;
    size_t currentStep;
    size_t allStepsCounter;
    size_t transfersCounter;
    std::vector<rank_t> neighbors;
    std::vector<size_t> cellsStepsCounters;
    std::vector<size_t> cellsParticleCounters;
    size_t iteration;
    size_t dynamicallyAdded;
    RDMA_Type rdma_type;
    size_t lastBuildGeneration;
    size_t startParticleCount = 0;
    size_t endParticleCount = 0;
    size_t initialParticleCount = 0;
    std::vector<size_t> beginningParticleCount;
    size_t handlerMemoryBytes = 0;

    std::vector<std::vector<MCParticle>> sendBuffers;
    std::vector<rank_t> sendBufferActiveRanks;
    std::vector<rank_t> readySendBufferRanks;
    std::vector<unsigned char> sendBufferActive;
    std::vector<unsigned char> sendBufferListed;
    std::vector<unsigned char> sendBufferReadyQueued;
    std::vector<rank_t> activeRanks;
    std::vector<rank_t> nextActiveRanks;
    size_t readySendBufferCursor;
    size_t sendBufferPendingRanks;
    size_t sendBufferCycleCounter;
    size_t sendBufferPendingParticles;
    size_t activeRankScanCursor;
    size_t activeRankScanRemaining;

    bool HandleAll(MonteCarloStepFinalData &stepData);

    void PutSelfParticles(std::vector<MCParticle> &&particles);

    void PrepareHandlers(void);

    void FreeHandlers(void);

    void AddParticles(const std::vector<MCParticle> &particles);

    void ResetAllBuffers(void);

    void ShrinkBuffers(void);

    std::vector<MCParticle> &GetSendBuffer(rank_t rank);

    void QueueReadySendBuffer(rank_t rank);

    void MarkSendBufferEmpty(rank_t rank);

    void ResetSendBuffers(void);

    void NoteSendBufferGrowth(rank_t rank, size_t previousSize, const std::vector<MCParticle> &buffer, size_t addedParticles);

    void NoteSendBufferFlush(rank_t rank, size_t flushedParticles);

    bool UsesAsyncReallocation(void) const;

    void PumpRMAProgress(void);

    void ProgressReallocations(void);

    void FlushSendBuffers(bool flushSmallBuffers);

    void FlushAllSendBuffers(void);

    bool AllSendBuffersEmpty(void) const;

    void PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum);

};

template<typename T, typename Grid>
MonteCarloManagerLegacy<T, Grid>::MonteCarloManagerLegacy(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition, const MonteCarloConfig &config, const MPI_Comm &comm, RDMA_Type rdma_type):
    grid(grid), config(config), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(MPI_COMM_NULL), tracker(comm), rdma_type(rdma_type)
{
    this->myIDCounter = 0;
    this->currentStep = 0;
    // this->progress = std::make_shared<ProgressCounter>(comm);
    this->comm_world = comm;
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->ranksOrder = GetRanksOrder(this->comm_world);
    this->communicators = std::vector<MPI_Comm>(this->size_world, MPI_COMM_NULL);

    this->rankHandlers = std::vector<RankHandler_t*>(this->size_world, nullptr);
    this->sendBuffers.resize(this->size_world);
    this->sendBufferActive.assign(this->size_world, 0);
    this->sendBufferListed.assign(this->size_world, 0);
    this->sendBufferReadyQueued.assign(this->size_world, 0);

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
            throw std::runtime_error("MonteCarloManagerLegacy: received async reallocation request for a missing handler");
        }
        return handler->LocalReallocate(factor);
    };

    auto metadataUpdateFunction = [this](rank_t rank, const ReallocationMetadata &metadata)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler == nullptr)
        {
            throw std::runtime_error("MonteCarloManagerLegacy: received async reallocation metadata for a missing handler");
        }
        handler->UpdatePeerRemoteInfo(metadata);
    };

    this->reallocationAgent = std::make_shared<ReallocationAgent>(this->comm_world, reallocationFunction,
                                                                  localReallocationFunction, metadataUpdateFunction);
    this->reallocationAgent->ConfigureAsyncPolling(
        this->config.asyncReallocationSendPollMinCycles,
        this->config.asyncReallocationIncomingPollActiveCycles,
        this->config.asyncReallocationIncomingPollIdleCycles,
        this->config.asyncReallocationMaxIncomingRequestsPerPoll);

    if(this->rank_world == 0)
    {
        std::cout << "Done initializing MonteCarloManagerLegacy" << std::endl;
    }
    this->cellsStepsCounters.assign(this->grid.GetPointNo(), 0);
    this->cellsParticleCounters.assign(this->grid.GetPointNo(), 0);
    this->lastBuildGeneration = std::numeric_limits<size_t>::max();
    this->readySendBufferCursor = 0;
    this->sendBufferPendingRanks = 0;
    this->sendBufferCycleCounter = 0;
    this->sendBufferPendingParticles = 0;
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::ClearCommunicator()
{
    if(this->comm_world == MPI_COMM_NULL)
    {
        return;
    }

    if(this->communicators.size() < this->size_world)
    {
        return;
    }

    auto clearRankComm = [this](rank_t _rank)
    {
        MPI_Comm &comm = this->communicators[_rank];
        if(comm == MPI_COMM_NULL)
        {
            return;
        }
        MPI_Comm_free(&comm);
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, clearRankComm);

    this->comm_world = MPI_COMM_NULL;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::FreeHandlers(void)
{
    auto freeHandler = [&](rank_t _rank)
    {
        RankHandler_t *handler = this->rankHandlers[_rank];
        if(handler != nullptr)
        {
            handler->Destroy();
            delete handler;
        }
        this->rankHandlers[_rank] = nullptr;
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, freeHandler);
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    using index_t = typename RankHandler_t::index_t;
    if(particles.empty())
    {
        return;
    }

    RankHandler_t *myHandler = this->rankHandlers[this->rank_world];

    // std::cout << "In add particles, handler size is " << myHandler->buffsize << ", particles size to add is " << particles.size() << std::endl;

    if(myHandler->av_length < particles.size())
    {
        double factor = std::max<double>(this->config.bufferReallocationFactor, std::ceil(static_cast<double>(particles.size() + myHandler->buffsize) / static_cast<double>(myHandler->buffsize)));
        myHandler->Reallocate(factor);
        assert(myHandler->av_length >= particles.size());
    }

    // set particles
    // update 'to handle' and 'available' lists accordingly
    index_t particlesNum = particles.size();
    myHandler->av_length -= particlesNum;
    index_t *avIndices = myHandler->av + (myHandler->av_length);
    index_t *thIndices = myHandler->th + (myHandler->th_length);
    myHandler->th_length += particlesNum;
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    for(size_t i = 0; i < particlesNum; i++)
    {
        index_t idx = avIndices[i];
        // copy particle
        MCParticle *particle = myHandler->particles + idx;
        std::memcpy(particle, &particles[i], sizeof(MCParticle));
        // set to handle
        thIndices[i] = idx;
        // set ID
        particle->rank = this->rank_world;
        particle->id = firstID + i;

        #ifdef STORM_DEBUG
            particle->checkedHere = true;
            particle->nextRank = std::numeric_limits<rank_t>::max();
            particle->removedFromRank = false;
            particle->sentByRank = std::numeric_limits<rank_t>::max();
            particle->lastSeen = 0;
        #endif // STORM_DEBUG

        #ifdef STORM_DEBUG
        if(not this->grid.IsPointInCell(particle->location, particle->cellIndex))
        {
            const T &declaredCell = this->grid.GetMeshPoint(particle->cellIndex);
            size_t containingIdx = this->grid.GetContainingCell(particle->location);
            const T &containingCell = this->grid.GetMeshPoint(containingIdx);
            STORMError eo("MonteCarloManagerLegacy<T, Grid>::AddParticles");
            eo.addEntry("rank", this->rank_world);
            eo.addEntry("Particle", *particle);
            eo.addEntry("Declared Cell Index", particle->cellIndex);
            eo.addEntry("Declared Cell", declaredCell);
            eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle->location));
            eo.addEntry("Real Containing Cell Index", containingIdx);
            eo.addEntry("Real Containing Cell", containingCell);
            eo.addEntry("Real Cell - Distance", abs(containingCell - particle->location));
            throw eo;
        }
        #endif // STORM_DEBUG
    }

    this->localDecrementAmount -= static_cast<typename AmountManager::counter_t>(particlesNum);
    // std::cout << "Done add particles" << std::endl;
}

template<typename T, typename Grid>
MonteCarloManagerLegacy<T, Grid>::~MonteCarloManagerLegacy()
{
    if(not std::uncaught_exceptions())
    {
        this->FreeHandlers();
        this->ClearCommunicator();
    }
}

template<typename T, typename Grid>
bool MonteCarloManagerLegacy<T, Grid>::UsesAsyncReallocation(void) const
{
    RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                             ? RMAFactory::ResolveAutoRDMA()
                             : this->rdma_type;
    return resolved == RDMA_Type::OFI_RDMA or resolved == RDMA_Type::IBV_RDMA;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::PumpRMAProgress(void)
{
    if(this->UsesAsyncReallocation())
    {
        RMAFactory::MakeProgress(this->rdma_type);
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::ProgressReallocations(void)
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
MonteCarloManagerLegacy<T, Grid>::Tracker::Tracker(const MPI_Comm &comm): comm(comm)
{}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::Tracker::Reset(void)
{
    this->track.clear();
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::Tracker::ReportParticle(MCParticle &particle)
{
    if(this->track.find(particle.id) == this->track.end())
    {
        this->track[particle.id] = std::vector<MCParticle>();
    }
    this->track[particle.id].push_back(particle);
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerLegacy<T, Grid>::MCParticle> MonteCarloManagerLegacy<T, Grid>::Tracker::GetLocalTrackParticleRoute(size_t id) const
{
    auto it = this->track.find(id);
    if(it == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return it->second;
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerLegacy<T, Grid>::MCParticle> MonteCarloManagerLegacy<T, Grid>::Tracker::GetTrackParticleRoute(size_t id) const
{
    std::vector<MCParticle> local = this->GetLocalTrackParticleRoute(id);
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
    using index_t = typename RankHandler_t::index_t;

    #ifdef STORM_DEBUG
    boost::container::flat_set<std::pair<rank_t, size_t>> particlesSet;
    for(const MCParticle &particle : particles)
    {
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            continue;
        }
        std::pair<rank_t, size_t> particleSetKey = {particle.rank, particle.id};
        if(particlesSet.find(particleSetKey) != particlesSet.end())
        {
            STORMError eo("Particle with the same ID is being added to the same rank twice");
            eo.addEntry("Particle", particle);
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("ID", particle.id);
            throw eo;
        }
        particlesSet.insert(particleSetKey);
    }
    #endif // STORM_DEBUG

    size_t particlesNum = particles.size();
    if(particlesNum == 0)
    {
        return;
    }

    RankHandler_t *handler = this->rankHandlers[this->rank_world];

    if(static_cast<size_t>(handler->av_length) < particlesNum)
    {
        double factor = std::max<double>(this->config.bufferReallocationFactor, std::ceil(static_cast<double>(particlesNum) / static_cast<double>(handler->buffsize)));
        handler->Reallocate(factor);
    }

    handler->av_length -= static_cast<int>(particlesNum);
    index_t *av_indices = handler->av + handler->av_length;
    int oldTHLength = handler->th_length;
    handler->th_length += particlesNum;

    for(size_t i = 0; i < particlesNum; i++)
    {
        size_t particleIdx = av_indices[i];
        handler->th[oldTHLength + i] = particleIdx;
        std::memcpy(handler->particles + particleIdx, &particles[i], sizeof(MCParticle));
        MCParticle &particle = handler->particles[particleIdx];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
    }

    // don't waste memory - remove current particles from the input vector
    std::vector<MCParticle> empty;
    particles.swap(empty);
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::TransferParticles(rank_t fromRank, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    #ifdef STORM_DEBUG
        boost::container::flat_map<size_t, rank_t> sentAndToWhom;
    #endif // STORM_DEBUG
    RankHandler_t *currRankHandler = this->rankHandlers[fromRank];

    for(size_t i = 0; i < num; i++)
    {
        const size_t &indexInToHandle = indicesInToHandle[i];
        const rank_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        size_t particleIdx = currRankHandler->th[indexInToHandle];
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<MCParticle>();
        }

        #ifdef STORM_DEBUG
            if(sentAndToWhom.find(particleIdx) == sentAndToWhom.end())
            {
                sentAndToWhom[particleIdx] = toRank;
            }
            else
            {
                STORMError eo("Particle is being sent to multiple ranks");
                eo.addEntry("Particle Index", particleIdx);
                eo.addEntry("Particle", currRankHandler->particles[particleIdx]);
                eo.addEntry("I am rank", this->rank_world);
                eo.addEntry("From Rank Buffer", fromRank);
                eo.addEntry("Already Sent To", sentAndToWhom[particleIdx]);
                eo.addEntry("Now Sending To", toRank);
                throw eo;
            }
        #endif // STORM_DEBUG
        MCParticle &particle = currRankHandler->particles[particleIdx];
        particle.sent = false; // reset

        // std::cout << "Rank " << this->rank_world << " transfers particle TH = " << indexInToHandle << ", particle index " << particleIdx << " (particle: " << particle << ") to rank " << toRank << std::endl;

        if(toRank == this->rank_world)
        {
            STORMError eo("Trying to transfer particle to the same rank");
            eo.addEntry("Particle", particle);
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }
        #ifdef STORM_DEBUG
        if(std::find_if(rankToParticles[toRank].begin(), rankToParticles[toRank].end(),
                        [&particle](const MCParticle &p) { return p == particle; }) != rankToParticles[toRank].end())
        {
            STORMError eo("Particle with the same ID is being sent to the same rank twice");
            eo.addEntry("Index In Transfer Queue", i);
            eo.addEntry("Particle Index", particleIdx);
            eo.addEntry("Particle", particle);
            for(size_t j = 0; j < i; j++)
            {
                size_t indexInToHandle2 = indicesInToHandle[j];
                rank_t toRank2 = transferRanks[j];
                size_t particle2Idx = currRankHandler->th[indexInToHandle2];
                const MCParticle &particle2 = currRankHandler->particles[particle2Idx];
                if(toRank2 == toRank and particle2 == particle)
                {
                    eo.addEntry("Already Appeared In Index", j);
                    eo.addEntry("Particle2", particle2);
                    eo.addEntry("Particle 2 Index", particle2Idx);
                    break;
                }
            }
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }
        #endif // STORM_DEBUG

        rankToParticles[toRank].push_back(particle);

        #ifdef STORM_DEBUG
        if(toRank != particle.nextRank)
        {
            STORMError eo("Particle will not be sent to the expected rank #1");
            eo.addEntry("Particle", particle);
            eo.addEntry("Origin", particle.sentByRank);
            eo.addEntry("Expected Rank", toRank);
            eo.addEntry("Next Rank", particle.nextRank);
            throw eo;
        }
        #endif // STORM_DEBUG
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef STORM_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            STORMError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // STORM_DEBUG
        bool transferred = remoteHandler->TransferParticles(particles);
        if(not transferred)
        {
            std::vector<MCParticle> &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.insert(buffer.end(), particles.begin(), particles.end());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    #ifdef STORM_DEBUG
        boost::container::flat_map<std::pair<rank_t, size_t>, rank_t> sentAndToWhom;
    #endif // STORM_DEBUG

    assert(rankBuffers.size() == indicesInToHandle.size());

    size_t numRanks = rankBuffers.size();
    for(size_t i = 0; i < numRanks; i++)
    {
        const rank_t &fromRank = rankBuffers[i];
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myTHIndices = indicesInToHandle[i];
        size_t numToHandle = myTHIndices.size();
        const std::vector<rank_t> &myTransferRanks = transferRanks[i];

        for(size_t j = 0; j < numToHandle; j++)
        {
            const size_t &indexInToHandle = myTHIndices[j];
            const rank_t &toRank = myTransferRanks[j];

            assert(toRank != this->rank_world); // can't send to self
            size_t particleIdx = currRankHandler->th[indexInToHandle];
            auto it = rankToParticles.find(toRank);
            if(it == rankToParticles.end())
            {
                rankToParticles[toRank] = std::vector<MCParticle>();
            }

            #ifdef STORM_DEBUG
                std::pair<rank_t, size_t> particleKey = {fromRank, particleIdx};
                if(sentAndToWhom.find(particleKey) == sentAndToWhom.end())
                {
                    sentAndToWhom[particleKey] = toRank;
                }
                else
                {
                    STORMError eo("Particle is being sent to multiple ranks");
                    eo.addEntry("Particle Index", particleIdx);
                    eo.addEntry("Particle", currRankHandler->particles[particleIdx]);
                    eo.addEntry("I am rank", this->rank_world);
                    eo.addEntry("From Rank Buffer", fromRank);
                    eo.addEntry("Already Sent To", sentAndToWhom[particleKey]);
                    eo.addEntry("Now Sending To", toRank);
                    throw eo;
                }
            #endif // STORM_DEBUG
            MCParticle &particle = currRankHandler->particles[particleIdx];
            particle.sent = false; // reset

            // std::cout << "Rank " << this->rank_world << " transfers particle TH = " << indexInToHandle << ", particle index " << particleIdx << " (particle: " << particle << ") to rank " << toRank << std::endl;

            if(toRank == this->rank_world)
            {
                STORMError eo("Trying to transfer particle to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }
            #ifdef STORM_DEBUG
            if(std::find_if(rankToParticles[toRank].begin(), rankToParticles[toRank].end(),
                            [&particle](const MCParticle &p) { return p == particle; }) != rankToParticles[toRank].end())
            {
                STORMError eo("Particle with the same ID is being sent to the same rank twice");
                eo.addEntry("Index In Transfer Queue", i);
                eo.addEntry("Particle Index", particleIdx);
                eo.addEntry("Particle", particle);
                for(size_t j = 0; j < i; j++)
                {
                    size_t indexInToHandle2 = myTHIndices[j];
                    rank_t toRank2 = myTransferRanks[j];
                    size_t particle2Idx = currRankHandler->th[indexInToHandle2];
                    const MCParticle &particle2 = currRankHandler->particles[particle2Idx];
                    if(toRank2 == toRank and particle2 == particle)
                    {
                        eo.addEntry("Already Appeared In Index", j);
                        eo.addEntry("Particle2", particle2);
                        eo.addEntry("Particle 2 Index", particle2Idx);
                        break;
                    }
                }
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }
            #endif // STORM_DEBUG

            rankToParticles[toRank].push_back(particle);

            #ifdef STORM_DEBUG
            if(toRank != particle.nextRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #1");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
            #endif // STORM_DEBUG
        }
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef STORM_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            STORMError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // STORM_DEBUG
        bool transferred = remoteHandler->TransferParticles(particles);
        if(not transferred)
        {
            std::vector<MCParticle> &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.insert(buffer.end(), particles.begin(), particles.end());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid>
bool MonteCarloManagerLegacy<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<std::vector<size_t>> removeParticlesVec;
    static std::vector<std::vector<rank_t>> transferToRanks;
    static std::vector<std::vector<size_t>> transferParticlesVec;
    static std::vector<MCParticle> particlesToAdd;
    static size_t progressStepCounter;
    std::vector<rank_t> &active_ranks = this->activeRanks;
    std::vector<rank_t> &next_active_ranks = this->nextActiveRanks;

    // static std::uniform_real_distribution<double> dist(0, 1);
    // static std::mt19937 re(this->rank_world);

    next_active_ranks.clear();
    bool completedNeighborSweep = true;
    if(active_ranks.empty())
    {
        // std::cout << "active_ranks is empty" << std::endl;
        const int PREFETCH_DISTANCE = 3;
        size_t N = this->neighbors.size();
        if(N > 0 and this->activeRankScanRemaining == 0)
        {
            this->activeRankScanRemaining = N;
            this->activeRankScanCursor %= N;
        }
        size_t scanCount = (N == 0) ? 0 :
            std::min(this->activeRankScanRemaining,
                     std::min(N, std::max<size_t>(1, this->config.activeRankScanChunk)));

        for (size_t scanOffset = 0; scanOffset < scanCount; ++scanOffset)
        {
            size_t i = (this->activeRankScanCursor + scanOffset) % N;
            // Prefetch future data to hide memory latency
            if (scanOffset + PREFETCH_DISTANCE < scanCount)
            {
                size_t futureIndex = (this->activeRankScanCursor + scanOffset + PREFETCH_DISTANCE) % N;
                rank_t future_rank = this->neighbors[futureIndex];

                // Prefetch the RankHandler *object (heap-allocated, likely scattered)
                RankHandler_t *future_handler = this->rankHandlers[future_rank];
                __builtin_prefetch(future_handler, 0, 1);                  // bring RankHandler into cache
                __builtin_prefetch((const void*) &(future_handler->th_length), 0, 1);      // bring th_length into cache
            }

            // Access current handler
            rank_t _rank = this->neighbors[i];
            RankHandler_t *handler = this->rankHandlers[_rank];

            int len = handler->th_length;

            if(len)
            {
                if(len < 0 || len > static_cast<int>(handler->buffsize))
                {
                    std::cerr << "HandleAll: corrupt th_length=" << len
                              << " for neighbor rank " << _rank << " (my rank " << this->rank_world
                              << ", buffsize=" << handler->buffsize
                              << ", iteration=" << this->iteration << ")" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                active_ranks.push_back(_rank);
            }
        }
        if(N > 0)
        {
            this->activeRankScanCursor = (this->activeRankScanCursor + scanCount) % N;
            assert(this->activeRankScanRemaining >= scanCount);
            this->activeRankScanRemaining -= scanCount;
            completedNeighborSweep = (this->activeRankScanRemaining == 0);
        }
        {
            RankHandler_t *handler = this->rankHandlers[this->rank_world];
            if(handler->th_length > 0)
            {
                active_ranks.push_back(this->rank_world);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();

    auto eliminateParticle = [&](size_t rankIndex, size_t particleTH)
    {
        removeParticlesVec[rankIndex].push_back(particleTH);
    };

    auto transferParticle = [&](size_t rankIndex, size_t particleTH, rank_t toRank)
    {
        assert(toRank != this->rank_world); // can't send to self
        transferToRanks[rankIndex].push_back(toRank);
        transferParticlesVec[rankIndex].push_back(particleTH);
        eliminateParticle(rankIndex, particleTH);
    };

    auto removeParticle = [&](size_t rankIndex, size_t particleTH)
    {
        eliminateParticle(rankIndex, particleTH);
        this->localDecrementAmount += 1;
    };

    transferParticlesVec.clear();
    transferToRanks.clear();
    removeParticlesVec.clear();


    for(size_t index = 0; index < activeRanksNum; index++)
    {
        rank_t _rank = active_ranks[index];
        RankHandler_t *handler = this->rankHandlers[_rank];
        int length = handler->th_length;

        transferParticlesVec.emplace_back();
        transferToRanks.emplace_back();
        removeParticlesVec.emplace_back();

        for(int i = 0; i < length; i++)
        {
            if(i >= static_cast<int>(handler->buffsize))
            {
                std::cerr << "HandleAll: th index " << i << " >= buffsize " << handler->buffsize
                          << " for rank " << _rank << " (my rank " << this->rank_world
                          << ", th_length=" << length << ", iteration=" << this->iteration << ")" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            size_t particleIndex = handler->th[i];
            if(particleIndex >= handler->buffsize)
            {
                std::cerr << "HandleAll: particleIndex " << particleIndex << " >= buffsize " << handler->buffsize
                          << " for rank " << _rank << " (my rank " << this->rank_world
                          << ", th[" << i << "]=" << particleIndex
                          << ", th_length=" << length << ", iteration=" << this->iteration << ")" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MCParticle &particle = handler->particles[particleIndex];
            bool debug = false; // (particle.rank == 5 and particle.id == 518987);

            try
            {
                #ifdef STORM_DEBUG
                if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
                {
                    STORMError eo("Particle was already handled in this iteration");
                    eo.addEntry("My Rank", this->rank_world);
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Iteration", this->iteration);
                    eo.addEntry("In Rank Buffer (1)", particle.lastSeenRankBuf);
                    eo.addEntry("In TH Index (1)", particle.lastSeenIndex);
                    eo.addEntry("In Rank Buffer (2)", _rank);
                    eo.addEntry("In TH Index (2)", i);
                    throw eo;
                }
                particle.lastSeen = this->iteration;
                particle.lastSeenRankBuf = _rank;
                particle.lastSeenRank = this->rank_world;
                particle.lastSeenIndex = i;
                #endif // STORM_DEBUG

                isEmpty = false;
                while(true)
                {
                    ++progressStepCounter;
                    if((progressStepCounter & 0x3FF) == 0)
                    {
                        this->PumpRMAProgress();
                    }
                    // TODO: shouldn't be, there's a bug
                    // if(particle.sent)
                    // {
                    //     continue;
                    // }

                    // std::cout << "Rank " << this->rank_world << " handles TH = " << i << ", which is index " << particleIndex << ", particle: " << particle << std::endl;

                    const size_t traceStep = particle.steps;
                    if(particle.on_track)
                    {
                        MCParticle trackedParticle = particle;
                        trackedParticle.steps = traceStep * 2;
                        this->tracker.ReportParticle(trackedParticle);
                    }
                    particle.steps++;
                    this->cellsStepsCounters[particle.cellIndex]++;

                    // std::cout << "Rank " << this->rank_world << " handles particle " << particle.id << " of rank " << particle.rank << ", step " << particle.steps << std::endl;

                    #ifdef STORM_DEBUG
                    if(particle.cellIndex >= this->Ncells)
                    {
                        STORMError eo("Particle has invalid cell index (ghost)");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Cell Index", particle.cellIndex);
                        eo.addEntry("Rank", this->rank_world);
                        eo.addEntry("Buffer of Rank", _rank);
                        throw eo;
                    }
                    if(particle.removedFromRank)
                    {
                        continue;
                        STORMError eo("Particle was removed from rank, but still in the list");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Rank", this->rank_world);
                        eo.addEntry("Buffer of Rank", _rank);
                        throw eo;
                    }
                    if(not particle.checkedHere)
                    {
                        if(particle.nextRank != this->rank_world)
                        {
                            // particle is in the right cell, but not in the right place
                            STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Origin", particle.sentByRank);
                            eo.addEntry("Particle Previous Location", particle.previousLocation);
                            eo.addEntry("Cell Index In Origin (Before Movement)", particle.cellIndexInPrevRank);
                            eo.addEntry("Expected", particle.nextRank);
                            eo.addEntry("Got (me)", this->rank_world);
                            eo.addEntry("The Particle Index In Last Rank", particle.particleIndexInLastRank);
                            eo.addEntry("Particle Index In This Rank", particleIndex);
                            eo.addEntry("The Particle TH In Last Rank", particle.particleTHInLastRank);
                            eo.addEntry("Particle TH In This Rank", i);
                            eo.addEntry("New Cell Index Should Be", particle.cellIndex);
                            eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                            throw eo;
                        }
                        particle.checkedHere = true;
                        particle.nextRank = std::numeric_limits<rank_t>::max();
                        particle.removedFromRank = false;
                        particle.sentByRank = std::numeric_limits<rank_t>::max();
                    }
                    if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
                    {
                        const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                        size_t containingIdx = this->grid.GetContainingCell(particle.location);
                        const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                        if(containingIdx != particle.cellIndex)
                        {
                            if(not this->grid.IsPointInCell(particle.location, containingIdx))
                            {
                                // particle is in the right cell, but not in the right place
                                STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                                eo.addEntry("My Rank", this->rank_world);
                                eo.addEntry("Transferred From Rank", _rank);
                                eo.addEntry("Particle", particle);
                                eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                                eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                                eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                                eo.addEntry("Declared Cell Index", particle.cellIndex);
                                eo.addEntry("Declared Cell", declaredCell);
                                eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                                eo.addEntry("Real Containing Cell Index", containingIdx);
                                eo.addEntry("Real Containing Cell", containingCell);
                                eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                                eo.addEntry("Particle Previous Location", particle.previousLocation);
                                eo.addEntry("Particle Previous Cell Index", particle.cellIndexInPrevRank);
                                throw eo;
                            }
                        }
                        if(abs(abs(declaredCell - particle.location) - abs(containingCell - particle.location)) >= 1e-12)
                        {
                            STORMError eo("Particle is in Wrong Location After Transfer");
                            eo.addEntry("My Rank", this->rank_world);
                            eo.addEntry("Transferred From Rank", _rank);
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                            eo.addEntry("Particle Previous Location", particle.previousLocation);
                            eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                            eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                            eo.addEntry("Declared Cell Index", particle.cellIndex);
                            eo.addEntry("Declared Cell", declaredCell);
                            eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                            eo.addEntry("Real Containing Cell Index", containingIdx);
                            eo.addEntry("Real Containing Cell", containingCell);
                            eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                            for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                            {
                                eo.addEntry("Face Index", faceIdx);
                                eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                                eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                                eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                            }
                            throw eo;
                        }
                    }

                    prevLoc = particle.location;
                    particle.previousLocation = particle.location;
                    #endif // STORM_DEBUG

                    if(particle.sent)
                    {
                        particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                            MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                        particle.sent = false;
                    }

                    if(debug)
                    {
                        std::cout << "Before running particle step, particle is " << particle << std::endl;
                    }

                    MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle, particlesToAdd);

                    if(particle.on_track)
                    {
                        MCParticle trackedParticle = particle;
                        trackedParticle.steps = traceStep * 2 + 1;
                        this->tracker.ReportParticle(trackedParticle);
                    }

                    #ifdef STORM_WITH_TRACING_HISTORY
                        particle.recordHistory(particle.cellIndex, static_cast<int>(this->rank_world), static_cast<int>(functionality.change));
                    #endif // STORM_WITH_TRACING_HISTORY

                    // std::cout << "Handling particle " << particle << ", functionality is " << functionality.change << std::endl;
                    if(debug)
                    {
                        std::cout << "Particle " << particle << ", functionality is " << functionality.change << std::endl;
                    }

                    if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
                    {
                        size_t nextCellIndex = functionality.nextCellIndex;

                        assert(nextCellIndex != particle.cellIndex);
                        assert(particle.timeLeft >= 0);

                        if(__builtin_expect(nextCellIndex < this->Ncells, 1))
                        {
                            // local neighbor
                            #ifdef STORM_DEBUG
                            size_t previousCell = particle.cellIndex;
                            #endif // STORM_DEBUG
                            particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                            particle.cellIndex = nextCellIndex;
                            #ifdef STORM_DEBUG
                            if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
                            {
                                const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                                size_t containingIdx = this->grid.GetContainingCell(particle.location);
                                const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                                STORMError eo("Particle is in Wrong Location");
                                eo.addEntry("rank", this->rank_world);
                                eo.addEntry("Particle", particle);
                                eo.addEntry("Previous Cell Index", previousCell);
                                eo.addEntry("Previous Cell", this->grid.GetMeshPoint(previousCell));
                                eo.addEntry("Previous Location", prevLoc);
                                eo.addEntry("Last location is in previous cell?", this->grid.IsPointInCell(prevLoc, previousCell));
                                eo.addEntry("Declared Cell Index", particle.cellIndex);
                                eo.addEntry("Declared Cell", declaredCell);
                                eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                                eo.addEntry("Real Containing Cell Index", containingIdx);
                                eo.addEntry("Real Containing Cell", containingCell);
                                eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                                for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                                {
                                    eo.addEntry("Face Index", faceIdx);
                                    eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                                    eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                                    eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                                }
                                throw eo;
                            }
                            #endif // STORM_DEBUG
                        }
                        else
                        {
                            // a ghost point, check rank and index in rank
                            auto it = ranks_ghost_map.find(nextCellIndex);
                            if(it == ranks_ghost_map.end())
                            {
                                // leaving domain
                                #ifdef STORM_WITH_TRACING_HISTORY
                                    T preReflectLoc = particle.location;
                                    T preReflectVel = particle.velocity;
                                #endif // STORM_WITH_TRACING_HISTORY
                                MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                                if(debug)
                                {
                                    std::cout << "Particle " << particle << ", leaving domain. status from bounday condition: " << status << std::endl;
                                }
                                if(status == MonteCarloParticleStatus::REFLECT)
                                {
                                    #ifdef STORM_WITH_TRACING_HISTORY
                                        particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
                                    #endif // STORM_WITH_TRACING_HISTORY
                                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                        MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                                }
                                else if(status == MonteCarloParticleStatus::REMOVE)
                                {
                                    stepData.leavingCount++;
                                    this->allStepsCounter += particle.steps;
                                    // remove particle from current list
                                    removeParticle(index, i);
                                }
                                else
                                {
                                    STORMError eo("Unknown boundary condition for particle");
                                    eo.addEntry("Particle", particle);
                                    eo.addEntry("Status", status);
                                    throw eo;
                                }
                                break;
                            }

                            particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                            auto [otherRank, neighborIndexInRank] = it->second;
                            #ifdef STORM_DEBUG
                            particle.checkedHere = false; // reset checked here flag
                            if(particle.nextRank != std::numeric_limits<rank_t>::max())
                            {
                                STORMError eo("Particle was already sent, and not sent again");
                                eo.addEntry("Particle", particle);
                                eo.addEntry("Already Transferred To Rank", particle.nextRank);
                                eo.addEntry("Being Transferred To Rank", otherRank);
                                eo.addEntry("Being Transferred To Index In Rank", neighborIndexInRank);
                                throw eo;
                            }
                            const std::vector<rank_t> &neighbors = this->grid.GetDuplicatedProcs();
                            if(std::find(neighbors.cbegin(), neighbors.cend(), otherRank) == neighbors.cend())
                            {
                                STORMError eo("Particle is going to be transffered to a non-neighboring rank");
                                eo.addEntry("Particle", particle);
                                eo.addEntry("My Rank", this->rank_world);
                                eo.addEntry("Next Rank", otherRank);
                                eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                                throw eo;
                            }
                            particle.cellIndexInPrevRank = particle.cellIndex;
                            particle.sentByRank = this->rank_world;
                            particle.ghostIndex = nextCellIndex;
                            particle.newCellValue = this->grid.GetMeshPoint(nextCellIndex);
                            particle.particleIndexInLastRank = particleIndex;
                            particle.particleTHInLastRank = i;
                            particle.nextRank = otherRank;

                            if(particle.nextRank == this->rank_world)
                            {
                                STORMError eo("Particle is going to be sent to the same rank");
                                eo.addEntry("Particle", particle);
                                eo.addEntry("My Rank", this->rank_world);
                                eo.addEntry("Next Rank", otherRank);
                                eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                                throw eo;
                            }
                            #endif // STORM_DEBUG
                            particle.sent = true;
                            particle.cellIndex = neighborIndexInRank;

                            #ifdef STORM_DEBUG
                            if(not TransferParticlesVecOfRank.empty())
                            {
                                size_t lastTHIndex = TransferParticlesVecOfRank.back();
                                size_t lastParticleIndex = handler->th[lastTHIndex];
                                const MCParticle &lastParticle = handler->particles[lastParticleIndex];
                                if(lastParticle == particle)
                                {
                                    STORMError eo("Particle is already in the transfer list");
                                    eo.addEntry("Iteration", this->iteration);
                                    eo.addEntry("Particle", particle);
                                    eo.addEntry("My Rank", this->rank_world);
                                    eo.addEntry("TH Index 1", lastTHIndex);
                                    eo.addEntry("TH Index 2", i);
                                    eo.addEntry("Length of Transfer List", TransferParticlesVecOfRank.size());
                                    eo.addEntry("In Rank Buffer", _rank);
                                    eo.addEntry("Sent to Rank", otherRank);
                                    throw eo;
                                }
                            }
                            #endif // STORM_DEBUG

                            transferParticle(index, i, otherRank);
                            break;
                        }
                    }
                    else if(functionality.change == MonteCarloParticleStatus::REMOVE)
                    {
                        this->allStepsCounter += particle.steps;
                        removeParticle(index, i);
                        break;
                    }
                    else if(functionality.change == MonteCarloParticleStatus::DONE)
                    {
                        stepData.remaining.push_back(particle);
                        this->allStepsCounter += particle.steps;
                        // remove particle from current list
                        removeParticle(index, i);
                        break;
                    }
                }
            }
            catch(STORMError &eo)
            {
                eo.addEntry("Particle TH index", i);
                eo.addEntry("Particle index", particleIndex);
                eo.addEntry("Handler rank buffer", _rank);
                eo.addEntry("Handler TH length", handler->th_length);
                eo.addEntry("Handler AV length", handler->av_length);
                eo.addEntry("Handler buffer size", handler->buffsize);
                int particleIndexAVPosition = -1;
                int handlerAVLength = handler->av_length;
                if(handlerAVLength >= 0)
                {
                    for(int avIndex = 0; avIndex < handlerAVLength; avIndex++)
                    {
                        if(handler->av[avIndex] == particleIndex)
                        {
                            particleIndexAVPosition = avIndex;
                            break;
                        }
                    }
                }
                eo.addEntry("Particle index AV position", particleIndexAVPosition);
                int duplicateParticleIndex = -1;
                int duplicateTHFirst = -1;
                int duplicateTHSecond = -1;
                int handlerTHLength = handler->th_length;
                if(handlerTHLength >= 0)
                {
                    static thread_local std::vector<int> seenTHIndex;
                    seenTHIndex.assign(handler->buffsize, -1);
                    for(int thIndex = 0; thIndex < handlerTHLength; thIndex++)
                    {
                        size_t thParticleIndex = handler->th[thIndex];
                        if(thParticleIndex >= handler->buffsize)
                        {
                            continue;
                        }
                        if(seenTHIndex[thParticleIndex] >= 0)
                        {
                            duplicateParticleIndex = static_cast<int>(thParticleIndex);
                            duplicateTHFirst = seenTHIndex[thParticleIndex];
                            duplicateTHSecond = thIndex;
                            break;
                        }
                        seenTHIndex[thParticleIndex] = thIndex;
                    }
                }
                eo.addEntry("Duplicate TH particle index", duplicateParticleIndex);
                eo.addEntry("Duplicate TH first index", duplicateTHFirst);
                eo.addEntry("Duplicate TH second index", duplicateTHSecond);
                throw eo;
            }
            // this->reallocationAgent->HandleAllWaitingReallocations();
        }

        // this->reallocationAgent->HandleAllWaitingReallocations();
    }


    for(size_t i = 0; i < activeRanksNum; i++)
    {
        const rank_t &fromRank = active_ranks[i];
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myTHIndices = transferParticlesVec[i];
        const std::vector<rank_t> &myTransferRanks = transferToRanks[i];
        size_t numToTransfer = myTHIndices.size();

        for(size_t j = 0; j < numToTransfer; j++)
        {
            const size_t &indexInToHandle = myTHIndices[j];
            const rank_t &toRank = myTransferRanks[j];
            assert(toRank != this->rank_world);
            size_t particleIdx = currRankHandler->th[indexInToHandle];
            MCParticle &particle = currRankHandler->particles[particleIdx];
            particle.sent = false;
            std::vector<MCParticle> &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.push_back(particle);
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, 1);
        }
    }

    for(size_t i = 0; i < activeRanksNum; i++)
    {
        rank_t _rank = active_ranks[i];
        const std::vector<size_t> &rankRemoveParticlesVec = removeParticlesVec[i];
        if(rankRemoveParticlesVec.empty())
        {
            continue; // nothing to remove
        }
        RankHandler_t *handler = this->rankHandlers[_rank];
        handler->RemoveParticles(rankRemoveParticlesVec, rankRemoveParticlesVec.size());
    }

    for(size_t i = 0; i < activeRanksNum; i++)
    {
        rank_t _rank = active_ranks[i];
        RankHandler_t *handler = this->rankHandlers[_rank];
        if(handler->th_length > 0)
        {
            next_active_ranks.push_back(_rank);
        }
    }
    active_ranks.swap(next_active_ranks);

    if(not isEmpty)
    {
        this->activeRankScanRemaining = 0;
        completedNeighborSweep = false;
    }
    bool toReturn = isEmpty and completedNeighborSweep and particlesToAdd.empty();
    if(not particlesToAdd.empty())
    {
        this->dynamicallyAdded += particlesToAdd.size();
        this->AddParticles(particlesToAdd);
        particlesToAdd.clear();
    }

    return toReturn;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::ResetAllBuffers(void)
{
    auto resetHandler = [this](rank_t rank)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->rankHandlers.size()))
        {
            return;
        }
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler != nullptr)
        {
            handler->Reset();
        }
    };

    resetHandler(this->rank_world);
    for(rank_t rank : this->neighbors)
    {
        if(rank != this->rank_world)
        {
            resetHandler(rank);
        }
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::ShrinkBuffers(void)
{
    std::vector<rank_t> shrinkList;
    boost::container::flat_set<rank_t> neighbors(this->neighbors.cbegin(), this->neighbors.cend());
    for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
    {
        if(r != this->rank_world and this->rankHandlers[r] != nullptr and this->rankHandlers[r]->buffsize > this->config.minimalBuffSize)
        {
            if(neighbors.find(r) == neighbors.end())
            {
                // no longer neighbor
                shrinkList.push_back(r);
            }
        }
    }
    const size_t localOriginalRequested = shrinkList.size();
    double shrinkPercent = this->config.shrinkPercent;
    shrinkPercent = std::max(0.0, std::min(1.0, shrinkPercent));
    if(not shrinkList.empty() and shrinkPercent < 1.0)
    {
        size_t shrinkBudget = 0;
        if(shrinkPercent > 0.0)
        {
            shrinkBudget = std::max<size_t>(
                1,
                static_cast<size_t>(std::ceil(shrinkPercent * static_cast<double>(shrinkList.size()))));
        }
        if(shrinkBudget < shrinkList.size())
        {
            shrinkList.resize(shrinkBudget);
        }
    }

    std::vector<std::vector<rank_t>> shrinkRequests(this->size_world);
    for(rank_t r : shrinkList)
    {
        shrinkRequests[r].push_back(this->rank_world);
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkRequests =
        MPI_Exchange_sparse_by_rank(shrinkRequests, this->comm_world, MPI_EXCHANGE_SPARSE_TAG + 20);

    boost::container::flat_set<rank_t> shrinkCandidates(shrinkList.cbegin(), shrinkList.cend());
    for(const auto &[requestingRank, ignoredPayload] : incomingShrinkRequests)
    {
        (void)ignoredPayload;
        shrinkCandidates.insert(requestingRank);
    }

    std::vector<std::vector<rank_t>> shrinkConfirmations(this->size_world);
    size_t candidatesWithoutHandler = 0;
    for(rank_t r : shrinkCandidates)
    {
        if(r == this->rank_world)
        {
            continue;
        }

        if(this->rankHandlers[r] != nullptr)
        {
            shrinkConfirmations[r].push_back(this->rank_world);
        }
        else
        {
            candidatesWithoutHandler++;
        }
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkConfirmations =
        MPI_Exchange_sparse_by_rank(shrinkConfirmations, this->comm_world, MPI_EXCHANGE_SPARSE_TAG + 21);

    boost::container::flat_set<rank_t> confirmedByPeer;
    for(const auto &[confirmingRank, ignoredPayload] : incomingShrinkConfirmations)
    {
        (void)ignoredPayload;
        confirmedByPeer.insert(confirmingRank);
    }

    std::vector<rank_t> shrinkPartners;
    shrinkPartners.reserve(shrinkCandidates.size());
    size_t missingPeerConfirmation = 0;
    for(rank_t r : shrinkCandidates)
    {
        if(r == this->rank_world)
        {
            continue;
        }

        if(this->rankHandlers[r] == nullptr)
        {
            continue;
        }

        if(confirmedByPeer.find(r) == confirmedByPeer.end())
        {
            missingPeerConfirmation++;
            continue;
        }

        shrinkPartners.push_back(r);
    }

    std::sort(shrinkPartners.begin(), shrinkPartners.end());

    (void)localOriginalRequested;
    (void)candidatesWithoutHandler;
    (void)missingPeerConfirmation;

    auto shrinkBuffer = [this](rank_t _rank)
    {
        double factor;
        if(std::find(this->neighbors.cbegin(), this->neighbors.cend(), _rank) != this->neighbors.cend())
        {
            factor = this->config.bufferShrinkNeighborFactor;
        }
        else
        {
            factor = this->config.bufferShrinkFactor;
        }
        this->rankHandlers[_rank]->requestedFactor = factor;
        this->rankHandlers[_rank]->Reallocate(factor);
    };

    for(rank_t r : shrinkPartners)
    {
        shrinkBuffer(r);
    }
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerLegacy<T, Grid>::MCParticle> &MonteCarloManagerLegacy<T, Grid>::GetSendBuffer(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBuffers.size()));
    return this->sendBuffers[static_cast<size_t>(rank)];
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::QueueReadySendBuffer(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBufferReadyQueued.size()));
    size_t rankIndex = static_cast<size_t>(rank);
    if(this->sendBufferReadyQueued[rankIndex])
    {
        return;
    }
    this->sendBufferReadyQueued[rankIndex] = 1;
    this->readySendBufferRanks.push_back(rank);
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::MarkSendBufferEmpty(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBuffers.size()));
    size_t rankIndex = static_cast<size_t>(rank);
    this->sendBuffers[rankIndex].clear();
    this->sendBufferReadyQueued[rankIndex] = 0;
    if(this->sendBufferActive[rankIndex])
    {
        this->sendBufferActive[rankIndex] = 0;
        assert(this->sendBufferPendingRanks > 0);
        this->sendBufferPendingRanks--;
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::ResetSendBuffers(void)
{
    for(rank_t rank : this->sendBufferActiveRanks)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            continue;
        }
        this->sendBuffers[static_cast<size_t>(rank)].clear();
    }
    std::fill(this->sendBufferActive.begin(), this->sendBufferActive.end(), 0);
    std::fill(this->sendBufferListed.begin(), this->sendBufferListed.end(), 0);
    std::fill(this->sendBufferReadyQueued.begin(), this->sendBufferReadyQueued.end(), 0);
    this->sendBufferActiveRanks.clear();
    this->readySendBufferRanks.clear();
    this->readySendBufferCursor = 0;
    this->sendBufferPendingRanks = 0;
    this->sendBufferCycleCounter = 0;
    this->sendBufferPendingParticles = 0;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::NoteSendBufferGrowth(rank_t rank, size_t previousSize, const std::vector<MCParticle> &buffer, size_t addedParticles)
{
    this->sendBufferPendingParticles += addedParticles;
    if(addedParticles > 0 and previousSize == 0 and not buffer.empty())
    {
        assert(rank >= 0);
        assert(rank < static_cast<rank_t>(this->sendBufferActive.size()));
        size_t rankIndex = static_cast<size_t>(rank);
        if(not this->sendBufferActive[rankIndex])
        {
            this->sendBufferActive[rankIndex] = 1;
            this->sendBufferPendingRanks++;
        }
        if(not this->sendBufferListed[rankIndex])
        {
            this->sendBufferListed[rankIndex] = 1;
            this->sendBufferActiveRanks.push_back(rank);
        }
    }
    if(previousSize < this->config.sendBufferMinSize and buffer.size() >= this->config.sendBufferMinSize)
    {
        this->QueueReadySendBuffer(rank);
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::NoteSendBufferFlush(rank_t rank, size_t flushedParticles)
{
    assert(this->sendBufferPendingParticles >= flushedParticles);
    this->sendBufferPendingParticles -= flushedParticles;
    this->MarkSendBufferEmpty(rank);
}


template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::FlushSendBuffers(bool flushSmallBuffers)
{
    size_t pendingRanks = this->sendBufferPendingRanks;

    if(pendingRanks > 0)
    {
        this->sendBufferCycleCounter++;
    }
    else
    {
        this->sendBufferCycleCounter = 0;
    }

    bool allowIdleDrain = flushSmallBuffers;
    bool heldIdleDrain = false;
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    if(flushSmallBuffers and this->config.holdSmallIdleFlushes)
    {
        size_t holdoffCycles = std::max<size_t>(1, this->config.GetSmallIdleFlushHoldoffCycles());
        allowIdleDrain = this->sendBufferCycleCounter >= holdoffCycles;
    }

    if(pendingRanks == 0)
    {
        return;
    }

    auto flushRankIfReady = [&](rank_t toRank, bool allowIdleFlush)
    {
        if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            return;
        }
        size_t rankIndex = static_cast<size_t>(toRank);
        if(not this->sendBufferActive[rankIndex])
        {
            return;
        }
        std::vector<MCParticle> &particles = this->sendBuffers[rankIndex];
        if(particles.empty())
        {
            this->MarkSendBufferEmpty(toRank);
            return;
        }
        bool thresholdFlush = particles.size() >= this->config.sendBufferMinSize;
        bool idleFlush = allowIdleFlush &&
            (particles.size() >= this->config.sendBufferMinIdleDrainSize ||
             this->sendBufferCycleCounter >= this->config.sendBufferIdleDrainPatienceCycles);
        if(not thresholdFlush and not idleFlush)
        {
            return;
        }
        if(usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank))
        {
            if(thresholdFlush)
            {
                this->QueueReadySendBuffer(toRank);
            }
            return;
        }
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        size_t flushedParticles = particles.size();
        bool transferred = remoteHandler->TransferParticles(particles);
        if(not transferred)
        {
            this->ProgressReallocations();
            if(thresholdFlush)
            {
                this->QueueReadySendBuffer(toRank);
            }
            return;
        }
        this->transfersCounter++;
        this->NoteSendBufferFlush(toRank, flushedParticles);
    };

    size_t readyEntries = this->readySendBufferRanks.size() - this->readySendBufferCursor;
    for(size_t readyIndex = 0; readyIndex < readyEntries; readyIndex++)
    {
        rank_t toRank = this->readySendBufferRanks[this->readySendBufferCursor++];
        if(toRank >= 0 and toRank < static_cast<rank_t>(this->sendBufferReadyQueued.size()))
        {
            this->sendBufferReadyQueued[static_cast<size_t>(toRank)] = 0;
        }
        flushRankIfReady(toRank, false);
    }
    if(this->readySendBufferCursor >= this->readySendBufferRanks.size())
    {
        this->readySendBufferRanks.clear();
        this->readySendBufferCursor = 0;
    }
    else if(this->readySendBufferCursor > 1024 and this->readySendBufferCursor * 2 > this->readySendBufferRanks.size())
    {
        this->readySendBufferRanks.erase(this->readySendBufferRanks.begin(),
                                         this->readySendBufferRanks.begin() + static_cast<std::ptrdiff_t>(this->readySendBufferCursor));
        this->readySendBufferCursor = 0;
    }

    if(allowIdleDrain)
    {
        for(size_t index = 0; index < this->sendBufferActiveRanks.size();)
        {
            rank_t toRank = this->sendBufferActiveRanks[index];
            if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
            {
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            size_t rankIndex = static_cast<size_t>(toRank);
            if(not this->sendBufferActive[rankIndex])
            {
                this->sendBufferListed[rankIndex] = 0;
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            flushRankIfReady(toRank, true);
            if(not this->sendBufferActive[rankIndex])
            {
                this->sendBufferListed[rankIndex] = 0;
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            if(flushSmallBuffers)
            {
                heldIdleDrain = true;
            }
            index++;
        }
    }
    else if(flushSmallBuffers and this->sendBufferPendingRanks > 0)
    {
        heldIdleDrain = true;
    }

    if(this->sendBufferPendingRanks == 0)
    {
        this->sendBufferCycleCounter = 0;
    }
    (void)heldIdleDrain;
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::FlushAllSendBuffers(void)
{
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    for(size_t index = 0; index < this->sendBufferActiveRanks.size();)
    {
        rank_t toRank = this->sendBufferActiveRanks[index];
        if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        size_t rankIndex = static_cast<size_t>(toRank);
        if(not this->sendBufferActive[rankIndex])
        {
            this->sendBufferListed[rankIndex] = 0;
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        std::vector<MCParticle> &particles = this->sendBuffers[rankIndex];
        if(particles.empty())
        {
            this->MarkSendBufferEmpty(toRank);
            this->sendBufferListed[rankIndex] = 0;
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        size_t flushedParticles = particles.size();
        if(usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank))
        {
            index++;
            continue;
        }
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        bool transferred = remoteHandler->TransferParticles(particles);
        if(not transferred)
        {
            this->ProgressReallocations();
            index++;
            continue;
        }
        this->transfersCounter++;
        this->NoteSendBufferFlush(toRank, flushedParticles);
        this->sendBufferListed[rankIndex] = 0;
        this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
        this->sendBufferActiveRanks.pop_back();
    }
    if(this->sendBufferPendingRanks == 0)
    {
        this->sendBufferCycleCounter = 0;
    }
}

template<typename T, typename Grid>
bool MonteCarloManagerLegacy<T, Grid>::AllSendBuffersEmpty(void) const
{
    if(this->sendBufferPendingParticles == 0)
    {
        return true;
    }
    for(rank_t rank : this->sendBufferActiveRanks)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            continue;
        }
        size_t rankIndex = static_cast<size_t>(rank);
        if(this->sendBufferActive[rankIndex] and not this->sendBuffers[rankIndex].empty())
        {
            return false;
        }
    }
    return false;
}


template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum)
{
    using index_t = typename RankHandler_t::index_t;
    const size_t bytesPerSlot = sizeof(MCParticle) + 2 * sizeof(index_t);
    size_t localHandlerMemory = 0;
    for(const RankHandler_t *h : this->rankHandlers)
    {
        if(h == nullptr) continue;
        localHandlerMemory += h->buffsize * bytesPerSlot;
    }

    struct { double val; int rank; } myMem, maxMem;
    myMem.val = static_cast<double>(localHandlerMemory);
    myMem.rank = this->rank_world;
    MPI_Allreduce(&myMem, &maxMem, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm_world);

    double avgMem = static_cast<double>(localHandlerMemory);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &avgMem, &avgMem, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    size_t preLoopInitial = initialParticlesNum;
    size_t preLoopPreStep = preStepParticlesNum;
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopInitial, &preLoopInitial, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopPreStep, &preLoopPreStep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    std::string handlerBreakdown;
    if(this->rank_world == maxMem.rank)
    {
        boost::container::flat_set<rank_t> neighborSet(this->neighbors.begin(), this->neighbors.end());
        std::ostringstream ss;
        double selfTotal = 0, neighborTotal = 0, nonNeighborTotal = 0;
        for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
        {
            const RankHandler_t *h = this->rankHandlers[r];
            if(h == nullptr) continue;
            double handlerMB = h->buffsize * bytesPerSlot / (1024.0 * 1024.0);
            bool isSelf = (h->peer_rank_world == this->rank_world);
            bool isNeighbor = !isSelf && neighborSet.count(r) > 0;
            (isSelf ? selfTotal : isNeighbor ? neighborTotal : nonNeighborTotal) += handlerMB;
            // const char *tag = isSelf ? " [self]" : isNeighbor ? " [neighbor]" : " [non-neighbor]";
            // ss << "  [" << h->peer_rank_world << "]: "
            //    << handlerMB << " MB (buffsize=" << h->buffsize << ")"
            //    << tag << "\n";
        }
        ss << "  Totals: self=" << selfTotal << " MB, neighbors=" << neighborTotal << " MB, non-neighbors=" << nonNeighborTotal << " MB\n";
        handlerBreakdown = ss.str();
    }

    if(maxMem.rank != 0)
    {
        if(this->rank_world == maxMem.rank)
        {
            int strLen = static_cast<int>(handlerBreakdown.size());
            MPI_Send(&strLen, 1, MPI_INT, 0, 999, this->comm_world);
            MPI_Send(handlerBreakdown.data(), strLen, MPI_CHAR, 0, 1000, this->comm_world);
        }
        else if(this->rank_world == 0)
        {
            int strLen = 0;
            MPI_Recv(&strLen, 1, MPI_INT, maxMem.rank, 999, this->comm_world, MPI_STATUS_IGNORE);
            handlerBreakdown.resize(strLen);
            MPI_Recv(&handlerBreakdown[0], strLen, MPI_CHAR, maxMem.rank, 1000, this->comm_world, MPI_STATUS_IGNORE);
        }
    }

    if(this->rank_world == 0)
    {
        avgMem /= this->size_world;
        std::cout << "RankHandler memory: max=" << maxMem.val / (1024.0 * 1024.0) << " MB (rank " << maxMem.rank << "), avg=" << avgMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << handlerBreakdown;
        std::cout << "Starting with " << (preLoopInitial + preLoopPreStep) << ". Came with " << preLoopInitial << ". Generated " << preLoopPreStep << " particles in preStep." << std::endl;
    }
}

template<typename T, typename Grid>
void MonteCarloManagerLegacy<T, Grid>::PrepareHandlers(void)
{

    boost::container::flat_set<rank_t> oldNeighbors(this->neighbors.cbegin(), this->neighbors.cend());

    this->neighbors = GetNeighborList(this->grid, this->ranks_ghost_map);

    // Self handler: 1-process communicator, no coordination needed
    if(this->rankHandlers[this->rank_world] == nullptr)
    {
        MPI_Comm_dup(MPI_COMM_SELF, &this->communicators[this->rank_world]);
        this->rankHandlers[this->rank_world] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, this->communicators[this->rank_world], this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
    }
    std::vector<rank_t> newNeighbors;
    for(rank_t rank : this->neighbors)
    {
        if(oldNeighbors.find(rank) == oldNeighbors.end() and this->rankHandlers[rank] == nullptr)
        {
            newNeighbors.push_back(rank);
        }
    }

    int numNewNeighbors = newNeighbors.size();
    MPI_Allreduce(MPI_IN_PLACE, &numNewNeighbors, 1, MPI_INT, MPI_SUM, this->comm_world);

    if(numNewNeighbors > 0)
    {
        auto createHandler = [this](rank_t rank, MPI_Comm pair_comm)
        {
            if(this->rankHandlers[rank] != nullptr)
            {
                return;
            }

            this->communicators[rank] = pair_comm;
            this->rankHandlers[rank] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, pair_comm, this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
            if(this->rankHandlers[rank]->peer_rank_world != rank)
            {
                STORMError eo("Peer rank world does not match");
                eo.addEntry("Rank", rank);
                eo.addEntry("Peer Rank World", this->rankHandlers[rank]->peer_rank_world);
                throw eo;
            }
        };
        ForEachRankSyncByList(this->comm_world, newNeighbors, createHandler);
    }

    this->ResetAllBuffers();
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerLegacy<T, Grid>::MCParticle> MonteCarloManagerLegacy<T, Grid>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    // if(this->Ncells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rank_world << ": " << this->Ncells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }

    this->Ncells = this->grid.GetPointNo();
    this->ranks_ghost_map = GetGhostMap(this->grid);
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();

    this->PrepareHandlers();

    this->ResetSendBuffers();
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;


    bool didRebalance = this->grid.DidRebalance() and (this->lastBuildGeneration != this->grid.GetBuildGeneration());
    if(didRebalance)
    {
        this->ShrinkBuffers();
    }
    this->lastBuildGeneration = this->grid.GetBuildGeneration();

    size_t initialParticlesNum = particleList.size();
    this->initialParticleCount = initialParticlesNum;
    this->cellsParticleCounters.assign(this->Ncells, 0);
    for(const auto &p : particleList) this->cellsParticleCounters[p.cellIndex]++;
    this->PutSelfParticles(std::move(particleList));
    this->physics->updateGridData();

    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);


    size_t preStepParticlesNum = newParticles1.size();
    for(const auto &p : newParticles1) this->cellsParticleCounters[p.cellIndex]++;
    this->startParticleCount = initialParticlesNum + preStepParticlesNum;
    this->beginningParticleCount = this->cellsParticleCounters;

    this->resetTracker();
    this->currentStep++;
    this->iteration = 0;
    this->allStepsCounter = 0;
    this->dynamicallyAdded = 0;
    // this->neighbors = this->grid.GetDuplicatedProcs();
    this->cellsStepsCounters.assign(this->Ncells, 0);
    this->transfersCounter = 0;

    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }

        int length = handler->th_length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            #ifdef STORM_DEBUG
            p.checkedHere = true;
            p.nextRank = std::numeric_limits<rank_t>::max();
            p.removedFromRank = false;
            p.sentByRank = std::numeric_limits<rank_t>::max();
            p.lastSeen = 0;
            #endif // STORM_DEBUG
            #ifdef STORM_WITH_TRACING_HISTORY
            p.tracingHistoryIndex = 0;
            p.tracingHistoryCount = 0;
            #endif // STORM_WITH_TRACING_HISTORY
            p.timeLeft = fullDt;
            p.initialWeight = p.weight;
            p.steps = 0;
        }
    }
    {
        this->AddParticles(newParticles1);
        std::vector<MCParticle>().swap(newParticles1);
    }
    MPI_Barrier(this->comm_world);

    size_t numParticles = initialParticlesNum + preStepParticlesNum;

    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;
    AmountManager amountManager(this->comm_world);
    amountManager.Initialize(startingParticleNum);

    MonteCarloStepFinalData data;
    size_t numOfCounterDecrementations = 0;

    {
        const size_t bytesPerSlot = sizeof(MCParticle) + 2 * sizeof(typename RankHandler_t::index_t);
        this->handlerMemoryBytes = 0;
        for(const RankHandler_t *h : this->rankHandlers)
        {
            if(h != nullptr)
            {
                this->handlerMemoryBytes += h->buffsize * bytesPerSlot;
            }
        }
    }

    this->PrintMemoryDiagnostics(initialParticlesNum, preStepParticlesNum);

    const bool &verify = amountManager.GetVerifyRef();
    const bool &done = amountManager.GetDoneRef();

    MEMORY_DEBUG_PRINT("Before main loop in MCM");

    const size_t amountProgressMinCycles = std::max<size_t>(1, this->config.amountProgressMinCycles);
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    const size_t reallocationProgressMinCycles = usesAsyncReallocation
        ? std::max<size_t>(1, this->config.asyncReallocationProgressMinCycles)
        : 1;
    auto loopStart = std::chrono::high_resolution_clock::now();
    try
    {
        while(not done)
        {
            this->PumpRMAProgress();
            bool shouldProgressReallocations = (not usesAsyncReallocation) or
                (this->iteration % reallocationProgressMinCycles == 0) or
                this->reallocationAgent->HasPendingAsyncReallocations();
            if(shouldProgressReallocations)
            {
                this->ProgressReallocations();
            }
            bool localWorkDone = this->HandleAll(data);
            this->PumpRMAProgress();
            this->FlushSendBuffers(localWorkDone);

            amountManager.Decrease(this->localDecrementAmount);
            this->localDecrementAmount = 0;

            if(this->iteration % amountProgressMinCycles == 0)
            {
                amountManager.Progress();
            }

            if(verify)
            {
                this->FlushAllSendBuffers();
                this->ProgressReallocations();
                bool ok = this->AllSendBuffersEmpty() and not this->reallocationAgent->HasPendingAsyncReallocations();
                amountManager.Verify(ok);
            }

            this->iteration++;
        }
    }
    catch(const STORMError &eo)
    {
        reportError(eo);
        throw;
    }

    auto loopEnd = std::chrono::high_resolution_clock::now();
    double loopTime = std::chrono::duration_cast<std::chrono::duration<double>>(loopEnd - loopStart).count();
    double localStepCount = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        localStepCount += static_cast<double>(counter);
    }
    double avgSteps = localStepCount;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &avgSteps, &avgSteps, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);
    avgSteps /= this->size_world;
    double maxSteps = localStepCount;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &maxSteps, &maxSteps, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "Loop time: " << loopTime << " seconds, max steps: " << maxSteps << ", avg steps: " << avgSteps << std::endl;
    }


    data.remaining = this->populationControl->activate(data.remaining);
    this->physics->postStep(data.remaining, fullDt);

    size_t newParticlesNum = data.remaining.size();
    this->endParticleCount = newParticlesNum;

    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        if(handler->th_length != 0)
        {
            STORMError eo("End of MonteCarloManagerLegacy::step: th length is not 0");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("TH Length", handler->th_length);
            eo.addEntry("Peer Rank", handler->peer_rank_world);
            throw eo;
        }
    }

    if(not didRebalance)
    {
        if(this->currentStep > 0 and this->config.shrinkBuffersCycle > 0 and this->currentStep % this->config.shrinkBuffersCycle == 0)
        {
            this->ShrinkBuffers();
        }
    }


    return data.remaining;
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // MONTE_CARLO_MANAGER_LEGACY_HPP
