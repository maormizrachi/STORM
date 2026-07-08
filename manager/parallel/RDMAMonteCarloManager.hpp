#ifndef RDMA_MONTE_CARLO_MANAGER_HPP
#define RDMA_MONTE_CARLO_MANAGER_HPP

#ifdef STORM_WITH_MPI

#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <boost/container/flat_set.hpp>
#include <mpi_utils/mpi_commands.hpp>
#include <mpi_utils/AmountManager.hpp>
#include "../../particle/Particle.hpp"
#include "../../physics/MonteCarloPhysics.hpp"
#include "../../population/PopulationControl.hpp"
#include "../../boundary/BoundaryCondition.hpp"
#include "../../utils/GhostMap.hpp"
#include "../../utils/RankSync.hpp"
#include "RankHandler2.hpp"
#include "ReallocationAgent.hpp"
#ifdef MEMORY_DEBUG
#include "misc/memory_debug.hpp"
#else
#ifndef MEMORY_DEBUG_PRINT
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#endif
#endif
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mpi.h>
#include "../MonteCarloConfig.hpp"
#include "../../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

#define RW_PROGRESS_TAG 9941
#define MC_PROGRESS_COUNTERS 6

enum MCProgressCounterIndex : size_t
{
    MC_PROGRESS_RW_STEPS = 0,
    MC_PROGRESS_DDMC_STEPS,
    MC_PROGRESS_DDMC_LEAKS,
    MC_PROGRESS_DDMC_CENSUS,
    MC_PROGRESS_DDMC_UPSCATTER,
    MC_PROGRESS_DDMC_FALLBACK,
    MC_PROGRESS_COUNTER_COUNT
};

static_assert(MC_PROGRESS_COUNTER_COUNT == MC_PROGRESS_COUNTERS,
              "Update MC_PROGRESS_COUNTERS when progress fields change");

template<typename T>
double MaxAxisRelativeDrift(const T &drift, const T &boxSize)
{
    double maxRelDrift = 0.0;
    auto update = [&maxRelDrift](double delta, double size)
    {
        if(size > 0.0)
            maxRelDrift = std::max(maxRelDrift, std::abs(delta) / size);
    };

    update(std::abs(drift.x), std::abs(boxSize.x));
    update(std::abs(drift.y), std::abs(boxSize.y));
    update(std::abs(drift.z), std::abs(boxSize.z));
    return maxRelDrift;
}

template<typename T>
void ComputeBoxDriftDiagnostics(const T &location, const T &boxLL, const T &boxUR,
                                double &relativeDrift, double &maxAxisRelativeDrift)
{
    T boxSize = boxUR - boxLL;
    T clamped = location;
    clamped.x = std::max(boxLL.x, std::min(boxUR.x, clamped.x));
    clamped.y = std::max(boxLL.y, std::min(boxUR.y, clamped.y));
    clamped.z = std::max(boxLL.z, std::min(boxUR.z, clamped.z));

    T drift = location - clamped;
    relativeDrift = abs(drift) / abs(boxSize);
    maxAxisRelativeDrift = MaxAxisRelativeDrift(drift, boxSize);
}

template<typename Grid>
std::vector<rank_t> GetNeighborList2(const Grid &tess, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &ghostsMap)
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
class RDMAMonteCarloManager
{
    using MCParticle = MonteCarloParticle<T, Grid>;
    using RankHandler_t = RankHandler2<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        size_t leavingCount = 0;
    };

    RDMAMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MonteCarloConfig &config = MonteCarloConfig(),
                    const MPI_Comm &comm = MPI_COMM_WORLD, RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA);

    ~RDMAMonteCarloManager();

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

    inline size_t GetPreStepParticleCount(void) const {return this->preStepParticleCount;}

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
    class RegisteredSendBuffer
    {
    public:
        RegisteredSendBuffer() = default;
        RegisteredSendBuffer(const RegisteredSendBuffer&) = delete;
        RegisteredSendBuffer &operator=(const RegisteredSendBuffer&) = delete;

        RegisteredSendBuffer(RegisteredSendBuffer &&other) noexcept:
            storage(std::move(other.storage)),
            registrationHandler(other.registrationHandler),
            sourceRegistration(other.sourceRegistration)
        {
            other.registrationHandler = nullptr;
            other.sourceRegistration = {};
        }

        RegisteredSendBuffer &operator=(RegisteredSendBuffer &&other) noexcept
        {
            if(this != &other)
            {
                this->ReleaseRegistration();
                this->storage = std::move(other.storage);
                this->registrationHandler = other.registrationHandler;
                this->sourceRegistration = other.sourceRegistration;
                other.registrationHandler = nullptr;
                other.sourceRegistration = {};
            }
            return *this;
        }

        ~RegisteredSendBuffer()
        {
            this->ReleaseRegistration();
        }

        size_t size(void) const {return this->storage.size();}
        size_t capacity(void) const {return this->storage.capacity();}
        bool empty(void) const {return this->storage.empty();}
        MCParticle *data(void) {return this->storage.data();}
        const MCParticle *data(void) const {return this->storage.data();}

        void clear(void)
        {
            this->storage.clear();
        }

        void push_back(const MCParticle &particle)
        {
            this->EnsureCapacity(this->storage.size() + 1);
            this->storage.push_back(particle);
        }

        void Append(const MCParticle *particles, size_t particlesNum)
        {
            if(particlesNum == 0)
            {
                return;
            }
            this->EnsureCapacity(this->storage.size() + particlesNum);
            this->storage.insert(this->storage.end(), particles, particles + particlesNum);
        }

        uint32_t SourceLkey(RankHandler_t *handler, bool useRegistration)
        {
            if(not useRegistration or handler == nullptr or this->storage.empty())
            {
                return 0;
            }
            if(this->sourceRegistration.handle != 0 and this->registrationHandler == handler)
            {
                return this->sourceRegistration.lkey;
            }
            this->ReleaseRegistration();
            this->sourceRegistration = handler->RegisterSendSource(this->storage.data(), this->storage.capacity());
            this->registrationHandler = (this->sourceRegistration.handle == 0)? nullptr : handler;
            return this->sourceRegistration.lkey;
        }

        void ReleaseRegistration(void)
        {
            if(this->sourceRegistration.handle != 0)
            {
                assert(this->registrationHandler != nullptr);
                this->registrationHandler->DeregisterSendSource(this->sourceRegistration.handle);
            }
            this->sourceRegistration = {};
            this->registrationHandler = nullptr;
        }

    private:
        void EnsureCapacity(size_t desiredCapacity)
        {
            if(desiredCapacity <= this->storage.capacity())
            {
                return;
            }
            this->ReleaseRegistration();
            size_t newCapacity = std::max<size_t>(desiredCapacity, std::max<size_t>(1, this->storage.capacity() * 2));
            this->storage.reserve(newCapacity);
        }

        std::vector<MCParticle> storage;
        RankHandler_t *registrationHandler = nullptr;
        typename RemoteMemoryAgent<MCParticle>::SourceRegistration sourceRegistration{};
    };

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
    std::chrono::high_resolution_clock::time_point progressStartTime_;
    double lastProgressPrintTime_ = 0.0;
    int64_t progressStartParticles_ = 0;
    size_t progressRemovedCount_ = 0;
public:
    const void* progressCellsPtr_ = nullptr;
    const void* progressOpacityPtr_ = nullptr;
private:
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
    size_t preStepParticleCount = 0;
    std::vector<size_t> beginningParticleCount;
    size_t handlerMemoryBytes = 0;

    std::vector<RegisteredSendBuffer> sendBuffers;
    std::vector<rank_t> sendBufferActiveRanks;
    std::vector<rank_t> readySendBufferRanks;
    std::vector<unsigned char> sendBufferActive;
    std::vector<unsigned char> sendBufferListed;
    std::vector<unsigned char> sendBufferReadyQueued;
    std::vector<std::vector<MCParticle>> detachedRankParticles;
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

    RegisteredSendBuffer &GetSendBuffer(rank_t rank);

    void QueueReadySendBuffer(rank_t rank);

    void MarkSendBufferEmpty(rank_t rank);

    void ResetSendBuffers(void);

    void ReleaseSendBufferRegistrations(void);

    void NoteSendBufferGrowth(rank_t rank, size_t previousSize, const RegisteredSendBuffer &buffer, size_t addedParticles);

    void NoteSendBufferFlush(rank_t rank, size_t flushedParticles);

    bool UsesAsyncReallocation(void) const;

    void ProgressReallocations(void);

    void FlushSendBuffers(bool flushSmallBuffers);

    void FlushAllSendBuffers(void);

    bool AllSendBuffersEmpty(void) const;

    void PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum);

};

template<typename T, typename Grid>
RDMAMonteCarloManager<T, Grid>::RDMAMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
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
    this->detachedRankParticles.resize(this->size_world);

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
            throw std::runtime_error("RDMAMonteCarloManager: received async reallocation request for a missing handler");
        }
        return handler->LocalReallocate(factor);
    };

    auto metadataUpdateFunction = [this](rank_t rank, const ReallocationMetadata &metadata)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler == nullptr)
        {
            throw std::runtime_error("RDMAMonteCarloManager: received async reallocation metadata for a missing handler");
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
        std::cout << "Done initializing RDMAMonteCarloManager" << std::endl;
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
void RDMAMonteCarloManager<T, Grid>::ClearCommunicator()
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
void RDMAMonteCarloManager<T, Grid>::FreeHandlers(void)
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
void RDMAMonteCarloManager<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    if(particles.empty())
    {
        return;
    }

    RankHandler_t *myHandler = this->rankHandlers[this->rank_world];
    size_t particlesNum = particles.size();
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    myHandler->AppendLocalParticles(particlesNum, [this, &particles, firstID](MCParticle &destination, size_t i)
    {
        std::memcpy(&destination, &particles[i], sizeof(MCParticle));
        destination.rank = this->rank_world;
        destination.id = firstID + i;

        #ifdef STORM_DEBUG
            destination.checkedHere = true;
            destination.nextRank = std::numeric_limits<rank_t>::max();
            destination.removedFromRank = false;
            destination.sentByRank = std::numeric_limits<rank_t>::max();
            destination.lastSeen = 0;
        #endif // STORM_DEBUG

        #ifdef STORM_DEBUG
        if(not this->grid.IsPointInCell(destination.location, destination.cellIndex))
        {
            const T &declaredCell = this->grid.GetMeshPoint(destination.cellIndex);
            size_t containingIdx = this->grid.GetContainingCell(destination.location);
            const T &containingCell = this->grid.GetMeshPoint(containingIdx);
            STORMError eo("RDMAMonteCarloManager<T, Grid>::AddParticles");
            eo.addEntry("rank", this->rank_world);
            eo.addEntry("Particle", destination);
            eo.addEntry("Declared Cell Index", destination.cellIndex);
            eo.addEntry("Declared Cell", declaredCell);
            eo.addEntry("Declared Cell - Distance", abs(declaredCell - destination.location));
            eo.addEntry("Real Containing Cell Index", containingIdx);
            eo.addEntry("Real Containing Cell", containingCell);
            eo.addEntry("Real Cell - Distance", abs(containingCell - destination.location));
            throw eo;
        }
        #endif // STORM_DEBUG
    });

    this->localDecrementAmount -= static_cast<typename AmountManager::counter_t>(particlesNum);
}

template<typename T, typename Grid>
RDMAMonteCarloManager<T, Grid>::~RDMAMonteCarloManager()
{
    if(not std::uncaught_exceptions())
    {
        this->ReleaseSendBufferRegistrations();
        this->FreeHandlers();
        this->ClearCommunicator();
    }
}

template<typename T, typename Grid>
bool RDMAMonteCarloManager<T, Grid>::UsesAsyncReallocation(void) const
{
    RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                             ? RMAFactory::ResolveAutoRDMA()
                             : this->rdma_type;
    return resolved == RDMA_Type::IBV_RDMA;
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::ProgressReallocations(void)
{
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
RDMAMonteCarloManager<T, Grid>::Tracker::Tracker(const MPI_Comm &comm): comm(comm)
{}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::Tracker::Reset(void)
{
    this->track.clear();
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::Tracker::ReportParticle(MCParticle &particle)
{
    if(this->track.find(particle.id) == this->track.end())
    {
        this->track[particle.id] = std::vector<MCParticle>();
    }
    this->track[particle.id].push_back(particle);
}

template<typename T, typename Grid>
std::vector<typename RDMAMonteCarloManager<T, Grid>::MCParticle> RDMAMonteCarloManager<T, Grid>::Tracker::GetLocalTrackParticleRoute(size_t id) const
{
    auto it = this->track.find(id);
    if(it == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return it->second;
}

template<typename T, typename Grid>
std::vector<typename RDMAMonteCarloManager<T, Grid>::MCParticle> RDMAMonteCarloManager<T, Grid>::Tracker::GetTrackParticleRoute(size_t id) const
{
    std::vector<MCParticle> local = this->GetLocalTrackParticleRoute(id);
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
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
    handler->AppendLocalParticles(particlesNum, [this, &particles](MCParticle &destination, size_t i)
    {
        std::memcpy(&destination, &particles[i], sizeof(MCParticle));
        if(destination.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            destination.rank = this->rank_world;
            destination.id = this->myIDCounter++;
        }
    });

    // don't waste memory - remove current particles from the input vector
    std::vector<MCParticle> empty;
    particles.swap(empty);
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::TransferParticles(rank_t fromRank, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    RankHandler_t *currRankHandler = this->rankHandlers[fromRank];

    for(size_t i = 0; i < num; i++)
    {
        const size_t &particleIdx = indicesInToHandle[i];
        const rank_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        assert(particleIdx < currRankHandler->LocalSize());
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<MCParticle>();
        }

        MCParticle &particle = currRankHandler->LocalParticleAt(particleIdx);
        particle.sent = false; // reset

        if(toRank == this->rank_world)
        {
            STORMError eo("Trying to transfer particle to the same rank");
            eo.addEntry("Particle", particle);
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }

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
            RegisteredSendBuffer &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.Append(particles.data(), particles.size());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;

    assert(rankBuffers.size() == indicesInToHandle.size());

    size_t numRanks = rankBuffers.size();
    for(size_t i = 0; i < numRanks; i++)
    {
        const rank_t &fromRank = rankBuffers[i];
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myParticleIndices = indicesInToHandle[i];
        size_t numToHandle = myParticleIndices.size();
        const std::vector<rank_t> &myTransferRanks = transferRanks[i];

        for(size_t j = 0; j < numToHandle; j++)
        {
            const size_t &particleIdx = myParticleIndices[j];
            const rank_t &toRank = myTransferRanks[j];

            assert(toRank != this->rank_world); // can't send to self
            assert(particleIdx < currRankHandler->LocalSize());
            auto it = rankToParticles.find(toRank);
            if(it == rankToParticles.end())
            {
                rankToParticles[toRank] = std::vector<MCParticle>();
            }

            MCParticle &particle = currRankHandler->LocalParticleAt(particleIdx);
            particle.sent = false; // reset

            if(toRank == this->rank_world)
            {
                STORMError eo("Trying to transfer particle to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }

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
            RegisteredSendBuffer &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.Append(particles.data(), particles.size());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid>
bool RDMAMonteCarloManager<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<MCParticle> particlesToAdd;
    static size_t progressStepCounter;
    std::vector<rank_t> &active_ranks = this->activeRanks;
    std::vector<rank_t> &next_active_ranks = this->nextActiveRanks;

    next_active_ranks.clear();
    bool completedNeighborSweep = true;
    auto hasDetachedParticles = [this](rank_t rank)
    {
        return rank >= 0 and rank < static_cast<rank_t>(this->detachedRankParticles.size()) and
               not this->detachedRankParticles[static_cast<size_t>(rank)].empty();
    };

    if(active_ranks.empty())
    {
        const int PREFETCH_DISTANCE = 3;
        size_t neighborsNum = this->neighbors.size();
        if(neighborsNum > 0 and this->activeRankScanRemaining == 0)
        {
            this->activeRankScanRemaining = neighborsNum;
            this->activeRankScanCursor %= neighborsNum;
        }
        size_t scanCount = (neighborsNum == 0) ? 0 :
            std::min(this->activeRankScanRemaining,
                     std::min(neighborsNum, std::max<size_t>(1, this->config.activeRankScanChunk)));

        for(size_t scanOffset = 0; scanOffset < scanCount; ++scanOffset)
        {
            size_t i = (this->activeRankScanCursor + scanOffset) % neighborsNum;
            if(scanOffset + PREFETCH_DISTANCE < scanCount)
            {
                size_t futureIndex = (this->activeRankScanCursor + scanOffset + PREFETCH_DISTANCE) % neighborsNum;
                RankHandler_t *future_handler = this->rankHandlers[this->neighbors[futureIndex]];
                __builtin_prefetch(future_handler, 0, 1);
                __builtin_prefetch((const void*) &(future_handler->tail), 0, 1);
            }

            rank_t _rank = this->neighbors[i];
            RankHandler_t *handler = this->rankHandlers[_rank];
            if(hasDetachedParticles(_rank))
            {
                active_ranks.push_back(_rank);
                continue;
            }
            size_t len = handler->LocalSize();
            if(len)
            {
                active_ranks.push_back(_rank);
            }
        }
        if(neighborsNum > 0)
        {
            this->activeRankScanCursor = (this->activeRankScanCursor + scanCount) % neighborsNum;
            assert(this->activeRankScanRemaining >= scanCount);
            this->activeRankScanRemaining -= scanCount;
            completedNeighborSweep = (this->activeRankScanRemaining == 0);
        }
        {
            RankHandler_t *handler = this->rankHandlers[this->rank_world];
            if(hasDetachedParticles(this->rank_world) or not handler->LocalEmpty())
            {
                active_ranks.push_back(this->rank_world);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();


    for(size_t index = 0; index < activeRanksNum; index++)
    {
        rank_t _rank = active_ranks[index];
        RankHandler_t *handler = this->rankHandlers[_rank];
        std::vector<MCParticle> &deferredParticles = this->detachedRankParticles[static_cast<size_t>(_rank)];
        std::vector<MCParticle> localParticles;

        if(not deferredParticles.empty())
        {
            localParticles.swap(deferredParticles);
        }
        else
        {
            handler->DetachLocalParticles(localParticles);
        }

        while(not localParticles.empty())
        {
                int currentN = static_cast<int>(localParticles.size());
                size_t particleIndex = static_cast<size_t>(currentN - 1);
                MCParticle &particle = localParticles[particleIndex];
                bool removeCurrent = false;
                bool debug = false;

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
                        eo.addEntry("In List Index (1)", particle.lastSeenIndex);
                        eo.addEntry("In Rank Buffer (2)", _rank);
                        eo.addEntry("In List Index (2)", particleIndex);
                        throw eo;
                    }
                    particle.lastSeen = this->iteration;
                    particle.lastSeenRankBuf = _rank;
                    particle.lastSeenRank = this->rank_world;
                    particle.lastSeenIndex = particleIndex;
                    #endif // STORM_DEBUG

                    isEmpty = false;
                    while(true)
                    {
                        ++progressStepCounter;
                        if((progressStepCounter & 0x3FFFF) == 0 && particle.steps > 100000)
                        {
                            std::cerr << "[StuckParticle] rank=" << this->rank_world
                                      << " localPts=" << this->grid.GetPointNo()
                                      << " " << particle
                                      << " freq=" << particle.frequency
                                      << " w/w0=" << (particle.initialWeight > 0 ? particle.weight / particle.initialWeight : 0.0) << std::endl;
                            if(this->progressCellsPtr_ && particle.cellIndex < this->Ncells)
                            {
                                std::cerr << " cellIndex=" << particle.cellIndex << std::endl;
                            }
                            std::string accelInfo = this->physics->getAccelerationDebugInfo(particle.cellIndex, particle.frequency);
                            if(!accelInfo.empty())
                                std::cerr << accelInfo << std::endl;
                            std::cerr << std::endl;
                        }

                        const size_t traceStep = particle.steps;
                        if(particle.on_track)
                        {
                            MCParticle trackedParticle = particle;
                            trackedParticle.steps = traceStep * 2;
                            this->tracker.ReportParticle(trackedParticle);
                        }
                        particle.steps++;
                        this->cellsStepsCounters[particle.cellIndex]++;

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
                                eo.addEntry("Particle TH In This Rank", particleIndex);
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
                            if(containingIdx != particle.cellIndex and
                               not this->grid.IsPointInCell(particle.location, containingIdx))
                            {
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
                        #endif // STORM_DEBUG

                        if(particle.sent)
                        {
                            particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                            particle.sent = false;
                        }

                        #ifdef STORM_DEBUG
                        T prevLoc = particle.location;
                        particle.previousLocation = particle.location;
                        #endif // STORM_DEBUG

                        if(debug)
                        {
                            std::cout << "Before running particle step, particle is " << particle << std::endl;
                        }

                        #ifdef STORM_DEBUG
                        const T beforeStepLocation = particle.location;
                        const T beforeStepVelocity = particle.velocity;
                        const dt_t beforeStepTimeLeft = particle.timeLeft;
                        if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            STORMError eo("RDMAMonteCarloManager: particle outside box before physics step");
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Particle before step", particle);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->Ncells);
                            if(particle.cellIndex < this->Ncells)
                            {
                                eo.addEntry("Cell index", particle.cellIndex);
                                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                eo.addEntry("Inside declared cell before step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                            }
                            throw eo;
                        }
                        #endif // STORM_DEBUG

                        MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle, particlesToAdd);

                        #ifdef STORM_DEBUG
                        if(__builtin_expect(functionality.change != MonteCarloParticleStatus::REMOVE &&
                                          functionality.change != MonteCarloParticleStatus::CELL_MOVE &&
                                          this->grid.IsPointOutsideBox(particle.location), 0))
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            STORMError eo("RDMAMonteCarloManager: physics step moved particle outside the box");
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Particle after step", particle);
                            eo.addEntry("Functionality", MonteCarloParticleStatusToString(functionality.change));
                            eo.addEntry("Next cell index", functionality.nextCellIndex);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->Ncells);
                            if(particle.cellIndex < this->Ncells)
                            {
                                eo.addEntry("Cell index", particle.cellIndex);
                                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                            }
                            throw eo;
                        }
                        #endif // STORM_DEBUG

                        if(particle.on_track)
                        {
                            MCParticle trackedParticle = particle;
                            trackedParticle.steps = traceStep * 2 + 1;
                            this->tracker.ReportParticle(trackedParticle);
                        }

                        #ifdef STORM_WITH_TRACING_HISTORY
                            particle.recordHistory(particle.cellIndex, static_cast<int>(this->rank_world), static_cast<int>(functionality.change));
                        #endif // STORM_WITH_TRACING_HISTORY

                        if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
                        {
                            size_t nextCellIndex = functionality.nextCellIndex;

                            assert(nextCellIndex != particle.cellIndex);
                            assert(particle.timeLeft >= 0);

                            #ifdef STORM_DEBUG
                            auto throwCellMoveOutsideBox = [&](const std::string &cellMoveTarget)
                            {
                                auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                                double relDrift = 0.0;
                                double maxAxisRelDrift = 0.0;
                                ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                                STORMError eo("RDMAMonteCarloManager: CELL_MOVE moved particle outside box before a non-boundary cell move");
                                eo.addEntry("Rank", this->rank_world);
                                eo.addEntry("Particle after step", particle);
                                eo.addEntry("Cell move target", cellMoveTarget);
                                eo.addEntry("Next cell index", nextCellIndex);
                                eo.addEntry("Location before step", beforeStepLocation);
                                eo.addEntry("Velocity before step", beforeStepVelocity);
                                eo.addEntry("Time left before step", beforeStepTimeLeft);
                                eo.addEntry("Box lower", boxLL);
                                eo.addEntry("Box upper", boxUR);
                                eo.addEntry("Relative drift", relDrift);
                                eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                                eo.addEntry("Cell count", this->Ncells);
                                if(particle.cellIndex < this->Ncells)
                                {
                                    eo.addEntry("Cell index", particle.cellIndex);
                                    eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                    eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                                }
                                throw eo;
                            };
                            #endif // STORM_DEBUG

                            if(__builtin_expect(nextCellIndex < this->Ncells, 1))
                            {
                                #ifdef STORM_DEBUG
                                if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
                                    throwCellMoveOutsideBox("local cell move");
                                #endif // STORM_DEBUG

                                #ifdef STORM_DEBUG
                                size_t previousCell = particle.cellIndex;
                                #endif // STORM_DEBUG
                                particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                    MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
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
                                    throw eo;
                                }
                                #endif // STORM_DEBUG
                            }
                            else
                            {
                                auto it = ranks_ghost_map.find(nextCellIndex);
                                if(it == ranks_ghost_map.end())
                                {
                                    #ifdef STORM_WITH_TRACING_HISTORY
                                        T preReflectLoc = particle.location;
                                        T preReflectVel = particle.velocity;
                                    #endif // STORM_WITH_TRACING_HISTORY
                                    MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                                    if(status == MonteCarloParticleStatus::REFLECT)
                                    {
                                        #ifdef STORM_WITH_TRACING_HISTORY
                                            particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
                                        #endif // STORM_WITH_TRACING_HISTORY
                                        continue;
                                    }
                                    else if(status == MonteCarloParticleStatus::REMOVE)
                                    {
                                        stepData.leavingCount++;
                                        this->allStepsCounter += particle.steps;
                                        this->localDecrementAmount += 1;
                                        ++this->progressRemovedCount_;
                                        removeCurrent = true;
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

                                #ifdef STORM_DEBUG
                                if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
                                    throwCellMoveOutsideBox("remote rank transfer");
                                #endif // STORM_DEBUG

                                particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                    MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                                auto [otherRank, neighborIndexInRank] = it->second;
                                #ifdef STORM_DEBUG
                                particle.checkedHere = false;
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
                                particle.particleTHInLastRank = particleIndex;
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
                                particle.sent = false;
                                RegisteredSendBuffer &buffer = this->GetSendBuffer(otherRank);
                                size_t previousSize = buffer.size();
                                buffer.push_back(particle);
                                this->NoteSendBufferGrowth(otherRank, previousSize, buffer, 1);
                                removeCurrent = true;
                                break;
                            }
                        }
                        else if(functionality.change == MonteCarloParticleStatus::REMOVE)
                        {
                            this->allStepsCounter += particle.steps;
                            this->localDecrementAmount += 1;
                            removeCurrent = true;
                            break;
                        }
                        else if(functionality.change == MonteCarloParticleStatus::DONE)
                        {
                            stepData.remaining.push_back(particle);
                            this->allStepsCounter += particle.steps;
                            this->localDecrementAmount += 1;
                            removeCurrent = true;
                            break;
                        }
                        else if(functionality.change == MonteCarloParticleStatus::NO_CELL_MOVE)
                        {
                            continue;
                        }
                        else
                        {
                            STORMError eo("Unknown Monte Carlo particle status");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Status", functionality.change);
                            throw eo;
                        }
                    }
                }
                catch(STORMError &eo)
                {
                    eo.addEntry("Particle list index", particleIndex);
                    eo.addEntry("Handler rank buffer", _rank);
                    eo.addEntry("Handler head", static_cast<size_t>(handler->head));
                    eo.addEntry("Handler tail", static_cast<size_t>(handler->tail));
                    eo.addEntry("Handler buffer size", handler->buffsize);
                    throw eo;
                }

                assert(removeCurrent);
                localParticles.pop_back();
        }

        if(not localParticles.empty())
        {
            assert(deferredParticles.empty());
            deferredParticles.swap(localParticles);
        }
    }


    for(size_t i = 0; i < activeRanksNum; i++)
    {
        rank_t _rank = active_ranks[i];
        RankHandler_t *handler = this->rankHandlers[_rank];
        if(hasDetachedParticles(_rank) or not handler->LocalEmpty())
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
void RDMAMonteCarloManager<T, Grid>::ResetAllBuffers(void)
{
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    for(std::vector<MCParticle> &particles : this->detachedRankParticles)
    {
        particles.clear();
    }

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
void RDMAMonteCarloManager<T, Grid>::ShrinkBuffers(void)
{
    if(this->rank_world == 0)
    {
        std::cout << "Shrinking buffers." << std::endl;
    }
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
typename RDMAMonteCarloManager<T, Grid>::RegisteredSendBuffer &RDMAMonteCarloManager<T, Grid>::GetSendBuffer(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBuffers.size()));
    return this->sendBuffers[static_cast<size_t>(rank)];
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::QueueReadySendBuffer(rank_t rank)
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
void RDMAMonteCarloManager<T, Grid>::MarkSendBufferEmpty(rank_t rank)
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
void RDMAMonteCarloManager<T, Grid>::ResetSendBuffers(void)
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
void RDMAMonteCarloManager<T, Grid>::ReleaseSendBufferRegistrations(void)
{
    for(RegisteredSendBuffer &buffer : this->sendBuffers)
    {
        buffer.ReleaseRegistration();
    }
}

template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::NoteSendBufferGrowth(rank_t rank, size_t previousSize, const RegisteredSendBuffer &buffer, size_t addedParticles)
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
void RDMAMonteCarloManager<T, Grid>::NoteSendBufferFlush(rank_t rank, size_t flushedParticles)
{
    assert(this->sendBufferPendingParticles >= flushedParticles);
    this->sendBufferPendingParticles -= flushedParticles;
    this->MarkSendBufferEmpty(rank);
}


template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::FlushSendBuffers(bool flushSmallBuffers)
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
    const bool useRegisteredSendSources = usesAsyncReallocation;
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
        RegisteredSendBuffer &particles = this->sendBuffers[rankIndex];
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
        uint32_t sourceLkey = particles.SourceLkey(remoteHandler, useRegisteredSendSources);
        bool transferred = remoteHandler->TransferParticles(particles.data(), particles.size(), sourceLkey);
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
void RDMAMonteCarloManager<T, Grid>::FlushAllSendBuffers(void)
{
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    const bool useRegisteredSendSources = usesAsyncReallocation;
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
        RegisteredSendBuffer &particles = this->sendBuffers[rankIndex];
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
        uint32_t sourceLkey = particles.SourceLkey(remoteHandler, useRegisteredSendSources);
        bool transferred = remoteHandler->TransferParticles(particles.data(), particles.size(), sourceLkey);
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
bool RDMAMonteCarloManager<T, Grid>::AllSendBuffersEmpty(void) const
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
void RDMAMonteCarloManager<T, Grid>::PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum)
{
    const size_t bytesPerSlot = sizeof(MCParticle);
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
void RDMAMonteCarloManager<T, Grid>::PrepareHandlers(void)
{

    boost::container::flat_set<rank_t> oldNeighbors(this->neighbors.cbegin(), this->neighbors.cend());

    this->neighbors = GetNeighborList2(this->grid, this->ranks_ghost_map);

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
std::vector<typename RDMAMonteCarloManager<T, Grid>::MCParticle> RDMAMonteCarloManager<T, Grid>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
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
    this->preStepParticleCount = preStepParticlesNum;
    for(const auto &p : newParticles1) this->cellsParticleCounters[p.cellIndex]++;
    this->startParticleCount = initialParticlesNum + preStepParticlesNum;
    this->beginningParticleCount = this->cellsParticleCounters;

    unsigned long long globalInitialParticles = static_cast<unsigned long long>(this->initialParticleCount);
    unsigned long long globalPreStepParticles = static_cast<unsigned long long>(this->preStepParticleCount);
    unsigned long long globalStartParticles = static_cast<unsigned long long>(this->startParticleCount);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalInitialParticles, &globalInitialParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalPreStepParticles, &globalPreStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalStartParticles, &globalStartParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "MC particle counts before transport:"
                  << " initial=" << globalInitialParticles
                  << " prestep_generated=" << globalPreStepParticles
                  << " active_after_prestep=" << globalStartParticles
                  << std::endl;
    }

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

        handler->ForEachLocalParticle([fullDt](MCParticle &p, size_t)
        {
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
            p.initialWeight = std::abs(p.weight);
            p.steps = 0;
        });
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
        const size_t bytesPerSlot = sizeof(MCParticle);
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
    this->progressStartTime_ = loopStart;
    this->lastProgressPrintTime_ = 0.0;
    int64_t globalInitialForProgress = amountManager.GetValue();
    this->progressStartParticles_ = globalInitialForProgress;
    this->progressRemovedCount_ = 0;
#ifdef STORM_WITH_MPI
    std::vector<std::array<unsigned long long, MC_PROGRESS_COUNTERS>> progressCountersByRank(this->size_world);
    MPI_Request progressReportSendReq = MPI_REQUEST_NULL;
    std::array<unsigned long long, MC_PROGRESS_COUNTERS> progressReportSendValue{};
    double progressLastReportSendTime = 0.0;
#endif

    auto buildProgressCounters = [this]()
    {
        std::array<unsigned long long, MC_PROGRESS_COUNTERS> counters{};
        counters[MC_PROGRESS_RW_STEPS] =
            static_cast<unsigned long long>(this->physics->getRandomWalkStepCount());
        counters[MC_PROGRESS_DDMC_STEPS] =
            static_cast<unsigned long long>(this->physics->getDDMCStepCount());
        counters[MC_PROGRESS_DDMC_LEAKS] =
            static_cast<unsigned long long>(this->physics->getDDMCLeakCount());
        counters[MC_PROGRESS_DDMC_CENSUS] =
            static_cast<unsigned long long>(this->physics->getDDMCCensusCount());
        counters[MC_PROGRESS_DDMC_UPSCATTER] =
            static_cast<unsigned long long>(this->physics->getDDMCUpscatterCount());
        counters[MC_PROGRESS_DDMC_FALLBACK] =
            static_cast<unsigned long long>(this->physics->getDDMCFallbackCount());
        return counters;
    };

    try
    {
        while(not done)
        {
            bool shouldProgressReallocations = (not usesAsyncReallocation) or
                (this->iteration % reallocationProgressMinCycles == 0) or
                this->reallocationAgent->HasPendingAsyncReallocations();
            if(shouldProgressReallocations)
            {
                this->ProgressReallocations();
            }
            bool localWorkDone = this->HandleAll(data);
            this->FlushSendBuffers(localWorkDone);

            amountManager.Decrease(this->localDecrementAmount);
            this->localDecrementAmount = 0;

            if(this->iteration % amountProgressMinCycles == 0)
            {
                amountManager.Progress();

                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - this->progressStartTime_).count();

#ifdef STORM_WITH_MPI
                if(this->rank_world == 0)
                {
                    progressCountersByRank[0] = buildProgressCounters();

                    int hasMsg = 0;
                    MPI_Status status;
                    while(true)
                    {
                        MPI_Iprobe(MPI_ANY_SOURCE, RW_PROGRESS_TAG, this->comm_world, &hasMsg, &status);
                        if(!hasMsg)
                            break;
                        std::array<unsigned long long, MC_PROGRESS_COUNTERS> recvCounters{};
                        MPI_Recv(recvCounters.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, status.MPI_SOURCE,
                                 RW_PROGRESS_TAG, this->comm_world, MPI_STATUS_IGNORE);
                        progressCountersByRank[status.MPI_SOURCE] = recvCounters;
                    }
                }
                else if(elapsed_s - progressLastReportSendTime >= 5.0)
                {
                    if(progressReportSendReq != MPI_REQUEST_NULL)
                    {
                        int sendDone = 0;
                        MPI_Test(&progressReportSendReq, &sendDone, MPI_STATUS_IGNORE);
                        if(sendDone)
                            progressReportSendReq = MPI_REQUEST_NULL;
                    }
                    if(progressReportSendReq == MPI_REQUEST_NULL)
                    {
                        progressReportSendValue = buildProgressCounters();
                        MPI_Isend(progressReportSendValue.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, 0,
                                  RW_PROGRESS_TAG, this->comm_world, &progressReportSendReq);
                        progressLastReportSendTime = elapsed_s;
                    }
                }
#endif

                if(this->rank_world == 0 && elapsed_s - this->lastProgressPrintTime_ >= 10.0)
                {
                    this->lastProgressPrintTime_ = elapsed_s;
                    std::array<unsigned long long, MC_PROGRESS_COUNTERS> globalCounters{};
#ifdef STORM_WITH_MPI
                    for(const auto &counters : progressCountersByRank)
                    {
                        for(size_t i = 0; i < globalCounters.size(); ++i)
                            globalCounters[i] += counters[i];
                    }
#else
                    globalCounters = buildProgressCounters();
#endif
                    int64_t globalRemaining = amountManager.GetValue();
                    int64_t globalDone = globalInitialForProgress - globalRemaining;
                    double done_frac = (globalInitialForProgress > 0) ? static_cast<double>(globalDone) / static_cast<double>(globalInitialForProgress) : 0.0;
                    double rate = (elapsed_s > 0) ? static_cast<double>(globalDone) / elapsed_s : 0.0;
                    double eta = (rate > 0) ? static_cast<double>(globalRemaining) / rate : 0.0;
                    RankHandler_t *selfHandler = this->rankHandlers[this->rank_world];
                    int localRemaining = selfHandler ? static_cast<int>(selfHandler->LocalSize()) : 0;
                    std::cerr << "[Progress] ~"
                              << (done_frac * 100.0) << "% done, "
                              << elapsed_s << "s elapsed, "
                              << "~" << eta << "s ETA, "
                              << "global_done=" << globalDone << "/" << globalInitialForProgress
                              << " rank0_local_remaining=" << localRemaining
                              << " rw_steps_total=" << globalCounters[MC_PROGRESS_RW_STEPS]
                              << " ddmc_steps_total=" << globalCounters[MC_PROGRESS_DDMC_STEPS]
                              << " ddmc_leaks=" << globalCounters[MC_PROGRESS_DDMC_LEAKS]
                              << " ddmc_census=" << globalCounters[MC_PROGRESS_DDMC_CENSUS]
                              << " ddmc_upscatter=" << globalCounters[MC_PROGRESS_DDMC_UPSCATTER]
                              << " ddmc_fallback=" << globalCounters[MC_PROGRESS_DDMC_FALLBACK]
                              << " eta_is_count_based=1"
                              << std::endl;
                }

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

#ifdef STORM_WITH_MPI
    if(this->rank_world != 0 && progressReportSendReq != MPI_REQUEST_NULL)
        MPI_Wait(&progressReportSendReq, MPI_STATUS_IGNORE);
#endif

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
        size_t localSize = handler->LocalSize();
        if(localSize != 0)
        {
            STORMError eo("End of RDMAMonteCarloManager::step: queue is not empty");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Head", static_cast<size_t>(handler->head));
            eo.addEntry("Tail", static_cast<size_t>(handler->tail));
            eo.addEntry("Particles", localSize);
            eo.addEntry("Peer Rank", handler->peer_rank_world);
            throw eo;
        }
    }
    for(rank_t rank = 0; rank < static_cast<rank_t>(this->detachedRankParticles.size()); rank++)
    {
        const std::vector<MCParticle> &particles = this->detachedRankParticles[static_cast<size_t>(rank)];
        if(not particles.empty())
        {
            STORMError eo("End of RDMAMonteCarloManager::step: detached particle list is not empty");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Peer Rank", rank);
            eo.addEntry("Detached Particles", particles.size());
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

#endif // RDMA_MONTE_CARLO_MANAGER_HPP
