#ifndef RDMA_MONTE_CARLO_MANAGER_HPP
#define RDMA_MONTE_CARLO_MANAGER_HPP

#include <cassert>
#include "mpi/mpi_commands.hpp"
#include "mpi/mpi_commands.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
#include "utils/amountManager/AmountManager.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "monte/utils/GhostMap.hpp"
#include "monte/utils/RankSync.hpp"
#include "utils/debug/vtune.h" // TODO: remove
#include "monte/manager/rdma/RankHandler2.hpp"
#include "monte/manager/ReallocationAgent.hpp"
#include "utils/debug/SmartTimer.hpp"
#include "misc/memory_debug.hpp"
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
#include "monte/manager/MonteCarloConfig.hpp"

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

#ifdef TIMING
namespace MonteCarloTimingDetail2
{
    struct Section
    {
        const char *name;
        double seconds;
    };

    inline double SecondsSince(const std::chrono::high_resolution_clock::time_point &start)
    {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    }

    inline void PrintTimingBreakdown(const MPI_Comm &comm, rank_t rank, const char *label, std::initializer_list<Section> sections)
    {
        rank_t size = 1;
        MPI_Comm_size(comm, &size);

        std::vector<const char*> names;
        std::vector<double> localSeconds;
        names.reserve(sections.size());
        localSeconds.reserve(sections.size());
        for(const Section &section : sections)
        {
            names.push_back(section.name);
            localSeconds.push_back(section.seconds);
        }

        std::vector<double> maxSeconds(localSeconds.size(), 0);
        std::vector<double> sumSeconds(localSeconds.size(), 0);
        MPI_Reduce(localSeconds.data(), maxSeconds.data(), static_cast<int>(localSeconds.size()), MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(localSeconds.data(), sumSeconds.data(), static_cast<int>(localSeconds.size()), MPI_DOUBLE, MPI_SUM, 0, comm);

        if(rank == 0)
        {
            std::cout << label << " (max/avg): ";
            for(size_t i = 0; i < localSeconds.size(); i++)
            {
                if(i > 0)
                {
                    std::cout << ", ";
                }
                std::cout << names[i] << "=" << maxSeconds[i] << "/" << sumSeconds[i] / size << "s";
            }
            std::cout << std::endl;
        }
    }
}

enum class SendFlushReason2
{
    Threshold,
    IdleDrain,
    VerifyDrain,
    FinalDrain
};
#endif // TIMING

template<typename T, typename Grid>
class RDMAMonteCarloManager
{
    using MCParticle = MonteCarloParticle<T, Grid>;
    using RankHandler_t = ::RankHandler2<T, Grid>;

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

    inline double GetPureComputeTime(void) const {
        #ifdef TIMING
        return this->pureComputeTime;
        #else
        return 0;
        #endif
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

        #ifdef RICH_MPI
            std::vector<MCParticle> GetLocalTrackParticleRoute(size_t id) const;
        #endif // RICH_MPI

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
    #ifdef TIMING
    double pureComputeTime = 0;
    size_t sendBufferFlushCalls = 0;
    size_t sendBufferFlushedParticles = 0;
    size_t sendBufferPeakRanks = 0;
    size_t sendBufferPeakParticles = 0;
    size_t sendBufferFlushThresholdCalls = 0;
    size_t sendBufferFlushIdleDrainCalls = 0;
    size_t sendBufferFlushVerifyDrainCalls = 0;
    size_t sendBufferFlushFinalDrainCalls = 0;
    size_t sendBufferThresholdParticles = 0;
    size_t sendBufferIdleDrainParticles = 0;
    size_t sendBufferVerifyDrainParticles = 0;
    size_t sendBufferFinalDrainParticles = 0;
    size_t sendBufferMaxBatchParticles = 0;
    size_t sendBufferMinNonzeroBatchParticles = 0;
    size_t sendBufferIdleHoldoffSkips = 0;
    size_t sendBufferIdleHoldoffPendingParticles = 0;
    #endif // TIMING

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

    #ifdef TIMING
    void FlushSendBuffers(bool flushSmallBuffers, double &transferTime);

    void FlushAllSendBuffers(SendFlushReason2 reason = SendFlushReason2::FinalDrain);
    #else
    void FlushSendBuffers(bool flushSmallBuffers);

    void FlushAllSendBuffers(void);
    #endif // TIMING

    bool AllSendBuffersEmpty(void) const;

    void PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum);

    #ifdef TIMING
    void RecordSendBufferFlush(size_t flushedParticles, SendFlushReason2 reason);

    void PrintTransferDiagnostics(double elapsed, double flushScanTime, double flushTransferTime) const;

    struct PrepareHandlersTiming
    {
        double oldNeighborSet = 0;
        double neighborList = 0;
        double selfHandler = 0;
        double findNewNeighbors = 0;
        double newNeighborAllreduce = 0;
        double createHandlers = 0;
        double handlerCtorRma = 0;
        double handlerCtorMutex = 0;
        double handlerCtorReset = 0;
        double handlerCtorPeerInfo = 0;
        double handlerCtorTotal = 0;
        double resetBuffers = 0;
        double total = 0;
        int localNewNeighbors = 0;
        int globalNewNeighbors = 0;
    };

    PrepareHandlersTiming lastPrepareHandlersTiming;
    #endif // TIMING
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
            UniversalError eo("RDMAMonteCarloManager<T, Grid>::AddParticles");
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
            UniversalError eo("Particle with the same ID is being added to the same rank twice");
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
            UniversalError eo("Trying to transfer particle to the same rank");
            eo.addEntry("Particle", particle);
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }

        rankToParticles[toRank].push_back(particle);

        #ifdef STORM_DEBUG
        if(toRank != particle.nextRank)
        {
            UniversalError eo("Particle will not be sent to the expected rank #1");
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
            UniversalError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #2");
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
                UniversalError eo("Trying to transfer particle to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }

            rankToParticles[toRank].push_back(particle);

            #ifdef STORM_DEBUG
            if(toRank != particle.nextRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #1");
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
            UniversalError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #2");
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

    #ifdef TIMING
    auto computeLoopStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING

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
                bool keepCurrent = false;
                bool debug = false;

                try
                {
                    #ifdef STORM_DEBUG
                    if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
                    {
                        UniversalError eo("Particle was already handled in this iteration");
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
                            UniversalError eo("Particle has invalid cell index (ghost)");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Cell Index", particle.cellIndex);
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Buffer of Rank", _rank);
                            throw eo;
                        }
                        if(particle.removedFromRank)
                        {
                            UniversalError eo("Particle was removed from rank, but still in the list");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Buffer of Rank", _rank);
                            throw eo;
                        }
                        if(not particle.checkedHere)
                        {
                            if(particle.nextRank != this->rank_world)
                            {
                                UniversalError eo("Particle Arrived to a Wrong Rank After Transfer");
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
                                UniversalError eo("Particle Arrived to a Wrong Rank After Transfer");
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

                        if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
                        {
                            size_t nextCellIndex = functionality.nextCellIndex;

                            assert(nextCellIndex != particle.cellIndex);
                            assert(particle.timeLeft >= 0);

                            if(BOOST_LIKELY(nextCellIndex < this->Ncells))
                            {
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
                                    UniversalError eo("Particle is in Wrong Location");
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
                                        keepCurrent = true;
                                    }
                                    else if(status == MonteCarloParticleStatus::REMOVE)
                                    {
                                        stepData.leavingCount++;
                                        this->allStepsCounter += particle.steps;
                                        this->localDecrementAmount += 1;
                                        removeCurrent = true;
                                    }
                                    else
                                    {
                                        UniversalError eo("Unknown boundary condition for particle");
                                        eo.addEntry("Particle", particle);
                                        eo.addEntry("Status", status);
                                        throw eo;
                                    }
                                    break;
                                }

                                particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                    MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                                auto [otherRank, neighborIndexInRank] = it->second;
                                #ifdef STORM_DEBUG
                                particle.checkedHere = false;
                                if(particle.nextRank != std::numeric_limits<rank_t>::max())
                                {
                                    UniversalError eo("Particle was already sent, and not sent again");
                                    eo.addEntry("Particle", particle);
                                    eo.addEntry("Already Transferred To Rank", particle.nextRank);
                                    eo.addEntry("Being Transferred To Rank", otherRank);
                                    eo.addEntry("Being Transferred To Index In Rank", neighborIndexInRank);
                                    throw eo;
                                }
                                const std::vector<rank_t> &neighbors = this->grid.GetDuplicatedProcs();
                                if(std::find(neighbors.cbegin(), neighbors.cend(), otherRank) == neighbors.cend())
                                {
                                    UniversalError eo("Particle is going to be transffered to a non-neighboring rank");
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
                                    UniversalError eo("Particle is going to be sent to the same rank");
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
                            UniversalError eo("Unknown Monte Carlo particle status");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Status", functionality.change);
                            throw eo;
                        }
                    }
                }
                catch(UniversalError &eo)
                {
                    eo.addEntry("Particle list index", particleIndex);
                    eo.addEntry("Handler rank buffer", _rank);
                    eo.addEntry("Handler head", static_cast<size_t>(handler->head));
                    eo.addEntry("Handler tail", static_cast<size_t>(handler->tail));
                    eo.addEntry("Handler buffer size", handler->buffsize);
                    throw eo;
                }

                if(removeCurrent)
                {
                    localParticles.pop_back();
                }
                else
                {
                    assert(keepCurrent);
                    break;
                }
        }

        if(not localParticles.empty())
        {
            assert(deferredParticles.empty());
            deferredParticles.swap(localParticles);
        }
    }

    #ifdef TIMING
    this->pureComputeTime += MonteCarloTimingDetail2::SecondsSince(computeLoopStart);
    #endif // TIMING

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

    #ifdef TIMING
    {
        size_t localRequested = shrinkList.size();
        size_t localCandidates = shrinkCandidates.size();
        size_t localPartners = shrinkPartners.size();
        size_t localWithoutHandler = candidatesWithoutHandler;
        size_t localMissingConfirmation = missingPeerConfirmation;

        size_t maxOriginalRequested = localOriginalRequested, sumOriginalRequested = localOriginalRequested;
        size_t maxRequested = localRequested, sumRequested = localRequested;
        size_t maxCandidates = localCandidates, sumCandidates = localCandidates;
        size_t maxPartners = localPartners, sumPartners = localPartners;
        size_t totalWithoutHandler = localWithoutHandler;
        size_t totalMissingConfirmation = localMissingConfirmation;

        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxOriginalRequested, &maxOriginalRequested, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumOriginalRequested, &sumOriginalRequested, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxRequested, &maxRequested, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumRequested, &sumRequested, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxCandidates, &maxCandidates, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumCandidates, &sumCandidates, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxPartners, &maxPartners, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumPartners, &sumPartners, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &totalWithoutHandler, &totalWithoutHandler, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &totalMissingConfirmation, &totalMissingConfirmation, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

        if(this->rank_world == 0)
        {
            std::cout << "ShrinkBuffers: originalRequested total=" << sumOriginalRequested
                      << ", localMax=" << maxOriginalRequested
                      << ", shrinkPercent=" << shrinkPercent
                      << ", requested total=" << sumRequested
                      << ", localMax=" << maxRequested
                      << ", candidates total=" << sumCandidates
                      << ", localMax=" << maxCandidates
                      << ", partners total=" << sumPartners
                      << ", localMax=" << maxPartners
                      << ", withoutHandler=" << totalWithoutHandler
                      << ", missingConfirmation=" << totalMissingConfirmation
                      << std::endl;
        }
    }
    #else
    (void)localOriginalRequested;
    (void)candidatesWithoutHandler;
    (void)missingPeerConfirmation;
    #endif // TIMING

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

#ifdef TIMING
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::RecordSendBufferFlush(size_t flushedParticles, SendFlushReason2 reason)
{
    this->sendBufferFlushCalls++;
    this->sendBufferFlushedParticles += flushedParticles;
    this->sendBufferMaxBatchParticles = std::max(this->sendBufferMaxBatchParticles, flushedParticles);
    if(flushedParticles > 0)
    {
        if(this->sendBufferMinNonzeroBatchParticles == 0)
        {
            this->sendBufferMinNonzeroBatchParticles = flushedParticles;
        }
        else
        {
            this->sendBufferMinNonzeroBatchParticles = std::min(this->sendBufferMinNonzeroBatchParticles, flushedParticles);
        }
    }

    switch(reason)
    {
        case SendFlushReason2::Threshold:
            this->sendBufferFlushThresholdCalls++;
            this->sendBufferThresholdParticles += flushedParticles;
            break;
        case SendFlushReason2::IdleDrain:
            this->sendBufferFlushIdleDrainCalls++;
            this->sendBufferIdleDrainParticles += flushedParticles;
            break;
        case SendFlushReason2::VerifyDrain:
            this->sendBufferFlushVerifyDrainCalls++;
            this->sendBufferVerifyDrainParticles += flushedParticles;
            break;
        case SendFlushReason2::FinalDrain:
            this->sendBufferFlushFinalDrainCalls++;
            this->sendBufferFinalDrainParticles += flushedParticles;
            break;
    }
}
#endif // TIMING

#ifdef TIMING
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::FlushSendBuffers(bool flushSmallBuffers, double &transferTime)
#else
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::FlushSendBuffers(bool flushSmallBuffers)
#endif // TIMING
{
    #ifdef TIMING
    size_t pendingParticles = this->sendBufferPendingParticles;
    size_t pendingRanks = this->sendBufferPendingRanks;
    #else
    size_t pendingRanks = this->sendBufferPendingRanks;
    #endif // TIMING

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
        #ifdef TIMING
        this->sendBufferPeakRanks = std::max(this->sendBufferPeakRanks, pendingRanks);
        this->sendBufferPeakParticles = std::max(this->sendBufferPeakParticles, pendingParticles);
        #endif // TIMING
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
        #ifdef TIMING
        auto transferStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        bool transferred = remoteHandler->TransferParticles(particles.data(), particles.size(), sourceLkey);
        #ifdef TIMING
        transferTime += MonteCarloTimingDetail2::SecondsSince(transferStart);
        #endif // TIMING
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
        #ifdef TIMING
        this->RecordSendBufferFlush(flushedParticles, thresholdFlush ? SendFlushReason2::Threshold : SendFlushReason2::IdleDrain);
        #endif // TIMING
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
    #ifdef TIMING
    if(heldIdleDrain)
    {
        this->sendBufferIdleHoldoffSkips++;
        this->sendBufferIdleHoldoffPendingParticles += pendingParticles;
    }
    this->sendBufferPeakRanks = std::max(this->sendBufferPeakRanks, pendingRanks);
    this->sendBufferPeakParticles = std::max(this->sendBufferPeakParticles, pendingParticles);
    #else
    (void)heldIdleDrain;
    #endif // TIMING
}

#ifdef TIMING
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::FlushAllSendBuffers(SendFlushReason2 reason)
#else
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::FlushAllSendBuffers(void)
#endif // TIMING
{
    #ifdef TIMING
    size_t pendingParticles = this->sendBufferPendingParticles;
    size_t pendingRanks = this->sendBufferPendingRanks;
    #endif // TIMING
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
        #ifdef TIMING
        this->RecordSendBufferFlush(flushedParticles, reason);
        #endif // TIMING
        this->sendBufferListed[rankIndex] = 0;
        this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
        this->sendBufferActiveRanks.pop_back();
    }
    #ifdef TIMING
    this->sendBufferPeakRanks = std::max(this->sendBufferPeakRanks, pendingRanks);
    this->sendBufferPeakParticles = std::max(this->sendBufferPeakParticles, pendingParticles);
    #endif // TIMING
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

#ifdef TIMING
template<typename T, typename Grid>
void RDMAMonteCarloManager<T, Grid>::PrintTransferDiagnostics(double elapsed, double flushScanTime, double flushTransferTime) const
{
    if(this->config.transferDiagnosticsLevel == MonteCarloTransferDiagnosticsLevel::Off)
    {
        return;
    }
    size_t every = std::max<size_t>(1, this->config.transferDiagnosticsEveryNSteps);
    if(this->currentStep % every != 0)
    {
        return;
    }

    enum CountIndex
    {
        FlushCalls,
        FlushedParticles,
        ThresholdCalls,
        IdleDrainCalls,
        VerifyDrainCalls,
        FinalDrainCalls,
        ThresholdParticles,
        IdleDrainParticles,
        VerifyDrainParticles,
        FinalDrainParticles,
        IdleHoldoffSkips,
        IdleHoldoffPendingParticles,
        MaxBatchParticles,
        PeakPendingRanks,
        PeakPendingParticles,
        TransferCalls,
        RemoteLockCalls,
        TransferReallocationRequests,
        TransferCallsWithReallocation,
        ContiguousPutCalls,
        ScatterPutCalls,
        ContiguousParticles,
        ScatterParticles,
        TransferCallsWithContiguousAllocation,
        TransferCallsWithoutContiguousAllocation,
        CountIndexSize
    };

    std::array<unsigned long long, CountIndexSize> localCounts{};
    localCounts[FlushCalls] = static_cast<unsigned long long>(this->sendBufferFlushCalls);
    localCounts[FlushedParticles] = static_cast<unsigned long long>(this->sendBufferFlushedParticles);
    localCounts[ThresholdCalls] = static_cast<unsigned long long>(this->sendBufferFlushThresholdCalls);
    localCounts[IdleDrainCalls] = static_cast<unsigned long long>(this->sendBufferFlushIdleDrainCalls);
    localCounts[VerifyDrainCalls] = static_cast<unsigned long long>(this->sendBufferFlushVerifyDrainCalls);
    localCounts[FinalDrainCalls] = static_cast<unsigned long long>(this->sendBufferFlushFinalDrainCalls);
    localCounts[ThresholdParticles] = static_cast<unsigned long long>(this->sendBufferThresholdParticles);
    localCounts[IdleDrainParticles] = static_cast<unsigned long long>(this->sendBufferIdleDrainParticles);
    localCounts[VerifyDrainParticles] = static_cast<unsigned long long>(this->sendBufferVerifyDrainParticles);
    localCounts[FinalDrainParticles] = static_cast<unsigned long long>(this->sendBufferFinalDrainParticles);
    localCounts[IdleHoldoffSkips] = static_cast<unsigned long long>(this->sendBufferIdleHoldoffSkips);
    localCounts[IdleHoldoffPendingParticles] = static_cast<unsigned long long>(this->sendBufferIdleHoldoffPendingParticles);
    localCounts[MaxBatchParticles] = static_cast<unsigned long long>(this->sendBufferMaxBatchParticles);
    localCounts[PeakPendingRanks] = static_cast<unsigned long long>(this->sendBufferPeakRanks);
    localCounts[PeakPendingParticles] = static_cast<unsigned long long>(this->sendBufferPeakParticles);

    enum TimeIndex
    {
        ManagerFlushScan,
        ManagerFlushTransfer,
        HandlerTransferTotal,
        HandlerLockWait,
        HandlerReallocationWait,
        HandlerAvailReserve,
        HandlerAvailIndexGet,
        HandlerParticlePut,
        HandlerTHLengthGet,
        HandlerTHPut,
        HandlerTHLengthPublish,
        HandlerAVLengthFlush,
        HandlerUnlock,
        TimeIndexSize
    };

    std::array<double, TimeIndexSize> localTimes{};
    localTimes[ManagerFlushScan] = flushScanTime;
    localTimes[ManagerFlushTransfer] = flushTransferTime;

    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        localCounts[TransferCalls] += static_cast<unsigned long long>(handler->transferCallsThisStep);
        localCounts[RemoteLockCalls] += static_cast<unsigned long long>(handler->remoteLockCallsThisStep);
        localCounts[TransferReallocationRequests] += static_cast<unsigned long long>(handler->transferReallocationRequestsThisStep);
        localCounts[TransferCallsWithReallocation] += static_cast<unsigned long long>(handler->transferCallsWithReallocationThisStep);
        localCounts[ContiguousPutCalls] += static_cast<unsigned long long>(handler->contiguousParticlePutsThisStep);
        localCounts[ScatterPutCalls] += static_cast<unsigned long long>(handler->scatterParticlePutsThisStep);
        localCounts[ContiguousParticles] += static_cast<unsigned long long>(handler->contiguousParticlesThisStep);
        localCounts[ScatterParticles] += static_cast<unsigned long long>(handler->scatterParticlesThisStep);
        localCounts[TransferCallsWithContiguousAllocation] += static_cast<unsigned long long>(handler->transferCallsWithContiguousAllocationThisStep);
        localCounts[TransferCallsWithoutContiguousAllocation] += static_cast<unsigned long long>(handler->transferCallsWithoutContiguousAllocationThisStep);

        localTimes[HandlerTransferTotal] += handler->transferTotalTimeThisStep;
        localTimes[HandlerLockWait] += handler->transferLockWaitTimeThisStep;
        localTimes[HandlerReallocationWait] += handler->transferReallocationWaitTimeThisStep;
        localTimes[HandlerAvailReserve] += handler->transferAvailReserveTimeThisStep;
        localTimes[HandlerAvailIndexGet] += handler->transferAvailIndexGetTimeThisStep;
        localTimes[HandlerParticlePut] += handler->transferParticlePutTimeThisStep;
        localTimes[HandlerTHLengthGet] += handler->transferTHLengthGetTimeThisStep;
        localTimes[HandlerTHPut] += handler->transferTHPutTimeThisStep;
        localTimes[HandlerTHLengthPublish] += handler->transferTHLengthPublishTimeThisStep;
        localTimes[HandlerAVLengthFlush] += handler->transferAVLengthFlushTimeThisStep;
        localTimes[HandlerUnlock] += handler->transferUnlockTimeThisStep;
    }

    std::array<unsigned long long, CountIndexSize> sumCounts{};
    std::array<unsigned long long, CountIndexSize> maxCounts{};
    MPI_Reduce(localCounts.data(), sumCounts.data(), CountIndexSize, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(localCounts.data(), maxCounts.data(), CountIndexSize, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);

    std::array<double, TimeIndexSize> sumTimes{};
    std::array<double, TimeIndexSize> maxTimes{};
    MPI_Reduce(localTimes.data(), sumTimes.data(), TimeIndexSize, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(localTimes.data(), maxTimes.data(), TimeIndexSize, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);

    auto [maxTransferRank, maxTransferTime] = MPI_Max_loc(localTimes[HandlerTransferTotal], this->comm_world);
    auto [maxParticlePutRank, maxParticlePutTime] = MPI_Max_loc(localTimes[HandlerParticlePut], this->comm_world);
    auto [maxLockRank, maxLockTime] = MPI_Max_loc(localTimes[HandlerLockWait], this->comm_world);
    auto [maxReallocationRank, maxReallocationTime] = MPI_Max_loc(localTimes[HandlerReallocationWait], this->comm_world);
    auto [maxFlushFractionRank, maxFlushFraction] = MPI_Max_loc(elapsed > 0 ? flushTransferTime / elapsed : 0, this->comm_world);

    if(this->rank_world != 0)
    {
        return;
    }

    double particlesPerFlush = sumCounts[FlushCalls] > 0
        ? static_cast<double>(sumCounts[FlushedParticles]) / static_cast<double>(sumCounts[FlushCalls])
        : 0;
    unsigned long long totalMovedParticles = sumCounts[ContiguousParticles] + sumCounts[ScatterParticles];
    double scatterFraction = totalMovedParticles > 0
        ? static_cast<double>(sumCounts[ScatterParticles]) / static_cast<double>(totalMovedParticles)
        : 0;

    auto avgTime = [this, &sumTimes](TimeIndex idx)
    {
        return sumTimes[idx] / this->size_world;
    };

    std::cout << "Transfer diagnostics: flush calls sum/localMax=" << sumCounts[FlushCalls] << "/" << maxCounts[FlushCalls]
              << ", particles sum=" << sumCounts[FlushedParticles]
              << ", particlesPerFlush=" << particlesPerFlush
              << ", maxBatch localMax=" << maxCounts[MaxBatchParticles]
              << ", pending ranks/particles peak=" << maxCounts[PeakPendingRanks] << "/" << maxCounts[PeakPendingParticles]
              << std::endl;
    std::cout << "Transfer diagnostics reasons: calls threshold/idle/verify/final="
              << sumCounts[ThresholdCalls] << "/" << sumCounts[IdleDrainCalls] << "/"
              << sumCounts[VerifyDrainCalls] << "/" << sumCounts[FinalDrainCalls]
              << ", particles threshold/idle/verify/final="
              << sumCounts[ThresholdParticles] << "/" << sumCounts[IdleDrainParticles] << "/"
              << sumCounts[VerifyDrainParticles] << "/" << sumCounts[FinalDrainParticles]
              << ", idleHoldoff skips/pendingParticles=" << sumCounts[IdleHoldoffSkips]
              << "/" << sumCounts[IdleHoldoffPendingParticles]
              << std::endl;
    std::cout << "Transfer diagnostics puts: contiguous/scatter calls="
              << sumCounts[ContiguousPutCalls] << "/" << sumCounts[ScatterPutCalls]
              << ", contiguous/scatter particles=" << sumCounts[ContiguousParticles] << "/" << sumCounts[ScatterParticles]
              << ", scatterFraction=" << scatterFraction
              << ", transfer calls contiguousAllocation yes/no="
              << sumCounts[TransferCallsWithContiguousAllocation] << "/"
              << sumCounts[TransferCallsWithoutContiguousAllocation]
              << ", transfer calls sum/localMax=" << sumCounts[TransferCalls] << "/" << maxCounts[TransferCalls]
              << ", remote locks sum/localMax=" << sumCounts[RemoteLockCalls] << "/" << maxCounts[RemoteLockCalls]
              << ", realloc requests sum/localMax=" << sumCounts[TransferReallocationRequests] << "/" << maxCounts[TransferReallocationRequests]
              << ", transfer calls with realloc sum/localMax=" << sumCounts[TransferCallsWithReallocation] << "/" << maxCounts[TransferCallsWithReallocation]
              << std::endl;
    std::cout << "Transfer diagnostics timing (max/avg): managerScan=" << maxTimes[ManagerFlushScan] << "/" << avgTime(ManagerFlushScan)
              << "s, managerTransfer=" << maxTimes[ManagerFlushTransfer] << "/" << avgTime(ManagerFlushTransfer)
              << "s, handlerTransfer=" << maxTimes[HandlerTransferTotal] << "/" << avgTime(HandlerTransferTotal)
              << "s, lock=" << maxTimes[HandlerLockWait] << "/" << avgTime(HandlerLockWait)
              << "s, reallocWait=" << maxTimes[HandlerReallocationWait] << "/" << avgTime(HandlerReallocationWait)
              << "s, reserve=" << maxTimes[HandlerAvailReserve] << "/" << avgTime(HandlerAvailReserve)
              << "s, avGet=" << maxTimes[HandlerAvailIndexGet] << "/" << avgTime(HandlerAvailIndexGet)
              << "s, particlePut=" << maxTimes[HandlerParticlePut] << "/" << avgTime(HandlerParticlePut)
              << "s, thGet=" << maxTimes[HandlerTHLengthGet] << "/" << avgTime(HandlerTHLengthGet)
              << "s, thPut=" << maxTimes[HandlerTHPut] << "/" << avgTime(HandlerTHPut)
              << "s, thPublish=" << maxTimes[HandlerTHLengthPublish] << "/" << avgTime(HandlerTHLengthPublish)
              << "s, avFlush=" << maxTimes[HandlerAVLengthFlush] << "/" << avgTime(HandlerAVLengthFlush)
              << "s, unlock=" << maxTimes[HandlerUnlock] << "/" << avgTime(HandlerUnlock)
              << "s" << std::endl;
    std::cout << "Transfer diagnostics max ranks: handlerTransfer=" << maxTransferRank << "(" << maxTransferTime
              << "s), particlePut=" << maxParticlePutRank << "(" << maxParticlePutTime
              << "s), lock=" << maxLockRank << "(" << maxLockTime
              << "s), reallocWait=" << maxReallocationRank << "(" << maxReallocationTime
              << "s), sameRankFlushTransferFraction=" << maxFlushFractionRank << "(" << maxFlushFraction << ")"
              << std::endl;
}
#endif // TIMING

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
    #ifdef TIMING
    PrepareHandlersTiming timing;
    auto totalStart = std::chrono::high_resolution_clock::now();
    auto sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING

    boost::container::flat_set<rank_t> oldNeighbors(this->neighbors.cbegin(), this->neighbors.cend());

    #ifdef TIMING
    timing.oldNeighborSet = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->neighbors = GetNeighborList2(this->grid, this->ranks_ghost_map);
    #ifdef TIMING
    timing.neighborList = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    #endif // TIMING

    // Self handler: 1-process communicator, no coordination needed
    #ifdef TIMING
    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    if(this->rankHandlers[this->rank_world] == nullptr)
    {
        MPI_Comm_dup(MPI_COMM_SELF, &this->communicators[this->rank_world]);
        this->rankHandlers[this->rank_world] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, this->communicators[this->rank_world], this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
        #ifdef TIMING
        timing.handlerCtorRma += this->rankHandlers[this->rank_world]->constructionRmaTime;
        timing.handlerCtorMutex += this->rankHandlers[this->rank_world]->constructionMutexTime;
        timing.handlerCtorReset += this->rankHandlers[this->rank_world]->constructionResetTime;
        timing.handlerCtorPeerInfo += this->rankHandlers[this->rank_world]->constructionPeerInfoTime;
        timing.handlerCtorTotal += this->rankHandlers[this->rank_world]->constructionTotalTime;
        #endif // TIMING
    }
    #ifdef TIMING
    timing.selfHandler = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    std::vector<rank_t> newNeighbors;
    for(rank_t rank : this->neighbors)
    {
        if(oldNeighbors.find(rank) == oldNeighbors.end() and this->rankHandlers[rank] == nullptr)
        {
            newNeighbors.push_back(rank);
        }
    }
    #ifdef TIMING
    timing.findNewNeighbors = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    #endif // TIMING

    int numNewNeighbors = newNeighbors.size();
    #ifdef TIMING
    timing.localNewNeighbors = numNewNeighbors;
    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    MPI_Allreduce(MPI_IN_PLACE, &numNewNeighbors, 1, MPI_INT, MPI_SUM, this->comm_world);
    #ifdef TIMING
    timing.newNeighborAllreduce = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    timing.globalNewNeighbors = numNewNeighbors;
    #endif // TIMING

    if(numNewNeighbors > 0)
    {
        #ifdef TIMING
        sectionStart = std::chrono::high_resolution_clock::now();
        auto createHandler = [this, &timing](rank_t rank, MPI_Comm pair_comm)
        #else
        auto createHandler = [this](rank_t rank, MPI_Comm pair_comm)
        #endif // TIMING
        {
            if(this->rankHandlers[rank] != nullptr)
            {
                return;
            }

            this->communicators[rank] = pair_comm;
            this->rankHandlers[rank] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, pair_comm, this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
            #ifdef TIMING
            timing.handlerCtorRma += this->rankHandlers[rank]->constructionRmaTime;
            timing.handlerCtorMutex += this->rankHandlers[rank]->constructionMutexTime;
            timing.handlerCtorReset += this->rankHandlers[rank]->constructionResetTime;
            timing.handlerCtorPeerInfo += this->rankHandlers[rank]->constructionPeerInfoTime;
            timing.handlerCtorTotal += this->rankHandlers[rank]->constructionTotalTime;
            #endif // TIMING
            if(this->rankHandlers[rank]->peer_rank_world != rank)
            {
                UniversalError eo("Peer rank world does not match");
                eo.addEntry("Rank", rank);
                eo.addEntry("Peer Rank World", this->rankHandlers[rank]->peer_rank_world);
                throw eo;
            }
        };
        ForEachRankSyncByList(this->comm_world, newNeighbors, createHandler);
        #ifdef TIMING
        timing.createHandlers = MonteCarloTimingDetail2::SecondsSince(sectionStart);
        #endif // TIMING
    }

    #ifdef TIMING
    if(this->rank_world == 0)
    {
        std::cout << "Number of new neighbors: " << numNewNeighbors << std::endl;
    }

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->ResetAllBuffers();
    #ifdef TIMING
    timing.resetBuffers = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    timing.total = MonteCarloTimingDetail2::SecondsSince(totalStart);
    this->lastPrepareHandlersTiming = timing;
    #endif // TIMING
}

template<typename T, typename Grid>
std::vector<typename RDMAMonteCarloManager<T, Grid>::MCParticle> RDMAMonteCarloManager<T, Grid>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    // if(this->Ncells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rank_world << ": " << this->Ncells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }
    #ifdef TIMING
    auto managerStepStart = std::chrono::high_resolution_clock::now();
    START_TIMER_PREEMPTIVE("Initialization");

    auto initStart = std::chrono::high_resolution_clock::now();
    double initGridMetaTime = 0;
    double initPrepareHandlersTime = 0;
    double initClearSendBuffersTime = 0;
    double initShrinkBuffersTime = 0;
    double initPutSelfParticlesTime = 0;
    double initUpdateGridDataTime = 0;
    double initPreStepTime = 0;
    double initResetParticleStateTime = 0;
    double initAddParticlesTime = 0;
    double initBarrierTime = 0;
    double amountManagerTime = 0;
    double memoryDiagnosticsTime = 0;
    double endStepShrinkBuffersTime = 0;

    auto sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING

    this->Ncells = this->grid.GetPointNo();
    this->ranks_ghost_map = GetGhostMap(this->grid);
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();

    #ifdef TIMING
    initGridMetaTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->PrepareHandlers();

    #ifdef TIMING
    initPrepareHandlersTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->ResetSendBuffers();
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;

    #ifdef TIMING
    initClearSendBuffersTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    #endif // TIMING

    bool didRebalance = this->grid.DidRebalance() and (this->lastBuildGeneration != this->grid.GetBuildGeneration());
    if(didRebalance)
    {
        #ifdef TIMING
        if(this->rank_world == 0)
        {
            std::cout << "Doing shrink because of rebalance" << std::endl;
        }
        sectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        this->ShrinkBuffers();
        #ifdef TIMING
        initShrinkBuffersTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
        #endif // TIMING
    }
    this->lastBuildGeneration = this->grid.GetBuildGeneration();

    size_t initialParticlesNum = particleList.size();
    this->initialParticleCount = initialParticlesNum;
    this->cellsParticleCounters.assign(this->Ncells, 0);
    for(const auto &p : particleList) this->cellsParticleCounters[p.cellIndex]++;
    #ifdef TIMING
    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->PutSelfParticles(std::move(particleList));
    #ifdef TIMING
    initPutSelfParticlesTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    START_TIMER_PREEMPTIVE("Prestep");

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->physics->updateGridData();

    #ifdef TIMING
    initUpdateGridDataTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    std::chrono::high_resolution_clock::time_point preStepStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);

    #ifdef TIMING
    std::chrono::high_resolution_clock::time_point preStepEnd = std::chrono::high_resolution_clock::now();

    double preStepSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(preStepEnd - preStepStart).count();
    initPreStepTime = preStepSeconds;
    auto [maxPreStepRank, maxPreStepTime] = MPI_Max_loc(preStepSeconds, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepSeconds, &preStepSeconds, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    if(this->rank_world == 0)
    {
        std::cout << "Prestep time: avg=" << preStepSeconds / this->size_world << "s, max=" << maxPreStepTime << "s (rank " << maxPreStepRank << ")" << std::endl;
    }
    #endif // TIMING

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

    #ifdef TIMING
    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        #ifdef TIMING
        handler->reallocationTime = 0;
        handler->reallocationsThisStep = 0;
        handler->peakBufferUsage = 0;
        handler->transferCallsThisStep = 0;
        handler->contiguousParticlePutsThisStep = 0;
        handler->contiguousParticlesThisStep = 0;
        handler->scatterParticlePutsThisStep = 0;
        handler->scatterParticlesThisStep = 0;
        handler->transferCallsWithContiguousAllocationThisStep = 0;
        handler->transferCallsWithoutContiguousAllocationThisStep = 0;
        handler->transferReallocationRequestsThisStep = 0;
        handler->transferCallsWithReallocationThisStep = 0;
        handler->remoteLockCallsThisStep = 0;
        handler->transferTotalTimeThisStep = 0;
        handler->transferLockWaitTimeThisStep = 0;
        handler->transferReallocationWaitTimeThisStep = 0;
        handler->transferAvailReserveTimeThisStep = 0;
        handler->transferAvailIndexGetTimeThisStep = 0;
        handler->transferParticlePutTimeThisStep = 0;
        handler->transferTHLengthGetTimeThisStep = 0;
        handler->transferTHPutTimeThisStep = 0;
        handler->transferTHLengthPublishTimeThisStep = 0;
        handler->transferAVLengthFlushTimeThisStep = 0;
        handler->transferUnlockTimeThisStep = 0;
        #endif // TIMING

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
            p.initialWeight = p.weight;
            p.steps = 0;
        });
    }
    #ifdef TIMING
    initResetParticleStateTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    auto addParticlesStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    {
        #ifdef TIMING
        START_TIMER("Adding Particles");
        #endif // TIMING
        this->AddParticles(newParticles1);
        std::vector<MCParticle>().swap(newParticles1);
    }
    #ifdef TIMING
    initAddParticlesTime = MonteCarloTimingDetail2::SecondsSince(addParticlesStart);

    sectionStart = std::chrono::high_resolution_clock::now();
    MPI_Barrier(this->comm_world);
    initBarrierTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    double initTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - initStart).count();
    #else
    MPI_Barrier(this->comm_world);
    #endif // TIMING

    size_t numParticles = initialParticlesNum + preStepParticlesNum;
    #ifdef TIMING
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &numParticles, &numParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    {
        int localParticleCount = static_cast<int>(initialParticlesNum + preStepParticlesNum);
        auto [maxParticleRank, maxParticleCount] = MPI_Max_loc(localParticleCount, this->comm_world);
        if(this->rank_world == 0)
        {
            std::cout << "Particles per rank: avg=" << static_cast<double>(numParticles) / this->size_world
                      << ", max=" << maxParticleCount << " (rank " << maxParticleRank << ")" << std::endl;
        }
    }
    #endif // TIMING

    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;
    #ifdef TIMING
    this->pureComputeTime = 0;
    this->sendBufferFlushCalls = 0;
    this->sendBufferFlushedParticles = 0;
    this->sendBufferPeakRanks = 0;
    this->sendBufferPeakParticles = 0;
    this->sendBufferFlushThresholdCalls = 0;
    this->sendBufferFlushIdleDrainCalls = 0;
    this->sendBufferFlushVerifyDrainCalls = 0;
    this->sendBufferFlushFinalDrainCalls = 0;
    this->sendBufferThresholdParticles = 0;
    this->sendBufferIdleDrainParticles = 0;
    this->sendBufferVerifyDrainParticles = 0;
    this->sendBufferFinalDrainParticles = 0;
    this->sendBufferMaxBatchParticles = 0;
    this->sendBufferMinNonzeroBatchParticles = 0;
    this->sendBufferIdleHoldoffSkips = 0;
    this->sendBufferIdleHoldoffPendingParticles = 0;

    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    AmountManager amountManager(this->comm_world);
    amountManager.Initialize(startingParticleNum);
    #ifdef TIMING
    amountManagerTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    #endif // TIMING

    MonteCarloStepFinalData data;
    size_t numOfCounterDecrementations = 0;
    #ifdef TIMING
    double addParticlesTime = initAddParticlesTime;
    #endif // TIMING

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

    #ifdef TIMING
    sectionStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->PrintMemoryDiagnostics(initialParticlesNum, preStepParticlesNum);
    #ifdef TIMING
    memoryDiagnosticsTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);

    auto start = std::chrono::high_resolution_clock::now();
    #endif // TIMING

    const bool &verify = amountManager.GetVerifyRef();
    const bool &done = amountManager.GetDoneRef();
    #ifdef TIMING
    double mainReallocationPollTime = 0;
    double mainHandleAllTime = 0;
    double mainFlushSendBuffersTime = 0;
    double mainFlushSendScanTime = 0;
    double mainFlushSendTransferTime = 0;
    double mainAmountDecreaseTime = 0;
    double mainAmountProgressTime = 0;
    double mainVerifyTime = 0;
    double mainExitBarrierTime = 0;
    size_t mainVerifyCount = 0;
    size_t mainAmountProgressCalls = 0;
    #endif // TIMING

    MEMORY_DEBUG_PRINT("Before main loop in MCM");
    #ifdef TIMING
    START_TIMER_PREEMPTIVE("Main Loop");
    #endif // TIMING

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
            bool shouldProgressReallocations = (not usesAsyncReallocation) or
                (this->iteration % reallocationProgressMinCycles == 0) or
                this->reallocationAgent->HasPendingAsyncReallocations();
            #ifdef TIMING
            auto mainLoopSectionStart = std::chrono::high_resolution_clock::now();
            #endif // TIMING
            if(shouldProgressReallocations)
            {
                this->ProgressReallocations();
            }
            #ifdef TIMING
            mainReallocationPollTime += MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);

            mainLoopSectionStart = std::chrono::high_resolution_clock::now();
            #endif // TIMING
            bool localWorkDone = this->HandleAll(data);
            #ifdef TIMING
            mainHandleAllTime += MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);

            mainLoopSectionStart = std::chrono::high_resolution_clock::now();
            double flushTransferTime = 0;
            this->FlushSendBuffers(localWorkDone, flushTransferTime);
            double flushTotalTime = MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);
            mainFlushSendBuffersTime += flushTotalTime;
            mainFlushSendTransferTime += flushTransferTime;
            mainFlushSendScanTime += std::max(0.0, flushTotalTime - flushTransferTime);
            #else
            this->FlushSendBuffers(localWorkDone);
            #endif // TIMING

            #ifdef TIMING
            mainLoopSectionStart = std::chrono::high_resolution_clock::now();
            #endif // TIMING
            amountManager.Decrease(this->localDecrementAmount);
            this->localDecrementAmount = 0;
            #ifdef TIMING
            mainAmountDecreaseTime += MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);
            #endif // TIMING

            if(this->iteration % amountProgressMinCycles == 0)
            {
                #ifdef TIMING
                mainLoopSectionStart = std::chrono::high_resolution_clock::now();
                #endif // TIMING
                amountManager.Progress();
                #ifdef TIMING
                mainAmountProgressTime += MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);
                mainAmountProgressCalls++;
                #endif // TIMING
            }

            if(verify)
            {
                #ifdef TIMING
                mainVerifyCount++;
                mainLoopSectionStart = std::chrono::high_resolution_clock::now();
                this->FlushAllSendBuffers(SendFlushReason2::VerifyDrain);
                #else
                this->FlushAllSendBuffers();
                #endif // TIMING
                this->ProgressReallocations();
                bool ok = this->AllSendBuffersEmpty() and not this->reallocationAgent->HasPendingAsyncReallocations();
                amountManager.Verify(ok);
                #ifdef TIMING
                mainVerifyTime += MonteCarloTimingDetail2::SecondsSince(mainLoopSectionStart);
                #endif // TIMING
            }

            this->iteration++;
        }
    }
    catch(const UniversalError &eo)
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

    #ifdef TIMING
    sectionStart = std::chrono::high_resolution_clock::now();
    MPI_Barrier(this->comm_world);
    mainExitBarrierTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    {
        size_t myNumSteps = 0;
        for(size_t counter : this->cellsStepsCounters)
        {
            myNumSteps += counter;
        }
        double localCompute = this->pureComputeTime;
        struct { double val; int rank; } myCompute, maxCompute;
        myCompute.val = localCompute;
        myCompute.rank = this->rank_world;
        MPI_Allreduce(&myCompute, &maxCompute, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm_world);
        double avgCompute = localCompute;
        MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &avgCompute, &avgCompute, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

        struct { long val; int rank; } myParticles, minParticles, maxParticles;
        myParticles.val = static_cast<long>(this->startParticleCount);
        myParticles.rank = this->rank_world;
        MPI_Allreduce(&myParticles, &minParticles, 1, MPI_LONG_INT, MPI_MINLOC, this->comm_world);
        MPI_Allreduce(&myParticles, &maxParticles, 1, MPI_LONG_INT, MPI_MAXLOC, this->comm_world);

        struct { long val; int rank; } mySteps, minSteps, maxSteps;
        mySteps.val = static_cast<long>(myNumSteps);
        mySteps.rank = this->rank_world;
        MPI_Allreduce(&mySteps, &minSteps, 1, MPI_LONG_INT, MPI_MINLOC, this->comm_world);
        MPI_Allreduce(&mySteps, &maxSteps, 1, MPI_LONG_INT, MPI_MAXLOC, this->comm_world);

        size_t steps_maxComputeRank = myNumSteps;
        MPI_Bcast(&steps_maxComputeRank, 1, MPI_UNSIGNED_LONG_LONG, maxCompute.rank, this->comm_world);
        size_t steps_minParticlesRank = myNumSteps;
        MPI_Bcast(&steps_minParticlesRank, 1, MPI_UNSIGNED_LONG_LONG, minParticles.rank, this->comm_world);
        size_t steps_maxParticlesRank = myNumSteps;
        MPI_Bcast(&steps_maxParticlesRank, 1, MPI_UNSIGNED_LONG_LONG, maxParticles.rank, this->comm_world);

        size_t particles_maxComputeRank = this->startParticleCount;
        MPI_Bcast(&particles_maxComputeRank, 1, MPI_UNSIGNED_LONG_LONG, maxCompute.rank, this->comm_world);
        size_t particles_minStepsRank = this->startParticleCount;
        MPI_Bcast(&particles_minStepsRank, 1, MPI_UNSIGNED_LONG_LONG, minSteps.rank, this->comm_world);
        size_t particles_maxStepsRank = this->startParticleCount;
        MPI_Bcast(&particles_maxStepsRank, 1, MPI_UNSIGNED_LONG_LONG, maxSteps.rank, this->comm_world);

        size_t myCells = this->Ncells;
        size_t cells_minParticlesRank = myCells;
        MPI_Bcast(&cells_minParticlesRank, 1, MPI_UNSIGNED_LONG_LONG, minParticles.rank, this->comm_world);
        size_t cells_maxParticlesRank = myCells;
        MPI_Bcast(&cells_maxParticlesRank, 1, MPI_UNSIGNED_LONG_LONG, maxParticles.rank, this->comm_world);
        size_t cells_maxComputeRank = myCells;
        MPI_Bcast(&cells_maxComputeRank, 1, MPI_UNSIGNED_LONG_LONG, maxCompute.rank, this->comm_world);
        size_t cells_minStepsRank = myCells;
        MPI_Bcast(&cells_minStepsRank, 1, MPI_UNSIGNED_LONG_LONG, minSteps.rank, this->comm_world);
        size_t cells_maxStepsRank = myCells;
        MPI_Bcast(&cells_maxStepsRank, 1, MPI_UNSIGNED_LONG_LONG, maxSteps.rank, this->comm_world);

        double compute_minParticlesRank = localCompute;
        MPI_Bcast(&compute_minParticlesRank, 1, MPI_DOUBLE, minParticles.rank, this->comm_world);
        double compute_maxParticlesRank = localCompute;
        MPI_Bcast(&compute_maxParticlesRank, 1, MPI_DOUBLE, maxParticles.rank, this->comm_world);
        double compute_minStepsRank = localCompute;
        MPI_Bcast(&compute_minStepsRank, 1, MPI_DOUBLE, minSteps.rank, this->comm_world);
        double compute_maxStepsRank = localCompute;
        MPI_Bcast(&compute_maxStepsRank, 1, MPI_DOUBLE, maxSteps.rank, this->comm_world);

        if(this->rank_world == 0)
        {
            std::cout << "Pure particle compute time: max=" << maxCompute.val << "s (rank " << maxCompute.rank << ", " << particles_maxComputeRank << " particles), avg=" << avgCompute / this->size_world << "s" << std::endl;
            std::cout << "Step counts: "
                      << "least-particles rank " << minParticles.rank << " (" << minParticles.val << " particles, " << steps_minParticlesRank << " steps, " << cells_minParticlesRank << " cells, " << compute_minParticlesRank << "s compute), "
                      << "most-particles rank " << maxParticles.rank << " (" << maxParticles.val << " particles, " << steps_maxParticlesRank << " steps, " << cells_maxParticlesRank << " cells, " << compute_maxParticlesRank << "s compute), "
                      << "max-compute rank " << maxCompute.rank << " (" << particles_maxComputeRank << " particles, " << steps_maxComputeRank << " steps, " << cells_maxComputeRank << " cells, " << maxCompute.val << "s compute)"
                      << std::endl;
            std::cout << "Step extremes: "
                      << "min-steps rank " << minSteps.rank << " (" << minSteps.val << " steps, " << particles_minStepsRank << " particles, " << cells_minStepsRank << " cells, " << compute_minStepsRank << "s compute), "
                      << "max-steps rank " << maxSteps.rank << " (" << maxSteps.val << " steps, " << particles_maxStepsRank << " particles, " << cells_maxStepsRank << " cells, " << compute_maxStepsRank << "s compute)"
                      << std::endl;
        }
    }
    #endif // TIMING

    #ifdef TIMING
    START_TIMER_PREEMPTIVE("Boundary Condition");

    MPI_Barrier(this->comm_world);
    start = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    data.remaining = this->populationControl->activate(data.remaining);
    #ifdef TIMING
    MPI_Barrier(this->comm_world);
    end = std::chrono::high_resolution_clock::now();
    double populationControlTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    START_TIMER_PREEMPTIVE("Poststep");
    MPI_Barrier(this->comm_world);
    start = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    this->physics->postStep(data.remaining, fullDt);
    #ifdef TIMING
    MPI_Barrier(this->comm_world);
    end = std::chrono::high_resolution_clock::now();
    double postStepTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    START_TIMER_PREEMPTIVE("Diagnostics");
    auto diagnosticsStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING

    size_t newParticlesNum = data.remaining.size();
    this->endParticleCount = newParticlesNum;
    #ifdef TIMING
    size_t leavingNumber = data.leavingCount;

    size_t totalSteps = this->allStepsCounter;
    size_t totalCounterDecrementations = numOfCounterDecrementations;
    size_t callsToTransfer = this->transfersCounter;

    int myStepsCount = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        myStepsCount += static_cast<int>(counter);
    }
    auto [maxStepsRank, maxStepsVal] = MPI_Max_loc(myStepsCount, this->comm_world);
    auto [maxTransfersRank, maxTransfersVal] = MPI_Max_loc(static_cast<int>(this->transfersCounter), this->comm_world);

    double reallocationTime = 0, maxReallocationTime = 0;
    size_t totalReallocations = 0;
    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        reallocationTime += handler->reallocationTime;
        totalReallocations += handler->reallocationsThisStep;
    }

    double localReallocationTime = reallocationTime;

    // std::cout << "leavingNumber = " << leavingNumber << " and newParticlesNum = " << newParticlesNum << std::endl;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &initialParticlesNum, &initialParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepParticlesNum, &preStepParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &leavingNumber, &leavingNumber, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &newParticlesNum, &newParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalSteps, &totalSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalCounterDecrementations, &totalCounterDecrementations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &startingParticleNum, &startingParticleNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &callsToTransfer, &callsToTransfer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(&reallocationTime, &maxReallocationTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &reallocationTime, &reallocationTime, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    if(this->rank_world == 0)
    {
        std::cout << "Elapsed: " << elapsed << " seconds, max " << maxReallocationTime << " in reallocation (average: " << reallocationTime / this->size_world << ")" << std::endl;
        std::cout << "Started with " << startingParticleNum << ". Came with " << initialParticlesNum << ". Generated " << preStepParticlesNum << " particles in preStep. ";
        std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << ". ";
        std::cout << "Total steps: " << totalSteps << ", total counter decrementations: " << totalCounterDecrementations << std::endl;
        std::cout << "Population control time: " << populationControlTime << ", post step time: " << postStepTime << std::endl;
    }
    std::cout.flush();
    MPI_Barrier(this->comm_world);
    #endif // TIMING

    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        size_t localSize = handler->LocalSize();
        if(localSize != 0)
        {
            UniversalError eo("End of RDMAMonteCarloManager::step: queue is not empty");
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
            UniversalError eo("End of RDMAMonteCarloManager::step: detached particle list is not empty");
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
            #ifdef TIMING
            if(this->rank_world == 0)
            {
                std::cout << "Doing shrink becuase of step number (currentStep=" << this->currentStep << ", shrinkBuffersCycle=" << this->config.shrinkBuffersCycle << ")" << std::endl;
            }
            sectionStart = std::chrono::high_resolution_clock::now();
            #endif // TIMING
            this->ShrinkBuffers();
            #ifdef TIMING
            endStepShrinkBuffersTime = MonteCarloTimingDetail2::SecondsSince(sectionStart);
            #endif // TIMING
        }
    }

    #ifdef TIMING
    double diagnosticsTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - diagnosticsStart).count();
    double managerTotalTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - managerStepStart).count();
    double localInitTime = initTime;
    double localManagerTotalTime = managerTotalTime;
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &initTime, &initTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &addParticlesTime, &addParticlesTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &diagnosticsTime, &diagnosticsTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &managerTotalTime, &managerTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        double accounted = maxPreStepTime + elapsed + populationControlTime + postStepTime;
        std::cout << "Manager breakdown (max): init=" << initTime << "s, addParticles=" << addParticlesTime
                  << "s, prestep=" << maxPreStepTime << "s, mainLoop=" << elapsed
                  << "s, popControl=" << populationControlTime << "s, postStep=" << postStepTime
                  << "s, diagnostics=" << diagnosticsTime << "s, total=" << managerTotalTime
                  << "s, previously accounted=" << accounted << "s" << std::endl;
    }

    MonteCarloTimingDetail2::PrintTimingBreakdown(this->comm_world, this->rank_world, "Main loop detail",
        {
            {"pollRealloc", mainReallocationPollTime},
            {"handleAll", mainHandleAllTime},
            {"flushSends", mainFlushSendBuffersTime},
            {"flushScan", mainFlushSendScanTime},
            {"flushTransfer", mainFlushSendTransferTime},
            {"amountDecrease", mainAmountDecreaseTime},
            {"amountProgress", mainAmountProgressTime},
            {"verify", mainVerifyTime},
            {"exitBarrier", mainExitBarrierTime},
            {"total", elapsed}
        });

    {
        unsigned long long localIterations = static_cast<unsigned long long>(this->iteration);
        unsigned long long maxIterations = localIterations;
        unsigned long long sumIterations = localIterations;
        unsigned long long localVerifyCount = static_cast<unsigned long long>(mainVerifyCount);
        unsigned long long sumVerifyCount = localVerifyCount;
        unsigned long long localAmountProgressCalls = static_cast<unsigned long long>(mainAmountProgressCalls);
        unsigned long long maxAmountProgressCalls = localAmountProgressCalls;
        unsigned long long sumAmountProgressCalls = localAmountProgressCalls;
        unsigned long long localFlushCalls = static_cast<unsigned long long>(this->sendBufferFlushCalls);
        unsigned long long maxFlushCalls = localFlushCalls;
        unsigned long long sumFlushCalls = localFlushCalls;
        unsigned long long localFlushedParticles = static_cast<unsigned long long>(this->sendBufferFlushedParticles);
        unsigned long long sumFlushedParticles = localFlushedParticles;
        unsigned long long localContiguousPutCalls = 0;
        unsigned long long localScatterPutCalls = 0;
        unsigned long long localContiguousPutParticles = 0;
        unsigned long long localScatterPutParticles = 0;
        unsigned long long localTransferCallsWithContiguousAllocation = 0;
        unsigned long long localTransferCallsWithoutContiguousAllocation = 0;
        for(const RankHandler_t *h : this->rankHandlers)
        {
            if(h == nullptr)
            {
                continue;
            }
            localContiguousPutCalls += static_cast<unsigned long long>(h->contiguousParticlePutsThisStep);
            localScatterPutCalls += static_cast<unsigned long long>(h->scatterParticlePutsThisStep);
            localContiguousPutParticles += static_cast<unsigned long long>(h->contiguousParticlesThisStep);
            localScatterPutParticles += static_cast<unsigned long long>(h->scatterParticlesThisStep);
            localTransferCallsWithContiguousAllocation += static_cast<unsigned long long>(h->transferCallsWithContiguousAllocationThisStep);
            localTransferCallsWithoutContiguousAllocation += static_cast<unsigned long long>(h->transferCallsWithoutContiguousAllocationThisStep);
        }
        unsigned long long sumContiguousPutCalls = localContiguousPutCalls;
        unsigned long long sumScatterPutCalls = localScatterPutCalls;
        unsigned long long sumContiguousPutParticles = localContiguousPutParticles;
        unsigned long long sumScatterPutParticles = localScatterPutParticles;
        unsigned long long sumTransferCallsWithContiguousAllocation = localTransferCallsWithContiguousAllocation;
        unsigned long long sumTransferCallsWithoutContiguousAllocation = localTransferCallsWithoutContiguousAllocation;
        unsigned long long localPeakRanks = static_cast<unsigned long long>(this->sendBufferPeakRanks);
        unsigned long long maxPeakRanks = localPeakRanks;
        unsigned long long localPeakParticles = static_cast<unsigned long long>(this->sendBufferPeakParticles);
        unsigned long long maxPeakParticles = localPeakParticles;

        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxIterations, &maxIterations, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumIterations, &sumIterations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumVerifyCount, &sumVerifyCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxAmountProgressCalls, &maxAmountProgressCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumAmountProgressCalls, &sumAmountProgressCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxFlushCalls, &maxFlushCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumFlushCalls, &sumFlushCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumFlushedParticles, &sumFlushedParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumContiguousPutCalls, &sumContiguousPutCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumScatterPutCalls, &sumScatterPutCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumContiguousPutParticles, &sumContiguousPutParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumScatterPutParticles, &sumScatterPutParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumTransferCallsWithContiguousAllocation, &sumTransferCallsWithContiguousAllocation, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &sumTransferCallsWithoutContiguousAllocation, &sumTransferCallsWithoutContiguousAllocation, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxPeakRanks, &maxPeakRanks, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(this->rank_world == 0 ? MPI_IN_PLACE : &maxPeakParticles, &maxPeakParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);

        if(this->rank_world == 0)
        {
            std::cout << "Main loop counts: iterations max/avg=" << maxIterations << "/" << static_cast<double>(sumIterations) / this->size_world
                      << ", verify total=" << sumVerifyCount
                      << ", amountProgress calls total/localMax=" << sumAmountProgressCalls << "/" << maxAmountProgressCalls
                      << ", sendFlush calls total/localMax=" << sumFlushCalls << "/" << maxFlushCalls
                      << ", particlesFlushed total=" << sumFlushedParticles
                      << ", contiguous/scatter put calls=" << sumContiguousPutCalls << "/" << sumScatterPutCalls
                      << ", contiguous/scatter particles=" << sumContiguousPutParticles << "/" << sumScatterPutParticles
                      << ", transfer calls contiguousAllocation yes/no=" << sumTransferCallsWithContiguousAllocation
                      << "/" << sumTransferCallsWithoutContiguousAllocation
                      << ", sendBuffer pending ranks peak=" << maxPeakRanks
                      << ", pending particles peak=" << maxPeakParticles
                      << std::endl;
        }
    }

    this->PrintTransferDiagnostics(elapsed, mainFlushSendScanTime, mainFlushSendTransferTime);

    MonteCarloTimingDetail2::PrintTimingBreakdown(this->comm_world, this->rank_world, "Init detail",
        {
            {"gridMeta", initGridMetaTime},
            {"prepareHandlers", initPrepareHandlersTime},
            {"clearSendBuffers", initClearSendBuffersTime},
            {"shrinkBuffers", initShrinkBuffersTime},
            {"putSelfParticles", initPutSelfParticlesTime},
            {"updateGridData", initUpdateGridDataTime},
            {"preStep", initPreStepTime},
            {"resetParticleState", initResetParticleStateTime},
            {"addParticles", initAddParticlesTime},
            {"initBarrier", initBarrierTime},
            {"total", localInitTime}
        });

    const PrepareHandlersTiming &prepareTiming = this->lastPrepareHandlersTiming;
    MonteCarloTimingDetail2::PrintTimingBreakdown(this->comm_world, this->rank_world, "PrepareHandlers detail",
        {
            {"oldNeighborSet", prepareTiming.oldNeighborSet},
            {"neighborList", prepareTiming.neighborList},
            {"selfHandler", prepareTiming.selfHandler},
            {"findNewNeighbors", prepareTiming.findNewNeighbors},
            {"newNeighborAllreduce", prepareTiming.newNeighborAllreduce},
            {"createHandlers", prepareTiming.createHandlers},
            {"resetBuffers", prepareTiming.resetBuffers},
            {"total", prepareTiming.total}
        });

    MonteCarloTimingDetail2::PrintTimingBreakdown(this->comm_world, this->rank_world, "Handler constructor detail",
        {
            {"rmaCreate", prepareTiming.handlerCtorRma},
            {"mutexCreate", prepareTiming.handlerCtorMutex},
            {"reset", prepareTiming.handlerCtorReset},
            {"peerInfo", prepareTiming.handlerCtorPeerInfo},
            {"total", prepareTiming.handlerCtorTotal}
        });

    int maxLocalNewNeighbors = 0;
    int sumLocalNewNeighbors = 0;
    int localNewNeighbors = prepareTiming.localNewNeighbors;
    MPI_Reduce(&localNewNeighbors, &maxLocalNewNeighbors, 1, MPI_INT, MPI_MAX, 0, this->comm_world);
    MPI_Reduce(&localNewNeighbors, &sumLocalNewNeighbors, 1, MPI_INT, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "PrepareHandlers counts: newNeighborsTotal=" << prepareTiming.globalNewNeighbors
                  << ", localMax=" << maxLocalNewNeighbors
                  << ", localAvg=" << static_cast<double>(sumLocalNewNeighbors) / this->size_world
                  << std::endl;
    }

    MonteCarloTimingDetail2::PrintTimingBreakdown(this->comm_world, this->rank_world, "Pre/post-main detail",
        {
            {"amountManager", amountManagerTime},
            {"memoryDiagnostics", memoryDiagnosticsTime},
            {"endStepShrinkBuffers", endStepShrinkBuffersTime},
            {"managerTotal", localManagerTotalTime}
        });

    {
        MonteCarloConfig::StepStats adaptStats;
        unsigned long long globalSendFlushCalls = static_cast<unsigned long long>(this->sendBufferFlushCalls);
        unsigned long long globalSendFlushedParticles = static_cast<unsigned long long>(this->sendBufferFlushedParticles);
        unsigned long long globalSendIdleDrainFlushCalls = static_cast<unsigned long long>(this->sendBufferFlushIdleDrainCalls);
        unsigned long long globalSendIdleDrainFlushedParticles = static_cast<unsigned long long>(this->sendBufferIdleDrainParticles);
        unsigned long long globalMaxPendingSendBufferParticles = static_cast<unsigned long long>(this->sendBufferPeakParticles);
        double globalFlushTransferTime = mainFlushSendTransferTime;
        double globalElapsedTime = elapsed;
        MPI_Allreduce(MPI_IN_PLACE, &globalSendFlushCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalSendFlushedParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalSendIdleDrainFlushCalls, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalSendIdleDrainFlushedParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalMaxPendingSendBufferParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalFlushTransferTime, 1, MPI_DOUBLE, MPI_SUM, this->comm_world);
        MPI_Allreduce(MPI_IN_PLACE, &globalElapsedTime, 1, MPI_DOUBLE, MPI_SUM, this->comm_world);
        adaptStats.totalReallocationTime = localReallocationTime;
        adaptStats.totalReallocations = totalReallocations;
        adaptStats.mainLoopTime = elapsed;
        adaptStats.totalIterations = this->iteration;
        adaptStats.totalTransfers = this->transfersCounter;
        adaptStats.totalSendFlushCalls = static_cast<size_t>(globalSendFlushCalls);
        adaptStats.totalSendFlushedParticles = static_cast<size_t>(globalSendFlushedParticles);
        adaptStats.totalSendIdleDrainFlushCalls = static_cast<size_t>(globalSendIdleDrainFlushCalls);
        adaptStats.totalSendIdleDrainFlushedParticles = static_cast<size_t>(globalSendIdleDrainFlushedParticles);
        adaptStats.avgFlushTransferFraction = globalElapsedTime > 0 ? globalFlushTransferTime / globalElapsedTime : 0;
        adaptStats.maxPendingSendBufferParticles = static_cast<size_t>(globalMaxPendingSendBufferParticles);
        size_t maxPeak = 0;
        size_t handlerCount = 0;
        for(RankHandler_t *h : this->rankHandlers)
        {
            if(h == nullptr)
            {
                continue;
            }
            handlerCount++;
            maxPeak = std::max(maxPeak, h->peakBufferUsage);
            h->peakBufferUsage = 0;
        }
        adaptStats.numHandlers = handlerCount;
        adaptStats.peakBufferUsage = maxPeak;
        this->config.Adapt(adaptStats, this->rank_world);

        unsigned long long syncShrinkCycle = this->config.shrinkBuffersCycle;
        MPI_Allreduce(MPI_IN_PLACE, &syncShrinkCycle, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm_world);
        this->config.shrinkBuffersCycle = static_cast<size_t>(syncShrinkCycle);
        unsigned long long syncSendBufferMinSize = static_cast<unsigned long long>(this->config.sendBufferMinSize);
        MPI_Bcast(&syncSendBufferMinSize, 1, MPI_UNSIGNED_LONG_LONG, 0, this->comm_world);
        this->config.sendBufferMinSize = static_cast<size_t>(syncSendBufferMinSize);
        unsigned long long syncSmallIdleFlushHoldoffCycles = static_cast<unsigned long long>(this->config.GetSmallIdleFlushHoldoffCycles());
        MPI_Bcast(&syncSmallIdleFlushHoldoffCycles, 1, MPI_UNSIGNED_LONG_LONG, 0, this->comm_world);
        this->config.SyncSmallIdleFlushHoldoffCycles(static_cast<size_t>(syncSmallIdleFlushHoldoffCycles));
    }
    #endif // TIMING

    return data.remaining;
}

#endif // RDMA_MONTE_CARLO_MANAGER_HPP
