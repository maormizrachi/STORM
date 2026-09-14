#ifndef STORM_MONTE_CARLO_MANAGER_HPP
#define STORM_MONTE_CARLO_MANAGER_HPP

#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <boost/container/flat_set.hpp>

#include "../particle/Particle.hpp"
#include "../physics/MonteCarloPhysics.hpp"
#include "../population/PopulationControl.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "../utils/GhostMap.hpp"
#include "../gpu/ProfileRegion.hpp"
#ifdef STORM_WITH_GPU
#include "../gpu/KokkosLocalTransportExecutor.hpp"
#include "../gpu/DevicePopulationContext.hpp"
#include "../gpu/DeviceSourceContext.hpp"
#endif // STORM_WITH_GPU
#include "communication/CommunicationEngine.hpp"
#include "communication/SerialCommunicationEngine.hpp"

#ifdef MEMORY_DEBUG
#include "misc/memory_debug.hpp"
#else
#ifndef MEMORY_DEBUG_PRINT
#define MEMORY_DEBUG_PRINT(label) ((void)0)
#endif
#endif
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef STORM_WITH_MPI
#include <mpi.h>
#include <mpi_utils/AmountManager.hpp>
#endif
#include "MonteCarloConfig.hpp"
#include "MonteCarloTracker.hpp"
#include "../elementary/PointOps.hpp"

namespace STORM
{

using namespace STORM::fallback;

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
        {
            maxRelDrift = std::max(maxRelDrift, std::abs(delta) / size);
        }
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

template<typename T, typename Grid, typename Physics = MonteCarloPhysics<T, Grid>>
class MonteCarloManager
{
    static_assert(std::is_base_of<MonteCarloPhysics<T, Grid>, Physics>::value, "Physics must derive from MonteCarloPhysics<T, Grid>");

    using MCParticle = MonteCarloParticle<T>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        size_t leavingCount = 0;
    };

    using Tracker = MonteCarloTracker<MCParticle>;

    MonteCarloManager(const Grid &grid, const std::shared_ptr<Physics> &physics,
                          const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                          const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                          const MonteCarloConfig &config = MonteCarloConfig(),
                          std::unique_ptr<CommunicationEngine<T>> engine = std::make_unique<SerialCommunicationEngine<T>>());

    ~MonteCarloManager() = default;

    inline size_t GetStepCounter(void) const
    {
        return this->allStepsCounter;
    }

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const
    {
        return this->cellsStepsCounters;
    }

    inline std::vector<size_t> &GetCellsStepsCounters(void)
    {
        return this->cellsStepsCounters;
    }

    inline const std::vector<size_t> &GetCellsParticleCounters(void) const
    {
        return this->cellsParticleCounters;
    }

    inline size_t GetStartParticleCount(void) const
    {
        return this->startParticleCount;
    }

    inline size_t GetEndParticleCount(void) const
    {
        return this->endParticleCount;
    }

    inline size_t GetInitialParticleCount(void) const
    {
        return this->initialParticleCount;
    }

    inline size_t GetPreStepParticleCount(void) const
    {
        return this->preStepParticleCount;
    }

    inline double GetPureComputeTime(void) const
    {
        return 0;
    }

    inline const std::vector<size_t> &GetBeginningParticleCount(void) const
    {
        return this->beginningParticleCount;
    }

    inline std::vector<size_t> &GetBeginningParticleCount(void)
    {
        return this->beginningParticleCount;
    }

    inline size_t GetHandlerMemoryBytes(void) const
    {
        return this->handlerMemoryBytes;
    }

    /// Advances the owned particle census. References returned by
    /// getParticles() are invalidated by this call.
    void step(dt_t fullDt);

    std::size_t GetParticleCount() const
    {
#ifdef STORM_WITH_GPU
        if(not this->hostParticlesValid and this->deviceCensusValid and this->gpuTransportExecutor)
            return this->gpuTransportExecutor->PendingCensusCount();
#endif
        return this->ownedParticles.size();
    }

    const std::vector<MCParticle> &getParticles(void) const
    {
        this->MaterializeDeviceCensus();
        return this->ownedParticles;
    }

    std::vector<MCParticle> &getParticles(void)
    {
        this->MaterializeDeviceCensus();
        this->particlesChanged = true;
        return this->ownedParticles;
    }

    inline const Tracker &getTracker(void)
    {
        return this->tracker;
    }

    inline void resetTracker(void)
    {
        this->tracker.Reset();
    }

private:
    using Reduction = typename CommunicationEngine<T>::Reduction;
    using CompletionCounter = long long;

    const Grid &grid;
    MonteCarloConfig config;
#ifdef STORM_WITH_MPI
    MPI_Comm commWorld;
#endif
    rank_t rankWorld, sizeWorld;
    size_t nCells;
    // std::shared_ptr<ProgressCounter> progress;
    CompletionCounter localDecrementAmount;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranksGhostMap;
    T ll, ur;
    std::shared_ptr<Physics> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    mutable size_t myIDCounter;
    size_t currentStep;
    size_t allStepsCounter;
    size_t transfersCounter;
    std::chrono::high_resolution_clock::time_point progressStartTime;
    double lastProgressPrintTime = 0.0;
    int64_t progressStartParticles = 0;
    size_t progressRemovedCount = 0;

public:
    const void *progressCellsPtr = nullptr;
    const void *progressOpacityPtr = nullptr;

private:
    std::vector<rank_t> neighbors;
    std::vector<size_t> cellsStepsCounters;
    std::vector<size_t> cellsParticleCounters;
    size_t iteration;
    size_t dynamicallyAdded;
    size_t lastBuildGeneration;
    size_t startParticleCount = 0;
    size_t endParticleCount = 0;
    size_t initialParticleCount = 0;
    size_t preStepParticleCount = 0;
    std::vector<size_t> beginningParticleCount;
    size_t handlerMemoryBytes = 0;
    mutable std::vector<MCParticle> ownedParticles;
    mutable bool hostParticlesValid = true;
    bool particlesChanged = false;
    bool deviceCensusValid = false;
    std::uint64_t populationActivationEpoch = 0;

    std::unique_ptr<CommunicationEngine<T>> engine;
#ifdef STORM_WITH_MPI
    std::unique_ptr<AmountManager> amountManager;
#endif
    CompletionCounter completionRemaining = 0;
    bool completionDone = false;
    std::vector<std::vector<MCParticle>> detachedRankParticles;
    std::vector<rank_t> activeRanks, nextActiveRanks;
    std::vector<unsigned char> activeRankListed;
    size_t hostTransportVisit = 0;
    size_t activeRankScanCursor = 0, activeRankScanRemaining = 0;
#ifdef STORM_WITH_GPU
    std::unique_ptr<gpu::KokkosLocalTransportExecutor> gpuTransportExecutor;
    double gpuPackSeconds = 0.0;
    double gpuDeviceSeconds = 0.0;
    double gpuCopyBackSeconds = 0.0;
    double gpuProgressSeconds = 0.0;
    double gpuHostEventSeconds = 0.0;
    unsigned long long gpuLaunchCount = 0;
    unsigned long long gpuParticleCount = 0;
    unsigned long long gpuIngestCount = 0;
    unsigned long long gpuPhysicsStepCount = 0;
    unsigned long long gpuHoldCount = 0;
    unsigned long long gpuElidedRemovalCount = 0;
    mutable std::size_t gpuDeferredD2HBytes = 0;
    size_t gpuHoldSkips = 0;
    std::size_t gpuLastStepMaxActive = 0;
#endif // STORM_WITH_GPU

    // Reused across transport rounds so the merge does not reallocate and
    // page-fault the whole local population on every iteration.
    std::vector<MCParticle> mergedParticleBuffer;
    std::vector<MCParticle> mergeScratchBuffer;
    std::vector<MCParticle> particlesToAdd;
    size_t progressStepCounter = 0;

    // Splits the transport loop time that the GPU phase timers do not cover.
    double loopCommunicationSeconds = 0.0;
    double loopAmountSeconds = 0.0;
    double loopHandleSeconds = 0.0;
    double loopMergeSeconds = 0.0;
    unsigned long long loopRounds = 0;
    unsigned long long loopIdleRounds = 0;

    // Outcome of applying one physics event to a particle: either the particle
    // stays local and keeps stepping, or it left this rank's transport loop
    // (censused, removed, or handed to a send buffer).
    enum class TransportEventAction
    {
        Continue,
        Finished
    };

    // Pre-step state that only the STORM_DEBUG diagnostics read. Empty in
    // release so the shared event handler keeps the same signature at no cost.
    struct TransportStepContext
    {
#ifdef STORM_DEBUG
        T beforeStepLocation;
        T beforeStepVelocity;
        dt_t beforeStepTimeLeft;
        T previousLocation;
#endif // STORM_DEBUG
    };

    TransportEventAction ApplyTransportEvent(MCParticle &particle,
                                             const MonteCarloFunctionality &functionality,
                                             rank_t bufferRank, size_t particleIndex,
                                             MonteCarloStepFinalData &stepData,
                                             const TransportStepContext &context);

    bool HandleAll(MonteCarloStepFinalData &stepData, bool fullNeighborSweep = false);

    bool HaveParticlesChanged(void) const
    {
        return this->particlesChanged;
    }

    void ClearParticlesChanged(void)
    {
        this->particlesChanged = false;
    }

    void MaterializeDeviceCensus(void) const
    {
#ifdef STORM_WITH_GPU
        if(this->hostParticlesValid or not this->deviceCensusValid or not this->gpuTransportExecutor)
        {
            return;
        }
        const std::size_t assigned = this->gpuTransportExecutor->AssignPendingCensusIdentities(this->rankWorld, static_cast<particle_id_t>(this->myIDCounter));
        this->myIDCounter += assigned;
        const std::size_t d2hBefore = this->gpuTransportExecutor->Metrics().d2hBytes;
        gpu::CompletedBatch census = this->gpuTransportExecutor->SnapshotPendingCensus();
        this->gpuDeferredD2HBytes += this->gpuTransportExecutor->Metrics().d2hBytes - d2hBefore;
        this->ownedParticles.clear();
        this->ownedParticles.reserve(census.census.size);
        for(const gpu::CompletedTransport &transported : census.census)
        {
            MCParticle particle;
            gpu::UnpackParticle(transported.particle, transported.cold, particle);
            this->ownedParticles.push_back(std::move(particle));
        }
        this->hostParticlesValid = true;
#endif
    }

#ifdef STORM_WITH_GPU
    bool TransportBatchOnDevice(std::vector<MCParticle> &localParticles, rank_t bufferRank, MonteCarloStepFinalData &stepData, bool &isEmpty);

    void IngestHostParticlesForDevice(std::vector<MCParticle> &arrivals);

    void CollectHostParticlesForDevice(std::vector<MCParticle> &arrivals);

    bool TransportResidentOnDevice(std::vector<MCParticle> &arrivals, MonteCarloStepFinalData &stepData, bool &isEmpty);

    void ApplyDeviceCompletions(gpu::CompletedBatch &completed,
                                MonteCarloStepFinalData &stepData,
                                bool transportInFlight = false);

    void DrainDeviceCensus(MonteCarloStepFinalData &stepData);
#endif // STORM_WITH_GPU

    void PutSelfParticles(std::vector<MCParticle> &&particles);

    void StageLocalParticlesForDevice(std::vector<MCParticle> &&particles, bool assignNewIDs);

    void AddParticles(const std::vector<MCParticle> &particles);

    void PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum);
};

#include "MonteCarloTransport.hpp"
#include "MonteCarloLifecycle.hpp"

} // namespace STORM

#endif // STORM_MONTE_CARLO_MANAGER_HPP
