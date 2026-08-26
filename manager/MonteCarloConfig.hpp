#ifndef STORM_MONTE_CARLO_CONFIG_HPP
#define STORM_MONTE_CARLO_CONFIG_HPP

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace STORM {

constexpr double MONTECARLO_EPSILON = 1e-8;

enum class MonteCarloTransferDiagnosticsLevel
{
    Off,
    StepSummary,
    StepSummaryAndHistograms
};

struct MonteCarloConfig
{
private:
    static constexpr size_t shrinkBuffersCycleMin = 50;
    static constexpr size_t defaultSendBufferMinSize = 1024;
    static constexpr size_t sendBufferMinSizeMin = 1024;
    static constexpr size_t sendBufferMinSizeMax = 16384;
    static constexpr size_t sendBufferTargetParticlesPerFlush = 2048;
    static constexpr double sendBufferHighTransferFraction = 0.20;
    static constexpr double sendBufferLowTransferFraction = 0.08;
    static constexpr size_t smallIdleFlushHoldoffCyclesMin = 512;
    static constexpr size_t smallIdleFlushHoldoffCyclesMax = 16384;
    static constexpr double smallIdleFlushHighCallFraction = 0.80;
    static constexpr double smallIdleFlushLowCallFraction = 0.50;
    static constexpr double smallIdleFlushLowParticleFraction = 0.20;
    static constexpr size_t smallIdleFlushPendingSoftLimitFactor = 512;

    size_t smallIdleFlushHoldoffCycles = smallIdleFlushHoldoffCyclesMin;

public:
    size_t initialBufferSize          = 5000;
    size_t shrinkBuffersCycle         = shrinkBuffersCycleMin;
    size_t sendBufferMinSize          = defaultSendBufferMinSize;
    size_t amountProgressMinCycles    = 16;
    size_t asyncReallocationProgressMinCycles = 16;
    size_t asyncReallocationSendPollMinCycles = 8;
    size_t asyncReallocationIncomingPollActiveCycles = 8;
    size_t asyncReallocationIncomingPollIdleCycles = 8;
    size_t asyncReallocationMaxIncomingRequestsPerPoll = 4;
    size_t activeRankScanChunk        = 64;
    // Bound one local compute slice so the host returns regularly to MPI/RMA
    // progress.  This is also the initial host-to-device packet batch size.
    size_t localTransportBatchSize    = 4096;
#ifdef STORM_WITH_GPU
    // Events per particle per device wave. A wavefront retires only when its
    // slowest lane stops, so a long cap makes thin lanes idle behind thick
    // ones; 64 bounds that tail and beat 4096 by ~1.7x on CrookedPipe.
    size_t gpuMaxInnerSteps           = 64;
    // Unused by the resident pool (the whole active set is launched). Kept so
    // callers can still cap a wave later without an ABI break.
    size_t gpuLaunchSize              = 0;
    // Do not launch a wave while the device pool is below this size if host
    // arrivals are still arriving. 0 disables the hold.
    size_t gpuMinLaunchSize           = 0;
    // Force a wave after this many consecutive holds, even if the pool is
    // still small. 0 means hold until arrivals stop (empty Collect).
    size_t gpuHoldMaxSkips            = 0;
#endif
    size_t transferDiagnosticsEveryNSteps = 1;
    double bufferReallocationFactor   = 1.5;
    size_t minimalBuffSize            = 50;
    double bufferShrinkFactor         = 0.5;
    double bufferShrinkNeighborFactor = 0.5;
    double shrinkPercent              = 0.25;
    bool retireStaleHandlers          = true;
    bool holdSmallIdleFlushes = true;
    size_t sendBufferMinIdleDrainSize = 512;
    size_t sendBufferIdleDrainPatienceCycles = 16384;
    MonteCarloTransferDiagnosticsLevel transferDiagnosticsLevel = MonteCarloTransferDiagnosticsLevel::StepSummary;

    size_t GetSmallIdleFlushHoldoffCycles(void) const
    {
        return this->smallIdleFlushHoldoffCycles;
    }

    void SyncSmallIdleFlushHoldoffCycles(size_t value)
    {
        this->smallIdleFlushHoldoffCycles = std::min<size_t>(smallIdleFlushHoldoffCyclesMax, std::max<size_t>(smallIdleFlushHoldoffCyclesMin, value));
    }

    static MonteCarloConfig Auto(size_t particlesPerRank, size_t numNeighbors)
    {
        MonteCarloConfig cfg;
        numNeighbors = std::max<size_t>(numNeighbors, 1);
        cfg.initialBufferSize = std::max<size_t>(100, particlesPerRank / numNeighbors);
        cfg.minimalBuffSize   = std::max<size_t>(10, cfg.initialBufferSize / 10);
        cfg.sendBufferMinSize = std::max<size_t>(sendBufferMinSizeMin, particlesPerRank / (numNeighbors * 4));
        return cfg;
    }

};

} // namespace STORM

#endif // STORM_MONTE_CARLO_CONFIG_HPP
