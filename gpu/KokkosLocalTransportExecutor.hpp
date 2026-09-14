#ifndef STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "DeviceParticle.hpp"
#include <Kokkos_ScatterView.hpp>
#ifdef KOKKOS_ENABLE_OPENMP
#include <omp.h>
#endif
#include "GreyIMCKernel.hpp"
#include "ProfileRegion.hpp"
#include "ExecutionEvent.hpp"

namespace STORM
{
namespace gpu
{

template<typename PhysicsT, typename = void>
struct HasDeviceTransport : std::false_type
{};

template<typename PhysicsT>
struct HasDeviceTransport<PhysicsT, std::void_t<
    decltype(std::declval<const PhysicsT &>().UsesDeviceTransport()),
    decltype(std::declval<const PhysicsT &>().GetDeviceTransportViews())>>
    : std::true_type
{};

template<typename PhysicsT, typename = void>
struct HasDeviceCensusPostStep : std::false_type
{};

template<typename PhysicsT>
struct HasDeviceCensusPostStep<PhysicsT, std::void_t<
    decltype(std::declval<const PhysicsT &>().
                 SupportsDeviceCensusPostStep())>>
    : std::true_type
{};

struct CompletedTransport
{
    DeviceParticle particle;
    DeviceParticleCold cold;
    TransportResult result;
};
static_assert(std::is_trivially_copyable<CompletedTransport>::value,
              "CompletedTransport must be safe for device-host copies");

struct CompletedTransportSpan
{
    const CompletedTransport *data = nullptr;
    std::size_t size = 0;

    const CompletedTransport *begin() const { return data; }
    const CompletedTransport *end() const
    {
        return data == nullptr ? nullptr : data + size;
    }
    bool empty() const { return size == 0; }
};

struct CompletedBatch
{
    // Production callers consume these borrowed spans before invoking the
    // executor again. Execute() uses particles only because it accumulates
    // results across multiple waves.
    CompletedTransportSpan terminals;
    CompletedTransportSpan fallbacks;
    CompletedTransportSpan remotes;
    CompletedTransportSpan census;
    std::vector<CompletedTransport> particles;
    double packSeconds = 0.0;
    double deviceSeconds = 0.0;
    double copyBackSeconds = 0.0;
    double progressSeconds = 0.0;
    std::size_t launchCount = 0;
    std::size_t physicsSteps = 0;
    std::size_t launchedParticles = 0;
    std::size_t censusCount = 0;
    std::size_t censusSteps = 0;
    std::size_t createdParticles = 0;
};

struct WaveCounters
{
    std::size_t terminal = 0;
    std::size_t fallback = 0;
    std::size_t remote = 0;
    std::size_t survivor = 0;
    std::size_t census = 0;
    std::size_t physicsSteps = 0;
    std::size_t censusSteps = 0;
    std::size_t appended = 0;
    std::size_t overflow = 0;
};
static_assert(std::is_trivially_copyable<WaveCounters>::value,
              "WaveCounters must be safe for device-host copies");

struct TransportExecutorMetrics
{
    std::size_t h2dBytes = 0;
    std::size_t d2hBytes = 0;
    std::size_t eliminatedHostCopyBytes = 0;
    std::size_t reallocationCount = 0;
    std::size_t synchronizationCount = 0;
    std::size_t terminalCount = 0;
    std::size_t fallbackCount = 0;
    std::size_t remoteCount = 0;
    std::size_t censusCopyCount = 0;
    std::size_t splitCreatedCount = 0;
    std::size_t maxIngestCount = 0;
    std::size_t maxActiveCount = 0;
    std::size_t progressPollCount = 0;
    std::size_t pipelinedRemoteCount = 0;
};

class KokkosLocalTransportExecutor
{
#if defined(KOKKOS_ENABLE_HIP)
    using PinnedHostSpace = Kokkos::HIPHostPinnedSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
    using PinnedHostSpace = Kokkos::CudaHostPinnedSpace;
#else
    using PinnedHostSpace = Kokkos::HostSpace;
#endif

    static constexpr bool hostAccessible = Kokkos::SpaceAccessibility<
        Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;

public:
    explicit KokkosLocalTransportExecutor(std::size_t maximumInnerSteps,
                                          bool overlapCommunication = false)
        : maximumInnerSteps_(std::max<std::size_t>(1, maximumInnerSteps)),
          overlapCommunication_(overlapCommunication),
          hostPackets_("storm_transport_host_packets", 0),
          hostColdPackets_("storm_transport_host_cold_packets", 0),
          packets_("storm_transport_packets", 0),
          nextPackets_("storm_transport_next_packets", 0),
          coldPackets_("storm_transport_cold_packets", 0),
          nextColdPackets_("storm_transport_next_cold_packets", 0),
          survivorFlags_("storm_transport_survivor_flags", 0),
          survivorSplitCounts_("storm_transport_survivor_split_counts", 0),
          compactedSurvivorSplitCounts_(
              "storm_transport_compacted_survivor_split_counts", 0),
          completedTransports_("storm_transport_completed", 0),
          fallbackTransports_("storm_transport_fallbacks", 0),
          pendingRemotes_("storm_transport_pending_remotes", 0),
          censusPackets_("storm_transport_census_packets", 0),
          censusCold_("storm_transport_census_cold", 0),
          censusCellCounts_("storm_transport_census_cell_counts", 0),
          hostEventTransports_("storm_transport_host_events", 0),
          hostRemoteTransports_("storm_transport_host_remotes", 0),
          waveCounters_("storm_transport_wave_counters"),
          hostWaveCounters_("storm_transport_host_wave_counters"),
          remoteCopySpace_(Kokkos::Experimental::partition_space(
              Kokkos::DefaultExecutionSpace{}, 1).front())
    {}

    ~KokkosLocalTransportExecutor()
    {
        // Includes exception unwinding: NIC-facing host spans and device
        // source buffers must outlive any outstanding D2H operation.
        remoteCopySpace_.fence("STORM remote copy teardown");
        Kokkos::DefaultExecutionSpace{}.fence("STORM transport teardown");
    }

    void Reset()
    {
        if(this->HasPendingRemoteCopy())
            throw std::runtime_error("Cannot reset transport with undelivered remotes");
        this->ResetStepMetrics();
        this->activeCount_ = 0;
        this->pendingRemoteCount_ = 0;
        this->pendingCensusCount_ = 0;
    }

    void ResetStepMetrics()
    {
        if(this->privateMaterialTarget_ || this->privateRadiationTarget_)
            throw std::runtime_error("Flush private energy tallies before resetting transport");
        this->remoteHoldSkips_ = 0;
        this->metrics_ = {};
        if constexpr(hostAccessible)
        {
            Kokkos::deep_copy(this->cellSteps_, std::size_t(0));
            if(this->cellSteps_.extent(0) != 0)
                this->cellStepScatter_.reset();
        }
    }

    // Restore the work weights used by the host mesh load balancer.
    void AddCellSteps(std::vector<std::size_t> &counts)
    {
        if constexpr(hostAccessible)
        {
            if(this->cellSteps_.extent(0) != 0)
                Kokkos::Experimental::contribute(this->cellSteps_, this->cellStepScatter_);
            Kokkos::DefaultExecutionSpace{}.fence("STORM cell work counts");
            for(std::size_t i = 0; i < std::min(counts.size(), this->cellSteps_.extent(0)); ++i)
                counts[i] += this->cellSteps_(i);
        }
    }

    // Opt-in because direct executor users may consume tallies after each wave.
    void EnablePrivateEnergyTallies()
    {
#ifdef KOKKOS_ENABLE_OPENMP
        if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>)
            this->privateEnergyTallies_ = true;
#endif
    }

    void FlushEnergyTallies()
    {
        if(!this->privateMaterialTarget_ && !this->privateRadiationTarget_)
            return;
        auto tallies = this->threadEnergyTallies_;
        double *material = this->privateMaterialTarget_;
        double *radiation = this->privateRadiationTarget_;
        Kokkos::parallel_for("storm_merge_private_energy", tallies.extent(2),
            KOKKOS_LAMBDA(std::size_t cell)
            {
                double m = 0.0, r = 0.0;
                for(std::size_t thread = 0; thread < tallies.extent(0); ++thread)
                {
                    m += tallies(thread, 0, cell);
                    r += tallies(thread, 1, cell);
                    tallies(thread, 0, cell) = 0.0;
                    tallies(thread, 1, cell) = 0.0;
                }
                if(material) material[cell] += m;
                if(radiation) radiation[cell] += r;
            });
        Kokkos::DefaultExecutionSpace{}.fence("STORM merge private energy");
        this->privateMaterialTarget_ = nullptr;
        this->privateRadiationTarget_ = nullptr;
    }

    std::size_t ActiveCount() const
    {
        return this->activeCount_;
    }

    std::size_t PendingRemoteCount() const
    {
        return this->pendingRemoteCount_;
    }

    std::size_t PendingCensusCount() const
    {
        return this->pendingCensusCount_;
    }

    Kokkos::View<DeviceParticle *> PendingCensusPackets()
    {
        return this->censusPackets_;
    }

    Kokkos::View<DeviceParticleCold *> PendingCensusCold()
    {
        return this->censusCold_;
    }

    void ClearPendingCensus()
    {
        this->pendingCensusCount_ = 0;
    }

    void ReplacePendingCensus(
        const Kokkos::View<DeviceParticle *> &packets,
        const Kokkos::View<DeviceParticleCold *> &cold,
        const std::size_t count)
    {
        this->censusPackets_ = packets;
        this->censusCold_ = cold;
        this->pendingCensusCount_ = count;
        this->metrics_.maxActiveCount =
            std::max(this->metrics_.maxActiveCount, count);
    }

    bool DeviceBusy() const
    {
        return this->activeCount_ > 0 or this->pendingRemoteCount_ > 0 or this->HasPendingRemoteCopy();
    }

    bool HasPendingRemoteCopy() const { return this->inFlightRemoteCount_ != 0; }

    const TransportExecutorMetrics &Metrics() const
    {
        return this->metrics_;
    }

    // Size device and pinned-host staging once per step so mid-transport
    // ingests do not trigger a cascade of Kokkos::resize calls.
    void ReservePoolCapacity(const std::size_t activeCapacity,
                             const std::size_t hostIngestCapacity)
    {
        if(activeCapacity == 0 && hostIngestCapacity == 0)
        {
            return;
        }
        if(!Kokkos::is_initialized())
        {
            return;
        }

        const std::size_t activeTarget =
            std::max(activeCapacity, this->poolReservedActiveCapacity_);
        const std::size_t hostTarget =
            std::max(hostIngestCapacity, this->poolReservedHostIngest_);
        if(activeTarget == this->poolReservedActiveCapacity_ &&
           hostTarget == this->poolReservedHostIngest_)
        {
            return;
        }

        if constexpr(!hostAccessible)
        {
            this->EnsureCapacity(this->hostPackets_, hostTarget);
            this->EnsureCapacity(this->hostColdPackets_, hostTarget);
        }
        this->EnsureCapacity(this->packets_, activeTarget);
        this->EnsureCapacity(this->coldPackets_, activeTarget);
        this->poolReservedActiveCapacity_ = activeTarget;
        this->poolReservedHostIngest_ = hostTarget;
    }

    void PromotePendingCensus(const dt_t fullDt)
    {
        if(this->DeviceBusy())
        {
            throw std::logic_error(
                "Cannot promote census while device transport is active");
        }
        const std::size_t count = this->pendingCensusCount_;
        if(count == 0)
        {
            return;
        }
        this->EnsureCapacity(this->packets_, count);
        this->EnsureCapacity(this->coldPackets_, count);
        auto censusPackets = this->censusPackets_;
        auto censusCold = this->censusCold_;
        auto packets = this->packets_;
        auto coldPackets = this->coldPackets_;
        Kokkos::parallel_for(
            "storm_promote_device_census",
            Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                DeviceParticle particle = censusPackets(i);
                particle.timeLeft = fullDt;
                particle.initialWeight =
                    particle.weight < 0.0
                        ? -particle.weight
                        : particle.weight;
                particle.steps = 0;
                packets(i) = particle;
                coldPackets(i) = censusCold(i);
            });
        this->activeCount_ = count;
        this->pendingCensusCount_ = 0;
        this->metrics_.maxActiveCount =
            std::max(this->metrics_.maxActiveCount, count);
    }

    CompletedBatch SnapshotPendingCensus()
    {
        CompletedBatch completed;
        this->CopyCensusToHost(completed, false);
        return completed;
    }

    std::size_t AssignPendingCensusIdentities(
        const rank_t rank,
        const particle_id_t firstID)
    {
        const std::size_t count = this->pendingCensusCount_;
        if(count == 0)
        {
            return 0;
        }
        auto censusCold = this->censusCold_;
        std::size_t missingCount = 0;
        Kokkos::parallel_reduce(
            "storm_count_missing_census_ids",
            Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(
                const std::size_t i,
                std::size_t &missing)
            {
                if(censusCold(i).id ==
                   std::numeric_limits<particle_id_t>::max())
                {
                    ++missing;
                }
            },
            missingCount);
        Kokkos::parallel_scan(
            "storm_assign_device_census_ids",
            Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(
                const std::size_t i,
                std::size_t &offset,
                const bool final)
            {
                if(censusCold(i).id ==
                   std::numeric_limits<particle_id_t>::max())
                {
                    if(final)
                    {
                        censusCold(i).id =
                            firstID + offset;
                        censusCold(i).rank = rank;
                    }
                    ++offset;
                }
            });
        return missingCount;
    }

    void CopyPendingCensusCellCounts(
        const std::size_t cellCount,
        std::vector<std::size_t> &counts)
    {
        counts.assign(cellCount, 0);
        if(cellCount == 0 || this->pendingCensusCount_ == 0)
        {
            return;
        }
        this->EnsureCapacity(this->censusCellCounts_, cellCount);
        Kokkos::deep_copy(this->censusCellCounts_, std::size_t(0));
        auto censusPackets = this->censusPackets_;
        auto deviceCounts = this->censusCellCounts_;
        const std::size_t censusCount = this->pendingCensusCount_;
        Kokkos::parallel_for(
            "storm_count_device_census_cells",
            Kokkos::RangePolicy<>(0, censusCount),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                const std::size_t cellIndex =
                    static_cast<std::size_t>(
                        censusPackets(i).cellIndex);
                if(cellIndex < cellCount)
                {
                    Kokkos::atomic_increment(
                        &deviceCounts(cellIndex));
                }
            });
        auto hostCounts = Kokkos::create_mirror_view(
            this->censusCellCounts_);
        Kokkos::deep_copy(hostCounts, this->censusCellCounts_);
        for(std::size_t i = 0; i < cellCount; ++i)
        {
            counts[i] = hostCounts(i);
        }
    }

    template<typename PointT>
    void Ingest(const std::vector<Particle<PointT>> &arrivals)
    {
        STORM_PROFILE_REGION("storm/pack");
        if(arrivals.empty())
        {
            return;
        }
        if(!Kokkos::is_initialized())
        {
            throw std::runtime_error("Kokkos must be initialized before GPU transport");
        }

        const std::size_t incoming = arrivals.size();
        const std::size_t offset = this->activeCount_;
        const std::size_t required = offset + incoming;
        this->ReserveForIngest(incoming);

        const Kokkos::DefaultExecutionSpace execution;
        if constexpr(hostAccessible)
        {
            // Capacity changes and previous kernels must finish before host writes.
            execution.fence("STORM direct ingest");
            for(std::size_t i = 0; i < incoming; ++i)
                PackParticle(arrivals[i], this->packets_(offset + i), this->coldPackets_(offset + i));
            ++this->metrics_.synchronizationCount;
        }
        else
        {
            for(std::size_t i = 0; i < incoming; ++i)
                PackParticle(arrivals[i], this->hostPackets_(i), this->hostColdPackets_(i));
            Kokkos::deep_copy(execution, Kokkos::subview(this->packets_, std::pair<std::size_t, std::size_t>(offset, required)),
                                Kokkos::subview(this->hostPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
            Kokkos::deep_copy(execution, Kokkos::subview(this->coldPackets_, std::pair<std::size_t, std::size_t>(offset, required)),
                                Kokkos::subview(this->hostColdPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
            execution.fence("STORM ingest staging reuse");
            this->metrics_.h2dBytes += incoming *
                (sizeof(DeviceParticle) + sizeof(DeviceParticleCold));
            this->metrics_.synchronizationCount += 2;
        }
        this->activeCount_ = required;
        this->metrics_.maxIngestCount =
            std::max(this->metrics_.maxIngestCount, incoming);
        this->metrics_.maxActiveCount =
            std::max(this->metrics_.maxActiveCount, required);
    }

    // Reserve device slots for source packets written directly by a kernel.
    std::size_t AllocateActiveSlots(const std::size_t incoming)
    {
        if(incoming == 0)
        {
            return this->activeCount_;
        }
        if(!Kokkos::is_initialized())
        {
            throw std::runtime_error(
                "Kokkos must be initialized before GPU source emission");
        }
        const std::size_t offset = this->activeCount_;
        this->ReserveForIngest(incoming);
        this->activeCount_ = offset + incoming;
        this->metrics_.maxActiveCount =
            std::max(this->metrics_.maxActiveCount, this->activeCount_);
        return offset;
    }

    Kokkos::View<DeviceParticle*> ActivePackets()
    {
        return this->packets_;
    }

    Kokkos::View<DeviceParticleCold*> ActiveColdPackets()
    {
        return this->coldPackets_;
    }

    template<typename ProgressFunction>
    CompletedBatch AdvanceWave(const GreyIMCViews<DeviceVec3> &views, ProgressFunction progress)
    {
        return this->AdvanceWave(views, progress, std::size_t(1), std::size_t(0));
    }

    CompletedBatch FlushPendingRemotes(const std::size_t minRemoteCopy,
                                       const std::size_t maxRemoteHolds,
                                       const bool force)
    {
        if(this->HasPendingRemoteCopy())
            throw std::logic_error("Drain pipelined remotes with their AdvanceWave completion consumer");
        CompletedBatch completed;
        this->FlushRemotesIfNeeded(completed, minRemoteCopy, maxRemoteHolds, force);
        return completed;
    }

    CompletedBatch FlushPendingCensus()
    {
        CompletedBatch completed;
        this->CopyCensusToHost(completed);
        return completed;
    }

    template<typename ProgressFunction>
    CompletedBatch AdvanceWave(const GreyIMCViews<DeviceVec3> &views,
                               ProgressFunction progress,
                               const std::size_t minRemoteCopy,
                               const std::size_t maxRemoteHolds)
    {
        // Compatibility callers consume all results on return.
        if(this->HasPendingRemoteCopy())
            throw std::logic_error("Cannot discard the completion consumer of a pipelined wave");
        return this->AdvanceWave(views, progress, minRemoteCopy, maxRemoteHolds,
                                 [](CompletedBatch &) {}, false);
    }

    template<typename ProgressFunction, typename CompletionFunction>
    CompletedBatch AdvanceWave(const GreyIMCViews<DeviceVec3> &views,
                               ProgressFunction progress,
                               const std::size_t minRemoteCopy,
                               const std::size_t maxRemoteHolds,
                               CompletionFunction consumeRemotes,
                               const bool pipelineRemotes = true)
    {
        STORM_PROFILE_REGION("storm/transport/wave");
        CompletedBatch completed;
        if(this->activeCount_ == 0)
        {
            this->ConsumeRemoteCopy(completed, progress, consumeRemotes);
            this->FlushRemotesIfNeeded(completed, minRemoteCopy, maxRemoteHolds, true);
            return completed;
        }
        if(!Kokkos::is_initialized())
        {
            throw std::runtime_error("Kokkos must be initialized before GPU transport");
        }

        const std::size_t activeCount = this->activeCount_;
        // Cap the launch so event buffers stay a few GiB. Unlaunched
        // particles stay in packets_[launchCount, activeCount) and are
        // compacted behind this wave's survivors.
        constexpr std::size_t kMaxWaveParticles = 4u * 1024u * 1024u;
        const std::size_t launchCount =
            std::min(activeCount, kMaxWaveParticles);
        const std::size_t remoteOffset = this->pendingRemoteCount_;
        const std::size_t censusOffset = this->pendingCensusCount_;
        this->ReserveForWave(launchCount, remoteOffset, censusOffset, true);

        const std::chrono::steady_clock::time_point deviceStart = std::chrono::steady_clock::now();
        const Kokkos::DefaultExecutionSpace execution;
        Kokkos::deep_copy(execution, this->waveCounters_, WaveCounters{});
        {
            STORM_PROFILE_REGION("storm/transport/kernel");
            this->LaunchGreyIMCTransport(views, launchCount, remoteOffset);
            this->CompactSurvivors(launchCount);
        }

        Kokkos::deep_copy(execution, this->hostWaveCounters_, this->waveCounters_);
        this->kernelReady_.Record(execution);
        // The previous wave's remote buffer is independent of this kernel's
        // output buffer. Apply/send those packets while this wave computes.
        this->ConsumeRemoteCopy(completed, progress, consumeRemotes);
        if(this->overlapCommunication_)
        {
            this->WaitWithProgress(this->kernelReady_, progress, completed);
        }
        else
        {
            execution.fence("STORM transport counters");
        }
        const WaveCounters counters = this->hostWaveCounters_();
        ++this->metrics_.synchronizationCount;
        if(counters.overflow != 0)
        {
            throw std::runtime_error("GPU transport event buffers overflowed");
        }
        if(counters.appended > std::numeric_limits<std::size_t>::max() - counters.survivor)
        {
            throw std::overflow_error("KokkosLocalTransportExecutor split output overflow");
        }
        const std::size_t totalSurvivors = counters.survivor + counters.appended;
        {
            STORM_PROFILE_REGION("storm/transport/harvest_census");
            this->HarvestWaveCensus(censusOffset, counters.census, launchCount);
        }
        if(counters.appended > 0)
        {
            this->ExpandInterfaceSplits(totalSurvivors, counters.survivor);
        }
        completed.deviceSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - deviceStart).count();
        completed.launchCount = 1 + (counters.appended > 0 ? 1 : 0);
        completed.physicsSteps = counters.physicsSteps;
        completed.launchedParticles = launchCount;
        completed.censusCount = counters.census;
        completed.censusSteps = counters.censusSteps;
        completed.createdParticles = counters.appended;
        this->metrics_.splitCreatedCount += counters.appended;

        this->pendingRemoteCount_ = remoteOffset + counters.remote;
        this->pendingCensusCount_ = censusOffset + counters.census;
        {
            STORM_PROFILE_REGION("storm/transport/compact");
            this->CompactCappedWave(launchCount, totalSurvivors, activeCount);
            this->metrics_.maxActiveCount = std::max(this->metrics_.maxActiveCount, this->activeCount_);
        }

        {
            STORM_PROFILE_REGION("storm/copy_back");
            this->CopyFinishedToHost(completed, counters.terminal, counters.fallback);
            // Queue a device-to-pinned-host copy without waiting. The next
            // wave delivers it; the empty-wave path explicitly drains it.
            if(this->overlapCommunication_ && pipelineRemotes)
                this->QueueRemoteCopy(minRemoteCopy, maxRemoteHolds, this->activeCount_ == 0);
            else
                this->FlushRemotesIfNeeded(completed, minRemoteCopy, maxRemoteHolds, this->activeCount_ == 0);
        }
        return completed;
    }

    template<typename PointT>
    CompletedBatch Execute(const std::vector<Particle<PointT>> &particles, std::size_t maximumParticles, const GreyIMCViews<DeviceVec3> &views)
    {
        return this->Execute(particles, maximumParticles, views, [](){});
    }

    template<typename PointT, typename ProgressFunction>
    CompletedBatch Execute(const std::vector<Particle<PointT>> &particles, std::size_t maximumParticles, const GreyIMCViews<DeviceVec3> &views, ProgressFunction progress)
    {
        this->Reset();
        CompletedBatch completed;
        const std::size_t count = std::min(maximumParticles, particles.size());
        if(count == 0)
        {
            return completed;
        }

        const std::chrono::steady_clock::time_point packStart = std::chrono::steady_clock::now();
        std::vector<Particle<PointT>> arrivals(particles.end() - static_cast<std::ptrdiff_t>(count), particles.end());
        this->Ingest(arrivals);
        completed.packSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();

        while(this->activeCount_ > 0 or this->pendingRemoteCount_ > 0)
        {
            CompletedBatch wave = this->AdvanceWave(views, progress, std::size_t(1), std::size_t(0));
            completed.deviceSeconds += wave.deviceSeconds;
            completed.copyBackSeconds += wave.copyBackSeconds;
            completed.progressSeconds += wave.progressSeconds;
            completed.launchCount += wave.launchCount;
            completed.physicsSteps += wave.physicsSteps;
            completed.launchedParticles += wave.launchedParticles;
            completed.createdParticles += wave.createdParticles;
            AppendSpan(completed.particles, wave.terminals);
            AppendSpan(completed.particles, wave.fallbacks);
            AppendSpan(completed.particles, wave.remotes);
        }
        CompletedBatch census = this->FlushPendingCensus();
        completed.copyBackSeconds += census.copyBackSeconds;
        AppendSpan(completed.particles, census.census);
        completed.censusCount = 0;
        completed.censusSteps = 0;
        return completed;
    }

private:
    template<typename ProgressFunction>
    void WaitWithProgress(const ExecutionEvent &event, ProgressFunction progress,
                          CompletedBatch &completed)
    {
        STORM_PROFILE_REGION("storm/transport/progress");
        auto nextProgress = std::chrono::steady_clock::time_point::min();
        while(!event.Ready())
        {
            const auto start = std::chrono::steady_clock::now();
            // A tight loop can issue millions of empty OFI progress calls
            // per step and contend with useful communication on other ranks.
            // Keep querying device completion, but bound network polling to
            // one pass per 20 us (long callbacks naturally run less often).
            if(start < nextProgress) continue;
            progress();
            completed.progressSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            nextProgress = start + std::chrono::microseconds(20);
            ++this->metrics_.progressPollCount;
        }
    }

    template<typename ProgressFunction, typename CompletionFunction>
    void ConsumeRemoteCopy(CompletedBatch &completed, ProgressFunction progress,
                           CompletionFunction consume)
    {
        if(!this->HasPendingRemoteCopy()) return;
        this->WaitWithProgress(this->remoteReady_, progress, completed);
        CompletedBatch remotes;
        if constexpr(hostAccessible)
            remotes.remotes = {this->inFlightDeviceRemotes_.data(), this->inFlightRemoteCount_};
        else
            remotes.remotes = {this->inFlightHostRemotes_.data(), this->inFlightRemoteCount_};
        consume(remotes);
        // The consumer owns MPI serialization. It must finish reading the
        // borrowed span before returning; only then may these buffers swap.
        this->inFlightRemoteCount_ = 0;
    }

    void QueueRemoteCopy(const std::size_t minRemoteCopy,
                         const std::size_t maxRemoteHolds, const bool force)
    {
        if(this->pendingRemoteCount_ == 0)
        {
            this->remoteHoldSkips_ = 0;
            return;
        }
        if(!force && minRemoteCopy > this->pendingRemoteCount_ &&
           !(maxRemoteHolds > 0 && this->remoteHoldSkips_ >= maxRemoteHolds))
        {
            ++this->remoteHoldSkips_;
            return;
        }
        if(this->HasPendingRemoteCopy())
            throw std::runtime_error("Remote copy buffer is still in use");
        this->remoteHoldSkips_ = 0;
        // Preallocate the alternate pair before starting the copy. During
        // steady state neither pair is resized while a copy owns it.
        this->EnsureCapacity(this->inFlightDeviceRemotes_, this->pendingRemotes_.extent(0));
        if constexpr(!hostAccessible)
        {
            this->EnsureCapacity(this->inFlightHostRemotes_, this->hostRemoteTransports_.extent(0));
        }
        std::swap(this->inFlightDeviceRemotes_, this->pendingRemotes_);
        std::swap(this->inFlightHostRemotes_, this->hostRemoteTransports_);
        this->inFlightRemoteCount_ = this->pendingRemoteCount_;
        this->pendingRemoteCount_ = 0;
        const auto range = std::make_pair(std::size_t(0), this->inFlightRemoteCount_);
        if constexpr(!hostAccessible)
        {
            Kokkos::deep_copy(this->remoteCopySpace_,
                Kokkos::subview(this->inFlightHostRemotes_, range),
                Kokkos::subview(this->inFlightDeviceRemotes_, range));
            const std::size_t bytes = this->inFlightRemoteCount_ * sizeof(CompletedTransport);
            this->metrics_.d2hBytes += bytes;
            this->metrics_.eliminatedHostCopyBytes += bytes;
        }
        // Keep the existing deferred-consumption and buffer ownership protocol.
        this->remoteReady_.Record(this->remoteCopySpace_);
        this->metrics_.remoteCount += this->inFlightRemoteCount_;
        this->metrics_.pipelinedRemoteCount += this->inFlightRemoteCount_;
    }

public:
    // NVCC must be able to name the enclosing function of an extended lambda.
    // Only the kernel launch helpers are public; executor state remains private.
    void ExpandInterfaceSplits(const std::size_t totalSurvivors,
                               const std::size_t primarySurvivors)
    {
        STORM_PROFILE_REGION("storm/transport/splits");
        this->EnsureCapacity(this->nextPackets_, totalSurvivors);
        this->EnsureCapacity(this->nextColdPackets_, totalSurvivors);
        auto expandedPackets = this->nextPackets_;
        auto expandedColdPackets = this->nextColdPackets_;
        auto splitCounts = this->survivorSplitCounts_;
        Kokkos::parallel_scan("storm_expand_ddmc_interface_splits",
            Kokkos::RangePolicy<>(0, primarySurvivors),
            KOKKOS_LAMBDA(const std::size_t survivor, std::size_t &splitOffset, const bool final)
            {
                const std::size_t copies = splitCounts(survivor);
                if(final && copies > 0)
                {
                    const DeviceParticle primary =
                        expandedPackets(survivor);
                    DeviceParticleCold splitCold =
                        expandedColdPackets(survivor);
                    splitCold.id =
                        std::numeric_limits<particle_id_t>::max();
                    const std::size_t output =
                        primarySurvivors + splitOffset;
                    for(std::size_t copy = 0; copy < copies; ++copy)
                    {
                        expandedPackets(output + copy) = primary;
                        expandedColdPackets(output + copy) = splitCold;
                    }
                }
                splitOffset += copies;
            });
        Kokkos::fence("KokkosLocalTransportExecutor split expansion");
        ++this->metrics_.synchronizationCount;
        }

    void LaunchGreyIMCTransport(const GreyIMCViews<DeviceVec3> &views, const std::size_t launchCount, const std::size_t remoteOffset)
    {
        this->PrepareHostTallies(views);
        auto threadViews = this->threadTransportViews_;
        const bool usePrivateEnergy = this->privateEnergyTallies_ && views.grid.cellCount != 0;
        auto cellSteps = this->cellSteps_;
        auto cellStepScatter = this->cellStepScatter_;
        const std::size_t maximumInnerSteps = this->maximumInnerSteps_;
        auto packets = this->packets_;
        auto nextPackets = this->nextPackets_;
        auto coldPackets = this->coldPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto survivorFlags = this->survivorFlags_;
        auto survivorSplitCounts = this->survivorSplitCounts_;
        auto completedTransports = this->completedTransports_;
        auto fallbackTransports = this->fallbackTransports_;
        auto pendingRemotes = this->pendingRemotes_;
        auto waveCounters = this->waveCounters_;
        const std::size_t completedCapacity = completedTransports.extent(0);
        const std::size_t fallbackCapacity = fallbackTransports.extent(0);
        const std::size_t remoteCapacity = pendingRemotes.extent(0);

        // CPU particles have uneven path lengths; let idle threads take chunks.
        using TransportSchedule = std::conditional_t<hostAccessible,
            Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::Schedule<Kokkos::Static>>;
        using CounterView = Kokkos::View<std::size_t,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        CounterView physicsSteps(&this->waveCounters_.data()->physicsSteps);
        Kokkos::parallel_reduce(
            "storm_grey_imc_transport",
            Kokkos::RangePolicy<TransportSchedule>(0, launchCount).set_chunk_size(hostAccessible ? 4 : 1),
            KOKKOS_LAMBDA(const std::size_t i, std::size_t &stepSum)
            {
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
            const auto &particleViews = usePrivateEnergy ? threadViews(omp_get_thread_num()) : views;
#else
            const auto &particleViews = views;
#endif
            const auto workCellCount = cellSteps.extent(0);
            auto cellStepAccess = cellStepScatter.access();
            DeviceParticle particle = packets(i);
            DeviceParticleCold cold;
            AssignCold(cold, coldPackets(i));
            survivorFlags(i) = 0;
            TransportResult result;
            std::size_t pendingExtraSplits = 0;
            std::size_t taken = 0;
            for(std::size_t step = 0; step < maximumInnerSteps; ++step)
            {
                ++taken;
                ++particle.steps;
                if constexpr(hostAccessible)
                {
                    if(static_cast<std::size_t>(particle.cellIndex) < workCellCount)
                        cellStepAccess(particle.cellIndex) += std::size_t(1);
                }
                result = gpu::AdvanceOne(particle, cold, particleViews);
                if(result.ddmcExtraSplits > 0)
                {
                    pendingExtraSplits = result.ddmcExtraSplits;
                }
                if(TryKeepPacketOnDevice(particle, cold, result, particleViews))
                {
                    if(pendingExtraSplits > 0)
                    {
                        break;
                    }
                    continue;
                }
                break;
            }

            stepSum += taken;
            if(result.error == TransportError::HostFallback)
            {
                CompletedTransport transport;
                transport.particle = particle;
                AssignCold(transport.cold, cold);
                transport.result = result;
                const std::size_t fallbackIndex = Kokkos::atomic_fetch_add(&waveCounters().fallback, std::size_t(1));
                if(fallbackIndex >= fallbackCapacity)
                {
                    Kokkos::atomic_fetch_add(
                        &waveCounters().overflow, std::size_t(1));
                    return;
                }
                fallbackTransports(fallbackIndex) = transport;
                return;
            }
            if(result.error == TransportError::None &&
               result.step.change == ParticleStatus::NO_CELL_MOVE)
            {
                packets(i) = particle;
                AssignCold(coldPackets(i), cold);
                survivorSplitCounts(i) = pendingExtraSplits;
                survivorFlags(i) = 1;
                if(pendingExtraSplits != 0)
                    Kokkos::atomic_fetch_add(&waveCounters().appended, pendingExtraSplits);
                return;
            }

            CompletedTransport transport;
            transport.particle = particle;
            AssignCold(transport.cold, cold);
            transport.result = result;
            if(IsCensusTerminal(particle, result))
            {
                const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
                if(particleViews.grid.cellIDs != nullptr and cellIndex < particleViews.grid.cellCount)
                {
                    cold.cellID = particleViews.grid.cellIDs[cellIndex];
                }
                AccumulateCensusEnergy(particle, particleViews);
                const std::size_t censusIndex = Kokkos::atomic_fetch_add(&waveCounters().census, std::size_t(1));
                Kokkos::atomic_fetch_add(&waveCounters().censusSteps, static_cast<std::size_t>(particle.steps));
                nextPackets(launchCount - 1 - censusIndex) = particle;
                AssignCold(nextColdPackets(launchCount - 1 - censusIndex), cold);
            }
            else if(IsRankHopTerminal(particle, result, particleViews))
            {
                const std::size_t remoteIndex = remoteOffset + Kokkos::atomic_fetch_add(&waveCounters().remote, std::size_t(1));
                if(remoteIndex >= remoteCapacity)
                {
                    Kokkos::atomic_fetch_add(
                        &waveCounters().overflow, std::size_t(1));
                    return;
                }
                pendingRemotes(remoteIndex) = transport;
            }
            else
            {
                const std::size_t terminalIndex = Kokkos::atomic_fetch_add(&waveCounters().terminal, std::size_t(1));
                if(terminalIndex >= completedCapacity)
                {
                    Kokkos::atomic_fetch_add(
                        &waveCounters().overflow, std::size_t(1));
                    return;
                }
                completedTransports(terminalIndex) = transport;
            }
            }, physicsSteps);
    }

    void CompactSurvivors(const std::size_t launchCount)
    {
        auto packets = this->packets_;
        auto coldPackets = this->coldPackets_;
        auto survivorFlags = this->survivorFlags_;
        auto splitCounts = this->survivorSplitCounts_;
        auto nextPackets = this->nextPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto compactedSplitCounts =
            this->compactedSurvivorSplitCounts_;
        auto waveCounters = this->waveCounters_;

        Kokkos::parallel_scan(
            "storm_compact_transport_survivors",
            Kokkos::RangePolicy<>(0, launchCount),
            KOKKOS_LAMBDA(
                const std::size_t i,
                std::size_t &output,
                const bool final)
            {
                const bool keep = survivorFlags(i) != 0;
                if(final && keep)
                {
                    nextPackets(output) = packets(i);
                    AssignCold(nextColdPackets(output), coldPackets(i));
                    compactedSplitCounts(output) = splitCounts(i);
                }
                if(keep)
                {
                    ++output;
                }
                if(final && i + 1 == launchCount)
                {
                    waveCounters().survivor = output;
                }
            });

        std::swap(
            this->survivorSplitCounts_,
            this->compactedSurvivorSplitCounts_);
    }

private:
    template<typename ViewT>
    void EnsureCapacity(ViewT &view, std::size_t required)
    {
        if(view.extent(0) < required)
        {
            Kokkos::resize(view, required);
            ++this->metrics_.reallocationCount;
        }
    }

public:
    void CompactCappedWave(const std::size_t launchCount, const std::size_t totalSurvivors, const std::size_t previousActiveCount)
    {
        const std::size_t leftoverCount = previousActiveCount - launchCount;
        if(leftoverCount == 0)
        {
            this->activeCount_ = totalSurvivors;
            std::swap(this->packets_, this->nextPackets_);
            std::swap(this->coldPackets_, this->nextColdPackets_);
            return;
        }
        if(totalSurvivors > launchCount)
        {
            throw std::runtime_error("GPU transport split expansion exceeded the capped wave while unlaunched particles remain");
        }
        const std::size_t newActive = totalSurvivors + leftoverCount;
        this->EnsureCapacity(this->packets_, newActive);
        this->EnsureCapacity(this->coldPackets_, newActive);
        auto packets = this->packets_;
        auto coldPackets = this->coldPackets_;
        auto nextPackets = this->nextPackets_;
        auto nextCold = this->nextColdPackets_;
        Kokkos::parallel_for("storm_copy_wave_survivors",
            Kokkos::RangePolicy<>(0, totalSurvivors),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                packets(i) = nextPackets(i);
                coldPackets(i) = nextCold(i);
            });
        if(totalSurvivors < launchCount)
        {
            std::size_t dest = totalSurvivors;
            std::size_t src = launchCount;
            std::size_t remaining = leftoverCount;
            const std::size_t scratch = this->nextPackets_.extent(0);
            if(scratch == 0)
            {
                throw std::runtime_error("GPU transport leftover compact needs a scratch buffer");
            }
            while(remaining > 0)
            {
                const std::size_t chunk = std::min(scratch, remaining);
                auto destP = this->packets_;
                auto destC = this->coldPackets_;
                auto scratchP = this->nextPackets_;
                auto scratchC = this->nextColdPackets_;
                Kokkos::parallel_for("storm_scratch_leftover",
                    Kokkos::RangePolicy<>(0, chunk),
                    KOKKOS_LAMBDA(const std::size_t i)
                    {
                        scratchP(i) = destP(src + i);
                        scratchC(i) = destC(src + i);
                    });
                Kokkos::parallel_for("storm_place_leftover",
                    Kokkos::RangePolicy<>(0, chunk),
                    KOKKOS_LAMBDA(const std::size_t i)
                    {
                        destP(dest + i) = scratchP(i);
                        destC(dest + i) = scratchC(i);
                    });
                dest += chunk;
                src += chunk;
                remaining -= chunk;
            }
        }
        this->activeCount_ = newActive;
    }

    void HarvestWaveCensus(const std::size_t censusOffset, const std::size_t censusCount, const std::size_t activeCount)
    {
        if(censusCount == 0)
        {
            return;
        }
        const std::size_t required = censusOffset + censusCount;
        this->EnsureCapacity(this->censusPackets_, required);
        this->EnsureCapacity(this->censusCold_, required);
        auto nextPackets = this->nextPackets_;
        auto nextCold = this->nextColdPackets_;
        auto censusPackets = this->censusPackets_;
        auto censusCold = this->censusCold_;
        Kokkos::parallel_for("storm_harvest_wave_census",
            Kokkos::RangePolicy<>(0, censusCount),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                const std::size_t source = activeCount - censusCount + i;
                censusPackets(censusOffset + i) = nextPackets(source);
                censusCold(censusOffset + i) = nextCold(source);
            });
    }

private:
    // Reuse thread-private energy storage and per-cell work counters across waves.
    void PrepareHostTallies(const GreyIMCViews<DeviceVec3> &views)
    {
        if(this->privateEnergyTallies_ && views.grid.cellCount != 0)
        {
            const auto threads = Kokkos::DefaultExecutionSpace::concurrency();
            if(this->threadEnergyTallies_.extent(2) != views.grid.cellCount ||
               this->threadEnergyTallies_.extent(0) != static_cast<std::size_t>(threads))
                this->threadEnergyTallies_ = decltype(this->threadEnergyTallies_)(
                    "storm_private_energy", threads, 2, views.grid.cellCount);
            this->privateMaterialTarget_ = views.pendingMaterialEnergy;
            this->privateRadiationTarget_ = views.pendingRadiationEnergy;
            if(this->threadTransportViews_.extent(0) != static_cast<std::size_t>(threads))
                this->threadTransportViews_ = decltype(this->threadTransportViews_)("storm_thread_views", threads);
            for(int thread = 0; thread < threads; ++thread)
            {
                auto &threadView = this->threadTransportViews_(thread);
                threadView = views;
                threadView.privateEnergyTallies = true;
                if(views.pendingMaterialEnergy)
                    threadView.pendingMaterialEnergy = &this->threadEnergyTallies_(thread, 0, 0);
                if(views.pendingRadiationEnergy)
                    threadView.pendingRadiationEnergy = &this->threadEnergyTallies_(thread, 1, 0);
            }
        }
        if constexpr(hostAccessible)
        {
            if(this->cellSteps_.extent(0) != views.grid.cellCount)
            {
                this->cellSteps_ = Kokkos::View<std::size_t*>("storm_cell_work", views.grid.cellCount);
                this->cellStepScatter_ = Kokkos::Experimental::create_scatter_view(this->cellSteps_);
            }
        }
    }

    void ReserveForIngest(const std::size_t incoming)
    {
        const std::size_t required = this->activeCount_ + incoming;
        if constexpr(!hostAccessible)
        {
            this->EnsureCapacity(this->hostPackets_, incoming);
            this->EnsureCapacity(this->hostColdPackets_, incoming);
        }
        this->EnsureCapacity(this->packets_, required);
        this->EnsureCapacity(this->nextPackets_, required);
        this->EnsureCapacity(this->coldPackets_, required);
        this->EnsureCapacity(this->nextColdPackets_, required);
        this->EnsureCapacity(this->survivorFlags_, required);
        this->EnsureCapacity(this->survivorSplitCounts_, required);
        this->EnsureCapacity(
            this->compactedSurvivorSplitCounts_, required);
        this->EnsureCapacity(this->completedTransports_, incoming);
        this->EnsureCapacity(this->fallbackTransports_, incoming);
        this->EnsureCapacity(this->pendingRemotes_, this->pendingRemoteCount_ + incoming);
        if constexpr(!hostAccessible)
        {
            this->EnsureCapacity(this->hostEventTransports_, incoming);
            this->EnsureCapacity(this->hostRemoteTransports_, this->pendingRemoteCount_ + incoming);
        }
    }

    void ReserveForWave(const std::size_t activeCount, const std::size_t remoteOffset, const std::size_t, const bool fullEventBuffers)
    {
        this->EnsureCapacity(this->nextPackets_, activeCount);
        this->EnsureCapacity(this->nextColdPackets_, activeCount);
        this->EnsureCapacity(this->survivorFlags_, activeCount);
        this->EnsureCapacity(this->survivorSplitCounts_, activeCount);
        this->EnsureCapacity(
            this->compactedSurvivorSplitCounts_, activeCount);
        const std::size_t eventGuess = fullEventBuffers
            ? activeCount
            : std::max<std::size_t>(65536, activeCount / 3);
        this->EnsureCapacity(this->completedTransports_, eventGuess);
        this->EnsureCapacity(
            this->fallbackTransports_,
            fullEventBuffers
                ? activeCount
                : std::max<std::size_t>(4096, eventGuess / 16));
        this->EnsureCapacity(
            this->pendingRemotes_,
            remoteOffset +
                (fullEventBuffers
                     ? activeCount
                     : std::max<std::size_t>(65536, activeCount / 8)));
        if constexpr(!hostAccessible)
        {
            this->EnsureCapacity(this->hostEventTransports_, eventGuess);
            this->EnsureCapacity(this->hostRemoteTransports_, remoteOffset + eventGuess);
        }
    }

    static void AppendSpan(std::vector<CompletedTransport> &destination,
                           const CompletedTransportSpan span)
    {
        if(span.empty())
        {
            return;
        }
        destination.insert(destination.end(), span.begin(), span.end());
    }

    CompletedTransportSpan CopyDeviceTransports(
                              Kokkos::View<CompletedTransport*> source,
                              Kokkos::View<CompletedTransport*, PinnedHostSpace> destination,
                              const std::size_t destinationOffset,
                              const std::size_t count)
    {
        if(count == 0)
        {
            return {};
        }
        if constexpr(hostAccessible)
        {
            Kokkos::DefaultExecutionSpace{}.fence("STORM direct completion access");
            ++this->metrics_.synchronizationCount;
            // Borrowed until the next executor call, just like the staged span.
            return {source.data(), count};
        }
        Kokkos::deep_copy(
            Kokkos::subview(destination,
                            std::pair<std::size_t, std::size_t>(
                                destinationOffset,
                                destinationOffset + count)),
            Kokkos::subview(source, std::pair<std::size_t, std::size_t>(0, count)));
        const std::size_t bytes = count * sizeof(CompletedTransport);
        this->metrics_.d2hBytes += bytes;
        this->metrics_.eliminatedHostCopyBytes += bytes;
        ++this->metrics_.synchronizationCount;
        return {destination.data() + destinationOffset, count};
    }

    void CopyFinishedToHost(CompletedBatch &completed,
                            const std::size_t terminalCount,
                            const std::size_t fallbackCount)
    {
        const std::chrono::steady_clock::time_point copyBackStart =
            std::chrono::steady_clock::now();
        completed.terminals = this->CopyDeviceTransports(
            this->completedTransports_, this->hostEventTransports_,
            0, terminalCount);
        completed.fallbacks = this->CopyDeviceTransports(
            this->fallbackTransports_, this->hostEventTransports_,
            terminalCount, fallbackCount);
        this->metrics_.terminalCount += terminalCount;
        this->metrics_.fallbackCount += fallbackCount;
        completed.copyBackSeconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - copyBackStart).count();
    }

    void FlushRemotesIfNeeded(CompletedBatch &completed,
                              const std::size_t minRemoteCopy,
                              const std::size_t maxRemoteHolds,
                              const bool force)
    {
        if(this->pendingRemoteCount_ == 0)
        {
            this->remoteHoldSkips_ = 0;
            return;
        }
        const bool fatEnough =
            minRemoteCopy == 0 or this->pendingRemoteCount_ >= minRemoteCopy;
        const bool heldTooLong =
            maxRemoteHolds > 0 and this->remoteHoldSkips_ >= maxRemoteHolds;
        if(not force and not fatEnough and not heldTooLong)
        {
            ++this->remoteHoldSkips_;
            return;
        }
        this->remoteHoldSkips_ = 0;
        const std::chrono::steady_clock::time_point copyBackStart =
            std::chrono::steady_clock::now();
        completed.remotes = this->CopyDeviceTransports(
            this->pendingRemotes_, this->hostRemoteTransports_,
            0, this->pendingRemoteCount_);
        this->metrics_.remoteCount += this->pendingRemoteCount_;
        completed.copyBackSeconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - copyBackStart).count();
        this->pendingRemoteCount_ = 0;
    }

    void CopyCensusToHost(CompletedBatch &completed, const bool clear = true)
    {
        if(this->pendingCensusCount_ == 0)
        {
            return;
        }
        const std::size_t count = this->pendingCensusCount_;
        const std::chrono::steady_clock::time_point copyBackStart = std::chrono::steady_clock::now();
        auto hostPackets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                Kokkos::subview(this->censusPackets_, std::pair<std::size_t, std::size_t>(0, count)));
        auto hostCold = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                Kokkos::subview(this->censusCold_, std::pair<std::size_t, std::size_t>(0, count)));
        this->EnsureCapacity(this->hostEventTransports_, count);
        for(std::size_t i = 0; i < count; ++i)
        {
            this->hostEventTransports_(i).particle = hostPackets(i);
            this->hostEventTransports_(i).cold = hostCold(i);
            this->hostEventTransports_(i).result = TransportResult{};
            // The dedicated census pool contains completed packets only.
            this->hostEventTransports_(i).result.step.change = MonteCarloParticleStatus::DONE;
        }
        completed.census = CompletedTransportSpan{this->hostEventTransports_.data(), count};
        this->metrics_.censusCopyCount += this->pendingCensusCount_;
        completed.copyBackSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - copyBackStart).count();
        if(clear)
        {
            this->pendingCensusCount_ = 0;
        }
    }

    Kokkos::View<GreyIMCViews<DeviceVec3>*, Kokkos::HostSpace> threadTransportViews_;
    bool privateEnergyTallies_ = false;
    Kokkos::View<double***, Kokkos::LayoutRight> threadEnergyTallies_;
    double *privateMaterialTarget_ = nullptr;
    double *privateRadiationTarget_ = nullptr;
    Kokkos::View<std::size_t*> cellSteps_;
    decltype(Kokkos::Experimental::create_scatter_view(
        std::declval<Kokkos::View<std::size_t*>>())) cellStepScatter_;
    std::size_t maximumInnerSteps_;
    bool overlapCommunication_ = false;
    std::size_t poolReservedActiveCapacity_ = 0;
    std::size_t poolReservedHostIngest_ = 0;
    std::size_t activeCount_ = 0;
    std::size_t pendingRemoteCount_ = 0;
    std::size_t pendingCensusCount_ = 0;
    std::size_t remoteHoldSkips_ = 0;
    Kokkos::View<DeviceParticle*, PinnedHostSpace> hostPackets_;
    Kokkos::View<DeviceParticleCold*, PinnedHostSpace> hostColdPackets_;
    Kokkos::View<DeviceParticle*> packets_;
    Kokkos::View<DeviceParticle*> nextPackets_;
    Kokkos::View<DeviceParticleCold*> coldPackets_;
    Kokkos::View<DeviceParticleCold*> nextColdPackets_;
    Kokkos::View<std::uint8_t*> survivorFlags_;
    Kokkos::View<std::size_t*> survivorSplitCounts_;
    Kokkos::View<std::size_t*> compactedSurvivorSplitCounts_;
    Kokkos::View<CompletedTransport*> completedTransports_;
    Kokkos::View<CompletedTransport*> fallbackTransports_;
    Kokkos::View<CompletedTransport*> pendingRemotes_;
    Kokkos::View<DeviceParticle*> censusPackets_;
    Kokkos::View<DeviceParticleCold*> censusCold_;
    Kokkos::View<std::size_t*> censusCellCounts_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostEventTransports_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostRemoteTransports_;
    Kokkos::View<WaveCounters> waveCounters_;
    Kokkos::View<WaveCounters, PinnedHostSpace> hostWaveCounters_;
    Kokkos::DefaultExecutionSpace remoteCopySpace_;
    ExecutionEvent kernelReady_, remoteReady_;
    Kokkos::View<CompletedTransport*> inFlightDeviceRemotes_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> inFlightHostRemotes_;
    std::size_t inFlightRemoteCount_ = 0;
    TransportExecutorMetrics metrics_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
