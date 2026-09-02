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
#include "GreyIMCKernel.hpp"
#include "ProfileRegion.hpp"

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
};

class KokkosLocalTransportExecutor
{
#if defined(KOKKOS_ENABLE_HIP)
    using PinnedHostSpace = Kokkos::HIPHostPinnedSpace;
#else
    using PinnedHostSpace = Kokkos::HostSpace;
#endif

public:
    explicit KokkosLocalTransportExecutor(std::size_t maximumInnerSteps)
        : maximumInnerSteps_(std::max<std::size_t>(1, maximumInnerSteps)),
          hostPackets_("storm_transport_host_packets", 0),
          hostColdPackets_("storm_transport_host_cold_packets", 0),
          packets_("storm_transport_packets", 0),
          nextPackets_("storm_transport_next_packets", 0),
          coldPackets_("storm_transport_cold_packets", 0),
          nextColdPackets_("storm_transport_next_cold_packets", 0),
          survivorSplitCounts_("storm_transport_survivor_split_counts", 0),
          completedTransports_("storm_transport_completed", 0),
          fallbackTransports_("storm_transport_fallbacks", 0),
          pendingRemotes_("storm_transport_pending_remotes", 0),
          censusPackets_("storm_transport_census_packets", 0),
          censusCold_("storm_transport_census_cold", 0),
          censusCellCounts_("storm_transport_census_cell_counts", 0),
          hostEventTransports_("storm_transport_host_events", 0),
          hostRemoteTransports_("storm_transport_host_remotes", 0),
          waveCounters_("storm_transport_wave_counters"),
          waveOverflow_("storm_transport_wave_overflow")
    {}

    void Reset()
    {
        this->activeCount_ = 0;
        this->pendingRemoteCount_ = 0;
        this->pendingCensusCount_ = 0;
        this->ResetStepMetrics();
    }

    void ResetStepMetrics()
    {
        this->remoteHoldSkips_ = 0;
        this->metrics_ = {};
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
        return this->activeCount_ > 0 or this->pendingRemoteCount_ > 0;
    }

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

        this->EnsureCapacity(this->hostPackets_, hostTarget);
        this->EnsureCapacity(this->hostColdPackets_, hostTarget);
        this->EnsureCapacity(this->packets_, activeTarget);
        this->EnsureCapacity(this->coldPackets_, activeTarget);
        this->poolReservedActiveCapacity_ = activeTarget;
        this->poolReservedHostIngest_ = hostTarget;
    }

    void PromotePendingCensus(const dt_t fullDt)
    {
        if(this->activeCount_ != 0 || this->pendingRemoteCount_ != 0)
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
        this->ShrinkTo(this->censusPackets_, 0);
        this->ShrinkTo(this->censusCold_, 0);
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

        for(std::size_t i = 0; i < incoming; ++i)
        {
            PackParticle(arrivals[i], this->hostPackets_(i), this->hostColdPackets_(i));
        }
        Kokkos::deep_copy(Kokkos::subview(this->packets_, std::pair<std::size_t, std::size_t>(offset, required)),
                            Kokkos::subview(this->hostPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
        Kokkos::deep_copy(Kokkos::subview(this->coldPackets_, std::pair<std::size_t, std::size_t>(offset, required)),
                            Kokkos::subview(this->hostColdPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
        this->activeCount_ = required;
        this->metrics_.h2dBytes += incoming *
            (sizeof(DeviceParticle) + sizeof(DeviceParticleCold));
        this->metrics_.synchronizationCount += 2;
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
        STORM_PROFILE_REGION("storm/transport/wave");
        CompletedBatch completed;
        if(this->activeCount_ == 0)
        {
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
        Kokkos::deep_copy(this->waveCounters_, WaveCounters{});
        Kokkos::deep_copy(this->waveOverflow_, 0);
        {
            STORM_PROFILE_REGION("storm/transport/kernel");
            this->LaunchGreyIMCTransport(views, launchCount, remoteOffset);
        }

        int overflow = 0;
        Kokkos::deep_copy(overflow, this->waveOverflow_);
        if(overflow != 0)
        {
            throw std::runtime_error(
                "GPU transport event buffers overflowed");
        }

        const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
        {
            STORM_PROFILE_REGION("storm/transport/progress");
            progress();
        }
        completed.progressSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - progressStart).count();

        WaveCounters counters;
        Kokkos::deep_copy(counters, this->waveCounters_);
        this->metrics_.synchronizationCount += 2;
        if(counters.appended >
           std::numeric_limits<std::size_t>::max() - counters.survivor)
        {
            throw std::overflow_error(
                "KokkosLocalTransportExecutor split output overflow");
        }
        const std::size_t totalSurvivors =
            counters.survivor + counters.appended;
        {
            STORM_PROFILE_REGION("storm/transport/harvest_census");
            this->HarvestWaveCensus(
                censusOffset, counters.census, launchCount);
        }
        if(counters.appended > 0)
        {
            STORM_PROFILE_REGION("storm/transport/splits");
            this->EnsureCapacity(this->nextPackets_, totalSurvivors);
            this->EnsureCapacity(this->nextColdPackets_, totalSurvivors);
            auto expandedPackets = this->nextPackets_;
            auto expandedColdPackets = this->nextColdPackets_;
            auto splitCounts = this->survivorSplitCounts_;
            const std::size_t primarySurvivors = counters.survivor;
            Kokkos::parallel_scan(
                "storm_expand_ddmc_interface_splits",
                Kokkos::RangePolicy<>(0, primarySurvivors),
                KOKKOS_LAMBDA(
                    const std::size_t survivor,
                    std::size_t &splitOffset,
                    const bool final)
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
            Kokkos::fence(
                "KokkosLocalTransportExecutor split expansion");
            ++this->metrics_.synchronizationCount;
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
            this->metrics_.maxActiveCount =
                std::max(this->metrics_.maxActiveCount, this->activeCount_);
            this->ShrinkTo(this->packets_, this->activeCount_);
            this->ShrinkTo(this->coldPackets_, this->activeCount_);
            this->ShrinkTo(this->nextPackets_, 0);
            this->ShrinkTo(this->nextColdPackets_, 0);
        }

        {
            STORM_PROFILE_REGION("storm/copy_back");
            this->CopyFinishedToHost(
                completed, counters.terminal, counters.fallback);
            this->FlushRemotesIfNeeded(
                completed, minRemoteCopy, maxRemoteHolds, this->activeCount_ == 0);
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
    void LaunchGreyIMCTransport(const GreyIMCViews<DeviceVec3> &views,
                                const std::size_t launchCount,
                                const std::size_t remoteOffset)
    {
        const std::size_t maximumInnerSteps = this->maximumInnerSteps_;
        auto packets = this->packets_;
        auto nextPackets = this->nextPackets_;
        auto coldPackets = this->coldPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto survivorSplitCounts = this->survivorSplitCounts_;
        auto completedTransports = this->completedTransports_;
        auto fallbackTransports = this->fallbackTransports_;
        auto pendingRemotes = this->pendingRemotes_;
        auto waveCounters = this->waveCounters_;
        auto waveOverflow = this->waveOverflow_;
        const std::size_t completedCapacity = completedTransports.extent(0);
        const std::size_t fallbackCapacity = fallbackTransports.extent(0);
        const std::size_t remoteCapacity = pendingRemotes.extent(0);

        Kokkos::parallel_for(
            "storm_grey_imc_transport",
            Kokkos::RangePolicy<>(0, launchCount),
            KOKKOS_LAMBDA(const std::size_t i)
            {
            DeviceParticle particle = packets(i);
            DeviceParticleCold cold;
            AssignCold(cold, coldPackets(i));
            TransportResult result;
            std::size_t pendingExtraSplits = 0;
            std::size_t taken = 0;
            for(std::size_t step = 0; step < maximumInnerSteps; ++step)
            {
                ++taken;
                ++particle.steps;
                result = gpu::AdvanceOne(particle, cold, views);
                if(result.ddmcExtraSplits > 0)
                {
                    pendingExtraSplits = result.ddmcExtraSplits;
                }
                if(TryKeepPacketOnDevice(
                       particle, cold, result, views))
                {
                    if(pendingExtraSplits > 0)
                    {
                        break;
                    }
                    continue;
                }
                break;
            }

            Kokkos::atomic_fetch_add(
                &waveCounters().physicsSteps, taken);
            if(result.error == TransportError::HostFallback)
            {
                CompletedTransport transport;
                transport.particle = particle;
                AssignCold(transport.cold, cold);
                transport.result = result;
                const std::size_t fallbackIndex =
                    Kokkos::atomic_fetch_add(
                        &waveCounters().fallback, std::size_t(1));
                if(fallbackIndex >= fallbackCapacity)
                {
                    Kokkos::atomic_fetch_add(&waveOverflow(), 1);
                    return;
                }
                fallbackTransports(fallbackIndex) = transport;
                return;
            }
            if(result.error == TransportError::None &&
               result.step.change == ParticleStatus::NO_CELL_MOVE)
            {
                const std::size_t survivorIndex =
                    Kokkos::atomic_fetch_add(
                        &waveCounters().survivor, std::size_t(1));
                nextPackets(survivorIndex) = particle;
                AssignCold(nextColdPackets(survivorIndex), cold);
                survivorSplitCounts(survivorIndex) = pendingExtraSplits;
                Kokkos::atomic_fetch_add(
                    &waveCounters().appended, pendingExtraSplits);
                return;
            }

            CompletedTransport transport;
            transport.particle = particle;
            AssignCold(transport.cold, cold);
            transport.result = result;
            if(IsCensusTerminal(particle, result))
            {
                const std::size_t cellIndex =
                    static_cast<std::size_t>(
                        particle.cellIndex);
                if(views.grid.cellIDs != nullptr &&
                   cellIndex < views.grid.cellCount)
                {
                    cold.cellID =
                        views.grid.cellIDs[cellIndex];
                }
                AccumulateCensusEnergy(particle, views);
                const std::size_t censusIndex =
                    Kokkos::atomic_fetch_add(
                        &waveCounters().census, std::size_t(1));
                Kokkos::atomic_fetch_add(
                    &waveCounters().censusSteps,
                    static_cast<std::size_t>(particle.steps));
                nextPackets(launchCount - 1 - censusIndex) = particle;
                AssignCold(nextColdPackets(launchCount - 1 - censusIndex), cold);
            }
            else if(IsRankHopTerminal(particle, result, views))
            {
                const std::size_t remoteIndex =
                    remoteOffset +
                    Kokkos::atomic_fetch_add(
                        &waveCounters().remote, std::size_t(1));
                if(remoteIndex >= remoteCapacity)
                {
                    Kokkos::atomic_fetch_add(&waveOverflow(), 1);
                    return;
                }
                pendingRemotes(remoteIndex) = transport;
            }
            else
            {
                const std::size_t terminalIndex =
                    Kokkos::atomic_fetch_add(
                        &waveCounters().terminal, std::size_t(1));
                if(terminalIndex >= completedCapacity)
                {
                    Kokkos::atomic_fetch_add(&waveOverflow(), 1);
                    return;
                }
                completedTransports(terminalIndex) = transport;
            }
            });
    }

    template<typename ViewT>
    void EnsureCapacity(ViewT &view, std::size_t required)
    {
        if(view.extent(0) < required)
        {
            Kokkos::resize(view, required);
            ++this->metrics_.reallocationCount;
        }
    }

    template<typename ViewT>
    void ShrinkTo(ViewT &view, const std::size_t used)
    {
        if(view.extent(0) <= used || view.extent(0) <= 65536)
        {
            return;
        }
        if(used * 2 >= view.extent(0))
        {
            return;
        }
        Kokkos::resize(view, used);
        ++this->metrics_.reallocationCount;
    }

    void CompactCappedWave(const std::size_t launchCount,
                           const std::size_t totalSurvivors,
                           const std::size_t previousActiveCount)
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
            throw std::runtime_error(
                "GPU transport split expansion exceeded the capped wave "
                "while unlaunched particles remain");
        }
        const std::size_t newActive = totalSurvivors + leftoverCount;
        this->EnsureCapacity(this->packets_, newActive);
        this->EnsureCapacity(this->coldPackets_, newActive);
        auto packets = this->packets_;
        auto coldPackets = this->coldPackets_;
        auto nextPackets = this->nextPackets_;
        auto nextCold = this->nextColdPackets_;
        Kokkos::parallel_for(
            "storm_copy_wave_survivors",
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
                throw std::runtime_error(
                    "GPU transport leftover compact needs a scratch buffer");
            }
            while(remaining > 0)
            {
                const std::size_t chunk = std::min(scratch, remaining);
                auto destP = this->packets_;
                auto destC = this->coldPackets_;
                auto scratchP = this->nextPackets_;
                auto scratchC = this->nextColdPackets_;
                Kokkos::parallel_for(
                    "storm_scratch_leftover",
                    Kokkos::RangePolicy<>(0, chunk),
                    KOKKOS_LAMBDA(const std::size_t i)
                    {
                        scratchP(i) = destP(src + i);
                        scratchC(i) = destC(src + i);
                    });
                Kokkos::parallel_for(
                    "storm_place_leftover",
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

    void HarvestWaveCensus(
        const std::size_t censusOffset,
        const std::size_t censusCount,
        const std::size_t activeCount)
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
        Kokkos::parallel_for(
            "storm_harvest_wave_census",
            Kokkos::RangePolicy<>(0, censusCount),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                const std::size_t source = activeCount - censusCount + i;
                censusPackets(censusOffset + i) = nextPackets(source);
                censusCold(censusOffset + i) = nextCold(source);
            });
    }

    void ReserveForIngest(const std::size_t incoming)
    {
        const std::size_t required = this->activeCount_ + incoming;
        this->EnsureCapacity(this->hostPackets_, incoming);
        this->EnsureCapacity(this->hostColdPackets_, incoming);
        this->EnsureCapacity(this->packets_, required);
        this->EnsureCapacity(this->nextPackets_, required);
        this->EnsureCapacity(this->coldPackets_, required);
        this->EnsureCapacity(this->nextColdPackets_, required);
        this->EnsureCapacity(this->survivorSplitCounts_, required);
        this->EnsureCapacity(this->completedTransports_, incoming);
        this->EnsureCapacity(this->fallbackTransports_, incoming);
        this->EnsureCapacity(
            this->pendingRemotes_, this->pendingRemoteCount_ + incoming);
        this->EnsureCapacity(
            this->hostEventTransports_, incoming);
        this->EnsureCapacity(
            this->hostRemoteTransports_, this->pendingRemoteCount_ + incoming);
    }

    void ReserveForWave(const std::size_t activeCount,
                        const std::size_t remoteOffset,
                        const std::size_t,
                        const bool fullEventBuffers)
    {
        this->EnsureCapacity(this->nextPackets_, activeCount);
        this->EnsureCapacity(this->nextColdPackets_, activeCount);
        this->EnsureCapacity(this->survivorSplitCounts_, activeCount);
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
        this->EnsureCapacity(this->hostEventTransports_, eventGuess);
        this->EnsureCapacity(
            this->hostRemoteTransports_,
            remoteOffset + eventGuess);
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

    void CopyCensusToHost(
        CompletedBatch &completed,
        const bool clear = true)
    {
        if(this->pendingCensusCount_ == 0)
        {
            return;
        }
        const std::size_t count = this->pendingCensusCount_;
        const std::chrono::steady_clock::time_point copyBackStart =
            std::chrono::steady_clock::now();
        auto hostPackets = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview(
                this->censusPackets_,
                std::pair<std::size_t, std::size_t>(0, count)));
        auto hostCold = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview(
                this->censusCold_,
                std::pair<std::size_t, std::size_t>(0, count)));
        this->EnsureCapacity(this->hostEventTransports_, count);
        for(std::size_t i = 0; i < count; ++i)
        {
            this->hostEventTransports_(i).particle = hostPackets(i);
            this->hostEventTransports_(i).cold = hostCold(i);
            this->hostEventTransports_(i).result = TransportResult{};
        }
        completed.census = CompletedTransportSpan{
            this->hostEventTransports_.data(), count};
        this->metrics_.censusCopyCount += this->pendingCensusCount_;
        completed.copyBackSeconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - copyBackStart).count();
        if(clear)
        {
            this->pendingCensusCount_ = 0;
        }
    }

    std::size_t maximumInnerSteps_;
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
    Kokkos::View<std::size_t*> survivorSplitCounts_;
    Kokkos::View<CompletedTransport*> completedTransports_;
    Kokkos::View<CompletedTransport*> fallbackTransports_;
    Kokkos::View<CompletedTransport*> pendingRemotes_;
    Kokkos::View<DeviceParticle*> censusPackets_;
    Kokkos::View<DeviceParticleCold*> censusCold_;
    Kokkos::View<std::size_t*> censusCellCounts_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostEventTransports_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostRemoteTransports_;
    Kokkos::View<WaveCounters> waveCounters_;
    Kokkos::View<int> waveOverflow_;
    TransportExecutorMetrics metrics_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
