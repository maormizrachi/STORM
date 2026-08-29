#ifndef STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "DeviceParticle.hpp"
#include "GreyIMCKernel.hpp"

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
          pendingCensus_("storm_transport_pending_census", 0),
          hostEventTransports_("storm_transport_host_events", 0),
          hostRemoteTransports_("storm_transport_host_remotes", 0),
          waveCounters_("storm_transport_wave_counters")
    {}

    void Reset()
    {
        this->activeCount_ = 0;
        this->pendingRemoteCount_ = 0;
        this->pendingCensusCount_ = 0;
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

    bool DeviceBusy() const
    {
        return this->activeCount_ > 0 or this->pendingRemoteCount_ > 0;
    }

    const TransportExecutorMetrics &Metrics() const
    {
        return this->metrics_;
    }

    template<typename PointT>
    void Ingest(const std::vector<Particle<PointT>> &arrivals)
    {
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
        const std::size_t maximumInnerSteps = this->maximumInnerSteps_;
        const std::size_t remoteOffset = this->pendingRemoteCount_;
        const std::size_t censusOffset = this->pendingCensusCount_;
        this->ReserveForWave(activeCount, remoteOffset, censusOffset);
        auto packets = this->packets_;
        auto nextPackets = this->nextPackets_;
        auto coldPackets = this->coldPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto survivorSplitCounts = this->survivorSplitCounts_;
        auto completedTransports = this->completedTransports_;
        auto fallbackTransports = this->fallbackTransports_;
        auto pendingRemotes = this->pendingRemotes_;
        auto pendingCensus = this->pendingCensus_;
        auto waveCounters = this->waveCounters_;

        const std::chrono::steady_clock::time_point deviceStart = std::chrono::steady_clock::now();
        Kokkos::deep_copy(this->waveCounters_, WaveCounters{});
        Kokkos::parallel_for(
            "storm_grey_imc_transport",
            Kokkos::RangePolicy<>(0, activeCount),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                DeviceParticle particle = packets(i);
                DeviceParticleCold cold = coldPackets(i);
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
                    if(TryKeepPacketOnDevice(particle, result, views))
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
                    transport.cold = cold;
                    transport.result = result;
                    const std::size_t fallbackIndex =
                        Kokkos::atomic_fetch_add(
                            &waveCounters().fallback, std::size_t(1));
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
                    nextColdPackets(survivorIndex) = cold;
                    survivorSplitCounts(survivorIndex) = pendingExtraSplits;
                    Kokkos::atomic_fetch_add(
                        &waveCounters().appended, pendingExtraSplits);
                    return;
                }

                CompletedTransport transport;
                transport.particle = particle;
                transport.cold = cold;
                transport.result = result;
                if(IsCensusTerminal(particle, result))
                {
                    const std::size_t censusIndex =
                        censusOffset +
                        Kokkos::atomic_fetch_add(
                            &waveCounters().census, std::size_t(1));
                    Kokkos::atomic_fetch_add(
                        &waveCounters().censusSteps,
                        static_cast<std::size_t>(particle.steps));
                    pendingCensus(censusIndex) = transport;
                }
                else if(IsRankHopTerminal(particle, result, views))
                {
                    const std::size_t remoteIndex =
                        remoteOffset +
                        Kokkos::atomic_fetch_add(
                            &waveCounters().remote, std::size_t(1));
                    pendingRemotes(remoteIndex) = transport;
                }
                else
                {
                    const std::size_t terminalIndex =
                        Kokkos::atomic_fetch_add(
                            &waveCounters().terminal, std::size_t(1));
                    completedTransports(terminalIndex) = transport;
                }
            });

        const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
        progress();
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
        if(counters.appended > 0)
        {
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
        completed.launchedParticles = activeCount;
        completed.censusCount = counters.census;
        completed.censusSteps = counters.censusSteps;
        completed.createdParticles = counters.appended;
        this->metrics_.splitCreatedCount += counters.appended;

        this->activeCount_ = totalSurvivors;
        this->metrics_.maxActiveCount =
            std::max(this->metrics_.maxActiveCount, totalSurvivors);
        this->pendingRemoteCount_ = remoteOffset + counters.remote;
        this->pendingCensusCount_ = censusOffset + counters.census;
        std::swap(this->packets_, this->nextPackets_);
        std::swap(this->coldPackets_, this->nextColdPackets_);

        this->CopyFinishedToHost(
            completed, counters.terminal, counters.fallback);
        this->FlushRemotesIfNeeded(
            completed, minRemoteCopy, maxRemoteHolds, this->activeCount_ == 0);
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
    template<typename ViewT>
    void EnsureCapacity(ViewT &view, std::size_t required)
    {
        if(view.extent(0) < required)
        {
            const std::size_t current = view.extent(0);
            const std::size_t increment =
                std::max<std::size_t>(1024, current / 2);
            const std::size_t geometric =
                current > std::numeric_limits<std::size_t>::max() - increment
                    ? required
                    : current + increment;
            Kokkos::resize(view, std::max(required, geometric));
            ++this->metrics_.reallocationCount;
        }
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
        this->EnsureCapacity(this->completedTransports_, required);
        this->EnsureCapacity(this->fallbackTransports_, required);
        this->EnsureCapacity(
            this->pendingRemotes_, this->pendingRemoteCount_ + incoming);
        this->EnsureCapacity(
            this->pendingCensus_, this->pendingCensusCount_ + incoming);
        this->EnsureCapacity(
            this->hostEventTransports_,
            std::max(required, this->pendingCensusCount_ + incoming));
        this->EnsureCapacity(
            this->hostRemoteTransports_, this->pendingRemoteCount_ + incoming);
    }

    void ReserveForWave(const std::size_t activeCount,
                        const std::size_t remoteOffset,
                        const std::size_t censusOffset)
    {
        this->EnsureCapacity(this->nextPackets_, activeCount);
        this->EnsureCapacity(this->nextColdPackets_, activeCount);
        this->EnsureCapacity(this->survivorSplitCounts_, activeCount);
        this->EnsureCapacity(this->completedTransports_, activeCount);
        this->EnsureCapacity(this->fallbackTransports_, activeCount);
        this->EnsureCapacity(this->pendingRemotes_, remoteOffset + activeCount);
        this->EnsureCapacity(this->pendingCensus_, censusOffset + activeCount);
        this->EnsureCapacity(
            this->hostEventTransports_,
            std::max(activeCount, censusOffset + activeCount));
        this->EnsureCapacity(
            this->hostRemoteTransports_, remoteOffset + activeCount);
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

    void CopyCensusToHost(CompletedBatch &completed)
    {
        if(this->pendingCensusCount_ == 0)
        {
            return;
        }
        const std::chrono::steady_clock::time_point copyBackStart =
            std::chrono::steady_clock::now();
        completed.census = this->CopyDeviceTransports(
            this->pendingCensus_, this->hostEventTransports_,
            0, this->pendingCensusCount_);
        this->metrics_.censusCopyCount += this->pendingCensusCount_;
        completed.copyBackSeconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - copyBackStart).count();
        this->pendingCensusCount_ = 0;
    }

    std::size_t maximumInnerSteps_;
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
    Kokkos::View<CompletedTransport*> pendingCensus_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostEventTransports_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostRemoteTransports_;
    Kokkos::View<WaveCounters> waveCounters_;
    TransportExecutorMetrics metrics_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
