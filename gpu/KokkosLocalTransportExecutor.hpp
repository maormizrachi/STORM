#ifndef STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

struct CompletedBatch
{
    std::vector<CompletedTransport> particles;
    std::vector<std::size_t> cellSteps;
    double packSeconds = 0.0;
    double deviceSeconds = 0.0;
    double copyBackSeconds = 0.0;
    double progressSeconds = 0.0;
    std::size_t launchCount = 0;
    std::size_t physicsSteps = 0;
    std::size_t launchedParticles = 0;
    std::size_t censusCount = 0;
    std::size_t censusSteps = 0;
};

struct WaveCounters
{
    std::size_t finished = 0;
    std::size_t remote = 0;
    std::size_t survivor = 0;
    std::size_t census = 0;
    std::size_t physicsSteps = 0;
    std::size_t censusSteps = 0;
};
static_assert(std::is_trivially_copyable<WaveCounters>::value,
              "WaveCounters must be safe for device-host copies");

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
          completedTransports_("storm_transport_completed", 0),
          pendingRemotes_("storm_transport_pending_remotes", 0),
          pendingCensus_("storm_transport_pending_census", 0),
          hostCompletedTransports_("storm_transport_host_completed", 0),
          waveCounters_("storm_transport_wave_counters")
    {}

    void Reset()
    {
        this->activeCount_ = 0;
        this->pendingRemoteCount_ = 0;
        this->pendingCensusCount_ = 0;
        this->remoteHoldSkips_ = 0;
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
        EnsureCapacity(this->hostPackets_, incoming);
        EnsureCapacity(this->hostColdPackets_, incoming);
        EnsureCapacity(this->packets_, required);
        EnsureCapacity(this->nextPackets_, required);
        EnsureCapacity(this->coldPackets_, required);
        EnsureCapacity(this->nextColdPackets_, required);
        EnsureCapacity(this->completedTransports_, incoming);
        EnsureCapacity(this->pendingRemotes_, this->pendingRemoteCount_ + incoming);
        EnsureCapacity(this->pendingCensus_, this->pendingCensusCount_ + incoming);
        EnsureCapacity(this->hostCompletedTransports_,
                       std::max(incoming, std::max(this->pendingRemoteCount_,
                                                   this->pendingCensusCount_)));

        for(std::size_t i = 0; i < incoming; ++i)
        {
            PackParticle(arrivals[i], this->hostPackets_(i), this->hostColdPackets_(i));
        }
        Kokkos::deep_copy(Kokkos::subview(this->packets_, std::pair<std::size_t, std::size_t>(offset, required)),
                            Kokkos::subview(this->hostPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
        Kokkos::deep_copy(Kokkos::subview(this->coldPackets_, std::pair<std::size_t, std::size_t>(offset, required)),
                            Kokkos::subview(this->hostColdPackets_, std::pair<std::size_t, std::size_t>(0, incoming)));
        this->activeCount_ = required;
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
        EnsureCapacity(this->nextPackets_, activeCount);
        EnsureCapacity(this->nextColdPackets_, activeCount);
        EnsureCapacity(this->completedTransports_, activeCount);
        EnsureCapacity(this->pendingRemotes_, remoteOffset + activeCount);
        EnsureCapacity(this->pendingCensus_, censusOffset + activeCount);
        auto packets = this->packets_;
        auto nextPackets = this->nextPackets_;
        auto coldPackets = this->coldPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto completedTransports = this->completedTransports_;
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
                std::size_t taken = 0;
                for(std::size_t step = 0; step < maximumInnerSteps; ++step)
                {
                    ++taken;
                    ++particle.steps;
                    result = gpu::AdvanceOne(particle, cold, views);
                    if(TryKeepPacketOnDevice(particle, result, views))
                    {
                        continue;
                    }
                    break;
                }

                Kokkos::atomic_fetch_add(
                    &waveCounters().physicsSteps, taken);
                if(result.error == TransportError::None &&
                   result.step.change == ParticleStatus::NO_CELL_MOVE)
                {
                    const std::size_t survivorIndex =
                        Kokkos::atomic_fetch_add(
                            &waveCounters().survivor, std::size_t(1));
                    nextPackets(survivorIndex) = particle;
                    nextColdPackets(survivorIndex) = cold;
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
                    const std::size_t finishedIndex =
                        Kokkos::atomic_fetch_add(
                            &waveCounters().finished, std::size_t(1));
                    completedTransports(finishedIndex) = transport;
                }
            });

        const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
        progress();
        completed.progressSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - progressStart).count();

        WaveCounters counters;
        Kokkos::deep_copy(counters, this->waveCounters_);
        completed.deviceSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - deviceStart).count();
        completed.launchCount = 1;
        completed.physicsSteps = counters.physicsSteps;
        completed.launchedParticles = activeCount;
        completed.censusCount = counters.census;
        completed.censusSteps = counters.censusSteps;

        this->activeCount_ = counters.survivor;
        this->pendingRemoteCount_ = remoteOffset + counters.remote;
        this->pendingCensusCount_ = censusOffset + counters.census;
        std::swap(this->packets_, this->nextPackets_);
        std::swap(this->coldPackets_, this->nextColdPackets_);

        this->CopyFinishedToHost(completed, counters.finished);
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
            completed.particles.insert(completed.particles.end(), std::make_move_iterator(wave.particles.begin()), std::make_move_iterator(wave.particles.end()));
        }
        CompletedBatch census = this->FlushPendingCensus();
        completed.copyBackSeconds += census.copyBackSeconds;
        completed.particles.insert(completed.particles.end(),
                                   std::make_move_iterator(census.particles.begin()),
                                   std::make_move_iterator(census.particles.end()));
        completed.censusCount = 0;
        completed.censusSteps = 0;
        return completed;
    }

private:
    template<typename ViewT>
    static void EnsureCapacity(ViewT &view, std::size_t required)
    {
        if(view.extent(0) < required)
        {
            Kokkos::resize(view, required);
        }
    }

    void CopyDeviceTransports(CompletedBatch &completed,
                              Kokkos::View<CompletedTransport*> source,
                              const std::size_t count)
    {
        if(count == 0)
        {
            return;
        }
        const std::size_t offset = completed.particles.size();
        completed.particles.resize(offset + count);
        EnsureCapacity(this->hostCompletedTransports_, count);
        Kokkos::deep_copy(
            Kokkos::subview(this->hostCompletedTransports_,
                            std::pair<std::size_t, std::size_t>(0, count)),
            Kokkos::subview(source, std::pair<std::size_t, std::size_t>(0, count)));
        std::memcpy(completed.particles.data() + offset,
                    this->hostCompletedTransports_.data(),
                    count * sizeof(CompletedTransport));
    }

    void CopyFinishedToHost(CompletedBatch &completed, const std::size_t finishedCount)
    {
        const std::chrono::steady_clock::time_point copyBackStart =
            std::chrono::steady_clock::now();
        this->CopyDeviceTransports(
            completed, this->completedTransports_, finishedCount);
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
        this->CopyDeviceTransports(
            completed, this->pendingRemotes_, this->pendingRemoteCount_);
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
        this->CopyDeviceTransports(
            completed, this->pendingCensus_, this->pendingCensusCount_);
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
    Kokkos::View<CompletedTransport*> completedTransports_;
    Kokkos::View<CompletedTransport*> pendingRemotes_;
    Kokkos::View<CompletedTransport*> pendingCensus_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostCompletedTransports_;
    Kokkos::View<WaveCounters> waveCounters_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
