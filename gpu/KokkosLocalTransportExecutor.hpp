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
          results_("storm_transport_results", 0),
          completedTransports_("storm_transport_completed", 0),
          hostCompletedTransports_("storm_transport_host_completed", 0),
          terminalCount_("storm_transport_terminal_count")
    {}

    void Reset()
    {
        this->activeCount_ = 0;
    }

    std::size_t ActiveCount() const
    {
        return this->activeCount_;
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
        EnsureCapacity(this->results_, required);
        EnsureCapacity(this->completedTransports_, required);
        EnsureCapacity(this->hostCompletedTransports_, required);

        for(std::size_t i = 0; i < incoming; ++i)
        {
            PackParticle(arrivals[i], this->hostPackets_(i), this->hostColdPackets_(i));
        }
        Kokkos::deep_copy(
            Kokkos::subview(
                this->packets_,
                std::pair<std::size_t, std::size_t>(offset, required)),
            Kokkos::subview(
                this->hostPackets_,
                std::pair<std::size_t, std::size_t>(0, incoming)));
        Kokkos::deep_copy(
            Kokkos::subview(
                this->coldPackets_,
                std::pair<std::size_t, std::size_t>(offset, required)),
            Kokkos::subview(
                this->hostColdPackets_,
                std::pair<std::size_t, std::size_t>(0, incoming)));
        this->activeCount_ = required;
    }

    template<typename ProgressFunction>
    CompletedBatch AdvanceWave(const GreyIMCViews<DeviceVec3> &views,
                               ProgressFunction progress)
    {
        CompletedBatch completed;
        if(this->activeCount_ == 0)
        {
            return completed;
        }
        if(!Kokkos::is_initialized())
        {
            throw std::runtime_error("Kokkos must be initialized before GPU transport");
        }

        const std::size_t activeCount = this->activeCount_;
        const std::size_t maximumInnerSteps = this->maximumInnerSteps_;
        auto packets = this->packets_;
        auto nextPackets = this->nextPackets_;
        auto coldPackets = this->coldPackets_;
        auto nextColdPackets = this->nextColdPackets_;
        auto results = this->results_;
        auto completedTransports = this->completedTransports_;

        const std::chrono::steady_clock::time_point deviceStart = std::chrono::steady_clock::now();
        std::size_t physicsSteps = 0;
        Kokkos::parallel_reduce(
            "storm_grey_imc_transport",
            Kokkos::RangePolicy<>(0, activeCount),
            KOKKOS_LAMBDA(const std::size_t i, std::size_t &localSteps)
            {
                DeviceParticle particle = packets(i);
                TransportResult result;
                std::size_t taken = 0;
                for(std::size_t step = 0; step < maximumInnerSteps; ++step)
                {
                    ++taken;
                    ++particle.steps;
                    result = AdvanceOne(particle, views);
                    if(result.error != TransportError::None)
                    {
                        break;
                    }
                    if(result.step.change == ParticleStatus::CELL_MOVE &&
                       result.step.nextCellIndex < views.grid.cellCount)
                    {
                        particle.cellIndex = result.step.nextCellIndex;
                        const DeviceVec3 &center = views.grid.cellCenters[particle.cellIndex];
                        constexpr double epsilon = 1.0e-8;
                        particle.location.x =
                            (1.0 - epsilon) * particle.location.x + epsilon * center.x;
                        particle.location.y =
                            (1.0 - epsilon) * particle.location.y + epsilon * center.y;
                        particle.location.z =
                            (1.0 - epsilon) * particle.location.z + epsilon * center.z;
                        result.step.change = ParticleStatus::NO_CELL_MOVE;
                        continue;
                    }
                    if(result.step.change == ParticleStatus::NO_CELL_MOVE)
                    {
                        continue;
                    }
                    break;
                }
                packets(i) = particle;
                results(i) = result;
                localSteps += taken;
            },
            physicsSteps);

        const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
        progress();
        completed.progressSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - progressStart).count();

        std::size_t zeroTerminals = 0;
        Kokkos::deep_copy(this->terminalCount_, zeroTerminals);
        Kokkos::parallel_scan(
            "storm_grey_imc_compact",
            Kokkos::RangePolicy<>(0, activeCount),
            KOKKOS_LAMBDA(const std::size_t i, std::size_t &terminalPrefix, const bool final)
            {
                const TransportResult result = results(i);
                const bool terminal =
                    result.error != TransportError::None ||
                    result.step.change != ParticleStatus::NO_CELL_MOVE;
                if(terminal)
                {
                    const std::size_t terminalIndex = terminalPrefix++;
                    if(final)
                    {
                        CompletedTransport transport;
                        transport.particle = packets(i);
                        transport.cold = coldPackets(i);
                        transport.result = result;
                        completedTransports(terminalIndex) = transport;
                    }
                }
                else if(final)
                {
                    const std::size_t survivorIndex = i - terminalPrefix;
                    nextPackets(survivorIndex) = packets(i);
                    nextColdPackets(survivorIndex) = coldPackets(i);
                }
            },
            this->terminalCount_);

        std::size_t terminalCount = 0;
        Kokkos::deep_copy(terminalCount, this->terminalCount_);
        completed.deviceSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - deviceStart).count();
        completed.launchCount = 1;
        completed.physicsSteps = physicsSteps;
        completed.launchedParticles = activeCount;

        this->activeCount_ = activeCount - terminalCount;
        std::swap(this->packets_, this->nextPackets_);
        std::swap(this->coldPackets_, this->nextColdPackets_);

        const std::chrono::steady_clock::time_point copyBackStart = std::chrono::steady_clock::now();
        completed.particles.resize(terminalCount);
        if(terminalCount > 0)
        {
            Kokkos::deep_copy(
                Kokkos::subview(this->hostCompletedTransports_,
                                std::pair<std::size_t, std::size_t>(0, terminalCount)),
                Kokkos::subview(this->completedTransports_,
                                std::pair<std::size_t, std::size_t>(0, terminalCount)));
            std::memcpy(
                completed.particles.data(),
                this->hostCompletedTransports_.data(),
                terminalCount * sizeof(CompletedTransport));
        }
        completed.copyBackSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - copyBackStart).count();
        return completed;
    }

    template<typename PointT>
    CompletedBatch Execute(const std::vector<Particle<PointT>> &particles,
                           std::size_t maximumParticles,
                           const GreyIMCViews<DeviceVec3> &views)
    {
        return this->Execute(particles, maximumParticles, views, [](){});
    }

    template<typename PointT, typename ProgressFunction>
    CompletedBatch Execute(const std::vector<Particle<PointT>> &particles,
                           std::size_t maximumParticles,
                           const GreyIMCViews<DeviceVec3> &views,
                           ProgressFunction progress)
    {
        this->Reset();
        CompletedBatch completed;
        const std::size_t count = std::min(maximumParticles, particles.size());
        if(count == 0)
        {
            return completed;
        }

        const std::chrono::steady_clock::time_point packStart = std::chrono::steady_clock::now();
        std::vector<Particle<PointT>> arrivals(
            particles.end() - static_cast<std::ptrdiff_t>(count),
            particles.end());
        this->Ingest(arrivals);
        completed.packSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();

        while(this->activeCount_ > 0)
        {
            CompletedBatch wave = this->AdvanceWave(views, progress);
            completed.deviceSeconds += wave.deviceSeconds;
            completed.copyBackSeconds += wave.copyBackSeconds;
            completed.progressSeconds += wave.progressSeconds;
            completed.launchCount += wave.launchCount;
            completed.physicsSteps += wave.physicsSteps;
            completed.launchedParticles += wave.launchedParticles;
            completed.particles.insert(
                completed.particles.end(),
                std::make_move_iterator(wave.particles.begin()),
                std::make_move_iterator(wave.particles.end()));
        }
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

    std::size_t maximumInnerSteps_;
    std::size_t activeCount_ = 0;
    Kokkos::View<DeviceParticle*, PinnedHostSpace> hostPackets_;
    Kokkos::View<DeviceParticleCold*, PinnedHostSpace> hostColdPackets_;
    Kokkos::View<DeviceParticle*> packets_;
    Kokkos::View<DeviceParticle*> nextPackets_;
    Kokkos::View<DeviceParticleCold*> coldPackets_;
    Kokkos::View<DeviceParticleCold*> nextColdPackets_;
    Kokkos::View<TransportResult*> results_;
    Kokkos::View<CompletedTransport*> completedTransports_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostCompletedTransports_;
    Kokkos::View<std::size_t> terminalCount_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
