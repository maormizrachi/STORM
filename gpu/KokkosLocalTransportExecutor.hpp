#ifndef STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_DualView.hpp>

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
    TransportResult result;
    std::size_t sourceIndex = 0;
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
          packets_("storm_transport_packets", 0),
          nextPackets_("storm_transport_next_packets", 0),
          results_("storm_transport_results", 0),
          sourceIndices_("storm_transport_source_indices", 0),
          nextSourceIndices_("storm_transport_next_source_indices", 0),
          completedTransports_("storm_transport_completed", 0),
          hostCompletedTransports_("storm_transport_host_completed", 0),
          terminalCount_("storm_transport_terminal_count"),
          cellSteps_("storm_transport_cell_steps", 0)
    {}

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
        if(!Kokkos::is_initialized())
        {
            throw std::runtime_error("Kokkos must be initialized before GPU transport");
        }

        const std::size_t count = std::min(
            std::max<std::size_t>(1, maximumParticles),
            particles.size());
        EnsureCapacity(this->hostPackets_, count);
        EnsureCapacity(this->packets_, count);
        EnsureCapacity(this->nextPackets_, count);
        EnsureCapacity(this->results_, count);
        EnsureCapacity(this->sourceIndices_, count);
        EnsureCapacity(this->nextSourceIndices_, count);
        EnsureCapacity(this->completedTransports_, count);
        EnsureCapacity(this->hostCompletedTransports_, count);
        EnsureCapacity(this->cellSteps_, views.grid.cellCount);
        Kokkos::deep_copy(this->cellSteps_.d_view, std::size_t(0));
        this->cellSteps_.modify_device();

        CompletedBatch completed;
        const std::chrono::steady_clock::time_point packStart = std::chrono::steady_clock::now();
        for(std::size_t i = 0; i < count; ++i)
        {
            const std::size_t sourceIndex = particles.size() - 1 - i;
            this->hostPackets_(i) = PackParticle(particles[sourceIndex]);
        }
        Kokkos::deep_copy(
            Kokkos::subview(
                this->packets_,
                std::pair<std::size_t, std::size_t>(0, count)),
            Kokkos::subview(
                this->hostPackets_,
                std::pair<std::size_t, std::size_t>(0, count)));
        std::size_t *sourceIndicesData = this->sourceIndices_.data();
        const std::size_t sourceSize = particles.size();
        Kokkos::parallel_for(
            "storm_transport_initialize_source_indices",
            Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(const std::size_t i)
            {
                sourceIndicesData[i] = sourceSize - 1 - i;
            });
        completed.packSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();

        const std::size_t maximumInnerSteps = this->maximumInnerSteps_;
        std::size_t activeCount = count;
        std::size_t completedCount = 0;
        while(activeCount > 0)
        {
            auto packets = this->packets_;
            auto nextPackets = this->nextPackets_;
            auto results = this->results_;
            auto sourceIndices = this->sourceIndices_;
            auto nextSourceIndices = this->nextSourceIndices_;
            auto completedTransports = this->completedTransports_;
            typename Kokkos::DualView<std::size_t*>::t_dev cellSteps = this->cellSteps_.d_view;

            const std::chrono::steady_clock::time_point deviceStart = std::chrono::steady_clock::now();
            Kokkos::parallel_for(
                "storm_grey_imc_transport",
                Kokkos::RangePolicy<>(0, activeCount),
                KOKKOS_LAMBDA(const std::size_t i)
                {
                    DeviceParticle particle = packets(i);
                    TransportResult result;
                    for(std::size_t step = 0; step < maximumInnerSteps; ++step)
                    {
                        ++particle.steps;
                        Kokkos::atomic_add(&cellSteps(particle.cellIndex), std::size_t(1));
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
                });

            const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
            progress();
            completed.progressSeconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - progressStart).count();

            const std::size_t completedOffset = completedCount;
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
                            transport.result = result;
                            transport.sourceIndex = sourceIndices(i);
                            completedTransports(completedOffset + terminalIndex) = transport;
                        }
                    }
                    else if(final)
                    {
                        const std::size_t survivorIndex = i - terminalPrefix;
                        nextPackets(survivorIndex) = packets(i);
                        nextSourceIndices(survivorIndex) = sourceIndices(i);
                    }
                },
                this->terminalCount_);

            std::size_t terminalCount = 0;
            Kokkos::deep_copy(terminalCount, this->terminalCount_);
            completed.deviceSeconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - deviceStart).count();
            ++completed.launchCount;

            activeCount -= terminalCount;
            completedCount += terminalCount;
            std::swap(this->packets_, this->nextPackets_);
            std::swap(this->sourceIndices_, this->nextSourceIndices_);
        }

        const std::chrono::steady_clock::time_point copyBackStart = std::chrono::steady_clock::now();
        completed.particles.resize(completedCount);
        if(completedCount > 0)
        {
            Kokkos::deep_copy(
                Kokkos::subview(this->hostCompletedTransports_,
                                std::pair<std::size_t, std::size_t>(0, completedCount)),
                Kokkos::subview(this->completedTransports_,
                                std::pair<std::size_t, std::size_t>(0, completedCount)));
            std::memcpy(
                completed.particles.data(),
                this->hostCompletedTransports_.data(),
                completedCount * sizeof(CompletedTransport));
        }
        completed.copyBackSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - copyBackStart).count();

        this->cellSteps_.sync_host();
        completed.cellSteps.resize(views.grid.cellCount);
        for(std::size_t i = 0; i < views.grid.cellCount; ++i)
        {
            completed.cellSteps[i] = this->cellSteps_.h_view(i);
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
    Kokkos::View<DeviceParticle*, PinnedHostSpace> hostPackets_;
    Kokkos::View<DeviceParticle*> packets_;
    Kokkos::View<DeviceParticle*> nextPackets_;
    Kokkos::View<TransportResult*> results_;
    Kokkos::View<std::size_t*> sourceIndices_;
    Kokkos::View<std::size_t*> nextSourceIndices_;
    Kokkos::View<CompletedTransport*> completedTransports_;
    Kokkos::View<CompletedTransport*, PinnedHostSpace> hostCompletedTransports_;
    Kokkos::View<std::size_t> terminalCount_;
    Kokkos::DualView<std::size_t*> cellSteps_;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_LOCAL_TRANSPORT_EXECUTOR_HPP
