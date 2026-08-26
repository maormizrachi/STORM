#ifndef STORM_GPU_GREY_IMC_KERNEL_HPP
#define STORM_GPU_GREY_IMC_KERNEL_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "FlatGridView.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "../particle/StepResult.hpp"
#include "../utils/CounterRNG.hpp"

namespace STORM
{
namespace gpu
{

template<typename PointT>
struct GreyIMCViews
{
    FlatGridView<PointT> grid;
    const double *absorptionOpacities = nullptr;
    const double *scatteringOpacities = nullptr;
    const double *fleckFactors = nullptr;
    double *pendingMaterialEnergy = nullptr;
    double *pendingRadiationEnergy = nullptr;
    double speedOfLight = 0.0;
    std::uint8_t depositMaterialEnergy = 1;
};

enum class TransportError : std::uint8_t
{
    None,
    InvalidCell,
    NoIntersection,
    InvalidOpacity
};

struct TransportResult
{
    StepResult step;
    TransportError error = TransportError::None;
};

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
TransportResult AdvanceOne(ParticleT &particle, const GreyIMCViews<PointT> &views)
{
    TransportResult result;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount)
    {
        result.error = TransportError::InvalidCell;
        return result;
    }

    const double speed = Kokkos::sqrt(particle.velocity.x * particle.velocity.x +
                                      particle.velocity.y * particle.velocity.y +
                                      particle.velocity.z * particle.velocity.z);
    const Intersection intersection = FindIntersection(particle, views.grid, speed);
    if(!intersection.valid)
    {
        result.error = TransportError::NoIntersection;
        return result;
    }

    const double absorptionOpacity = views.absorptionOpacities[cellIndex];
    const double scatteringOpacity = views.scatteringOpacities[cellIndex];
    const double fleck = views.fleckFactors[cellIndex];
    if(!Kokkos::isfinite(absorptionOpacity) ||
       !Kokkos::isfinite(scatteringOpacity) ||
       absorptionOpacity < 0.0 ||
       scatteringOpacity < 0.0 ||
       fleck < 0.0 ||
       fleck > 1.0)
    {
        result.error = TransportError::InvalidOpacity;
        return result;
    }

    const double effectiveAbsorptionOpacity = (1.0 - fleck) * absorptionOpacity;
    const double eventOpacity = scatteringOpacity + effectiveAbsorptionOpacity;
    const double randomDistance = -Kokkos::log(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
    const double scatteringDistance = eventOpacity > 0.0
        ? randomDistance / eventOpacity
        : DBL_MAX;
    const double scatteringTime = speed > 0.0
        ? scatteringDistance / speed
        : DBL_MAX;

    enum Event : std::uint8_t
    {
        IntersectionEvent,
        ScatteringEvent,
        CensusEvent
    };

    Event event = IntersectionEvent;
    double dt = intersection.time;
    if(scatteringTime < dt)
    {
        event = ScatteringEvent;
        dt = scatteringTime;
    }
    if(particle.timeLeft < dt)
    {
        event = CensusEvent;
        dt = particle.timeLeft;
    }

    particle.timeLeft -= dt;
    const double decayRate = absorptionOpacity * fleck * views.speedOfLight;
    const double exponent = -dt * decayRate;
    const double expFactor = Kokkos::exp(exponent) - 1.0;
    double integratedEnergy = particle.weight * dt;
    if(Kokkos::abs(decayRate * dt) >= 1.0e-12)
    {
        integratedEnergy = particle.weight * expFactor * (-1.0 / decayRate);
    }

    particle.location.x += particle.velocity.x * dt;
    particle.location.y += particle.velocity.y * dt;
    particle.location.z += particle.velocity.z * dt;

    if(views.depositMaterialEnergy)
    {
        Kokkos::atomic_add(&views.pendingMaterialEnergy[cellIndex], -expFactor * particle.weight);
    }
    Kokkos::atomic_add(&views.pendingRadiationEnergy[cellIndex], integratedEnergy);

    particle.weight *= 1.0 + expFactor;
    if(Kokkos::abs(particle.weight) < particle.initialWeight * 1.0e-3)
    {
        if(views.depositMaterialEnergy)
        {
            Kokkos::atomic_add(&views.pendingMaterialEnergy[cellIndex], particle.weight);
        }
        result.step.change = ParticleStatus::REMOVE;
        return result;
    }

    if(event == IntersectionEvent)
    {
        const bool deviceReflect =
            intersection.boundaryCrossing &&
            views.grid.deviceBoundaryBehaviors != nullptr &&
            views.grid.deviceBoundaryBehaviors[intersection.directedFace] ==
                static_cast<std::uint8_t>(
                    DeviceBoundaryFaceBehavior::ReflectingRigid);
        if(deviceReflect)
        {
            const PointT &normal =
                views.grid.normals[intersection.directedFace];
            const double normalVelocity =
                particle.velocity.x * normal.x +
                particle.velocity.y * normal.y +
                particle.velocity.z * normal.z;
            particle.velocity.x -= 2.0 * normalVelocity * normal.x;
            particle.velocity.y -= 2.0 * normalVelocity * normal.y;
            particle.velocity.z -= 2.0 * normalVelocity * normal.z;

            // Preserve the established host CrookedPipe semantics: its
            // boundary applies one centre nudge and the manager applies a
            // second after REFLECT.
            constexpr double epsilon = 1.0e-8;
            const PointT &center = views.grid.cellCenters[cellIndex];
            for(std::uint8_t nudge = 0; nudge < 2; ++nudge)
            {
                particle.location.x =
                    (1.0 - epsilon) * particle.location.x +
                    epsilon * center.x;
                particle.location.y =
                    (1.0 - epsilon) * particle.location.y +
                    epsilon * center.y;
                particle.location.z =
                    (1.0 - epsilon) * particle.location.z +
                    epsilon * center.z;
            }
            result.step.change = ParticleStatus::NO_CELL_MOVE;
            return result;
        }
        result.step.change = ParticleStatus::CELL_MOVE;
        result.step.nextCellIndex = intersection.nextCellIndex;
        result.step.boundaryCrossing = intersection.boundaryCrossing;
    }
    else if(event == ScatteringEvent)
    {
        // Preserve the host RNG sequence: event selection consumes one
        // random number even though both grey event types are isotropic.
        (void) CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
        const double random1 = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
        const double random2 = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
        const double mu = 1.0 - 2.0 * random1;
        const double phi = 6.28318530717958647692 * random2;
        const double radial = 1.0 - mu * mu;
        const double sinTheta = Kokkos::sqrt(radial > 0.0 ? radial : 0.0);
        particle.velocity.x = sinTheta * Kokkos::cos(phi) * views.speedOfLight;
        particle.velocity.y = sinTheta * Kokkos::sin(phi) * views.speedOfLight;
        particle.velocity.z = mu * views.speedOfLight;
        result.step.change = ParticleStatus::NO_CELL_MOVE;
    }
    else
    {
        result.step.change = ParticleStatus::DONE;
    }

    return result;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_IMC_KERNEL_HPP
