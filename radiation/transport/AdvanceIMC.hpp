#ifndef STORM_RADIATION_TRANSPORT_ADVANCE_IMC_HPP
#define STORM_RADIATION_TRANSPORT_ADVANCE_IMC_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "TransportPortability.hpp"
#include "../ddmc/AdvanceDDMC.hpp"
#include "../../boundary/BoundaryCondition.hpp"
#include "../../gpu/FlatGridView.hpp"
#include "../../particle/StepResult.hpp"
#include "../../utils/CounterRNG.hpp"
#include "../ddmc/DDMCSampling.hpp"

namespace STORM
{
namespace transport
{

enum class TransportError : std::uint8_t
{
    None,
    InvalidCell,
    NoIntersection,
    InvalidOpacity,
    InvalidDoppler,
    HostFallback
};

struct TransportResult
{
    StepResult step;
    TransportError error = TransportError::None;
    std::size_t directedFace = std::numeric_limits<std::size_t>::max();
    ddmc::HostFallbackReason hostFallbackReason =
        ddmc::HostFallbackReason::None;
    std::size_t pendingLeakFace = std::numeric_limits<std::size_t>::max();
    std::size_t ddmcExtraSplits = 0;
};

struct IMCOpacityState
{
    double absorption = 0.0;
    double scattering = 0.0;
    double fleck = 1.0;
    std::size_t group = 0;
};

struct GreyOpacityPolicy
{
    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    IMCOpacityState Evaluate(const ParticleT &,
                             const ViewsT &views,
                             const std::size_t cellIndex,
                             const double) const
    {
        IMCOpacityState result;
        result.absorption =
            views.absorptionOpacities[cellIndex];
        result.scattering =
            views.scatteringOpacities[cellIndex];
        result.fleck = views.fleckFactors[cellIndex];
        return result;
    }

    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    void Scatter(ParticleT &particle,
                 const ViewsT &views,
                 const std::size_t,
                 const IMCOpacityState &,
                 const bool,
                 const double) const
    {
        const double random1 = CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
        const double random2 = CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
        const double mu = 1.0 - 2.0 * random1;
        const double phi =
            6.28318530717958647692 * random2;
        const double radial = 1.0 - mu * mu;
        const double sinTheta =
            Sqrt(radial > 0.0 ? radial : 0.0);
        particle.velocity.x =
            sinTheta * Cos(phi) * views.speedOfLight;
        particle.velocity.y =
            sinTheta * Sin(phi) * views.speedOfLight;
        particle.velocity.z = mu * views.speedOfLight;
    }

    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    void TallyGroupRadiation(const ParticleT &,
                             const ViewsT &,
                             const std::size_t,
                             const IMCOpacityState &,
                             const double) const
    {}
};

struct SpectralTableOpacityPolicy
{
    template<typename ViewsT>
    STORM_TRANSPORT_INLINE
    std::size_t FindGroup(const ViewsT &views, const double frequency) const
    {
        const std::size_t groupCount = views.groupCount;
        if(groupCount == 0 || views.energyBoundaries == nullptr)
        {
            return 0;
        }
        for(std::size_t group = 0; group + 1 < groupCount; ++group)
        {
            if(frequency < views.energyBoundaries[group + 1])
            {
                return group;
            }
        }
        return groupCount - 1;
    }

    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    IMCOpacityState Evaluate(const ParticleT &,
                             const ViewsT &views,
                             const std::size_t cellIndex,
                             const double transportFrequency) const
    {
        IMCOpacityState result;
        result.group = this->FindGroup(views, transportFrequency);
        double energy = transportFrequency;
        if(views.energyBoundaries != nullptr)
        {
            energy = transportFrequency > views.energyBoundaries[0]
                ? transportFrequency
                : views.energyBoundaries[0];
        }
        const double energyCubed = energy * energy * energy;
        result.absorption =
            (views.spectralAbsorptionScale != nullptr &&
             energyCubed > 0.0)
                ? views.spectralAbsorptionScale[cellIndex] /
                    energyCubed
                : views.absorptionOpacities[cellIndex];
        result.scattering =
            views.scatteringOpacities[cellIndex];
        result.fleck = views.fleckFactors[cellIndex];
        return result;
    }

    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    void Scatter(ParticleT &particle,
                 const ViewsT &views,
                 const std::size_t cellIndex,
                 const IMCOpacityState &,
                 const bool effectiveScatter,
                 const double dopplerShift) const
    {
        const double random1 = CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
        const double random2 = CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
        const double mu = 1.0 - 2.0 * random1;
        const double phi =
            6.28318530717958647692 * random2;
        const double radial = 1.0 - mu * mu;
        const double sinTheta =
            Sqrt(radial > 0.0 ? radial : 0.0);
        particle.velocity.x =
            sinTheta * Cos(phi) * views.speedOfLight;
        particle.velocity.y =
            sinTheta * Sin(phi) * views.speedOfLight;
        particle.velocity.z = mu * views.speedOfLight;

        particle.frequency *= dopplerShift;
        if(!effectiveScatter ||
           views.thermalEmissionCdf == nullptr ||
           views.energyBoundaries == nullptr ||
           views.groupCount == 0)
        {
            return;
        }

        const double reemitRandom = CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
        particle.frequency = ddmc::SampleFrequencyFromCellCdf(
            views.energyBoundaries,
            views.thermalEmissionCdf,
            views.groupCount,
            cellIndex,
            reemitRandom);
    }

    template<typename ParticleT, typename ViewsT>
    STORM_TRANSPORT_INLINE
    void TallyGroupRadiation(
        const ParticleT &,
        const ViewsT &views,
        const std::size_t cellIndex,
        const IMCOpacityState &opacityState,
        const double integratedEnergy) const
    {
        if(views.pendingGroupRadiationEnergy == nullptr ||
           opacityState.group >= views.groupCount)
        {
            return;
        }
        STORM_TRANSPORT_ACCUMULATE(
            views.pendingGroupRadiationEnergy[
                cellIndex * views.groupCount +
                opacityState.group],
            integratedEnergy);
    }
};

template<typename PointT>
STORM_TRANSPORT_INLINE
double Dot(const PointT &left, const PointT &right)
{
    return left.x * right.x +
           left.y * right.y +
           left.z * right.z;
}

template<typename ViewsT>
STORM_TRANSPORT_INLINE
void AddMomentum(const ViewsT &views,
                 const std::size_t cellIndex,
                 const double x,
                 const double y,
                 const double z)
{
    if(!views.depositMomentum)
    {
        return;
    }
    STORM_TRANSPORT_ACCUMULATE(
        views.pendingMomentum[cellIndex].x, x);
    STORM_TRANSPORT_ACCUMULATE(
        views.pendingMomentum[cellIndex].y, y);
    STORM_TRANSPORT_ACCUMULATE(
        views.pendingMomentum[cellIndex].z, z);
}

template<typename ParticleT, typename PointT>
STORM_TRANSPORT_INLINE
bool TransformToLab(ParticleT &particle,
                    const PointT &cellVelocity,
                    const double speedOfLight)
{
    const double velocitySquared =
        Dot(cellVelocity, cellVelocity);
    if(velocitySquared < 1.0e-30)
    {
        return true;
    }
    const double inverseC2 =
        1.0 / (speedOfLight * speedOfLight);
    const double gammaArgument =
        1.0 - velocitySquared * inverseC2;
    if(!(gammaArgument > 0.0) ||
       !IsFinite(gammaArgument))
    {
        return false;
    }
    const double gamma = 1.0 / Sqrt(gammaArgument);
    const double dopplerShift = gamma *
        (1.0 + Dot(cellVelocity, particle.velocity) *
                   inverseC2);
    if(!(dopplerShift > 0.0) ||
       !IsFinite(dopplerShift))
    {
        return false;
    }
    particle.frequency *= dopplerShift;
    particle.weight *= dopplerShift;

    const double velocityDotNegativeCell =
        -Dot(particle.velocity, cellVelocity);
    const double factor =
        (gamma - 1.0) * velocityDotNegativeCell /
            velocitySquared -
        gamma;
    particle.velocity.x -= cellVelocity.x * factor;
    particle.velocity.y -= cellVelocity.y * factor;
    particle.velocity.z -= cellVelocity.z * factor;

    const double newSpeed = Sqrt(
        Dot(particle.velocity, particle.velocity));
    if(newSpeed > 0.0)
    {
        const double scale = speedOfLight / newSpeed;
        particle.velocity.x *= scale;
        particle.velocity.y *= scale;
        particle.velocity.z *= scale;
    }
    return true;
}

// Shared IMC event kernel. Host code supplies ordinary pointers and device
// code supplies device pointers through the same small Views value. Opacity
// and frequency-changing scatter are isolated in a policy so spectral models
// can be added without duplicating tracking, event selection, or tallies.
template<typename ParticleT, typename ViewsT, typename OpacityPolicyT>
STORM_TRANSPORT_INLINE
TransportResult AdvanceIMC(ParticleT &particle,
                           const ViewsT &views,
                           const OpacityPolicyT &opacity)
{
    TransportResult result;
    const std::size_t cellIndex =
        static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount)
    {
        result.error = TransportError::InvalidCell;
        return result;
    }

    const double speed = Sqrt(
        particle.velocity.x * particle.velocity.x +
        particle.velocity.y * particle.velocity.y +
        particle.velocity.z * particle.velocity.z);
    const gpu::Intersection intersection =
        gpu::FindIntersection(particle, views.grid, speed);
    if(!intersection.valid)
    {
        result.error = TransportError::NoIntersection;
        return result;
    }

    double dopplerShift = 1.0;
    if(views.comovingTransport)
    {
        if(views.cellVelocities == nullptr)
        {
            result.error = TransportError::InvalidDoppler;
            return result;
        }
        const auto &cellVelocity =
            views.cellVelocities[cellIndex];
        const double velocitySquared =
            Dot(cellVelocity, cellVelocity);
        if(velocitySquared >= 1.0e-30)
        {
            const double inverseC2 =
                1.0 /
                (views.speedOfLight * views.speedOfLight);
            const double gammaArgument =
                1.0 - velocitySquared * inverseC2;
            if(!(gammaArgument > 0.0) ||
               !IsFinite(gammaArgument))
            {
                result.error =
                    TransportError::InvalidDoppler;
                return result;
            }
            const double gamma =
                1.0 / Sqrt(gammaArgument);
            dopplerShift = gamma *
                (1.0 -
                 Dot(cellVelocity, particle.velocity) *
                     inverseC2);
            if(!(dopplerShift > 0.0) ||
               !IsFinite(dopplerShift))
            {
                result.error =
                    TransportError::InvalidDoppler;
                return result;
            }
        }
    }

    const double transportFrequency =
        particle.frequency * dopplerShift;
    const IMCOpacityState opacityState =
        opacity.Evaluate(
            particle, views, cellIndex,
            transportFrequency);
    const double absorptionOpacity =
        opacityState.absorption;
    const double scatteringOpacity =
        opacityState.scattering;
    const double fleck = opacityState.fleck;
    if(!IsFinite(absorptionOpacity) ||
       !IsFinite(scatteringOpacity) ||
       absorptionOpacity < 0.0 ||
       scatteringOpacity < 0.0 ||
       fleck < 0.0 ||
       fleck > 1.0)
    {
        result.error = TransportError::InvalidOpacity;
        return result;
    }

    const double effectiveAbsorptionOpacity =
        (1.0 - fleck) * absorptionOpacity;
    const double eventOpacity =
        scatteringOpacity + effectiveAbsorptionOpacity;
    const double distanceRandom = CounterRNG::unitOpen(
        particle.rngKey, particle.rngCounter++);
    const double randomDistance =
        -Log1p(distanceRandom - 1.0);
    const double scatteringDistance = eventOpacity > 0.0
        ? randomDistance /
            (eventOpacity * dopplerShift)
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
    const double decayRate =
        absorptionOpacity * fleck * views.speedOfLight;
    const double materialExpFactor =
        Expm1(-dt * decayRate);
    const double weightExpFactor =
        Expm1(-dt * decayRate * dopplerShift);
    double integratedEnergy = particle.weight * dt;
    if(Abs(decayRate * dt) >= 1.0e-12)
    {
        integratedEnergy =
            particle.weight * materialExpFactor *
            (-1.0 / decayRate);
    }

    particle.location.x += particle.velocity.x * dt;
    particle.location.y += particle.velocity.y * dt;
    particle.location.z += particle.velocity.z * dt;

    if(views.depositMaterialEnergy)
    {
        STORM_TRANSPORT_ACCUMULATE(
            views.pendingMaterialEnergy[cellIndex],
            -materialExpFactor * particle.weight);
    }
    const double inverseC2 =
        1.0 / (views.speedOfLight * views.speedOfLight);
    AddMomentum(
        views, cellIndex,
        -weightExpFactor * particle.weight *
            particle.velocity.x * inverseC2,
        -weightExpFactor * particle.weight *
            particle.velocity.y * inverseC2,
        -weightExpFactor * particle.weight *
            particle.velocity.z * inverseC2);
    STORM_TRANSPORT_ACCUMULATE(
        views.pendingRadiationEnergy[cellIndex],
        integratedEnergy);
    opacity.TallyGroupRadiation(
        particle, views, cellIndex, opacityState,
        integratedEnergy);

    particle.weight *= 1.0 + weightExpFactor;
    if(Abs(particle.weight) < particle.initialWeight * 1.0e-3)
    {
        if(views.depositMaterialEnergy)
        {
            STORM_TRANSPORT_ACCUMULATE(
                views.pendingMaterialEnergy[cellIndex],
                particle.weight);
        }
        result.step.change = ParticleStatus::REMOVE;
        return result;
    }

    if(event == IntersectionEvent)
    {
        result.directedFace = intersection.directedFace;
        const bool deviceReflect =
            intersection.boundaryCrossing &&
            views.grid.deviceBoundaryBehaviors != nullptr &&
            views.grid.deviceBoundaryBehaviors[
                intersection.directedFace] ==
                static_cast<std::uint8_t>(
                    DeviceBoundaryFaceBehavior::ReflectingRigid);
        if(deviceReflect)
        {
            const auto &normal =
                views.grid.normals[intersection.directedFace];
            const double normalVelocity =
                particle.velocity.x * normal.x +
                particle.velocity.y * normal.y +
                particle.velocity.z * normal.z;
            particle.velocity.x -=
                2.0 * normalVelocity * normal.x;
            particle.velocity.y -=
                2.0 * normalVelocity * normal.y;
            particle.velocity.z -=
                2.0 * normalVelocity * normal.z;

            // Preserve the established host CrookedPipe semantics: its
            // boundary applies one centre nudge and the manager applies a
            // second after REFLECT.
            constexpr double epsilon = 1.0e-8;
            const auto &center = views.grid.cellCenters[cellIndex];
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
        result.step.boundaryCrossing =
            intersection.boundaryCrossing;
    }
    else if(event == ScatteringEvent)
    {
        const double oldVelocityX = particle.velocity.x;
        const double oldVelocityY = particle.velocity.y;
        const double oldVelocityZ = particle.velocity.z;
        const double eventRandom =
            CounterRNG::unitOpen(
                particle.rngKey, particle.rngCounter++) *
            eventOpacity;
        const bool effectiveScatter =
            eventRandom >= scatteringOpacity;
        opacity.Scatter(
            particle, views, cellIndex, opacityState,
            effectiveScatter, dopplerShift);
        if(views.comovingTransport)
        {
            const double weightBeforeTransform =
                particle.weight;
            particle.weight *= dopplerShift;
            if(!TransformToLab(
                   particle,
                   views.cellVelocities[cellIndex],
                   views.speedOfLight))
            {
                result.error =
                    TransportError::InvalidDoppler;
                return result;
            }
            AddMomentum(
                views, cellIndex,
                (weightBeforeTransform * oldVelocityX -
                 particle.weight * particle.velocity.x) *
                    inverseC2,
                (weightBeforeTransform * oldVelocityY -
                 particle.weight * particle.velocity.y) *
                    inverseC2,
                (weightBeforeTransform * oldVelocityZ -
                 particle.weight * particle.velocity.z) *
                    inverseC2);
        }
        result.step.change = ParticleStatus::NO_CELL_MOVE;
    }
    else
    {
        result.step.change = ParticleStatus::DONE;
    }

    return result;
}

template<typename ParticleT, typename ViewsT>
STORM_TRANSPORT_INLINE
TransportResult AdvanceIMC(ParticleT &particle,
                           const ViewsT &views)
{
    if(views.spectralEnabled)
    {
        return AdvanceIMC(
            particle, views, SpectralTableOpacityPolicy{});
    }
    return AdvanceIMC(
        particle, views, GreyOpacityPolicy{});
}

} // namespace transport
} // namespace STORM

#endif // STORM_RADIATION_TRANSPORT_ADVANCE_IMC_HPP
