#ifndef STORM_GPU_GREY_RANDOM_WALK_KERNEL_HPP
#define STORM_GPU_GREY_RANDOM_WALK_KERNEL_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "FlatGridView.hpp"
#include "../particle/StepResult.hpp"
#include "../utils/CounterRNG.hpp"

namespace STORM
{
namespace gpu
{

struct RandomWalkTableView
{
    const double *tau = nullptr;
    const double *survival = nullptr;
    const double *radius = nullptr;
    std::size_t tableSize = 0;
    std::size_t radiusTableSize = 0;
    double tauMin = 0.0;
    double tauMax = 0.0;
};

struct GreyRandomWalkView
{
    const std::uint8_t *cellEligible = nullptr;
    const double *cellTotalOpacity = nullptr;
    RandomWalkTableView tables;
    std::size_t *stepCounter = nullptr;
    double minimumParticleOpticalDepth = 0.0;
    std::uint8_t enabled = 0;
};

struct GreyRandomWalkResult
{
    StepResult step;
    std::uint8_t taken = 0;
    std::uint8_t invalid = 0;
};

STORM_GPU_INLINE_FUNCTION
double SampleRandomWalkLeakTime(const RandomWalkTableView &tables,
                                const double xi)
{
    const double target = 1.0 - xi;
    if(target >= tables.survival[0])
    {
        return tables.tau[0];
    }
    if(target <= tables.survival[tables.tableSize - 1])
    {
        return tables.tau[tables.tableSize - 1];
    }

    std::size_t lower = 1;
    std::size_t upper = tables.tableSize - 1;
    while(lower < upper)
    {
        const std::size_t middle = lower + (upper - lower) / 2;
        if(tables.survival[middle] > target)
        {
            lower = middle + 1;
        }
        else
        {
            upper = middle;
        }
    }

    const std::size_t index = lower;
    const double survival0 = tables.survival[index - 1];
    const double survival1 = tables.survival[index];
    const double tau0 = tables.tau[index - 1];
    const double tau1 = tables.tau[index];
    const double fraction =
        (survival0 - target) / (survival0 - survival1);
    return Kokkos::exp(
        Kokkos::log(tau0) +
        fraction * (Kokkos::log(tau1) - Kokkos::log(tau0)));
}

STORM_GPU_INLINE_FUNCTION
double SampleRandomWalkRadius(const RandomWalkTableView &tables,
                              const double tau,
                              const double xi)
{
    const double clampedTau =
        tau < tables.tauMin ? tables.tauMin :
        (tau > tables.tauMax ? tables.tauMax : tau);
    const double logMin = Kokkos::log(tables.tauMin);
    const double logMax = Kokkos::log(tables.tauMax);
    const double tauPosition =
        (Kokkos::log(clampedTau) - logMin) / (logMax - logMin) *
        static_cast<double>(tables.tableSize - 1);
    std::size_t tauIndex = static_cast<std::size_t>(tauPosition);
    if(tauIndex > tables.tableSize - 2)
    {
        tauIndex = tables.tableSize - 2;
    }
    const double tauFraction =
        tauPosition - static_cast<double>(tauIndex);

    const double clampedXi = xi < 0.0 ? 0.0 : (xi > 1.0 ? 1.0 : xi);
    const double xiPosition =
        clampedXi * static_cast<double>(tables.radiusTableSize - 1);
    std::size_t xiIndex = static_cast<std::size_t>(xiPosition);
    if(xiIndex > tables.radiusTableSize - 2)
    {
        xiIndex = tables.radiusTableSize - 2;
    }
    const double xiFraction =
        xiPosition - static_cast<double>(xiIndex);

    const double radius00 =
        tables.radius[tauIndex * tables.radiusTableSize + xiIndex];
    const double radius01 =
        tables.radius[tauIndex * tables.radiusTableSize + xiIndex + 1];
    const double radius10 =
        tables.radius[(tauIndex + 1) * tables.radiusTableSize + xiIndex];
    const double radius11 =
        tables.radius[(tauIndex + 1) * tables.radiusTableSize + xiIndex + 1];
    const double radius0 =
        radius00 + xiFraction * (radius01 - radius00);
    const double radius1 =
        radius10 + xiFraction * (radius11 - radius10);
    const double radius =
        radius0 + tauFraction * (radius1 - radius0);
    return radius < 0.0 ? 0.0 : (radius > 1.0 ? 1.0 : radius);
}

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
GreyRandomWalkResult TryAdvanceGreyRandomWalk(
    ParticleT &particle,
    const FlatGridView<PointT> &grid,
    const GreyRandomWalkView &randomWalk,
    const double *absorptionOpacities,
    const double *fleckFactors,
    double *pendingMaterialEnergy,
    double *pendingRadiationEnergy,
    const double speedOfLight,
    const std::uint8_t depositMaterialEnergy)
{
    GreyRandomWalkResult result;
    if(!randomWalk.enabled)
    {
        return result;
    }

    const std::size_t cellIndex =
        static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= grid.cellCount ||
       randomWalk.cellEligible == nullptr ||
       randomWalk.cellTotalOpacity == nullptr ||
       randomWalk.tables.tau == nullptr ||
       randomWalk.tables.survival == nullptr ||
       randomWalk.tables.radius == nullptr ||
       randomWalk.tables.tableSize < 2 ||
       randomWalk.tables.radiusTableSize < 2)
    {
        result.invalid = 1;
        return result;
    }
    if(!randomWalk.cellEligible[cellIndex])
    {
        return result;
    }

    double radius = DBL_MAX;
    const std::size_t faceBegin = grid.cellFaceOffsets[cellIndex];
    const std::size_t faceEnd = grid.cellFaceOffsets[cellIndex + 1];
    for(std::size_t face = faceBegin; face < faceEnd; ++face)
    {
        const PointT &normal = grid.normals[face];
        const double distance =
            particle.location.x * normal.x +
            particle.location.y * normal.y +
            particle.location.z * normal.z -
            grid.facePlaneOffsets[face];
        if(distance < radius)
        {
            radius = distance;
        }
    }
    if(!(radius > 0.0))
    {
        return result;
    }

    const double totalOpacity = randomWalk.cellTotalOpacity[cellIndex];
    if(!(totalOpacity > 0.0) ||
       radius * totalOpacity <
           randomWalk.minimumParticleOpticalDepth)
    {
        return result;
    }

    const double diffusionCoefficient =
        speedOfLight / (3.0 * totalOpacity);
    if(!(diffusionCoefficient > 0.0))
    {
        return result;
    }

    result.taken = 1;
    if(randomWalk.stepCounter != nullptr)
    {
        Kokkos::atomic_add(randomWalk.stepCounter, std::size_t(1));
    }

    const double tauLeak = SampleRandomWalkLeakTime(
        randomWalk.tables,
        CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
    const double leakTime =
        tauLeak * radius * radius / diffusionCoefficient;
    const bool leak = leakTime <= particle.timeLeft;
    const double dt = leak ? leakTime : particle.timeLeft;

    const double absorptionOpacity = absorptionOpacities[cellIndex];
    const double fleck = fleckFactors[cellIndex];
    const double absorptionRate =
        absorptionOpacity * fleck * speedOfLight;
    const double expFactor =
        Kokkos::expm1(-dt * absorptionRate);
    if(depositMaterialEnergy)
    {
        Kokkos::atomic_add(
            &pendingMaterialEnergy[cellIndex],
            -expFactor * particle.weight);
    }
    if(absorptionRate > 0.0)
    {
        Kokkos::atomic_add(
            &pendingRadiationEnergy[cellIndex],
            particle.weight * expFactor * (-1.0 / absorptionRate));
    }
    particle.weight *= 1.0 + expFactor;
    particle.timeLeft -= dt;

    if(Kokkos::abs(particle.weight) <
       particle.initialWeight * 1.0e-4)
    {
        if(depositMaterialEnergy)
        {
            Kokkos::atomic_add(
                &pendingMaterialEnergy[cellIndex], particle.weight);
        }
        result.step.change = ParticleStatus::REMOVE;
        return result;
    }

    constexpr double pi = 3.14159265358979323846;
    const double cosine =
        2.0 * CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++) - 1.0;
    const double sine =
        Kokkos::sqrt(cosine * cosine < 1.0 ?
            1.0 - cosine * cosine : 0.0);
    const double phi =
        2.0 * pi * CounterRNG::unitOpen(
            particle.rngKey, particle.rngCounter++);
    const double directionX = sine * Kokkos::cos(phi);
    const double directionY = sine * Kokkos::sin(phi);
    const double directionZ = cosine;

    double displacement = radius;
    if(!leak)
    {
        const double tauPosition =
            diffusionCoefficient * dt / (radius * radius);
        displacement = radius * SampleRandomWalkRadius(
            randomWalk.tables, tauPosition,
            CounterRNG::unitOpen(
                particle.rngKey, particle.rngCounter++));
    }
    if(displacement > radius * (1.0 + 1.0e-12))
    {
        displacement = radius;
    }

    const double centerX = particle.location.x;
    const double centerY = particle.location.y;
    const double centerZ = particle.location.z;
    particle.location.x = centerX + displacement * directionX;
    particle.location.y = centerY + displacement * directionY;
    particle.location.z = centerZ + displacement * directionZ;

    constexpr double nudge = 1.0e-6;
    const PointT &cellCenter = grid.cellCenters[cellIndex];
    particle.location.x =
        particle.location.x * (1.0 - nudge) +
        nudge * cellCenter.x;
    particle.location.y =
        particle.location.y * (1.0 - nudge) +
        nudge * cellCenter.y;
    particle.location.z =
        particle.location.z * (1.0 - nudge) +
        nudge * cellCenter.z;

    bool contained = false;
    std::size_t containmentCorrections = 0;
    while(!contained && containmentCorrections < 4096)
    {
        contained = true;
        for(std::size_t face = faceBegin; face < faceEnd; ++face)
        {
            const PointT &normal = grid.normals[face];
            const double distance =
                particle.location.x * normal.x +
                particle.location.y * normal.y +
                particle.location.z * normal.z -
                grid.facePlaneOffsets[face];
            if(distance < 0.0)
            {
                displacement *= 0.99;
                particle.location.x =
                    centerX + displacement * directionX;
                particle.location.y =
                    centerY + displacement * directionY;
                particle.location.z =
                    centerZ + displacement * directionZ;
                contained = false;
                ++containmentCorrections;
                break;
            }
        }
    }
    if(!contained)
    {
        particle.location.x = centerX;
        particle.location.y = centerY;
        particle.location.z = centerZ;
    }

    const double directionRandom1 = CounterRNG::unitOpen(
        particle.rngKey, particle.rngCounter++);
    const double directionRandom2 = CounterRNG::unitOpen(
        particle.rngKey, particle.rngCounter++);
    const double velocityCosine = 1.0 - 2.0 * directionRandom1;
    const double velocityPhi = 2.0 * pi * directionRandom2;
    const double velocitySine =
        Kokkos::sqrt(velocityCosine * velocityCosine < 1.0 ?
            1.0 - velocityCosine * velocityCosine : 0.0);
    particle.velocity.x =
        velocitySine * Kokkos::cos(velocityPhi) * speedOfLight;
    particle.velocity.y =
        velocitySine * Kokkos::sin(velocityPhi) * speedOfLight;
    particle.velocity.z = velocityCosine * speedOfLight;

    result.step.change =
        leak ? ParticleStatus::NO_CELL_MOVE : ParticleStatus::DONE;
    return result;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_RANDOM_WALK_KERNEL_HPP
