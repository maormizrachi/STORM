#ifndef STORM_RADIATION_ADVANCE_RANDOM_WALK_HPP
#define STORM_RADIATION_ADVANCE_RANDOM_WALK_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "../RandomWalk.hpp"
#include "../transport/TransportPortability.hpp"
#include "../../particle/StepResult.hpp"
#include "../../utils/CounterRNG.hpp"

namespace STORM
{
namespace transport
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

struct RandomWalkView
{
    const std::uint8_t *cellEligible = nullptr;
    const double *cellTotalOpacity = nullptr;
    const PGRWCellData *pgrwCells = nullptr;
    RandomWalkTableView tables;
    std::size_t *stepCounter = nullptr;
    double minimumParticleOpticalDepth = 0.0;
    std::uint8_t enabled = 0;
    std::uint8_t spectralEnabled = 0;
};

struct RandomWalkResult
{
    StepResult step;
    std::uint8_t taken = 0;
    std::uint8_t invalid = 0;
};

STORM_TRANSPORT_INLINE
double SampleRandomWalkLeakTime(const RandomWalkTableView &tables, const double xi)
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
    const double fraction = (survival0 - target) / (survival0 - survival1);
    return Exp(Log(tau0) + fraction * (Log(tau1) - Log(tau0)));
}

STORM_TRANSPORT_INLINE
double SampleRandomWalkRadius(const RandomWalkTableView &tables, const double tau, const double xi)
{
    const double clampedTau = (tau < tables.tauMin)? tables.tauMin : ((tau > tables.tauMax)? tables.tauMax : tau);
    const double logMin = Log(tables.tauMin);
    const double logMax = Log(tables.tauMax);
    const double tauPosition = (Log(clampedTau) - logMin) / (logMax - logMin) * static_cast<double>(tables.tableSize - 1);
    std::size_t tauIndex = static_cast<std::size_t>(tauPosition);
    if(tauIndex > tables.tableSize - 2)
    {
        tauIndex = tables.tableSize - 2;
    }
    const double tauFraction = tauPosition - static_cast<double>(tauIndex);

    const double clampedXi = xi < 0.0 ? 0.0 : (xi > 1.0 ? 1.0 : xi);
    const double xiPosition = clampedXi * static_cast<double>(tables.radiusTableSize - 1);
    std::size_t xiIndex = static_cast<std::size_t>(xiPosition);
    if(xiIndex > tables.radiusTableSize - 2)
    {
        xiIndex = tables.radiusTableSize - 2;
    }
    const double xiFraction = xiPosition - static_cast<double>(xiIndex);

    const double radius00 = tables.radius[tauIndex * tables.radiusTableSize + xiIndex];
    const double radius01 = tables.radius[tauIndex * tables.radiusTableSize + xiIndex + 1];
    const double radius10 = tables.radius[(tauIndex + 1) * tables.radiusTableSize + xiIndex];
    const double radius11 = tables.radius[(tauIndex + 1) * tables.radiusTableSize + xiIndex + 1];
    const double radius0 = radius00 + xiFraction * (radius01 - radius00);
    const double radius1 = radius10 + xiFraction * (radius11 - radius10);
    const double radius = radius0 + tauFraction * (radius1 - radius0);
    return radius < 0.0 ? 0.0 : (radius > 1.0 ? 1.0 : radius);
}

template<typename ParticleT, typename ViewsT>
STORM_TRANSPORT_INLINE
RandomWalkResult TryAdvanceRandomWalk(ParticleT &particle, const ViewsT &views)
{
    RandomWalkResult result;
    const RandomWalkView &randomWalk = views.randomWalk;
    if(not randomWalk.enabled)
    {
        return result;
    }

    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount || randomWalk.cellEligible == nullptr || randomWalk.cellTotalOpacity == nullptr ||
       randomWalk.tables.tau == nullptr || randomWalk.tables.survival == nullptr || randomWalk.tables.radius == nullptr ||
       randomWalk.tables.tableSize < 2 || randomWalk.tables.radiusTableSize < 2 ||
       (randomWalk.spectralEnabled && (randomWalk.pgrwCells == nullptr || views.energyBoundaries == nullptr ||
                                       views.thermalEmissionCdf == nullptr || views.groupCount == 0)))
    {
        result.invalid = 1;
        return result;
    }
    if(not randomWalk.cellEligible[cellIndex])
    {
        return result;
    }

    double radius = DBL_MAX;
    const std::size_t faceBegin = views.grid.cellFaceOffsets[cellIndex];
    const std::size_t faceEnd = views.grid.cellFaceOffsets[cellIndex + 1];
    for(std::size_t face = faceBegin; face < faceEnd; ++face)
    {
        const auto &normal = views.grid.normals[face];
        const double distance = particle.location.x * normal.x + particle.location.y * normal.y + particle.location.z * normal.z - views.grid.facePlaneOffsets[face];
        if(distance < radius)
        {
            radius = distance;
        }
    }
    if(not (radius > 0.0))
    {
        return result;
    }

    double totalOpacity = randomWalk.cellTotalOpacity[cellIndex];
    double absorptionOpacity = views.absorptionOpacities[cellIndex];
    double diffusionCoefficient = totalOpacity > 0.0 ? views.speedOfLight / (3.0 * totalOpacity) : 0.0;
    double gamma = 1.0;
    std::size_t groupCutoff = 0;
    if(randomWalk.spectralEnabled)
    {
        const PGRWCellData &cell = randomWalk.pgrwCells[cellIndex];
        totalOpacity = cell.sigmaT_bar;
        absorptionOpacity = cell.sigmaA_bar;
        diffusionCoefficient = cell.D;
        gamma = cell.gamma;
        groupCutoff = cell.groupCutoff;
        if(groupCutoff == 0 or groupCutoff > views.groupCount or particle.frequency >= views.energyBoundaries[groupCutoff])
        {
            return result;
        }
    }
    if(not (totalOpacity > 0.0) or not (diffusionCoefficient > 0.0) or radius * totalOpacity < randomWalk.minimumParticleOpticalDepth)
    {
        return result;
    }

    result.taken = 1;
    if(randomWalk.stepCounter != nullptr)
    {
        STORM_TRANSPORT_ACCUMULATE(*randomWalk.stepCounter, std::size_t(1));
    }

    const double tauLeak = SampleRandomWalkLeakTime(randomWalk.tables, CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
    const double leakTime = tauLeak * radius * radius / diffusionCoefficient;
    const double fleck = views.fleckFactors[cellIndex];
    double upscatterTime = DBL_MAX;
    if(randomWalk.spectralEnabled && gamma < 1.0 && absorptionOpacity > 0.0 && fleck > 0.0)
    {
        const double upscatterRate = views.speedOfLight * (1.0 - fleck) * absorptionOpacity * (1.0 - gamma);
        if(upscatterRate > 0.0)
        {
            upscatterTime = -Log(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++)) / upscatterRate;
        }
    }
    const bool leak = leakTime <= particle.timeLeft && leakTime <= upscatterTime;
    const bool census = particle.timeLeft <= upscatterTime && !leak;
    const double dt = leak ? leakTime : (census ? particle.timeLeft : upscatterTime);

    const double absorptionRate = absorptionOpacity * fleck * views.speedOfLight;
    const double expFactor = Expm1(-dt * absorptionRate);
    if(views.depositMaterialEnergy)
    {
        STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], -expFactor * particle.weight);
    }
    if(absorptionRate > 0.0)
    {
        const double integratedEnergy = particle.weight * expFactor * (-1.0 / absorptionRate);
        STORM_TRANSPORT_ACCUMULATE(views.pendingRadiationEnergy[cellIndex], integratedEnergy);
        if(randomWalk.spectralEnabled && views.pendingGroupRadiationEnergy != nullptr)
        {
            std::size_t group = 0;
            while(group + 1 < views.groupCount && particle.frequency >= views.energyBoundaries[group + 1])
            {
                ++group;
            }
            STORM_TRANSPORT_ACCUMULATE(views.pendingGroupRadiationEnergy[cellIndex * views.groupCount + group], integratedEnergy);
        }
    }
    particle.weight *= 1.0 + expFactor;
    particle.timeLeft -= dt;

    if(Abs(particle.weight) < particle.initialWeight * 1.0e-4)
    {
        if(views.depositMaterialEnergy)
        {
            STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], particle.weight);
        }
        result.step.change = ParticleStatus::REMOVE;
        return result;
    }

    constexpr double pi = 3.14159265358979323846;
    const double cosine = 2.0 * CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) - 1.0;
    const double sine = Sqrt(cosine * cosine < 1.0 ? 1.0 - cosine * cosine : 0.0);
    const double phi = 2.0 * pi * CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double directionX = sine * Cos(phi);
    const double directionY = sine * Sin(phi);
    const double directionZ = cosine;

    double displacement = radius;
    if(!leak)
    {
        const double tauPosition = diffusionCoefficient * dt / (radius * radius);
        displacement = radius * SampleRandomWalkRadius(randomWalk.tables, tauPosition, CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
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
    const auto &cellCenter = views.grid.cellCenters[cellIndex];
    particle.location.x = particle.location.x * (1.0 - nudge) + nudge * cellCenter.x;
    particle.location.y = particle.location.y * (1.0 - nudge) + nudge * cellCenter.y;
    particle.location.z = particle.location.z * (1.0 - nudge) + nudge * cellCenter.z;

    bool contained = false;
    std::size_t containmentCorrections = 0;
    while(!contained && containmentCorrections < 4096)
    {
        contained = true;
        for(std::size_t face = faceBegin; face < faceEnd; ++face)
        {
            const auto &normal = views.grid.normals[face];
            const double distance = particle.location.x * normal.x + particle.location.y * normal.y + particle.location.z * normal.z - views.grid.facePlaneOffsets[face];
            if(distance < 0.0)
            {
                displacement *= 0.99;
                particle.location.x = centerX + displacement * directionX;
                particle.location.y = centerY + displacement * directionY;
                particle.location.z = centerZ + displacement * directionZ;
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

    const double directionRandom1 = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double directionRandom2 = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double velocityCosine = 1.0 - 2.0 * directionRandom1;
    const double velocityPhi = 2.0 * pi * directionRandom2;
    const double velocitySine = Sqrt(velocityCosine * velocityCosine < 1.0 ? 1.0 - velocityCosine * velocityCosine : 0.0);
    particle.velocity.x = velocitySine * Cos(velocityPhi) * views.speedOfLight;
    particle.velocity.y = velocitySine * Sin(velocityPhi) * views.speedOfLight;
    particle.velocity.z = velocityCosine * views.speedOfLight;

    if(!leak && !census && randomWalk.spectralEnabled)
    {
        const double *cdf = views.thermalEmissionCdf + cellIndex * (views.groupCount + 1);
        const double lower = cdf[groupCutoff];
        const double total = cdf[views.groupCount];
        if(total > lower && IsFinite(total))
        {
            const double target = lower + CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) * (total - lower);
            std::size_t group = groupCutoff;
            while(group + 1 < views.groupCount and cdf[group + 1] < target)
            {
                ++group;
            }
            const double cdfLower = cdf[group];
            const double cdfUpper = cdf[group + 1];
            const double width = cdfUpper - cdfLower;
            const double fraction = width > 0.0 ? (target - cdfLower) / width : 0.5;
            particle.frequency = views.energyBoundaries[group] + fraction * (views.energyBoundaries[group + 1] - views.energyBoundaries[group]);
        }
        else
        {
            particle.frequency = views.energyBoundaries[groupCutoff];
        }
    }

    result.step.change = census ? ParticleStatus::DONE : ParticleStatus::NO_CELL_MOVE;
    return result;
}

} // namespace transport
} // namespace STORM

#endif // STORM_RADIATION_ADVANCE_RANDOM_WALK_HPP
