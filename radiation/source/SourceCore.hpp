#ifndef STORM_RADIATION_SOURCE_CORE_HPP
#define STORM_RADIATION_SOURCE_CORE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../utils/CounterRNG.hpp"
#include "../../types.hpp"
#include "../transport/TransportPortability.hpp"

#ifdef STORM_WITH_GPU
#include "../../gpu/KokkosTypes.hpp"
#define STORM_SOURCE_INLINE STORM_GPU_INLINE_FUNCTION
#else
#define STORM_SOURCE_INLINE inline
#endif

namespace STORM
{
namespace source
{

struct Plan
{
    std::vector<std::size_t> nPhotons;
    std::vector<std::size_t> photonOffsets;
    std::vector<double> energyPerPhoton;
    std::vector<double> energyToCreate;
    std::vector<double> gamma;
    std::size_t totalPhotons = 0;
    std::uint64_t rngStreamBase = 0;
    double emittedEnergy = 0.0;
};

template<typename PointT>
struct SampleViews
{
    const std::size_t *tetOffsets = nullptr;
    const double *tetCumVolumes = nullptr;
    const std::uint32_t *tetTris = nullptr;
    const PointT *vertices = nullptr;
    const PointT *cellCenters = nullptr;
    const cell_id_t *cellIDs = nullptr;
    const double *thermalEmissionCdf = nullptr;
    const double *energyBoundaries = nullptr;
    const PointT *cellVelocities = nullptr;
    std::size_t cellCount = 0;
    std::size_t groupCount = 0;
    double speedOfLight = 0.0;
    double invClight2 = 0.0;
    std::uint8_t sampleFrequency = 0;
    std::uint8_t applyLabFrame = 0;
};

struct EmittedScalars
{
    double frequency = 0.0;
    double weight = 0.0;
    double initialWeight = 0.0;
    std::uint64_t rngKey = 0;
    std::uint64_t rngCounter = 0;
    cell_index_t cellIndex = 0;
    cell_id_t cellID = 0;
};

STORM_SOURCE_INLINE double CellEmissionEnergy(const double fleck, const double volume, const double temperature, const double planckOpacity,
                                                const double dt, const double arad, const double clight)
{
    const double t2 = temperature * temperature;
    const double t4 = t2 * t2;
    return fleck * volume * arad * t4 * planckOpacity * dt * clight;
}

STORM_SOURCE_INLINE std::size_t PhotonCount(const double cellEnergy, const double globalEnergy, const std::size_t totalParticles,
                                            const std::size_t nMin, const std::size_t nMax)
{
    if(not (cellEnergy > 0.0))
    {
        return 0;
    }
    const std::size_t proportional = (globalEnergy > 0.0)? static_cast<std::size_t>(cellEnergy / globalEnergy * static_cast<double>(totalParticles)) : nMin;
    const std::size_t lo = (nMin > proportional)? nMin : proportional;
    return (lo > nMax)? nMax : lo;
}

STORM_SOURCE_INLINE std::uint64_t MakeSourceRngKey(const std::uint64_t seed, const std::uint64_t rank, const std::uint64_t stream)
{
    return CounterRNG::makeKey(seed, rank, stream);
}

STORM_SOURCE_INLINE void SwapDoubles(double &left, double &right)
{
    const double tmp = left;
    left = right;
    right = tmp;
}

template<typename PointT>
STORM_SOURCE_INLINE PointT SampleTetrahedronPosition(const PointT &center, const PointT &a, const PointT &b, const PointT &c, double s, double t, double u)
{
    if(s > t)
    {
        SwapDoubles(s, t);
    }
    if(t > u)
    {
        SwapDoubles(t, u);
    }
    if(s > t)
    {
        SwapDoubles(s, t);
    }
    const double oneMinusU = 1.0 - u;
    return PointT(s * a.x + (t - s) * b.x + (u - t) * c.x + oneMinusU * center.x,
                    s * a.y + (t - s) * b.y + (u - t) * c.y + oneMinusU * center.y,
                    s * a.z + (t - s) * b.z + (u - t) * c.z + oneMinusU * center.z);
}

template<typename PointT>
STORM_SOURCE_INLINE PointT SamplePositionFromTetTables(const SampleViews<PointT> &views, const std::size_t cellIndex, const std::uint64_t rngKey, std::uint64_t &rngCounter)
{
    const PointT &center = views.cellCenters[cellIndex];
    if(views.tetOffsets == nullptr or views.tetCumVolumes == nullptr or views.tetTris == nullptr or views.vertices == nullptr)
    {
        return center;
    }
    const std::size_t begin = views.tetOffsets[cellIndex];
    const std::size_t end = views.tetOffsets[cellIndex + 1];
    if(begin >= end)
    {
        return center;
    }
    const double totalVolume = views.tetCumVolumes[end - 1];
    if(!(totalVolume > 0.0))
    {
        return center;
    }
    const double target = CounterRNG::unitOpen(rngKey, rngCounter++) * totalVolume;
    std::size_t tet = begin;
    while(tet + 1 < end && views.tetCumVolumes[tet] < target)
    {
        ++tet;
    }
    const std::size_t tri = tet * 3U;
    const PointT &a = views.vertices[views.tetTris[tri]];
    const PointT &b = views.vertices[views.tetTris[tri + 1]];
    const PointT &c = views.vertices[views.tetTris[tri + 2]];
    const double s = CounterRNG::unitOpen(rngKey, rngCounter++);
    const double t = CounterRNG::unitOpen(rngKey, rngCounter++);
    const double u = CounterRNG::unitOpen(rngKey, rngCounter++);
    return SampleTetrahedronPosition(center, a, b, c, s, t, u);
}

template<typename PointT>
STORM_SOURCE_INLINE PointT SampleIsotropicDirection(const std::uint64_t rngKey, std::uint64_t &rngCounter, const double speedOfLight)
{
    const double random1 = CounterRNG::unitOpen(rngKey, rngCounter++);
    const double random2 = CounterRNG::unitOpen(rngKey, rngCounter++);
    const double mu = 1.0 - 2.0 * random1;
    const double phi = 6.28318530717958647692 * random2;
    const double radial = 1.0 - mu * mu;
    const double sinTheta = transport::Sqrt(radial > 0.0 ? radial : 0.0);
    return PointT(sinTheta * transport::Cos(phi) * speedOfLight, sinTheta * transport::Sin(phi) * speedOfLight, mu * speedOfLight);
}

STORM_SOURCE_INLINE
double SampleFrequencyFromCdf(const double *boundaries, const double *cdf, const std::size_t groupCount, const std::size_t cellIndex, const double random)
{
    if(boundaries == nullptr or cdf == nullptr or groupCount == 0)
    {
        return 0.0;
    }
    const double *cellCdf = cdf + cellIndex * (groupCount + 1);
    const double total = cellCdf[groupCount];
    if(not (total > 0.0) or not transport::IsFinite(total))
    {
        return 0.5 * (boundaries[0] + boundaries[groupCount]);
    }
    const double target = random * total;
    std::size_t group = 0;
    while(group + 1 < groupCount and cellCdf[group + 1] < target)
    {
        ++group;
    }
    const double lower = cellCdf[group];
    const double upper = cellCdf[group + 1];
    const double width = upper - lower;
    const double fraction = (width > 0.0)? (target - lower) / width : 0.5;
    return boundaries[group] + fraction * (boundaries[group + 1] - boundaries[group]);
}

template<typename PointT>
STORM_SOURCE_INLINE double DopplerShift(const PointT &particleVelocity, const PointT &cellVelocity, const double invClight2)
{
    const double v2 = cellVelocity.x * cellVelocity.x + cellVelocity.y * cellVelocity.y + cellVelocity.z * cellVelocity.z;
    if(v2 < 1.0e-30)
    {
        return 1.0;
    }
    const double gamma = 1.0 / transport::Sqrt(1.0 - v2 * invClight2);
    const double vDotN = cellVelocity.x * particleVelocity.x + cellVelocity.y * particleVelocity.y + cellVelocity.z * particleVelocity.z;
    return gamma * (1.0 - vDotN * invClight2);
}

template<typename PointT>
STORM_SOURCE_INLINE void LorentzBoostToLab(PointT &velocity, double &frequency, double &weight, const PointT &cellVelocity, const double invClight2, const double speedOfLight)
{
    const double v2 = cellVelocity.x * cellVelocity.x + cellVelocity.y * cellVelocity.y + cellVelocity.z * cellVelocity.z;
    if(v2 < 1.0e-30)
    {
        return;
    }
    const double gamma = 1.0 / transport::Sqrt(1.0 - invClight2 * v2);
    const PointT negV(-cellVelocity.x, -cellVelocity.y, -cellVelocity.z);
    const double vDotP = velocity.x * negV.x + velocity.y * negV.y + velocity.z * negV.z;
    const double dopplerShift = gamma * (1.0 - vDotP * invClight2);
    frequency *= dopplerShift;
    weight *= dopplerShift;
    const double factor = (gamma - 1.0) * vDotP / v2 - gamma;
    velocity.x += negV.x * factor;
    velocity.y += negV.y * factor;
    velocity.z += negV.z * factor;
    const double newSpeed = transport::Sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    if(newSpeed > 0.0)
    {
        const double scale = speedOfLight / newSpeed;
        velocity.x *= scale;
        velocity.y *= scale;
        velocity.z *= scale;
    }
}

STORM_SOURCE_INLINE std::size_t CellFromPhotonSlot(const std::size_t *offsets, const std::size_t cellCount, const std::size_t slot)
{
    std::size_t lo = 0;
    std::size_t hi = cellCount;
    while(lo + 1 < hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;
        if(offsets[mid] <= slot)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

template<typename PointT>
STORM_SOURCE_INLINE void EmitThermalPacket(const SampleViews<PointT> &views, const std::size_t cellIndex, const std::uint64_t rngKey,
                                            const double energyPerPhoton, PointT &location, PointT &velocity, EmittedScalars &scalars)
{
    scalars.rngKey = rngKey;
    scalars.rngCounter = 0;
    scalars.cellIndex = static_cast<cell_index_t>(cellIndex);
    scalars.cellID = (views.cellIDs != nullptr)? views.cellIDs[cellIndex] : static_cast<cell_id_t>(cellIndex);
    scalars.frequency = 0.0;
    scalars.weight = energyPerPhoton;

    location = SamplePositionFromTetTables(views, cellIndex, rngKey, scalars.rngCounter);
    velocity = SampleIsotropicDirection<PointT>(rngKey, scalars.rngCounter, views.speedOfLight);

    if(views.applyLabFrame != 0 && views.cellVelocities != nullptr)
    {
        LorentzBoostToLab(velocity, scalars.frequency, scalars.weight, views.cellVelocities[cellIndex], views.invClight2, views.speedOfLight);
    }

    if(views.sampleFrequency != 0)
    {
        const double random = CounterRNG::unitOpen(rngKey, scalars.rngCounter++);
        const double freqCo = SampleFrequencyFromCdf(views.energyBoundaries, views.thermalEmissionCdf, views.groupCount, cellIndex, random);
        if(views.applyLabFrame != 0 and views.cellVelocities != nullptr)
        {
            const double doppler = DopplerShift(velocity, views.cellVelocities[cellIndex], views.invClight2);
            if(doppler > 0.0 and transport::IsFinite(doppler))
            {
                scalars.frequency = freqCo / doppler;
                scalars.weight = energyPerPhoton / doppler;
            }
            else
            {
                scalars.frequency = freqCo;
            }
        }
        else
        {
            scalars.frequency = freqCo;
        }
    }

    scalars.initialWeight = (scalars.weight < 0.0)? -scalars.weight : scalars.weight;
}

} // namespace source
} // namespace STORM

#endif // STORM_RADIATION_SOURCE_CORE_HPP
