#ifndef STORM_RADIATION_DDMC_ADVANCE_DDMC_HPP
#define STORM_RADIATION_DDMC_ADVANCE_DDMC_HPP

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "../transport/TransportPortability.hpp"
#include "../../particle/RadiationTransportState.hpp"
#include "../../particle/StepResult.hpp"
#include "../../types.hpp"
#include "../../utils/CounterRNG.hpp"
#include "DDMCSampling.hpp"
#include "DDMCTypes.hpp"

namespace STORM::ddmc
{

enum class AdvanceError : std::uint8_t
{
    None,
    InvalidCell,
    InvalidData,
    InvalidNormal
};

enum class AdvanceEvent : std::uint8_t
{
    None,
    Entry,
    AbsorptionCutoff,
    Census,
    DDMCLeak,
    IMCLeak,
    Upscatter,
    GroupCutoff
};

template<typename PointT>
struct DeviceView
{
    const std::uint8_t *cellEligible = nullptr;
    const double *sigmaEnergyAbs = nullptr;
    const double *sigmaParticleGate = nullptr;
    const double *totalLeakRate = nullptr;
    const double *gamma = nullptr;
    const double *velocityDivergence = nullptr;
    const double *cellTemperature = nullptr;
    const std::size_t *groupCutoff = nullptr;
    const cell_id_t *cellIDs = nullptr;
    const std::size_t *leakOffsets = nullptr;
    const double *leakRates = nullptr;
    const double *ddmcLeakRates = nullptr;
    const cell_index_t *nextCellIndices = nullptr;
    const std::uint8_t *faceKinds = nullptr;
    const std::uint8_t *targetDDMCEligible = nullptr;
    const std::size_t *targetGroupCutoff = nullptr;
    const PointT *outwardNormals = nullptr;
    const PointT *faceCenters = nullptr;
    PointT *fluxRhs = nullptr;
    std::size_t *stepCount = nullptr;
    std::size_t *leakCount = nullptr;
    std::size_t *residentLeakCount = nullptr;
    std::size_t *transportLeakCount = nullptr;
    std::size_t *remoteResidentLeakCount = nullptr;
    std::size_t *censusCount = nullptr;
    std::size_t cellCount = 0;
    double minimumParticleOpticalDepth = 0.0;
    std::uint8_t enabled = 0;
    std::uint8_t pgrwEnabled = 0;
};

template<typename PointT>
struct HostSnapshot
{
    std::vector<std::uint8_t> cellEligible;
    std::vector<double> sigmaEnergyAbs;
    std::vector<double> sigmaParticleGate;
    std::vector<double> totalLeakRate;
    std::vector<double> gamma;
    std::vector<double> velocityDivergence;
    std::vector<double> cellTemperature;
    std::vector<std::size_t> groupCutoff;
    std::vector<cell_id_t> cellIDs;
    std::vector<std::size_t> leakOffsets;
    std::vector<double> leakRates;
    std::vector<double> ddmcLeakRates;
    std::vector<cell_index_t> nextCellIndices;
    std::vector<std::uint8_t> faceKinds;
    std::vector<std::uint8_t> targetDDMCEligible;
    std::vector<std::size_t> targetGroupCutoff;
    std::vector<PointT> outwardNormals;
    std::vector<PointT> faceCenters;
    std::vector<PointT> fluxRhs;
    double minimumParticleOpticalDepth = 0.0;
    bool enabled = false;
    bool pgrwEnabled = false;

    template<typename GridT>
    void Build(const std::vector<CellData<PointT>> &cells,
               const GridT &grid,
               const double minimumOpticalDepth,
               const std::vector<double> &temperatures,
               const std::vector<std::size_t> &stableCellIDs,
               const bool useMultigroupPGRW)
    {
        this->cellEligible.resize(cells.size());
        this->sigmaEnergyAbs.resize(cells.size());
        this->sigmaParticleGate.resize(cells.size());
        this->totalLeakRate.resize(cells.size());
        this->gamma.resize(cells.size());
        this->velocityDivergence.resize(cells.size());
        this->cellTemperature.resize(cells.size());
        this->groupCutoff.resize(cells.size());
        this->cellIDs.resize(cells.size());
        this->leakOffsets.resize(cells.size() + 1);
        this->leakRates.clear();
        this->ddmcLeakRates.clear();
        this->nextCellIndices.clear();
        this->faceKinds.clear();
        this->targetDDMCEligible.clear();
        this->targetGroupCutoff.clear();
        this->outwardNormals.clear();
        this->faceCenters.clear();

        std::size_t leak = 0;
        for(std::size_t cellIndex = 0; cellIndex < cells.size(); ++cellIndex)
        {
            const CellData<PointT> &cell = cells[cellIndex];
            this->cellEligible[cellIndex] = cell.eligible ? 1u : 0u;
            this->sigmaEnergyAbs[cellIndex] = cell.sigmaEnergyAbs;
            this->sigmaParticleGate[cellIndex] = cell.sigmaParticleGate;
            this->totalLeakRate[cellIndex] = cell.totalLeakRate;
            this->gamma[cellIndex] = cell.gamma;
            this->velocityDivergence[cellIndex] = cell.velocityDivergence;
            this->cellTemperature[cellIndex] =
                cellIndex < temperatures.size() ? temperatures[cellIndex] : 0.0;
            this->groupCutoff[cellIndex] = cell.groupCutoff;
            if(cellIndex < stableCellIDs.size() &&
               stableCellIDs[cellIndex] !=
                   std::numeric_limits<std::size_t>::max())
            {
                this->cellIDs[cellIndex] =
                    static_cast<cell_id_t>(stableCellIDs[cellIndex]);
            }
            else
            {
                this->cellIDs[cellIndex] = static_cast<cell_id_t>(cellIndex);
            }
            this->leakOffsets[cellIndex] = leak;
            for(const FaceLeak<PointT> &face : cell.faceLeaks)
            {
                this->leakRates.push_back(face.rate);
                this->ddmcLeakRates.push_back(face.ddmcRate);
                this->nextCellIndices.push_back(
                    static_cast<cell_index_t>(face.nextCellIndex));
                this->faceKinds.push_back(
                    static_cast<std::uint8_t>(face.kind));
                this->targetDDMCEligible.push_back(
                    face.targetDDMCEligible ? 1u : 0u);
                this->targetGroupCutoff.push_back(face.targetGroupCutoff);
                this->outwardNormals.push_back(face.outwardNormal);
                this->faceCenters.push_back(grid.FaceCM(face.faceIndex));
                ++leak;
            }
        }
        this->leakOffsets[cells.size()] = leak;
        this->fluxRhs.assign(cells.size(), PointT{});
        this->minimumParticleOpticalDepth = minimumOpticalDepth;
        this->pgrwEnabled = useMultigroupPGRW;
        this->enabled = true;
    }

    DeviceView<PointT> View()
    {
        DeviceView<PointT> result;
        result.cellEligible = this->cellEligible.data();
        result.sigmaEnergyAbs = this->sigmaEnergyAbs.data();
        result.sigmaParticleGate = this->sigmaParticleGate.data();
        result.totalLeakRate = this->totalLeakRate.data();
        result.gamma = this->gamma.data();
        result.velocityDivergence = this->velocityDivergence.data();
        result.cellTemperature = this->cellTemperature.data();
        result.groupCutoff = this->groupCutoff.data();
        result.cellIDs = this->cellIDs.data();
        result.leakOffsets = this->leakOffsets.data();
        result.leakRates = this->leakRates.data();
        result.ddmcLeakRates = this->ddmcLeakRates.data();
        result.nextCellIndices = this->nextCellIndices.data();
        result.faceKinds = this->faceKinds.data();
        result.targetDDMCEligible = this->targetDDMCEligible.data();
        result.targetGroupCutoff = this->targetGroupCutoff.data();
        result.outwardNormals = this->outwardNormals.data();
        result.faceCenters = this->faceCenters.data();
        result.fluxRhs = this->fluxRhs.data();
        result.cellCount = this->cellEligible.size();
        result.minimumParticleOpticalDepth = this->minimumParticleOpticalDepth;
        result.enabled = this->enabled ? 1u : 0u;
        result.pgrwEnabled = this->pgrwEnabled ? 1u : 0u;
        return result;
    }
};

template<typename PointT>
struct ColdState
{
    PointT pendingFlux{};
};

template<typename PointT>
struct AdvanceResult
{
    StepResult step;
    PointT pendingFlux{};
    AdvanceError error = AdvanceError::None;
    AdvanceEvent event = AdvanceEvent::None;
    std::uint8_t taken = 0;
    std::uint8_t entered = 0;
    std::uint8_t remotePendingFlux = 0;
};

template<typename ParticleT, typename = void>
struct HasFlatRadiationFlags : std::false_type
{};

template<typename ParticleT>
struct HasFlatRadiationFlags<ParticleT, std::void_t<decltype(std::declval<ParticleT &>().radiationFlags)>>
    : std::true_type
{};

template<typename ParticleT>
STORM_TRANSPORT_INLINE
std::uint8_t &RadiationFlags(ParticleT &particle)
{
    if constexpr(HasFlatRadiationFlags<ParticleT>::value)
        return particle.radiationFlags;
    else
        return particle.radiationState.flags;
}

template<typename PointT>
STORM_TRANSPORT_INLINE
double Dot(const PointT &left, const PointT &right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template<typename PointT>
STORM_TRANSPORT_INLINE
PointT Scale(const PointT &point, const double scale)
{
    PointT result;
    result.x = point.x * scale;
    result.y = point.y * scale;
    result.z = point.z * scale;
    return result;
}

template<typename PointT>
STORM_TRANSPORT_INLINE
PointT Add(const PointT &left, const PointT &right)
{
    PointT result;
    result.x = left.x + right.x;
    result.y = left.y + right.y;
    result.z = left.z + right.z;
    return result;
}

template<typename PointT>
STORM_TRANSPORT_INLINE
PointT Cross(const PointT &left, const PointT &right)
{
    PointT result;
    result.x = left.y * right.z - left.z * right.y;
    result.y = left.z * right.x - left.x * right.z;
    result.z = left.x * right.y - left.y * right.x;
    return result;
}

template<typename PointT>
STORM_TRANSPORT_INLINE
PointT Normalize(const PointT &point)
{
    const double magnitude = transport::Sqrt(Dot(point, point));
    return magnitude > 0.0 ? Scale(point, 1.0 / magnitude) : PointT{};
}

template<typename ParticleT, typename PointT>
STORM_TRANSPORT_INLINE
void SampleIsotropicDirection(ParticleT &particle, const double speed)
{
    const double mu = 1.0 - 2.0 * CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double phi = 6.28318530717958647692 * CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double radial = 1.0 - mu * mu;
    const double sinTheta = transport::Sqrt(radial > 0.0 ? radial : 0.0);
    particle.velocity.x = speed * sinTheta * transport::Cos(phi);
    particle.velocity.y = speed * sinTheta * transport::Sin(phi);
    particle.velocity.z = speed * mu;
}

STORM_TRANSPORT_INLINE
double SampleAsymptoticMuPortable(double random)
{
    random = (random < 0.0) ? 0.0 : ((random > 1.0) ? 1.0 : random);
    double lo = 0.0;
    double hi = 1.0;
    for(int iteration = 0; iteration < 56; ++iteration)
    {
        const double mu = 0.5 * (lo + hi);
        const double cdf = 0.5 * mu * mu * (1.0 + mu);
        if(cdf < random)
        {
            lo = mu;
        }
        else
        {
            hi = mu;
        }
    }
    return 0.5 * (lo + hi);
}

template<typename ViewsT, typename PointT>
STORM_TRANSPORT_INLINE
void AddFlux(const ViewsT &views, const std::size_t cellIndex, const PointT &contribution)
{
    if(views.ddmc.fluxRhs == nullptr or cellIndex >= views.ddmc.cellCount)
    {
        return;
    }
    STORM_TRANSPORT_ACCUMULATE(views.ddmc.fluxRhs[cellIndex].x, contribution.x);
    STORM_TRANSPORT_ACCUMULATE(views.ddmc.fluxRhs[cellIndex].y, contribution.y);
    STORM_TRANSPORT_ACCUMULATE(views.ddmc.fluxRhs[cellIndex].z, contribution.z);
}

STORM_TRANSPORT_INLINE
double NextToward(const double from, const double to)
{
#ifdef STORM_WITH_GPU
    return nextafter(from, to);
#else
    return std::nextafter(from, to);
#endif
}

STORM_TRANSPORT_INLINE
double NextUp(const double value)
{
    return NextToward(value, DBL_MAX);
}

STORM_TRANSPORT_INLINE
double ClampFrequency(const double *boundaries,
                      const std::size_t groupCount,
                      double frequency)
{
    if(boundaries == nullptr || groupCount == 0)
    {
        return frequency;
    }
    if(frequency < boundaries[0])
    {
        return boundaries[0];
    }
    if(frequency > boundaries[groupCount])
    {
        return boundaries[groupCount];
    }
    return frequency;
}

template<typename ParticleT, typename PointT>
STORM_TRANSPORT_INLINE
void LorentzBoost(ParticleT &particle, const PointT &boostVelocity, const double speedOfLight)
{
    const double v2 = Dot(boostVelocity, boostVelocity);
    if(v2 < 1.0e-30 || !(speedOfLight > 0.0))
    {
        return;
    }
    const double inverseC2 = 1.0 / (speedOfLight * speedOfLight);
    const double gamma = 1.0 / transport::Sqrt(1.0 - v2 * inverseC2);
    const double doppler =
        gamma * (1.0 - Dot(boostVelocity, particle.velocity) * inverseC2);
    if(!(doppler > 0.0) || !transport::IsFinite(doppler))
    {
        return;
    }
    particle.frequency *= doppler;
    particle.weight *= doppler;
    const double vDotP = Dot(particle.velocity, boostVelocity);
    particle.velocity = Add(
        particle.velocity,
        Scale(boostVelocity, (gamma - 1.0) * vDotP / v2 - gamma));
    const double newSpeed = transport::Sqrt(Dot(particle.velocity, particle.velocity));
    if(newSpeed > 0.0)
    {
        particle.velocity = Scale(particle.velocity, speedOfLight / newSpeed);
    }
}

template<typename ParticleT, typename ViewsT>
STORM_TRANSPORT_INLINE
void LorentzToComoving(ParticleT &particle,
                       const ViewsT &views,
                       const std::size_t cellIndex)
{
    using PointT = typename ViewsT::point_type;
    if(!views.comovingTransport || views.cellVelocities == nullptr)
    {
        return;
    }
    LorentzBoost<ParticleT, PointT>(
        particle, views.cellVelocities[cellIndex], views.speedOfLight);
}

template<typename ParticleT, typename ViewsT>
STORM_TRANSPORT_INLINE
void LorentzToLab(ParticleT &particle,
                  const ViewsT &views,
                  const std::size_t cellIndex)
{
    using PointT = typename ViewsT::point_type;
    if(!views.comovingTransport || views.cellVelocities == nullptr)
    {
        return;
    }
    PointT negV = Scale(views.cellVelocities[cellIndex], -1.0);
    LorentzBoost<ParticleT, PointT>(particle, negV, views.speedOfLight);
}

template<typename ParticleT, typename ColdT>
STORM_TRANSPORT_INLINE
cell_id_t BypassCellID(const ParticleT &particle, const ColdT &cold)
{
    if constexpr(HasFlatRadiationFlags<ParticleT>::value)
    {
        return cold.bypassCellID;
    }
    else
    {
        (void) cold;
        return particle.radiationState.bypassCellID;
    }
}

template<typename ParticleT>
STORM_TRANSPORT_INLINE
void SamplePlanckBandFrequency(ParticleT &particle,
                               const double *boundaries,
                               const std::size_t groupCount,
                               const double kT,
                               const std::size_t beginGroup,
                               const std::size_t endGroup)
{
    if(boundaries == nullptr || beginGroup >= endGroup || endGroup > groupCount)
    {
        return;
    }
    const double bandMass = PlanckBandMass(boundaries, kT, beginGroup, endGroup);
    if(!(bandMass > 0.0))
    {
        return;
    }
    double remaining =
        CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) * bandMass;
    for(std::size_t group = beginGroup; group < endGroup; ++group)
    {
        const double groupMass = PlanckBandMass(boundaries, kT, group, group + 1);
        if(remaining <= groupMass || group + 1 == endGroup)
        {
            double localRandom = 0.0;
            if(groupMass > 0.0)
            {
                localRandom = remaining / groupMass;
                if(localRandom < 0.0)
                {
                    localRandom = 0.0;
                }
                else if(localRandom > 1.0)
                {
                    localRandom = 1.0;
                }
            }
            else
            {
                localRandom = CounterRNG::unitOpen(
                    particle.rngKey, particle.rngCounter++);
            }
            particle.frequency = SampleFrequencyInGroup(
                boundaries, groupCount, group, localRandom);
            return;
        }
        remaining -= groupMass;
    }
}

template<typename ParticleT, typename ViewsT>
STORM_TRANSPORT_INLINE
void ExitDDMCToTransport(ParticleT &particle,
                         std::uint8_t &radiationFlags,
                         const ViewsT &views,
                         const std::size_t cellIndex,
                         const bool sampleDirection,
                         const bool convertedIncomingToComoving,
                         const std::uint8_t ddmcFlags,
                         const std::uint8_t pending)
{
    using PointT = typename ViewsT::point_type;
    const bool packetInDDMC = (radiationFlags & ddmcFlags) != 0;
    if(!packetInDDMC && !convertedIncomingToComoving)
    {
        return;
    }
    if(packetInDDMC)
    {
        particle.location = views.grid.cellCenters[cellIndex];
        if(sampleDirection)
        {
            SampleIsotropicDirection<ParticleT, PointT>(
                particle, views.speedOfLight);
        }
    }
    if((radiationFlags & RadiationTransportState<PointT>::DDMCComovingFrame) != 0 ||
       convertedIncomingToComoving)
    {
        LorentzToLab(particle, views, cellIndex);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
    }
    radiationFlags = static_cast<std::uint8_t>(
        radiationFlags & ~(ddmcFlags | pending));
    particle.initialWeight = transport::Abs(particle.weight);
}

template<typename ParticleT, typename ColdT, typename ViewsT>
STORM_TRANSPORT_INLINE
AdvanceResult<typename ViewsT::point_type> AdvanceDDMC(ParticleT &particle, ColdT &cold, const ViewsT &views)
{
    using PointT = typename ViewsT::point_type;
    AdvanceResult<PointT> result;
    const DeviceView<PointT> &ddmc = views.ddmc;
    if(not ddmc.enabled)
    {
        return result;
    }

    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= ddmc.cellCount)
    {
        result.error = AdvanceError::InvalidCell;
        return result;
    }

    constexpr std::uint8_t mode = RadiationTransportState<PointT>::DDMCMode;
    constexpr std::uint8_t resident = RadiationTransportState<PointT>::DDMCCellResident;
    constexpr std::uint8_t comoving = RadiationTransportState<PointT>::DDMCComovingFrame;
    constexpr std::uint8_t pending = RadiationTransportState<PointT>::PendingFlux;
    constexpr std::uint8_t ddmcFlags = mode | resident | comoving;
    std::uint8_t &radiationFlags = RadiationFlags(particle);
    const bool packetInDDMC = (radiationFlags & mode) != 0;
    bool convertedIncomingToComoving = false;

    if((radiationFlags & pending) != 0)
    {
        AddFlux(views, cellIndex, cold.pendingFlux);
        cold.pendingFlux = PointT{};
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~pending);
    }

    if(ddmc.cellEligible == nullptr || !ddmc.cellEligible[cellIndex])
    {
        ExitDDMCToTransport(
            particle, radiationFlags, views, cellIndex, true,
            convertedIncomingToComoving, ddmcFlags, pending);
        return result;
    }
    if(ddmc.totalLeakRate == nullptr || ddmc.sigmaEnergyAbs == nullptr ||
       ddmc.sigmaParticleGate == nullptr || ddmc.leakOffsets == nullptr ||
       ddmc.leakRates == nullptr || ddmc.ddmcLeakRates == nullptr ||
       ddmc.nextCellIndices == nullptr || ddmc.targetDDMCEligible == nullptr ||
       ddmc.outwardNormals == nullptr || ddmc.faceCenters == nullptr)
    {
        result.error = AdvanceError::InvalidData;
        return result;
    }

    if(!packetInDDMC)
    {
        const cell_id_t bypass = BypassCellID(particle, cold);
        if(bypass != std::numeric_limits<cell_id_t>::max() &&
           ddmc.cellIDs != nullptr &&
           ddmc.cellIDs[cellIndex] == bypass)
        {
            return result;
        }
    }

    const double totalLeakRate = ddmc.totalLeakRate[cellIndex];
    const std::size_t leakBegin = ddmc.leakOffsets[cellIndex];
    const std::size_t leakEnd = ddmc.leakOffsets[cellIndex + 1];
    const double fleck = views.fleckFactors != nullptr
        ? views.fleckFactors[cellIndex]
        : 1.0;
    const double gamma = ddmc.gamma != nullptr ? ddmc.gamma[cellIndex] : 1.0;
    const double velocityDivergence =
        ddmc.velocityDivergence != nullptr
            ? ddmc.velocityDivergence[cellIndex]
            : 0.0;
    const std::size_t groupCutoff =
        ddmc.groupCutoff != nullptr ? ddmc.groupCutoff[cellIndex] : 0;
    const bool pgrw = ddmc.pgrwEnabled && views.energyBoundaries != nullptr &&
        views.groupCount > 0;

    if(not packetInDDMC)
    {
        double minimumDistance = DBL_MAX;
        const std::size_t faceBegin = views.grid.cellFaceOffsets[cellIndex];
        const std::size_t faceEnd = views.grid.cellFaceOffsets[cellIndex + 1];
        for(std::size_t face = faceBegin; face < faceEnd; ++face)
        {
            const PointT &normal = views.grid.normals[face];
            const double distance = Dot(particle.location, normal) - views.grid.facePlaneOffsets[face];
            if(distance < minimumDistance)
            {
                minimumDistance = distance;
            }
        }
        if(not (minimumDistance > 0.0) or minimumDistance * ddmc.sigmaParticleGate[cellIndex] < ddmc.minimumParticleOpticalDepth)
        {
            return result;
        }
    }

    if(pgrw)
    {
        if(groupCutoff == 0 || groupCutoff > views.groupCount)
        {
            ExitDDMCToTransport(
                particle, radiationFlags, views, cellIndex, true,
                convertedIncomingToComoving, ddmcFlags, pending);
            return result;
        }
        double probeFrequency = particle.frequency;
        if(!packetInDDMC && views.comovingTransport &&
           views.cellVelocities != nullptr)
        {
            ParticleT probe = particle;
            LorentzToComoving(probe, views, cellIndex);
            probeFrequency = probe.frequency;
        }
        probeFrequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, probeFrequency);
        if(probeFrequency >= views.energyBoundaries[groupCutoff])
        {
            ExitDDMCToTransport(
                particle, radiationFlags, views, cellIndex, true,
                convertedIncomingToComoving, ddmcFlags, pending);
            return result;
        }
    }

    if(!packetInDDMC && views.comovingTransport)
    {
        LorentzToComoving(particle, views, cellIndex);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
        convertedIncomingToComoving = true;
    }

    double upscatterRate = 0.0;
    if(pgrw && gamma < 1.0 && ddmc.sigmaEnergyAbs[cellIndex] > 0.0 &&
       (fleck > 0.0 || !views.depositMaterialEnergy))
    {
        upscatterRate = views.speedOfLight * (1.0 - fleck) *
            ddmc.sigmaEnergyAbs[cellIndex] * (1.0 - gamma);
    }
    const double eventRate = totalLeakRate + upscatterRate;
    if(!(eventRate > 0.0) || leakBegin > leakEnd ||
       (totalLeakRate > 0.0 && leakBegin >= leakEnd))
    {
        ExitDDMCToTransport(
            particle, radiationFlags, views, cellIndex, true,
            convertedIncomingToComoving, ddmcFlags, pending);
        return result;
    }

    result.taken = 1;
    const double tEvent =
        -transport::Log(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++)) /
        eventRate;
    const double tCensus = particle.timeLeft;
    double tCutoff = DBL_MAX;
    if(pgrw && groupCutoff > 0 && groupCutoff <= views.groupCount &&
       velocityDivergence < 0.0)
    {
        double frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
        const double cutoffFrequency = views.energyBoundaries[groupCutoff];
        const double growthRate = -velocityDivergence / 3.0;
        if(frequency > 0.0 && frequency < cutoffFrequency && growthRate > 0.0)
        {
            tCutoff = transport::Log(cutoffFrequency / frequency) / growthRate;
        }
    }
    const bool censusEvent = tCensus <= tEvent && tCensus <= tCutoff;
    const bool cutoffEvent = tCutoff < tEvent && tCutoff < tCensus;
    const double dt = censusEvent ? tCensus : (cutoffEvent ? tCutoff : tEvent);

    if(views.comovingTransport && velocityDivergence != 0.0)
    {
        double logShift = -velocityDivergence * dt / 3.0;
        if(transport::IsFinite(logShift) && logShift != 0.0)
        {
            if(logShift < -50.0)
            {
                logShift = -50.0;
            }
            else if(logShift > 50.0)
            {
                logShift = 50.0;
            }
            const double shift = transport::Exp(logShift);
            particle.frequency *= shift;
            particle.weight *= shift;
            particle.frequency = ClampFrequency(
                views.energyBoundaries, views.groupCount, particle.frequency);
        }
    }

    const double absorptionRate =
        ddmc.sigmaEnergyAbs[cellIndex] * fleck * views.speedOfLight;
    const double oldWeight = particle.weight;
    const double expFactor = transport::Expm1(-dt * absorptionRate);
    const double integratedEnergy =
        absorptionRate > 0.0
            ? oldWeight * expFactor * (-1.0 / absorptionRate)
            : oldWeight * dt;

    if(views.depositMaterialEnergy)
    {
        STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], -expFactor * oldWeight);
        if(views.depositMomentum && views.pendingMomentum != nullptr &&
           views.cellVelocities != nullptr)
        {
            const double absorbed = -expFactor * oldWeight;
            const double inverseC2 =
                1.0 / (views.speedOfLight * views.speedOfLight);
            const PointT &velocity = views.cellVelocities[cellIndex];
            STORM_TRANSPORT_ACCUMULATE(
                views.pendingMomentum[cellIndex].x,
                absorbed * velocity.x * inverseC2);
            STORM_TRANSPORT_ACCUMULATE(
                views.pendingMomentum[cellIndex].y,
                absorbed * velocity.y * inverseC2);
            STORM_TRANSPORT_ACCUMULATE(
                views.pendingMomentum[cellIndex].z,
                absorbed * velocity.z * inverseC2);
        }
    }
    STORM_TRANSPORT_ACCUMULATE(views.pendingRadiationEnergy[cellIndex], integratedEnergy);
    if(views.pendingGroupRadiationEnergy != nullptr && pgrw &&
       groupCutoff > 0 && groupCutoff <= views.groupCount &&
       ddmc.cellTemperature != nullptr)
    {
        const double kT = boltzmannConstant * ddmc.cellTemperature[cellIndex];
        const double bandMass = PlanckBandMass(
            views.energyBoundaries, kT, 0, groupCutoff);
        if(bandMass > 0.0)
        {
            for(std::size_t group = 0; group < groupCutoff; ++group)
            {
                const double groupMass = PlanckBandMass(
                    views.energyBoundaries, kT, group, group + 1);
                STORM_TRANSPORT_ACCUMULATE(
                    views.pendingGroupRadiationEnergy[
                        cellIndex * views.groupCount + group],
                    integratedEnergy * groupMass / bandMass);
            }
        }
    }
    particle.weight *= 1.0 + expFactor;
    particle.timeLeft -= dt;
    if(ddmc.stepCount != nullptr)
    {
        STORM_TRANSPORT_ACCUMULATE(*ddmc.stepCount, std::size_t(1));
    }

    if(transport::Abs(particle.weight) < particle.initialWeight * 1.0e-3)
    {
        if(views.depositMaterialEnergy)
        {
            STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], particle.weight);
            if(views.depositMomentum && views.pendingMomentum != nullptr &&
               views.cellVelocities != nullptr)
            {
                const double inverseC2 =
                    1.0 / (views.speedOfLight * views.speedOfLight);
                const PointT &velocity = views.cellVelocities[cellIndex];
                STORM_TRANSPORT_ACCUMULATE(
                    views.pendingMomentum[cellIndex].x,
                    particle.weight * velocity.x * inverseC2);
                STORM_TRANSPORT_ACCUMULATE(
                    views.pendingMomentum[cellIndex].y,
                    particle.weight * velocity.y * inverseC2);
                STORM_TRANSPORT_ACCUMULATE(
                    views.pendingMomentum[cellIndex].z,
                    particle.weight * velocity.z * inverseC2);
            }
        }
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~ddmcFlags);
        result.step.change = ParticleStatus::REMOVE;
        result.event = AdvanceEvent::AbsorptionCutoff;
        return result;
    }

    if(cutoffEvent)
    {
        particle.frequency = NextUp(views.energyBoundaries[groupCutoff]);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
        SampleIsotropicDirection<ParticleT, PointT>(
            particle, views.speedOfLight);
        LorentzToLab(particle, views, cellIndex);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
        particle.initialWeight = transport::Abs(particle.weight);
        radiationFlags = static_cast<std::uint8_t>(
            radiationFlags & ~(ddmcFlags | pending));
        result.step.change = ParticleStatus::NO_CELL_MOVE;
        result.event = AdvanceEvent::GroupCutoff;
        return result;
    }

    if(!packetInDDMC)
    {
        const double speed = transport::Sqrt(Dot(particle.velocity, particle.velocity));
        if(speed > 0.0)
        {
            AddFlux(views, cellIndex, Scale(particle.velocity, particle.weight / speed));
        }
        radiationFlags = static_cast<std::uint8_t>(radiationFlags | ddmcFlags);
        particle.location = views.grid.cellCenters[cellIndex];
        SampleIsotropicDirection<ParticleT, PointT>(
            particle, views.speedOfLight);
        result.entered = 1;
        result.event = AdvanceEvent::Entry;
    }

    if(censusEvent)
    {
        particle.location = views.grid.cellCenters[cellIndex];
        if(pgrw && groupCutoff > 0 && groupCutoff <= views.groupCount &&
           ddmc.cellTemperature != nullptr)
        {
            const double kT = boltzmannConstant * ddmc.cellTemperature[cellIndex];
            SamplePlanckBandFrequency(
                particle, views.energyBoundaries, views.groupCount, kT,
                0, groupCutoff);
            const double upperBand = views.energyBoundaries[groupCutoff];
            if(particle.frequency > NextUp(upperBand) ||
               particle.frequency >= upperBand)
            {
                particle.frequency = NextToward(
                    upperBand, views.energyBoundaries[0]);
            }
            particle.frequency = ClampFrequency(
                views.energyBoundaries, views.groupCount, particle.frequency);
        }
        else if(views.spectralEnabled && views.thermalEmissionCdf != nullptr)
        {
            particle.frequency = SampleFrequencyFromCellCdf(
                views.energyBoundaries,
                views.thermalEmissionCdf,
                views.groupCount,
                cellIndex,
                CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
            particle.frequency = ClampFrequency(
                views.energyBoundaries, views.groupCount, particle.frequency);
        }
        SampleIsotropicDirection<ParticleT, PointT>(particle, views.speedOfLight);
        LorentzToLab(particle, views, cellIndex);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~(ddmcFlags | pending));
        cold.pendingFlux = PointT{};
        result.step.change = ParticleStatus::DONE;
        result.event = AdvanceEvent::Census;
        if(ddmc.censusCount != nullptr)
            STORM_TRANSPORT_ACCUMULATE(*ddmc.censusCount, std::size_t(1));
        return result;
    }

    const double eventPick =
        CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) * eventRate;
    if(eventPick > totalLeakRate)
    {
        if(not pgrw)
        {
            SampleIsotropicDirection<ParticleT, PointT>(particle, views.speedOfLight);
            radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~(ddmcFlags | pending));
            result.step.change = ParticleStatus::DONE;
            result.event = AdvanceEvent::Census;
            if(ddmc.censusCount != nullptr)
            {
                STORM_TRANSPORT_ACCUMULATE(*ddmc.censusCount, std::size_t(1));
            }
            return result;
        }
        const double kT = (ddmc.cellTemperature != nullptr)? boltzmannConstant * ddmc.cellTemperature[cellIndex] : 0.0;
        SamplePlanckBandFrequency(particle, views.energyBoundaries, views.groupCount, kT, groupCutoff, views.groupCount);
        particle.frequency = ClampFrequency(views.energyBoundaries, views.groupCount, particle.frequency);
        SampleIsotropicDirection<ParticleT, PointT>(particle, views.speedOfLight);
        LorentzToLab(particle, views, cellIndex);
        particle.frequency = ClampFrequency(views.energyBoundaries, views.groupCount, particle.frequency);
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~(ddmcFlags | pending));
        particle.initialWeight = transport::Abs(particle.weight);
        result.step.change = ParticleStatus::NO_CELL_MOVE;
        result.event = AdvanceEvent::Upscatter;
        return result;
    }

    double facePick = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) * totalLeakRate;
    std::size_t chosen = leakEnd - 1;
    for(std::size_t leak = leakBegin; leak < leakEnd; ++leak)
    {
        facePick -= ddmc.leakRates[leak];
        if(facePick <= 0.0)
        {
            chosen = leak;
            break;
        }
    }

    if(ddmc.faceKinds != nullptr and ddmc.faceKinds[chosen] == static_cast<std::uint8_t>(FaceKind::ThermalizingBoundary))
    {
        result.error = AdvanceError::InvalidData;
        return result;
    }

    PointT normal = Normalize(ddmc.outwardNormals[chosen]);
    if(!(Dot(normal, normal) > 0.0))
    {
        result.error = AdvanceError::InvalidNormal;
        return result;
    }
    const double rate = ddmc.leakRates[chosen];
    const bool useDDMCChannel = (ddmc.ddmcLeakRates[chosen] > 0.0) and CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) < (ddmc.ddmcLeakRates[chosen] / rate);
    const double mu = useDDMCChannel? SampleAsymptoticMuPortable(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++)) : transport::Sqrt(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++));
    const double phi = 6.28318530717958647692 * CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double sinTheta = transport::Sqrt((1.0 - mu * mu) > 0.0 ? 1.0 - mu * mu : 0.0);

    PointT helper;
    if(transport::Abs(normal.x) < 0.9)
    {
        helper.x = 1.0;
    }
    else
    {
        helper.y = 1.0;
    }
    PointT e1 = Normalize(Add(helper, Scale(normal, -Dot(helper, normal))));
    const PointT e2 = Normalize(Cross(normal, e1));
    PointT direction = Add(Scale(normal, mu), Add(Scale(e1, sinTheta * transport::Cos(phi)), Scale(e2, sinTheta * transport::Sin(phi))));
    direction = Normalize(direction);

    particle.location = ddmc.faceCenters[chosen];
    particle.velocity = Scale(direction, views.speedOfLight);

    const bool targetDDMC = useDDMCChannel && ddmc.targetDDMCEligible[chosen];
    const double fluxWeight = particle.weight;
    if(!targetDDMC && pgrw)
    {
        std::size_t beginGroup = 0;
        const std::size_t targetCutoff =
            ddmc.targetGroupCutoff != nullptr
                ? ddmc.targetGroupCutoff[chosen]
                : groupCutoff;
        if(ddmc.targetDDMCEligible[chosen] && targetCutoff > 0 &&
           targetCutoff < groupCutoff)
        {
            beginGroup = targetCutoff;
        }
        const double kT = ddmc.cellTemperature != nullptr
            ? boltzmannConstant * ddmc.cellTemperature[cellIndex]
            : 0.0;
        SamplePlanckBandFrequency(
            particle, views.energyBoundaries, views.groupCount, kT,
            beginGroup, groupCutoff);
        particle.frequency = ClampFrequency(
            views.energyBoundaries, views.groupCount, particle.frequency);
    }

    if(not targetDDMC and views.comovingTransport)
    {
        LorentzToLab(particle, views, cellIndex);
        particle.frequency = ClampFrequency(views.energyBoundaries, views.groupCount, particle.frequency);
        particle.initialWeight = transport::Abs(particle.weight);
    }

    const PointT fluxContribution = Scale(direction, fluxWeight);
    AddFlux(views, cellIndex, fluxContribution);

    const std::size_t nextCell = static_cast<std::size_t>(ddmc.nextCellIndices[chosen]);
    if(targetDDMC)
    {
        radiationFlags = static_cast<std::uint8_t>(radiationFlags | ddmcFlags);
        if(nextCell < ddmc.cellCount)
        {
            AddFlux(views, nextCell, fluxContribution);
        }
        else
        {
            cold.pendingFlux = fluxContribution;
            radiationFlags = static_cast<std::uint8_t>(radiationFlags | pending);
            result.pendingFlux = fluxContribution;
            result.remotePendingFlux = 1;
        }
        result.event = AdvanceEvent::DDMCLeak;
        if(ddmc.leakCount != nullptr)
        {
            STORM_TRANSPORT_ACCUMULATE(*ddmc.leakCount, std::size_t(1));
        }
        if(ddmc.residentLeakCount != nullptr)
        {
            STORM_TRANSPORT_ACCUMULATE(*ddmc.residentLeakCount, std::size_t(1));
        }
        if(result.remotePendingFlux and ddmc.remoteResidentLeakCount != nullptr)
        {
            STORM_TRANSPORT_ACCUMULATE(*ddmc.remoteResidentLeakCount, std::size_t(1));
        }
    }
    else
    {
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~(ddmcFlags | pending));
        cold.pendingFlux = PointT{};
        result.event = AdvanceEvent::IMCLeak;
        if(ddmc.leakCount != nullptr)
        {
            STORM_TRANSPORT_ACCUMULATE(*ddmc.leakCount, std::size_t(1));
        }
        if(ddmc.transportLeakCount != nullptr)
        {
            STORM_TRANSPORT_ACCUMULATE(*ddmc.transportLeakCount, std::size_t(1));
        }
    }

    result.step.change = ParticleStatus::CELL_MOVE;
    result.step.nextCellIndex = ddmc.nextCellIndices[chosen];
    return result;
}

} // namespace STORM::ddmc

#endif
