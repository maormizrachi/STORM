#ifndef STORM_RADIATION_DDMC_ADVANCE_DDMC_HPP
#define STORM_RADIATION_DDMC_ADVANCE_DDMC_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "../transport/TransportPortability.hpp"
#include "../../particle/RadiationTransportState.hpp"
#include "../../particle/StepResult.hpp"
#include "../../utils/CounterRNG.hpp"
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
    IMCLeak
};

template<typename PointT>
struct DeviceView
{
    const std::uint8_t *cellEligible = nullptr;
    const double *sigmaEnergyAbs = nullptr;
    const double *sigmaParticleGate = nullptr;
    const double *totalLeakRate = nullptr;
    const std::size_t *leakOffsets = nullptr;
    const double *leakRates = nullptr;
    const double *ddmcLeakRates = nullptr;
    const cell_index_t *nextCellIndices = nullptr;
    const std::uint8_t *faceKinds = nullptr;
    const std::uint8_t *targetDDMCEligible = nullptr;
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
};

template<typename PointT>
struct HostSnapshot
{
    std::vector<std::uint8_t> cellEligible;
    std::vector<double> sigmaEnergyAbs;
    std::vector<double> sigmaParticleGate;
    std::vector<double> totalLeakRate;
    std::vector<std::size_t> leakOffsets;
    std::vector<double> leakRates;
    std::vector<double> ddmcLeakRates;
    std::vector<cell_index_t> nextCellIndices;
    std::vector<std::uint8_t> faceKinds;
    std::vector<std::uint8_t> targetDDMCEligible;
    std::vector<PointT> outwardNormals;
    std::vector<PointT> faceCenters;
    std::vector<PointT> fluxRhs;
    double minimumParticleOpticalDepth = 0.0;
    bool enabled = false;

    template<typename GridT>
    void Build(const std::vector<CellData<PointT>> &cells,
               const GridT &grid,
               const double minimumOpticalDepth)
    {
        this->cellEligible.resize(cells.size());
        this->sigmaEnergyAbs.resize(cells.size());
        this->sigmaParticleGate.resize(cells.size());
        this->totalLeakRate.resize(cells.size());
        this->leakOffsets.resize(cells.size() + 1);
        this->leakRates.clear();
        this->ddmcLeakRates.clear();
        this->nextCellIndices.clear();
        this->faceKinds.clear();
        this->targetDDMCEligible.clear();
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
                this->outwardNormals.push_back(face.outwardNormal);
                this->faceCenters.push_back(grid.FaceCM(face.faceIndex));
                ++leak;
            }
        }
        this->leakOffsets[cells.size()] = leak;
        this->fluxRhs.assign(cells.size(), PointT{});
        this->minimumParticleOpticalDepth = minimumOpticalDepth;
        this->enabled = true;
    }

    DeviceView<PointT> View()
    {
        DeviceView<PointT> result;
        result.cellEligible = this->cellEligible.data();
        result.sigmaEnergyAbs = this->sigmaEnergyAbs.data();
        result.sigmaParticleGate = this->sigmaParticleGate.data();
        result.totalLeakRate = this->totalLeakRate.data();
        result.leakOffsets = this->leakOffsets.data();
        result.leakRates = this->leakRates.data();
        result.ddmcLeakRates = this->ddmcLeakRates.data();
        result.nextCellIndices = this->nextCellIndices.data();
        result.faceKinds = this->faceKinds.data();
        result.targetDDMCEligible = this->targetDDMCEligible.data();
        result.outwardNormals = this->outwardNormals.data();
        result.faceCenters = this->faceCenters.data();
        result.fluxRhs = this->fluxRhs.data();
        result.cellCount = this->cellEligible.size();
        result.minimumParticleOpticalDepth = this->minimumParticleOpticalDepth;
        result.enabled = this->enabled ? 1u : 0u;
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

    if((radiationFlags & pending) != 0)
    {
        AddFlux(views, cellIndex, cold.pendingFlux);
        cold.pendingFlux = PointT{};
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~pending);
    }

    if(ddmc.cellEligible == nullptr || !ddmc.cellEligible[cellIndex])
    {
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

    const double totalLeakRate = ddmc.totalLeakRate[cellIndex];
    const std::size_t leakBegin = ddmc.leakOffsets[cellIndex];
    const std::size_t leakEnd = ddmc.leakOffsets[cellIndex + 1];
    if(not (totalLeakRate > 0.0) or leakBegin >= leakEnd)
    {
        return result;
    }

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

    result.taken = 1;
    const double tEvent = -transport::Log(CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++)) / totalLeakRate;
    const bool censusEvent = particle.timeLeft <= tEvent;
    const double dt = censusEvent ? particle.timeLeft : tEvent;
    const double absorptionRate = ddmc.sigmaEnergyAbs[cellIndex] * views.fleckFactors[cellIndex] * views.speedOfLight;
    const double oldWeight = particle.weight;
    const double expFactor = transport::Expm1(-dt * absorptionRate);
    const double integratedEnergy =
        absorptionRate > 0.0
            ? oldWeight * expFactor * (-1.0 / absorptionRate)
            : oldWeight * dt;

    if(views.depositMaterialEnergy)
    {
        STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], -expFactor * oldWeight);
    }
    STORM_TRANSPORT_ACCUMULATE(views.pendingRadiationEnergy[cellIndex], integratedEnergy);
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
        }
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~ddmcFlags);
        result.step.change = ParticleStatus::REMOVE;
        result.event = AdvanceEvent::AbsorptionCutoff;
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
        SampleIsotropicDirection<ParticleT, PointT>(particle, views.speedOfLight);
        radiationFlags = static_cast<std::uint8_t>(radiationFlags & ~(ddmcFlags | pending));
        cold.pendingFlux = PointT{};
        result.step.change = ParticleStatus::DONE;
        result.event = AdvanceEvent::Census;
        if(ddmc.censusCount != nullptr)
            STORM_TRANSPORT_ACCUMULATE(*ddmc.censusCount, std::size_t(1));
        return result;
    }

    // Preserve the CPU event stream: grey DDMC has no upscatter channel, but
    // the general event selector still consumes this draw before face choice.
    (void) CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
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

    PointT normal = Normalize(ddmc.outwardNormals[chosen]);
    if(!(Dot(normal, normal) > 0.0))
    {
        result.error = AdvanceError::InvalidNormal;
        return result;
    }
    const double rate = ddmc.leakRates[chosen];
    const bool useDDMCChannel =
        ddmc.ddmcLeakRates[chosen] > 0.0 &&
        CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) <
            ddmc.ddmcLeakRates[chosen] / rate;
    const double mu = useDDMCChannel
        ? SampleAsymptoticMuPortable(CounterRNG::unitOpen(
              particle.rngKey, particle.rngCounter++))
        : transport::Sqrt(CounterRNG::unitOpen(
              particle.rngKey, particle.rngCounter++));
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
    const PointT fluxContribution = Scale(direction, particle.weight);
    AddFlux(views, cellIndex, fluxContribution);

    const bool targetDDMC = useDDMCChannel && ddmc.targetDDMCEligible[chosen];
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
            STORM_TRANSPORT_ACCUMULATE(*ddmc.leakCount, std::size_t(1));
        if(ddmc.transportLeakCount != nullptr)
            STORM_TRANSPORT_ACCUMULATE(
                *ddmc.transportLeakCount, std::size_t(1));
    }

    result.step.change = ParticleStatus::CELL_MOVE;
    result.step.nextCellIndex = ddmc.nextCellIndices[chosen];
    return result;
}

} // namespace STORM::ddmc

#endif
