#ifndef STORM_GPU_GREY_IMC_KERNEL_HPP
#define STORM_GPU_GREY_IMC_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

#include "FlatGridView.hpp"
#include "../radiation/ddmc/AdvanceDDMC.hpp"
#include "../radiation/random_walk/AdvanceRandomWalk.hpp"
#include "../radiation/transport/AdvanceIMC.hpp"

namespace STORM
{
namespace gpu
{

template<typename PointT>
struct GreyIMCViews
{
    using point_type = PointT;

    FlatGridView<PointT> grid;
    const double *absorptionOpacities = nullptr;
    const double *scatteringOpacities = nullptr;
    const double *fleckFactors = nullptr;
    const PointT *cellVelocities = nullptr;
    double *pendingMaterialEnergy = nullptr;
    double *pendingRadiationEnergy = nullptr;
    PointT *pendingMomentum = nullptr;
    ddmc::DeviceView<PointT> ddmc;
    transport::RandomWalkView randomWalk;
    const double *energyBoundaries = nullptr;
    const double *spectralAbsorptionScale = nullptr;
    const double *thermalEmissionCdf = nullptr;
    double *pendingGroupRadiationEnergy = nullptr;
    double *censusRadiationEnergy = nullptr;
    double *censusGroupRadiationEnergy = nullptr;
    std::size_t groupCount = 0;
    double speedOfLight = 0.0;
    std::uint8_t depositMaterialEnergy = 1;
    std::uint8_t comovingTransport = 0;
    std::uint8_t depositMomentum = 0;
    std::uint8_t spectralEnabled = 0;
    std::uint8_t ddmcOnlyTransport = 0;
};

using TransportError = transport::TransportError;
using TransportResult = transport::TransportResult;

STORM_GPU_INLINE_FUNCTION
void AccumulateCensusValue(double *destination, double value)
{
#ifdef STORM_WITH_GPU
    Kokkos::atomic_add(destination, value);
#else
    *destination += value;
#endif
}

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
void AccumulateCensusEnergy(
    const ParticleT &particle,
    const GreyIMCViews<PointT> &views)
{
    const std::size_t cellIndex =
        static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount ||
       views.censusRadiationEnergy == nullptr)
    {
        return;
    }
    AccumulateCensusValue(
        &views.censusRadiationEnergy[cellIndex], particle.weight);
    if(views.spectralEnabled != 0 &&
       views.censusGroupRadiationEnergy != nullptr &&
       views.groupCount > 0)
    {
        const std::size_t group =
            transport::SpectralTableOpacityPolicy{}.FindGroup(
                views, particle.frequency);
        AccumulateCensusValue(
            &views.censusGroupRadiationEnergy[
                cellIndex * views.groupCount + group],
            particle.weight);
    }
}

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
void NudgeTowardCellCenter(ParticleT &particle,
                           const GreyIMCViews<PointT> &views,
                           const std::size_t cellIndex)
{
    const PointT &center = views.grid.cellCenters[cellIndex];
    constexpr double epsilon = 1.0e-8;
    particle.location.x = (1.0 - epsilon) * particle.location.x + epsilon * center.x;
    particle.location.y = (1.0 - epsilon) * particle.location.y + epsilon * center.y;
    particle.location.z = (1.0 - epsilon) * particle.location.z + epsilon * center.z;
}

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
void ApplyDeviceReflect(ParticleT &particle,
                        const GreyIMCViews<PointT> &views,
                        const std::size_t directedFace)
{
    const PointT &normal = views.grid.normals[directedFace];
    const double normalVelocity =
        particle.velocity.x * normal.x +
        particle.velocity.y * normal.y +
        particle.velocity.z * normal.z;
    particle.velocity.x -= 2.0 * normalVelocity * normal.x;
    particle.velocity.y -= 2.0 * normalVelocity * normal.y;
    particle.velocity.z -= 2.0 * normalVelocity * normal.z;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    for(std::uint8_t nudge = 0; nudge < 2; ++nudge)
    {
        NudgeTowardCellCenter(particle, views, cellIndex);
    }
}

// Rank-local CELL_MOVE and reflecting walls stay on the GCD and keep
// transporting. DONE is a timestep terminal: it leaves the active wave but
// stays on device until host Comb. REMOVE, HostOnly boundaries, and MPI rank
// hops still leave the GCD during the loop.
template<typename ParticleT, typename ColdParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
bool TryKeepPacketOnDevice(ParticleT &particle,
                           ColdParticleT &cold,
                           TransportResult &result,
                           const GreyIMCViews<PointT> &views)
{
    if(result.error != TransportError::None)
    {
        return false;
    }
    if(result.step.change == ParticleStatus::NO_CELL_MOVE)
    {
        return true;
    }
    if(result.step.change != ParticleStatus::CELL_MOVE)
    {
        return false;
    }

    const cell_index_t nextCell = result.step.nextCellIndex;
    if(nextCell < views.grid.cellCount)
    {
        particle.cellIndex = nextCell;
        if(views.grid.cellIDs != nullptr)
        {
            cold.cellID =
                views.grid.cellIDs[
                    static_cast<std::size_t>(nextCell)];
        }
        NudgeTowardCellCenter(particle, views, static_cast<std::size_t>(nextCell));
        result.step.change = ParticleStatus::NO_CELL_MOVE;
        return true;
    }

    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount ||
       views.grid.cellFaceOffsets == nullptr ||
       views.grid.nextCellIndices == nullptr)
    {
        return false;
    }
    const std::size_t begin = views.grid.cellFaceOffsets[cellIndex];
    const std::size_t end = views.grid.cellFaceOffsets[cellIndex + 1];
    for(std::size_t directedFace = begin; directedFace < end; ++directedFace)
    {
        if(views.grid.nextCellIndices[directedFace] != nextCell)
        {
            continue;
        }
        const bool reflecting =
            views.grid.deviceBoundaryBehaviors != nullptr &&
            views.grid.deviceBoundaryBehaviors[directedFace] ==
                static_cast<std::uint8_t>(
                    ::STORM::DeviceBoundaryFaceBehavior::ReflectingRigid);
        if(reflecting)
        {
            ApplyDeviceReflect(particle, views, directedFace);
            result.step.change = ParticleStatus::NO_CELL_MOVE;
            return true;
        }
        return false;
    }
    return false;
}

template<typename ParticleT>
STORM_GPU_INLINE_FUNCTION
bool IsCensusTerminal(const ParticleT &,
                      const TransportResult &result)
{
    return result.error == TransportError::None &&
           result.step.change == ParticleStatus::DONE;
}

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
bool IsRankHopTerminal(const ParticleT &particle,
                       const TransportResult &result,
                       const GreyIMCViews<PointT> &views)
{
    if(result.error != TransportError::None ||
       result.step.change != ParticleStatus::CELL_MOVE)
    {
        return false;
    }
    const cell_index_t nextCell = result.step.nextCellIndex;
    if(nextCell < views.grid.cellCount)
    {
        return false;
    }
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount ||
       views.grid.cellFaceOffsets == nullptr ||
       views.grid.nextCellIndices == nullptr)
    {
        return true;
    }
    const std::size_t begin = views.grid.cellFaceOffsets[cellIndex];
    const std::size_t end = views.grid.cellFaceOffsets[cellIndex + 1];
    for(std::size_t directedFace = begin; directedFace < end; ++directedFace)
    {
        if(views.grid.nextCellIndices[directedFace] != nextCell)
        {
            continue;
        }
        return views.grid.boundaryCrossings == nullptr ||
               views.grid.boundaryCrossings[directedFace] == 0;
    }
    return true;
}

template<typename ParticleT, typename ColdT, typename PointT>
STORM_GPU_INLINE_FUNCTION
ddmc::InterfaceResult ApplyDDMCInterface(ParticleT &particle,
                                           ColdT &cold,
                                           TransportResult &result,
                                           const GreyIMCViews<PointT> &views)
{
    ddmc::InterfaceResult interface;
    if(result.error != TransportError::None ||
       result.step.change != ParticleStatus::CELL_MOVE ||
       result.step.boundaryCrossing)
    {
        return interface;
    }
    const std::size_t sourceCell =
        static_cast<std::size_t>(particle.cellIndex);
    interface =
        ddmc::TryIMCToDDMCInterface(
            particle, cold, views, sourceCell,
            result.step.nextCellIndex, result.directedFace);
    if(interface.taken)
    {
        result.step = interface.step;
        result.ddmcExtraSplits = interface.extraSplitCount;
        return interface;
    }
    if(sourceCell < views.ddmc.cellCount &&
       views.ddmc.cellIDs != nullptr &&
       ddmc::BypassCellID(particle, cold) ==
           views.ddmc.cellIDs[sourceCell])
    {
        ddmc::SetBypassCellID(
            particle, cold,
            std::numeric_limits<cell_id_t>::max());
    }
    return interface;
}

template<typename ParticleT, typename ColdT, typename PointT>
STORM_GPU_INLINE_FUNCTION
TransportResult AdvanceOne(ParticleT &particle, ColdT &cold,
                           const GreyIMCViews<PointT> &views)
{
    const std::uint8_t radiationFlags = ddmc::RadiationFlags(particle);
    const bool packetInDDMC =
        (radiationFlags & RadiationTransportState<PointT>::DDMCMode) != 0;
    if(views.ddmcOnlyTransport && !packetInDDMC)
    {
        TransportResult result;
        result.error = TransportError::HostFallback;
        return result;
    }
    if(!packetInDDMC && views.randomWalk.enabled)
    {
        const transport::RandomWalkResult randomWalk =
            transport::TryAdvanceRandomWalk(particle, views);
        if(randomWalk.invalid)
        {
            TransportResult result;
            result.error = TransportError::InvalidOpacity;
            return result;
        }
        if(randomWalk.taken)
            return TransportResult{randomWalk.step, TransportError::None};
    }
    const ddmc::AdvanceResult<PointT> ddmcResult =
        ddmc::AdvanceDDMC(particle, cold, views);
    if(ddmcResult.error == ddmc::AdvanceError::HostFallback)
    {
        TransportResult result;
        result.error = TransportError::HostFallback;
        result.hostFallbackReason = ddmcResult.hostFallbackReason;
        result.pendingLeakFace = ddmcResult.pendingLeakFace;
        return result;
    }
    if(ddmcResult.error != ddmc::AdvanceError::None)
    {
        TransportResult result;
        result.error = TransportError::InvalidOpacity;
        return result;
    }
    if(ddmcResult.taken)
    {
        TransportResult result{ddmcResult.step, TransportError::None};
        result.ddmcExtraSplits = ddmcResult.extraSplitCount;
        if(views.ddmcOnlyTransport &&
           (ddmc::RadiationFlags(particle) &
            RadiationTransportState<PointT>::DDMCMode) == 0)
        {
            result.error = TransportError::HostFallback;
        }
        return result;
    }
    TransportResult result = transport::AdvanceIMC(particle, views);
    ApplyDDMCInterface(particle, cold, result, views);
    return result;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_IMC_KERNEL_HPP
