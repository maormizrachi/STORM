#ifndef STORM_GPU_GREY_IMC_KERNEL_HPP
#define STORM_GPU_GREY_IMC_KERNEL_HPP

#include <cstddef>
#include <cstdint>

#include "FlatGridView.hpp"
#include "GreyRandomWalkKernel.hpp"
#include "../radiation/transport/AdvanceIMC.hpp"

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
    const PointT *cellVelocities = nullptr;
    double *pendingMaterialEnergy = nullptr;
    double *pendingRadiationEnergy = nullptr;
    PointT *pendingMomentum = nullptr;
    GreyRandomWalkView randomWalk;
    const double *energyBoundaries = nullptr;
    const double *spectralAbsorptionScale = nullptr;
    const double *thermalEmissionCdf = nullptr;
    double *pendingGroupRadiationEnergy = nullptr;
    std::size_t groupCount = 0;
    double speedOfLight = 0.0;
    std::uint8_t depositMaterialEnergy = 1;
    std::uint8_t comovingTransport = 0;
    std::uint8_t depositMomentum = 0;
    std::uint8_t spectralEnabled = 0;
};

using TransportError = transport::TransportError;
using TransportResult = transport::TransportResult;

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
TransportResult AdvanceOne(ParticleT &particle, const GreyIMCViews<PointT> &views)
{
    return transport::AdvanceIMC(particle, views);
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_IMC_KERNEL_HPP
