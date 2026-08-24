#ifndef STORM_GPU_DEVICE_PARTICLE_HPP
#define STORM_GPU_DEVICE_PARTICLE_HPP

#include <cstdint>
#include <type_traits>

#include "KokkosTypes.hpp"
#include "../particle/Particle.hpp"

namespace STORM
{
namespace gpu
{

struct DeviceVec3
{
    using coord_type = double;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    STORM_GPU_INLINE_FUNCTION
    DeviceVec3() = default;

    STORM_GPU_INLINE_FUNCTION
    DeviceVec3(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_)
    {}
};

struct DeviceParticle
{
    particle_id_t id = 0;
    cell_id_t cellID = 0;
    cell_id_t sourceCellID = 0;
    DeviceVec3 location{};
    DeviceVec3 velocity{};
    cell_index_t cellIndex = 0;
    dt_t timeLeft = 0.0;
    double frequency = 0.0;
    double weight = 0.0;
    double initialWeight = 0.0;
    std::uint64_t rngKey = 0;
    std::uint64_t rngCounter = 0;
    std::uint8_t radiationFlags = 0;
    DeviceVec3 pendingFlux{};
    cell_id_t bypassCellID = 0;
#ifdef MONTECARLO_POLARIZATION
    double pendingMeanScatterings = 0.0;
    double stokesQ = 0.0;
    double stokesU = 0.0;
    DeviceVec3 polarizationBasis{};
    std::uint8_t polarizationInitialized = 0;
#endif
    particle_step_t steps = 0;
    std::uint8_t onTrack = 0;
};

static_assert(std::is_trivially_copyable<DeviceVec3>::value, "DeviceVec3 must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceParticle>::value, "DeviceParticle must be trivially copyable");

template<typename PointT>
DeviceParticle PackParticle(const ParticleTransportData<PointT> &source)
{
    DeviceParticle result;
    result.id = source.id;
    result.cellID = source.cellID;
    result.sourceCellID = source.sourceCellID;
    result.location = DeviceVec3(source.location.x, source.location.y, source.location.z);
    result.velocity = DeviceVec3(source.velocity.x, source.velocity.y, source.velocity.z);
    result.cellIndex = source.cellIndex;
    result.timeLeft = source.timeLeft;
    result.frequency = source.frequency;
    result.weight = source.weight;
    result.initialWeight = source.initialWeight;
    result.rngKey = source.rngKey;
    result.rngCounter = source.rngCounter;
    result.radiationFlags = source.radiationState.flags;
    result.pendingFlux = DeviceVec3(source.radiationState.pendingFlux.x, source.radiationState.pendingFlux.y,
                                    source.radiationState.pendingFlux.z);
    result.bypassCellID = source.radiationState.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    result.pendingMeanScatterings = source.radiationState.pendingMeanScatterings;
    result.stokesQ = source.stokesQ;
    result.stokesU = source.stokesU;
    result.polarizationBasis = DeviceVec3(source.polarizationBasis.x, source.polarizationBasis.y,
                                          source.polarizationBasis.z);
    result.polarizationInitialized = source.polarizationInitialized;
#endif
    result.steps = source.steps;
    result.onTrack = source.on_track;
    return result;
}

template<typename PointT>
void UnpackParticle(const DeviceParticle &source, ParticleTransportData<PointT> &destination)
{
    destination.id = source.id;
    destination.cellID = source.cellID;
    destination.sourceCellID = source.sourceCellID;
    destination.location = PointT(source.location.x, source.location.y, source.location.z);
    destination.velocity = PointT(source.velocity.x, source.velocity.y, source.velocity.z);
    destination.cellIndex = source.cellIndex;
    destination.timeLeft = source.timeLeft;
    destination.frequency = source.frequency;
    destination.weight = source.weight;
    destination.initialWeight = source.initialWeight;
    destination.rngKey = source.rngKey;
    destination.rngCounter = source.rngCounter;
    destination.radiationState.flags = source.radiationFlags;
    destination.radiationState.pendingFlux = PointT(source.pendingFlux.x, source.pendingFlux.y, source.pendingFlux.z);
    destination.radiationState.bypassCellID = source.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    destination.radiationState.pendingMeanScatterings = source.pendingMeanScatterings;
    destination.stokesQ = source.stokesQ;
    destination.stokesU = source.stokesU;
    destination.polarizationBasis = PointT(source.polarizationBasis.x, source.polarizationBasis.y,
                                           source.polarizationBasis.z);
    destination.polarizationInitialized = source.polarizationInitialized;
#endif
    destination.steps = source.steps;
    destination.on_track = source.onTrack;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_DEVICE_PARTICLE_HPP
