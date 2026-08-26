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
    DeviceVec3 location{};
    DeviceVec3 velocity{};
    cell_index_t cellIndex = 0;
    dt_t timeLeft = 0.0;
    double weight = 0.0;
    double initialWeight = 0.0;
    double frequency = 0.0;
    std::uint64_t rngKey = 0;
    std::uint64_t rngCounter = 0;
    particle_step_t steps = 0;
    std::uint8_t radiationFlags = 0;
};

// State that IMC transport does not inspect every event. Keep it in a parallel
// array so the hot kernel only loads and writes DeviceParticle.
struct DeviceParticleCold
{
    particle_id_t id = 0;
    cell_id_t cellID = 0;
    cell_id_t sourceCellID = 0;
    DeviceVec3 pendingFlux{};
    cell_id_t bypassCellID = 0;
#ifdef MONTECARLO_POLARIZATION
    double pendingMeanScatterings = 0.0;
    double stokesQ = 0.0;
    double stokesU = 0.0;
    DeviceVec3 polarizationBasis{};
    std::uint8_t polarizationInitialized = 0;
#endif
    std::uint8_t onTrack = 0;
};

static_assert(std::is_trivially_copyable<DeviceVec3>::value, "DeviceVec3 must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceParticle>::value, "DeviceParticle must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceParticleCold>::value,
              "DeviceParticleCold must be trivially copyable");

template<typename PointT>
void PackParticle(const ParticleTransportData<PointT> &source,
                  DeviceParticle &particle,
                  DeviceParticleCold &cold)
{
    particle.location = DeviceVec3(source.location.x, source.location.y, source.location.z);
    particle.velocity = DeviceVec3(source.velocity.x, source.velocity.y, source.velocity.z);
    particle.cellIndex = source.cellIndex;
    particle.timeLeft = source.timeLeft;
    particle.weight = source.weight;
    particle.initialWeight = source.initialWeight;
    particle.frequency = source.frequency;
    particle.rngKey = source.rngKey;
    particle.rngCounter = source.rngCounter;
    particle.steps = source.steps;
    particle.radiationFlags = source.radiationState.flags;

    cold.id = source.id;
    cold.cellID = source.cellID;
    cold.sourceCellID = source.sourceCellID;
    cold.pendingFlux = DeviceVec3(source.radiationState.pendingFlux.x,
                                 source.radiationState.pendingFlux.y,
                                 source.radiationState.pendingFlux.z);
    cold.bypassCellID = source.radiationState.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    cold.pendingMeanScatterings = source.radiationState.pendingMeanScatterings;
    cold.stokesQ = source.stokesQ;
    cold.stokesU = source.stokesU;
    cold.polarizationBasis = DeviceVec3(source.polarizationBasis.x,
                                        source.polarizationBasis.y,
                                        source.polarizationBasis.z);
    cold.polarizationInitialized = source.polarizationInitialized;
#endif
    cold.onTrack = source.on_track;
}

template<typename PointT>
void UnpackParticle(const DeviceParticle &particle,
                    const DeviceParticleCold &cold,
                    ParticleTransportData<PointT> &destination)
{
    destination.id = cold.id;
    destination.cellID = cold.cellID;
    destination.sourceCellID = cold.sourceCellID;
    destination.location = PointT(particle.location.x, particle.location.y, particle.location.z);
    destination.velocity = PointT(particle.velocity.x, particle.velocity.y, particle.velocity.z);
    destination.cellIndex = particle.cellIndex;
    destination.timeLeft = particle.timeLeft;
    destination.frequency = particle.frequency;
    destination.weight = particle.weight;
    destination.initialWeight = particle.initialWeight;
    destination.rngKey = particle.rngKey;
    destination.rngCounter = particle.rngCounter;
    destination.radiationState.flags = particle.radiationFlags;
    destination.radiationState.pendingFlux =
        PointT(cold.pendingFlux.x, cold.pendingFlux.y, cold.pendingFlux.z);
    destination.radiationState.bypassCellID = cold.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    destination.radiationState.pendingMeanScatterings = cold.pendingMeanScatterings;
    destination.stokesQ = cold.stokesQ;
    destination.stokesU = cold.stokesU;
    destination.polarizationBasis = PointT(cold.polarizationBasis.x,
                                           cold.polarizationBasis.y,
                                           cold.polarizationBasis.z);
    destination.polarizationInitialized = cold.polarizationInitialized;
#endif
    destination.steps = particle.steps;
    destination.on_track = cold.onTrack;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_DEVICE_PARTICLE_HPP
