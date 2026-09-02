#ifndef STORM_GPU_DEVICE_PARTICLE_HPP
#define STORM_GPU_DEVICE_PARTICLE_HPP

#include <cstdint>
#include <limits>
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
    DeviceVec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_)
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
    rank_t rank = 0;
    cell_id_t cellID = 0;
    cell_id_t sourceCellID = 0;
    DeviceVec3 pendingFlux{};
    cell_id_t bypassCellID = std::numeric_limits<cell_id_t>::max();
#ifdef MONTECARLO_POLARIZATION
    double pendingMeanScatterings = 0.0;
    double stokesQ = 0.0;
    double stokesU = 0.0;
    DeviceVec3 polarizationBasis{};
    std::uint8_t polarizationInitialized = 0;
#endif
    std::uint8_t onTrack = 0;
};

// Whole-struct assignment lowers to a 32-byte memcpy covering pendingFlux and
// its neighbouring field. In IMC-only kernels nothing reads those members, so
// SROA cannot split the slice and leaves it in memory; the backend then
// promotes it to LDS and caps occupancy. Copying member by member keeps the
// whole struct in registers.
STORM_GPU_INLINE_FUNCTION
void AssignCold(DeviceParticleCold &destination, const DeviceParticleCold &source)
{
    destination.id = source.id;
    destination.rank = source.rank;
    destination.cellID = source.cellID;
    destination.sourceCellID = source.sourceCellID;
    destination.pendingFlux.x = source.pendingFlux.x;
    destination.pendingFlux.y = source.pendingFlux.y;
    destination.pendingFlux.z = source.pendingFlux.z;
    destination.bypassCellID = source.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    destination.pendingMeanScatterings = source.pendingMeanScatterings;
    destination.stokesQ = source.stokesQ;
    destination.stokesU = source.stokesU;
    destination.polarizationBasis.x = source.polarizationBasis.x;
    destination.polarizationBasis.y = source.polarizationBasis.y;
    destination.polarizationBasis.z = source.polarizationBasis.z;
    destination.polarizationInitialized = source.polarizationInitialized;
#endif
    destination.onTrack = source.onTrack;
}

namespace detail
{
// Mirrors DeviceParticleCold. Adding a member there without adding it here (and
// to AssignCold) changes the size and trips the assertion below.
struct DeviceParticleColdLayout
{
    particle_id_t id;
    rank_t rank;
    cell_id_t cellID;
    cell_id_t sourceCellID;
    DeviceVec3 pendingFlux;
    cell_id_t bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    double pendingMeanScatterings;
    double stokesQ;
    double stokesU;
    DeviceVec3 polarizationBasis;
    std::uint8_t polarizationInitialized;
#endif
    std::uint8_t onTrack;
};
} // namespace detail

static_assert(sizeof(DeviceParticleCold) == sizeof(detail::DeviceParticleColdLayout),
              "DeviceParticleCold gained a member; add it to AssignCold and to "
              "detail::DeviceParticleColdLayout");

static_assert(std::is_trivially_copyable<DeviceVec3>::value, "DeviceVec3 must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceParticle>::value, "DeviceParticle must be trivially copyable");
static_assert(std::is_trivially_copyable<DeviceParticleCold>::value, "DeviceParticleCold must be trivially copyable");

template<typename PointT>
void PackParticle(const ParticleTransportData<PointT> &source, DeviceParticle &particle, DeviceParticleCold &cold)
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
    cold.pendingFlux = DeviceVec3(source.radiationState.pendingFlux.x, source.radiationState.pendingFlux.y, source.radiationState.pendingFlux.z);
    cold.bypassCellID = source.radiationState.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    cold.pendingMeanScatterings = source.radiationState.pendingMeanScatterings;
    cold.stokesQ = source.stokesQ;
    cold.stokesU = source.stokesU;
    cold.polarizationBasis = DeviceVec3(source.polarizationBasis.x, source.polarizationBasis.y, source.polarizationBasis.z);
    cold.polarizationInitialized = source.polarizationInitialized;
#endif
    cold.onTrack = source.on_track;
}

template<typename PointT>
void UnpackParticle(const DeviceParticle &particle, const DeviceParticleCold &cold, ParticleTransportData<PointT> &destination)
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
    destination.radiationState.pendingFlux = PointT(cold.pendingFlux.x, cold.pendingFlux.y, cold.pendingFlux.z);
    destination.radiationState.bypassCellID = cold.bypassCellID;
#ifdef MONTECARLO_POLARIZATION
    destination.radiationState.pendingMeanScatterings = cold.pendingMeanScatterings;
    destination.stokesQ = cold.stokesQ;
    destination.stokesU = cold.stokesU;
    destination.polarizationBasis = PointT(cold.polarizationBasis.x, cold.polarizationBasis.y, cold.polarizationBasis.z);
    destination.polarizationInitialized = cold.polarizationInitialized;
#endif
    destination.steps = particle.steps;
    destination.on_track = cold.onTrack;
}

template<typename PointT>
void PackParticle(
    const Particle<PointT> &source,
    DeviceParticle &particle,
    DeviceParticleCold &cold)
{
    PackParticle(
        static_cast<const ParticleTransportData<PointT> &>(source),
        particle, cold);
#ifdef STORM_WITH_MPI
    cold.rank = source.rank;
#endif
}

template<typename PointT>
void UnpackParticle(
    const DeviceParticle &particle,
    const DeviceParticleCold &cold,
    Particle<PointT> &destination)
{
    UnpackParticle(
        particle, cold,
        static_cast<ParticleTransportData<PointT> &>(destination));
#ifdef STORM_WITH_MPI
    destination.rank = cold.rank;
#endif
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_DEVICE_PARTICLE_HPP
