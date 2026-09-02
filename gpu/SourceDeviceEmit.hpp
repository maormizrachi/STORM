#ifndef STORM_GPU_SOURCE_DEVICE_EMIT_HPP
#define STORM_GPU_SOURCE_DEVICE_EMIT_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../radiation/source/SourceCore.hpp"
#include "../utils/CounterRNG.hpp"
#include "DeviceSourceContext.hpp"
#include "GreyIMCData.hpp"
#include "KokkosTypes.hpp"

namespace STORM
{
namespace gpu
{

inline void EmitSourcesOnDevice(DeviceSourceContext &context)
{
    if(context.executor == nullptr)
    {
        throw std::logic_error(
            "Device source context has no executor");
    }
    if(context.gpuData == nullptr)
    {
        throw std::logic_error(
            "Device source context has no GPU data");
    }
    if(context.plan == nullptr)
    {
        throw std::logic_error(
            "Device source context has no source plan");
    }

    const source::Plan &plan = *context.plan;
    context.emittedCount = plan.totalPhotons;
    if(plan.totalPhotons == 0)
    {
        return;
    }

    KokkosLocalTransportExecutor &executor = *context.executor;
    GreyIMCData &gpuData = *context.gpuData;
    const std::size_t offset =
        executor.AllocateActiveSlots(plan.totalPhotons);

    Kokkos::View<std::size_t *> photonOffsets(
        "storm_source_photon_offsets", plan.photonOffsets.size());
    Kokkos::View<double *> energyPerPhoton(
        "storm_source_energy_per_photon", plan.energyPerPhoton.size());
    auto hostOffsets = Kokkos::create_mirror_view(photonOffsets);
    auto hostEnergy = Kokkos::create_mirror_view(energyPerPhoton);
    for(std::size_t i = 0; i < plan.photonOffsets.size(); ++i)
    {
        hostOffsets(i) = plan.photonOffsets[i];
    }
    for(std::size_t i = 0; i < plan.energyPerPhoton.size(); ++i)
    {
        hostEnergy(i) = plan.energyPerPhoton[i];
    }
    Kokkos::deep_copy(photonOffsets, hostOffsets);
    Kokkos::deep_copy(energyPerPhoton, hostEnergy);

    const GreyIMCViews<DeviceVec3> views = gpuData.Views(
        context.speedOfLight,
        false,
        context.applyLabFrame != 0,
        false);

    source::SampleViews<DeviceVec3> sampleViews;
    sampleViews.tetOffsets = gpuData.SourceTetOffsets();
    sampleViews.tetCumVolumes = gpuData.SourceTetCumVolumes();
    sampleViews.tetTris = gpuData.SourceTetTris();
    sampleViews.vertices = gpuData.SourceVertices();
    sampleViews.cellCenters = views.grid.cellCenters;
    sampleViews.cellIDs = views.grid.cellIDs;
    sampleViews.thermalEmissionCdf = views.thermalEmissionCdf;
    sampleViews.energyBoundaries = views.energyBoundaries;
    sampleViews.cellVelocities = views.cellVelocities;
    sampleViews.cellCount = views.grid.cellCount;
    sampleViews.groupCount = views.groupCount;
    sampleViews.speedOfLight = context.speedOfLight;
    sampleViews.invClight2 = context.invClight2;
    sampleViews.sampleFrequency = context.sampleFrequency;
    sampleViews.applyLabFrame = context.applyLabFrame;

    auto packets = executor.ActivePackets();
    auto coldPackets = executor.ActiveColdPackets();
    const std::size_t cellCount = plan.nPhotons.size();
    const std::size_t totalPhotons = plan.totalPhotons;
    const std::uint64_t rngSeed = context.particleRngSeed;
    const std::uint64_t creationRank = context.creationRank;
    const std::uint64_t rngStreamBase = plan.rngStreamBase;
    const particle_id_t firstId = context.firstParticleId;
#ifdef STORM_WITH_MPI
    const rank_t rank = context.rank;
#endif
    const dt_t fullDt = context.fullDt;

    Kokkos::parallel_for(
        "storm_emit_thermal_sources",
        Kokkos::RangePolicy<>(0, totalPhotons),
        KOKKOS_LAMBDA(const std::size_t slot)
        {
            const std::size_t cellIndex = source::CellFromPhotonSlot(
                photonOffsets.data(), cellCount, slot);
            const std::uint64_t rngKey = source::MakeSourceRngKey(
                rngSeed, creationRank, rngStreamBase + slot);
            DeviceVec3 location{};
            DeviceVec3 velocity{};
            source::EmittedScalars scalars;
            source::EmitThermalPacket(
                sampleViews,
                cellIndex,
                rngKey,
                energyPerPhoton(cellIndex),
                location,
                velocity,
                scalars);

            DeviceParticle particle;
            particle.location = location;
            particle.velocity = velocity;
            particle.cellIndex = scalars.cellIndex;
            particle.timeLeft = fullDt;
            particle.weight = scalars.weight;
            particle.initialWeight = scalars.initialWeight;
            particle.frequency = scalars.frequency;
            particle.rngKey = scalars.rngKey;
            particle.rngCounter = scalars.rngCounter;
            particle.steps = 0;
            particle.radiationFlags = 0;

            DeviceParticleCold cold;
            cold.id = firstId + static_cast<particle_id_t>(slot);
#ifdef STORM_WITH_MPI
            cold.rank = rank;
#endif
            cold.cellID = scalars.cellID;
            cold.sourceCellID = scalars.cellID;

            packets(offset + slot) = particle;
            coldPackets(offset + slot) = cold;
        });
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_SOURCE_DEVICE_EMIT_HPP
