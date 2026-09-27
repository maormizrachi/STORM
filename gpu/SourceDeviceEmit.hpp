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
    const std::size_t entryCount = plan.entryCount();
    context.emittedCount = plan.totalPhotons;
    context.emittedFourMomentum.assign(4 * entryCount, 0.0);
    if(plan.totalPhotons == 0)
    {
        return;
    }

    KokkosLocalTransportExecutor &executor = *context.executor;
    GreyIMCData &gpuData = *context.gpuData;
    const std::size_t offset =
        executor.AllocateActiveSlots(plan.totalPhotons);

    Kokkos::View<std::size_t *> entryOffsets(
        "storm_source_entry_offsets", entryCount + 1);
    Kokkos::View<std::size_t *> entryCell(
        "storm_source_entry_cell", entryCount);
    Kokkos::View<double *> energyPerPhoton(
        "storm_source_energy_per_photon", entryCount);
    Kokkos::View<double *> fixedFrequency(
        "storm_source_fixed_frequency", plan.fixedFrequencies ? entryCount : 0);
    auto hostOffsets = Kokkos::create_mirror_view(entryOffsets);
    auto hostCell = Kokkos::create_mirror_view(entryCell);
    auto hostEnergy = Kokkos::create_mirror_view(energyPerPhoton);
    auto hostFrequency = Kokkos::create_mirror_view(fixedFrequency);
    for(std::size_t entry = 0; entry <= entryCount; ++entry)
    {
        hostOffsets(entry) = plan.entryOffsets[entry];
    }
    for(std::size_t entry = 0; entry < entryCount; ++entry)
    {
        hostCell(entry) = plan.entryCell[entry];
        hostEnergy(entry) = plan.energyPerPhoton[entry];
        if(plan.fixedFrequencies)
        {
            hostFrequency(entry) = plan.fixedFrequency[entry];
        }
    }
    Kokkos::deep_copy(entryOffsets, hostOffsets);
    Kokkos::deep_copy(entryCell, hostCell);
    Kokkos::deep_copy(energyPerPhoton, hostEnergy);
    Kokkos::deep_copy(fixedFrequency, hostFrequency);
    const bool fixedFrequencies = plan.fixedFrequencies;

    const GreyIMCViews<DeviceVec3> views = gpuData.Views(
        context.speedOfLight,
        false,
        context.applyLabFrame != 0,
        false,
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
    sampleViews.thermalKT = views.thermalKT;
    sampleViews.thermalFrequencyLaw = views.thermalFrequencyLaw;
    sampleViews.fullDt = context.fullDt;
    sampleViews.cellCount = views.grid.cellCount;
    sampleViews.groupCount = views.groupCount;
    sampleViews.speedOfLight = context.speedOfLight;
    sampleViews.invClight2 = context.invClight2;
    sampleViews.sampleFrequency = context.sampleFrequency;
    sampleViews.applyLabFrame = context.applyLabFrame;
    sampleViews.clampLabFrequency = plan.clampLabFrequency ? 1 : 0;

    auto packets = executor.ActivePackets();
    auto coldPackets = executor.ActiveColdPackets();
    const std::size_t totalPhotons = plan.totalPhotons;
    const std::uint64_t rngSeed = context.particleRngSeed;
    const std::uint64_t creationRank = context.creationRank;
    const std::uint64_t rngStreamBase = plan.rngStreamBase;
    const particle_id_t firstId = context.firstParticleId;
#ifdef STORM_WITH_MPI
    const rank_t rank = context.rank;
#endif

    Kokkos::parallel_for(
        "storm_emit_sources",
        Kokkos::RangePolicy<>(0, totalPhotons),
        KOKKOS_LAMBDA(const std::size_t slot)
        {
            const std::size_t entry = source::EntryFromPhotonSlot(
                entryOffsets.data(), entryCount, slot);
            const std::size_t cellIndex = entryCell(entry);
            const std::uint64_t rngKey = source::MakeSourceRngKey(
                rngSeed, creationRank, rngStreamBase + slot);
            DeviceVec3 location{};
            DeviceVec3 velocity{};
            source::EmittedScalars scalars;
            source::EmitSourcePacket(
                sampleViews,
                cellIndex,
                rngKey,
                energyPerPhoton(entry),
                fixedFrequencies ? &fixedFrequency(entry) : nullptr,
                location,
                velocity,
                scalars);

            DeviceParticle particle;
            particle.location = location;
            particle.velocity = velocity;
            particle.cellIndex = scalars.cellIndex;
            particle.timeLeft = scalars.timeLeft;
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

    // Sum each entry's packets in slot order (no atomics), so the material
    // debit is bitwise reproducible.
    Kokkos::View<double *> fourMomentum(
        "storm_source_four_momentum", 4 * entryCount);
    const double invClight2 = context.invClight2;
    Kokkos::parallel_for(
        "storm_sum_source_four_momentum",
        Kokkos::RangePolicy<>(0, entryCount),
        KOKKOS_LAMBDA(const std::size_t entry)
        {
            double energy = 0.0;
            DeviceVec3 momentum{};
            for(std::size_t slot = entryOffsets(entry); slot < entryOffsets(entry + 1); ++slot)
            {
                const DeviceParticle &particle = packets(offset + slot);
                source::AddPacketFourMomentum(particle.weight, particle.velocity, invClight2, energy, momentum);
            }
            fourMomentum(4 * entry) = energy;
            fourMomentum(4 * entry + 1) = momentum.x;
            fourMomentum(4 * entry + 2) = momentum.y;
            fourMomentum(4 * entry + 3) = momentum.z;
        });
    auto hostFourMomentum = Kokkos::create_mirror_view(fourMomentum);
    Kokkos::deep_copy(hostFourMomentum, fourMomentum);
    for(std::size_t i = 0; i < context.emittedFourMomentum.size(); ++i)
    {
        context.emittedFourMomentum[i] = hostFourMomentum(i);
    }
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_SOURCE_DEVICE_EMIT_HPP
