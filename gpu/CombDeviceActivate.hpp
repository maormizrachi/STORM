#ifndef STORM_GPU_COMB_DEVICE_ACTIVATE_HPP
#define STORM_GPU_COMB_DEVICE_ACTIVATE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include "../population/CombCore.hpp"
#include "../utils/CounterRNG.hpp"
#include "DevicePopulationContext.hpp"
#include "KokkosTypes.hpp"
#include "ProfileRegion.hpp"

namespace STORM
{
namespace gpu
{

struct DeviceCombEmitter
{
    const DeviceParticle *censusPackets = nullptr;
    const DeviceParticleCold *censusCold = nullptr;
    DeviceParticle *outputPackets = nullptr;
    DeviceParticleCold *outputCold = nullptr;
    std::size_t *outputIndex = nullptr;
    std::uint64_t binRngKey = 0;

    STORM_GPU_INLINE_FUNCTION
    void operator()(const std::size_t sourceIndex, const double weight, const double initialWeight, const std::size_t cellIndex, const bool resetIdentity) const
    {
        DeviceParticle particle = this->censusPackets[sourceIndex];
        DeviceParticleCold cold = this->censusCold[sourceIndex];
        particle.weight = weight;
        particle.initialWeight = initialWeight;
        particle.cellIndex = static_cast<cell_index_t>(cellIndex);
        particle.timeLeft = 0.0;
        particle.steps = 0;
        if(resetIdentity)
        {
            comb::RekeyClone(particle, this->binRngKey, *this->outputIndex);
            cold.id = std::numeric_limits<particle_id_t>::max();
            cold.rank = std::numeric_limits<rank_t>::max();
        }
        const std::size_t writeIndex = (*this->outputIndex)++;
        this->outputPackets[writeIndex] = particle;
        this->outputCold[writeIndex] = cold;
    }
};

inline void ActivateCombOnDeviceCensus(DevicePopulationContext &context)
{
    STORM_PROFILE_REGION("storm/comb");
    if(context.executor == nullptr)
    {
        throw std::logic_error("Device population control context has no executor");
    }
    KokkosLocalTransportExecutor &executor = *context.executor;
    const std::size_t censusCount = executor.PendingCensusCount();

    // Ranks with an empty census must still reach the reductions below, so
    // there is no early exit here: every rank that enters this function has to
    // take part in the same collectives.
    const comb::Parameters &parameters = context.parameters;
    const std::size_t cellCount = context.cellCount;
    const std::uint64_t activationEpoch = context.activationEpoch;
    const rank_t rank = context.rank;

    Kokkos::View<std::size_t*> cellCounts("storm_comb_cell_counts", cellCount);
    Kokkos::View<double*> cellWeights("storm_comb_cell_weights", cellCount);
    Kokkos::View<std::size_t*> sortedIndices("storm_comb_sorted_indices", censusCount);
    Kokkos::View<std::size_t*> cellOffsets("storm_comb_cell_offsets", cellCount + 1);
    Kokkos::View<std::size_t*> cellWriteCursor("storm_comb_cell_write_cursor", cellCount);
    Kokkos::View<std::size_t*> outputCounts("storm_comb_output_counts", cellCount);
    Kokkos::View<std::size_t*> outputOffsets("storm_comb_output_offsets", cellCount + 1);
    Kokkos::View<std::size_t*> rankKeys("storm_comb_rank_keys", censusCount);
    Kokkos::View<particle_id_t*> idKeys("storm_comb_id_keys", censusCount);
    Kokkos::View<double*> combOffsets("storm_comb_offsets", cellCount);
    Kokkos::View<std::size_t*> targets("storm_comb_targets", cellCount);

    Kokkos::deep_copy(cellCounts, std::size_t(0));
    Kokkos::deep_copy(cellWeights, 0.0);

    Kokkos::View<DeviceParticle*> censusPackets = executor.PendingCensusPackets();
    Kokkos::View<DeviceParticleCold*> censusCold = executor.PendingCensusCold();
    Kokkos::parallel_for("storm_comb_accumulate_cell_weights",
        Kokkos::RangePolicy<>(0, censusCount),
        KOKKOS_LAMBDA(const std::size_t i)
        {
            const std::size_t cellIndex = static_cast<std::size_t>(censusPackets(i).cellIndex);
            if(cellIndex >= cellCount)
            {
                return;
            }
            const double weight = censusPackets(i).weight;
            if(weight <= 0.0)
            {
                return;
            }
            Kokkos::atomic_add(&cellWeights(cellIndex), weight);
            Kokkos::atomic_increment(&cellCounts(cellIndex));
        });

    Kokkos::View<double *>::HostMirror hostCellWeights = Kokkos::create_mirror_view(cellWeights);
    Kokkos::View<std::size_t *>::HostMirror hostCellCounts = Kokkos::create_mirror_view(cellCounts);
    Kokkos::deep_copy(hostCellWeights, cellWeights);
    Kokkos::deep_copy(hostCellCounts, cellCounts);

    double localTotalWeight = 0.0;
    for(std::size_t cell = 0; cell < cellCount; ++cell)
    {
        localTotalWeight += hostCellWeights(cell);
    }

    std::size_t globalCellCount = cellCount;
    double globalTotalWeight = localTotalWeight;
#ifdef STORM_WITH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized && context.communicator != MPI_COMM_NULL)
    {
        MPI_Allreduce(MPI_IN_PLACE, &globalCellCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, context.communicator);
        MPI_Allreduce(MPI_IN_PLACE, &globalTotalWeight, 1, MPI_DOUBLE, MPI_SUM, context.communicator);
    }
#endif
    if(globalTotalWeight <= 0.0)
    {
        executor.ClearPendingCensus();
        return;
    }

    const std::size_t globalBudget = comb::GlobalBudget(globalCellCount, parameters);
    Kokkos::View<std::size_t*>::HostMirror hostTargets = Kokkos::create_mirror_view(targets);
    Kokkos::View<double *>::HostMirror hostCombOffsets = Kokkos::create_mirror_view(combOffsets);
    for(std::size_t cell = 0; cell < cellCount; ++cell)
    {
        hostTargets(cell) = comb::TargetParticleCount(hostCellWeights(cell), globalTotalWeight, globalBudget, parameters.Nmin);
        const std::uint64_t rngKey = comb::MakeBinRngKey(activationEpoch, static_cast<std::uint64_t>(rank), cell);
        hostCombOffsets(cell) = CounterRNG::unitOpen(rngKey, 0);
    }
    Kokkos::deep_copy(targets, hostTargets);
    Kokkos::deep_copy(combOffsets, hostCombOffsets);

    Kokkos::parallel_for("storm_comb_prefix_cell_counts",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int)
        {
            cellOffsets(0) = 0;
            for(std::size_t cell = 0; cell < cellCount; ++cell)
            {
                cellOffsets(cell + 1) = cellOffsets(cell) + cellCounts(cell);
            }
        });
    Kokkos::deep_copy(cellWriteCursor,
        Kokkos::subview(cellOffsets, std::pair<std::size_t, std::size_t>(0, cellCount)));

    Kokkos::parallel_for("storm_comb_scatter_by_cell",
        Kokkos::RangePolicy<>(0, censusCount),
        KOKKOS_LAMBDA(const std::size_t i)
        {
            const std::size_t cellIndex = static_cast<std::size_t>(censusPackets(i).cellIndex);
            if(cellIndex >= cellCount)
            {
                return;
            }
            const double weight = censusPackets(i).weight;
            if(weight <= 0.0)
            {
                return;
            }
            const std::size_t writeIndex = Kokkos::atomic_fetch_add(&cellWriteCursor(cellIndex), static_cast<std::size_t>(1));
            sortedIndices(writeIndex) = i;
            rankKeys(writeIndex) = static_cast<std::size_t>(censusCold(i).rank);
            idKeys(writeIndex) = censusCold(i).id;
        });

    // Order every cell by ascending (rank, id) and then apply the seeded
    // Fisher-Yates shuffle, so the survivors picked below do not depend on the
    // order packets happened to arrive in. This mirrors the host Comb
    // reference and runs on the host: a per-cell sort on the device gets one
    // thread per cell, which is quadratic in the packets of the busiest cell.
    {
        Kokkos::View<std::size_t *>::HostMirror hostSortedIndices = Kokkos::create_mirror_view(sortedIndices);
        Kokkos::View<std::size_t *>::HostMirror hostRankKeys = Kokkos::create_mirror_view(rankKeys);
        Kokkos::View<particle_id_t *>::HostMirror hostIDKeys = Kokkos::create_mirror_view(idKeys);
        Kokkos::View<std::size_t *>::HostMirror hostCellOffsets = Kokkos::create_mirror_view(cellOffsets);
        Kokkos::deep_copy(hostSortedIndices, sortedIndices);
        Kokkos::deep_copy(hostRankKeys, rankKeys);
        Kokkos::deep_copy(hostIDKeys, idKeys);
        Kokkos::deep_copy(hostCellOffsets, cellOffsets);

        std::vector<std::size_t> order;
        std::vector<std::size_t> reordered;
        for(std::size_t cell = 0; cell < cellCount; ++cell)
        {
            const std::size_t begin = hostCellOffsets(cell);
            const std::size_t count = hostCellOffsets(cell + 1) - begin;
            if(count == 0)
            {
                continue;
            }
            order.resize(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(),
                [&](const std::size_t left, const std::size_t right)
                {
                    return comb::LessParticleKey(
                        static_cast<rank_t>(hostRankKeys(begin + left)),
                        hostIDKeys(begin + left),
                        static_cast<rank_t>(hostRankKeys(begin + right)),
                        hostIDKeys(begin + right));
                });
            reordered.resize(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                reordered[i] = hostSortedIndices(begin + order[i]);
            }
            const std::uint64_t rngKey = comb::MakeBinRngKey(activationEpoch, static_cast<std::uint64_t>(rank), cell);
            comb::FisherYatesShuffle(reordered.data(), count, rngKey);
            for(std::size_t i = 0; i < count; ++i)
            {
                hostSortedIndices(begin + i) = reordered[i];
            }
        }
        Kokkos::deep_copy(sortedIndices, hostSortedIndices);
    }

    Kokkos::View<double*> particleWeights("storm_comb_particle_weights", censusCount);
    Kokkos::parallel_for("storm_comb_gather_weights",
        Kokkos::RangePolicy<>(0, censusCount),
        KOKKOS_LAMBDA(const std::size_t i)
        {
            particleWeights(i) = censusPackets(i).weight;
        });

    Kokkos::parallel_for("storm_comb_count_outputs",
        Kokkos::RangePolicy<>(0, cellCount),
        KOKKOS_LAMBDA(const std::size_t cell)
        {
            const std::size_t begin = cellOffsets(cell);
            const std::size_t end = cellOffsets(cell + 1);
            const std::size_t count = end - begin;
            outputCounts(cell) = comb::CountBin(particleWeights.data(), sortedIndices.data() + begin, count, cellWeights(cell), targets(cell), combOffsets(cell));
        });

    Kokkos::parallel_for("storm_comb_prefix_output_counts",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int)
        {
            outputOffsets(0) = 0;
            for(std::size_t cell = 0; cell < cellCount; ++cell)
            {
                outputOffsets(cell + 1) = outputOffsets(cell) + outputCounts(cell);
            }
        });

    Kokkos::View<std::size_t*>::HostMirror hostOutputOffsets = Kokkos::create_mirror_view(outputOffsets);
    Kokkos::deep_copy(hostOutputOffsets, outputOffsets);
    const std::size_t outputCount = hostOutputOffsets(cellCount);

    Kokkos::View<DeviceParticle*> outputPackets("storm_comb_output_packets", outputCount);
    Kokkos::View<DeviceParticleCold*> outputCold("storm_comb_output_cold", outputCount);
    Kokkos::View<std::size_t*> cellEmitCursor("storm_comb_emit_cursor", cellCount);
    Kokkos::deep_copy(cellEmitCursor,
        Kokkos::subview(outputOffsets, std::pair<std::size_t, std::size_t>(0, cellCount)));

    Kokkos::parallel_for("storm_comb_emit_outputs",
        Kokkos::RangePolicy<>(0, cellCount),
        KOKKOS_LAMBDA(const std::size_t cell)
        {
            const std::size_t begin = cellOffsets(cell);
            const std::size_t end = cellOffsets(cell + 1);
            const std::size_t count = end - begin;
            std::size_t outputIndex = outputOffsets(cell);
            DeviceCombEmitter emitter{
                censusPackets.data(),
                censusCold.data(),
                outputPackets.data(),
                outputCold.data(),
                &outputIndex,
                comb::MakeBinRngKey(activationEpoch, static_cast<std::uint64_t>(rank), cell)};
            comb::EmitBin(sortedIndices.data() + begin, particleWeights.data(), count, cellWeights(cell), targets(cell), cell, combOffsets(cell), emitter);
            cellEmitCursor(cell) = outputIndex;
        });

    executor.ReplacePendingCensus(outputPackets, outputCold, outputCount);
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_COMB_DEVICE_ACTIVATE_HPP
