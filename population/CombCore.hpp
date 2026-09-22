#ifndef STORM_COMB_CORE_HPP
#define STORM_COMB_CORE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../utils/CounterRNG.hpp"
#include "../types.hpp"

#ifdef STORM_WITH_GPU
#include "../gpu/KokkosTypes.hpp"
#define STORM_COMB_INLINE STORM_GPU_INLINE_FUNCTION
#else
#define STORM_COMB_INLINE inline
#endif

namespace STORM
{
namespace comb
{

struct Parameters
{
    std::size_t Nmin = 20;
    double totalParticlesFactor = 2.0;
    // Per-cell floor and cap. Zero keeps them tied to Nmin (floor Nmin, cap
    // 20*Nmin), the historical behaviour. Setting them separates the count a
    // negligible-weight cell is padded up to from the global budget and from
    // the ceiling a hot cell may reach. On a mesh whose cell count is dominated
    // by refinement in regions holding almost no radiation, it is the floor and
    // not the proportional share that sets the population.
    std::size_t Nfloor = 0;
    std::size_t Nmax = 0;
};

STORM_COMB_INLINE
std::uint64_t MakeBinRngKey(
    const std::uint64_t activationEpoch,
    const std::uint64_t rank,
    const std::size_t cellIndex,
    const std::size_t groupIndex = 0)
{
    // Preserve field order; equal fields must not cancel one another.
    const auto cellKey = CounterRNG::makeKey(
        activationEpoch, rank, static_cast<std::uint64_t>(cellIndex));
    return CounterRNG::mix(cellKey ^ static_cast<std::uint64_t>(groupIndex));
}

STORM_COMB_INLINE
std::size_t GlobalBudget(
    const std::size_t globalCellCount,
    const Parameters &parameters)
{
    return static_cast<std::size_t>(
        static_cast<double>(globalCellCount) *
        static_cast<double>(parameters.Nmin) *
        parameters.totalParticlesFactor);
}

// Resampling creates a new history. IDs are assigned later by the manager,
// but shared transport consumes RNG state directly, so initialize it here.
template<typename ParticleT>
STORM_COMB_INLINE
void RekeyClone(ParticleT &particle, const std::uint64_t binKey,
                const std::size_t outputIndex)
{
    particle.rngKey = CounterRNG::makeKey(particle.rngKey, binKey, outputIndex);
    particle.rngCounter = 0;
}

STORM_COMB_INLINE
std::size_t TargetParticleCount(
    const double binWeight,
    const double totalWeight,
    const std::size_t globalBudget,
    const std::size_t Nmin,
    const std::size_t Nfloor = 0,
    const std::size_t Nmax = 0)
{
    if(binWeight <= 0.0 || totalWeight <= 0.0)
    {
        return 0;
    }
    const std::size_t proportional = static_cast<std::size_t>(static_cast<double>(globalBudget) * binWeight / totalWeight);
    const std::size_t floorCount = (Nfloor > 0) ? Nfloor : Nmin;
    const std::size_t capCount = (Nmax > 0) ? Nmax : Nmin * 20;
    return std::min(capCount, std::max(floorCount, proportional));
}

STORM_COMB_INLINE
bool LessParticleKey(const rank_t leftRank, const particle_id_t leftID, const rank_t rightRank, const particle_id_t rightID)
{
#ifdef STORM_WITH_MPI
    if(leftRank != rightRank)
    {
        return leftRank < rightRank;
    }
#else
    (void)leftRank;
    (void)rightRank;
#endif
    return leftID < rightID;
}

STORM_COMB_INLINE
void FisherYatesShuffle(std::size_t *indices, const std::size_t count, const std::uint64_t rngKey)
{
    for(std::size_t i = count; i > 1; --i)
    {
        const std::uint64_t draw = CounterRNG::next(rngKey, count - i);
        const std::size_t j = draw % i;
        const std::size_t tmp = indices[i - 1];
        indices[i - 1] = indices[j];
        indices[j] = tmp;
    }
}

STORM_COMB_INLINE
std::size_t CountSplitCopies(const double weight, const double idealWeight)
{
    if(weight <= 2.0 * idealWeight || idealWeight <= 0.0)
    {
        return 1;
    }
    return static_cast<std::size_t>(std::ceil(weight / idealWeight));
}

STORM_COMB_INLINE
std::size_t CountUndersampledBin(const double *weights, const std::size_t *sourceIndices, const std::size_t count, const double binWeight, const std::size_t target)
{
    if(count == 0 || target == 0 || binWeight <= 0.0)
    {
        return 0;
    }
    const double idealWeight =
        binWeight / static_cast<double>(target);
    std::size_t outputs = 0;
    for(std::size_t i = 0; i < count; ++i)
    {
        outputs += CountSplitCopies(
            weights[sourceIndices[i]], idealWeight);
    }
    return outputs;
}

STORM_COMB_INLINE
std::size_t CountOversampledBin(
    const double *weights,
    const std::size_t *sourceIndices,
    const std::size_t count,
    const double binWeight,
    const std::size_t target,
    const double combOffset)
{
    if(count == 0 || target == 0 || binWeight <= 0.0)
    {
        return 0;
    }
    const double survivorWeight =
        binWeight / static_cast<double>(target);
    std::size_t outputs = 0;
    std::size_t combIndex = 0;
    double cumulativeWeight = 0.0;
    for(std::size_t i = 0; i < count; ++i)
    {
        const double weight = weights[sourceIndices[i]];
        while((cumulativeWeight + weight) >
                  (static_cast<double>(combIndex) + combOffset) *
                      survivorWeight &&
              combIndex < target)
        {
            ++combIndex;
            ++outputs;
        }
        cumulativeWeight += weight;
    }
    return outputs;
}

STORM_COMB_INLINE
std::size_t CountBin(const double *weights, const std::size_t *sourceIndices, const std::size_t count, const double binWeight, const std::size_t target, const double combOffset)
{
    if(count <= target)
    {
        return CountUndersampledBin(weights, sourceIndices, count, binWeight, target);
    }
    return CountOversampledBin(weights, sourceIndices, count, binWeight, target, combOffset);
}

template<typename EmitFn>
STORM_COMB_INLINE
void EmitUndersampledBin(
    const std::size_t *sourceIndices,
    const double *weights,
    const std::size_t count,
    const double binWeight,
    const std::size_t target,
    const std::size_t cellIndex,
    EmitFn emit)
{
    if(count == 0 || target == 0 || binWeight <= 0.0)
    {
        return;
    }
    const double idealWeight = binWeight / static_cast<double>(target);
    for(std::size_t i = 0; i < count; ++i)
    {
        const std::size_t sourceIndex = sourceIndices[i];
        const double weight = weights[sourceIndex];
        const std::size_t copies =
            CountSplitCopies(weight, idealWeight);
        if(copies == 1)
        {
            emit(sourceIndex, weight, weight, cellIndex, false);
            continue;
        }
        const double splitWeight = weight / static_cast<double>(copies);
        for(std::size_t copy = 0; copy < copies; ++copy)
        {
            emit(sourceIndex, splitWeight, splitWeight, cellIndex, true);
        }
    }
}

template<typename EmitFn>
STORM_COMB_INLINE
void EmitOversampledBin(
    const std::size_t *sourceIndices,
    const double *weights,
    const std::size_t count,
    const double binWeight,
    const std::size_t target,
    const std::size_t cellIndex,
    const double combOffset,
    EmitFn emit)
{
    if(count == 0 || target == 0 || binWeight <= 0.0)
    {
        return;
    }
    const double survivorWeight = binWeight / static_cast<double>(target);
    std::size_t combIndex = 0;
    double cumulativeWeight = 0.0;
    for(std::size_t i = 0; i < count; ++i)
    {
        const std::size_t sourceIndex = sourceIndices[i];
        const double weight = weights[sourceIndex];
        while((cumulativeWeight + weight) > (static_cast<double>(combIndex) + combOffset) * survivorWeight and combIndex < target)
        {
            ++combIndex;
            emit(sourceIndex, survivorWeight, survivorWeight, cellIndex, true);
        }
        cumulativeWeight += weight;
    }
}

template<typename EmitFn>
STORM_COMB_INLINE
void EmitBin(
    const std::size_t *sourceIndices,
    const double *weights,
    const std::size_t count,
    const double binWeight,
    const std::size_t target,
    const std::size_t cellIndex,
    const double combOffset,
    EmitFn emit)
{
    if(count <= target)
    {
        EmitUndersampledBin(sourceIndices, weights, count, binWeight, target, cellIndex, emit);
        return;
    }
    EmitOversampledBin(sourceIndices, weights, count, binWeight, target, cellIndex, combOffset, emit);
}

inline void AllocateStratifiedTargets(
    const std::vector<std::size_t> &activeGroups,
    const std::vector<double> &binWeights,
    const double cellWeight,
    const std::size_t cellTarget,
    const std::size_t groupCount,
    const std::size_t minParticlesPerGroup,
    std::vector<std::size_t> &targetByGroup)
{
    targetByGroup.assign(groupCount, 0);
    if(activeGroups.empty() || cellTarget == 0 || cellWeight <= 0.0)
    {
        return;
    }

    std::size_t allocated = 0;
    if(activeGroups.size() * minParticlesPerGroup <= cellTarget)
    {
        for(const std::size_t group : activeGroups)
        {
            targetByGroup[group] = minParticlesPerGroup;
            allocated += minParticlesPerGroup;
        }
    }
    else
    {
        std::vector<std::size_t> sortedGroups = activeGroups;
        std::sort(sortedGroups.begin(), sortedGroups.end(),
            [&](const std::size_t left, const std::size_t right)
            {
                return binWeights[left] > binWeights[right];
            });
        const std::size_t protectedGroups =
            std::min(sortedGroups.size(), cellTarget);
        for(std::size_t index = 0; index < protectedGroups; ++index)
        {
            targetByGroup[sortedGroups[index]] = 1;
            ++allocated;
        }
    }

    const std::size_t remaining = (cellTarget > allocated)? cellTarget - allocated : 0;
    std::size_t proportionalAllocated = 0;
    std::vector<double> fractional(groupCount, 0.0);
    for(const std::size_t group : activeGroups)
    {
        if(targetByGroup[group] == 0)
        {
            continue;
        }
        const double exactExtra = static_cast<double>(remaining) * binWeights[group] / cellWeight;
        const std::size_t extra = static_cast<std::size_t>(std::floor(exactExtra));
        targetByGroup[group] += extra;
        proportionalAllocated += extra;
        fractional[group] = exactExtra - static_cast<double>(extra);
    }
    while(proportionalAllocated < remaining)
    {
        std::size_t bestGroup = groupCount;
        double bestFraction = -1.0;
        for(const std::size_t group : activeGroups)
        {
            if(targetByGroup[group] > 0 and fractional[group] > bestFraction)
            {
                bestGroup = group;
                bestFraction = fractional[group];
            }
        }
        if(bestGroup == groupCount)
        {
            break;
        }
        ++targetByGroup[bestGroup];
        fractional[bestGroup] = 0.0;
        ++proportionalAllocated;
    }
}

} // namespace comb
} // namespace STORM

#undef STORM_COMB_INLINE

#endif // STORM_COMB_CORE_HPP
