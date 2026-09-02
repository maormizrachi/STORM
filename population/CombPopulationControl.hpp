#ifndef STORM_COMB_POPULATION_CONTROL_HPP
#define STORM_COMB_POPULATION_CONTROL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#ifdef STORM_WITH_MPI
    #include <mpi.h>
#endif
#include "CombCore.hpp"
#include "PopulationControl.hpp"
#include "../StormError.hpp"
#include "../utils/CounterRNG.hpp"
#ifdef STORM_WITH_GPU
#include "../gpu/CombDeviceActivate.hpp"
#include "../gpu/DevicePopulationContext.hpp"
#endif

namespace STORM {

template<typename T, typename Grid>
class CombPopulationControl : public PopulationControl<T, Grid>
{
public:
    CombPopulationControl(const Grid &grid, size_t Nmin = 20, double totalParticlesFactor = 2.0);

    std::vector<Particle<T>> activate(const std::vector<Particle<T>> &particles) override;

#ifdef STORM_WITH_GPU
    bool SupportsDeviceActivation() const override
    {
        return true;
    }

    void activateDevice(gpu::DevicePopulationContext &context) const override;
#endif

private:
    size_t Nmin;
    double totalParticlesFactor;
    mutable std::uint64_t activationEpoch_ = 0;

    comb::Parameters Parameters() const
    {
        return {this->Nmin, this->totalParticlesFactor};
    }

    template<typename MCParticle>
    void RepairOutsideBox(std::vector<MCParticle> &particles, size_t Ncells) const
    {
        for(MCParticle &particle : particles)
        {
            if(this->grid.IsPointOutsideBox(particle.location))
            {
                if(particle.cellIndex >= Ncells)
                {
                    STORMError eo("Comb Population Control: outside particle has invalid cell index");
                    eo.addEntry("Particle cell index", particle.cellIndex);
                    eo.addEntry("Cell count", Ncells);
                    throw eo;
                }
                T original = particle.location;
                T direction = this->grid.GetMeshPoint(particle.cellIndex) - original;
                double t = 1e-6;
                while(this->grid.IsPointOutsideBox(particle.location) && t < 1.0)
                {
                    particle.location = original + t * direction;
                    t *= 2;
                }
                if(this->grid.IsPointOutsideBox(particle.location))
                {
                    particle.location = this->grid.GetMeshPoint(particle.cellIndex);
                }
            }
        }
    }
};

template<typename T, typename Grid>
class StratifiedCombPopulationControl : public PopulationControl<T, Grid>
{
public:
    using MCParticle = Particle<T>;
    using Classifier = std::function<size_t(const MCParticle&)>;

    StratifiedCombPopulationControl(const Grid &grid,
                                    size_t groupCount,
                                    Classifier classifier,
                                    size_t Nmin = 20,
                                    double totalParticlesFactor = 2.0,
                                    size_t minParticlesPerGroup = 2);

    std::vector<MCParticle> activate(const std::vector<MCParticle> &particles) override;

private:
    size_t groupCount;
    Classifier classifier;
    size_t Nmin;
    double totalParticlesFactor;
    size_t minParticlesPerGroup;
    mutable std::uint64_t activationEpoch_ = 0;

    comb::Parameters Parameters() const
    {
        return {this->Nmin, this->totalParticlesFactor};
    }

    template<typename ParticleType>
    void RepairOutsideBox(std::vector<ParticleType> &particles, size_t Ncells) const
    {
        for(ParticleType &particle : particles)
        {
            if(this->grid.IsPointOutsideBox(particle.location))
            {
                if(particle.cellIndex >= Ncells)
                {
                    STORMError eo("Stratified Comb Population Control: outside particle has invalid cell index");
                    eo.addEntry("Particle cell index", particle.cellIndex);
                    eo.addEntry("Cell count", Ncells);
                    throw eo;
                }
                T original = particle.location;
                T direction = this->grid.GetMeshPoint(particle.cellIndex) - original;
                double t = 1e-6;
                while(this->grid.IsPointOutsideBox(particle.location) && t < 1.0)
                {
                    particle.location = original + t * direction;
                    t *= 2;
                }
                if(this->grid.IsPointOutsideBox(particle.location))
                {
                    particle.location = this->grid.GetMeshPoint(particle.cellIndex);
                }
            }
        }
    }
};

namespace comb_host_detail
{

template<typename MCParticle>
inline void ValidateParticle(
    const MCParticle &particle,
    size_t Ncells,
    const std::string &label)
{
    if(particle.weight == 0.0)
    {
        return;
    }
    if(!std::isfinite(particle.weight))
    {
        STORMError eo(label + ": particle weight is not finite");
        eo.addEntry("Particle weight", particle.weight);
        eo.addEntry("Particle cell index", particle.cellIndex);
        throw eo;
    }
    if(particle.weight < 0.0)
    {
        STORMError eo(label + ": particle weight is negative");
        eo.addEntry("Particle weight", particle.weight);
        eo.addEntry("Particle cell index", particle.cellIndex);
        throw eo;
    }
    if(particle.cellIndex >= Ncells)
    {
        STORMError eo(label + ": input particle has invalid cell index");
        eo.addEntry("Particle cell index", particle.cellIndex);
        eo.addEntry("Cell count", Ncells);
        throw eo;
    }
}

template<typename MCParticle>
inline void SortParticleIndices(
    std::size_t *indices,
    const std::size_t count,
    const std::vector<const MCParticle *> &particles)
{
    std::sort(
        indices,
        indices + count,
        [&](const std::size_t left, const std::size_t right)
        {
#ifdef STORM_WITH_MPI
            return comb::LessParticleKey(particles[left]->rank, particles[left]->id, particles[right]->rank, particles[right]->id);
#else
            (void)particles[left]->rank;
            (void)particles[right]->rank;
            return comb::LessParticleKey(0, particles[left]->id, 0, particles[right]->id);
#endif
        });
}

template<typename MCParticle>
inline void AppendCombBin(
    const std::vector<const MCParticle *> &bin,
    double binWeight,
    size_t target,
    size_t cellIndex,
    std::uint64_t activationEpoch,
    rank_t rank,
    size_t groupIndex,
    std::vector<MCParticle> &result)
{
    if(bin.empty() or target == 0 or binWeight <= 0.0)
    {
        return;
    }

    const std::size_t count = bin.size();
    std::vector<double> weights(count);
    std::vector<std::size_t> indices(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        weights[i] = bin[i]->weight;
        indices[i] = i;
    }

    SortParticleIndices(indices.data(), count, bin);
    const std::uint64_t rngKey = comb::MakeBinRngKey(
        activationEpoch,
        static_cast<std::uint64_t>(rank),
        cellIndex,
        groupIndex);
    comb::FisherYatesShuffle(indices.data(), count, rngKey);
    const double combOffset = CounterRNG::unitOpen(rngKey, 0);

    const auto emit = [&](
                          const std::size_t sourceIndex,
                          const double weight,
                          const double initialWeight,
                          const std::size_t outputCellIndex,
                          const bool resetIdentity)
    {
        MCParticle particle = *bin[sourceIndex];
        particle.weight = weight;
        particle.initialWeight = initialWeight;
        particle.cellIndex = outputCellIndex;
        particle.timeLeft = 0;
        particle.steps = 0;
        if(resetIdentity)
        {
            particle.id = std::numeric_limits<size_t>::max();
#ifdef STORM_WITH_MPI
            particle.rank = std::numeric_limits<rank_t>::max();
#endif
        }
        result.push_back(std::move(particle));
    };

    comb::EmitBin(
        indices.data(),
        weights.data(),
        count,
        binWeight,
        target,
        cellIndex,
        combOffset,
        emit);
}

inline double ReduceTotalWeight(
    double localTotalWeight,
#ifdef STORM_WITH_MPI
    MPI_Comm communicator
#else
    int communicator
#endif
)
{
#ifdef STORM_WITH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized && communicator != MPI_COMM_NULL)
    {
        MPI_Allreduce(
            MPI_IN_PLACE,
            &localTotalWeight,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            communicator);
    }
#else
    (void)communicator;
#endif
    return localTotalWeight;
}

inline std::size_t ReduceGlobalCellCount(
    std::size_t localCellCount,
#ifdef STORM_WITH_MPI
    MPI_Comm communicator
#else
    int communicator
#endif
)
{
#ifdef STORM_WITH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized && communicator != MPI_COMM_NULL)
    {
        MPI_Allreduce(
            MPI_IN_PLACE,
            &localCellCount,
            1,
            MPI_UNSIGNED_LONG_LONG,
            MPI_SUM,
            communicator);
    }
#else
    (void)communicator;
#endif
    return localCellCount;
}

} // namespace comb_host_detail

template<typename T, typename Grid>
CombPopulationControl<T, Grid>::CombPopulationControl(
    const Grid &grid,
    size_t Nmin,
    double totalParticlesFactor)
    : PopulationControl<T, Grid>(grid),
      Nmin(Nmin),
      totalParticlesFactor(totalParticlesFactor)
{}

#ifdef STORM_WITH_GPU
template<typename T, typename Grid>
void CombPopulationControl<T, Grid>::activateDevice(
    gpu::DevicePopulationContext &context) const
{
    context.parameters = this->Parameters();
    gpu::ActivateCombOnDeviceCensus(context);
}
#endif

template<typename T, typename Grid>
std::vector<Particle<T>> CombPopulationControl<T, Grid>::activate(
    const std::vector<Particle<T>> &particles)
{
    using MCParticle = Particle<T>;

    #ifdef STORM_WITH_MPI
        rank_t rank = 0;
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
    #else
        rank_t rank = 0;
    #endif

    const std::uint64_t activationEpoch = this->activationEpoch_++;
    std::vector<MCParticle> result;
    size_t Ncells = this->grid.GetPointNo();
    size_t globalCellCount = comb_host_detail::ReduceGlobalCellCount(
        Ncells,
#ifdef STORM_WITH_MPI
        MPI_COMM_WORLD
#else
        0
#endif
        );
    std::vector<double> weights(Ncells, 0);
    std::vector<std::vector<const MCParticle *>> particlesInCells(Ncells);

    double totalWeight = 0.0;
    for(const MCParticle &particle : particles)
    {
        if(this->grid.IsPointOutsideBox(particle.location))
        {
            STORMError eo("Comb Population Control: input particle is outside the box");
            eo.addEntry("Particle", particle);
            eo.addEntry("Cell count", Ncells);
            if(particle.cellIndex < Ncells)
            {
                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                eo.addEntry("Inside declared cell", this->grid.IsPointInCell(particle.location, particle.cellIndex));
            }
            throw eo;
        }
        comb_host_detail::ValidateParticle(
            particle, Ncells, "Comb Population Control");
        if(particle.weight == 0.0)
        {
            continue;
        }
        weights[particle.cellIndex] += particle.weight;
        totalWeight += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    totalWeight = comb_host_detail::ReduceTotalWeight(
        totalWeight,
#ifdef STORM_WITH_MPI
        MPI_COMM_WORLD
#else
        0
#endif
        );
    if(!std::isfinite(totalWeight))
    {
        STORMError eo("Comb Population Control: total particle weight is not finite");
        eo.addEntry("Total weight", totalWeight);
        eo.addEntry("Local particle count", particles.size());
        throw eo;
    }
    if(totalWeight == 0.0)
    {
        return result;
    }

    const std::size_t globalBudget =
        comb::GlobalBudget(globalCellCount, this->Parameters());
    for(size_t i = 0; i < Ncells; i++)
    {
        if(weights[i] <= 0.0)
        {
            continue;
        }
        const size_t target = comb::TargetParticleCount(
            weights[i], totalWeight, globalBudget, this->Nmin);
        comb_host_detail::AppendCombBin<MCParticle>(
            particlesInCells[i],
            weights[i],
            target,
            i,
            activationEpoch,
            rank,
            0,
            result);
    }

    this->RepairOutsideBox(result, Ncells);
    return result;
}

template<typename T, typename Grid>
StratifiedCombPopulationControl<T, Grid>::StratifiedCombPopulationControl(
    const Grid &grid,
    size_t groupCount,
    typename StratifiedCombPopulationControl<T, Grid>::Classifier classifier,
    size_t Nmin,
    double totalParticlesFactor,
    size_t minParticlesPerGroup)
    : PopulationControl<T, Grid>(grid),
      groupCount(std::max<size_t>(1, groupCount)),
      classifier(std::move(classifier)),
      Nmin(Nmin),
      totalParticlesFactor(totalParticlesFactor),
      minParticlesPerGroup(minParticlesPerGroup)
{}

template<typename T, typename Grid>
std::vector<Particle<T>> StratifiedCombPopulationControl<T, Grid>::activate(
    const std::vector<Particle<T>> &particles)
{
    using MCParticle = Particle<T>;

    #ifdef STORM_WITH_MPI
        rank_t rank = 0;
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
    #else
        rank_t rank = 0;
    #endif

    const std::uint64_t activationEpoch = this->activationEpoch_++;
    std::vector<MCParticle> result;
    size_t const Ncells = this->grid.GetPointNo();
    size_t globalCellCount = comb_host_detail::ReduceGlobalCellCount(
        Ncells,
#ifdef STORM_WITH_MPI
        MPI_COMM_WORLD
#else
        0
#endif
        );

    std::vector<double> cellWeights(Ncells, 0.0);
    std::vector<std::vector<std::vector<const MCParticle*>>> particlesInBins(
        Ncells, std::vector<std::vector<const MCParticle*>>(this->groupCount));
    std::vector<std::vector<double>> binWeights(Ncells, std::vector<double>(this->groupCount, 0.0));

    double totalWeight = 0.0;
    for(const MCParticle &particle : particles)
    {
        if(this->grid.IsPointOutsideBox(particle.location))
        {
            STORMError eo("Stratified Comb Population Control: input particle is outside the box");
            eo.addEntry("Particle", particle);
            eo.addEntry("Cell count", Ncells);
            throw eo;
        }
        comb_host_detail::ValidateParticle(
            particle, Ncells, "Stratified Comb Population Control");
        if(particle.weight == 0.0)
        {
            continue;
        }
        size_t group = this->classifier(particle);
        if(group >= this->groupCount)
        {
            group = this->groupCount - 1;
        }
        cellWeights[particle.cellIndex] += particle.weight;
        binWeights[particle.cellIndex][group] += particle.weight;
        totalWeight += particle.weight;
        particlesInBins[particle.cellIndex][group].push_back(&particle);
    }

    totalWeight = comb_host_detail::ReduceTotalWeight(
        totalWeight,
#ifdef STORM_WITH_MPI
        MPI_COMM_WORLD
#else
        0
#endif
        );
    if(!std::isfinite(totalWeight))
    {
        STORMError eo("Stratified Comb Population Control: total particle weight is not finite");
        eo.addEntry("Total weight", totalWeight);
        eo.addEntry("Local particle count", particles.size());
        throw eo;
    }
    if(totalWeight == 0.0)
    {
        return result;
    }

    const std::size_t globalBudget =
        comb::GlobalBudget(globalCellCount, this->Parameters());
    for(size_t i = 0; i < Ncells; i++)
    {
        if(cellWeights[i] <= 0.0)
        {
            continue;
        }
        const size_t cellTarget = comb::TargetParticleCount(
            cellWeights[i], totalWeight, globalBudget, this->Nmin);

        std::vector<size_t> activeGroups;
        for(size_t g = 0; g < this->groupCount; g++)
        {
            if(!particlesInBins[i][g].empty())
            {
                activeGroups.push_back(g);
            }
        }
        if(activeGroups.empty())
        {
            continue;
        }

        std::vector<size_t> targetByGroup;
        comb::AllocateStratifiedTargets(
            activeGroups,
            binWeights[i],
            cellWeights[i],
            cellTarget,
            this->groupCount,
            this->minParticlesPerGroup,
            targetByGroup);

        for(size_t g : activeGroups)
        {
            comb_host_detail::AppendCombBin<MCParticle>(
                particlesInBins[i][g],
                binWeights[i][g],
                targetByGroup[g],
                i,
                activationEpoch,
                rank,
                g,
                result);
        }
    }

    this->RepairOutsideBox(result, Ncells);
    return result;
}

} // namespace STORM

#endif // STORM_COMB_POPULATION_CONTROL_HPP
