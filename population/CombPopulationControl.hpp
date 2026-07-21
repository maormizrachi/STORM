#ifndef STORM_COMB_POPULATION_CONTROL_HPP
#define STORM_COMB_POPULATION_CONTROL_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <functional>
#include <limits>
#include <utility>
#include <boost/random/mersenne_twister.hpp>
#ifdef STORM_WITH_MPI
    #include <mpi.h>
#endif
#include "PopulationControl.hpp"
#include "../StormError.hpp"

namespace STORM {

template<typename T, typename Grid>
class CombPopulationControl : public PopulationControl<T, Grid>
{
public:
    CombPopulationControl(const Grid &grid, size_t Nmin = 20, double totalParticlesFactor = 2.0);

    std::vector<Particle<T, Grid>> activate(const std::vector<Particle<T, Grid>> &particles) override;

private:
    size_t Nmin;
    double totalParticlesFactor;
    boost::random::mt19937_64 gen;
};

template<typename T, typename Grid>
class StratifiedCombPopulationControl : public PopulationControl<T, Grid>
{
public:
    using MCParticle = Particle<T, Grid>;
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
};

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
CombPopulationControl<T, Grid>::CombPopulationControl(const Grid &grid, size_t Nmin, double totalParticlesFactor)
    : PopulationControl<T, Grid>(grid), Nmin(Nmin), totalParticlesFactor(totalParticlesFactor)
{}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> CombPopulationControl<T, Grid>::activate(const std::vector<Particle<T, Grid>> &particles)
{
    using MCParticle = Particle<T, Grid>;

    #ifdef STORM_WITH_MPI
        rank_t rank = 0;
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        static std::mt19937_64 gen((873 * rank) + particles.size());
    #else
        static std::mt19937_64 gen(particles.size());
    #endif

    std::vector<MCParticle> result;
    size_t Ncells = this->grid.GetPointNo();
    size_t Ntotal = Ncells;
    #ifdef STORM_WITH_MPI
        { int mpiInit = 0; MPI_Initialized(&mpiInit); if(mpiInit) { MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); } }
    #endif
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
        if(!std::isfinite(particle.weight))
        {
            STORMError eo("Comb Population Control: particle weight is not finite");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight < 0.0)
        {
            STORMError eo("Comb Population Control: particle weight is negative");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight == 0.0)
            continue;
        assert(particle.cellIndex < Ncells);
        weights[particle.cellIndex] += particle.weight;
        totalWeight += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    #ifdef STORM_WITH_MPI
        { int mpiInit = 0; MPI_Initialized(&mpiInit); if(mpiInit) { MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); } }
    #endif
    if(!std::isfinite(totalWeight))
    {
        STORMError eo("Comb Population Control: total particle weight is not finite");
        eo.addEntry("Total weight", totalWeight);
        eo.addEntry("Local particle count", particles.size());
        throw eo;
    }
    if(totalWeight == 0.0)
        return result;

    Ntotal = static_cast<size_t>(Ntotal * this->Nmin * this->totalParticlesFactor);

    std::uniform_real_distribution<double> dist(0, 1);

    for(size_t i = 0; i < Ncells; i++)
    {
        size_t NinCell = std::min(this->Nmin * 20, std::max<size_t>(this->Nmin, static_cast<size_t>(Ntotal * weights[i] / totalWeight)));
        if(particlesInCells[i].size() <= NinCell)
        {
            double weight_ideal = weights[i] / NinCell;
            for(const MCParticle *particle : particlesInCells[i])
            {
                if(particle->weight > 2 * weight_ideal)
                {
                    MCParticle particleCpy = *particle;
                    size_t Nsplit = std::ceil(particle->weight / weight_ideal);
                    double weight_split = particle->weight / Nsplit;
                    particleCpy.weight = weight_split;
                    particleCpy.initialWeight = std::abs(weight_split);
                    particleCpy.id = std::numeric_limits<size_t>::max();
                    #ifdef STORM_WITH_MPI
                        particleCpy.rank = std::numeric_limits<rank_t>::max();
                    #endif
                    particleCpy.cellIndex = i;
                    particleCpy.timeLeft = 0;
                    particleCpy.steps = 0;
                    for(size_t j = 0; j < Nsplit; j++)
                    {
                        result.push_back(particleCpy);
                    }
                }
                else
                {
                    result.push_back(*particle);
                }
            }
            continue;
        }

        // sort for reproducibility
        #ifdef STORM_WITH_MPI
            std::sort(particlesInCells[i].begin(), particlesInCells[i].end(), [](const MCParticle *p1, const MCParticle *p2){ return (p1->rank) < (p2->rank) or ((p1->rank) == (p2->rank) and p1->id < p2->id); });
        #else
            std::sort(particlesInCells[i].begin(), particlesInCells[i].end(), [](const MCParticle *p1, const MCParticle *p2){ return p1->id < p2->id; });
        #endif

        std::shuffle(particlesInCells[i].begin(), particlesInCells[i].end(), gen);

        double new_energy = weights[i] / NinCell;

        double r = dist(gen);
        size_t comb_index = 0;
        double cum_sum_w = 0;
        for(const MCParticle *particle : particlesInCells[i])
        {
            while((cum_sum_w + particle->weight) > (comb_index + r) * new_energy)
            {
                ++comb_index;
                result.push_back(*particle);
                result.back().id = std::numeric_limits<size_t>::max();
                #ifdef STORM_WITH_MPI
                    result.back().rank = std::numeric_limits<rank_t>::max();
                #endif
                result.back().cellIndex = i;
                result.back().timeLeft = 0;
                result.back().weight = new_energy;
                result.back().initialWeight = new_energy;
            }
            cum_sum_w += particle->weight;
        }
    }

    for(MCParticle &p : result)
    {
        if(this->grid.IsPointOutsideBox(p.location))
        {
            if(p.cellIndex >= Ncells)
            {
                STORMError eo("Comb Population Control: outside particle has invalid cell index");
                eo.addEntry("Particle cell index", p.cellIndex);
                eo.addEntry("Cell count", Ncells);
                throw eo;
            }
            T original = p.location;
            T direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) && t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
            if(this->grid.IsPointOutsideBox(p.location))
                p.location = this->grid.GetMeshPoint(p.cellIndex);
        }
    }

    return result;
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> StratifiedCombPopulationControl<T, Grid>::activate(const std::vector<Particle<T, Grid>> &particles)
{
    using MCParticle = Particle<T, Grid>;

    #ifdef STORM_WITH_MPI
        rank_t rank = 0;
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        std::mt19937_64 gen((873 * rank) + particles.size());
    #else
        std::mt19937_64 gen(particles.size());
    #endif

    std::vector<MCParticle> result;
    size_t const Ncells = this->grid.GetPointNo();
    size_t Ntotal = Ncells;
    #ifdef STORM_WITH_MPI
        { int mpiInit = 0; MPI_Initialized(&mpiInit); if(mpiInit) { MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); } }
    #endif

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
        if(!std::isfinite(particle.weight))
        {
            STORMError eo("Stratified Comb Population Control: particle weight is not finite");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight < 0.0)
        {
            STORMError eo("Stratified Comb Population Control: particle weight is negative");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight == 0.0)
            continue;
        if(particle.cellIndex >= Ncells)
        {
            STORMError eo("Stratified Comb Population Control: input particle has invalid cell index");
            eo.addEntry("Particle cell index", particle.cellIndex);
            eo.addEntry("Cell count", Ncells);
            throw eo;
        }
        assert(particle.cellIndex < Ncells);
        size_t group = this->classifier(particle);
        if(group >= this->groupCount)
            group = this->groupCount - 1;
        cellWeights[particle.cellIndex] += particle.weight;
        binWeights[particle.cellIndex][group] += particle.weight;
        totalWeight += particle.weight;
        particlesInBins[particle.cellIndex][group].push_back(&particle);
    }

    #ifdef STORM_WITH_MPI
        { int mpiInit = 0; MPI_Initialized(&mpiInit); if(mpiInit) { MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); } }
    #endif
    if(!std::isfinite(totalWeight))
    {
        STORMError eo("Stratified Comb Population Control: total particle weight is not finite");
        eo.addEntry("Total weight", totalWeight);
        eo.addEntry("Local particle count", particles.size());
        throw eo;
    }
    if(totalWeight == 0.0)
        return result;

    Ntotal = static_cast<size_t>(Ntotal * this->Nmin * this->totalParticlesFactor);
    std::uniform_real_distribution<double> dist(0, 1);

    auto appendCombBin = [&](const std::vector<const MCParticle*> &bin,
                             double binWeight,
                             size_t target,
                             size_t cellIndex)
    {
        if(bin.empty() || target == 0 || binWeight <= 0.0)
            return;
        if(bin.size() <= target)
        {
            double const idealWeight = binWeight / static_cast<double>(target);
            for(const MCParticle *particle : bin)
            {
                if(particle->weight > 2.0 * idealWeight)
                {
                    size_t const splitCount = static_cast<size_t>(std::ceil(particle->weight / idealWeight));
                    double const splitWeight = particle->weight / static_cast<double>(splitCount);
                    MCParticle particleCopy = *particle;
                    particleCopy.weight = splitWeight;
                    particleCopy.initialWeight = splitWeight;
                    particleCopy.id = std::numeric_limits<size_t>::max();
                    #ifdef STORM_WITH_MPI
                    particleCopy.rank = std::numeric_limits<rank_t>::max();
                    #endif
                    particleCopy.cellIndex = cellIndex;
                    particleCopy.timeLeft = 0;
                    particleCopy.steps = 0;
                    for(size_t j = 0; j < splitCount; j++)
                        result.push_back(particleCopy);
                }
                else
                {
                    result.push_back(*particle);
                }
            }
            return;
        }

        std::vector<const MCParticle*> shuffled = bin;
        #ifdef STORM_WITH_MPI
            std::sort(shuffled.begin(), shuffled.end(), [](const MCParticle *p1, const MCParticle *p2){return (p1->rank) < (p2->rank) or ((p1->rank) == (p2->rank) and p1->id < p2->id);});
        #else
            std::sort(shuffled.begin(), shuffled.end(), [](const MCParticle *p1, const MCParticle *p2){return p1->id < p2->id;});
        #endif
        std::shuffle(shuffled.begin(), shuffled.end(), gen);

        double const newWeight = binWeight / static_cast<double>(target);
        double const r = dist(gen);
        size_t combIndex = 0;
        double cumWeight = 0.0;
        for(const MCParticle *particle : shuffled)
        {
            while((cumWeight + particle->weight) > (static_cast<double>(combIndex) + r) * newWeight &&
                  combIndex < target)
            {
                ++combIndex;
                result.push_back(*particle);
                result.back().id = std::numeric_limits<size_t>::max();
                #ifdef STORM_WITH_MPI
                result.back().rank = std::numeric_limits<rank_t>::max();
                #endif
                result.back().cellIndex = cellIndex;
                result.back().timeLeft = 0;
                result.back().weight = newWeight;
                result.back().initialWeight = newWeight;
            }
            cumWeight += particle->weight;
        }
    };

    for(size_t i = 0; i < Ncells; i++)
    {
        if(cellWeights[i] <= 0.0)
            continue;
        size_t cellTarget = std::min(this->Nmin * 20, std::max(this->Nmin, static_cast<size_t>(Ntotal * cellWeights[i] / totalWeight)));

        std::vector<size_t> activeGroups;
        for(size_t g = 0; g < this->groupCount; g++)
            if(!particlesInBins[i][g].empty())
                activeGroups.push_back(g);
        if(activeGroups.empty())
            continue;

        std::vector<size_t> targetByGroup(this->groupCount, 0);
        size_t allocated = 0;
        if(activeGroups.size() * this->minParticlesPerGroup <= cellTarget)
        {
            for(size_t g : activeGroups)
            {
                targetByGroup[g] = this->minParticlesPerGroup;
                allocated += this->minParticlesPerGroup;
            }
        }
        else
        {
            std::sort(activeGroups.begin(), activeGroups.end(), [&](size_t a, size_t b)
            {
                return binWeights[i][a] > binWeights[i][b];
            });
            size_t const protectedGroups = std::min(activeGroups.size(), cellTarget);
            for(size_t j = 0; j < protectedGroups; j++)
            {
                targetByGroup[activeGroups[j]] = 1;
                ++allocated;
            }
        }

        size_t const remaining = (cellTarget > allocated) ? cellTarget - allocated : 0;
        size_t proportionalAllocated = 0;
        std::vector<double> fractional(this->groupCount, 0.0);
        for(size_t g : activeGroups)
        {
            if(targetByGroup[g] == 0)
                continue;
            double const exactExtra = static_cast<double>(remaining) * binWeights[i][g] / cellWeights[i];
            size_t const extra = static_cast<size_t>(std::floor(exactExtra));
            targetByGroup[g] += extra;
            proportionalAllocated += extra;
            fractional[g] = exactExtra - static_cast<double>(extra);
        }
        while(proportionalAllocated < remaining)
        {
            size_t bestGroup = this->groupCount;
            double bestFraction = -1.0;
            for(size_t g : activeGroups)
            {
                if(targetByGroup[g] > 0 && fractional[g] > bestFraction)
                {
                    bestGroup = g;
                    bestFraction = fractional[g];
                }
            }
            if(bestGroup == this->groupCount)
                break;
            ++targetByGroup[bestGroup];
            fractional[bestGroup] = 0.0;
            ++proportionalAllocated;
        }

        for(size_t g : activeGroups)
            appendCombBin(particlesInBins[i][g], binWeights[i][g], targetByGroup[g], i);
    }

    for(MCParticle &p : result)
    {
        if(this->grid.IsPointOutsideBox(p.location))
        {
            if(p.cellIndex >= Ncells)
            {
                STORMError eo("Stratified Comb Population Control: outside particle has invalid cell index");
                eo.addEntry("Particle cell index", p.cellIndex);
                eo.addEntry("Cell count", Ncells);
                throw eo;
            }
            T original = p.location;
            T direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) && t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
            if(this->grid.IsPointOutsideBox(p.location))
                p.location = this->grid.GetMeshPoint(p.cellIndex);
        }
    }

    return result;
}

} // namespace STORM

#endif // STORM_COMB_POPULATION_CONTROL_HPP
