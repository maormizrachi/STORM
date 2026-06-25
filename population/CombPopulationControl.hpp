#ifndef STORM_COMB_POPULATION_CONTROL_HPP
#define STORM_COMB_POPULATION_CONTROL_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <boost/random/mersenne_twister.hpp>
#ifdef STORM_WITH_MPI
    #include <mpi.h>
#endif
#include "PopulationControl.hpp"

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
CombPopulationControl<T, Grid>::CombPopulationControl(const Grid &grid, size_t Nmin, double totalParticlesFactor)
    : PopulationControl<T, Grid>(grid), Nmin(Nmin), totalParticlesFactor(totalParticlesFactor)
{}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> CombPopulationControl<T, Grid>::activate(const std::vector<Particle<T, Grid>> &particles)
{
    using MCParticle = Particle<T, Grid>;

    #ifdef STORM_WITH_MPI
        rank_t rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        static std::mt19937_64 gen((873 * rank) + particles.size());
    #else
        static std::mt19937_64 gen(particles.size());
    #endif

    std::vector<MCParticle> result;

    size_t Ncells = this->grid.GetPointNo();
    size_t Ntotal = Ncells;
    #ifdef STORM_WITH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif
    std::vector<double> weights(Ncells, 0);
    std::vector<std::vector<const MCParticle *>> particlesInCells(Ncells);

    double totalWeight = 0.0;
    for(const MCParticle &particle : particles)
    {
        assert(particle.cellIndex < Ncells);
        weights[particle.cellIndex] += particle.weight;
        totalWeight += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    #ifdef STORM_WITH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif

    Ntotal = std::llround(Ntotal * this->Nmin * this->totalParticlesFactor);

    std::uniform_real_distribution<double> dist(0, 1);

    for(size_t i = 0; i < Ncells; i++)
    {
        size_t NinCell = std::min(this->Nmin * 20, std::max<size_t>(this->Nmin, std::llround(Ntotal * weights[i] / totalWeight)));
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
                    particleCpy.id = std::numeric_limits<size_t>::max();
                    #ifdef STORM_WITH_MPI
                        particleCpy.rank = std::numeric_limits<rank_t>::max();
                    #endif
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
            T original = p.location;
            T direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) && t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
        }
    }

    return result;
}

} // namespace STORM

#endif // STORM_COMB_POPULATION_CONTROL_HPP
