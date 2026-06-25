#ifndef STORM_SIMPLE_RADIATION_PHYSICS_HPP
#define STORM_SIMPLE_RADIATION_PHYSICS_HPP

#include <cmath>
#include <random>
#include <vector>
#include <memory>
#include <cassert>
#include <boost/math/special_functions/pow.hpp>
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/radiation/RadiationCell.hpp"
#include "monte/radiation/OpacityModel.hpp"
#include "monte/PhysicalConstants.hpp"

namespace STORM {

template<typename T, typename Grid>
class SimpleRadiationPhysics : public MonteCarloPhysics<T, Grid>
{
public:
    using MCParticle = Particle<T, Grid>;

    SimpleRadiationPhysics(const Grid &grid,
                           const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary,
                           std::vector<RadiationCell> &cells,
                           const std::shared_ptr<OpacityModel> &opacity,
                           size_t newPhotonsPerCell = 5);

    std::vector<MCParticle> preStep(double fullDt) override;
    StepResult<T, Grid> step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd) override;
    void postStep(const std::vector<MCParticle> &particles, double fullDt) override;

    const std::vector<double> &GetEradTimeAvg() const { return this->eradTimeAvg; }

private:
    std::vector<RadiationCell> &cells;
    std::shared_ptr<OpacityModel> opacity;
    size_t newPhotonsPerCell;
    std::vector<double> planckOpacities;
    std::vector<double> fleckFactors;
    std::vector<double> eradTimeAvg;
    std::mt19937_64 re;
    std::uniform_real_distribution<double> dist;

    MCParticle GenerateSingleParticle(size_t cellIndex, double fullDt);
    std::vector<MCParticle> GenerateParticles(double fullDt);
};

template<typename T, typename Grid>
SimpleRadiationPhysics<T, Grid>::SimpleRadiationPhysics(const Grid &grid,
                                                         const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary,
                                                         std::vector<RadiationCell> &cells,
                                                         const std::shared_ptr<OpacityModel> &opacity,
                                                         size_t newPhotonsPerCell):
    MonteCarloPhysics<T, Grid>(grid, boundary),
    cells(cells), opacity(opacity), newPhotonsPerCell(newPhotonsPerCell),
    re(42), dist(0.0, 1.0)
{}

template<typename T, typename Grid>
typename SimpleRadiationPhysics<T, Grid>::MCParticle SimpleRadiationPhysics<T, Grid>::GenerateSingleParticle(size_t cellIndex, double fullDt)
{
    MCParticle p;
    p.cellIndex = cellIndex;
    p.location = this->grid.GetMeshPoint(cellIndex);

    double theta = 2.0 * M_PI * this->dist(this->re);
    double cosTheta = 2.0 * this->dist(this->re) - 1.0;
    double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
    p.velocity = T(sinTheta * std::cos(theta), sinTheta * std::sin(theta), cosTheta);
    p.velocity *= constants::clight;

    p.timeLeft = fullDt * this->dist(this->re);
    p.frequency = 0;
    p.steps = 0;
    return p;
}

template<typename T, typename Grid>
std::vector<typename SimpleRadiationPhysics<T, Grid>::MCParticle> SimpleRadiationPhysics<T, Grid>::GenerateParticles(double fullDt)
{
    std::vector<MCParticle> newParticles;
    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        double T_cell = this->cells[i].temperature;
        double kappaP = this->planckOpacities[i];
        double f = this->fleckFactors[i];
        double volume = this->grid.GetVolume(i);
        double energyToCreate = f * volume * constants::arad * boost::math::pow<4>(T_cell) * kappaP * fullDt * constants::clight;
        if(energyToCreate <= 0)
        {
            continue;
        }
        this->cells[i].internalEnergy -= energyToCreate / volume;
        double energyPerParticle = energyToCreate / this->newPhotonsPerCell;
        for(size_t j = 0; j < this->newPhotonsPerCell; j++)
        {
            MCParticle p = this->GenerateSingleParticle(i, fullDt);
            p.weight = energyPerParticle;
            p.initialWeight = energyPerParticle;
            newParticles.push_back(p);
        }
    }
    return newParticles;
}

template<typename T, typename Grid>
std::vector<typename SimpleRadiationPhysics<T, Grid>::MCParticle> SimpleRadiationPhysics<T, Grid>::preStep(double fullDt)
{
    size_t Ncells = this->grid.GetPointNo();
    this->planckOpacities.resize(Ncells);
    this->fleckFactors.resize(Ncells);
    this->eradTimeAvg.assign(Ncells, 0.0);

    for(size_t i = 0; i < Ncells; i++)
    {
        double T_cell = this->cells[i].temperature;
        this->planckOpacities[i] = this->opacity->PlanckOpacity(i, T_cell);
        double cv = this->cells[i].cv;
        if(cv > 0 and T_cell > 0)
        {
            double denom = 1.0 + 4.0 * constants::arad * boost::math::pow<3>(T_cell) * this->planckOpacities[i] * constants::clight * fullDt / cv;
            this->fleckFactors[i] = 1.0 / denom;
        }
        else
        {
            this->fleckFactors[i] = 1.0;
        }
    }

    std::vector<MCParticle> newParticles = this->GenerateParticles(fullDt);

    std::vector<MCParticle> boundaryParticles = this->boundary->generateNewBoundaryParticles(fullDt);
    newParticles.insert(newParticles.end(), boundaryParticles.begin(), boundaryParticles.end());

    return newParticles;
}

template<typename T, typename Grid>
StepResult<T, Grid> SimpleRadiationPhysics<T, Grid>::step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd)
{
    (void) particlesToAdd;

    StepResult<T, Grid> functionality;
    size_t cellIndex = particle.cellIndex;

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);

    double kappaP = this->planckOpacities[cellIndex];
    double kappaS = this->opacity->ScatteringOpacity(cellIndex, this->cells[cellIndex].temperature);
    double f = this->fleckFactors[cellIndex];
    double effectiveSigma = kappaS + (1.0 - f) * kappaP;
    double timeScatter = std::numeric_limits<double>::infinity();
    if(effectiveSigma > 0)
    {
        timeScatter = -std::log(this->dist(this->re)) / (effectiveSigma * constants::clight);
    }

    double timeLeft = particle.timeLeft;
    double dt;
    enum { EVT_FACE, EVT_SCATTER, EVT_CENSUS } event;
    if(timeLeft <= timeIntersect and timeLeft <= timeScatter)
    {
        event = EVT_CENSUS;
        dt = timeLeft;
    }
    else if(timeIntersect <= timeScatter)
    {
        event = EVT_FACE;
        dt = timeIntersect;
    }
    else
    {
        event = EVT_SCATTER;
        dt = timeScatter;
    }

    particle.location += particle.velocity * dt;
    particle.timeLeft -= dt;

    double kappaAf = kappaP * f;
    if(kappaAf > 0 and dt > 0)
    {
        double expm1Val = std::expm1(-dt * kappaAf * constants::clight);
        particle.weight *= (1.0 + expm1Val);
        double absorbedEnergy = particle.weight * (-expm1Val / (1.0 + expm1Val));
        this->eradTimeAvg[cellIndex] += absorbedEnergy * dt;
    }

    if(particle.weight < 1e-3 * particle.initialWeight and particle.initialWeight > 0)
    {
        functionality.change = ParticleStatus::REMOVE;
        return functionality;
    }

    switch(event)
    {
        case EVT_FACE:
            functionality.change = ParticleStatus::CELL_MOVE;
            functionality.nextCellIndex = nextCellIndex;
            break;
        case EVT_SCATTER:
        {
            double theta = 2.0 * M_PI * this->dist(this->re);
            double cosTheta = 2.0 * this->dist(this->re) - 1.0;
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            particle.velocity = T(sinTheta * std::cos(theta), sinTheta * std::sin(theta), cosTheta);
            particle.velocity *= constants::clight;
            functionality.change = ParticleStatus::NO_CELL_MOVE;
            break;
        }
        case EVT_CENSUS:
            functionality.change = ParticleStatus::DONE;
            break;
    }
    return functionality;
}

template<typename T, typename Grid>
void SimpleRadiationPhysics<T, Grid>::postStep(const std::vector<MCParticle> &particles, double fullDt)
{
    size_t Ncells = this->grid.GetPointNo();

    for(size_t i = 0; i < Ncells; i++)
    {
        double volume = this->grid.GetVolume(i);
        if(volume > 0 and fullDt > 0)
        {
            this->eradTimeAvg[i] /= (fullDt * volume);
        }
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        this->cells[i].Erad = 0;
    }
    for(const MCParticle &p : particles)
    {
        if(p.cellIndex < Ncells)
        {
            double volume = this->grid.GetVolume(p.cellIndex);
            if(volume > 0)
            {
                this->cells[p.cellIndex].Erad += p.weight / volume;
            }
        }
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        if(this->cells[i].cv > 0)
        {
            this->cells[i].temperature = this->cells[i].internalEnergy / this->cells[i].cv;
            if(this->cells[i].temperature < 0)
            {
                this->cells[i].temperature = 0;
            }
        }
    }
}

} // namespace STORM

#endif // STORM_SIMPLE_RADIATION_PHYSICS_HPP
