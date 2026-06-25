#ifndef RDMONT_NO_PHYSICS_HPP
#define RDMONT_NO_PHYSICS_HPP

#include "MonteCarloPhysics.hpp"

namespace RDMont {

template<typename T, typename Grid>
class NoPhysics : public MonteCarloPhysics<T, Grid>
{
public:
    using MCParticle = Particle<T, Grid>;
    using Functionality = StepResult<T, Grid>;

    NoPhysics(Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary);

    std::vector<MCParticle> preStep(double fullDt) override;

    Functionality step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd) override;

    void postStep(const std::vector<MCParticle> &particles, double fullDt) override;
};

template<typename T, typename Grid>
NoPhysics<T, Grid>::NoPhysics(Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary):
    MonteCarloPhysics<T, Grid>(grid, boundary)
{}

template<typename T, typename Grid>
typename NoPhysics<T, Grid>::Functionality NoPhysics<T, Grid>::step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd)
{
    (void) particlesToAdd;
    Functionality functionality;

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect > 0);

    dt_t timeLeft = particle.timeLeft;
    dt_t dt = std::numeric_limits<dt_t>::infinity();

    if(timeLeft < timeIntersect)
    {
        functionality.change = ParticleStatus::DONE;
        dt = timeLeft;
    }
    else
    {
        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
        dt = timeIntersect;
    }

    particle.timeLeft -= dt;
    particle.location += particle.velocity * dt;
    return functionality;
}

template<typename T, typename Grid>
std::vector<typename NoPhysics<T, Grid>::MCParticle> NoPhysics<T, Grid>::preStep(double fullDt)
{
    (void) fullDt;
    return std::vector<MCParticle>();
}

template<typename T, typename Grid>
void NoPhysics<T, Grid>::postStep(const std::vector<MCParticle> &particles, double fullDt)
{
    (void) particles;
    (void) fullDt;
}

} // namespace RDMont

// Back-compat alias
template<typename T, typename Grid>
using NoMonteCarloPhysics = RDMont::NoPhysics<T, Grid>;

#endif // RDMONT_NO_PHYSICS_HPP
