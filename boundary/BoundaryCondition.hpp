#ifndef STORM_BOUNDARY_CONDITION_HPP
#define STORM_BOUNDARY_CONDITION_HPP

#include "monte/particle/Particle.hpp"
#include "monte/particle/ParticleStatus.hpp"

namespace STORM {

template<typename T, typename Grid>
class BoundaryCondition
{
public:
    BoundaryCondition(const Grid &grid);

    virtual ~BoundaryCondition() = default;

    virtual ParticleStatus apply(Particle<T, Grid> &particle) = 0;

    virtual std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) = 0;

protected:
    const Grid &grid;
};

template<typename T, typename Grid>
BoundaryCondition<T, Grid>::BoundaryCondition(const Grid &grid)
    : grid(grid)
{}

} // namespace STORM

#endif // STORM_BOUNDARY_CONDITION_HPP
