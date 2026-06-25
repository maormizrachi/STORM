#ifndef RDMONT_POPULATION_CONTROL_HPP
#define RDMONT_POPULATION_CONTROL_HPP

#include "monte/particle/Particle.hpp"

namespace RDMont {

template<typename T, typename Grid>
class PopulationControl
{
public:
    using MCParticle = Particle<T, Grid>;

    PopulationControl(const Grid &grid);

    virtual ~PopulationControl() = default;

    virtual std::vector<MCParticle> activate(const std::vector<MCParticle> &particles) = 0;

protected:
    const Grid &grid;
};

template<typename T, typename Grid>
PopulationControl<T, Grid>::PopulationControl(const Grid &grid)
    : grid(grid)
{}

} // namespace RDMont

#endif // RDMONT_POPULATION_CONTROL_HPP
