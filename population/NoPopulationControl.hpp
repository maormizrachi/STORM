#ifndef STORM_NO_POPULATION_CONTROL_HPP
#define STORM_NO_POPULATION_CONTROL_HPP

#include "PopulationControl.hpp"

namespace STORM {

template<typename T, typename Grid>
class NoPopulationControl : public PopulationControl<T, Grid>
{
public:
    using MCParticle = Particle<T, Grid>;

    NoPopulationControl(const Grid &grid);

    std::vector<MCParticle> activate(const std::vector<MCParticle> &particles) override;
};

template<typename T, typename Grid>
NoPopulationControl<T, Grid>::NoPopulationControl(const Grid &grid)
    : PopulationControl<T, Grid>(grid)
{}

template<typename T, typename Grid>
std::vector<typename NoPopulationControl<T, Grid>::MCParticle> NoPopulationControl<T, Grid>::activate(const std::vector<MCParticle> &particles)
{
    return particles;
}

} // namespace STORM

#endif // STORM_NO_POPULATION_CONTROL_HPP
