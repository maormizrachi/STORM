#ifndef STORM_POPULATION_CONTROL_HPP
#define STORM_POPULATION_CONTROL_HPP

#include <stdexcept>
#include <vector>

#include "../particle/Particle.hpp"

#ifdef STORM_WITH_GPU
namespace STORM
{
namespace gpu
{
struct DevicePopulationContext;
}
}
#endif

namespace STORM {

template<typename T, typename Grid>
class PopulationControl
{
public:
    using MCParticle = Particle<T>;

    PopulationControl(const Grid &grid);

    virtual ~PopulationControl() = default;

    virtual std::vector<MCParticle> activate(const std::vector<MCParticle> &particles) = 0;
    virtual bool IsIdentity() const
    {
        return false;
    }

#ifdef STORM_WITH_GPU
    virtual bool SupportsDeviceActivation() const
    {
        return false;
    }

    virtual void activateDevice(gpu::DevicePopulationContext &context) const;
#endif

protected:
    const Grid &grid;
};

template<typename T, typename Grid>
PopulationControl<T, Grid>::PopulationControl(const Grid &grid)
    : grid(grid)
{}

#ifdef STORM_WITH_GPU
template<typename T, typename Grid>
void PopulationControl<T, Grid>::activateDevice(gpu::DevicePopulationContext &context) const
{
    (void)context;
    throw std::logic_error("Population control does not support device activation");
}
#endif

} // namespace STORM

#endif // STORM_POPULATION_CONTROL_HPP
