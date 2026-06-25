#ifndef STORM_OPACITY_MODEL_HPP
#define STORM_OPACITY_MODEL_HPP

#include <cstddef>

namespace STORM {

class OpacityModel
{
public:
    virtual ~OpacityModel() = default;

    virtual double PlanckOpacity(size_t cellIndex, double temperature) const = 0;

    virtual double ScatteringOpacity(size_t cellIndex, double temperature) const
    {
        (void) cellIndex;
        (void) temperature;
        return 0;
    }
};

} // namespace STORM

#endif // STORM_OPACITY_MODEL_HPP
