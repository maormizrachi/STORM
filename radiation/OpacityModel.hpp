#ifndef RDMONT_OPACITY_MODEL_HPP
#define RDMONT_OPACITY_MODEL_HPP

#include <cstddef>

namespace RDMont {

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

} // namespace RDMont

#endif // RDMONT_OPACITY_MODEL_HPP
