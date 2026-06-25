#ifndef RDMONT_HOHLRAUM_OPACITY_HPP
#define RDMONT_HOHLRAUM_OPACITY_HPP

#include <cmath>
#include <vector>
#include "monte/radiation/OpacityModel.hpp"

namespace RDMont {
namespace examples {

class HohlraumOpacity : public OpacityModel
{
public:
    HohlraumOpacity(const std::vector<int> &materialFlags)
        : materialFlags_(materialFlags) {}

    double PlanckOpacity(size_t cellIndex, double temperature) const override
    {
        if(this->materialFlags_[cellIndex])
        {
            double T_keV = temperature / kev_kelvin;
            T_keV = std::max(T_keV, 1e-4);
            return 300.0 * std::pow(T_keV, -3.0);
        }
        return 1e-20;
    }

    double ScatteringOpacity(size_t cellIndex, double temperature) const override
    {
        (void) cellIndex;
        (void) temperature;
        return 0;
    }

private:
    static constexpr double ev = 1.602176634e-12;
    static constexpr double kev = 1e3 * ev;
    static constexpr double kev_kelvin = kev / 1.380649e-16;
    const std::vector<int> &materialFlags_;
};

} // namespace examples
} // namespace RDMont

#endif // RDMONT_HOHLRAUM_OPACITY_HPP
