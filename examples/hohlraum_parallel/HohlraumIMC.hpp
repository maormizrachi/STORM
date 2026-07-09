#ifndef STORM_HOHLRAUM_IMC_HPP
#define STORM_HOHLRAUM_IMC_HPP

#include <cmath>
#include <string>
#include <vector>

#include "PhysicalConstants.hpp"

namespace STORM {
namespace examples {

class HohlraumEOS
{
public:
    HohlraumEOS(double cvPerVolumeMaterial, double cvPerVolumeVacuum,
                double densityMaterial, double densityVacuum)
        : cvPerMassMaterial_(cvPerVolumeMaterial / densityMaterial),
          cvPerMassVacuum_(cvPerVolumeVacuum / densityVacuum),
          densityMaterial_(densityMaterial),
          densityVacuum_(densityVacuum)
    {}

    double dT2cv(double density, double /*temperature*/,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return isMaterial(density) ? cvPerMassMaterial_ : cvPerMassVacuum_;
    }

    double de2T(double density, double specificEnergy,
                const std::vector<double> &tracers, const std::vector<std::string> &tracerNames) const
    {
        double cvPerMass = dT2cv(density, 0.0, tracers, tracerNames);
        return (cvPerMass > 0.0) ? specificEnergy / cvPerMass : 0.0;
    }

private:
    bool isMaterial(double density) const
    {
        return std::abs(density - densityMaterial_) < std::abs(density - densityVacuum_);
    }

    double cvPerMassMaterial_;
    double cvPerMassVacuum_;
    double densityMaterial_;
    double densityVacuum_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_HOHLRAUM_IMC_HPP
