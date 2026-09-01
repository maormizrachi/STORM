#ifndef STORM_CROOKED_PIPE_EOS_HPP
#define STORM_CROOKED_PIPE_EOS_HPP

#include <cmath>
#include <string>
#include <vector>

namespace STORM {
namespace examples {

// The two crooked-pipe materials are told apart by their density, which is the only
// per-cell quantity an EOS receives. RadiationIMC feeds dT2cv into the Fleck factor
// next to a*T^4 terms, so dT2cv must return heat capacity per unit volume, while
// de2T receives energy per unit mass.
class CrookedPipeEOS
{
public:
    CrookedPipeEOS(double thickCvPerVolume, double thinCvPerVolume, double thickDensity, double thinDensity)
        : thickCvPerVolume_(thickCvPerVolume),
          thinCvPerVolume_(thinCvPerVolume),
          thickDensity_(thickDensity),
          thinDensity_(thinDensity)
    {}

    double dT2cv(double density, double, const std::vector<double> &, const std::vector<std::string> &) const
    {
        return IsThick(density) ? thickCvPerVolume_ : thinCvPerVolume_;
    }

    double de2T(double density, double specificEnergy, const std::vector<double> &, const std::vector<std::string> &) const
    {
        return specificEnergy * density / (IsThick(density) ? thickCvPerVolume_ : thinCvPerVolume_);
    }

private:
    bool IsThick(double density) const
    {
        return std::abs(density - thickDensity_) < std::abs(density - thinDensity_);
    }

    double thickCvPerVolume_;
    double thinCvPerVolume_;
    double thickDensity_;
    double thinDensity_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_CROOKED_PIPE_EOS_HPP
