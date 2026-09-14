#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <units/units.hpp>
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationOpacityModel.hpp"

// Su & Olson (1996): rho Cv=alpha T^3, epsilon=4a/alpha=1.
// sigma_a=sigma_t=1/cm; x_dimensionless=sqrt(3)*sigma*x,
// tau=epsilon*c*sigma*t; u=E/(a Tb^4), v=(T/Tb)^4.
class SuOlsonEOS
{
  public:
    double dT2cv(double, double T, const std::vector<double> &,
                 const std::vector<std::string> &) const
    {
        return 4 * units::arad * T * T * T; // volumetric Cv for STORM's Fleck expression
    }
    double de2T(double rho, double e, const std::vector<double> &,
                const std::vector<std::string> &) const
    {
        if(e < 0 || !std::isfinite(e))
        {
            throw std::runtime_error("Invalid material energy");
        }
        return std::pow(rho * e / units::arad, 0.25);
    }
};
template <class P, class Grid>
class SuOlsonOpacity : public STORM::RadiationOpacityModel<P, Grid, STORM::RadiationCell, 1>
{
  public:
    double CalcPlanckOpacity(const STORM::RadiationCell &) override
    {
        return 1.;
    }
    double CalcAbsorptionOpacity(const STORM::RadiationCell &, double) override
    {
        return 1.;
    }
};
