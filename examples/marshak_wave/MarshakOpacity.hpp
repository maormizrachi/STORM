#ifndef STORM_MARSHAK_OPACITY_HPP
#define STORM_MARSHAK_OPACITY_HPP

#include <cmath>
#include <vector>
#include "radiation/OpacityModel.hpp"
#include "PhysicalConstants.hpp"

namespace STORM {
namespace examples {

/*
 * Power-law opacity for the Marshak wave benchmarks:
 *
 *   kappa(T, rho) = kappa0 * (keV_K / T)^alpha * rho^beta
 *
 * PlanckOpacity returns kappa_P, and ScatteringOpacity returns kappa_R - kappa_P
 * so that the total transport opacity equals kappa_R (correct diffusion limit).
 */
class MarshakOpacity : public OpacityModel
{
public:
    MarshakOpacity(double kappaP0, double kappaR0, double alpha, double betaRho,
                   const std::vector<double> &densities)
        : kappaP0_(kappaP0), kappaR0_(kappaR0), alpha_(alpha), betaRho_(betaRho), densities_(densities)
    {}

    double PlanckOpacity(size_t cellIndex, double temperature) const override
    {
        double rhoFactor = (betaRho_ != 0.0) ? std::pow(densities_[cellIndex], betaRho_) : 1.0;
        double Tfactor = std::pow(constants::kev_kelvin / std::max(temperature, 1.0), alpha_);
        return kappaP0_ * Tfactor * rhoFactor;
    }

    double ScatteringOpacity(size_t cellIndex, double temperature) const override
    {
        double rhoFactor = (betaRho_ != 0.0) ? std::pow(densities_[cellIndex], betaRho_) : 1.0;
        double Tfactor = std::pow(constants::kev_kelvin / std::max(temperature, 1.0), alpha_);
        double kappaR = kappaR0_ * Tfactor * rhoFactor;
        double kappaP = kappaP0_ * Tfactor * rhoFactor;
        return std::max(kappaR - kappaP, 0.0);
    }

private:
    double kappaP0_, kappaR0_;
    double alpha_;
    double betaRho_;
    const std::vector<double> &densities_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_MARSHAK_OPACITY_HPP
