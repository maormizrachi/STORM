#ifndef STORM_MARSHAK_OPACITY_HPP
#define STORM_MARSHAK_OPACITY_HPP

#include <cmath>
#include <cstddef>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/RadiationCell.hpp"
#include <units/units.hpp>

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
template<typename PointT, typename GridT>
class MarshakOpacity : public RadiationOpacityModel<PointT, GridT, RadiationCell, 1>
{
public:
    MarshakOpacity(double kappaP0, double kappaR0, double alpha, double betaRho,
                   const std::vector<double> &densities, const std::vector<RadiationCell> &cells)
        : kappaP0_(kappaP0), kappaR0_(kappaR0), alpha_(alpha), betaRho_(betaRho),
          densities_(densities), cells_(&cells)
    {}

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        double rhoFactor = (betaRho_ != 0.0) ? std::pow(densities_[idx], betaRho_) : 1.0;
        double Tfactor = std::pow(units::kev_kelvin / std::max(cell.temperature, 1.0), alpha_);
        return kappaP0_ * Tfactor * rhoFactor;
    }

    double CalcScatteringOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        double rhoFactor = (betaRho_ != 0.0) ? std::pow(densities_[idx], betaRho_) : 1.0;
        double Tfactor = std::pow(units::kev_kelvin / std::max(cell.temperature, 1.0), alpha_);
        double kappaR = kappaR0_ * Tfactor * rhoFactor;
        double kappaP = kappaP0_ * Tfactor * rhoFactor;
        return std::max(kappaR - kappaP, 0.0);
    }

private:
    std::size_t cellIndex(const RadiationCell &cell) const
    {
        return static_cast<std::size_t>(&cell - cells_->data());
    }

    double kappaP0_, kappaR0_;
    double alpha_;
    double betaRho_;
    const std::vector<double> &densities_;
    const std::vector<RadiationCell> *cells_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_MARSHAK_OPACITY_HPP
