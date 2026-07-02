#ifndef STORM_DENSMORE_OPACITY_HPP
#define STORM_DENSMORE_OPACITY_HPP

#include <cmath>
#include <vector>
#include "radiation/OpacityModel.hpp"
#include "PhysicalConstants.hpp"

namespace STORM {
namespace examples {

/*
 * Gray Planck-mean opacity for the Densmore et al. (2012) heterogeneous
 * step-opacity problem.
 *
 * The frequency-dependent opacity is sigma(E) = sigma0 / (sqrt(kT) * E^3),
 * with sigma0 = 10 keV^{3.5}/cm for x < x_step, 1000 keV^{3.5}/cm otherwise.
 *
 * The Planck mean is computed numerically over the bounded energy range
 * [Emin, Emax] using a simple sum over Ngroups log-spaced energy groups:
 *
 *   sigma_P = sum_g[ sigma_g * B_g ] / sum_g[ B_g ]
 *
 * where B_g = integral of Planck function over group g.
 */
class DensmoreOpacity : public OpacityModel
{
public:
    DensmoreOpacity(const std::vector<int> &regionFlags, size_t Ngroups = 30)
        : regionFlags_(regionFlags), Ngroups_(Ngroups)
    {
        double Emin = constants::kev * 1e-4;
        double Emax = constants::kev * 1e2;
        groupBoundaries_.resize(Ngroups + 1);
        groupCenters_.resize(Ngroups);
        groupBoundaries_[0] = Emin;
        for(size_t g = 0; g < Ngroups; g++)
        {
            groupBoundaries_[g + 1] = std::pow(Emax / Emin, 1.0 / Ngroups) * groupBoundaries_[g];
            groupCenters_[g] = 0.5 * (groupBoundaries_[g] + groupBoundaries_[g + 1]);
        }

        sigma0_left_ = 10.0 * std::pow(constants::kev, 3.5);
        sigma0_right_ = 1000.0 * std::pow(constants::kev, 3.5);
    }

    double PlanckOpacity(size_t cellIndex, double temperature) const override
    {
        double sigma0 = regionFlags_[cellIndex] ? sigma0_left_ : sigma0_right_;
        double kT = constants::k_boltz * std::max(temperature, 1.0);
        double sqrtKT = std::sqrt(kT);

        double weightedSum = 0;
        double totalWeight = 0;
        for(size_t g = 0; g < Ngroups_; g++)
        {
            double a = groupBoundaries_[g] / kT;
            double b = groupBoundaries_[g + 1] / kT;
            double Bg = PlanckIntegral(a, b);
            double sigma_g = sigma0 / (sqrtKT * groupCenters_[g] * groupCenters_[g] * groupCenters_[g]);
            weightedSum += sigma_g * Bg;
            totalWeight += Bg;
        }
        return (totalWeight > 0) ? weightedSum / totalWeight : 1e-20;
    }

    double ScatteringOpacity(size_t /*cellIndex*/, double /*temperature*/) const override
    {
        return 0;
    }

private:
    static double PlanckIntegral(double a, double b)
    {
        size_t N = 64;
        double h = (b - a) / N;
        double sum = 0;
        for(size_t i = 0; i <= N; i++)
        {
            double x = a + i * h;
            double f = 0;
            if(x > 0 and x < 500)
            {
                f = x * x * x / (std::exp(x) - 1.0);
            }
            double w = (i == 0 or i == N) ? 0.5 : 1.0;
            sum += w * f;
        }
        return sum * h;
    }

    const std::vector<int> &regionFlags_;
    size_t Ngroups_;
    double sigma0_left_, sigma0_right_;
    std::vector<double> groupBoundaries_;
    std::vector<double> groupCenters_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_DENSMORE_OPACITY_HPP
