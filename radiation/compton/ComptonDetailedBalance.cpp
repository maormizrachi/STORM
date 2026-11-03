#include "radiation/compton/ComptonDetailedBalance.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include <planck_integral/planck_integral.hpp>
#include <units/units.hpp>

namespace STORM {

namespace {

// Centre-form Planck group energies B_g and occupations n_g at T (CMMC's set_Bg_ng).
void planckGroups(const std::vector<double> &centers,
                  const std::vector<double> &boundaries,
                  double temperature,
                  std::vector<double> &B,
                  std::vector<double> &n)
{
    std::size_t const groups = centers.size();
    B.assign(groups, 0.0);
    n.assign(groups, 0.0);
    double const pi = 3.141592653589793238462643383279502884;
    double const occupationFactor = std::pow(units::clight, 3) / (8.0 * pi * units::planck_constant);
    for(std::size_t g = 0; g < groups; ++g)
    {
        B[g] = planck_integral::planck_energy_density_group_integral(boundaries[g], boundaries[g + 1], temperature);
        double const nu = centers[g] / units::planck_constant;
        double const dnu = (boundaries[g + 1] - boundaries[g]) / units::planck_constant;
        n[g] = occupationFactor * B[g] / (nu * nu * nu * dnu);
    }
}

double balanceFactor(const std::vector<double> &centers,
                     const std::vector<double> &B,
                     const std::vector<double> &n,
                     std::size_t i,
                     std::size_t j)
{
    return (1.0 + n[j]) * B[i] * centers[j] / ((1.0 + n[i]) * B[j] * centers[i]);
}

} // namespace

void EnforceComptonDetailedBalance(const std::vector<double> &centers,
                                   const std::vector<double> &boundaries,
                                   double temperature,
                                   bool derivativesValid,
                                   bool adjustRates,
                                   ComptonMatrixData &data)
{
    std::size_t const groups = centers.size();
    std::vector<double> B, n, BUp, nUp, BDown, nDown;
    planckGroups(centers, boundaries, temperature, B, n);
    double constexpr relativeStep = 1e-4;
    if(derivativesValid)
    {
        planckGroups(centers, boundaries, temperature * (1.0 + relativeStep), BUp, nUp);
        planckGroups(centers, boundaries, temperature * (1.0 - relativeStep), BDown, nDown);
    }
    for(std::size_t i = 0; i < groups; ++i)
    {
        for(std::size_t j = i + 1; j < groups; ++j)
        {
            // CMMC's guard: skip pairs whose Planck weights underflow.
            if(B[j] * centers[i] < std::numeric_limits<double>::min() * 1e40)
            {
                continue;
            }
            double const f = balanceFactor(centers, B, n, i, j);
            if(std::isnan(f))
            {
                continue;
            }
            double df = 0.0;
            bool slope = derivativesValid;
            if(slope)
            {
                double const fUp = balanceFactor(centers, BUp, nUp, i, j);
                double const fDown = balanceFactor(centers, BDown, nDown, i, j);
                df = (fUp - fDown) / (2.0 * relativeStep * temperature);
                slope = std::isfinite(df);
            }
            if(f < 1.0)
            {
                if(adjustRates)
                {
                    data.tau[j][i] = data.tau[i][j] * f;
                }
                if(slope)
                {
                    data.dtau_dUm[j][i] = data.dtau_dUm[i][j] * f + data.tau[i][j] * df;
                }
            }
            else
            {
                if(adjustRates)
                {
                    data.tau[i][j] = data.tau[j][i] / f;
                }
                if(slope)
                {
                    // d(tau_ji / f)/dT, written to avoid overflowing f * f in the Wien tail.
                    data.dtau_dUm[i][j] = data.dtau_dUm[j][i] / f - (data.tau[j][i] / f) * (df / f);
                }
            }
        }
    }
}

} // namespace STORM
