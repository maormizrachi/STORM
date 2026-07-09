#ifndef STORM_MOVING_SLAB_OPACITY_HPP
#define STORM_MOVING_SLAB_OPACITY_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "utils/LinearInterpolation.hpp"
#include <planck_integral/planck_integral.hpp>
#include <units/units.hpp>

namespace STORM {
namespace examples {

struct OpacityRow { double nu_min; double nu_max; double kappa; };

static const OpacityRow OPACITY_TABLE_124[124] = {
    {1.000e-03, 1.229e-03, 1.000e+04}, {1.229e-03, 1.510e-03, 1.000e+04},
    {1.510e-03, 1.856e-03, 1.000e+04}, {1.856e-03, 2.281e-03, 1.000e+04},
    {2.281e-03, 2.804e-03, 1.000e+04}, {2.804e-03, 3.446e-03, 1.000e+04},
    {3.446e-03, 4.234e-03, 1.000e+04}, {4.234e-03, 5.204e-03, 1.000e+04},
    {5.204e-03, 6.396e-03, 1.000e+04}, {6.396e-03, 7.860e-03, 1.000e+04},
    {7.860e-03, 9.660e-03, 1.000e+04}, {9.660e-03, 1.187e-02, 1.000e+04},
    {1.187e-02, 1.459e-02, 1.000e+04}, {1.459e-02, 1.793e-02, 1.000e+04},
    {1.793e-02, 2.204e-02, 1.000e+04}, {2.204e-02, 2.708e-02, 8.933e+03},
    {2.708e-02, 3.328e-02, 8.569e+03}, {3.328e-02, 4.090e-02, 7.335e+03},
    {4.090e-02, 5.027e-02, 5.656e+03}, {5.027e-02, 6.178e-02, 4.031e+03},
    {6.178e-02, 7.593e-02, 2.710e+03}, {7.593e-02, 9.331e-02, 1.770e+03},
    {9.331e-02, 1.147e-01, 1.184e+03}, {1.147e-01, 1.409e-01, 7.924e+02},
    {1.409e-01, 1.732e-01, 5.061e+02}, {1.732e-01, 2.129e-01, 3.230e+02},
    {2.129e-01, 2.616e-01, 2.062e+02}, {2.616e-01, 3.215e-01, 2.100e+02},
    {3.215e-01, 3.951e-01, 1.229e+02}, {3.951e-01, 4.856e-01, 7.579e+01},
    {4.856e-01, 5.968e-01, 4.905e+01}, {5.968e-01, 7.334e-01, 3.110e+01},
    {7.334e-01, 9.014e-01, 1.947e+01}, {9.014e-01, 1.000e+00, 1.196e+01},
    {1.000e+00, 1.014e+00, 1.187e+01}, {1.014e+00, 1.028e+00, 1.149e+01},
    {1.028e+00, 1.042e+00, 1.112e+01}, {1.042e+00, 1.057e+00, 1.076e+01},
    {1.057e+00, 1.072e+00, 1.041e+01}, {1.072e+00, 1.087e+00, 1.007e+01},
    {1.087e+00, 1.102e+00, 9.740e+00}, {1.102e+00, 1.117e+00, 9.416e+00},
    {1.117e+00, 1.133e+00, 9.098e+00}, {1.133e+00, 1.149e+00, 8.785e+00},
    {1.149e+00, 1.165e+00, 8.477e+00}, {1.165e+00, 1.181e+00, 8.180e+00},
    {1.181e+00, 1.198e+00, 7.900e+00}, {1.198e+00, 1.214e+00, 7.635e+00},
    {1.214e+00, 1.231e+00, 7.381e+00}, {1.231e+00, 1.248e+00, 7.138e+00},
    {1.248e+00, 1.266e+00, 6.902e+00}, {1.266e+00, 1.283e+00, 6.674e+00},
    {1.283e+00, 1.301e+00, 6.452e+00}, {1.301e+00, 1.319e+00, 6.237e+00},
    {1.319e+00, 1.338e+00, 6.029e+00}, {1.338e+00, 1.357e+00, 5.827e+00},
    {1.357e+00, 1.375e+00, 5.631e+00}, {1.375e+00, 1.395e+00, 5.438e+00},
    {1.395e+00, 1.414e+00, 5.250e+00}, {1.414e+00, 1.434e+00, 5.066e+00},
    {1.434e+00, 1.454e+00, 4.886e+00}, {1.454e+00, 1.474e+00, 4.709e+00},
    {1.474e+00, 1.495e+00, 4.542e+00}, {1.495e+00, 1.516e+00, 4.387e+00},
    {1.516e+00, 1.537e+00, 4.243e+00}, {1.537e+00, 1.558e+00, 4.117e+00},
    {1.558e+00, 1.580e+00, 4.310e+00}, {1.580e+00, 1.602e+00, 1.572e+01},
    {1.602e+00, 1.625e+00, 4.834e+00}, {1.625e+00, 1.647e+00, 3.726e+00},
    {1.647e+00, 1.670e+00, 3.758e+00}, {1.670e+00, 1.694e+00, 4.706e+00},
    {1.694e+00, 1.717e+00, 3.394e+01}, {1.717e+00, 1.741e+00, 9.034e+02},
    {1.741e+00, 1.765e+00, 1.615e+01}, {1.765e+00, 1.790e+00, 4.098e+00},
    {1.790e+00, 1.815e+00, 3.420e+00}, {1.815e+00, 1.840e+00, 3.389e+00},
    {1.840e+00, 1.866e+00, 3.986e+00}, {1.866e+00, 1.892e+00, 4.350e+00},
    {1.892e+00, 1.919e+00, 3.933e+00}, {1.919e+00, 1.945e+00, 4.258e+00},
    {1.945e+00, 1.972e+00, 4.861e+00}, {1.972e+00, 1.995e+00, 6.836e+00},
    {1.995e+00, 2.089e+00, 4.674e+01}, {2.089e+00, 2.188e+00, 2.108e+01},
    {2.188e+00, 2.291e+00, 2.281e+01}, {2.291e+00, 2.399e+00, 1.963e+01},
    {2.399e+00, 2.512e+00, 1.749e+01}, {2.512e+00, 2.630e+00, 1.590e+01},
    {2.630e+00, 2.754e+00, 1.442e+01}, {2.754e+00, 2.884e+00, 1.294e+01},
    {2.884e+00, 3.020e+00, 1.144e+01}, {3.020e+00, 3.162e+00, 1.014e+01},
    {3.162e+00, 3.311e+00, 9.047e+00}, {3.311e+00, 3.467e+00, 8.057e+00},
    {3.467e+00, 3.631e+00, 7.118e+00}, {3.631e+00, 3.802e+00, 6.219e+00},
    {3.802e+00, 3.981e+00, 5.474e+00}, {3.981e+00, 4.169e+00, 4.861e+00},
    {4.169e+00, 4.365e+00, 4.311e+00}, {4.365e+00, 4.571e+00, 3.792e+00},
    {4.571e+00, 4.786e+00, 3.296e+00}, {4.786e+00, 5.012e+00, 2.888e+00},
    {5.012e+00, 5.248e+00, 2.555e+00}, {5.248e+00, 5.495e+00, 2.258e+00},
    {5.495e+00, 5.754e+00, 1.978e+00}, {5.754e+00, 6.026e+00, 1.713e+00},
    {6.026e+00, 6.310e+00, 1.496e+00}, {6.310e+00, 6.607e+00, 1.320e+00},
    {6.607e+00, 6.918e+00, 1.163e+00}, {6.918e+00, 7.244e+00, 1.016e+00},
    {7.244e+00, 7.586e+00, 8.770e-01}, {7.586e+00, 7.943e+00, 7.641e-01},
    {7.943e+00, 8.318e+00, 6.729e-01}, {8.318e+00, 8.710e+00, 5.919e-01},
    {8.710e+00, 9.120e+00, 5.160e-01}, {9.120e+00, 9.550e+00, 4.442e-01},
    {9.550e+00, 1.070e+01, 3.862e-01}, {1.070e+01, 1.315e+01, 2.385e-01},
    {1.315e+01, 1.616e+01, 1.309e-01}, {1.616e+01, 1.986e+01, 7.143e-02},
    {1.986e+01, 2.441e+01, 3.867e-02}, {2.441e+01, 3.000e+01, 2.076e-02},
};
constexpr size_t N_OPACITY_GROUPS = 124;

template<typename PointT, typename GridT, typename CellT>
class MovingSlabOpacity : public RadiationOpacityModel<PointT, GridT, CellT, N_OPACITY_GROUPS>
{
public:
    using Base = RadiationOpacityModel<PointT, GridT, CellT, N_OPACITY_GROUPS>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, N_OPACITY_GROUPS + 1>;

    MovingSlabOpacity(double rhoSlab, const std::vector<CellT> &cells)
        : rhoSlab_(rhoSlab), cells_(&cells)
    {
        for(size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            sigmaG_[g] = rhoSlab_ * OPACITY_TABLE_124[g].kappa;
        }
    }

    double CalcPlanckOpacity(const CellT &cell) override
    {
        if(cell.density < 0.5 * rhoSlab_)
        {
            return 1e-12;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        for(size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            double Elo = OPACITY_TABLE_124[g].nu_min * units::kev;
            double Ehi = OPACITY_TABLE_124[g].nu_max * units::kev;
            double Bg = planck_integral::planck_energy_density_group_integral(Elo, Ehi, cell.temperature);
            numerator += sigmaG_[g] * Bg;
            denominator += Bg;
        }
        if(denominator <= 0.0)
        {
            return 0.0;
        }
        return numerator / denominator;
    }

    double CalcAbsorptionOpacity(const CellT &cell, double energy) override
    {
        if(cell.density < 0.5 * rhoSlab_)
        {
            return 1e-12;
        }
        size_t g = findGroupByEnergy(energy);
        if(g >= N_OPACITY_GROUPS)
        {
            return 1e-100;
        }
        return sigmaG_[g];
    }

    double CalcScatteringOpacity(const CellT &) override { return 0.0; }
    double CalcScatteringOpacity(const CellT &, double) override { return 0.0; }

    double GetThermalEnergy(const CellT &cell, double random,
                            const GroupBoundaries &boundaries) const override
    {
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        double total = cumulative[N_OPACITY_GROUPS];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }
        double r = clampUnitOpen(random);
        return LinearInterpolation(cumulative, boundaries, r * total);
    }

    double SampleThermalEnergyInGroup(const CellT &cell, std::size_t group, double random,
                                       const GroupBoundaries &boundaries) const override
    {
        group = std::min<std::size_t>(group, N_OPACITY_GROUPS - 1);
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        double c0 = cumulative[group];
        double c1 = cumulative[group + 1];
        if(c1 <= c0 || !std::isfinite(c1 - c0))
        {
            return 0.5 * (boundaries[group] + boundaries[group + 1]);
        }
        double r = clampUnitOpen(random);
        return LinearInterpolation(cumulative, boundaries, c0 + r * (c1 - c0));
    }

    GroupArray GetThermalGroupPdf(const CellT &cell, const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        double total = cumulative[N_OPACITY_GROUPS];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return pdf;
        }
        for(std::size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            double weight = cumulative[g + 1] - cumulative[g];
            pdf[g] = (weight > 0.0 && std::isfinite(weight)) ? weight / total : 0.0;
        }
        return pdf;
    }

    GroupArray GetCumulativeOpacity(const CellT &cell, const GroupBoundaries &boundaries) const override
    {
        GroupArray cumulativeUpper{};
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        for(std::size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            cumulativeUpper[g] = cumulative[g + 1];
        }
        return cumulativeUpper;
    }

    GroupArray getEnergyCenters(const GroupBoundaries &boundaries) const override
    {
        GroupArray centers{};
        for(std::size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            centers[g] = 0.5 * (boundaries[g] + boundaries[g + 1]);
        }
        return centers;
    }

private:
    static double clampUnitOpen(double random)
    {
        double upper = std::nextafter(1.0, 0.0);
        return std::isfinite(random) ? std::clamp(random, 0.0, upper) : 0.5;
    }

    size_t findGroupByEnergy(double energy) const
    {
        for(size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            if(energy < OPACITY_TABLE_124[g].nu_max * units::kev)
            {
                return g;
            }
        }
        return N_OPACITY_GROUPS - 1;
    }

    GroupCdf computeCumulativePlanck(const CellT &cell, const GroupBoundaries &boundaries) const
    {
        GroupCdf cdf{};
        cdf[0] = 0.0;
        double kT = constants::k_boltz * cell.temperature;
        bool isVacuum = (cell.density < 0.5 * rhoSlab_);
        for(std::size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            double a = boundaries[g] / kT;
            double b = boundaries[g + 1] / kT;
            double bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 0.0;
            double sigma = isVacuum ? 1e-12 : sigmaG_[g];
            double weight = (sigma > 0.0 && std::isfinite(sigma) && std::isfinite(bg)) ? sigma * bg : 0.0;
            cdf[g + 1] = cdf[g] + weight;
        }
        return cdf;
    }

    double rhoSlab_;
    const std::vector<CellT> *cells_;
    double sigmaG_[N_OPACITY_GROUPS];
};

} // namespace examples
} // namespace STORM

#endif // STORM_MOVING_SLAB_OPACITY_HPP
