#ifndef STORM_MOVING_SLAB_OPACITY_HPP
#define STORM_MOVING_SLAB_OPACITY_HPP

#include <cstddef>
#include <cmath>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/RadiationCell.hpp"

/*
 * 124-group opacity table for the moving slab benchmark
 * (McClarren & Gentile 2021).  Each row gives [nu_min, nu_max] in keV
 * and the mass absorption coefficient kappa in cm^2/g.
 */
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

/*
 * Gray Planck-mean opacity for the moving slab benchmark.
 *
 * Computes kappa_P = sum_g(sigma_g * B_g) / sum_g(B_g) at each cell's
 * temperature using the 124-group table above.  Vacuum cells (density <
 * 0.5 * rho_slab) return a near-zero opacity so photons free-stream.
 */
template<typename PointT, typename GridT>
class MovingSlabOpacity : public RadiationOpacityModel<PointT, GridT, RadiationCell, 1>
{
public:
    MovingSlabOpacity(double rhoSlab, const std::vector<double> &densities,
                      const std::vector<RadiationCell> &cells)
        : rhoSlab_(rhoSlab), densities_(densities), cells_(&cells)
    {
        for(size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            sigmaG_[g] = rhoSlab_ * OPACITY_TABLE_124[g].kappa;
        }
    }

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        if(densities_[idx] < 0.5 * rhoSlab_)
        {
            return 1e-30;
        }

        double numerator = 0.0;
        double denominator = 0.0;
        for(size_t g = 0; g < N_OPACITY_GROUPS; ++g)
        {
            double Elo = OPACITY_TABLE_124[g].nu_min * KEV_ERG;
            double Ehi = OPACITY_TABLE_124[g].nu_max * KEV_ERG;
            double Bg = PlanckGroupIntegral(Elo, Ehi, cell.temperature);
            numerator += sigmaG_[g] * Bg;
            denominator += Bg;
        }
        if(denominator <= 0.0)
        {
            return 0.0;
        }
        return numerator / denominator;
    }

    double CalcScatteringOpacity(const RadiationCell &) override
    {
        return 0.0;
    }

private:
    std::size_t cellIndex(const RadiationCell &cell) const
    {
        return static_cast<std::size_t>(&cell - cells_->data());
    }

    double rhoSlab_;
    std::vector<double> densities_;
    const std::vector<RadiationCell> *cells_;
    double sigmaG_[N_OPACITY_GROUPS];

    static constexpr double KEV_ERG = 1.602176634e-9;
    static constexpr double K_BOLTZ = 1.380649e-16;
    static constexpr double HPLANCK = 6.62607015e-27;
    static constexpr double CLIGHT  = 2.99792458e10;

    static double PlanckGroupIntegral(double Elo, double Ehi, double T_kelvin)
    {
        double kT = K_BOLTZ * T_kelvin;
        if(kT <= 0)
        {
            return 0.0;
        }
        return PlanckPrimitive(Ehi / kT) - PlanckPrimitive(Elo / kT);
    }

    static double PlanckPrimitive(double x)
    {
        if(x <= 0)
        {
            return 0.0;
        }
        double prefactor = 8.0 * M_PI / (HPLANCK * CLIGHT * HPLANCK * CLIGHT * HPLANCK * CLIGHT);
        prefactor *= K_BOLTZ * K_BOLTZ * K_BOLTZ * K_BOLTZ;

        if(x < 2.0)
        {
            double x2 = x * x;
            double x3 = x2 * x;
            return prefactor * x3 * (1.0 / 3.0 + x * (-1.0 / 8.0 + x * (1.0 / 60.0
                + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 272160.0
                + x2 * (-1.0 / 13305600.0 + x2 / 622702080.0))))));
        }

        double sum = 0.0;
        for(int n = 1; n <= 5; ++n)
        {
            double emx = std::exp(-n * x);
            double inv_n = 1.0 / n;
            sum += emx * (x * x * x * inv_n + 3.0 * x * x * inv_n * inv_n
                + 6.0 * x * inv_n * inv_n * inv_n
                + 6.0 * inv_n * inv_n * inv_n * inv_n);
        }
        return prefactor * sum;
    }
};

} // namespace examples
} // namespace STORM

#endif // STORM_MOVING_SLAB_OPACITY_HPP
