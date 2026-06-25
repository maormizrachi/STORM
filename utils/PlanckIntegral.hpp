#ifndef RDMONT_PLANCK_INTEGRAL_HPP
#define RDMONT_PLANCK_INTEGRAL_HPP

#include <boost/math/special_functions/pow.hpp>
#include <cmath>
#include <cassert>

namespace RDMont {
namespace planck_integral {

static int constexpr N_clark = 5;
static double constexpr x_clark = 2.;

inline double Clark_Taylor(double const x)
{
    double const x2 = x * x;
    double const x3 = x2 * x;
    return x3 * (1. / 3. + x * (-1. / 8. + x * (1. / 60. + x2 * (-1. / 5040. + x2 * (1. / 272160. + x2 * (-1. / 13305600. + x2 / 622702080.))))));
}

inline double Clark_series(double const x)
{
    double const x2 = x * x;
    double const x3 = x2 * x;
    double sum = 0.;
    double const exp_val = std::exp(-x);
    double expn = exp_val;
    for(int n = 1; n <= N_clark; ++n)
    {
        double const in = 1. / static_cast<double>(n);
        sum += in * (x3 + in * (3. * x2 + 6. * in * (x + in))) * expn;
        expn *= exp_val;
    }
    return -sum;
}

inline double PlanckIntegral(double const a, double const b)
{
    assert(a < b);
    using boost::math::pow;
    static double constexpr coeff = 15. / pow<4>(M_PI);
    if(a > x_clark)
    {
        return coeff * (Clark_series(b) - Clark_series(a));
    }
    if(b < x_clark)
    {
        return coeff * (Clark_Taylor(b) - Clark_Taylor(a));
    }
    return 1.0 + coeff * (Clark_series(b) - Clark_Taylor(a));
}

} // namespace planck_integral
} // namespace RDMont

#endif // RDMONT_PLANCK_INTEGRAL_HPP
