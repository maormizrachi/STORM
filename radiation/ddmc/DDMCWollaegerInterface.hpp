#ifndef STORM_DDMC_WOLLAEGER_INTERFACE_HPP
#define STORM_DDMC_WOLLAEGER_INTERFACE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace STORM::ddmc {

// Wollaeger et al., ApJS 209, 36 (2013), Eq. (60).  The table is built
// lazily and interpolates mu*K(mu), preserving the grazing-angle behavior.
// This header deliberately has no dependency on a mesh or on RICH types.
constexpr double ExtrapolationLength = 0.7104;
constexpr double MinimumTabulatedMu = 1.0e-12;
constexpr double MaximumTabulatedMu = 1.0 - 1.0e-12;

namespace detail {

constexpr long double Pi = 3.141592653589793238462643383279502884L;

inline long double ClampOpenUnit(long double x)
{
    constexpr long double eps = 1.0e-15L;
    return std::max(eps, std::min(1.0L - eps, x));
}

inline long double Lambda(long double x)
{
    x = ClampOpenUnit(x);
    return 1.0L - x * std::atanh(x);
}

inline long double H(long double x)
{
    x = ClampOpenUnit(x);
    return 1.0L + 1.5L * x
         - 0.5L * x * (1.0L + 3.0L * x) * std::log1p(1.0L / x)
         + x * (2.5L + 3.0L * x) / (2.0L * (1.0L + x));
}

inline long double N(long double x)
{
    x = ClampOpenUnit(x);
    long double const lambda = Lambda(x);
    long double const imaginary = 0.5L * Pi * x;
    return x * (lambda * lambda + imaginary * imaginary);
}

inline long double Q(long double x)
{
    x = ClampOpenUnit(x);
    return x * H(x) / ((2.0L + 3.0L * x) * N(x));
}

inline long double QDerivative(long double x)
{
    x = ClampOpenUnit(x);
    long double const room = std::min(x, 1.0L - x);
    long double const dx = std::max(1.0e-10L,
        std::min(1.0e-5L, 1.0e-3L * room));
    long double const lo = ClampOpenUnit(x - dx);
    long double const hi = ClampOpenUnit(x + dx);
    return (Q(hi) - Q(lo)) / (hi - lo);
}

struct GaussLegendreRule
{
    std::vector<long double> x;
    std::vector<long double> w;

    explicit GaussLegendreRule(std::size_t n) : x(n), w(n)
    {
        std::size_t const half = (n + 1) / 2;
        for(std::size_t i = 0; i < half; ++i)
        {
            long double z = std::cos(Pi *
                (static_cast<long double>(i) + 0.75L) /
                (static_cast<long double>(n) + 0.5L));
            long double pp = 0.0L;
            for(int iteration = 0; iteration < 64; ++iteration)
            {
                long double p1 = 1.0L;
                long double p2 = 0.0L;
                for(std::size_t j = 1; j <= n; ++j)
                {
                    long double const p3 = p2;
                    p2 = p1;
                    p1 = ((2.0L * static_cast<long double>(j) - 1.0L) * z * p2
                        - (static_cast<long double>(j) - 1.0L) * p3) /
                        static_cast<long double>(j);
                }
                pp = static_cast<long double>(n) * (z * p1 - p2) /
                     (z * z - 1.0L);
                long double const next = z - p1 / pp;
                if(std::abs(next - z) < 1.0e-28L)
                {
                    z = next;
                    break;
                }
                z = next;
            }
            long double const mappedLo = 0.5L * (1.0L - z);
            long double const mappedHi = 0.5L * (1.0L + z);
            long double const weight = 1.0L /
                ((1.0L - z * z) * pp * pp);
            x[i] = mappedLo;
            x[n - 1 - i] = mappedHi;
            w[i] = weight;
            w[n - 1 - i] = weight;
        }
    }
};

inline long double EvaluateKernel(long double mu,
                                  GaussLegendreRule const &rule)
{
    mu = ClampOpenUnit(mu);
    long double const qMu = Q(mu);
    long double integral = 0.0L;
    for(std::size_t i = 0; i < rule.x.size(); ++i)
    {
        long double const x = rule.x[i];
        long double const dx = x - mu;
        long double dividedDifference;
        if(std::abs(dx) <= 1.0e-8L * std::max(1.0L, std::abs(mu)))
            dividedDifference = QDerivative(mu);
        else
            dividedDifference = (Q(x) - qMu) / dx;
        integral += rule.w[i] * dividedDifference;
    }

    long double const denominator = (2.0L + 3.0L * mu) * N(mu);
    long double const first = Lambda(mu) * H(mu) / denominator;
    long double const logarithmic = 0.5L * mu * H(mu) / denominator *
        std::log((1.0L - mu) / mu);
    return first + 0.5L * integral + logarithmic;
}

class KernelTable
{
public:
    KernelTable()
    {
        constexpr std::size_t lowCount = 512;
        constexpr std::size_t highCount = 512;
        constexpr std::size_t quadratureOrder = 192;
        constexpr double joinMu = 1.0e-2;

        mu_.reserve(lowCount + highCount);
        scaledKernel_.reserve(lowCount + highCount);
        GaussLegendreRule const rule(quadratureOrder);

        double const logLo = std::log(MinimumTabulatedMu);
        double const logHi = std::log(joinMu);
        for(std::size_t i = 0; i < lowCount; ++i)
        {
            double const s = static_cast<double>(i) /
                static_cast<double>(lowCount - 1);
            double const mu = std::exp(logLo + s * (logHi - logLo));
            mu_.push_back(mu);
            scaledKernel_.push_back(mu *
                static_cast<double>(EvaluateKernel(mu, rule)));
        }
        for(std::size_t i = 1; i < highCount; ++i)
        {
            double const s = static_cast<double>(i) /
                static_cast<double>(highCount - 1);
            double const mu = joinMu + s * (MaximumTabulatedMu - joinMu);
            mu_.push_back(mu);
            scaledKernel_.push_back(mu *
                static_cast<double>(EvaluateKernel(mu, rule)));
        }
    }

    double operator()(double mu) const
    {
        if(!std::isfinite(mu) || !(mu > 0.0) || mu > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if(mu < MinimumTabulatedMu)
            return scaledKernel_.front() / mu;
        if(mu > MaximumTabulatedMu)
            return scaledKernel_.back() / mu;

        auto upper = std::lower_bound(mu_.begin(), mu_.end(), mu);
        if(upper == mu_.begin())
            return scaledKernel_.front() / mu;
        if(upper == mu_.end())
            return scaledKernel_.back() / mu;
        std::size_t const hi = static_cast<std::size_t>(upper - mu_.begin());
        std::size_t const lo = hi - 1;
        double fraction;
        if(mu_[hi] <= 1.0e-2)
        {
            fraction = (std::log(mu) - std::log(mu_[lo])) /
                       (std::log(mu_[hi]) - std::log(mu_[lo]));
        }
        else
        {
            fraction = (mu - mu_[lo]) / (mu_[hi] - mu_[lo]);
        }
        double const scaled = scaledKernel_[lo] + fraction *
            (scaledKernel_[hi] - scaledKernel_[lo]);
        return scaled / mu;
    }

private:
    std::vector<double> mu_;
    std::vector<double> scaledKernel_;
};

} // namespace detail

inline double Kernel(double mu)
{
    static detail::KernelTable const table;
    double const value = table(mu);
    return std::isfinite(value)
        ? value : std::numeric_limits<double>::quiet_NaN();
}

inline double MovingFactor(double mu, double normalVelocityOverC)
{
    double const kernel = Kernel(mu);
    if(!std::isfinite(kernel) || !std::isfinite(normalVelocityOverC))
        return std::numeric_limits<double>::quiet_NaN();
    return 1.0 + 2.0 * normalVelocityOverC * kernel;
}

inline double StaticAdmissionProbability(double mu,
                                         double transportOpacity,
                                         double centerToFaceDistance)
{
    if(!(mu > 0.0) || !(transportOpacity > 0.0) ||
       !(centerToFaceDistance > 0.0))
        return 0.0;
    double const denominator = 3.0 *
        (transportOpacity * centerToFaceDistance + ExtrapolationLength);
    return std::clamp(2.0 * (1.0 + 1.5 * mu) / denominator, 0.0, 1.0);
}

inline double SampleAsymptoticMu(double random)
{
    random = std::clamp(random, 0.0, 1.0);
    double lo = 0.0;
    double hi = 1.0;
    // Detailed balance with the static admission law gives
    // p(mu) proportional to mu*(1 + 3*mu/2), whose normalized CDF is
    // mu^2*(1 + mu)/2.
    for(int iteration = 0; iteration < 56; ++iteration)
    {
        double const mu = 0.5 * (lo + hi);
        double const cdf = 0.5 * mu * mu * (1.0 + mu);
        if(cdf < random)
            lo = mu;
        else
            hi = mu;
    }
    return 0.5 * (lo + hi);
}

inline double BoundaryLeakRate(double area,
                               double volume,
                               double transportOpacity,
                               double centerToFaceDistance,
                               double lightSpeed)
{
    if(!(area > 0.0) || !(volume > 0.0) || !(transportOpacity > 0.0) ||
       !(centerToFaceDistance > 0.0) || !(lightSpeed > 0.0))
        return 0.0;
    double const opticalDenominator =
        transportOpacity * centerToFaceDistance + ExtrapolationLength;
    return lightSpeed * area /
        (3.0 * volume * opticalDenominator);
}

} // namespace STORM::ddmc

#endif // STORM_DDMC_WOLLAEGER_INTERFACE_HPP
