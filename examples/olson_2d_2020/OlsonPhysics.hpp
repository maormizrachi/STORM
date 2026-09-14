#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <units/units.hpp>
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationOpacityModel.hpp"

namespace Olson
{
constexpr double rho = 0.001;
// Fig. 4 geometry: the bottom half-block is cut by the reflecting y=0 axis.
inline bool IsAluminum(double x, double y)
{
    return (x >= .5 && x < 1.5 && y >= .5 && y < 1.5) ||
           (x >= 1.5 && x < 2.5 && ((y >= 1.5 && y < 2.5) || y < .5));
}
struct Cell : STORM::RadiationCell
{
    std::vector<double> tracers{0.};
};
inline double Alpha(double T, double chi)
{
    const double q = std::exp(-chi / T);
    return 2 * std::sqrt(q) / (std::sqrt(q + 4) + std::sqrt(q));
}
inline double Energy(double T, bool al)
{
    const double chi = al ? .3 : .1, H = al ? .5 : .1;
    return units::arad * std::pow(units::kev_kelvin, 4) * H * (T + (T + chi) * Alpha(T, chi));
}
inline double Cv(double T, bool al)
{
    const double chi = al ? .3 : .1, H = al ? .5 : .1, q = std::exp(-chi / T);
    const double a = Alpha(T, chi);
    const double da = chi / (T * T) * (a - std::sqrt(q / (q + 4)));
    return units::arad * std::pow(units::kev_kelvin, 3) * H * (1 + a + (T + chi) * da);
}
class EOS
{
  public:
    // STORM currently divides 4aT^3*sigma_P*c*dt by this return value,
    // without a rho factor. Hence this adapter returns VOLUMETRIC Cv.
    // de2T, separately, receives specific energy in erg/g.
    double dT2cv(double, double T, const std::vector<double> &tr, const std::vector<std::string> &) const
    {
        return Cv(T / units::kev_kelvin, tr.at(0) > .5);
    }
    double de2T(double density, double e, const std::vector<double> &tr, const std::vector<std::string> &) const
    {
        if(!(e > 0) || !std::isfinite(e))
        {
            throw std::runtime_error("Invalid Olson material energy");
        }
        const bool al = tr.at(0) > .5;
        double lo = 1e-12, hi = 1.;
        while(Energy(hi, al) < density * e)
        {
            hi *= 2;
        }
        for(int i = 0; i < 48; ++i)
        {
            double m = (lo + hi) / 2;
            if(Energy(m, al) < density * e)
            {
                lo = m;
            }
            else
            {
                hi = m;
            }
        }
        return (lo + hi) / 2 * units::kev_kelvin;
    }
};
// Equations (5),(9) of Steinberg & Heizler (2023), restating Olson (2020).
// Input photon energy and temperature in keV; OUTPUT MACROSCOPIC sigma [1/cm].
inline double Sigma(double E, double T, bool al)
{
    double k;
    if(al)
    {
        if(E < .01)
        {
            k = std::min(1e7, 1e8 * T);
        }
        else
        {
            const double a = 1e7 * std::pow(.01 / E, 2) / (1 + 20 * std::pow(T, 1.5));
            if(E < .1)
            {
                k = a;
            }
            else if(E < 1.5)
            {
                k = a + 1e6 * std::pow(.1 / E, 2) / (1 + 200 * T * T);
            }
            else
            {
                k = a * std::sqrt(1.5 / E) + 1e5 * std::pow(1.5 / E, 2.5) / (1 + 1000 * T * T);
            }
        }
    }
    else
    {
        if(E < .008)
        {
            k = std::min(1e7, 1e9 * T * T);
        }
        else
        {
            const double a = 3e6 * std::pow(.008 / E, 2) / (1 + 200 * std::pow(T, 1.5));
            k = E < .3 ? a : a * std::sqrt(.3 / E) + 4e4 * std::pow(.3 / E, 2.5) / (1 + 8000 * T * T);
        }
    }
    return rho * k;
}
// Continuous-frequency transport. Only emission integrals/CDFs are tabulated;
// every transport absorption uses the exact piecewise sigma(E,T) above.
// Log-energy quadrature is split exactly at every absorption edge.
class Spectrum
{
    static constexpr size_t NT = 2049;
    std::vector<double> logE_, table_;
    // Include the T=0.1 keV opacity cap transition exactly.
    static double tableTemperature(size_t i)
    {
        return i <= 1536 ? .0001 * std::exp(std::log(1000.) * i / 1536.)
                         : .1 * std::exp(std::log(20.) * (i - 1536) / 512.);
    }
    size_t stride_;
    std::pair<size_t, double> index(double T) const
    {
        if(T < .0001 || T > 2 || !std::isfinite(T))
        {
            throw std::runtime_error("Temperature outside verified spectral table");
        }
        double j = T <= .1 ? std::log(T / .0001) / std::log(1000.) * 1536
                           : 1536 + std::log(T / .1) / std::log(20.) * 512;
        size_t i = std::min(NT - 2, size_t(std::max(0., j)));
        return {i, j - i};
    }

  public:
    Spectrum()
    {
        for(double edge : {1e-7, .008, .01, .1, .3, 1.5, 100.})
        {
            logE_.push_back(std::log(edge));
        }
        for(int i = 1; i < 1536; ++i)
        {
            logE_.push_back(std::log(1e-7) + std::log(1e9) * i / 1536.);
        }
        std::sort(logE_.begin(), logE_.end());
        stride_ = logE_.size();
        table_.resize(2 * NT * stride_, 0.);
        constexpr double norm = 15 / (M_PI * M_PI * M_PI * M_PI);
        for(size_t m = 0; m < 2; ++m)
        {
            for(size_t t = 0; t < NT; ++t)
            {
                double T = tableTemperature(t);
                size_t base = (m * NT + t) * stride_;
                for(size_t j = 0; j + 1 < stride_; ++j)
                {
                    double w = 0;
                    // Two-point Gauss-Legendre in log E, split at every edge.
                    // b(E,T)dE = (15/pi^4) x^4/(exp(x)-1) dlogE.
                    for(double q : {-.5773502691896257, .5773502691896257})
                    {
                        double E = std::exp((logE_[j] + logE_[j + 1]) / 2 + q * (logE_[j + 1] - logE_[j]) / 2), x = E / T;
                        if(x < 700)
                        {
                            w += .5 * norm * std::pow(x, 4) / std::expm1(x) * Sigma(E, T, m);
                        }
                    }
                    table_[base + j + 1] = table_[base + j] + w * (logE_[j + 1] - logE_[j]);
                }
            }
        }
    }
    double Mean(double T, bool al) const
    {
        std::pair<size_t, double> indexResult = index(T);
        size_t i = indexResult.first;
        double f = indexResult.second;
        size_t b = (size_t(al) * NT + i) * stride_ + stride_ - 1;
        return (1 - f) * table_[b] + f * table_[b + stride_];
    }
    double Sample(double T, bool al, double u) const
    {
        std::pair<size_t, double> indexResult = index(T);
        size_t i = indexResult.first;
        double f = indexResult.second;
        size_t b = (size_t(al) * NT + i) * stride_;
        auto cdf = [&](size_t j)
        { return (1 - f) * table_[b + j] + f * table_[b + stride_ + j]; };
        double q = std::clamp(u, 0., std::nextafter(1., 0.)) * cdf(stride_ - 1);
        size_t lo = 0, hi = stride_ - 1;
        while(hi - lo > 1)
        {
            size_t m = (lo + hi) / 2;
            if(cdf(m) <= q)
            {
                lo = m;
            }
            else
            {
                hi = m;
            }
        }
        double w = cdf(hi) - cdf(lo), v = w > 0 ? (q - cdf(lo)) / w : .5;
        return std::exp(logE_[lo] + v * (logE_[hi] - logE_[lo]));
    }
};
template <class P, class Grid>
class Opacity : public STORM::RadiationOpacityModel<P, Grid, Cell, 1>
{
    Spectrum spectrum_;

  public:
    using Bounds = std::array<double, 2>;
    double CalcPlanckOpacity(const Cell &c) override
    {
        return spectrum_.Mean(c.temperature / units::kev_kelvin, c.tracers[0] > .5);
    }
    double CalcAbsorptionOpacity(const Cell &c, double E) override
    {
        return Sigma(E / units::kev, c.temperature / units::kev_kelvin, c.tracers[0] > .5);
    }
    double GetThermalEnergy(const Cell &c, double u, const Bounds &) const override
    {
        return units::kev * spectrum_.Sample(c.temperature / units::kev_kelvin, c.tracers[0] > .5, u);
    }
    double SampleThermalEnergyInGroup(const Cell &c, size_t, double u, const Bounds &b) const override
    {
        return GetThermalEnergy(c, u, b);
    }
    std::array<double, 1> GetThermalGroupPdf(const Cell &, const Bounds &) const override
    {
        return {1.};
    }
    std::array<double, 1> GetCumulativeOpacity(const Cell &, const Bounds &) const override
    {
        return {1.};
    }
};
} // namespace Olson
