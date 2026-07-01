#ifndef STORM_RANDOM_WALK_HPP
#define STORM_RANDOM_WALK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace STORM {

struct PGRWCellData
{
    std::size_t groupCutoff = 0;
    double sigmaA_bar = 0.0;
    double sigmaT_bar = 0.0;
    double D = 0.0;
    double gamma = 1.0;
};

class RandomWalk
{
public:
    RandomWalk()
    {
        tauTable_.resize(tableSize_);
        survivalTable_.resize(tableSize_);

        double logMin = std::log(tauMin_);
        double logMax = std::log(tauMax_);
        for(std::size_t i = 0; i < tableSize_; ++i)
        {
            double frac = static_cast<double>(i) / static_cast<double>(tableSize_ - 1);
            double tau = std::exp(logMin + frac * (logMax - logMin));
            tauTable_[i] = tau;
            survivalTable_[i] = computeSurvival(tau);
        }

        static constexpr std::size_t xGridSize = 512;
        radiusTable_.resize(tableSize_ * radiusTableSize_);

        for(std::size_t i = 0; i < tableSize_; ++i)
        {
            double tau = tauTable_[i];
            std::vector<double> gGrid(xGridSize);
            for(std::size_t k = 0; k < xGridSize; ++k)
            {
                double x = static_cast<double>(k) / static_cast<double>(xGridSize - 1);
                gGrid[k] = computeRadialCDF(x, tau);
            }

            for(std::size_t j = 0; j < radiusTableSize_; ++j)
            {
                double xi = static_cast<double>(j) / static_cast<double>(radiusTableSize_ - 1);
                if(xi <= gGrid[0])
                {
                    radiusTable_[i * radiusTableSize_ + j] = 0.0;
                }
                else if(xi >= gGrid[xGridSize - 1])
                {
                    radiusTable_[i * radiusTableSize_ + j] = 1.0;
                }
                else
                {
                    auto it = std::lower_bound(gGrid.begin(), gGrid.end(), xi);
                    std::size_t idx = static_cast<std::size_t>(it - gGrid.begin());
                    if(idx == 0)
                    {
                        idx = 1;
                    }
                    double g0 = gGrid[idx - 1];
                    double g1 = gGrid[idx];
                    double x0 = static_cast<double>(idx - 1) / static_cast<double>(xGridSize - 1);
                    double x1 = static_cast<double>(idx) / static_cast<double>(xGridSize - 1);
                    double f = (g1 > g0) ? (xi - g0) / (g1 - g0) : 0.5;
                    radiusTable_[i * radiusTableSize_ + j] = x0 + f * (x1 - x0);
                }
            }
        }
    }

    double sampleLeakTime(double xi) const
    {
        double target = 1.0 - xi;
        if(target >= survivalTable_.front())
        {
            return tauTable_.front();
        }
        if(target <= survivalTable_.back())
        {
            return tauTable_.back();
        }

        auto it = std::lower_bound(survivalTable_.begin(), survivalTable_.end(), target,
                                   [](double a, double b) { return a > b; });
        if(it == survivalTable_.begin())
        {
            return tauTable_.front();
        }
        if(it == survivalTable_.end())
        {
            return tauTable_.back();
        }

        std::size_t idx = static_cast<std::size_t>(it - survivalTable_.begin());
        double S0 = survivalTable_[idx - 1];
        double S1 = survivalTable_[idx];
        double t0 = tauTable_[idx - 1];
        double t1 = tauTable_[idx];
        double frac = (S0 - target) / (S0 - S1);
        double logT = std::log(t0) + frac * (std::log(t1) - std::log(t0));
        return std::exp(logT);
    }

    double sampleRadius(double tau, double xi) const
    {
        double logTau = std::log(std::clamp(tau, tauMin_, tauMax_));
        double logMin = std::log(tauMin_);
        double logMax = std::log(tauMax_);
        double tauPos = (logTau - logMin) / (logMax - logMin) * static_cast<double>(tableSize_ - 1);
        std::size_t tauIdx = std::min(static_cast<std::size_t>(tauPos), tableSize_ - 2);
        double tauFrac = tauPos - static_cast<double>(tauIdx);

        double xiClamped = std::clamp(xi, 0.0, 1.0);
        double xiPos = xiClamped * static_cast<double>(radiusTableSize_ - 1);
        std::size_t xiIdx = std::min(static_cast<std::size_t>(xiPos), radiusTableSize_ - 2);
        double xiFrac = xiPos - static_cast<double>(xiIdx);

        double r00 = radiusTable_[tauIdx * radiusTableSize_ + xiIdx];
        double r01 = radiusTable_[tauIdx * radiusTableSize_ + xiIdx + 1];
        double r10 = radiusTable_[(tauIdx + 1) * radiusTableSize_ + xiIdx];
        double r11 = radiusTable_[(tauIdx + 1) * radiusTableSize_ + xiIdx + 1];

        double r0 = r00 + xiFrac * (r01 - r00);
        double r1 = r10 + xiFrac * (r11 - r10);

        return std::clamp(r0 + tauFrac * (r1 - r0), 0.0, 1.0);
    }

    double computeSurvival(double tau) const
    {
        if(tau <= 0.0)
        {
            return 1.0;
        }
        if(tau < 1e-4)
        {
            return 1.0;
        }
        return survivalEigen(tau);
    }

    double computeRadialCDF(double x, double tau) const
    {
        if(tau <= 0.0)
        {
            return (x > 0.0) ? 1.0 : 0.0;
        }
        if(x <= 0.0)
        {
            return 0.0;
        }
        if(x >= 1.0)
        {
            return 1.0;
        }
        return (tau < tauFreeSpace_) ? radialCdfFreeSpace(x, tau) : radialCdfEigen(x, tau);
    }

private:
    static constexpr double PI_ = 3.14159265358979323846;
    static constexpr int eigenTerms_ = 500;
    static constexpr double tauFreeSpace_ = 0.01;
    static constexpr double tauMin_ = 1e-8;
    static constexpr double tauMax_ = 64.0;
    static constexpr std::size_t tableSize_ = 1024;
    static constexpr std::size_t radiusTableSize_ = 256;

    double survivalEigen(double tau) const
    {
        double sum = 0.0;
        for(int n = 1; n <= eigenTerms_; ++n)
        {
            double arg = -static_cast<double>(n * n) * PI_ * PI_ * tau;
            if(arg < -700.0)
            {
                break;
            }
            double term = std::exp(arg);
            sum += (n % 2 == 1) ? term : -term;
        }
        return std::clamp(2.0 * sum, 0.0, 1.0);
    }

    static double radialCdfFreeSpace(double x, double tau)
    {
        if(x <= 0.0)
        {
            return 0.0;
        }
        if(x >= 1.0)
        {
            return 1.0;
        }
        double a = x / (2.0 * std::sqrt(tau));
        return std::clamp(std::erf(a) - 2.0 * a * std::exp(-a * a) / std::sqrt(PI_), 0.0, 1.0);
    }

    double radialCdfEigen(double x, double tau) const
    {
        if(x <= 0.0)
        {
            return 0.0;
        }
        if(x >= 1.0)
        {
            return 1.0;
        }
        double S = computeSurvival(tau);
        if(S < 1e-30)
        {
            return 1.0;
        }
        double sum = 0.0;
        for(int n = 1; n <= eigenTerms_; ++n)
        {
            double arg = -static_cast<double>(n * n) * PI_ * PI_ * tau;
            if(arg < -700.0)
            {
                break;
            }
            double npi = static_cast<double>(n) * PI_;
            double npx = npi * x;
            double term = std::exp(arg) * (std::sin(npx) / npi - x * std::cos(npx));
            sum += term;
        }
        return std::clamp(2.0 * sum / S, 0.0, 1.0);
    }

    std::vector<double> tauTable_;
    std::vector<double> survivalTable_;
    std::vector<double> radiusTable_;
};

} // namespace STORM

#endif // STORM_RANDOM_WALK_HPP
