#ifndef STORM_DENSMORE_OPACITY_HPP
#define STORM_DENSMORE_OPACITY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/RadiationCell.hpp"
#include "PhysicalConstants.hpp"
#include "utils/LinearInterpolation.hpp"
#include <planck_integral/planck_integral.hpp>

namespace STORM {
namespace examples {

constexpr size_t N_DENSMORE_GROUPS = 30;

template<typename PointT, typename GridT>
class DensmoreOpacity : public RadiationOpacityModel<PointT, GridT, RadiationCell, N_DENSMORE_GROUPS>
{
public:
    using Base = RadiationOpacityModel<PointT, GridT, RadiationCell, N_DENSMORE_GROUPS>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, N_DENSMORE_GROUPS + 1>;

    DensmoreOpacity(const std::vector<int> &regionFlags, const std::vector<RadiationCell> &cells)
        : regionFlags_(regionFlags), cells_(&cells)
    {
        sigma0_left_ = 10.0 * std::pow(constants::kev, 3.5);
        sigma0_right_ = 1000.0 * std::pow(constants::kev, 3.5);
    }

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        double sigma0 = regionFlags_[idx] ? sigma0_left_ : sigma0_right_;
        double kT = constants::k_boltz * std::max(cell.temperature, 1.0);
        double sqrtKT = std::sqrt(kT);

        double weightedSum = 0;
        double totalWeight = 0;
        for(size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double Ec = 0.5 * (groupBounds_[g] + groupBounds_[g + 1]);
            double sigma_g = sigma0 / (sqrtKT * Ec * Ec * Ec);
            double a = groupBounds_[g] / kT;
            double b = groupBounds_[g + 1] / kT;
            double Bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 0.0;
            weightedSum += sigma_g * Bg;
            totalWeight += Bg;
        }
        return (totalWeight > 0) ? weightedSum / totalWeight : 1e-20;
    }

    double CalcAbsorptionOpacity(const RadiationCell &cell, double frequency) override
    {
        std::size_t idx = cellIndex(cell);
        double sigma0 = regionFlags_[idx] ? sigma0_left_ : sigma0_right_;
        double kT = constants::k_boltz * std::max(cell.temperature, 1.0);
        double sqrtKT = std::sqrt(kT);
        double E = std::max(frequency, groupBounds_[0]);
        return sigma0 / (sqrtKT * E * E * E);
    }

    double CalcScatteringOpacity(const RadiationCell &) override { return 0.0; }
    double CalcScatteringOpacity(const RadiationCell &, double) override { return 0.0; }

    double GetThermalEnergy(const RadiationCell &cell, double random,
                            const GroupBoundaries &boundaries) const override
    {
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        double total = cumulative[N_DENSMORE_GROUPS];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }
        double r = clampUnitOpen(random);
        return LinearInterpolation(cumulative, boundaries, r * total);
    }

    double SampleThermalEnergyInGroup(const RadiationCell &cell, std::size_t group, double random,
                                       const GroupBoundaries &boundaries) const override
    {
        group = std::min<std::size_t>(group, N_DENSMORE_GROUPS - 1);
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

    GroupArray GetThermalGroupPdf(const RadiationCell &cell, const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        double total = cumulative[N_DENSMORE_GROUPS];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return pdf;
        }
        for(std::size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double weight = cumulative[g + 1] - cumulative[g];
            pdf[g] = (weight > 0.0 && std::isfinite(weight)) ? weight / total : 0.0;
        }
        return pdf;
    }

    GroupArray GetCumulativeOpacity(const RadiationCell &cell, const GroupBoundaries &boundaries) const override
    {
        GroupArray cumulativeUpper{};
        GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        for(std::size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            cumulativeUpper[g] = cumulative[g + 1];
        }
        return cumulativeUpper;
    }

    void setGroupBoundaries(const GroupBoundaries &bounds)
    {
        groupBounds_ = bounds;
    }

private:
    static double clampUnitOpen(double random)
    {
        double upper = std::nextafter(1.0, 0.0);
        return std::isfinite(random) ? std::clamp(random, 0.0, upper) : 0.5;
    }

    std::size_t cellIndex(const RadiationCell &cell) const
    {
        return static_cast<std::size_t>(&cell - cells_->data());
    }

    GroupCdf computeCumulativePlanck(const RadiationCell &cell, const GroupBoundaries &boundaries) const
    {
        GroupCdf cdf{};
        cdf[0] = 0.0;
        std::size_t idx = cellIndex(cell);
        double sigma0 = regionFlags_[idx] ? sigma0_left_ : sigma0_right_;
        double kT = constants::k_boltz * std::max(cell.temperature, 1.0);
        double sqrtKT = std::sqrt(kT);
        for(std::size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double a = boundaries[g] / kT;
            double b = boundaries[g + 1] / kT;
            double bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 0.0;
            double Ec = 0.5 * (boundaries[g] + boundaries[g + 1]);
            double sigma_g = sigma0 / (sqrtKT * Ec * Ec * Ec);
            double weight = (sigma_g > 0.0 && std::isfinite(sigma_g) && std::isfinite(bg)) ? sigma_g * bg : 0.0;
            cdf[g + 1] = cdf[g] + weight;
        }
        return cdf;
    }

    const std::vector<int> &regionFlags_;
    const std::vector<RadiationCell> *cells_;
    double sigma0_left_, sigma0_right_;
    GroupBoundaries groupBounds_{};
};

} // namespace examples
} // namespace STORM

#endif // STORM_DENSMORE_OPACITY_HPP
