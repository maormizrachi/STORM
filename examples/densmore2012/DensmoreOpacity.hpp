#ifndef STORM_DENSMORE_OPACITY_HPP
#define STORM_DENSMORE_OPACITY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/ddmc/DDMCSampling.hpp"
#include <units/units.hpp>
#include <planck_integral/planck_integral.hpp>

namespace STORM {
namespace examples {

constexpr size_t N_DENSMORE_GROUPS = 30;

template<typename PointT, typename GridT>
class DensmoreOpacity final : public RadiationOpacityModel<PointT, GridT, RadiationCell, N_DENSMORE_GROUPS>
{
public:
    STORM::PortableAbsorptionLaw GetPortableAbsorptionLaw() const override
    { return STORM::PortableAbsorptionLaw::InverseCube; }
    STORM::ThermalFrequencyLaw GetPortableThermalFrequencyLaw() const override
    { return STORM::ThermalFrequencyLaw::BoseEinstein0; }

    using Base = RadiationOpacityModel<PointT, GridT, RadiationCell, N_DENSMORE_GROUPS>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, N_DENSMORE_GROUPS + 1>;

    DensmoreOpacity(
        const std::vector<int> &regionFlags,
        const std::vector<RadiationCell> &cells,
        const double opacityScale = 1.0)
        : regionFlags_(regionFlags), cells_(&cells)
    {
        sigma0_left_ =
            opacityScale * 10.0 * std::pow(units::kev, 3.5);
        sigma0_right_ =
            opacityScale * 1000.0 * std::pow(units::kev, 3.5);
    }

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        double sigma0 = regionFlags_[idx] ? sigma0_left_ : sigma0_right_;
        double kT = units::k_boltz * std::max(cell.temperature, 1.0);
        const GroupBoundaries &bounds = groupBounds_;

        double weightedOpacityIntegral = 0.0;
        double planckWeight = 0.0;
        for(size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double a = bounds[g] / kT;
            double b = bounds[g + 1] / kT;
            if(!(a > 0.0 && b > a))
            {
                continue;
            }
            const double bose0 = ddmc::BoseEinstein0Integral(a, b);
            const double planck = planck_integral::planck_integral(a, b);
            if(bose0 > 0.0 && std::isfinite(bose0))
            {
                weightedOpacityIntegral += bose0;
            }
            if(planck > 0.0 && std::isfinite(planck))
            {
                planckWeight += planck;
            }
        }
        if(!(planckWeight > 0.0) || !(weightedOpacityIntegral > 0.0))
        {
            return 1e-20;
        }
        // σ(E)B(E)dE / B(E)dE, with B ∝ x^3/(e^x-1) and σ ∝ E^{-3}.
        // planck_integral includes 15/π^4, so restore that factor on ∫dx/(e^x-1).
        constexpr double pi = 3.14159265358979323846;
        constexpr double planckNorm = 15.0 / (pi * pi * pi * pi);
        const double kT3 = kT * kT * kT;
        const double sigmaP =
            (sigma0 / (std::sqrt(kT) * kT3)) *
            (planckNorm * weightedOpacityIntegral / planckWeight);
        return (std::isfinite(sigmaP) && sigmaP > 0.0) ? sigmaP : 1e-20;
    }

    double CalcAbsorptionOpacity(const RadiationCell &cell, double frequency) override
    {
        std::size_t idx = cellIndex(cell);
        double sigma0 = regionFlags_[idx] ? sigma0_left_ : sigma0_right_;
        double kT = units::k_boltz * std::max(cell.temperature, 1.0);
        double sqrtKT = std::sqrt(kT);
        double E = std::max(frequency, groupBounds_[0]);
        return sigma0 / (sqrtKT * E * E * E);
    }

    double CalcScatteringOpacity(const RadiationCell &) override { return 0.0; }
    double CalcScatteringOpacity(const RadiationCell &, double) override { return 0.0; }

    double GetThermalEnergy(const RadiationCell &cell, double random,
                            const GroupBoundaries &boundaries) const override
    {
        GroupCdf cumulative = computeCumulativeOpacityWeightedPlanck(cell, boundaries);
        double total = cumulative[N_DENSMORE_GROUPS];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }
        double r = clampUnitOpen(random) * total;
        std::size_t group = 0;
        while(group + 1 < N_DENSMORE_GROUPS && cumulative[group + 1] < r)
        {
            ++group;
        }
        const double lower = cumulative[group];
        const double width = cumulative[group + 1] - lower;
        const double local =
            (width > 0.0 && std::isfinite(width)) ? (r - lower) / width : 0.5;
        return SampleThermalEnergyInGroup(cell, group, local, boundaries);
    }

    double SampleThermalEnergyInGroup(const RadiationCell &cell, std::size_t group, double random,
                                       const GroupBoundaries &boundaries) const override
    {
        group = std::min<std::size_t>(group, N_DENSMORE_GROUPS - 1);
        double kT = units::k_boltz * std::max(cell.temperature, 1.0);
        return ddmc::SampleBoseEinstein0FrequencyInGroup(
            boundaries.data(), N_DENSMORE_GROUPS, group, kT, clampUnitOpen(random));
    }

    GroupArray GetThermalGroupPdf(const RadiationCell &cell, const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        GroupCdf cumulative = computeCumulativeOpacityWeightedPlanck(cell, boundaries);
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
        GroupCdf cumulative = computeCumulativeOpacityWeightedPlanck(cell, boundaries);
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

    GroupCdf computeCumulativeOpacityWeightedPlanck(const RadiationCell &cell, const GroupBoundaries &boundaries) const
    {
        GroupCdf cdf{};
        cdf[0] = 0.0;
        double kT = units::k_boltz * std::max(cell.temperature, 1.0);
        for(std::size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double a = boundaries[g] / kT;
            double b = boundaries[g + 1] / kT;
            const double weight = ddmc::BoseEinstein0Integral(a, b);
            cdf[g + 1] = cdf[g] + ((weight > 0.0 && std::isfinite(weight)) ? weight : 0.0);
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
