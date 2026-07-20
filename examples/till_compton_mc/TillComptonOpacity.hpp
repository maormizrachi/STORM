#ifndef STORM_TILL_COMPTON_OPACITY_HPP
#define STORM_TILL_COMPTON_OPACITY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "radiation/RadiationOpacityModel.hpp"
#include "utils/LinearInterpolation.hpp"
#include <planck_integral/planck_integral.hpp>
#include <units/units.hpp>

namespace STORM {
namespace examples {

/*
 * The opacity used by the original RICH Till-Compton case.  The old case
 * evaluates the free-free opacity in c.g.s. units at photon energy E.  This
 * class keeps that convention explicit: STORM passes photon energies (not
 * frequencies) to CalcAbsorptionOpacity.
 */
template<typename PointT, typename GridT, typename CellT, std::size_t NumGroups>
class TillComptonOpacity
    : public RadiationOpacityModel<PointT, GridT, CellT, NumGroups>
{
public:
    using Base = RadiationOpacityModel<PointT, GridT, CellT, NumGroups>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, NumGroups + 1>;

    TillComptonOpacity(GroupBoundaries energyBoundaries,
                       bool includePlasmaCutoff)
        : energyBoundaries_(energyBoundaries),
          includePlasmaCutoff_(includePlasmaCutoff)
    {}

    double CalcPlanckOpacity(const CellT &cell) override
    {
        const double temperature = safeTemperature(cell.temperature);
        const double kT = units::k_boltz * temperature;
        double weightedOpacity = 0.0;
        double totalWeight = 0.0;

        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            const double lower = energyBoundaries_[group];
            const double upper = energyBoundaries_[group + 1];
            const double weight = planck_integral::planck_integral(lower / kT, upper / kT);
            const double center = 0.5 * (lower + upper);
            const double opacity = calcAbsorptionOpacity(cell, center);
            if(weight > 0.0 && std::isfinite(weight) &&
               opacity >= 0.0 && std::isfinite(opacity))
            {
                weightedOpacity += opacity * weight;
                totalWeight += weight;
            }
        }

        return totalWeight > 0.0 ? weightedOpacity / totalWeight : 0.0;
    }

    double CalcAbsorptionOpacity(const CellT &cell, double energy) override
    {
        return calcAbsorptionOpacity(cell, energy);
    }

    double CalcScatteringOpacity(const CellT &) override { return 0.0; }
    double CalcScatteringOpacity(const CellT &, double) override { return 0.0; }

    double GetThermalEnergy(const CellT &cell, double random,
                            const GroupBoundaries &boundaries) const override
    {
        const GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        const double total = cumulative[NumGroups];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }
        return LinearInterpolation(cumulative, boundaries,
                                    clampUnitOpen(random) * total);
    }

    double SampleThermalEnergyInGroup(const CellT &cell, std::size_t group,
                                      double random,
                                      const GroupBoundaries &boundaries) const override
    {
        group = std::min(group, NumGroups - 1);
        const GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        const double lower = cumulative[group];
        const double upper = cumulative[group + 1];
        if(!(upper > lower) || !std::isfinite(upper - lower))
        {
            return 0.5 * (boundaries[group] + boundaries[group + 1]);
        }
        return LinearInterpolation(cumulative, boundaries,
                                   lower + clampUnitOpen(random) * (upper - lower));
    }

    GroupArray GetThermalGroupPdf(const CellT &cell,
                                  const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        const GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        const double total = cumulative[NumGroups];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return pdf;
        }
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            const double weight = cumulative[group + 1] - cumulative[group];
            pdf[group] = weight > 0.0 && std::isfinite(weight) ? weight / total : 0.0;
        }
        return pdf;
    }

    GroupArray GetCumulativeOpacity(const CellT &cell,
                                    const GroupBoundaries &boundaries) const override
    {
        const GroupCdf cumulative = computeCumulativePlanck(cell, boundaries);
        GroupArray cumulativeOpacity{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            cumulativeOpacity[group] = cumulative[group + 1];
        }
        return cumulativeOpacity;
    }

private:
    double calcAbsorptionOpacity(const CellT &cell, double energy) const
    {
        constexpr double protonMass = 1.6726231e-24;
        constexpr double electronCharge = 4.8032e-10;
        constexpr double pi = 3.141592653589793238462643383279502884;

        const double temperature = safeTemperature(cell.temperature);
        const double frequency = std::max(energy, energyBoundaries_[0]) /
                                  units::planck_constant;
        const double ionDensity = std::max(0.0, cell.density) / protonMass;
        const double electronDensity = ionDensity; // fully ionized hydrogen
        const double kT = units::k_boltz * temperature;

        if(!(frequency > 0.0) || !(ionDensity > 0.0) || !(electronDensity > 0.0))
        {
            return 0.0;
        }

        const double gauntArgument = std::exp(5.960) * std::pow(temperature, 1.5) /
                                     frequency;
        const double gauntFactor = std::max(1.0, std::log(std::max(gauntArgument, 1.0)));

        double plasmaCutoffFactor = 1.0;
        if(includePlasmaCutoff_)
        {
            const double plasmaFrequency =
                std::sqrt(4.0 * pi * electronDensity * electronCharge * electronCharge /
                          units::me) /
                (2.0 * pi);
            if(frequency <= plasmaFrequency)
            {
                return 0.0;
            }
            const double ratio = plasmaFrequency / frequency;
            plasmaCutoffFactor = 1.0 / std::sqrt(std::max(1.0 - ratio * ratio, 0.0));
        }

        const double stimulatedFactor = -std::expm1(-energy / kT);
        const double absorption = 3.7e8 * electronDensity * ionDensity /
                                  std::sqrt(temperature) * stimulatedFactor /
                                  std::pow(frequency, 3.0);
        const double result = gauntFactor * absorption * plasmaCutoffFactor;
        return std::isfinite(result) && result >= 0.0 ? result : 0.0;
    }
    static double safeTemperature(double temperature)
    {
        return std::max(temperature, 1.0);
    }

    static double clampUnitOpen(double random)
    {
        const double upper = std::nextafter(1.0, 0.0);
        return std::isfinite(random) ? std::clamp(random, 0.0, upper) : 0.5;
    }

    GroupCdf computeCumulativePlanck(const CellT &cell,
                                     const GroupBoundaries &boundaries) const
    {
        GroupCdf cumulative{};
        cumulative[0] = 0.0;
        const double temperature = safeTemperature(cell.temperature);
        const double kT = units::k_boltz * temperature;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            const double lower = boundaries[group];
            const double upper = boundaries[group + 1];
            const double planckWeight = planck_integral::planck_integral(lower / kT,
                                                                          upper / kT);
            const double center = 0.5 * (lower + upper);
            const double weight = planckWeight * calcAbsorptionOpacity(cell, center);
            cumulative[group + 1] = cumulative[group] +
                (weight > 0.0 && std::isfinite(weight) ? weight : 0.0);
        }
        return cumulative;
    }

    GroupBoundaries energyBoundaries_{};
    bool includePlasmaCutoff_ = true;
};

} // namespace examples
} // namespace STORM

#endif // STORM_TILL_COMPTON_OPACITY_HPP
