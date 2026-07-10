#ifndef STORM_RADIATION_OPACITY_MODEL_HPP
#define STORM_RADIATION_OPACITY_MODEL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "elementary/PointOps.hpp"
#include <units/units.hpp>

namespace STORM {

using namespace STORM::fallback;

template<typename PointT, typename GridT, typename CellT, std::size_t NumGroups>
class RadiationOpacityModel
{
public:
    using Point = PointT;
    using Grid = GridT;
    using Cell = CellT;
    using GroupArray = std::array<double, NumGroups>;
    using GroupBoundaries = std::array<double, NumGroups + 1>;

    virtual ~RadiationOpacityModel() = default;

    virtual double CalcPlanckOpacity(const CellT &cell) = 0;

    virtual double CalcAbsorptionOpacity(const CellT &cell, double frequency)
    {
        (void) frequency;
        return this->CalcPlanckOpacity(cell);
    }

    virtual double CalcScatteringOpacity(const CellT &cell)
    {
        (void) cell;
        return 0.0;
    }

    virtual double CalcScatteringOpacity(const CellT &cell, double frequency)
    {
        (void) frequency;
        return this->CalcScatteringOpacity(cell);
    }

    virtual PointT getRandomVelocity(const CellT &cell,
                                     std::mt19937_64 &rng,
                                     std::uniform_real_distribution<double> &dist)
    {
        (void) cell;
        return isotropicVelocity(rng, dist);
    }

    virtual PointT getNewScatterVelocity(const CellT &cell,
                                         const PointT &oldVelocity,
                                         double frequency,
                                         std::mt19937_64 &rng,
                                         std::uniform_real_distribution<double> &dist)
    {
        (void) oldVelocity;
        (void) frequency;
        return this->getRandomVelocity(cell, rng, dist);
    }

    virtual bool ComptonIncludedInTransport() const { return false; }

    virtual void reseed(std::uint64_t seed)
    {
        (void) seed;
    }

    virtual std::size_t findGroup(double frequency, const GroupBoundaries &boundaries) const
    {
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            if(frequency < boundaries[g + 1])
            {
                return g;
            }
        }
        return NumGroups - 1;
    }

    virtual double GetThermalEnergy(const CellT &cell, double random, const GroupBoundaries &boundaries) const
    {
        (void) cell;
        return boundaries[0] + random * (boundaries[NumGroups] - boundaries[0]);
    }

    virtual double SampleThermalEnergyInGroup(const CellT &cell, std::size_t group, double random, const GroupBoundaries &boundaries) const
    {
        (void) cell;
        return boundaries[group] + random * (boundaries[group + 1] - boundaries[group]);
    }

    virtual GroupArray GetThermalGroupPdf(const CellT &cell, const GroupBoundaries &boundaries) const
    {
        (void) cell;
        GroupArray pdf{};
        double total = boundaries[NumGroups] - boundaries[0];
        if(total <= 0.0)
        {
            return pdf;
        }
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            pdf[g] = (boundaries[g + 1] - boundaries[g]) / total;
        }
        return pdf;
    }

    virtual GroupArray GetCumulativeOpacity(const CellT &cell, const GroupBoundaries &boundaries) const
    {
        (void) cell;
        (void) boundaries;
        GroupArray cumOp{};
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            cumOp[g] = static_cast<double>(g + 1) / static_cast<double>(NumGroups);
        }
        return cumOp;
    }

    virtual GroupArray getEnergyCenters(const GroupBoundaries &boundaries) const
    {
        GroupArray centers{};
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            centers[g] = 0.5 * (boundaries[g] + boundaries[g + 1]);
        }
        return centers;
    }

protected:
    static PointT isotropicVelocity(std::mt19937_64 &rng,
                                    std::uniform_real_distribution<double> &dist)
    {
        constexpr double twoPi = 6.283185307179586476925286766559;
        double theta = twoPi * dist(rng);
        double cosTheta = 2.0 * dist(rng) - 1.0;
        double sinThetaSquared = std::max(0.0, 1.0 - cosTheta * cosTheta);
        double sinTheta = std::sqrt(sinThetaSquared);
        return PointT(sinTheta * std::cos(theta),
                      sinTheta * std::sin(theta),
                      cosTheta) * units::clight;
    }
};

} // namespace STORM

#endif // STORM_RADIATION_OPACITY_MODEL_HPP
