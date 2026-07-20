#ifndef STORM_DDMC_GEOMETRY_HPP
#define STORM_DDMC_GEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace STORM::ddmc {

inline double TwoSidedResistance(double sourceDistance,
                                 double sourceDiffusion,
                                 double targetDistance,
                                 double targetDiffusion)
{
    if(!(sourceDistance > 0.0) || !(targetDistance > 0.0) ||
       !(sourceDiffusion > 0.0) || !(targetDiffusion > 0.0))
        return std::numeric_limits<double>::infinity();
    double const resistance = sourceDistance / sourceDiffusion +
        targetDistance / targetDiffusion;
    return std::isfinite(resistance) && resistance > 0.0
        ? resistance : std::numeric_limits<double>::infinity();
}

inline double TwoSidedConductance(double area,
                                  double sourceDistance,
                                  double sourceDiffusion,
                                  double targetDistance,
                                  double targetDiffusion)
{
    double const resistance = TwoSidedResistance(
        sourceDistance, sourceDiffusion, targetDistance, targetDiffusion);
    if(!(area > 0.0) || !std::isfinite(resistance))
        return 0.0;
    double const conductance = area / resistance;
    return std::isfinite(conductance) && conductance > 0.0
        ? conductance : 0.0;
}

inline double LeakageRate(double area,
                          double volume,
                          double sourceDistance,
                          double sourceDiffusion,
                          double targetDistance,
                          double targetDiffusion)
{
    double const conductance = TwoSidedConductance(
        area, sourceDistance, sourceDiffusion,
        targetDistance, targetDiffusion);
    if(!(volume > 0.0))
        return 0.0;
    double const rate = conductance / volume;
    return std::isfinite(rate) && rate > 0.0 ? rate : 0.0;
}

inline double ReciprocityResidual(double sourceVolume,
                                  double sourceRate,
                                  double targetVolume,
                                  double targetRate)
{
    double const lhs = sourceVolume * sourceRate;
    double const rhs = targetVolume * targetRate;
    double const scale = std::max({std::abs(lhs), std::abs(rhs),
                                   std::numeric_limits<double>::min()});
    return std::abs(lhs - rhs) / scale;
}

} // namespace STORM::ddmc

#endif // STORM_DDMC_GEOMETRY_HPP
