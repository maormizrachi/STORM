#ifndef STORM_DDMC_SAMPLING_HPP
#define STORM_DDMC_SAMPLING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <planck_integral/planck_integral.hpp>

namespace STORM::ddmc {

template<typename Boundaries>
double PlanckBandMass(const Boundaries &boundaries,
                      double kT,
                      std::size_t beginGroup,
                      std::size_t endGroup)
{
    if(!(kT > 0.0) || !std::isfinite(kT) || beginGroup >= endGroup)
        return 0.0;

    double mass = 0.0;
    for(std::size_t group = beginGroup; group < endGroup; ++group)
    {
        double const contribution = planck_integral::planck_integral(
            boundaries[group] / kT, boundaries[group + 1] / kT);
        if(std::isfinite(contribution) && contribution > 0.0)
            mass += contribution;
    }
    return std::isfinite(mass) && mass > 0.0 ? mass : 0.0;
}

template<typename Boundaries>
double PlanckBandFraction(const Boundaries &boundaries,
                          double kT,
                          std::size_t beginGroup,
                          std::size_t endGroup,
                          std::size_t totalEndGroup)
{
    double const denominator = PlanckBandMass(
        boundaries, kT, beginGroup, totalEndGroup);
    if(!(denominator > 0.0))
        return 0.0;
    return std::clamp(PlanckBandMass(
        boundaries, kT, beginGroup, endGroup) / denominator, 0.0, 1.0);
}

} // namespace STORM::ddmc

#endif // STORM_DDMC_SAMPLING_HPP
