#ifndef STORM_DDMC_SAMPLING_HPP
#define STORM_DDMC_SAMPLING_HPP

#include <cstddef>

#include "../transport/TransportPortability.hpp"

namespace STORM::ddmc {

// Matches source/Radiation/planck_integral/planck_integral.hpp (cgs Boltzmann).
static constexpr double boltzmannConstant = 1.380649e-16;

STORM_TRANSPORT_INLINE
double ClarkTaylor(const double x)
{
    const double x2 = x * x;
    const double x3 = x2 * x;
    return x3 * (1.0 / 3.0 +
                 x * (-1.0 / 8.0 +
                      x * (1.0 / 60.0 +
                           x2 * (-1.0 / 5040.0 +
                                 x2 * (1.0 / 272160.0 +
                                       x2 * (-1.0 / 13305600.0 +
                                             x2 / 622702080.0))))));
}

STORM_TRANSPORT_INLINE
double ClarkSeries(const double x)
{
    const double x2 = x * x;
    const double x3 = x2 * x;
    double sum = 0.0;
    for(int n = 1; n <= 5; ++n)
    {
        const double in = 1.0 / static_cast<double>(n);
        sum += in *
            (x3 + in * (3.0 * x2 + 6.0 * in * (x + in))) *
            transport::Exp(-(x * static_cast<double>(n)));
    }
    return -sum;
}

STORM_TRANSPORT_INLINE
double PlanckIntegral(const double a, const double b)
{
    if(!(a < b) || !transport::IsFinite(a) || !transport::IsFinite(b))
    {
        return 0.0;
    }
    constexpr double pi = 3.14159265358979323846;
    constexpr double coeff = 15.0 / (pi * pi * pi * pi);
    constexpr double xClark = 2.0;
    if(a > xClark)
    {
        return coeff * (ClarkSeries(b) - ClarkSeries(a));
    }
    if(b < xClark)
    {
        return coeff * (ClarkTaylor(b) - ClarkTaylor(a));
    }
    return 1.0 + coeff * (ClarkSeries(b) - ClarkTaylor(a));
}

template<typename Boundaries>
STORM_TRANSPORT_INLINE
double PlanckBandMass(const Boundaries &boundaries,
                      const double kT,
                      const std::size_t beginGroup,
                      const std::size_t endGroup)
{
    if(!(kT > 0.0) || !transport::IsFinite(kT) || beginGroup >= endGroup)
    {
        return 0.0;
    }

    double mass = 0.0;
    for(std::size_t group = beginGroup; group < endGroup; ++group)
    {
        const double contribution =
            PlanckIntegral(boundaries[group] / kT, boundaries[group + 1] / kT);
        if(transport::IsFinite(contribution) && contribution > 0.0)
        {
            mass += contribution;
        }
    }
    return (transport::IsFinite(mass) && mass > 0.0) ? mass : 0.0;
}

template<typename Boundaries>
STORM_TRANSPORT_INLINE
double PlanckBandFraction(const Boundaries &boundaries,
                          const double kT,
                          const std::size_t beginGroup,
                          const std::size_t endGroup,
                          const std::size_t totalEndGroup)
{
    const double denominator =
        PlanckBandMass(boundaries, kT, beginGroup, totalEndGroup);
    if(!(denominator > 0.0))
    {
        return 0.0;
    }
    const double value =
        PlanckBandMass(boundaries, kT, beginGroup, endGroup) / denominator;
    if(value < 0.0)
    {
        return 0.0;
    }
    if(value > 1.0)
    {
        return 1.0;
    }
    return value;
}

STORM_TRANSPORT_INLINE
double SampleFrequencyInGroup(const double *boundaries,
                              const std::size_t groupCount,
                              const std::size_t group,
                              double random)
{
    if(boundaries == nullptr || group >= groupCount)
    {
        return 0.0;
    }
    if(random < 0.0)
    {
        random = 0.0;
    }
    else if(random > 1.0)
    {
        random = 1.0;
    }
    return boundaries[group] +
        random * (boundaries[group + 1] - boundaries[group]);
}

STORM_TRANSPORT_INLINE
double SampleFrequencyFromCellCdf(const double *boundaries,
                                  const double *cdf,
                                  const std::size_t groupCount,
                                  const std::size_t cellIndex,
                                  const double random)
{
    if(boundaries == nullptr || cdf == nullptr || groupCount == 0)
    {
        return 0.0;
    }
    const double *cellCdf = cdf + cellIndex * (groupCount + 1);
    const double total = cellCdf[groupCount];
    if(!(total > 0.0) || !transport::IsFinite(total))
    {
        return 0.5 * (boundaries[0] + boundaries[groupCount]);
    }
    const double target = random * total;
    std::size_t group = 0;
    while(group + 1 < groupCount && cellCdf[group + 1] < target)
    {
        ++group;
    }
    const double lower = cellCdf[group];
    const double upper = cellCdf[group + 1];
    const double width = upper - lower;
    const double fraction = width > 0.0 ? (target - lower) / width : 0.5;
    return SampleFrequencyInGroup(boundaries, groupCount, group, fraction);
}

} // namespace STORM::ddmc

#endif
