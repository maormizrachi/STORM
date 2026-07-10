#ifndef STORM_LORENTZ_TRANSFORMATION_HPP
#define STORM_LORENTZ_TRANSFORMATION_HPP

#include <algorithm>
#include <cmath>
#include <units/units.hpp>
#include "particle/Particle.hpp"
#include "elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

template<typename T, typename Grid>
double DopplerShift(const Particle<T, Grid> &particle, const T &velocity)
{
    double const v2 = ScalarProd(velocity, velocity);
    double const invClight2 = 1.0 / (units::clight * units::clight);
    double const gamma = 1.0 / std::sqrt(1.0 - invClight2 * v2);
    return gamma * (1.0 - ScalarProd(velocity, particle.velocity) * invClight2);
}

template<typename T, typename Grid>
void LorentzTransformation(
    Particle<T, Grid> &particle,
    const T &velocity,
    const std::vector<double> *energyBoundaries = nullptr)
{
    double const v2 = ScalarProd(velocity, velocity);
    if(v2 < 1e-30)
        return;

    double const invClight2 = 1.0 / (units::clight * units::clight);
    double const gamma = 1.0 / std::sqrt(1.0 - invClight2 * v2);
    double const dopplerShift = DopplerShift(particle, velocity);
    particle.frequency *= dopplerShift;
    if(energyBoundaries != nullptr && energyBoundaries->size() >= 2)
    {
        particle.frequency = std::clamp(
            particle.frequency,
            (*energyBoundaries)[0],
            energyBoundaries->back());
    }
    particle.weight *= dopplerShift;
    particle.velocity = particle.velocity + velocity * ((gamma - 1.0) * ScalarProd(particle.velocity, velocity) / v2 - gamma);
    particle.velocity *= units::clight / abs(particle.velocity);
}

} // namespace STORM

#endif // STORM_LORENTZ_TRANSFORMATION_HPP
