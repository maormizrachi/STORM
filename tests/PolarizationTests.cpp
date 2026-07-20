#include "examples/Vector3D.hpp"
#include "particle/Particle.hpp"
#include "radiation/Polarization.hpp"
#include "radiation/SphericalObserver.hpp"

#include <cassert>
#include <cmath>
#include <random>

namespace {

struct Grid {};
using Point = Vector3D;
using Particle = STORM::Particle<Point, Grid>;

double uniform(std::mt19937_64 &rng)
{
    static std::uniform_real_distribution<double> dist(
        std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
    return dist(rng);
}

void assertPhysical(const Particle &p)
{
    assert(std::isfinite(p.stokesQ));
    assert(std::isfinite(p.stokesU));
    assert(p.stokesQ * p.stokesQ + p.stokesU * p.stokesU <= 1.0 + 1.0e-12);
    assert(std::abs(STORM::fallback::ScalarProd(
        p.velocity, p.polarizationBasis)) <= 1.0e-10 * std::max(1.0, STORM::fallback::abs(p.velocity)));
}

void testPolarizedThomsonScatter()
{
    Particle p;
    p.velocity = Point(0.0, 0.0, 1.0);
    p.stokesQ = 0.35;
    p.stokesU = -0.2;
    p.polarizationBasis = Point(1.0, 0.0, 0.0);
    p.polarizationInitialized = true;

    std::mt19937_64 rng(17);
    auto u01 = [&]() { return uniform(rng); };
    Point const oldVelocity = p.velocity;
    Point const newVelocity = STORM::polarization::samplePolarizedThomsonDirection(
        p, oldVelocity, u01);
    STORM::polarization::applyThomsonScatter<Point>(p, oldVelocity, newVelocity);
    assertPhysical(p);
}

void testAcceleratedHistory()
{
    Particle p;
    p.velocity = Point(1.0, 0.0, 0.0);
    STORM::polarization::resetUnpolarized<Point>(p);
    p.stokesQ = 1.0;

    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> dist(
        std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
    STORM::polarization::applyAcceleratedPolarizationHistory<Point>(
        p, 1.0e-12, 1.0e-10, 0.0, Point(0.0, 1.0, 0.0),
        4, 2.0, rng, dist);
    assert(p.radiationState.pendingMeanScatterings == 0.0);
    assertPhysical(p);
}

void testObserverFlag()
{
    STORM::SphericalObserver<Point, 1> observer(
        Point(0.0, 0.0, 0.0), 1.0, 16, {0.0, 1.0});
    observer.setPolarizationEnabled(true);
    auto crossing = observer.nextOutwardCrossing(
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), 2.0);
    STORM::ObserverCrossingRecord<Point> record;
    record.crossingPoint = crossing.point;
    record.direction = Point(1.0, 0.0, 0.0);
    record.frequency = 0.5;
    record.weight = 2.0;
    record.stokesQ = 0.5;
    record.polarizationBasis = Point(0.0, 1.0, 0.0);
    record.polarizationInitialized = true;
    observer.recordCrossing(record);
    double totalQ = 0.0;
    for(double q : observer.stokesQ()) totalQ += q;
    assert(std::abs(totalQ) > 0.0);
}

} // namespace

int main()
{
    testPolarizedThomsonScatter();
    testAcceleratedHistory();
    testObserverFlag();
    return 0;
}
