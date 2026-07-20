#include "examples/Vector3D.hpp"
#include "radiation/Compton.hpp"
#include "radiation/SphericalObserver.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <memory>

namespace {

using Point = Vector3D;

struct Cell {};
struct Grid {};

void testFrozenKernelValidation()
{
    using Result = STORM::ComptonKernelResult<2>;
    Result result;
    result.rates[0][1] = 1.0;
    result.rates[1][0] = 0.5;
    STORM_TEST_CHECK(STORM::isValidComptonResult(result));
    result.rates[0][1] = -1.0;
    STORM_TEST_CHECK(!STORM::isValidComptonResult(result));

    using Kernel = STORM::FunctionalComptonKernel<Point, Grid, Cell, 2>;
    auto kernel = Kernel([](const Cell&, double, double,
                            const std::array<double, 3>&,
                            const std::array<double, 2>&,
                            double, std::size_t, std::mt19937_64&) {
        Result generated;
        generated.rates[0][1] = 2.0;
        generated.hasResidualSource = true;
        generated.residualSource[1] = -0.25;
        return generated;
    });
    std::mt19937_64 rng(7);
    auto generated = kernel.build(Cell{}, 1.0, 1.0, {1.0, 2.0, 3.0},
                                  {1.5, 2.5}, 1.0, 8, rng);
    STORM_TEST_CHECK(generated.rates[0][1] == 2.0);
    STORM_TEST_CHECK(generated.residualSource[1] == -0.25);
}

void testSignedSphericalTally()
{
    STORM::SphericalObserver<Point, 2> observer(
        Point(0.0, 0.0, 0.0), 1.0, 64, {1.0, 2.0, 3.0});
    auto crossing = observer.nextOutwardCrossing(
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), 2.0);
    STORM_TEST_CHECK(crossing.hit);
    STORM_TEST_CHECK(std::abs(crossing.time - 1.0) < 1.0e-12);

    STORM::ObserverCrossingRecord<Point> record;
    record.crossingPoint = crossing.point;
    record.direction = Point(1.0, 0.0, 0.0);
    record.weight = -2.5;
    record.frequency = 1.5;
    record.sourceCellID = 17;
    observer.recordCrossing(record);
    observer.addBoxEscapeEnergy(3.0);
    observer.addEmittedEnergy(4.0);
    observer.addEmittedEnergy(-1.0);
    observer.addEmittedEnergyComponents(4.0, 1.0);
    STORM_TEST_CHECK(observer.totalCrossingEnergy() == -2.5);
    STORM_TEST_CHECK(observer.sourceCellEscape().at(17) == -2.5);
    STORM_TEST_CHECK(observer.boxEscapeEnergy() == 3.0);
    STORM_TEST_CHECK(observer.emittedEnergy() == 3.0);
    STORM_TEST_CHECK(observer.emittedPositiveEnergy() == 4.0);
    STORM_TEST_CHECK(observer.emittedNegativeEnergy() == 1.0);
    double groupTotal = 0.0;
    for(const auto &row : observer.groupEnergy()) groupTotal += row[0];
    STORM_TEST_CHECK(groupTotal == -2.5);
    auto snapshot = observer.snapshot();
    STORM_TEST_CHECK(snapshot.totalCrossingEnergy == -2.5);
    STORM_TEST_CHECK(snapshot.emittedEnergy == 3.0);
    STORM_TEST_CHECK(snapshot.emittedPositiveEnergy == 4.0);
    STORM_TEST_CHECK(snapshot.emittedNegativeEnergy == 1.0);

    // Once a packet has crossed outward, the next ray segment starts outside
    // the sphere and must not report the same root again.
    auto outsideCrossing = observer.nextOutwardCrossing(
        crossing.point + Point(1.0e-8, 0.0, 0.0),
        Point(1.0, 0.0, 0.0), 2.0);
    STORM_TEST_CHECK(!outsideCrossing.hit);
}

} // namespace

int main()
{
    testFrozenKernelValidation();
    testSignedSphericalTally();
    return 0;
}
