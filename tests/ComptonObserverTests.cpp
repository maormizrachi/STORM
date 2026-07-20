#include "examples/Vector3D.hpp"
#include "radiation/Compton.hpp"
#include "radiation/SphericalObserver.hpp"

#include <cassert>
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
    assert(STORM::isValidComptonResult(result));
    result.rates[0][1] = -1.0;
    assert(!STORM::isValidComptonResult(result));

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
    assert(generated.rates[0][1] == 2.0);
    assert(generated.residualSource[1] == -0.25);
}

void testSignedSphericalTally()
{
    STORM::SphericalObserver<Point, 2> observer(
        Point(0.0, 0.0, 0.0), 1.0, 64, {1.0, 2.0, 3.0});
    auto crossing = observer.nextOutwardCrossing(
        Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), 2.0);
    assert(crossing.hit);
    assert(std::abs(crossing.time - 1.0) < 1.0e-12);

    STORM::ObserverCrossingRecord<Point> record;
    record.crossingPoint = crossing.point;
    record.direction = Point(1.0, 0.0, 0.0);
    record.weight = -2.5;
    record.frequency = 1.5;
    record.sourceCellID = 17;
    observer.recordCrossing(record);
    observer.addBoxEscapeEnergy(3.0);
    assert(observer.totalCrossingEnergy() == -2.5);
    assert(observer.sourceCellEscape().at(17) == -2.5);
    assert(observer.boxEscapeEnergy() == 3.0);
    double groupTotal = 0.0;
    for(const auto &row : observer.groupEnergy()) groupTotal += row[0];
    assert(groupTotal == -2.5);
    auto snapshot = observer.snapshot();
    assert(snapshot.totalCrossingEnergy == -2.5);
}

} // namespace

int main()
{
    testFrozenKernelValidation();
    testSignedSphericalTally();
    return 0;
}
