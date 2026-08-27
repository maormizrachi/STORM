#include <type_traits>

#include "DeviceParticle.hpp"
#include "GreyIMCData.hpp"
#include "KokkosLocalTransportExecutor.hpp"
#include "KokkosRuntime.hpp"

namespace storm_gpu_compile_test
{

using STORM::gpu::DeviceParticle;

struct TestPoint
{
    using coord_type = double;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    explicit TestPoint(double value = 0.0)
        : x(value), y(value), z(value)
    {}

    TestPoint(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_)
    {}
};

static_assert(std::is_trivially_copyable<DeviceParticle>::value,
              "The Kokkos packet must be trivially copyable");

KOKKOS_FUNCTION
STORM::transport::RandomWalkResult InstantiateRandomWalkKernel(
    DeviceParticle &particle,
    const STORM::gpu::GreyIMCViews<STORM::gpu::DeviceVec3> &views)
{
    return STORM::transport::TryAdvanceRandomWalk(particle, views);
}

void InstantiateHostSupport(
    STORM::gpu::GreyIMCData &data,
    STORM::gpu::KokkosLocalTransportExecutor &executor,
    const std::vector<STORM::Particle<TestPoint>> &particles)
{
    const std::vector<std::size_t> offsets = {0};
    const std::vector<TestPoint> points;
    const std::vector<double> planeOffsets;
    const std::vector<STORM::cell_index_t> cells;
    const std::vector<std::uint8_t> boundaries;
    std::vector<double> tables;
    std::vector<double> groupTallies;
    std::vector<double> radiationTallies;
    std::vector<TestPoint> momentumTallies;
    std::size_t randomWalkSteps = 0;
    const STORM::gpu::KokkosRuntime runtime;
    data.UploadGrid(
        offsets, points, points, planeOffsets, cells, boundaries,
        boundaries);
    data.UploadTables(tables, tables, tables);
    data.UploadHydro(points);
    (void) executor.Execute(
        particles,
        particles.size(),
        data.Views(1.0, true, true, true));
    data.AddTallies(
        tables,
        radiationTallies,
        groupTallies,
        momentumTallies,
        randomWalkSteps);
}

} // namespace storm_gpu_compile_test
