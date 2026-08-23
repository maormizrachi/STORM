#include <memory>
#include <type_traits>

#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "physics/NoPhysics.hpp"

namespace
{

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using Physics = STORM::NoPhysics<Vector3D, Grid>;
using StaticManager = STORM::RDMAMonteCarloManager<Vector3D, Grid, Physics>;
using ErasedManager = STORM::RDMAMonteCarloManager<Vector3D, Grid>;

static_assert(not std::is_same<StaticManager, ErasedManager>::value,
              "The concrete physics type must remain part of the RDMA manager type");
static_assert(std::is_constructible<StaticManager,
                                    const Grid &,
                                    const std::shared_ptr<Physics> &,
                                    const std::shared_ptr<STORM::PopulationControl<Vector3D, Grid>> &,
                                    const std::shared_ptr<STORM::BoundaryCondition<Vector3D, Grid>> &>::value,
              "The RDMA manager must accept concrete derived physics");

[[maybe_unused]] STORM::MonteCarloManager<Vector3D, Grid> InstantiateStaticPhysicsFactory(
    const Grid &grid,
    const std::shared_ptr<Physics> &physics,
    const std::shared_ptr<STORM::PopulationControl<Vector3D, Grid>> &populationControl,
    const std::shared_ptr<STORM::BoundaryCondition<Vector3D, Grid>> &boundaryCondition)
{
    return STORM::CreateMonteCarloManager<Vector3D, Grid>(
        grid, physics, populationControl, boundaryCondition, STORM::ManagerType::RDMA);
}

} // namespace
