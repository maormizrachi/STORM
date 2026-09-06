#pragma once
#ifdef STORM_WITH_MPI
#include "../MonteCarloManager.hpp"
#include "../communication/RDMACommunicationEngine.hpp"
namespace STORM
{
template<class T, class Grid, class Physics = MonteCarloPhysics<T, Grid>>
class MonteCarloManagerLegacy : public MonteCarloManager<T, Grid, Physics>
{
public:
    MonteCarloManagerLegacy(const Grid &grid, const std::shared_ptr<Physics> &physics,
                            const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                            const MonteCarloConfig &config = MonteCarloConfig(),
                            const MPI_Comm &comm = MPI_COMM_WORLD,
                            RDMA_Type type = RDMA_Type::AUTO_RDMA)
        : MonteCarloManager<T, Grid, Physics>(
              grid, physics, populationControl, boundaryCondition, config,
              std::make_unique<RDMACommunicationEngine<T, Grid>>(grid, config, comm, type))
    {
    }
};
#endif
} // namespace STORM
