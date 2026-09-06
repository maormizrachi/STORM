#ifndef STORM_MONTE_CARLO_MANAGER_FACTORY_HPP
#define STORM_MONTE_CARLO_MANAGER_FACTORY_HPP

#ifdef STORM_WITH_MPI

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <mpi.h>
#include "../particle/Particle.hpp"
#include "../physics/MonteCarloPhysics.hpp"
#include "../population/PopulationControl.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "MonteCarloConfig.hpp"
#include "MonteCarloManager.hpp"
#include "communication/RDMACommunicationEngine.hpp"
#include "communication/P2PCommunicationEngine.hpp"

namespace STORM
{

enum class ManagerType
{
    Auto,
    RDMA,
    Legacy,
    P2P
};

enum class RDMAEngine
{
    IBV,
    OFI,
    MPI,
    Auto
};

inline RDMA_Type ToRDMAType(RDMAEngine engine)
{
    switch(engine)
    {
    case RDMAEngine::IBV:
        return RDMA_Type::IBV_RDMA;
    case RDMAEngine::OFI:
        return RDMA_Type::OFI_RDMA;
    case RDMAEngine::MPI:
        return RDMA_Type::MPI_RMA;
    case RDMAEngine::Auto:
        return RDMA_Type::AUTO_RDMA;
    }
    return RDMA_Type::AUTO_RDMA;
}

template<class T, class Grid>
std::unique_ptr<CommunicationEngine<T>> CreateCommunicationEngine(
    const Grid &grid, ManagerType managerType, RDMAEngine rdmaEngine,
    const MonteCarloConfig &config, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    auto log = [&](const std::string &msg)
    {
        if(rank == 0)
        {
            std::cout << "[MonteCarloManager] " << msg << std::endl;
        }
    };

    std::unique_ptr<CommunicationEngine<T>> engine;
    switch(managerType)
    {
    case ManagerType::Auto:
        try
        {
            engine = std::make_unique<RDMACommunicationEngine<T, Grid>>(grid, config, comm, RDMA_Type::OFI_RDMA);
            log("Using RDMA with OFI (libfabric)");
            break;
        }
        catch(const std::exception &e)
        {
            log(std::string("RDMA+OFI unavailable: ") + e.what());
            engine = std::make_unique<P2PCommunicationEngine<T, Grid>>(grid, config, comm);
            log("Using P2P (two-sided MPI)");
        }
        break;
    case ManagerType::P2P:
        engine = std::make_unique<P2PCommunicationEngine<T, Grid>>(grid, config, comm);
        log("Using P2P (two-sided MPI)");
        break;
    case ManagerType::Legacy:
        log("Legacy manager is deprecated; using the unified manager with RDMA");
        engine = std::make_unique<RDMACommunicationEngine<T, Grid>>(grid, config, comm, ToRDMAType(rdmaEngine));
        break;
    case ManagerType::RDMA:
        engine = std::make_unique<RDMACommunicationEngine<T, Grid>>(grid, config, comm, ToRDMAType(rdmaEngine));
        break;
    default:
        throw std::runtime_error("Unknown ManagerType");
    }
    return engine;
}

template<typename T, typename Grid, typename Physics>
MonteCarloManager<T, Grid, Physics> CreateMonteCarloManager(
    const Grid &grid,
    const std::shared_ptr<Physics> &physics,
    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
    ManagerType managerType = ManagerType::Auto,
    RDMAEngine rdmaEngine = RDMAEngine::OFI,
    const MonteCarloConfig &config = MonteCarloConfig(),
    const MPI_Comm &comm = MPI_COMM_WORLD)
{
    static_assert(std::is_base_of<MonteCarloPhysics<T, Grid>, Physics>::value,
                  "Physics must derive from MonteCarloPhysics<T, Grid>");

    std::unique_ptr<CommunicationEngine<T>> engine = CreateCommunicationEngine<T>(grid, managerType, rdmaEngine, config, comm);
    return MonteCarloManager<T, Grid, Physics>(
        grid, physics, populationControl, boundaryCondition, config, std::move(engine));
}

} // namespace STORM
#endif // STORM_WITH_MPI
#endif // STORM_MONTE_CARLO_MANAGER_FACTORY_HPP
