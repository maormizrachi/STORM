#ifndef STORM_MONTE_CARLO_MANAGER_FACTORY_HPP
#define STORM_MONTE_CARLO_MANAGER_FACTORY_HPP

#ifdef STORM_WITH_MPI

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <mpi.h>
#include "../particle/Particle.hpp"
#include "../physics/MonteCarloPhysics.hpp"
#include "../population/PopulationControl.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "MonteCarloConfig.hpp"
#include "parallel/MonteCarloManagerLegacy.hpp"
#include "parallel/RDMAMonteCarloManager.hpp"
#include "parallel/TwoSidedMonteCarloManager.hpp"

namespace STORM {

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

template<typename T, typename Grid>
class MonteCarloManager
{
    using MCParticle = Particle<T, Grid>;

    struct Concept
    {
        virtual ~Concept() = default;
        virtual std::vector<MCParticle> step(std::vector<MCParticle> &&particles, dt_t fullDt) = 0;
        virtual std::vector<size_t> &GetCellsStepsCounters() = 0;
        virtual const std::vector<size_t> &GetCellsStepsCounters() const = 0;
        virtual std::vector<size_t> &GetBeginningParticleCount() = 0;
        virtual const std::vector<size_t> &GetBeginningParticleCount() const = 0;
    };

    template<typename Impl>
    struct Model : Concept
    {
        Impl impl;

        template<typename... Args>
        explicit Model(Args &&...args) : impl(std::forward<Args>(args)...) {}

        std::vector<MCParticle> step(std::vector<MCParticle> &&particles, dt_t fullDt) override
        {
            return impl.step(std::move(particles), fullDt);
        }
        std::vector<size_t> &GetCellsStepsCounters() override { return impl.GetCellsStepsCounters(); }
        const std::vector<size_t> &GetCellsStepsCounters() const override { return impl.GetCellsStepsCounters(); }
        std::vector<size_t> &GetBeginningParticleCount() override { return impl.GetBeginningParticleCount(); }
        const std::vector<size_t> &GetBeginningParticleCount() const override { return impl.GetBeginningParticleCount(); }
    };

    std::unique_ptr<Concept> impl_;

public:
    MonteCarloManager() = default;
    MonteCarloManager(MonteCarloManager &&) = default;
    MonteCarloManager &operator=(MonteCarloManager &&) = default;

    template<typename Impl, typename... Args>
    static MonteCarloManager Create(Args &&...args)
    {
        MonteCarloManager mgr;
        mgr.impl_ = std::make_unique<Model<Impl>>(std::forward<Args>(args)...);
        return mgr;
    }

    std::vector<MCParticle> step(std::vector<MCParticle> &&particles, dt_t fullDt)
    {
        return impl_->step(std::move(particles), fullDt);
    }

    std::vector<size_t> &GetCellsStepsCounters() { return impl_->GetCellsStepsCounters(); }
    const std::vector<size_t> &GetCellsStepsCounters() const { return impl_->GetCellsStepsCounters(); }
    std::vector<size_t> &GetBeginningParticleCount() { return impl_->GetBeginningParticleCount(); }
    const std::vector<size_t> &GetBeginningParticleCount() const { return impl_->GetBeginningParticleCount(); }
};

inline RDMA_Type ToRDMAType(RDMAEngine engine)
{
    switch(engine)
    {
        case RDMAEngine::IBV:  return RDMA_Type::IBV_RDMA;
        case RDMAEngine::OFI:  return RDMA_Type::OFI_RDMA;
        case RDMAEngine::MPI:  return RDMA_Type::MPI_RMA;
        case RDMAEngine::Auto: return RDMA_Type::AUTO_RDMA;
    }
    return RDMA_Type::AUTO_RDMA;
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid> CreateMonteCarloManager(
    const Grid &grid,
    const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
    ManagerType managerType = ManagerType::Auto,
    RDMAEngine rdmaEngine = RDMAEngine::OFI,
    const MonteCarloConfig &config = MonteCarloConfig(),
    const MPI_Comm &comm = MPI_COMM_WORLD)
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

    switch(managerType)
    {
        case ManagerType::Auto:
        {
            // Fallback chain: (1) native RDMA through OFI/libfabric -> (2) P2P.
            // The old IBV backend is kept for explicit requests only.
            try
            {
                log("Trying RDMA with OFI (libfabric)...");
                if(not RMAFactory::IsBackendAvailable(RDMA_Type::OFI_RDMA, comm))
                {
                    throw std::runtime_error("no hardware OFI/libfabric RDMA provider is available on all ranks");
                }
                auto mgr = MonteCarloManager<T, Grid>::template Create<RDMAMonteCarloManager<T, Grid>>(
                    grid, physics, populationControl, boundaryCondition, config, comm, RDMA_Type::OFI_RDMA);
                log("Using RDMA with OFI (libfabric)");
                return mgr;
            }
            catch(const std::exception &e)
            {
                log(std::string("RDMA+OFI unavailable: ") + e.what());
            }

            log("Falling back to P2P (two-sided MPI)...");
            auto mgr = MonteCarloManager<T, Grid>::template Create<TwoSidedMonteCarloManager<T, Grid>>(
                grid, physics, populationControl, boundaryCondition, comm);
            log("Using P2P (two-sided MPI)");
            return mgr;
        }

        case ManagerType::RDMA:
            return MonteCarloManager<T, Grid>::template Create<RDMAMonteCarloManager<T, Grid>>(
                grid, physics, populationControl, boundaryCondition, config, comm, ToRDMAType(rdmaEngine));

        case ManagerType::Legacy:
            return MonteCarloManager<T, Grid>::template Create<MonteCarloManagerLegacy<T, Grid>>(
                grid, physics, populationControl, boundaryCondition, config, comm, ToRDMAType(rdmaEngine));

        case ManagerType::P2P:
            return MonteCarloManager<T, Grid>::template Create<TwoSidedMonteCarloManager<T, Grid>>(
                grid, physics, populationControl, boundaryCondition, comm);
    }
    throw std::runtime_error("Unknown ManagerType");
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // STORM_MONTE_CARLO_MANAGER_FACTORY_HPP
