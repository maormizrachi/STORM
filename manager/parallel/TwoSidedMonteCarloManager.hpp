#ifndef TWO_SIDED_MONTE_CARLO_MANAGER_HPP
#define TWO_SIDED_MONTE_CARLO_MANAGER_HPP

#ifdef STORM_WITH_MPI

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <chrono>
#include <boost/container/flat_set.hpp>
#include <mpi.h>
#include <mpi_utils/AmountManager.hpp>
#include <mpi_utils/BuffersManager.hpp>
#include <mpi_utils/mpi_commands.hpp>
#include "../../utils/GhostMap.hpp"
#include "../../particle/Particle.hpp"
#include "../../physics/MonteCarloPhysics.hpp"
#include "../../population/PopulationControl.hpp"
#include "../../boundary/BoundaryCondition.hpp"
#include "../MonteCarloConfig.hpp"
#include "../MonteCarloParticleInitialization.hpp"
#include "../MonteCarloStepState.hpp"
#include "../MonteCarloTracker.hpp"
#include "../../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

#define PARTICLES_TAG 8817
#define RECV_BUFFER_MAX_SIZE 1000
#define SEND_BUFFER_DISPATCH_MIN_SIZE 500
#define SEND_BUFFER_DISPATCH_MIN_CYCLES 500

template<typename T, typename Grid>
class TwoSidedMonteCarloManager
{
    using MCParticle = MonteCarloParticle<T>;

public:
    using MonteCarloStepFinalData = MonteCarloStepState<MCParticle>;
    using Tracker = MonteCarloTracker<MCParticle>;

    TwoSidedMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~TwoSidedMonteCarloManager() = default;

    inline size_t GetStepCounter(void) const
    {
        return this->allStepsCounter;
    }

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const
    {
        return this->cellsStepsCounters;
    }

    inline std::vector<size_t> &GetCellsStepsCounters(void)
    {
        return this->cellsStepsCounters;
    }

    inline size_t GetStartParticleCount(void) const
    {
        return this->startParticleCount_;
    }

    inline size_t GetEndParticleCount(void) const
    {
        return this->endParticleCount_;
    }

    inline size_t GetInitialParticleCount(void) const
    {
        return this->initialParticleCount_;
    }

    inline size_t GetPreStepParticleCount(void) const
    {
        return this->preStepParticleCount_;
    }

    inline double GetPureComputeTime(void) const
    {
        return 0;
    }

    inline const std::vector<size_t> &GetBeginningParticleCount(void) const
    {
        return this->beginningParticleCount_;
    }

    inline std::vector<size_t> &GetBeginningParticleCount(void)
    {
        return this->beginningParticleCount_;
    }

    inline size_t GetHandlerMemoryBytes(void) const
    {
        return this->handlerMemoryBytes_;
    }

    // todo: should return that?
    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, dt_t fullDt);

    inline const Tracker &getTracker(void)
    {
        return this->tracker;
    }

    inline void resetTracker(void)
    {
        this->tracker.Reset();
    }

private:
    const Grid &grid;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;

    std::vector<MCParticle> particles;
    std::shared_ptr<BuffersManager<MCParticle>> buffersManager;
    typename AmountManager::counter_t localDecrementAmount;
    Tracker tracker;

    size_t allStepsCounter;
    std::vector<size_t> cellsStepsCounters;

    size_t iteration;
    size_t myIDCounter;
    size_t currentStep;
    size_t startParticleCount_ = 0;
    size_t endParticleCount_ = 0;
    size_t initialParticleCount_ = 0;
    size_t preStepParticleCount_ = 0;
    std::vector<size_t> beginningParticleCount_;
    size_t handlerMemoryBytes_ = 0;

    bool HandleAll(MonteCarloStepFinalData &stepData);

    void PutSelfParticles(const MCParticle *particles, size_t particlesNum);

    void RemoveParticles(const std::vector<size_t> &indices);
};

#include "TwoSidedMonteCarloLifecycle.hpp"
#include "TwoSidedMonteCarloTransport.hpp"

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // TWO_SIDED_MONTE_CARLO_MANAGER_HPP
