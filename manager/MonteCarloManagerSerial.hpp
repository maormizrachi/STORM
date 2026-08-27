#ifndef STORM_MONTE_CARLO_MANAGER_SERIAL_HPP
#define STORM_MONTE_CARLO_MANAGER_SERIAL_HPP

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <chrono>
#include <limits>
#include <tuple>
#include <utility>
#include "../particle/Particle.hpp"
#include "../physics/MonteCarloPhysics.hpp"
#include "../population/PopulationControl.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "MonteCarloConfig.hpp"
#include "LocalTransportExecutor.hpp"
#include "MonteCarloParticleInitialization.hpp"
#include "MonteCarloStepState.hpp"
#include "MonteCarloTracker.hpp"
#include "MonteCarloTransportCore.hpp"
#include "ParticleQueue.hpp"
#include "../StormError.hpp"

namespace STORM {

template<typename T, typename Grid>
class MonteCarloManagerSerial
{
    using MCParticle = Particle<T>;

public:
    using MonteCarloStepFinalData = MonteCarloStepState<MCParticle>;
    using Tracker = MonteCarloTracker<MCParticle>;

    MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition);

    virtual ~MonteCarloManagerSerial() = default;

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, dt_t fullDt);

    inline const Tracker &getTracker(void)
    {
        return this->tracker;
    }
    inline void resetTracker(void)
    {
        this->tracker.Reset();
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
    inline size_t GetInitialParticleCount(void) const
    {
        return this->initialParticleCount_;
    }
    inline size_t GetPreStepParticleCount(void) const
    {
        return this->preStepParticleCount_;
    }
    inline size_t GetEndParticleCount(void) const
    {
        return this->endParticleCount_;
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
        return 0;
    }

private:
    using TransportCore = MonteCarloTransportCore<HostLocalTransportExecutor<MCParticle>>;

    const Grid &grid;
    size_t Ncells;
    int progress;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    size_t myIDCounter;
    std::vector<size_t> cellsStepsCounters;
    size_t startParticleCount_ = 0;
    size_t endParticleCount_ = 0;
    size_t initialParticleCount_ = 0;
    size_t preStepParticleCount_ = 0;
    std::vector<size_t> beginningParticleCount_;
    ParticleQueue<MCParticle> particles;
    HostLocalTransportExecutor<MCParticle> localTransportExecutor;
    TransportCore transportCore;

    void HandleAll(MonteCarloStepFinalData &cache);
    void PutSelfParticles(std::vector<MCParticle> &&particles);
    void PrepareForStep(void);
    void AddParticles(const std::vector<MCParticle> &particles);
    void HandleParticle(MCParticle &particle, MonteCarloStepFinalData &stepData,
                        std::vector<MCParticle> &particlesToAdd);
};

#include "SerialMonteCarloLifecycle.hpp"
#include "SerialMonteCarloTransport.hpp"

} // namespace STORM

#endif // STORM_MONTE_CARLO_MANAGER_SERIAL_HPP
