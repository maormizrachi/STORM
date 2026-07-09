#ifndef STORM_MONTE_CARLO_MANAGER_SERIAL_HPP
#define STORM_MONTE_CARLO_MANAGER_SERIAL_HPP

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <cstring>
#include <numeric>
#include <boost/container/flat_map.hpp>
#include "../particle/Particle.hpp"
#include "../physics/MonteCarloPhysics.hpp"
#include "../population/PopulationControl.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "MonteCarloConfig.hpp"
#include "../StormError.hpp"

#define SERIAL_REALLOCATION_FACTOR 2

namespace STORM {

template<typename T, typename Grid>
class MonteCarloManagerSerial
{
    using index_t = uint32_t;
    using MCParticle = Particle<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        size_t leavingCount = 0;
    };

    MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition);

    virtual ~MonteCarloManagerSerial();

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, dt_t fullDt);

    class Tracker
    {
    public:
        Tracker(void) {}
        void Reset(void) { this->track.clear(); }
        std::vector<MCParticle> GetTrackParticleRoute(size_t id) const
        {
            auto it = this->track.find(id);
            if(it == this->track.end())
            {
                return std::vector<MCParticle>();
            }
            return it->second;
        }
        void ReportParticle(MCParticle &particle)
        {
            if(this->track.find(particle.id) == this->track.end())
            {
                this->track[particle.id] = std::vector<MCParticle>();
            }
            this->track[particle.id].push_back(particle);
        }
    private:
        boost::container::flat_map<size_t, std::vector<MCParticle>> track;
    };

    inline const Tracker &getTracker(void) { return this->tracker; }
    inline void resetTracker(void) { this->tracker.Reset(); }
    inline const std::vector<size_t> &GetCellsStepsCounters(void) const { return this->cellsStepsCounters; }
    inline std::vector<size_t> &GetCellsStepsCounters(void) { return this->cellsStepsCounters; }
    inline size_t GetStartParticleCount(void) const { return this->startParticleCount_; }
    inline size_t GetInitialParticleCount(void) const { return this->initialParticleCount_; }
    inline size_t GetPreStepParticleCount(void) const { return this->preStepParticleCount_; }
    inline size_t GetEndParticleCount(void) const { return this->endParticleCount_; }
    inline double GetPureComputeTime(void) const { return 0; }
    inline const std::vector<size_t> &GetBeginningParticleCount(void) const { return this->beginningParticleCount_; }
    inline std::vector<size_t> &GetBeginningParticleCount(void) { return this->beginningParticleCount_; }
    inline size_t GetHandlerMemoryBytes(void) const { return 0; }

private:
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

    struct
    {
        size_t buffsize;
        MCParticle *particles;
        index_t *av;
        size_t av_length;
        index_t *th;
        size_t th_length;
    } particlesData;

    void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);
    void HandleAll(MonteCarloStepFinalData &cache);
    void PutSelfParticles(std::vector<MCParticle> &&particles);
    void PrepareForStep(void);
    void AddParticles(const std::vector<MCParticle> &particles);
};

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), myIDCounter(0)
{
    this->particlesData.buffsize = 0;
    this->particlesData.particles = nullptr;
    this->particlesData.av = nullptr;
    this->particlesData.av_length = 0;
    this->particlesData.th = nullptr;
    this->particlesData.th_length = 0;
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::PrepareForStep(void)
{
    this->Ncells = this->grid.GetPointNo();
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    if(this->particlesData.av_length < particles.size())
    {
        size_t newBuffSize = std::max(this->particlesData.buffsize * SERIAL_REALLOCATION_FACTOR, this->particlesData.buffsize + particles.size());
        size_t oldBuffSize = this->particlesData.buffsize;
        this->particlesData.buffsize = newBuffSize;
        MCParticle *new_particles = new MCParticle[this->particlesData.buffsize];
        index_t *new_av = new index_t[this->particlesData.buffsize];
        index_t *new_th = new index_t[this->particlesData.buffsize];
        index_t difference = newBuffSize - oldBuffSize;
        if(oldBuffSize > 0)
        {
            std::memcpy(new_particles, this->particlesData.particles, oldBuffSize * sizeof(MCParticle));
            std::memcpy(new_av + difference, this->particlesData.av, oldBuffSize * sizeof(index_t));
            std::memcpy(new_th, this->particlesData.th, oldBuffSize * sizeof(index_t));
        }
        delete[] this->particlesData.particles;
        delete[] this->particlesData.av;
        delete[] this->particlesData.th;
        this->particlesData.particles = new_particles;
        this->particlesData.av = new_av;
        this->particlesData.th = new_th;
        assert(oldBuffSize < newBuffSize);
        std::iota(new_av, new_av + difference, oldBuffSize);
        this->particlesData.av_length += static_cast<int>(difference);
    }

    index_t particlesNum = particles.size();
    this->particlesData.av_length -= particlesNum;
    index_t *avIndices = this->particlesData.av + this->particlesData.av_length;
    index_t *thIndices = this->particlesData.th + this->particlesData.th_length;
    this->particlesData.th_length += particlesNum;
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    for(size_t i = 0; i < particlesNum; i++)
    {
        index_t idx = avIndices[i];
        MCParticle *particle = this->particlesData.particles + idx;
        std::memcpy(particle, &particles[i], sizeof(MCParticle));
        thIndices[i] = idx;
        particle->id = firstID + i;
    }
}

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::~MonteCarloManagerSerial()
{
    delete[] this->particlesData.av;
    this->particlesData.av = nullptr;
    delete[] this->particlesData.th;
    this->particlesData.th = nullptr;
    delete[] this->particlesData.particles;
    this->particlesData.particles = nullptr;
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
    size_t particlesNum = particles.size();
    bool reallocated = false;
    MCParticle *old_particles = this->particlesData.particles;
    index_t *old_av = this->particlesData.av;
    index_t *old_th = this->particlesData.th;
    if(this->particlesData.buffsize < particlesNum)
    {
        this->particlesData.buffsize = particlesNum;
        this->particlesData.particles = new MCParticle[this->particlesData.buffsize];
        this->particlesData.th = new index_t[this->particlesData.buffsize];
        this->particlesData.av = new index_t[this->particlesData.buffsize];
        reallocated = true;
    }
    this->particlesData.av_length = 0;
    this->particlesData.th_length = 0;
    std::memcpy(this->particlesData.particles, particles.data(), particles.size() * sizeof(MCParticle));
    if(reallocated)
    {
        delete[] old_particles;
        delete[] old_av;
        delete[] old_th;
    }
    this->particlesData.th_length = particlesNum;
    for(size_t i = 0; i < particlesNum; i++)
    {
        assert(i < this->particlesData.buffsize);
        this->particlesData.th[i] = i;
    }
    size_t availLength = this->particlesData.buffsize - particlesNum;
    this->particlesData.av_length = availLength;
    for(size_t i = 0; i < availLength; i++)
    {
        size_t idx = i + particlesNum;
        assert(idx < this->particlesData.buffsize);
        this->particlesData.av[i] = idx;
    }
    size_t firstID = this->myIDCounter;
    size_t assignedCounter = 0;
    for(size_t i = 0; i < particlesNum; i++)
    {
        if(this->particlesData.particles[i].id == std::numeric_limits<size_t>::max())
        {
            this->particlesData.particles[i].id = firstID + assignedCounter;
            assignedCounter++;
        }
    }
    this->myIDCounter += assignedCounter;
    std::vector<MCParticle> empty;
    particles.swap(empty);
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
{
    for(int i = static_cast<int>(num) - 1; i >= 0; i--)
    {
        const size_t &toHandleIndex = indicesInToHandle[i];
        assert(i == 0 or indicesInToHandle[i] > indicesInToHandle[i - 1]);
        assert(toHandleIndex < this->particlesData.th_length);
        index_t particleIdx = this->particlesData.th[toHandleIndex];
        assert(this->particlesData.av_length < this->particlesData.buffsize);
        this->particlesData.av[this->particlesData.av_length++] = particleIdx;
        this->particlesData.th[toHandleIndex] = this->particlesData.th[--this->particlesData.th_length];
        assert(this->particlesData.th_length >= 0);
    }
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<size_t> removeParticlesVec;
    static std::vector<MCParticle> particlesToAdd;
    size_t removeCounter = 0;

    auto removeParticle = [&](size_t i)
    {
        if(removeCounter >= removeParticlesVec.size())
        {
            removeParticlesVec.push_back(i);
            removeCounter++;
        }
        else
        {
            removeParticlesVec[removeCounter++] = i;
        }
    };

    int length = this->particlesData.th_length;
    for(int i = 0; i < length; i++)
    {
        assert(i < this->particlesData.buffsize);
        size_t particleIndex = this->particlesData.th[i];
        assert(particleIndex < this->particlesData.buffsize);
        MCParticle &particle = this->particlesData.particles[particleIndex];
        while(true)
        {
            const size_t traceStep = particle.steps;
            if(particle.on_track)
            {
                MCParticle trackedParticle = particle;
                trackedParticle.steps = traceStep * 2;
                this->tracker.ReportParticle(trackedParticle);
            }
            particle.steps++;
            this->cellsStepsCounters[particle.cellIndex]++;

            StepResult<T, Grid> functionality = this->physics->step(particle, particlesToAdd);

            if(particle.on_track)
            {
                MCParticle trackedParticle = particle;
                trackedParticle.steps = traceStep * 2 + 1;
                this->tracker.ReportParticle(trackedParticle);
            }

            #ifdef STORM_WITH_TRACING_HISTORY
                particle.recordHistory(particle.cellIndex, 0, static_cast<int>(functionality.change));
            #endif

            if(functionality.change == ParticleStatus::CELL_MOVE)
            {
                size_t nextCellIndex = functionality.nextCellIndex;
                assert(nextCellIndex != particle.cellIndex);
                assert(particle.timeLeft >= 0);
                if(__builtin_expect(nextCellIndex < this->Ncells, 1))
                {
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                    particle.cellIndex = nextCellIndex;
                }
                else
                {
                    #ifdef STORM_WITH_TRACING_HISTORY
                        T preReflectLoc = particle.location;
                        T preReflectVel = particle.velocity;
                    #endif
                    ParticleStatus status = this->boundaryCondition->apply(particle);
                    if(status == ParticleStatus::REFLECT)
                    {
                        #ifdef STORM_WITH_TRACING_HISTORY
                            particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
                        #endif
                    }
                    else if(status == ParticleStatus::REMOVE)
                    {
                        stepData.leavingCount++;
                        removeParticle(i);
                        break;
                    }
                    else
                    {
                        STORMError eo("Unknown boundary condition for particle");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Status", status);
                        throw eo;
                    }
                    continue;
                }
            }
            else if(functionality.change == ParticleStatus::REMOVE)
            {
                removeParticle(i);
                break;
            }
            else if(functionality.change == ParticleStatus::DONE)
            {
                stepData.remaining.push_back(particle);
                removeParticle(i);
                break;
            }
        }
    }

    if(not particlesToAdd.empty())
    {
        this->AddParticles(particlesToAdd);
        particlesToAdd.clear();
    }

    if(removeCounter > 0)
    {
        this->RemoveParticles(removeParticlesVec, removeCounter);
    }
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerSerial<T, Grid>::MCParticle> MonteCarloManagerSerial<T, Grid>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    this->PrepareForStep();
    this->physics->updateGridData();
    this->resetTracker();
    this->PutSelfParticles(std::move(particleList));

    int length = this->particlesData.th_length;
    for(int i = 0; i < length; i++)
    {
        size_t particleIndex = this->particlesData.th[i];
        MCParticle &p = this->particlesData.particles[particleIndex];
        #ifdef STORM_WITH_TRACING_HISTORY
        p.tracingHistoryIndex = 0;
        p.tracingHistoryCount = 0;
        #endif
        p.timeLeft = fullDt;
        p.initialWeight = std::abs(p.weight);
        p.steps = 0;
    }

    size_t initialParticlesNum = this->particlesData.th_length;
    this->initialParticleCount_ = initialParticlesNum;
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
    this->AddParticles(newParticles1);
    this->preStepParticleCount_ = newParticles1.size();
    this->startParticleCount_ = initialParticlesNum + newParticles1.size();
    std::cout << "MC particle counts before transport:"
              << " initial=" << this->initialParticleCount_
              << " prestep_generated=" << this->preStepParticleCount_
              << " active_after_prestep=" << this->startParticleCount_
              << std::endl;

    this->beginningParticleCount_.assign(this->Ncells, 0);
    for(size_t i = 0; i < this->particlesData.th_length; i++)
    {
        size_t particleIndex = this->particlesData.th[i];
        this->beginningParticleCount_[this->particlesData.particles[particleIndex].cellIndex]++;
    }

    this->cellsStepsCounters.assign(this->Ncells, 0);

    MonteCarloStepFinalData data;

    auto start = std::chrono::high_resolution_clock::now();
    double lastProgressPrint = 0.0;
    size_t const totalParticlesStart = this->startParticleCount_;

    try
    {
        while(this->particlesData.th_length != 0)
        {
            this->HandleAll(data);

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start).count();
            if(elapsed_s - lastProgressPrint >= 10.0)
            {
                lastProgressPrint = elapsed_s;
                size_t remaining = this->particlesData.th_length;
                double done_frac = 1.0 - static_cast<double>(remaining) / static_cast<double>(totalParticlesStart);
                double processed = static_cast<double>(totalParticlesStart - remaining);
                double rate = (elapsed_s > 0) ? processed / elapsed_s : 0.0;
                double eta = (rate > 0) ? remaining / rate : 0.0;
                std::cerr << "[Progress] " << std::fixed << std::setprecision(1)
                          << (done_frac * 100.0) << "% done, "
                          << elapsed_s << "s elapsed, "
                          << "~" << eta << "s remaining"
                          << std::endl;
            }
        }
    }
    catch(const STORMError &eo)
    {
        reportError(eo);
        throw;
    }

    std::vector<MCParticle> populationControlParticles = this->populationControl->activate(data.remaining);
    this->endParticleCount_ = populationControlParticles.size();
    this->physics->postStep(populationControlParticles, fullDt);

    return populationControlParticles;
}

} // namespace STORM

#endif // STORM_MONTE_CARLO_MANAGER_SERIAL_HPP
