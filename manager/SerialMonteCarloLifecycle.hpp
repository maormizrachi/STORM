#ifndef STORM_SERIAL_MONTE_CARLO_LIFECYCLE_HPP
#define STORM_SERIAL_MONTE_CARLO_LIFECYCLE_HPP

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition) :
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), myIDCounter(0), transportCore(this->localTransportExecutor)
{
    this->beginningParticleCount_.assign(this->grid.GetPointNo(), 0);
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
    if(particles.empty())
    {
        return;
    }

    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    this->particles.Append(particles.size(), [this, &particles, firstID](MCParticle &destination, size_t i)
    {
        destination = particles[i];
        destination.id = firstID + i;
    });
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
    this->particles.Clear();
    this->particles.Reserve(particles.size());

    size_t firstID = this->myIDCounter;
    size_t assignedCounter = 0;
    for(size_t i = 0; i < particles.size(); i++)
    {
        if(particles[i].id == std::numeric_limits<size_t>::max())
        {
            particles[i].id = firstID + assignedCounter;
            assignedCounter++;
        }
        this->particles.Append(std::move(particles[i]));
    }
    this->myIDCounter += assignedCounter;
    particles.clear();
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerSerial<T, Grid>::MCParticle> MonteCarloManagerSerial<T, Grid>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    this->PrepareForStep();
    this->physics->updateGridData();
    this->resetTracker();
    this->PutSelfParticles(std::move(particleList));

    MonteCarloParticleInitializer::InitializeStore(this->particles, fullDt);

    size_t initialParticlesNum = this->particles.Size();
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
    for(size_t i = 0; i < this->particles.Size(); i++)
    {
        this->beginningParticleCount_[this->particles.At(i).cellIndex]++;
    }

    this->cellsStepsCounters.assign(this->Ncells, 0);

    MonteCarloStepFinalData data;

    auto start = std::chrono::high_resolution_clock::now();
    double lastProgressPrint = 0.0;
    size_t const totalParticlesStart = this->startParticleCount_;

    try
    {
        while(not this->particles.Empty())
        {
            this->HandleAll(data);

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start).count();
            if(elapsed_s - lastProgressPrint >= 10.0)
            {
                lastProgressPrint = elapsed_s;
                size_t remaining = this->particles.Size();
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

#endif // STORM_SERIAL_MONTE_CARLO_LIFECYCLE_HPP
