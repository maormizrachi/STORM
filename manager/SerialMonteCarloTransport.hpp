#ifndef STORM_SERIAL_MONTE_CARLO_TRANSPORT_HPP
#define STORM_SERIAL_MONTE_CARLO_TRANSPORT_HPP

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    std::vector<MCParticle> particlesToAdd;
    auto processParticle = [this, &stepData, &particlesToAdd](MCParticle &particle, size_t)
    {
        this->HandleParticle(particle, stepData, particlesToAdd);
    };
    this->transportCore.HandleAll(this->particles, processParticle,
                                  this->particles.Size());

    if(not particlesToAdd.empty())
    {
        this->AddParticles(particlesToAdd);
    }
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::HandleParticle(MCParticle &particle,
                                                       MonteCarloStepFinalData &stepData,
                                                       std::vector<MCParticle> &particlesToAdd)
{
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

        StepResult functionality = this->physics->step(particle, particlesToAdd);

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
                this->physics->onBoundaryResult(particle, status, functionality.boundaryCrossing and this->boundaryCondition->isEscape(status));
                if(status == ParticleStatus::REFLECT)
                {
#ifdef STORM_WITH_TRACING_HISTORY
                    particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
#endif
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                }
                else if(status == ParticleStatus::REMOVE)
                {
                    stepData.leavingCount++;
                    return;
                }
                else
                {
                    STORMError eo("Unknown boundary condition for particle");
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Status", status);
                    throw eo;
                }
            }
        }
        else if(functionality.change == ParticleStatus::REMOVE)
        {
            return;
        }
        else if(functionality.change == ParticleStatus::DONE)
        {
            stepData.remaining.push_back(particle);
            return;
        }
    }
}

#endif // STORM_SERIAL_MONTE_CARLO_TRANSPORT_HPP
