#ifndef STORM_TWO_SIDED_MONTE_CARLO_TRANSPORT_HPP
#define STORM_TWO_SIDED_MONTE_CARLO_TRANSPORT_HPP

template<typename T, typename Grid>
bool TwoSidedMonteCarloManager<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<size_t> removeParticlesVec;
    static std::vector<MCParticle> particlesToAdd;
    removeParticlesVec.clear();

    auto eliminateParticle = [&](size_t particleIndex)
    {
        removeParticlesVec.push_back(particleIndex);
    };

    auto transferParticle = [&](size_t particleIndex, rank_t toRank)
    {
        assert(toRank != this->rank_world); // can't send to self
        this->buffersManager->Add(toRank, this->particles[particleIndex]);
        eliminateParticle(particleIndex);
    };

    auto removeParticle = [&](size_t particleIndex)
    {
        eliminateParticle(particleIndex);
        this->localDecrementAmount += 1;
    };

    this->iteration++;

    size_t length = this->particles.size();
    for(size_t i = 0; i < length; i++)
    {
        assert(i < this->particles.size());
        MCParticle &particle = this->particles[i];
        bool debug = false;

        #ifdef STORM_DEBUG
        if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
        {
            STORMError eo("Particle was already handled in this iteration");
            auto it = std::find(this->particles.cbegin() + i, this->particles.cend(), particle);
            if(it != this->particles.cend())
            {
                eo.addEntry("Second Location", std::distance(this->particles.cbegin(), it));
            }
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Particle", particle);
            eo.addEntry("Iteration", this->iteration);
            throw eo;
        }
        particle.lastSeen = this->iteration;
        particle.lastSeenRank = this->rank_world;
        particle.lastSeenIndex = i;
        #endif // STORM_DEBUG

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

            #ifdef STORM_DEBUG
            if(particle.cellIndex >= this->Ncells)
            {
                STORMError eo("Particle has invalid cell index (ghost)");
                eo.addEntry("Particle", particle);
                eo.addEntry("Cell Index", particle.cellIndex);
                eo.addEntry("Rank", this->rank_world);
                throw eo;
            }
            if(particle.removedFromRank)
            {
                continue;
                STORMError eo("Particle was removed from rank, but still in the list");
                eo.addEntry("Particle", particle);
                eo.addEntry("Rank", this->rank_world);
                throw eo;
            }
            if(not particle.checkedHere)
            {
                if(particle.nextRank != this->rank_world)
                {
                    STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Origin", particle.sentByRank);
                    eo.addEntry("Particle Previous Location", particle.previousLocation);
                    eo.addEntry("Cell Index In Origin (Before Movement)", particle.cellIndexInPrevRank);
                    eo.addEntry("Expected", particle.nextRank);
                    eo.addEntry("Got (me)", this->rank_world);
                    eo.addEntry("The Particle Index In Last Rank", particle.particleIndexInLastRank);
                    eo.addEntry("Particle Index In This Rank", i);
                    eo.addEntry("New Cell Index Should Be", particle.cellIndex);
                    eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                    throw eo;
                }
                particle.checkedHere = true;
                particle.nextRank = std::numeric_limits<rank_t>::max();
                particle.removedFromRank = false;
                particle.sentByRank = std::numeric_limits<rank_t>::max();
            }
            if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
            {
                const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                size_t containingIdx = this->grid.GetContainingCell(particle.location);
                const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                if(containingIdx != particle.cellIndex)
                {
                    if(not this->grid.IsPointInCell(particle.location, containingIdx))
                    {
                        STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                        eo.addEntry("My Rank", this->rank_world);
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                        eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                        eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                        eo.addEntry("Declared Cell Index", particle.cellIndex);
                        eo.addEntry("Declared Cell", declaredCell);
                        eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                        eo.addEntry("Real Containing Cell Index", containingIdx);
                        eo.addEntry("Real Containing Cell", containingCell);
                        eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                        eo.addEntry("Particle Previous Location", particle.previousLocation);
                        eo.addEntry("Particle Previous Cell Index", particle.cellIndexInPrevRank);
                        throw eo;
                    }
                }
                if(std::abs(abs(declaredCell - particle.location) - abs(containingCell - particle.location)) >= 1e-12)
                {
                    STORMError eo("Particle is in Wrong Location After Transfer");
                    eo.addEntry("My Rank", this->rank_world);
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                    eo.addEntry("Particle Previous Location", particle.previousLocation);
                    eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                    eo.addEntry("New Cell Value Should Be", particle.newCellValue);
                    eo.addEntry("Declared Cell Index", particle.cellIndex);
                    eo.addEntry("Declared Cell", declaredCell);
                    eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                    eo.addEntry("Real Containing Cell Index", containingIdx);
                    eo.addEntry("Real Containing Cell", containingCell);
                    eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                    for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                    {
                        eo.addEntry("Face Index", faceIdx);
                        eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                        eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                        eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                    }
                    throw eo;
                }
            }
            #endif // STORM_DEBUG
            if(particle.sent)
            {
                particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                    MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                particle.sent = false;
            }
            T prevLoc = particle.location;
            #ifdef STORM_DEBUG
                particle.previousLocation = particle.location;
            #endif // STORM_DEBUG
            MonteCarloFunctionality functionality = this->physics->step(particle, particlesToAdd);

            if(particle.on_track)
            {
                MCParticle trackedParticle = particle;
                trackedParticle.steps = traceStep * 2 + 1;
                this->tracker.ReportParticle(trackedParticle);
            }

            if(debug)
            {
                std::cout << "Particle " << particle << ", functionality is " << functionality.change << std::endl;
            }

            if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
            {
                size_t nextCellIndex = functionality.nextCellIndex;
                assert(nextCellIndex != particle.cellIndex);
                assert(particle.timeLeft >= 0);

                if(__builtin_expect(nextCellIndex < this->Ncells, 1))
                {
                    size_t previousCell = particle.cellIndex;
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                    particle.cellIndex = nextCellIndex;
                    #ifdef STORM_DEBUG
                    if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
                    {
                        const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                        size_t containingIdx = this->grid.GetContainingCell(particle.location);
                        const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                        STORMError eo("Particle is in Wrong Location");
                        eo.addEntry("rank", this->rank_world);
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Previous Cell Index", previousCell);
                        eo.addEntry("Previous Cell", this->grid.GetMeshPoint(previousCell));
                        eo.addEntry("Previous Location", prevLoc);
                        eo.addEntry("Last location is in previous cell?", this->grid.IsPointInCell(prevLoc, previousCell));
                        eo.addEntry("Declared Cell Index", particle.cellIndex);
                        eo.addEntry("Declared Cell", declaredCell);
                        eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                        eo.addEntry("Real Containing Cell Index", containingIdx);
                        eo.addEntry("Real Containing Cell", containingCell);
                        eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                        for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                        {
                            eo.addEntry("Face Index", faceIdx);
                            eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                            eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                            eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                        }
                        throw eo;
                    }
                    #endif // STORM_DEBUG
                }
                else
                {
                    auto it = ranks_ghost_map.find(nextCellIndex);
                    if(it == ranks_ghost_map.end())
                    {
                        MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                        this->physics->onBoundaryResult(
                            particle, status,
                            functionality.boundaryCrossing &&
                            this->boundaryCondition->isEscape(status));
                        if(debug)
                        {
                            std::cout << "Particle " << particle << ", leaving domain. status from bounday condition: " << status << std::endl;
                        }
                        if(status == MonteCarloParticleStatus::REFLECT)
                        {
                            particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                        }
                        else if(status == MonteCarloParticleStatus::REMOVE)
                        {
                            stepData.leavingCount++;
                            this->allStepsCounter += particle.steps;
                            removeParticle(i);
                        }
                        else
                        {
                            STORMError eo("Unknown boundary condition for particle");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Status", status);
                            throw eo;
                        }
                        break;
                    }

                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                    auto [otherRank, neighborIndexInRank] = it->second;
                    #ifdef STORM_DEBUG
                    particle.checkedHere = false;
                    if(particle.nextRank != std::numeric_limits<rank_t>::max())
                    {
                        STORMError eo("Particle was already sent, and not sent again");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Already Transferred To Rank", particle.nextRank);
                        eo.addEntry("Being Transferred To Rank", otherRank);
                        eo.addEntry("Being Transferred To Index In Rank", neighborIndexInRank);
                        throw eo;
                    }
                    const std::vector<rank_t> &neighbors = this->grid.GetDuplicatedProcs();
                    if(std::find(neighbors.cbegin(), neighbors.cend(), otherRank) == neighbors.cend())
                    {
                        STORMError eo("Particle is going to be transffered to a non-neighboring rank");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("My Rank", this->rank_world);
                        eo.addEntry("Next Rank", otherRank);
                        eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                        throw eo;
                    }
                    particle.cellIndexInPrevRank = particle.cellIndex;
                    particle.sentByRank = this->rank_world;
                    particle.ghostIndex = nextCellIndex;
                    particle.newCellValue = this->grid.GetMeshPoint(nextCellIndex);
                    particle.particleIndexInLastRank = i;
                    particle.nextRank = otherRank;

                    if(particle.nextRank == this->rank_world)
                    {
                        STORMError eo("Particle is going to be sent to the same rank");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("My Rank", this->rank_world);
                        eo.addEntry("Next Rank", otherRank);
                        eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                        throw eo;
                    }
                    #endif // STORM_DEBUG
                    particle.sent = true;
                    particle.cellIndex = neighborIndexInRank;

                    transferParticle(i, otherRank);
                    break;
                }
            }
            else if(functionality.change == MonteCarloParticleStatus::REMOVE)
            {
                this->allStepsCounter += particle.steps;
                removeParticle(i);
                break;
            }
            else if(functionality.change == MonteCarloParticleStatus::DONE)
            {
                stepData.remaining.push_back(particle);
                this->allStepsCounter += particle.steps;
                removeParticle(i);
                break;
            }
        }
    }

    if(not removeParticlesVec.empty())
    {
        this->RemoveParticles(removeParticlesVec);
    }
    if(not particlesToAdd.empty())
    {
        this->localDecrementAmount -= particlesToAdd.size();
        this->PutSelfParticles(particlesToAdd.data(), particlesToAdd.size());
        particlesToAdd.clear();
    }
    return (length == 0);
}

#endif // STORM_TWO_SIDED_MONTE_CARLO_TRANSPORT_HPP
