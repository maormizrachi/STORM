#ifndef STORM_MONTE_CARLO_TRANSPORT_HPP
#define STORM_MONTE_CARLO_TRANSPORT_HPP

template<typename T, typename Grid, typename Physics>
MonteCarloManager<T, Grid, Physics>::MonteCarloManager(
    const Grid &grid, const std::shared_ptr<Physics> &physics,
    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
    const MonteCarloConfig &config, std::unique_ptr<CommunicationEngine<T>> engine)
    : grid(grid), config(config),
#ifdef STORM_WITH_MPI
      commWorld(engine->Communicator()),
#endif
      rankWorld(engine->Rank()), sizeWorld(engine->Size()), physics(physics),
      populationControl(populationControl), boundaryCondition(boundaryCondition),
#ifdef STORM_WITH_MPI
      tracker(commWorld),
#endif
      myIDCounter(0), currentStep(0),
      lastBuildGeneration(std::numeric_limits<size_t>::max()),
      engine(std::move(engine)), detachedRankParticles(sizeWorld)
{
    cellsStepsCounters.assign(grid.GetPointNo(), 0);
    cellsParticleCounters.assign(grid.GetPointNo(), 0);
    beginningParticleCount.assign(grid.GetPointNo(), 0);
}

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::AddParticles(const std::vector<MCParticle> &particles)
{
    if(particles.empty())
    {
        return;
    }

    std::vector<MCParticle> initialized(particles);
    size_t particlesNum = particles.size();
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    for(size_t i = 0; i < particlesNum; ++i)
    {
        MCParticle &destination = initialized[i];
#ifdef STORM_WITH_MPI
        destination.rank = this->rankWorld;
#endif
        destination.id = firstID + i;

#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
        destination.checkedHere = true;
        destination.nextRank = std::numeric_limits<rank_t>::max();
        destination.removedFromRank = false;
        destination.sentByRank = std::numeric_limits<rank_t>::max();
        destination.lastSeen = 0;
        destination.lastSeenRank = std::numeric_limits<rank_t>::max();
        destination.lastSeenRankBuf = std::numeric_limits<rank_t>::max();
        destination.lastSeenIndex = std::numeric_limits<size_t>::max();
#endif // STORM_DEBUG

#ifdef STORM_DEBUG
        if(not this->grid.IsPointInCell(destination.location, destination.cellIndex))
        {
            const T &declaredCell = this->grid.GetMeshPoint(destination.cellIndex);
            size_t containingIdx = this->grid.GetContainingCell(destination.location);
            const T &containingCell = this->grid.GetMeshPoint(containingIdx);
            STORMError eo("MonteCarloManager<T, Grid>::AddParticles");
            eo.addEntry("rank", this->rankWorld);
            eo.addEntry("Particle", destination);
            eo.addEntry("Declared Cell Index", destination.cellIndex);
            eo.addEntry("Declared Cell", declaredCell);
            eo.addEntry("Declared Cell - Distance", abs(declaredCell - destination.location));
            eo.addEntry("Real Containing Cell Index", containingIdx);
            eo.addEntry("Real Containing Cell", containingCell);
            eo.addEntry("Real Cell - Distance", abs(containingCell - destination.location));
            throw eo;
        }
#endif // STORM_DEBUG
    }
    this->engine->AppendLocal(initialized);

    this->localDecrementAmount -= static_cast<CompletionCounter>(particlesNum);
}

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
    boost::container::flat_set<std::pair<rank_t, size_t>> particlesSet;
    for(const MCParticle &particle : particles)
    {
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            continue;
        }
        std::pair<rank_t, size_t> particleSetKey = {particle.rank, particle.id};
        if(particlesSet.find(particleSetKey) != particlesSet.end())
        {
            STORMError eo("Particle with the same ID is being added to the same rank twice");
            eo.addEntry("Particle", particle);
            eo.addEntry("Rank", this->rankWorld);
            eo.addEntry("ID", particle.id);
            throw eo;
        }
        particlesSet.insert(particleSetKey);
    }
#endif // STORM_DEBUG

    size_t particlesNum = particles.size();
    if(particlesNum == 0)
    {
        return;
    }

    for(MCParticle &destination : particles)
    {
        if(destination.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned, assign now
#ifdef STORM_WITH_MPI
            destination.rank = this->rankWorld;
#endif
            destination.id = this->myIDCounter++;
        }
    }
    this->engine->AppendLocal(particles);

    // don't waste memory - remove current particles from the input vector
    std::vector<MCParticle> empty;
    particles.swap(empty);
}

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::StageLocalParticlesForDevice(std::vector<MCParticle> &&particles, bool assignNewIDs)
{
    if(particles.empty())
    {
        return;
    }

    std::vector<MCParticle> &staged = this->detachedRankParticles[static_cast<size_t>(this->rankWorld)];
    staged.reserve(staged.size() + particles.size());
    for(MCParticle &particle : particles)
    {
        if(assignNewIDs or particle.id == std::numeric_limits<particle_id_t>::max())
        {
#ifdef STORM_WITH_MPI
            particle.rank = this->rankWorld;
#endif
            particle.id = this->myIDCounter++;
        }
        staged.push_back(std::move(particle));
    }
    std::vector<MCParticle>().swap(particles);
}
template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::PrintMemoryDiagnostics(size_t initial, size_t generated)
{
    if(rankWorld == 0)
    {
        std::cout << "Communication memory on rank 0: " << engine->MemoryBytes()
                  << " bytes; starting local particles: " << initial + generated << std::endl;
    }
}

// Applies one physics event to a particle. This is the single copy of the
// transport/communication policy used by both host and device completion
// paths.
template<typename T, typename Grid, typename Physics>
typename MonteCarloManager<T, Grid, Physics>::TransportEventAction
MonteCarloManager<T, Grid, Physics>::ApplyTransportEvent(MCParticle &particle,
                                                             const MonteCarloFunctionality &functionality,
                                                             rank_t bufferRank, size_t particleIndex,
                                                             MonteCarloStepFinalData &stepData,
                                                             const TransportStepContext &context)
{
    (void)bufferRank;
    (void)particleIndex;
    (void)context;

    if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
    {
        size_t nextCellIndex = functionality.nextCellIndex;

        assert(nextCellIndex != particle.cellIndex);
        assert(particle.timeLeft >= 0);

#ifdef STORM_DEBUG
        auto throwCellMoveOutsideBox = [&](const std::string &cellMoveTarget)
        {
            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
            double relDrift = 0.0;
            double maxAxisRelDrift = 0.0;
            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

            STORMError eo("MonteCarloManager: CELL_MOVE moved particle outside box before a non-boundary cell move");
            eo.addEntry("Rank", this->rankWorld);
            eo.addEntry("Particle after step", particle);
            eo.addEntry("Cell move target", cellMoveTarget);
            eo.addEntry("Next cell index", nextCellIndex);
            eo.addEntry("Location before step", context.beforeStepLocation);
            eo.addEntry("Velocity before step", context.beforeStepVelocity);
            eo.addEntry("Time left before step", context.beforeStepTimeLeft);
            eo.addEntry("Box lower", boxLL);
            eo.addEntry("Box upper", boxUR);
            eo.addEntry("Relative drift", relDrift);
            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
            eo.addEntry("Cell count", this->nCells);
            if(particle.cellIndex < this->nCells)
            {
                eo.addEntry("Cell index", particle.cellIndex);
                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
            }
            throw eo;
        };
#endif // STORM_DEBUG

        if(__builtin_expect(nextCellIndex < this->nCells, 1))
        {
#ifdef STORM_DEBUG
            if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
            {
                throwCellMoveOutsideBox("local cell move");
            }
#endif // STORM_DEBUG

#ifdef STORM_DEBUG
            size_t previousCell = particle.cellIndex;
#endif // STORM_DEBUG
            particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
            particle.cellIndex = nextCellIndex;
#ifdef STORM_DEBUG
            if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
            {
                const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                size_t containingIdx = this->grid.GetContainingCell(particle.location);
                const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                STORMError eo("Particle is in Wrong Location");
                eo.addEntry("rank", this->rankWorld);
                eo.addEntry("Particle", particle);
                eo.addEntry("Previous Cell Index", previousCell);
                eo.addEntry("Previous Cell", this->grid.GetMeshPoint(previousCell));
                eo.addEntry("Previous Location", context.previousLocation);
                eo.addEntry("Last location is in previous cell?", this->grid.IsPointInCell(context.previousLocation, previousCell));
                eo.addEntry("Declared Cell Index", particle.cellIndex);
                eo.addEntry("Declared Cell", declaredCell);
                eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                eo.addEntry("Real Containing Cell Index", containingIdx);
                eo.addEntry("Real Containing Cell", containingCell);
                eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                throw eo;
            }
#endif // STORM_DEBUG
            return TransportEventAction::Continue;
        }
        else
        {
            auto it = ranksGhostMap.find(nextCellIndex);
            if(it == ranksGhostMap.end())
            {
                const size_t physicalCell = ResolvePhysicalCellIndex(this->grid, nextCellIndex, this->nCells);
                if(physicalCell < this->nCells)
                {
                    ApplyPeriodicCellMove(this->grid, particle.location, nextCellIndex, particle.velocity);
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(physicalCell);
                    if(not this->grid.IsPointInCell(particle.location, physicalCell))
                    {
                        particle.location = this->grid.GetCellCM(physicalCell);
                    }
                    particle.cellIndex = physicalCell;
                    return TransportEventAction::Continue;
                }
#ifdef STORM_WITH_TRACING_HISTORY
                T preReflectLoc = particle.location;
                T preReflectVel = particle.velocity;
#endif // STORM_WITH_TRACING_HISTORY
                MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                this->physics->onBoundaryResult(particle, status, functionality.boundaryCrossing && this->boundaryCondition->isEscape(status));
                if(status == MonteCarloParticleStatus::REFLECT)
                {
#ifdef STORM_WITH_TRACING_HISTORY
                    particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
#endif // STORM_WITH_TRACING_HISTORY
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                    return TransportEventAction::Continue;
                }
                else if(status == MonteCarloParticleStatus::REMOVE)
                {
                    stepData.leavingCount++;
                    this->allStepsCounter += particle.steps;
                    this->localDecrementAmount += 1;
                    ++this->progressRemovedCount;
                }
                else
                {
                    STORMError eo("Unknown boundary condition for particle");
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Status", status);
                    throw eo;
                }
                return TransportEventAction::Finished;
            }

#ifdef STORM_DEBUG
            if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
            {
                throwCellMoveOutsideBox("remote rank transfer");
            }
#endif // STORM_DEBUG

            particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
            this->grid.WrapPeriodicPoint(particle.location);
            auto [otherRank, neighborIndexInRank] = it->second;
#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
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
                eo.addEntry("My Rank", this->rankWorld);
                eo.addEntry("Next Rank", otherRank);
                eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                throw eo;
            }
            particle.cellIndexInPrevRank = particle.cellIndex;
            particle.sentByRank = this->rankWorld;
            particle.ghostIndex = nextCellIndex;
            particle.newCellValue = this->grid.GetMeshPoint(nextCellIndex);
            particle.particleIndexInLastRank = particleIndex;
            particle.particleTHInLastRank = particleIndex;
            particle.nextRank = otherRank;
            if(particle.nextRank == this->rankWorld)
            {
                STORMError eo("Particle is going to be sent to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("My Rank", this->rankWorld);
                eo.addEntry("Next Rank", otherRank);
                eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                throw eo;
            }
#endif // STORM_DEBUG

            particle.sent = true;
            particle.cellIndex = neighborIndexInRank;
            particle.sent = false;
            this->engine->Send(otherRank, particle);
            return TransportEventAction::Finished;
        }
    }
    else if(functionality.change == MonteCarloParticleStatus::REMOVE)
    {
        this->allStepsCounter += particle.steps;
        this->localDecrementAmount += 1;
        return TransportEventAction::Finished;
    }
    else if(functionality.change == MonteCarloParticleStatus::DONE)
    {
        stepData.remaining.push_back(particle);
        this->allStepsCounter += particle.steps;
        this->localDecrementAmount += 1;
        return TransportEventAction::Finished;
    }
    else if(functionality.change == MonteCarloParticleStatus::NO_CELL_MOVE)
    {
        return TransportEventAction::Continue;
    }

    STORMError eo("Unknown Monte Carlo particle status");
    eo.addEntry("Particle", particle);
    eo.addEntry("Status", functionality.change);
    throw eo;
}

#ifdef STORM_WITH_GPU
template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::CollectHostParticlesForDevice(std::vector<MCParticle> &arrivals)
{
    arrivals.clear();
    const size_t handlerCount = static_cast<size_t>(this->sizeWorld);
    for(size_t rank = 0; rank < handlerCount; ++rank)
    {
        std::vector<MCParticle> &deferred = this->detachedRankParticles[rank];
        if(not deferred.empty())
        {
            arrivals.insert(arrivals.end(), std::make_move_iterator(deferred.begin()), std::make_move_iterator(deferred.end()));
            deferred.clear();
        }

        if(this->engine->LocalSize(rank) == 0)
        {
            continue;
        }
        this->mergeScratchBuffer.clear();
        this->engine->Detach(rank, this->mergeScratchBuffer);
        arrivals.insert(arrivals.end(), std::make_move_iterator(this->mergeScratchBuffer.begin()), std::make_move_iterator(this->mergeScratchBuffer.end()));
        this->mergeScratchBuffer.clear();
    }
}

// Ingest host arrivals into the resident device pool, advance one wave, and
// apply host policy only to packets that left the GCD (rank hops, HostOnly,
// REMOVE). DONE stays on device until Comb; survivors keep transporting.
template<typename T, typename Grid, typename Physics>
bool MonteCarloManager<T, Grid, Physics>::TransportResidentOnDevice(std::vector<MCParticle> &arrivals, MonteCarloStepFinalData &stepData, bool &isEmpty)
{
    if constexpr(not gpu::HasDeviceTransport<Physics>::value)
    {
        return false;
    }
    else
    {
        if(not this->physics->UsesDeviceTransport())
        {
            return false;
        }
        if(not this->gpuTransportExecutor)
        {
            this->gpuTransportExecutor = std::make_unique<gpu::KokkosLocalTransportExecutor>(
                        this->config.gpuMaxInnerSteps,
                        this->config.gpuOverlapCommunication && this->sizeWorld > 1);
        }

        for(MCParticle &particle : arrivals)
        {
            if(particle.sent)
            {
                particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                particle.sent = false;
            }
        }

        const std::size_t incoming = arrivals.size();
        const std::chrono::steady_clock::time_point packStart = std::chrono::steady_clock::now();
        this->gpuIngestCount += incoming;
        this->gpuTransportExecutor->Ingest(arrivals);
        this->gpuPackSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();
        arrivals.clear();

        if(this->gpuTransportExecutor->ActiveCount() == 0 and
           this->gpuTransportExecutor->PendingRemoteCount() == 0 and
           !this->gpuTransportExecutor->HasPendingRemoteCopy())
        {
            this->gpuHoldSkips = 0;
            return true;
        }

        isEmpty = false;
        const std::size_t activeCount = this->gpuTransportExecutor->ActiveCount();
        const std::size_t minLaunch = this->config.gpuMinLaunchSize;
        const bool fatEnough = minLaunch == 0 or activeCount >= minLaunch or
            this->gpuTransportExecutor->HasPendingRemoteCopy();
        const bool heldTooLong = (this->config.gpuHoldMaxSkips > 0) and (this->gpuHoldSkips >= this->config.gpuHoldMaxSkips);
        if(not fatEnough and not heldTooLong)
        {
            ++this->gpuHoldSkips;
            ++this->gpuHoldCount;
            gpu::CompletedBatch held = this->gpuTransportExecutor->FlushPendingRemotes(minLaunch, this->config.gpuHoldMaxSkips, false);
            this->gpuCopyBackSeconds += held.copyBackSeconds;
            this->ApplyDeviceCompletions(held, stepData);
            return true;
        }
        this->gpuHoldSkips = 0;

        gpu::CompletedBatch completed = this->gpuTransportExecutor->AdvanceWave(
            this->physics->GetDeviceTransportViews(),
            [this]()
            {
                // Keep queue credits/reallocation handshakes and pending
                // sends moving while the GPU owns the transport arrays.
                this->engine->Progress();
                this->engine->Flush(false);
            },
            minLaunch,
            this->config.gpuHoldMaxSkips,
            [this, &stepData](gpu::CompletedBatch &remotes)
            {
                this->ApplyDeviceCompletions(remotes, stepData, true);
                this->engine->Flush(false);
            });
        this->gpuDeviceSeconds += completed.deviceSeconds;
        this->gpuCopyBackSeconds += completed.copyBackSeconds;
        this->gpuProgressSeconds += completed.progressSeconds;
        this->gpuLaunchCount += completed.launchCount;
        this->gpuParticleCount += completed.launchedParticles;
        this->gpuPhysicsStepCount += completed.physicsSteps;
        this->localDecrementAmount += static_cast<CompletionCounter>(completed.censusCount);
        this->localDecrementAmount -= static_cast<CompletionCounter>(completed.createdParticles);
        this->dynamicallyAdded += completed.createdParticles;
        this->allStepsCounter += completed.censusSteps;
        this->ApplyDeviceCompletions(completed, stepData);
        return true;
    }
}

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::ApplyDeviceCompletions(gpu::CompletedBatch &completed, MonteCarloStepFinalData &stepData,
                                                               const bool transportInFlight)
{
    STORM_PROFILE_REGION("storm/host_events");
    if(completed.terminals.empty() and completed.fallbacks.empty() and completed.remotes.empty())
    {
        return;
    }

    const TransportStepContext context;
    std::vector<MCParticle> bounced;
    const std::chrono::steady_clock::time_point hostEventStart = std::chrono::steady_clock::now();
    auto ensureHostIdentity = [this](MCParticle &particle)
    {
        if(particle.id == std::numeric_limits<particle_id_t>::max())
        {
#ifdef STORM_WITH_MPI
            particle.rank = this->rankWorld;
#endif
            particle.id = this->myIDCounter++;
        }
    };

    auto processFallbacks = [this, &stepData, &context, &bounced, &ensureHostIdentity](const gpu::CompletedTransportSpan span)
    {
        for(const gpu::CompletedTransport &transported : span)
        {
            if(transported.result.error != gpu::TransportError::HostFallback)
            {
                STORMError eo("MonteCarloManager: non-fallback packet in fallback batch");
                eo.addEntry("Rank", this->rankWorld);
                eo.addEntry("Transport error", static_cast<int>(transported.result.error));
                throw eo;
            }
            MCParticle particle;
            gpu::UnpackParticle(transported.particle, transported.cold, particle);
            ensureHostIdentity(particle);
            std::vector<MCParticle> particlesToAdd;
            MonteCarloFunctionality functionality = this->physics->step(particle, particlesToAdd);
            if(functionality.change == MonteCarloParticleStatus::NO_CELL_MOVE or
               this->ApplyTransportEvent(particle, functionality, this->rankWorld, 0, stepData, context) == TransportEventAction::Continue)
            {
                bounced.push_back(std::move(particle));
            }
            this->localDecrementAmount -= static_cast<CompletionCounter>(particlesToAdd.size());
            this->dynamicallyAdded += particlesToAdd.size();
            for(MCParticle &extra : particlesToAdd)
            {
                ensureHostIdentity(extra);
                if(this->ApplyTransportEvent(extra, functionality, this->rankWorld, 0, stepData, context) == TransportEventAction::Continue)
                {
                    bounced.push_back(std::move(extra));
                }
            }
        }
    };

    auto processOrdinary = [this, &stepData, &context, &bounced, &ensureHostIdentity](const gpu::CompletedTransportSpan span)
    {
        for(const gpu::CompletedTransport &transported : span)
        {
            if(transported.result.error != gpu::TransportError::None)
            {
                STORMError eo("MonteCarloManager: device grey transport failed");
                eo.addEntry("Rank", this->rankWorld);
                eo.addEntry("Cell index", transported.particle.cellIndex);
                eo.addEntry("Transport error", static_cast<int>(transported.result.error));
                throw eo;
            }

            if(transported.result.step.change == MonteCarloParticleStatus::REMOVE)
            {
                this->allStepsCounter += transported.particle.steps;
                this->localDecrementAmount += 1;
                ++this->gpuElidedRemovalCount;
                continue;
            }

            MCParticle particle;
            gpu::UnpackParticle(transported.particle, transported.cold, particle);
            ensureHostIdentity(particle);
            if(transported.result.step.change == MonteCarloParticleStatus::NO_CELL_MOVE or
               this->ApplyTransportEvent(particle, transported.result.step, this->rankWorld, 0, stepData, context) == TransportEventAction::Continue)
            {
                bounced.push_back(std::move(particle));
            }
        }
    };

    processFallbacks(completed.fallbacks);
    processOrdinary(completed.terminals);
    processOrdinary(completed.remotes);
    this->gpuHostEventSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - hostEventStart).count();

    if(not bounced.empty())
    {
        if(transportInFlight)
            throw STORMError("A pipelined rank-hop unexpectedly returned to local transport");
        this->gpuIngestCount += bounced.size();
        this->gpuTransportExecutor->Ingest(bounced);
    }
}

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::DrainDeviceCensus(MonteCarloStepFinalData &stepData)
{
    if(not this->gpuTransportExecutor)
    {
        return;
    }
    gpu::CompletedBatch census = this->gpuTransportExecutor->FlushPendingCensus();
    if(census.census.empty())
    {
        return;
    }
    for(const gpu::CompletedTransport &transported : census.census)
    {
        MCParticle particle;
        gpu::UnpackParticle(transported.particle, transported.cold, particle);
        if(particle.id == std::numeric_limits<particle_id_t>::max())
        {
#ifdef STORM_WITH_MPI
            particle.rank = this->rankWorld;
#endif
            particle.id = this->myIDCounter++;
        }
        if(transported.result.error != gpu::TransportError::None)
        {
            STORMError eo("MonteCarloManager: device census packet failed");
            eo.addEntry("Rank", this->rankWorld);
            eo.addEntry("Particle", particle);
            eo.addEntry("Transport error", static_cast<int>(transported.result.error));
            throw eo;
        }
        if(transported.result.step.change != MonteCarloParticleStatus::DONE)
        {
            STORMError eo("MonteCarloManager: device census pool held a non-DONE packet");
            eo.addEntry("Rank", this->rankWorld);
            eo.addEntry("Particle", particle);
            eo.addEntry("Status", transported.result.step.change);
            throw eo;
        }
        stepData.remaining.push_back(std::move(particle));
    }
}

template<typename T, typename Grid, typename Physics>
bool MonteCarloManager<T, Grid, Physics>::TransportBatchOnDevice(std::vector<MCParticle> &localParticles,
                                                                     rank_t bufferRank,
                                                                     MonteCarloStepFinalData &stepData,
                                                                     bool &isEmpty)
{
    return this->TransportResidentOnDevice(localParticles, stepData, isEmpty);
}
#endif // STORM_WITH_GPU

template<typename T, typename Grid, typename Physics>
bool MonteCarloManager<T, Grid, Physics>::HandleAll(MonteCarloStepFinalData &stepData)
{
    std::vector<MCParticle> &particlesToAdd = this->particlesToAdd;
    size_t &progressStepCounter = this->progressStepCounter;
    std::vector<rank_t> &currentActiveRanks = this->activeRanks;
    std::vector<rank_t> &nextActiveRankList = this->nextActiveRanks;

    nextActiveRankList.clear();
    bool completedNeighborSweep = true;
    auto hasDetachedParticles = [this](rank_t rank)
    {
        return rank >= 0 and rank < static_cast<rank_t>(this->detachedRankParticles.size()) and
               not this->detachedRankParticles[static_cast<size_t>(rank)].empty();
    };

    bool scanForHostTransport = true;
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(this->physics->UsesDeviceTransport())
        {
            scanForHostTransport = false;
        }
    }
#endif

    if(currentActiveRanks.empty() and scanForHostTransport)
    {
        size_t neighborsNum = this->neighbors.size();
        if(neighborsNum > 0 and this->activeRankScanRemaining == 0)
        {
            this->activeRankScanRemaining = neighborsNum;
            this->activeRankScanCursor %= neighborsNum;
        }
        size_t scanCount = (neighborsNum == 0) ? 0 : std::min(this->activeRankScanRemaining, std::min(neighborsNum, std::max<size_t>(1, this->config.activeRankScanChunk)));

        for(size_t scanOffset = 0; scanOffset < scanCount; ++scanOffset)
        {
            size_t i = (this->activeRankScanCursor + scanOffset) % neighborsNum;
            rank_t rank = this->neighbors[i];
            if(hasDetachedParticles(rank))
            {
                currentActiveRanks.push_back(rank);
                continue;
            }
            size_t len = this->engine->LocalSize(rank);
            if(len)
            {
                currentActiveRanks.push_back(rank);
            }
        }
        if(neighborsNum > 0)
        {
            this->activeRankScanCursor = (this->activeRankScanCursor + scanCount) % neighborsNum;
            assert(this->activeRankScanRemaining >= scanCount);
            this->activeRankScanRemaining -= scanCount;
            completedNeighborSweep = (this->activeRankScanRemaining == 0);
        }
        {
            if(hasDetachedParticles(this->rankWorld) or this->engine->LocalSize(this->rankWorld) != 0)
            {
                currentActiveRanks.push_back(this->rankWorld);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = currentActiveRanks.size();

    bool handledDeviceSweep = false;
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(this->physics->UsesDeviceTransport())
        {
            const std::chrono::steady_clock::time_point mergeStart = std::chrono::steady_clock::now();
            std::vector<MCParticle> &arrivals = this->mergedParticleBuffer;
            this->CollectHostParticlesForDevice(arrivals);
            this->loopMergeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - mergeStart).count();

            if(this->TransportResidentOnDevice(arrivals, stepData, isEmpty))
            {
                handledDeviceSweep = true;
                completedNeighborSweep = true;
                currentActiveRanks.clear();
                nextActiveRankList.clear();
            }
        }
    }
#endif

    if(not handledDeviceSweep)
    {
        for(size_t index = 0; index < activeRanksNum; index++)
        {
            rank_t rank = currentActiveRanks[index];
            std::vector<MCParticle> &deferredParticles = this->detachedRankParticles[static_cast<size_t>(rank)];
            std::vector<MCParticle> localParticles;

            if(not deferredParticles.empty())
            {
                localParticles.swap(deferredParticles);
            }
            else
            {
                this->engine->Detach(rank, localParticles);
            }

#ifdef STORM_WITH_GPU
            if(this->TransportBatchOnDevice(localParticles, rank, stepData, isEmpty))
            {
                if(not localParticles.empty())
                {
                    assert(deferredParticles.empty());
                    deferredParticles.swap(localParticles);
                }
                continue;
            }
#endif // STORM_WITH_GPU

            auto processParticle = [&](MCParticle &particle, size_t particleIndex)
            {
                bool removeCurrent = false;
                bool debug = false;

                try
                {
#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
                    if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rankWorld)
                    {
                        STORMError eo("Particle was already handled in this iteration");
                        eo.addEntry("My Rank", this->rankWorld);
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Iteration", this->iteration);
                        eo.addEntry("In Rank Buffer (1)", particle.lastSeenRankBuf);
                        eo.addEntry("In List Index (1)", particle.lastSeenIndex);
                        eo.addEntry("In Rank Buffer (2)", rank);
                        eo.addEntry("In List Index (2)", particleIndex);
                        throw eo;
                    }
                    particle.lastSeen = this->iteration;
                    particle.lastSeenRankBuf = rank;
                    particle.lastSeenRank = this->rankWorld;
                    particle.lastSeenIndex = particleIndex;
#endif // STORM_DEBUG

                    isEmpty = false;
                    constexpr size_t communicationProgressInterval = 1024;
                    constexpr size_t stuckParticleWarnInterval = 256 * 1024;
                    while(true)
                    {
                        ++progressStepCounter;
                        if((progressStepCounter % communicationProgressInterval) == 0)
                        {
                            this->engine->Poll();
                        }
                        if((progressStepCounter % stuckParticleWarnInterval) == 0 && particle.steps > 100000)
                        {
                            std::cerr << "[StuckParticle] rank=" << this->rankWorld
                                      << " localPts=" << this->grid.GetPointNo()
                                      << " " << particle
                                      << " freq=" << particle.frequency
                                      << " w/w0=" << (particle.initialWeight > 0 ? particle.weight / particle.initialWeight : 0.0) << std::endl;
                            if(this->progressCellsPtr && particle.cellIndex < this->nCells)
                            {
                                std::cerr << " cellIndex=" << particle.cellIndex << std::endl;
                            }
                            std::string accelInfo = this->physics->getAccelerationDebugInfo(particle.cellIndex, particle.frequency);
                            if(!accelInfo.empty())
                            {
                                std::cerr << accelInfo << std::endl;
                            }
                            std::cerr << std::endl;
                        }

                        const size_t traceStep = particle.steps;
                        if(particle.on_track)
                        {
                            MCParticle trackedParticle = particle;
                            trackedParticle.steps = traceStep * 2;
                            this->tracker.ReportParticle(trackedParticle);
                        }
                        particle.steps++;
                        this->cellsStepsCounters[particle.cellIndex]++;

#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
                        if(particle.cellIndex >= this->nCells)
                        {
                            STORMError eo("Particle has invalid cell index (ghost)");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Cell Index", particle.cellIndex);
                            eo.addEntry("Rank", this->rankWorld);
                            eo.addEntry("Buffer of Rank", rank);
                            throw eo;
                        }
                        if(particle.removedFromRank)
                        {
                            STORMError eo("Particle was removed from rank, but still in the list");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Rank", this->rankWorld);
                            eo.addEntry("Buffer of Rank", rank);
                            throw eo;
                        }
                        if(not particle.checkedHere)
                        {
                            if(particle.nextRank != this->rankWorld)
                            {
                                STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                                eo.addEntry("Particle", particle);
                                eo.addEntry("Origin", particle.sentByRank);
                                eo.addEntry("Particle Previous Location", particle.previousLocation);
                                eo.addEntry("Cell Index In Origin (Before Movement)", particle.cellIndexInPrevRank);
                                eo.addEntry("Expected", particle.nextRank);
                                eo.addEntry("Got (me)", this->rankWorld);
                                eo.addEntry("The Particle Index In Last Rank", particle.particleIndexInLastRank);
                                eo.addEntry("Particle Index In This Rank", particleIndex);
                                eo.addEntry("The Particle TH In Last Rank", particle.particleTHInLastRank);
                                eo.addEntry("Particle TH In This Rank", particleIndex);
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
                            if(containingIdx != particle.cellIndex and
                               not this->grid.IsPointInCell(particle.location, containingIdx))
                            {
                                STORMError eo("Particle Arrived to a Wrong Rank After Transfer");
                                eo.addEntry("My Rank", this->rankWorld);
                                eo.addEntry("Transferred From Rank", rank);
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
#endif // STORM_DEBUG

                        if(particle.sent)
                        {
                            particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                                MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                            particle.sent = false;
                        }

#ifdef STORM_DEBUG
                        T prevLoc = particle.location;
#ifdef STORM_WITH_MPI
                        particle.previousLocation = particle.location;
#endif
#endif // STORM_DEBUG

                        if(debug)
                        {
                            std::cout << "Before running particle step, particle is " << particle << std::endl;
                        }

#ifdef STORM_DEBUG
                        const T beforeStepLocation = particle.location;
                        const T beforeStepVelocity = particle.velocity;
                        const dt_t beforeStepTimeLeft = particle.timeLeft;
                        if(__builtin_expect(this->grid.IsPointOutsideBox(particle.location), 0))
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            STORMError eo("MonteCarloManager: particle outside box before physics step");
                            eo.addEntry("Rank", this->rankWorld);
                            eo.addEntry("Particle before step", particle);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->nCells);
                            if(particle.cellIndex < this->nCells)
                            {
                                eo.addEntry("Cell index", particle.cellIndex);
                                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                eo.addEntry("Inside declared cell before step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                            }
                            throw eo;
                        }
#endif // STORM_DEBUG

                        MonteCarloFunctionality functionality = this->physics->step(particle, particlesToAdd);

#ifdef STORM_DEBUG
                        if(__builtin_expect(functionality.change != MonteCarloParticleStatus::REMOVE &&
                                                functionality.change != MonteCarloParticleStatus::CELL_MOVE &&
                                                this->grid.IsPointOutsideBox(particle.location),
                                            0))
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            STORMError eo("MonteCarloManager: physics step moved particle outside the box");
                            eo.addEntry("Rank", this->rankWorld);
                            eo.addEntry("Particle after step", particle);
                            eo.addEntry("Functionality", MonteCarloParticleStatusToString(functionality.change));
                            eo.addEntry("Next cell index", functionality.nextCellIndex);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->nCells);
                            if(particle.cellIndex < this->nCells)
                            {
                                eo.addEntry("Cell index", particle.cellIndex);
                                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                            }
                            throw eo;
                        }
#endif // STORM_DEBUG

                        if(particle.on_track)
                        {
                            MCParticle trackedParticle = particle;
                            trackedParticle.steps = traceStep * 2 + 1;
                            this->tracker.ReportParticle(trackedParticle);
                        }

#ifdef STORM_WITH_TRACING_HISTORY
                        particle.recordHistory(particle.cellIndex, static_cast<int>(this->rankWorld), static_cast<int>(functionality.change));
#endif // STORM_WITH_TRACING_HISTORY

                        TransportStepContext context;
#ifdef STORM_DEBUG
                        context.beforeStepLocation = beforeStepLocation;
                        context.beforeStepVelocity = beforeStepVelocity;
                        context.beforeStepTimeLeft = beforeStepTimeLeft;
                        context.previousLocation = prevLoc;
#endif // STORM_DEBUG

                        if(this->ApplyTransportEvent(particle, functionality, rank, particleIndex,
                                                     stepData, context) == TransportEventAction::Finished)
                        {
                            removeCurrent = true;
                            break;
                        }
                    }
                }
                catch(STORMError &eo)
                {
                    eo.addEntry("Particle list index", particleIndex);
                    eo.addEntry("Handler rank buffer", rank);
                    throw eo;
                }

                assert(removeCurrent);
            };
            const size_t particleCount = std::min(std::max<size_t>(1, this->config.localTransportBatchSize), localParticles.size());
            for(size_t processed = 0; processed < particleCount; ++processed)
            {
                const size_t particleIndex = localParticles.size() - 1;
                processParticle(localParticles.back(), particleIndex);
                localParticles.pop_back();
            }

            if(not localParticles.empty())
            {
                assert(deferredParticles.empty());
                deferredParticles.swap(localParticles);
            }
        }
    }

    if(not handledDeviceSweep)
    {
        for(size_t i = 0; i < activeRanksNum; i++)
        {
            rank_t rank = currentActiveRanks[i];
            if(hasDetachedParticles(rank) or this->engine->LocalSize(rank) != 0)
            {
                nextActiveRankList.push_back(rank);
            }
        }
        currentActiveRanks.swap(nextActiveRankList);
    }

    if(not isEmpty)
    {
        this->activeRankScanRemaining = 0;
        completedNeighborSweep = false;
    }
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(handledDeviceSweep)
        {
            completedNeighborSweep = true;
            const bool deviceBusy = this->gpuTransportExecutor and this->gpuTransportExecutor->DeviceBusy();
            isEmpty = not deviceBusy;
        }
    }
#endif
    bool toReturn = isEmpty and completedNeighborSweep and particlesToAdd.empty();
    if(not particlesToAdd.empty())
    {
        this->dynamicallyAdded += particlesToAdd.size();
        this->AddParticles(particlesToAdd);
        particlesToAdd.clear();
    }

    return toReturn;
}

#endif // STORM_MONTE_CARLO_TRANSPORT_HPP
