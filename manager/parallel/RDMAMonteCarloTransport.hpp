#ifndef STORM_RDMA_MONTE_CARLO_TRANSPORT_HPP
#define STORM_RDMA_MONTE_CARLO_TRANSPORT_HPP

// Applies one physics event to a particle. This is the single copy of the
// transport/communication policy used by both host and device completion
// paths.
template<typename T, typename Grid, typename Physics>
typename RDMAMonteCarloManager<T, Grid, Physics>::TransportEventAction
RDMAMonteCarloManager<T, Grid, Physics>::ApplyTransportEvent(MCParticle &particle,
                                                             const MonteCarloFunctionality &functionality,
                                                             rank_t bufferRank, size_t particleIndex,
                                                             MonteCarloStepFinalData &stepData,
                                                             const TransportStepContext &context)
{
    (void) bufferRank;
    (void) particleIndex;
    (void) context;

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

            STORMError eo("RDMAMonteCarloManager: CELL_MOVE moved particle outside box before a non-boundary cell move");
            eo.addEntry("Rank", this->rank_world);
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
            eo.addEntry("Cell count", this->Ncells);
            if(particle.cellIndex < this->Ncells)
            {
                eo.addEntry("Cell index", particle.cellIndex);
                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
            }
            throw eo;
        };
        #endif // STORM_DEBUG

        if(__builtin_expect(nextCellIndex < this->Ncells, 1))
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
                eo.addEntry("rank", this->rank_world);
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
            auto it = ranks_ghost_map.find(nextCellIndex);
            if(it == ranks_ghost_map.end())
            {
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
                    ++this->progressRemovedCount_;
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
            particle.particleIndexInLastRank = particleIndex;
            particle.particleTHInLastRank = particleIndex;
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
            particle.sent = false;
            RegisteredSendBuffer_t &buffer = this->GetSendBuffer(otherRank);
            size_t previousSize = buffer.size();
            buffer.push_back(particle);
            this->NoteSendBufferGrowth(otherRank, previousSize, buffer, 1);
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
void RDMAMonteCarloManager<T, Grid, Physics>::CollectHostParticlesForDevice(std::vector<MCParticle> &arrivals)
{
    arrivals.clear();
    const size_t handlerCount = this->rankHandlers.size();
    for(size_t rank = 0; rank < handlerCount; ++rank)
    {
        std::vector<MCParticle> &deferred = this->detachedRankParticles[rank];
        if(not deferred.empty())
        {
            arrivals.insert(arrivals.end(),
                            std::make_move_iterator(deferred.begin()),
                            std::make_move_iterator(deferred.end()));
            deferred.clear();
        }

        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler == nullptr or handler->LocalEmpty())
        {
            continue;
        }
        this->mergeScratchBuffer.clear();
        handler->DetachLocalParticles(this->mergeScratchBuffer);
        arrivals.insert(arrivals.end(),
                        std::make_move_iterator(this->mergeScratchBuffer.begin()),
                        std::make_move_iterator(this->mergeScratchBuffer.end()));
        this->mergeScratchBuffer.clear();
    }
}

// Ingest host arrivals into the resident device pool, advance one wave, and
// apply host policy only to packets that left the GCD (rank hops, HostOnly,
// REMOVE). DONE stays on device until Comb; survivors keep transporting.
template<typename T, typename Grid, typename Physics>
bool RDMAMonteCarloManager<T, Grid, Physics>::TransportResidentOnDevice(std::vector<MCParticle> &arrivals, MonteCarloStepFinalData &stepData, bool &isEmpty)
{
    if constexpr(not gpu::HasDeviceTransport<Physics>::value)
    {
        (void) arrivals;
        (void) stepData;
        (void) isEmpty;
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
            this->gpuTransportExecutor =
                std::make_unique<gpu::KokkosLocalTransportExecutor>(this->config.gpuMaxInnerSteps);
        }

        for(MCParticle &particle : arrivals)
        {
            if(particle.sent)
            {
                particle.location = (1 - MONTECARLO_EPSILON) * particle.location +
                                    MONTECARLO_EPSILON * this->grid.GetMeshPoint(particle.cellIndex);
                particle.sent = false;
            }
        }

        const std::size_t incoming = arrivals.size();
        const std::chrono::steady_clock::time_point packStart = std::chrono::steady_clock::now();
        this->gpuIngestCount += incoming;
        this->gpuTransportExecutor->Ingest(arrivals);
        this->gpuPackSeconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();
        arrivals.clear();

        if(this->gpuTransportExecutor->ActiveCount() == 0 and
           this->gpuTransportExecutor->PendingRemoteCount() == 0)
        {
            this->gpuHoldSkips = 0;
            return true;
        }

        isEmpty = false;
        const std::size_t activeCount = this->gpuTransportExecutor->ActiveCount();
        const std::size_t minLaunch = this->config.gpuMinLaunchSize;
        const bool fatEnough = minLaunch == 0 or activeCount >= minLaunch;
        const bool heldTooLong =
            this->config.gpuHoldMaxSkips > 0 and
            this->gpuHoldSkips >= this->config.gpuHoldMaxSkips;
        if(not fatEnough and not heldTooLong)
        {
            ++this->gpuHoldSkips;
            ++this->gpuHoldCount;
            gpu::CompletedBatch held = this->gpuTransportExecutor->FlushPendingRemotes(
                minLaunch, this->config.gpuHoldMaxSkips, false);
            this->gpuCopyBackSeconds += held.copyBackSeconds;
            this->ApplyDeviceCompletions(held, stepData);
            return true;
        }
        this->gpuHoldSkips = 0;

        gpu::CompletedBatch completed = this->gpuTransportExecutor->AdvanceWave(
            this->physics->GetDeviceTransportViews(),
            [this]()
            {
                this->PumpRMAProgress();
            },
            minLaunch,
            this->config.gpuHoldMaxSkips);
        this->gpuDeviceSeconds += completed.deviceSeconds;
        this->gpuCopyBackSeconds += completed.copyBackSeconds;
        this->gpuProgressSeconds += completed.progressSeconds;
        this->gpuLaunchCount += completed.launchCount;
        this->gpuParticleCount += completed.launchedParticles;
        this->gpuPhysicsStepCount += completed.physicsSteps;
        this->localDecrementAmount +=
            static_cast<typename AmountManager::counter_t>(completed.censusCount);
        this->localDecrementAmount -=
            static_cast<typename AmountManager::counter_t>(
                completed.createdParticles);
        this->dynamicallyAdded += completed.createdParticles;
        this->allStepsCounter += completed.censusSteps;
        this->ApplyDeviceCompletions(completed, stepData);
        return true;
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ApplyDeviceCompletions(
    gpu::CompletedBatch &completed,
    MonteCarloStepFinalData &stepData)
{
    if(completed.terminals.empty() &&
       completed.fallbacks.empty() &&
       completed.remotes.empty())
    {
        return;
    }

    const TransportStepContext context;
    std::vector<MCParticle> bounced;
    const std::chrono::steady_clock::time_point hostEventStart =
        std::chrono::steady_clock::now();
    auto ensureHostIdentity = [this](MCParticle &particle)
    {
        if(particle.id == std::numeric_limits<particle_id_t>::max())
        {
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
    };

    auto processFallbacks =
        [this, &stepData, &context, &bounced, &ensureHostIdentity](
            const gpu::CompletedTransportSpan span)
    {
        for(const gpu::CompletedTransport &transported : span)
        {
            if(transported.result.error != gpu::TransportError::HostFallback)
            {
                STORMError eo(
                    "RDMAMonteCarloManager: non-fallback packet in fallback batch");
                eo.addEntry("Rank", this->rank_world);
                eo.addEntry(
                    "Transport error",
                    static_cast<int>(transported.result.error));
                throw eo;
            }
            MCParticle particle;
            gpu::UnpackParticle(
                transported.particle, transported.cold, particle);
            ensureHostIdentity(particle);
            std::vector<MCParticle> particlesToAdd;
            MonteCarloFunctionality functionality =
                this->physics->step(particle, particlesToAdd);
            if(functionality.change == MonteCarloParticleStatus::NO_CELL_MOVE or
               this->ApplyTransportEvent(particle, functionality, this->rank_world, 0,
                                         stepData, context) == TransportEventAction::Continue)
            {
                bounced.push_back(std::move(particle));
            }
            this->localDecrementAmount -=
                static_cast<typename AmountManager::counter_t>(
                    particlesToAdd.size());
            this->dynamicallyAdded += particlesToAdd.size();
            for(MCParticle &extra : particlesToAdd)
            {
                ensureHostIdentity(extra);
                if(this->ApplyTransportEvent(extra, functionality, this->rank_world, 0,
                                            stepData, context) == TransportEventAction::Continue)
                {
                    bounced.push_back(std::move(extra));
                }
            }
        }
    };

    auto processOrdinary =
        [this, &stepData, &context, &bounced, &ensureHostIdentity](
            const gpu::CompletedTransportSpan span)
    {
        for(const gpu::CompletedTransport &transported : span)
        {
            if(transported.result.error != gpu::TransportError::None)
            {
                STORMError eo(
                    "RDMAMonteCarloManager: device grey transport failed");
                eo.addEntry("Rank", this->rank_world);
                eo.addEntry(
                    "Cell index", transported.particle.cellIndex);
                eo.addEntry(
                    "Transport error",
                    static_cast<int>(transported.result.error));
                throw eo;
            }

            if(transported.result.step.change ==
               MonteCarloParticleStatus::REMOVE)
            {
                this->allStepsCounter += transported.particle.steps;
                this->localDecrementAmount += 1;
                ++this->gpuElidedRemovalCount;
                continue;
            }

            MCParticle particle;
            gpu::UnpackParticle(
                transported.particle, transported.cold, particle);
            ensureHostIdentity(particle);
            if(transported.result.step.change ==
                   MonteCarloParticleStatus::NO_CELL_MOVE or
               this->ApplyTransportEvent(
                   particle, transported.result.step, this->rank_world, 0,
                   stepData, context) == TransportEventAction::Continue)
            {
                bounced.push_back(std::move(particle));
            }
        }
    };

    processFallbacks(completed.fallbacks);
    processOrdinary(completed.terminals);
    processOrdinary(completed.remotes);
    this->gpuHostEventSeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - hostEventStart).count();

    if(not bounced.empty())
    {
        this->gpuIngestCount += bounced.size();
        this->gpuTransportExecutor->Ingest(bounced);
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::DrainDeviceCensus(MonteCarloStepFinalData &stepData)
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
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
        if(transported.result.error != gpu::TransportError::None)
        {
            STORMError eo("RDMAMonteCarloManager: device census packet failed");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Particle", particle);
            eo.addEntry("Transport error", static_cast<int>(transported.result.error));
            throw eo;
        }
        if(transported.result.step.change != MonteCarloParticleStatus::DONE)
        {
            STORMError eo("RDMAMonteCarloManager: device census pool held a non-DONE packet");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Particle", particle);
            eo.addEntry("Status", transported.result.step.change);
            throw eo;
        }
        stepData.remaining.push_back(std::move(particle));
    }
}

template<typename T, typename Grid, typename Physics>
bool RDMAMonteCarloManager<T, Grid, Physics>::TransportBatchOnDevice(std::vector<MCParticle> &localParticles,
                                                                     rank_t bufferRank,
                                                                     MonteCarloStepFinalData &stepData,
                                                                     bool &isEmpty)
{
    (void) bufferRank;
    return this->TransportResidentOnDevice(localParticles, stepData, isEmpty);
}
#endif // STORM_WITH_GPU

template<typename T, typename Grid, typename Physics>
bool RDMAMonteCarloManager<T, Grid, Physics>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<MCParticle> particlesToAdd;
    static size_t progressStepCounter;
    std::vector<rank_t> &active_ranks = this->activeRanks;
    std::vector<rank_t> &next_active_ranks = this->nextActiveRanks;

    next_active_ranks.clear();
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

    if(active_ranks.empty() and scanForHostTransport)
    {
        const int PREFETCH_DISTANCE = 3;
        size_t neighborsNum = this->neighbors.size();
        if(neighborsNum > 0 and this->activeRankScanRemaining == 0)
        {
            this->activeRankScanRemaining = neighborsNum;
            this->activeRankScanCursor %= neighborsNum;
        }
        size_t scanCount = (neighborsNum == 0) ? 0 :
            std::min(this->activeRankScanRemaining,
                     std::min(neighborsNum, std::max<size_t>(1, this->config.activeRankScanChunk)));

        for(size_t scanOffset = 0; scanOffset < scanCount; ++scanOffset)
        {
            size_t i = (this->activeRankScanCursor + scanOffset) % neighborsNum;
            if(scanOffset + PREFETCH_DISTANCE < scanCount)
            {
                size_t futureIndex = (this->activeRankScanCursor + scanOffset + PREFETCH_DISTANCE) % neighborsNum;
                RankHandler_t *future_handler = this->rankHandlers[this->neighbors[futureIndex]];
                __builtin_prefetch(future_handler, 0, 1);
                __builtin_prefetch((const void*) &(future_handler->tail), 0, 1);
            }

            rank_t _rank = this->neighbors[i];
            RankHandler_t *handler = this->rankHandlers[_rank];
            if(hasDetachedParticles(_rank))
            {
                active_ranks.push_back(_rank);
                continue;
            }
            size_t len = handler->LocalSize();
            if(len)
            {
                active_ranks.push_back(_rank);
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
            RankHandler_t *handler = this->rankHandlers[this->rank_world];
            if(hasDetachedParticles(this->rank_world) or not handler->LocalEmpty())
            {
                active_ranks.push_back(this->rank_world);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();

    bool handledDeviceSweep = false;
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(this->physics->UsesDeviceTransport())
        {
            const std::chrono::steady_clock::time_point mergeStart =
                std::chrono::steady_clock::now();
            std::vector<MCParticle> &arrivals = this->mergedParticleBuffer;
            this->CollectHostParticlesForDevice(arrivals);
            this->loopMergeSeconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - mergeStart).count();

            if(this->TransportResidentOnDevice(arrivals, stepData, isEmpty))
            {
                handledDeviceSweep = true;
                completedNeighborSweep = true;
                active_ranks.clear();
                next_active_ranks.clear();
            }
        }
    }
#endif

    if(not handledDeviceSweep)
    {
        for(size_t index = 0; index < activeRanksNum; index++)
        {
            rank_t _rank = active_ranks[index];
            RankHandler_t *handler = this->rankHandlers[_rank];
            std::vector<MCParticle> &deferredParticles = this->detachedRankParticles[static_cast<size_t>(_rank)];
            std::vector<MCParticle> localParticles;

            if(not deferredParticles.empty())
            {
                localParticles.swap(deferredParticles);
            }
            else
            {
                handler->DetachLocalParticles(localParticles);
            }

#ifdef STORM_WITH_GPU
        if(this->TransportBatchOnDevice(localParticles, _rank, stepData, isEmpty))
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
                    #ifdef STORM_DEBUG
                    if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
                    {
                        STORMError eo("Particle was already handled in this iteration");
                        eo.addEntry("My Rank", this->rank_world);
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Iteration", this->iteration);
                        eo.addEntry("In Rank Buffer (1)", particle.lastSeenRankBuf);
                        eo.addEntry("In List Index (1)", particle.lastSeenIndex);
                        eo.addEntry("In Rank Buffer (2)", _rank);
                        eo.addEntry("In List Index (2)", particleIndex);
                        throw eo;
                    }
                    particle.lastSeen = this->iteration;
                    particle.lastSeenRankBuf = _rank;
                    particle.lastSeenRank = this->rank_world;
                    particle.lastSeenIndex = particleIndex;
                    #endif // STORM_DEBUG

                    isEmpty = false;
                    constexpr size_t RMA_PROGRESS_INTERVAL = 1024;
                    constexpr size_t STUCK_PARTICLE_WARN_INTERVAL = 256 * 1024;
                    while(true)
                    {
                        ++progressStepCounter;
                        if((progressStepCounter % RMA_PROGRESS_INTERVAL) == 0)
                        {
                            this->PumpRMAProgress();
                        }
                        if((progressStepCounter % STUCK_PARTICLE_WARN_INTERVAL) == 0 && particle.steps > 100000)
                        {
                            std::cerr << "[StuckParticle] rank=" << this->rank_world
                                      << " localPts=" << this->grid.GetPointNo()
                                      << " " << particle
                                      << " freq=" << particle.frequency
                                      << " w/w0=" << (particle.initialWeight > 0 ? particle.weight / particle.initialWeight : 0.0) << std::endl;
                            if(this->progressCellsPtr_ && particle.cellIndex < this->Ncells)
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

                        #ifdef STORM_DEBUG
                        if(particle.cellIndex >= this->Ncells)
                        {
                            STORMError eo("Particle has invalid cell index (ghost)");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Cell Index", particle.cellIndex);
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Buffer of Rank", _rank);
                            throw eo;
                        }
                        if(particle.removedFromRank)
                        {
                            STORMError eo("Particle was removed from rank, but still in the list");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Buffer of Rank", _rank);
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
                                eo.addEntry("My Rank", this->rank_world);
                                eo.addEntry("Transferred From Rank", _rank);
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
                        particle.previousLocation = particle.location;
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

                            STORMError eo("RDMAMonteCarloManager: particle outside box before physics step");
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Particle before step", particle);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->Ncells);
                            if(particle.cellIndex < this->Ncells)
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
                                          this->grid.IsPointOutsideBox(particle.location), 0))
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            STORMError eo("RDMAMonteCarloManager: physics step moved particle outside the box");
                            eo.addEntry("Rank", this->rank_world);
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
                            eo.addEntry("Cell count", this->Ncells);
                            if(particle.cellIndex < this->Ncells)
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
                            particle.recordHistory(particle.cellIndex, static_cast<int>(this->rank_world), static_cast<int>(functionality.change));
                        #endif // STORM_WITH_TRACING_HISTORY

                        TransportStepContext context;
                        #ifdef STORM_DEBUG
                        context.beforeStepLocation = beforeStepLocation;
                        context.beforeStepVelocity = beforeStepVelocity;
                        context.beforeStepTimeLeft = beforeStepTimeLeft;
                        context.previousLocation = prevLoc;
                        #endif // STORM_DEBUG

                        if(this->ApplyTransportEvent(particle, functionality, _rank, particleIndex,
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
                eo.addEntry("Handler rank buffer", _rank);
                eo.addEntry("Handler head", static_cast<size_t>(handler->head));
                eo.addEntry("Handler tail", static_cast<size_t>(handler->tail));
                eo.addEntry("Handler buffer size", handler->buffsize);
                throw eo;
            }

            assert(removeCurrent);
        };
        this->transportCore.HandleAll(localParticles, processParticle,
                                       this->config.localTransportBatchSize);

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
            rank_t _rank = active_ranks[i];
            RankHandler_t *handler = this->rankHandlers[_rank];
            if(hasDetachedParticles(_rank) or not handler->LocalEmpty())
            {
                next_active_ranks.push_back(_rank);
            }
        }
        active_ranks.swap(next_active_ranks);
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
            const bool deviceBusy =
                this->gpuTransportExecutor and
                this->gpuTransportExecutor->DeviceBusy();
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

#endif // STORM_RDMA_MONTE_CARLO_TRANSPORT_HPP
