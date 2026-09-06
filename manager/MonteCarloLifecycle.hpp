#ifndef STORM_MONTE_CARLO_LIFECYCLE_HPP
#define STORM_MONTE_CARLO_LIFECYCLE_HPP

template<typename T, typename Grid, typename Physics>
void MonteCarloManager<T, Grid, Physics>::step(dt_t fullDt)
{
    STORM_PROFILE_REGION("storm/step");
    // if(this->nCells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rankWorld << ": " << this->nCells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }

    std::chrono::high_resolution_clock::time_point stepStart = std::chrono::high_resolution_clock::now();

    this->nCells = this->grid.GetPointNo();
#ifdef STORM_WITH_MPI
    if(this->commWorld != MPI_COMM_NULL)
    {
        this->ranksGhostMap = GetGhostMap(this->grid);
    }
#endif
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();

    this->engine->Prepare();
    this->neighbors = this->engine->Neighbors();
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;
    bool reuseDeviceCensus = false;
#ifdef STORM_WITH_GPU
    reuseDeviceCensus = this->deviceCensusValid and not this->HaveParticlesChanged() and this->gpuTransportExecutor;
    if(this->deviceCensusValid and not reuseDeviceCensus)
    {
        this->MaterializeDeviceCensus();
    }
    this->gpuPackSeconds = 0.0;
    this->gpuDeviceSeconds = 0.0;
    this->gpuCopyBackSeconds = 0.0;
    this->gpuProgressSeconds = 0.0;
    this->gpuHostEventSeconds = 0.0;
    this->gpuLaunchCount = 0;
    this->gpuParticleCount = 0;
    this->gpuIngestCount = 0;
    this->gpuPhysicsStepCount = 0;
    this->gpuHoldCount = 0;
    this->gpuElidedRemovalCount = 0;
    this->gpuHoldSkips = 0;
    if(this->gpuTransportExecutor)
    {
        if(reuseDeviceCensus)
        {
            this->gpuTransportExecutor->ResetStepMetrics();
        }
        else
        {
            this->gpuTransportExecutor->Reset();
        }
    }
#endif
    this->loopCommunicationSeconds = 0.0;
    this->loopAmountSeconds = 0.0;
    this->loopHandleSeconds = 0.0;
    this->loopMergeSeconds = 0.0;
    this->loopRounds = 0;
    this->loopIdleRounds = 0;

    bool didRebalance = false;
#ifdef STORM_WITH_MPI
    if(this->commWorld != MPI_COMM_NULL)
    {
        didRebalance = this->grid.DidRebalance() && this->lastBuildGeneration != this->grid.GetBuildGeneration();
        this->lastBuildGeneration = this->grid.GetBuildGeneration();
    }
#endif
    if(didRebalance)
    {
        this->engine->ShrinkBuffers();
    }

    size_t initialParticlesNum = reuseDeviceCensus
#ifdef STORM_WITH_GPU
                                     ? this->gpuTransportExecutor->PendingCensusCount()
#else
                                     ? 0
#endif
                                     : this->ownedParticles.size();
    this->initialParticleCount = initialParticlesNum;
    this->cellsParticleCounters.assign(this->nCells, 0);
    if(reuseDeviceCensus)
    {
#ifdef STORM_WITH_GPU
        this->gpuTransportExecutor->CopyPendingCensusCellCounts(this->nCells, this->cellsParticleCounters);
#endif
    }
    else
    {
        for(const MCParticle &p : this->ownedParticles)
        {
            this->cellsParticleCounters[p.cellIndex]++;
        }
    }
#ifdef STORM_WITH_GPU
    if(reuseDeviceCensus)
    {
        this->gpuTransportExecutor->PromotePendingCensus(fullDt);
        this->deviceCensusValid = false;
        this->ownedParticles.clear();
        this->hostParticlesValid = false;
    }
    else if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(this->physics->UsesDeviceTransport())
        {
            this->StageLocalParticlesForDevice(std::move(this->ownedParticles), false);
        }
        else
        {
            this->PutSelfParticles(std::move(this->ownedParticles));
        }
    }
    else
    {
        this->PutSelfParticles(std::move(this->ownedParticles));
    }
#else
    this->PutSelfParticles(std::move(this->ownedParticles));
#endif
    this->hostParticlesValid = false;
    this->ClearParticlesChanged();
    this->physics->updateGridData();

    std::chrono::high_resolution_clock::time_point generationStart = std::chrono::high_resolution_clock::now();
    std::vector<MCParticle> newParticles1;
    std::size_t deviceEmitted = 0;
    {
        STORM_PROFILE_REGION("storm/generation");
#ifdef STORM_WITH_GPU
        if constexpr(gpu::HasDeviceSourceGeneration<Physics>::value)
        {
            if(this->physics->SupportsDeviceSourceGeneration())
            {
                gpu::DeviceSourceContext context;
                context.executor = this->gpuTransportExecutor.get();
                context.executorStorage = &this->gpuTransportExecutor;
                context.gpuMaxInnerSteps = this->config.gpuMaxInnerSteps;
                context.gpuOverlapCommunication =
                    this->config.gpuOverlapCommunication && this->sizeWorld > 1;
                context.firstParticleId =
                    static_cast<particle_id_t>(this->myIDCounter);
                context.rank = this->rankWorld;
                context.fullDt = fullDt;
                newParticles1 = this->physics->preStepOnDevice(context);
                deviceEmitted = context.emittedCount;
                this->myIDCounter += deviceEmitted;
            }
            else
            {
                newParticles1 = this->physics->preStep(fullDt);
            }
        }
        else
        {
            newParticles1 = this->physics->preStep(fullDt);
        }
#else
        newParticles1 = this->physics->preStep(fullDt);
#endif
    }
    double generationSeconds = std::chrono::duration<double>(
                                   std::chrono::high_resolution_clock::now() - generationStart)
                                   .count();

    size_t preStepParticlesNum = deviceEmitted + newParticles1.size();
    this->preStepParticleCount = preStepParticlesNum;
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceSourceGeneration<Physics>::value)
    {
        if(deviceEmitted > 0)
        {
            const std::vector<std::size_t> &nPhotons = this->physics->getLastSourcePhotonsPerCell();
            const std::size_t n = (nPhotons.size() < this->cellsParticleCounters.size()) ? nPhotons.size() : this->cellsParticleCounters.size();
            for(std::size_t i = 0; i < n; ++i)
            {
                this->cellsParticleCounters[i] += nPhotons[i];
            }
        }
    }
#endif
    for(const MCParticle &p : newParticles1)
    {
        this->cellsParticleCounters[p.cellIndex]++;
    }
    this->startParticleCount = initialParticlesNum + preStepParticlesNum;
    this->beginningParticleCount = this->cellsParticleCounters;

    unsigned long long globalInitialParticles = static_cast<unsigned long long>(this->initialParticleCount);
    unsigned long long globalPreStepParticles = static_cast<unsigned long long>(this->preStepParticleCount);
    unsigned long long globalStartParticles = static_cast<unsigned long long>(this->startParticleCount);
    const unsigned long long localStartParticles = globalStartParticles;
    unsigned long long maxStartParticles = 0;
    this->engine->Reduce(&localStartParticles, &maxStartParticles, 1, Reduction::Max, true);
    int maxStartRankCandidate = (localStartParticles == maxStartParticles) ? static_cast<int>(this->rankWorld) : std::numeric_limits<int>::max();
    int maxStartRank = 0;
    this->engine->Reduce(&maxStartRankCandidate, &maxStartRank, 1, Reduction::Min, true);
    this->engine->Reduce(&globalInitialParticles, &globalInitialParticles, 1, Reduction::Sum);
    this->engine->Reduce(&globalPreStepParticles, &globalPreStepParticles, 1, Reduction::Sum);
    this->engine->Reduce(&globalStartParticles, &globalStartParticles, 1, Reduction::Sum);
    if(this->rankWorld == 0)
    {
        const double averageStartParticles = static_cast<double>(globalStartParticles) / this->sizeWorld;
        const double maxToAverage = (averageStartParticles > 0) ? static_cast<double>(maxStartParticles) / averageStartParticles : 0.0;
        const double maxRawPayloadMiB = static_cast<double>(maxStartParticles) * sizeof(MCParticle) / (1 << 20);
        std::cout << "MC particle counts before transport:"
                  << " initial=" << globalInitialParticles
                  << " prestep_generated=" << globalPreStepParticles
                  << " active_after_prestep=" << globalStartParticles
                  << std::endl;
        std::cout << "MC particle distribution after generation:"
                  << " max=" << maxStartParticles
                  << " (rank " << maxStartRank << ")"
                  << " avg=" << averageStartParticles
                  << " max/avg=" << maxToAverage
                  << " particle_size=" << sizeof(MCParticle) << " B"
                  << " max_raw_payload=" << maxRawPayloadMiB << " MiB"
                  << std::endl;
    }

    this->resetTracker();
    this->currentStep++;
    this->iteration = 0;
    this->allStepsCounter = 0;
    this->dynamicallyAdded = 0;
    // this->neighbors = this->grid.GetDuplicatedProcs();
    this->cellsStepsCounters.assign(this->nCells, 0);
    this->transfersCounter = 0;

    auto initializeParticle = [fullDt](MCParticle &particle)
    {
#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
        particle.checkedHere = true;
        particle.nextRank = std::numeric_limits<rank_t>::max();
        particle.removedFromRank = false;
        particle.sentByRank = std::numeric_limits<rank_t>::max();
        particle.lastSeen = 0;
        particle.lastSeenRank = std::numeric_limits<rank_t>::max();
        particle.lastSeenRankBuf = std::numeric_limits<rank_t>::max();
        particle.lastSeenIndex = std::numeric_limits<size_t>::max();
#endif
#ifdef STORM_WITH_TRACING_HISTORY
        particle.tracingHistoryIndex = 0;
        particle.tracingHistoryCount = 0;
#endif
        particle.timeLeft = fullDt;
        particle.initialWeight = std::abs(particle.weight);
        particle.steps = 0;
    };

    this->engine->VisitLocal(initializeParticle);
    for(std::vector<MCParticle> &particles : this->detachedRankParticles)
    {
        for(MCParticle &particle : particles)
        {
            initializeParticle(particle);
        }
    }
    {
#ifdef STORM_WITH_GPU
        if constexpr(gpu::HasDeviceTransport<Physics>::value)
        {
            if(this->physics->UsesDeviceTransport() and this->config.gpuDevicePoolHeadroomFactor > 0.0)
            {
                if(!this->gpuTransportExecutor)
                {
                    this->gpuTransportExecutor = std::make_unique<gpu::KokkosLocalTransportExecutor>(
                        this->config.gpuMaxInnerSteps,
                        this->config.gpuOverlapCommunication && this->sizeWorld > 1);
                }
                const std::size_t peakEstimate = std::max(this->startParticleCount, this->gpuLastStepMaxActive);
                const std::size_t activeTarget = std::max(this->config.gpuDevicePoolMinCapacity,
                                                          static_cast<std::size_t>(std::ceil(static_cast<double>(peakEstimate) * this->config.gpuDevicePoolHeadroomFactor)));
                std::size_t hostIngestTarget = this->config.gpuHostIngestCapacity;
                if(hostIngestTarget == 0)
                {
                    hostIngestTarget = std::max<std::size_t>(262144, this->config.localTransportBatchSize);
                }
                this->gpuTransportExecutor->ReservePoolCapacity(activeTarget, hostIngestTarget);
            }
            if(this->physics->UsesDeviceTransport())
            {
                this->StageLocalParticlesForDevice(std::move(newParticles1), true);
            }
            else
            {
                this->AddParticles(newParticles1);
            }
        }
        else
        {
            this->AddParticles(newParticles1);
        }
#else
        this->AddParticles(newParticles1);
#endif
        std::vector<MCParticle>().swap(newParticles1);
    }
    this->engine->Barrier();

    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;
    this->engine->ResetCounterSnapshots();
#ifdef STORM_WITH_MPI
    this->amountManager.reset();
    if(this->commWorld != MPI_COMM_NULL)
    {
        this->amountManager = std::make_unique<AmountManager>(this->commWorld);
        this->amountManager->Initialize(startingParticleNum);
    }
#endif
    this->completionRemaining = startingParticleNum;
    this->completionDone = false;

    MonteCarloStepFinalData data;
    this->handlerMemoryBytes = this->engine->MemoryBytes();

    this->PrintMemoryDiagnostics(initialParticlesNum, preStepParticlesNum);

    MEMORY_DEBUG_PRINT("Before main loop in MCM");

    const size_t amountProgressMinCycles = std::max<size_t>(1, this->config.amountProgressMinCycles);
    std::chrono::high_resolution_clock::time_point loopStart = std::chrono::high_resolution_clock::now();
    double setupSeconds = std::chrono::duration<double>(loopStart - stepStart).count() - generationSeconds;
    this->progressStartTime = loopStart;
    this->lastProgressPrintTime = 0.0;
    int64_t globalInitialForProgress = this->completionRemaining;
#ifdef STORM_WITH_MPI
    if(this->amountManager)
    {
        globalInitialForProgress = this->amountManager->GetValue();
    }
#endif
    this->progressStartParticles = globalInitialForProgress;
    this->progressRemovedCount = 0;

    auto buildProgressCounters = [this]()
    {
        std::array<unsigned long long, MC_PROGRESS_COUNTERS> counters{};
        counters[MC_PROGRESS_RW_STEPS] = static_cast<unsigned long long>(this->physics->getRandomWalkStepCount());
        counters[MC_PROGRESS_DDMC_STEPS] = static_cast<unsigned long long>(this->physics->getDDMCStepCount());
        counters[MC_PROGRESS_DDMC_LEAKS] = static_cast<unsigned long long>(this->physics->getDDMCLeakCount());
        counters[MC_PROGRESS_DDMC_CENSUS] = static_cast<unsigned long long>(this->physics->getDDMCCensusCount());
        counters[MC_PROGRESS_DDMC_UPSCATTER] = static_cast<unsigned long long>(this->physics->getDDMCUpscatterCount());
        counters[MC_PROGRESS_DDMC_FALLBACK] = static_cast<unsigned long long>(this->physics->getDDMCFallbackCount());
        return counters;
    };

    {
        STORM_PROFILE_REGION("storm/loop");
        try
        {
            while(
#ifdef STORM_WITH_MPI
                this->amountManager ? not this->amountManager->GetDoneRef() : not this->completionDone
#else
                not this->completionDone
#endif
            )
            {
                ++this->loopRounds;
                std::chrono::steady_clock::time_point phaseStart = std::chrono::steady_clock::now();

                this->engine->Progress();
                std::chrono::steady_clock::time_point handleStart = std::chrono::steady_clock::now();
                this->loopCommunicationSeconds += std::chrono::duration<double>(handleStart - phaseStart).count();

                // HandleAll returns true when this rank found nothing to transport.
                bool localWorkDone = this->HandleAll(data);
                phaseStart = std::chrono::steady_clock::now();
                this->loopHandleSeconds += std::chrono::duration<double>(phaseStart - handleStart).count();
                if(localWorkDone)
                {
                    ++this->loopIdleRounds;
                }

                this->engine->Poll();
                this->engine->Flush(localWorkDone);
                std::chrono::steady_clock::time_point amountStart = std::chrono::steady_clock::now();
                this->loopCommunicationSeconds += std::chrono::duration<double>(amountStart - phaseStart).count();

                this->completionRemaining -= this->localDecrementAmount;
#ifdef STORM_WITH_MPI
                if(this->amountManager)
                {
                    this->amountManager->Decrease(this->localDecrementAmount);
                }
#endif
                this->localDecrementAmount = 0;

                if(this->iteration % amountProgressMinCycles == 0)
                {
#ifdef STORM_WITH_MPI
                    if(this->amountManager)
                    {
                        this->amountManager->Progress();
                    }
#endif
                    this->loopAmountSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - amountStart).count();

                    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
                    double elapsedSeconds = std::chrono::duration<double>(now - this->progressStartTime).count();

                    std::array<unsigned long long, MC_PROGRESS_COUNTERS> localCounters = buildProgressCounters();
                    this->engine->PublishCounters(localCounters.data(), localCounters.size(), elapsedSeconds);

                    if(this->rankWorld == 0 && elapsedSeconds - this->lastProgressPrintTime >= 10.0)
                    {
                        this->lastProgressPrintTime = elapsedSeconds;
                        std::vector<unsigned long long> globalCounters = this->engine->CounterTotals();
                        int64_t globalRemaining = this->completionRemaining;
#ifdef STORM_WITH_MPI
                        if(this->amountManager)
                        {
                            globalRemaining = this->amountManager->GetValue();
                        }
#endif
                        int64_t globalDone = globalInitialForProgress - globalRemaining;
                        double doneFraction = (globalInitialForProgress > 0) ? static_cast<double>(globalDone) / static_cast<double>(globalInitialForProgress) : 0.0;
                        double rate = (elapsedSeconds > 0) ? static_cast<double>(globalDone) / elapsedSeconds : 0.0;
                        double eta = (rate > 0) ? static_cast<double>(globalRemaining) / rate : 0.0;
                        size_t localRemaining = this->engine->LocalSize(this->rankWorld);
                        std::cerr << "[Progress] ~"
                                  << (doneFraction * 100.0) << "% done, "
                                  << elapsedSeconds << "s elapsed, "
                                  << "~" << eta << "s ETA, "
                                  << "global_done=" << globalDone << "/" << globalInitialForProgress
                                  << " rank0_local_remaining=" << localRemaining
                                  << " rw_steps_total=" << globalCounters[MC_PROGRESS_RW_STEPS]
                                  << " ddmc_steps_total=" << globalCounters[MC_PROGRESS_DDMC_STEPS]
                                  << " ddmc_leaks=" << globalCounters[MC_PROGRESS_DDMC_LEAKS]
                                  << " ddmc_census=" << globalCounters[MC_PROGRESS_DDMC_CENSUS]
                                  << " ddmc_upscatter=" << globalCounters[MC_PROGRESS_DDMC_UPSCATTER]
                                  << " ddmc_fallback=" << globalCounters[MC_PROGRESS_DDMC_FALLBACK]
                                  << " eta_is_count_based=1"
                                  << std::endl;
                    }
                }

                bool needsCompletionVerification = this->completionRemaining == 0;
#ifdef STORM_WITH_MPI
                if(this->amountManager)
                {
                    needsCompletionVerification = this->amountManager->GetVerifyRef();
                }
#endif
                if(needsCompletionVerification)
                {
                    this->engine->FlushAll();
                    this->engine->Progress();
                    const bool idle = localWorkDone && !this->engine->Pending();
#ifdef STORM_WITH_MPI
                    if(this->amountManager)
                    {
                        this->amountManager->Verify(idle);
                    }
                    else
#endif
                    {
                        this->completionDone = idle && this->completionRemaining == 0;
                    }
                }

                this->iteration++;
            }
        }
        catch(const STORMError &eo)
        {
            reportError(eo);
            throw;
        }
    }

    this->engine->FinishCounters();

    std::chrono::high_resolution_clock::time_point loopEnd = std::chrono::high_resolution_clock::now();
    double loopTime = std::chrono::duration_cast<std::chrono::duration<double>>(loopEnd - loopStart).count();
    double deviceCensusDrainSeconds = 0.0;
    bool usedDeviceCensusPostStep = false;
    bool usedDevicePopulationControl = false;
    bool retainDeviceCensus = false;
#ifdef STORM_WITH_GPU
    {
        STORM_PROFILE_REGION("storm/census/device_post_step");
        const std::chrono::high_resolution_clock::time_point deviceCensusStart =
            std::chrono::high_resolution_clock::now();
        if constexpr(gpu::HasDeviceCensusPostStep<Physics>::value)
        {
            bool canUseDevicePostStep =
                this->physics->SupportsDeviceCensusPostStep() &&
                data.remaining.empty();
#ifdef STORM_WITH_MPI
            // Device activation may run collectives, and data.remaining is a
            // per-rank quantity, so the whole communicator has to agree before
            // any rank enters activateDevice.
            int deviceCensusVote = canUseDevicePostStep ? 1 : 0;
            this->engine->Reduce(&deviceCensusVote, &deviceCensusVote, 1, Reduction::Min, true);
            canUseDevicePostStep = (deviceCensusVote != 0);
#endif
            if(canUseDevicePostStep)
            {
                if(this->populationControl->SupportsDeviceActivation())
                {
                    gpu::DevicePopulationContext context;
                    context.executor = this->gpuTransportExecutor.get();
                    context.cellCount = this->nCells;
                    context.activationEpoch =
                        this->populationActivationEpoch;
                    context.rank = this->rankWorld;
#ifdef STORM_WITH_MPI
                    context.communicator = this->commWorld;
#endif
                    this->populationControl->activateDevice(context);
                    ++this->populationActivationEpoch;
                    usedDevicePopulationControl = true;
                }
                if(this->populationControl->IsIdentity() ||
                   usedDevicePopulationControl)
                {
                    this->physics->postStepWithDeviceCensus(
                        data.remaining, fullDt);
                    usedDeviceCensusPostStep = true;
                }
            }
        }
        retainDeviceCensus = usedDeviceCensusPostStep and data.remaining.empty();
        if(not retainDeviceCensus and not usedDevicePopulationControl)
        {
            this->DrainDeviceCensus(data);
        }
        deviceCensusDrainSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - deviceCensusStart).count();
    }
#endif
    double localStepCount = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        localStepCount += static_cast<double>(counter);
    }
#ifdef STORM_WITH_GPU
    localStepCount += static_cast<double>(this->gpuPhysicsStepCount);
#endif
    double avgSteps = localStepCount;
    this->engine->Reduce(&avgSteps, &avgSteps, 1, Reduction::Sum);
    avgSteps /= this->sizeWorld;
    double maxSteps = localStepCount;
    this->engine->Reduce(&maxSteps, &maxSteps, 1, Reduction::Max);
    if(this->rankWorld == 0)
    {
        std::cout << "Loop time: " << loopTime << " seconds, max steps: " << maxSteps << ", avg steps: " << avgSteps << std::endl;
    }

#ifdef STORM_WITH_GPU
    double localGpuTimes[5] = {
        this->gpuPackSeconds,
        this->gpuDeviceSeconds,
        this->gpuCopyBackSeconds,
        this->gpuProgressSeconds,
        this->gpuHostEventSeconds};
    double maximumGpuTimes[5] = {};
    this->engine->Reduce(localGpuTimes, maximumGpuTimes, 5, Reduction::Max);
    gpu::TransportExecutorMetrics executorMetrics;
    if(this->gpuTransportExecutor)
    {
        executorMetrics = this->gpuTransportExecutor->Metrics();
        this->gpuLastStepMaxActive = executorMetrics.maxActiveCount;
    }
    unsigned long long localGpuCounts[17] = {
        this->gpuLaunchCount,
        this->gpuParticleCount,
        this->gpuIngestCount,
        this->gpuHoldCount,
        static_cast<unsigned long long>(executorMetrics.h2dBytes),
        static_cast<unsigned long long>(executorMetrics.d2hBytes + this->gpuDeferredD2HBytes),
        static_cast<unsigned long long>(executorMetrics.eliminatedHostCopyBytes),
        static_cast<unsigned long long>(executorMetrics.reallocationCount),
        static_cast<unsigned long long>(executorMetrics.synchronizationCount),
        static_cast<unsigned long long>(executorMetrics.terminalCount),
        static_cast<unsigned long long>(executorMetrics.fallbackCount),
        static_cast<unsigned long long>(executorMetrics.remoteCount),
        static_cast<unsigned long long>(executorMetrics.censusCopyCount),
        static_cast<unsigned long long>(executorMetrics.splitCreatedCount),
        this->gpuElidedRemovalCount,
        static_cast<unsigned long long>(executorMetrics.progressPollCount),
        static_cast<unsigned long long>(executorMetrics.pipelinedRemoteCount)};
    unsigned long long globalGpuCounts[17] = {};
    this->engine->Reduce(localGpuCounts, globalGpuCounts, 17, Reduction::Sum);
    this->gpuDeferredD2HBytes = 0;
    const unsigned long long globalGpuLaunches = globalGpuCounts[0];
    const unsigned long long globalGpuParticles = globalGpuCounts[1];
    const unsigned long long globalGpuIngest = globalGpuCounts[2];
    const unsigned long long globalGpuHolds = globalGpuCounts[3];
    if(this->rankWorld == 0 && globalGpuParticles > 0)
    {
        const double particlesPerLaunch = static_cast<double>(globalGpuParticles) / static_cast<double>(globalGpuLaunches);
        const double packPerLaunch = (globalGpuLaunches > 0) ? static_cast<double>(globalGpuIngest) / static_cast<double>(globalGpuLaunches) : 0.0;
        std::cout << "GPU transport max-rank time: pack=" << maximumGpuTimes[0]
                  << " s, device+compact=" << maximumGpuTimes[1]
                  << " s, compact-copy=" << maximumGpuTimes[2]
                  << " s, overlapped-rma=" << maximumGpuTimes[3]
                  << " s, host-events=" << maximumGpuTimes[4]
                  << " s; global launches=" << globalGpuLaunches
                  << ", launched=" << globalGpuParticles
                  << " (" << particlesPerLaunch << "/launch)"
                  << ", packed=" << globalGpuIngest
                  << " (" << packPerLaunch << "/launch)"
                  << ", holds=" << globalGpuHolds << std::endl;
    }
    unsigned long long localStagingMaxima[2] = {
        static_cast<unsigned long long>(executorMetrics.maxIngestCount),
        static_cast<unsigned long long>(executorMetrics.maxActiveCount)};
    unsigned long long globalStagingMaxima[2] = {};
    this->engine->Reduce(localStagingMaxima, globalStagingMaxima, 2, Reduction::Max);
    if(this->rankWorld == 0 && globalGpuParticles > 0)
    {
        std::cout << "GPU staging totals: h2d_bytes="
                  << globalGpuCounts[4]
                  << " d2h_bytes=" << globalGpuCounts[5]
                  << " host_copy_bytes_eliminated="
                  << globalGpuCounts[6]
                  << " reallocations=" << globalGpuCounts[7]
                  << " synchronizations=" << globalGpuCounts[8]
                  << " terminals=" << globalGpuCounts[9]
                  << " fallbacks=" << globalGpuCounts[10]
                  << " remotes=" << globalGpuCounts[11]
                  << " census=" << globalGpuCounts[12]
                  << " splits_created=" << globalGpuCounts[13]
                  << " removal_unpacks_elided=" << globalGpuCounts[14]
                  << " progress_polls=" << globalGpuCounts[15]
                  << " pipelined_remotes=" << globalGpuCounts[16]
                  << " max_ingest=" << globalStagingMaxima[0]
                  << " max_active=" << globalStagingMaxima[1]
                  << std::endl;
    }
#endif

    {
        double localLoopTimes[4] = {
            this->loopCommunicationSeconds,
            this->loopAmountSeconds,
            this->loopHandleSeconds,
            this->loopMergeSeconds};
        double maximumLoopTimes[4] = {};
        double minimumLoopTimes[4] = {};
        this->engine->Reduce(localLoopTimes, maximumLoopTimes, 4, Reduction::Max);
        this->engine->Reduce(localLoopTimes, minimumLoopTimes, 4, Reduction::Min);
        unsigned long long roundCounts[2] = {this->loopRounds, this->loopIdleRounds};
        unsigned long long maximumRounds[2] = {};
        unsigned long long totalRounds[2] = {};
        this->engine->Reduce(roundCounts, maximumRounds, 2, Reduction::Max);
        this->engine->Reduce(roundCounts, totalRounds, 2, Reduction::Sum);
        if(this->rankWorld == 0)
        {
            std::cout << "MC loop split max-rank: rma=" << maximumLoopTimes[0]
                      << " s, amount=" << maximumLoopTimes[1]
                      << " s, handle=" << maximumLoopTimes[2]
                      << " s, merge=" << maximumLoopTimes[3]
                      << " s; min-rank: rma=" << minimumLoopTimes[0]
                      << " s, amount=" << minimumLoopTimes[1]
                      << " s, handle=" << minimumLoopTimes[2]
                      << " s; rounds max=" << maximumRounds[0]
                      << " idle_max=" << maximumRounds[1]
                      << " rounds_sum=" << totalRounds[0]
                      << " idle_sum=" << totalRounds[1] << std::endl;
        }
    }

    std::chrono::high_resolution_clock::time_point censusStart = std::chrono::high_resolution_clock::now();
    {
        STORM_PROFILE_REGION("storm/census");
        if(retainDeviceCensus)
        {
            this->ownedParticles.clear();
            this->hostParticlesValid = false;
            this->deviceCensusValid = true;
#ifdef STORM_WITH_GPU
            const std::size_t assigned = this->gpuTransportExecutor->AssignPendingCensusIdentities(this->rankWorld, static_cast<particle_id_t>(this->myIDCounter));
            this->myIDCounter += assigned;
#endif
        }
        else
        {
            if(usedDeviceCensusPostStep and (this->populationControl->IsIdentity() or usedDevicePopulationControl))
            {
                this->ownedParticles = std::move(data.remaining);
            }
            else
            {
                this->ownedParticles = this->populationControl->activate(data.remaining);
            }
            this->hostParticlesValid = true;
            this->deviceCensusValid = false;
            if(not usedDeviceCensusPostStep)
            {
                this->physics->postStep(this->ownedParticles, fullDt);
            }
        }
    }
    double censusSeconds = deviceCensusDrainSeconds +
                           std::chrono::duration<double>(
                               std::chrono::high_resolution_clock::now() - censusStart)
                               .count();

    double localStepTimes[4] = {setupSeconds, generationSeconds, loopTime, censusSeconds};
    double maximumStepTimes[4] = {};
    this->engine->Reduce(localStepTimes, maximumStepTimes, 4, Reduction::Max);
    if(this->rankWorld == 0)
    {
        std::cout << "MC step max-rank time: setup=" << maximumStepTimes[0]
                  << " s, generation=" << maximumStepTimes[1]
                  << " s, loop=" << maximumStepTimes[2]
                  << " s, census=" << maximumStepTimes[3]
                  << " s" << std::endl;
    }

    size_t newParticlesNum = this->ownedParticles.size();
#ifdef STORM_WITH_GPU
    if(retainDeviceCensus and this->gpuTransportExecutor)
    {
        newParticlesNum += this->gpuTransportExecutor->PendingCensusCount();
    }
#endif
    this->endParticleCount = newParticlesNum;

    if(this->engine->Pending())
    {
        throw STORMError("End of MonteCarloManager::step: communication is pending");
    }
#ifdef STORM_WITH_MPI
    this->amountManager.reset();
#endif
    for(rank_t rank = 0; rank < static_cast<rank_t>(this->detachedRankParticles.size()); rank++)
    {
        const std::vector<MCParticle> &particles = this->detachedRankParticles[static_cast<size_t>(rank)];
        if(not particles.empty())
        {
            STORMError eo("End of MonteCarloManager::step: detached particle list is not empty");
            eo.addEntry("Rank", this->rankWorld);
            eo.addEntry("Peer Rank", rank);
            eo.addEntry("Detached Particles", particles.size());
            throw eo;
        }
    }
#ifdef STORM_WITH_GPU
    if(this->gpuTransportExecutor and
       (this->gpuTransportExecutor->DeviceBusy() or
        (this->gpuTransportExecutor->PendingCensusCount() > 0 &&
         !this->deviceCensusValid)))
    {
        STORMError eo("End of MonteCarloManager::step: device particle pool is not empty");
        eo.addEntry("Rank", this->rankWorld);
        eo.addEntry("Device particles", this->gpuTransportExecutor->ActiveCount());
        eo.addEntry("Pending remote packets", this->gpuTransportExecutor->PendingRemoteCount());
        eo.addEntry("Pending census packets", this->gpuTransportExecutor->PendingCensusCount());
        throw eo;
    }
#endif

    if(not didRebalance)
    {
        if(this->currentStep > 0 and this->config.shrinkBuffersCycle > 0 and this->currentStep % this->config.shrinkBuffersCycle == 0)
        {
            this->engine->ShrinkBuffers();
        }
    }

    this->ClearParticlesChanged();
}

#endif // STORM_MONTE_CARLO_LIFECYCLE_HPP
