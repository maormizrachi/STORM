#ifndef STORM_RDMA_STEP_LIFECYCLE_HPP
#define STORM_RDMA_STEP_LIFECYCLE_HPP

template<typename T, typename Grid, typename Physics>
std::vector<typename RDMAMonteCarloManager<T, Grid, Physics>::MCParticle> RDMAMonteCarloManager<T, Grid, Physics>::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    // if(this->Ncells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rank_world << ": " << this->Ncells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }

    auto stepStart = std::chrono::high_resolution_clock::now();

    this->Ncells = this->grid.GetPointNo();
    this->ranks_ghost_map = GetGhostMap(this->grid);
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();

    this->PrepareHandlers();

    this->ResetSendBuffers();
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;
#ifdef STORM_WITH_GPU
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
    this->gpuHoldSkips = 0;
    if(this->gpuTransportExecutor)
    {
        this->gpuTransportExecutor->Reset();
    }
#endif
    this->loopRmaSeconds = 0.0;
    this->loopAmountSeconds = 0.0;
    this->loopHandleSeconds = 0.0;
    this->loopMergeSeconds = 0.0;
    this->loopRounds = 0;
    this->loopIdleRounds = 0;

    bool didRebalance = this->grid.DidRebalance() and (this->lastBuildGeneration != this->grid.GetBuildGeneration());
    if(didRebalance)
    {
        this->ShrinkBuffers();
    }
    this->lastBuildGeneration = this->grid.GetBuildGeneration();

    size_t initialParticlesNum = particleList.size();
    this->initialParticleCount = initialParticlesNum;
    this->cellsParticleCounters.assign(this->Ncells, 0);
    for(const auto &p : particleList)
    {
        this->cellsParticleCounters[p.cellIndex]++;
    }
#ifdef STORM_WITH_GPU
    if constexpr(gpu::HasDeviceTransport<Physics>::value)
    {
        if(this->physics->UsesDeviceTransport())
        {
            this->StageLocalParticlesForDevice(std::move(particleList), false);
        }
        else
        {
            this->PutSelfParticles(std::move(particleList));
        }
    }
    else
    {
        this->PutSelfParticles(std::move(particleList));
    }
#else
    this->PutSelfParticles(std::move(particleList));
#endif
    this->physics->updateGridData();

    auto generationStart = std::chrono::high_resolution_clock::now();
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
    double generationSeconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - generationStart).count();

    size_t preStepParticlesNum = newParticles1.size();
    this->preStepParticleCount = preStepParticlesNum;
    for(const auto &p : newParticles1)
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
    MPI_Allreduce(&localStartParticles, &maxStartParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm_world);
    int maxStartRankCandidate = (localStartParticles == maxStartParticles)? static_cast<int>(this->rank_world) : std::numeric_limits<int>::max();
    int maxStartRank = 0;
    MPI_Allreduce(&maxStartRankCandidate, &maxStartRank, 1, MPI_INT, MPI_MIN, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalInitialParticles, &globalInitialParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalPreStepParticles, &globalPreStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalStartParticles, &globalStartParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        const double averageStartParticles = static_cast<double>(globalStartParticles) / this->size_world;
        const double maxToAverage = (averageStartParticles > 0)? static_cast<double>(maxStartParticles) / averageStartParticles : 0.0;
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
    this->cellsStepsCounters.assign(this->Ncells, 0);
    this->transfersCounter = 0;

    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }

        handler->ForEachLocalParticle([fullDt](MCParticle &particle, size_t)
        {
            MonteCarloParticleInitializer::Initialize(particle, fullDt);
        });
    }
    for(std::vector<MCParticle> &particles : this->detachedRankParticles)
    {
        MonteCarloParticleInitializer::Initialize(particles, fullDt);
    }
    {
#ifdef STORM_WITH_GPU
        if constexpr(gpu::HasDeviceTransport<Physics>::value)
        {
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
    MPI_Barrier(this->comm_world);

    size_t numParticles = initialParticlesNum + preStepParticlesNum;

    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;
    AmountManager amountManager(this->comm_world);
    amountManager.Initialize(startingParticleNum);

    MonteCarloStepFinalData data;
    size_t numOfCounterDecrementations = 0;

    {
        const size_t bytesPerSlot = sizeof(MCParticle);
        this->handlerMemoryBytes = 0;
        for(const RankHandler_t *h : this->rankHandlers)
        {
            if(h != nullptr)
            {
                this->handlerMemoryBytes += h->buffsize * bytesPerSlot;
            }
        }
    }

    this->PrintMemoryDiagnostics(initialParticlesNum, preStepParticlesNum);

    const bool &verify = amountManager.GetVerifyRef();
    const bool &done = amountManager.GetDoneRef();

    MEMORY_DEBUG_PRINT("Before main loop in MCM");

    const size_t amountProgressMinCycles = std::max<size_t>(1, this->config.amountProgressMinCycles);
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    const size_t reallocationProgressMinCycles = usesAsyncReallocation
        ? std::max<size_t>(1, this->config.asyncReallocationProgressMinCycles)
        : 1;
    auto loopStart = std::chrono::high_resolution_clock::now();
    double setupSeconds = std::chrono::duration<double>(loopStart - stepStart).count()
        - generationSeconds;
    this->progressStartTime_ = loopStart;
    this->lastProgressPrintTime_ = 0.0;
    int64_t globalInitialForProgress = amountManager.GetValue();
    this->progressStartParticles_ = globalInitialForProgress;
    this->progressRemovedCount_ = 0;
#ifdef STORM_WITH_MPI
    std::vector<std::array<unsigned long long, MC_PROGRESS_COUNTERS>> progressCountersByRank(this->size_world);
    MPI_Request progressReportSendReq = MPI_REQUEST_NULL;
    std::array<unsigned long long, MC_PROGRESS_COUNTERS> progressReportSendValue{};
    double progressLastReportSendTime = 0.0;
#endif

    auto buildProgressCounters = [this]()
    {
        std::array<unsigned long long, MC_PROGRESS_COUNTERS> counters{};
        counters[MC_PROGRESS_RW_STEPS] =
            static_cast<unsigned long long>(this->physics->getRandomWalkStepCount());
        counters[MC_PROGRESS_DDMC_STEPS] =
            static_cast<unsigned long long>(this->physics->getDDMCStepCount());
        counters[MC_PROGRESS_DDMC_LEAKS] =
            static_cast<unsigned long long>(this->physics->getDDMCLeakCount());
        counters[MC_PROGRESS_DDMC_CENSUS] =
            static_cast<unsigned long long>(this->physics->getDDMCCensusCount());
        counters[MC_PROGRESS_DDMC_UPSCATTER] =
            static_cast<unsigned long long>(this->physics->getDDMCUpscatterCount());
        counters[MC_PROGRESS_DDMC_FALLBACK] =
            static_cast<unsigned long long>(this->physics->getDDMCFallbackCount());
        return counters;
    };

    try
    {
        while(not done)
        {
            ++this->loopRounds;
            auto phaseStart = std::chrono::steady_clock::now();

            this->PumpRMAProgress();
            bool shouldProgressReallocations = (not usesAsyncReallocation) or
                (this->iteration % reallocationProgressMinCycles == 0) or
                this->reallocationAgent->HasPendingAsyncReallocations();
            if(shouldProgressReallocations)
            {
                this->ProgressReallocations();
            }
            this->MakeRDMAProgress();
            auto handleStart = std::chrono::steady_clock::now();
            this->loopRmaSeconds += std::chrono::duration<double>(handleStart - phaseStart).count();

            // HandleAll returns true when this rank found nothing to transport.
            bool localWorkDone = this->HandleAll(data);
            phaseStart = std::chrono::steady_clock::now();
            this->loopHandleSeconds += std::chrono::duration<double>(phaseStart - handleStart).count();
            if(localWorkDone)
            {
                ++this->loopIdleRounds;
            }

            this->PumpRMAProgress();
            this->FlushSendBuffers(localWorkDone);
            auto amountStart = std::chrono::steady_clock::now();
            this->loopRmaSeconds += std::chrono::duration<double>(amountStart - phaseStart).count();

            amountManager.Decrease(this->localDecrementAmount);
            this->localDecrementAmount = 0;

            if(this->iteration % amountProgressMinCycles == 0)
            {
                amountManager.Progress();
                this->loopAmountSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - amountStart).count();

                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - this->progressStartTime_).count();

#ifdef STORM_WITH_MPI
                if(this->rank_world == 0)
                {
                    progressCountersByRank[0] = buildProgressCounters();

                    int hasMsg = 0;
                    MPI_Status status;
                    while(true)
                    {
                        MPI_Iprobe(MPI_ANY_SOURCE, RW_PROGRESS_TAG, this->comm_world, &hasMsg, &status);
                        if(!hasMsg)
                        {
                            break;
                        }
                        std::array<unsigned long long, MC_PROGRESS_COUNTERS> recvCounters{};
                        MPI_Recv(recvCounters.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, status.MPI_SOURCE,
                                 RW_PROGRESS_TAG, this->comm_world, MPI_STATUS_IGNORE);
                        progressCountersByRank[status.MPI_SOURCE] = recvCounters;
                    }
                }
                else if(elapsed_s - progressLastReportSendTime >= 5.0)
                {
                    if(progressReportSendReq != MPI_REQUEST_NULL)
                    {
                        int sendDone = 0;
                        MPI_Test(&progressReportSendReq, &sendDone, MPI_STATUS_IGNORE);
                        if(sendDone)
                        {
                            progressReportSendReq = MPI_REQUEST_NULL;
                        }
                    }
                    if(progressReportSendReq == MPI_REQUEST_NULL)
                    {
                        progressReportSendValue = buildProgressCounters();
                        MPI_Isend(progressReportSendValue.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, 0,
                                  RW_PROGRESS_TAG, this->comm_world, &progressReportSendReq);
                        progressLastReportSendTime = elapsed_s;
                    }
                }
#endif

                if(this->rank_world == 0 && elapsed_s - this->lastProgressPrintTime_ >= 10.0)
                {
                    this->lastProgressPrintTime_ = elapsed_s;
                    std::array<unsigned long long, MC_PROGRESS_COUNTERS> globalCounters{};
#ifdef STORM_WITH_MPI
                    for(const auto &counters : progressCountersByRank)
                    {
                        for(size_t i = 0; i < globalCounters.size(); ++i)
                        {
                            globalCounters[i] += counters[i];
                        }
                    }
#else
                    globalCounters = buildProgressCounters();
#endif
                    int64_t globalRemaining = amountManager.GetValue();
                    int64_t globalDone = globalInitialForProgress - globalRemaining;
                    double done_frac = (globalInitialForProgress > 0) ? static_cast<double>(globalDone) / static_cast<double>(globalInitialForProgress) : 0.0;
                    double rate = (elapsed_s > 0) ? static_cast<double>(globalDone) / elapsed_s : 0.0;
                    double eta = (rate > 0) ? static_cast<double>(globalRemaining) / rate : 0.0;
                    RankHandler_t *selfHandler = this->rankHandlers[this->rank_world];
                    int localRemaining = selfHandler ? static_cast<int>(selfHandler->LocalSize()) : 0;
                    std::cerr << "[Progress] ~"
                              << (done_frac * 100.0) << "% done, "
                              << elapsed_s << "s elapsed, "
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

            if(verify)
            {
                this->FlushAllSendBuffers();
                this->ProgressReallocations();
                bool ok = this->AllSendBuffersEmpty() and not this->reallocationAgent->HasPendingAsyncReallocations();
                amountManager.Verify(ok);
            }

            this->iteration++;
        }
    }
    catch(const STORMError &eo)
    {
        reportError(eo);
        throw;
    }

#ifdef STORM_WITH_MPI
    if(this->rank_world != 0 && progressReportSendReq != MPI_REQUEST_NULL)
    {
        MPI_Wait(&progressReportSendReq, MPI_STATUS_IGNORE);
    }
#endif

    auto loopEnd = std::chrono::high_resolution_clock::now();
    double loopTime = std::chrono::duration_cast<std::chrono::duration<double>>(loopEnd - loopStart).count();
    double localStepCount = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        localStepCount += static_cast<double>(counter);
    }
#ifdef STORM_WITH_GPU
    localStepCount += static_cast<double>(this->gpuPhysicsStepCount);
#endif
    double avgSteps = localStepCount;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &avgSteps, &avgSteps, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);
    avgSteps /= this->size_world;
    double maxSteps = localStepCount;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &maxSteps, &maxSteps, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "Loop time: " << loopTime << " seconds, max steps: " << maxSteps << ", avg steps: " << avgSteps << std::endl;
    }

#ifdef STORM_WITH_GPU
    double localGpuTimes[5] = {
        this->gpuPackSeconds,
        this->gpuDeviceSeconds,
        this->gpuCopyBackSeconds,
        this->gpuProgressSeconds,
        this->gpuHostEventSeconds
    };
    double maximumGpuTimes[5] = {};
    MPI_Reduce(localGpuTimes, maximumGpuTimes, 5, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    unsigned long long globalGpuLaunches = this->gpuLaunchCount;
    unsigned long long globalGpuParticles = this->gpuParticleCount;
    unsigned long long globalGpuIngest = this->gpuIngestCount;
    unsigned long long globalGpuHolds = this->gpuHoldCount;
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalGpuLaunches,
               &globalGpuLaunches, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalGpuParticles,
               &globalGpuParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalGpuIngest,
               &globalGpuIngest, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalGpuHolds,
               &globalGpuHolds, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0 && globalGpuParticles > 0)
    {
        const double particlesPerLaunch =
            static_cast<double>(globalGpuParticles) /
            static_cast<double>(globalGpuLaunches);
        const double packPerLaunch =
            globalGpuLaunches > 0
                ? static_cast<double>(globalGpuIngest) /
                  static_cast<double>(globalGpuLaunches)
                : 0.0;
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
#endif

    {
        double localLoopTimes[4] = {
            this->loopRmaSeconds,
            this->loopAmountSeconds,
            this->loopHandleSeconds,
            this->loopMergeSeconds
        };
        double maximumLoopTimes[4] = {};
        double minimumLoopTimes[4] = {};
        MPI_Reduce(localLoopTimes, maximumLoopTimes, 4, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(localLoopTimes, minimumLoopTimes, 4, MPI_DOUBLE, MPI_MIN, 0, this->comm_world);
        unsigned long long roundCounts[2] = {this->loopRounds, this->loopIdleRounds};
        unsigned long long maximumRounds[2] = {};
        unsigned long long totalRounds[2] = {};
        MPI_Reduce(roundCounts, maximumRounds, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
        MPI_Reduce(roundCounts, totalRounds, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        if(this->rank_world == 0)
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

    auto censusStart = std::chrono::high_resolution_clock::now();
    data.remaining = this->populationControl->activate(data.remaining);
    this->physics->postStep(data.remaining, fullDt);
    double censusSeconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - censusStart).count();

    double localStepTimes[4] = {setupSeconds, generationSeconds, loopTime, censusSeconds};
    double maximumStepTimes[4] = {};
    MPI_Reduce(localStepTimes, maximumStepTimes, 4, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "MC step max-rank time: setup=" << maximumStepTimes[0]
                  << " s, generation=" << maximumStepTimes[1]
                  << " s, loop=" << maximumStepTimes[2]
                  << " s, census=" << maximumStepTimes[3]
                  << " s" << std::endl;
    }

    size_t newParticlesNum = data.remaining.size();
    this->endParticleCount = newParticlesNum;

    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        size_t localSize = handler->LocalSize();
        if(localSize != 0)
        {
            STORMError eo("End of RDMAMonteCarloManager::step: queue is not empty");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Head", static_cast<size_t>(handler->head));
            eo.addEntry("Tail", static_cast<size_t>(handler->tail));
            eo.addEntry("Particles", localSize);
            eo.addEntry("Peer Rank", handler->peer_rank_world);
            throw eo;
        }
    }
    for(rank_t rank = 0; rank < static_cast<rank_t>(this->detachedRankParticles.size()); rank++)
    {
        const std::vector<MCParticle> &particles = this->detachedRankParticles[static_cast<size_t>(rank)];
        if(not particles.empty())
        {
            STORMError eo("End of RDMAMonteCarloManager::step: detached particle list is not empty");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("Peer Rank", rank);
            eo.addEntry("Detached Particles", particles.size());
            throw eo;
        }
    }
#ifdef STORM_WITH_GPU
    if(this->gpuTransportExecutor and this->gpuTransportExecutor->ActiveCount() > 0)
    {
        STORMError eo("End of RDMAMonteCarloManager::step: device particle pool is not empty");
        eo.addEntry("Rank", this->rank_world);
        eo.addEntry("Device particles", this->gpuTransportExecutor->ActiveCount());
        throw eo;
    }
#endif

    if(not didRebalance)
    {
        if(this->currentStep > 0 and this->config.shrinkBuffersCycle > 0 and this->currentStep % this->config.shrinkBuffersCycle == 0)
        {
            this->ShrinkBuffers();
        }
    }

    return data.remaining;
}

#endif // STORM_RDMA_STEP_LIFECYCLE_HPP
