#ifndef STORM_TWO_SIDED_MONTE_CARLO_LIFECYCLE_HPP
#define STORM_TWO_SIDED_MONTE_CARLO_LIFECYCLE_HPP

template<typename T, typename Grid>
TwoSidedMonteCarloManager<T, Grid>::TwoSidedMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MPI_Comm &comm) :
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(comm), tracker(comm)
{
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->myIDCounter = 0;
    this->currentStep = 0;
    this->cellsStepsCounters.assign(this->grid.GetPointNo(), 0);
    this->beginningParticleCount_.assign(this->grid.GetPointNo(), 0);
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::PutSelfParticles(const MCParticle *newParticles, size_t particlesNum)
{
    #ifdef STORM_DEBUG
    boost::container::flat_set<std::pair<rank_t, size_t>> particlesSet;
    for(size_t i = 0; i < particlesNum; i++)
    {
        const MCParticle &particle = newParticles[i];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            continue;
        }
        std::pair<rank_t, size_t> particleSetKey = {particle.rank, particle.id};
        if(particlesSet.find(particleSetKey) != particlesSet.end())
        {
            STORMError eo("Particle with the same ID is being added to the same rank twice");
            eo.addEntry("Particle", particle);
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("ID", particle.id);
            throw eo;
        }
        particlesSet.insert(particleSetKey);
    }
    #endif // STORM_DEBUG

    size_t oldSize = this->particles.size();
    this->particles.insert(this->particles.end(), newParticles, newParticles + particlesNum);
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particlesNum;

    for(size_t i = 0; i < particlesNum; i++)
    {
        size_t idx = oldSize + i;
        MCParticle &particle = this->particles[idx];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            particle.rank = this->rank_world;
            particle.id = firstID + i;
        }
    }
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::RemoveParticles(const std::vector<size_t> &indices)
{
    assert(not indices.empty());
    for(long long int i = indices.size() - 1; i >= 0; i--)
    {
        size_t particleIndex = indices[i];
        if(i > 0)
        {
            assert(particleIndex > indices[i - 1]);
        }
        std::swap(this->particles[particleIndex], this->particles.back());
        this->particles.pop_back();
    }
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::step(dt_t fullDt)
{
    try
    {
        this->Ncells = this->grid.GetPointNo();
        this->ranks_ghost_map = GetGhostMap(this->grid);
        std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
        this->particles.clear();

        size_t initialParticlesNum = this->ownedParticles.size();
        this->initialParticleCount_ = initialParticlesNum;
        this->PutSelfParticles(this->ownedParticles.data(), this->ownedParticles.size());
        {
            std::vector<MCParticle> empty;
            this->ownedParticles.swap(empty);
        }
        this->ClearParticlesChanged();
        this->resetTracker();
        this->currentStep++;
        this->iteration = 0;
        this->allStepsCounter = 0;
        // this->neighbors = this->grid.GetDuplicatedProcs();
        this->cellsStepsCounters.assign(this->Ncells, 0);
        MPI_Barrier(this->comm_world);

        MonteCarloParticleInitializer::Initialize(this->particles, fullDt);

        MPI_Barrier(this->comm_world);

        this->physics->updateGridData();

        std::chrono::high_resolution_clock::time_point preStepStart = std::chrono::high_resolution_clock::now();
        std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
        std::chrono::high_resolution_clock::time_point preStepEnd = std::chrono::high_resolution_clock::now();

        double preStepSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(preStepEnd - preStepStart).count(); // todo: necessary?

        this->PutSelfParticles(newParticles1.data(), newParticles1.size());
        // MPI_Barrier(this->comm_world);

        size_t numParticles = this->particles.size();
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &numParticles, &numParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

        size_t preStepParticlesNum = newParticles1.size();
        this->preStepParticleCount_ = preStepParticlesNum;
        this->startParticleCount_ = initialParticlesNum + preStepParticlesNum;
        unsigned long long globalInitialParticles = static_cast<unsigned long long>(this->initialParticleCount_);
        unsigned long long globalPreStepParticles = static_cast<unsigned long long>(this->preStepParticleCount_);
        unsigned long long globalStartParticles = static_cast<unsigned long long>(this->startParticleCount_);
        const unsigned long long localStartParticles = globalStartParticles;
        unsigned long long maxStartParticles = 0;
        MPI_Allreduce(&localStartParticles, &maxStartParticles, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm_world);
        int maxStartRankCandidate =
            (localStartParticles == maxStartParticles)
                ? static_cast<int>(this->rank_world)
                : std::numeric_limits<int>::max();
        int maxStartRank = 0;
        MPI_Allreduce(&maxStartRankCandidate, &maxStartRank, 1, MPI_INT,
                      MPI_MIN, this->comm_world);
        MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalInitialParticles, &globalInitialParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalPreStepParticles, &globalPreStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalStartParticles, &globalStartParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        if(this->rank_world == 0)
        {
            const double averageStartParticles =
                static_cast<double>(globalStartParticles) / this->size_world;
            const double maxToAverage = averageStartParticles > 0
                ? static_cast<double>(maxStartParticles) /
                      averageStartParticles
                : 0.0;
            const double maxRawPayloadMiB =
                static_cast<double>(maxStartParticles) * sizeof(MCParticle) /
                (1024.0 * 1024.0);
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

        this->beginningParticleCount_.assign(this->Ncells, 0);
        for(const auto &p : this->particles)
        {
            this->beginningParticleCount_[p.cellIndex]++;
        }

        int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;
        // std::cout << "Rank " << this->rank_world << ", startingParticleNum is " << startingParticleNum << " = " << initialParticlesNum << " + " << preStepParticlesNum << std::endl;

        this->localDecrementAmount = 0;
        AmountManager amountManager(this->comm_world);
        amountManager.Initialize(startingParticleNum);

        MonteCarloStepFinalData data;
        // measure time
        // vtune_start();
        size_t numOfCounterDecrementations = 0;
        auto start = std::chrono::high_resolution_clock::now();
        size_t lastLocalDecrementAmount;
        size_t decrementTryCounter = 0;

        size_t i = 0;
        // volatile int &verify = *amountManager.shouldVerify;
        const bool &done = amountManager.GetDoneRef();
        const bool &verify = amountManager.GetVerifyRef();

        auto receiveCallback = [this](const MCParticle *newValues, size_t newValuesCount, rank_t fromRank)
                                    {
                                        // std::cout << "Rank " << this->rank_world << " is here, got " << newValuesCount << " new particles from rank " << fromRank << "." << std::endl;
                                        this->PutSelfParticles(newValues, newValuesCount);
                                    };
        this->buffersManager = std::make_shared<BuffersManager<MCParticle>>(this->comm_world, receiveCallback, PARTICLES_TAG, RECV_BUFFER_MAX_SIZE * sizeof(MCParticle), SEND_BUFFER_DISPATCH_MIN_SIZE * sizeof(MCParticle), SEND_BUFFER_DISPATCH_MIN_CYCLES, this->size_world, this->grid.GetDuplicatedProcs());

        bool printed = false; // todo remove

        while(not done)
        {
            i++;

            if(i % 20 == 0)
            {
                this->buffersManager->HandleIncomingOutcoming();
            }

            bool isEmpty = this->HandleAll(data);

            amountManager.Decrease(static_cast<AmountManager::counter_t>(this->localDecrementAmount));
            this->localDecrementAmount = 0;

            amountManager.Progress();

            if(verify)
            {
                bool ok = this->particles.empty() and this->buffersManager->CountOutcoming() == 0;
                amountManager.Verify(ok);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        this->ownedParticles = this->populationControl->activate(data.remaining);
        this->physics->postStep(this->ownedParticles, fullDt);

        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        // std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;

        double localStepCount = 0;
        for(size_t counter : this->cellsStepsCounters)
        {
            localStepCount += static_cast<double>(counter);
        }
        double avgSteps = localStepCount;
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &avgSteps, &avgSteps, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);
        avgSteps /= this->size_world;
        double maxStepsDouble = localStepCount;
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &maxStepsDouble, &maxStepsDouble, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
        if(this->rank_world == 0)
        {
            std::cout << "Loop time: " << seconds << " seconds, max steps: " << maxStepsDouble << ", avg steps: " << avgSteps << std::endl;
        }

        size_t newParticlesNum = this->ownedParticles.size();
        this->endParticleCount_ = newParticlesNum;
        size_t leavingNumber = data.leavingCount;

        size_t totalSteps = this->allStepsCounter;
        size_t totalCounterDecrementations = numOfCounterDecrementations;
        size_t callsToTransfer = this->buffersManager->GetSentCounter();

        struct
        {
            int x;
            int rank;
        } mySteps, maxSteps, myTransfers, maxTransfers;

        mySteps.x = 0;
        for(size_t counter : this->cellsStepsCounters)
        {
            mySteps.x += static_cast<int>(counter);
        }
        mySteps.rank = this->rank_world;

        MPI_Reduce(&mySteps, &maxSteps, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);

        myTransfers.x = static_cast<int>(callsToTransfer);
        myTransfers.rank = this->rank_world;
        MPI_Reduce(&myTransfers, &maxTransfers, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);

        // std::cout << "leavingNumber = " << leavingNumber << " and newParticlesNum = " << newParticlesNum << std::endl;
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &initialParticlesNum, &initialParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepParticlesNum, &preStepParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &leavingNumber, &leavingNumber, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &newParticlesNum, &newParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalSteps, &totalSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalCounterDecrementations, &totalCounterDecrementations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &startingParticleNum, &startingParticleNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &callsToTransfer, &callsToTransfer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

        int sent = this->buffersManager->GetSentCounter();
        int recv = this->buffersManager->GetRecvCounter();

        struct
        {
            int x;
            int rank;
        } recvMax, recvRanked = {recv, this->rank_world};
        MPI_Reduce(&recvRanked, &recvMax, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &sent, &sent, 1, MPI_INT, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &recv, &recv, 1, MPI_INT, MPI_SUM, 0, this->comm_world);

        if(this->rank_world == 0)
        {
            std::cout << "Started with " << startingParticleNum << ". Came with " << initialParticlesNum << ". Generated " << preStepParticlesNum << " particles in preStep. ";
            std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << ". ";
            std::cout << "Total steps: " << totalSteps << ", total counter decrementations: " << totalCounterDecrementations << std::endl;
            std::cout << "Total send communications: " << sent << ", total receive communications: " << recv << " (max: " << recvMax.x << " in rank " << recvMax.rank << ")" << std::endl;
            std::cout << "Max steps: " << maxSteps.x << " on rank " << maxSteps.rank << ", average is " << totalSteps / this->size_world << std::endl;
            std::cout << "Max calls to transfer: " << maxTransfers.x << " on rank " << maxTransfers.rank << ", average is " << callsToTransfer / this->size_world << std::endl;
            assert(sent == recv);
        }

        assert(this->particles.empty());
        MPI_Barrier(this->comm_world);

        this->handlerMemoryBytes_ = this->buffersManager->GetTotalMemoryBytes();
        this->buffersManager = nullptr; // TODO: good?
        this->ClearParticlesChanged();
        // if(this->rank_world == 0)
        // std::cout << "====================================" << std::endl;
        // MPI_Barrier(this->comm_world);
    }
    catch(const STORMError &eo)
    {
        reportError(eo);
        throw;
    }
}

#endif // STORM_TWO_SIDED_MONTE_CARLO_LIFECYCLE_HPP
