#ifndef STORM_RDMA_MANAGER_OPERATIONS_HPP
#define STORM_RDMA_MANAGER_OPERATIONS_HPP

template<typename T, typename Grid, typename Physics>
RDMAMonteCarloManager<T, Grid, Physics>::RDMAMonteCarloManager(const Grid &grid, const std::shared_ptr<Physics> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition, const MonteCarloConfig &config, const MPI_Comm &comm, RDMA_Type rdma_type) :
    grid(grid), config(config), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(MPI_COMM_NULL), tracker(comm), rdma_type(rdma_type), transportCore(this->localTransportExecutor)
{
    this->myIDCounter = 0;
    this->currentStep = 0;
#ifdef STORM_WITH_GPU
    // The executor allocates Kokkos DualViews, so it must not be built until
    // physics->preStep() has constructed KokkosRuntime.
#endif // STORM_WITH_GPU
    // this->progress = std::make_shared<ProgressCounter>(comm);
    this->comm_world = comm;
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    // OFIContext construction exchanges endpoint addresses over comm_world and
    // must therefore happen collectively before neighbor-specific handlers are built.
    RMAFactory::Initialize(this->rdma_type, this->comm_world);

    this->ranksOrder = GetRanksOrder(this->comm_world);
    this->communicators = std::vector<MPI_Comm>(this->size_world, MPI_COMM_NULL);

    this->rankHandlers = std::vector<RankHandler_t*>(this->size_world, nullptr);
    this->sendBuffers.resize(this->size_world);
    this->sendBufferActive.assign(this->size_world, 0);
    this->sendBufferListed.assign(this->size_world, 0);
    this->sendBufferReadyQueued.assign(this->size_world, 0);
    this->detachedRankParticles.resize(this->size_world);

    auto reallocationFunction = [this](rank_t rank)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        double factor = this->config.bufferReallocationFactor;
        handler->Reallocate(factor);
    };

    auto localReallocationFunction = [this](rank_t rank, double factor) -> ReallocationMetadata
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler == nullptr)
        {
            throw std::runtime_error("RDMAMonteCarloManager: received async reallocation request for a missing handler");
        }
        return handler->LocalReallocate(factor);
    };

    auto metadataUpdateFunction = [this](rank_t rank, const ReallocationMetadata &metadata)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler == nullptr)
        {
            throw std::runtime_error("RDMAMonteCarloManager: received async reallocation metadata for a missing handler");
        }
        handler->UpdatePeerRemoteInfo(metadata);
    };

    this->reallocationAgent = std::make_shared<ReallocationAgent>(this->comm_world, reallocationFunction,
                                                                  localReallocationFunction, metadataUpdateFunction);
    this->reallocationAgent->ConfigureAsyncPolling(
        this->config.asyncReallocationSendPollMinCycles,
        this->config.asyncReallocationIncomingPollActiveCycles,
        this->config.asyncReallocationIncomingPollIdleCycles,
        this->config.asyncReallocationMaxIncomingRequestsPerPoll);

    if(this->rank_world == 0)
    {
        std::cout << "Done initializing RDMAMonteCarloManager" << std::endl;
    }
    this->cellsStepsCounters.assign(this->grid.GetPointNo(), 0);
    this->cellsParticleCounters.assign(this->grid.GetPointNo(), 0);
    this->beginningParticleCount.assign(this->grid.GetPointNo(), 0);
    this->lastBuildGeneration = std::numeric_limits<size_t>::max();
    this->readySendBufferCursor = 0;
    this->sendBufferPendingRanks = 0;
    this->sendBufferCycleCounter = 0;
    this->sendBufferPendingParticles = 0;
    this->activeRankScanCursor = 0;
    this->activeRankScanRemaining = 0;
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ClearCommunicator()
{
    if(this->comm_world == MPI_COMM_NULL)
    {
        return;
    }

    if(this->communicators.size() < this->size_world)
    {
        return;
    }

    auto clearRankComm = [this](rank_t _rank)
    {
        MPI_Comm &comm = this->communicators[_rank];
        if(comm == MPI_COMM_NULL)
        {
            return;
        }
        MPI_Comm_free(&comm);
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, clearRankComm);

    this->comm_world = MPI_COMM_NULL;
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::FreeHandlers(void)
{
    auto freeHandler = [&](rank_t _rank)
    {
        RankHandler_t *handler = this->rankHandlers[_rank];
        if(handler != nullptr)
        {
            handler->Destroy();
            delete handler;
        }
        this->rankHandlers[_rank] = nullptr;
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, freeHandler);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::AddParticles(const std::vector<MCParticle> &particles)
{
    if(particles.empty())
    {
        return;
    }

    RankHandler_t *myHandler = this->rankHandlers[this->rank_world];
    size_t particlesNum = particles.size();
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    myHandler->AppendLocalParticles(particlesNum, [this, &particles, firstID](MCParticle &destination, size_t i)
    {
        destination = particles[i];
        destination.rank = this->rank_world;
        destination.id = firstID + i;

        #ifdef STORM_DEBUG
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
            STORMError eo("RDMAMonteCarloManager<T, Grid>::AddParticles");
            eo.addEntry("rank", this->rank_world);
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
    });

    this->localDecrementAmount -= static_cast<typename AmountManager::counter_t>(particlesNum);
}

template<typename T, typename Grid, typename Physics>
RDMAMonteCarloManager<T, Grid, Physics>::~RDMAMonteCarloManager()
{
    if(not std::uncaught_exceptions())
    {
        this->ReleaseSendBufferRegistrations();
        this->FreeHandlers();
        this->ClearCommunicator();
    }
}

template<typename T, typename Grid, typename Physics>
bool RDMAMonteCarloManager<T, Grid, Physics>::UsesAsyncReallocation(void) const
{
    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler and handler->UsesAsyncReallocation())
        {
            return true;
        }
    }
    return false;
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::PumpRMAProgress(void)
{
    if(this->UsesAsyncReallocation())
    {
        RMAFactory::MakeProgress(this->rdma_type);
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ProgressReallocations(void)
{
    this->PumpRMAProgress();
    if(this->UsesAsyncReallocation())
    {
        this->reallocationAgent->ProgressAsyncReallocations();
    }
    else
    {
        this->reallocationAgent->HandleAllWaitingReallocations();
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::MakeRDMAProgress(void)
{
    RMAFactory::MakeProgress(this->rdma_type);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::PutSelfParticles(std::vector<MCParticle> &&particles)
{
    #ifdef STORM_DEBUG
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
            eo.addEntry("Rank", this->rank_world);
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

    RankHandler_t *handler = this->rankHandlers[this->rank_world];
    handler->AppendLocalParticles(particlesNum, [this, &particles](MCParticle &destination, size_t i)
    {
        destination = particles[i];
        if(destination.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned, assign now
            destination.rank = this->rank_world;
            destination.id = this->myIDCounter++;
        }
    });

    // don't waste memory - remove current particles from the input vector
    std::vector<MCParticle> empty;
    particles.swap(empty);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::StageLocalParticlesForDevice(std::vector<MCParticle> &&particles, bool assignNewIDs)
{
    if(particles.empty())
    {
        return;
    }

    std::vector<MCParticle> &staged = this->detachedRankParticles[static_cast<size_t>(this->rank_world)];
    staged.reserve(staged.size() + particles.size());
    for(MCParticle &particle : particles)
    {
        if(assignNewIDs or particle.id == std::numeric_limits<particle_id_t>::max())
        {
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
        staged.push_back(std::move(particle));
    }
    std::vector<MCParticle>().swap(particles);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::TransferParticles(rank_t fromRank, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    RankHandler_t *currRankHandler = this->rankHandlers[fromRank];

    for(size_t i = 0; i < num; i++)
    {
        const size_t &particleIdx = indicesInToHandle[i];
        const rank_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        assert(particleIdx < currRankHandler->LocalSize());
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<MCParticle>();
        }

        MCParticle &particle = currRankHandler->LocalParticleAt(particleIdx);
        particle.sent = false; // reset

        if(toRank == this->rank_world)
        {
            STORMError eo("Trying to transfer particle to the same rank");
            eo.addEntry("Particle", particle);
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }

        rankToParticles[toRank].push_back(particle);

        #ifdef STORM_DEBUG
        if(toRank != particle.nextRank)
        {
            STORMError eo("Particle will not be sent to the expected rank #1");
            eo.addEntry("Particle", particle);
            eo.addEntry("Origin", particle.sentByRank);
            eo.addEntry("Expected Rank", toRank);
            eo.addEntry("Next Rank", particle.nextRank);
            throw eo;
        }
        #endif // STORM_DEBUG
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef STORM_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            STORMError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // STORM_DEBUG
        const bool usesAsyncReallocation = this->UsesAsyncReallocation();
        if(usesAsyncReallocation)
        {
            this->ProgressReallocations();
        }
        bool transferred = false;
        if(not (usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank)))
        {
            transferred = remoteHandler->TransferParticles(particles);
        }
        if(not transferred)
        {
            RegisteredSendBuffer_t &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.Append(particles.data(), particles.size());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;

    assert(rankBuffers.size() == indicesInToHandle.size());

    size_t numRanks = rankBuffers.size();
    for(size_t i = 0; i < numRanks; i++)
    {
        const rank_t &fromRank = rankBuffers[i];
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myParticleIndices = indicesInToHandle[i];
        size_t numToHandle = myParticleIndices.size();
        const std::vector<rank_t> &myTransferRanks = transferRanks[i];

        for(size_t j = 0; j < numToHandle; j++)
        {
            const size_t &particleIdx = myParticleIndices[j];
            const rank_t &toRank = myTransferRanks[j];

            assert(toRank != this->rank_world); // can't send to self
            assert(particleIdx < currRankHandler->LocalSize());
            auto it = rankToParticles.find(toRank);
            if(it == rankToParticles.end())
            {
                rankToParticles[toRank] = std::vector<MCParticle>();
            }

            MCParticle &particle = currRankHandler->LocalParticleAt(particleIdx);
            particle.sent = false; // reset

            if(toRank == this->rank_world)
            {
                STORMError eo("Trying to transfer particle to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }

            rankToParticles[toRank].push_back(particle);

            #ifdef STORM_DEBUG
            if(toRank != particle.nextRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #1");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
            #endif // STORM_DEBUG
        }
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef STORM_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            STORMError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                STORMError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // STORM_DEBUG
        const bool usesAsyncReallocation = this->UsesAsyncReallocation();
        if(usesAsyncReallocation)
        {
            this->ProgressReallocations();
        }
        bool transferred = false;
        if(not (usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank)))
        {
            transferred = remoteHandler->TransferParticles(particles);
        }
        if(not transferred)
        {
            RegisteredSendBuffer_t &buffer = this->GetSendBuffer(toRank);
            size_t previousSize = buffer.size();
            buffer.Append(particles.data(), particles.size());
            this->NoteSendBufferGrowth(toRank, previousSize, buffer, particles.size());
        }
        this->ProgressReallocations();
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ResetAllBuffers(void)
{
    this->activeRanks.clear();
    this->nextActiveRanks.clear();
    for(std::vector<MCParticle> &particles : this->detachedRankParticles)
    {
        particles.clear();
    }

    auto resetHandler = [this](rank_t rank)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->rankHandlers.size()))
        {
            return;
        }
        RankHandler_t *handler = this->rankHandlers[rank];
        if(handler != nullptr)
        {
            handler->Reset();
        }
    };

    resetHandler(this->rank_world);
    for(rank_t rank : this->neighbors)
    {
        if(rank != this->rank_world)
        {
            resetHandler(rank);
        }
    }
}

#endif // STORM_RDMA_MANAGER_OPERATIONS_HPP
