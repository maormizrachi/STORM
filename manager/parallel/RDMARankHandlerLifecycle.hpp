#ifndef STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP
#define STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP

#define HANDLER_BREAKDOWN_LEN_TAG 999
#define HANDLER_BREAKDOWN_DATA_TAG 1000

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ShrinkBuffers(void)
{
    if(this->rank_world == 0)
    {
        std::cout << "Shrinking buffers." << std::endl;
    }
    std::vector<rank_t> shrinkList;
    boost::container::flat_set<rank_t> neighbors(this->neighbors.cbegin(), this->neighbors.cend());
    size_t avgBuffSize = 0;
    size_t numHandlers = 0;
    for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
    {
        if(r != this->rank_world and this->rankHandlers[r] != nullptr)
        {
            avgBuffSize += this->rankHandlers[r]->buffsize;
            numHandlers++;
        }
    }
    if(numHandlers > 0)
    {
        avgBuffSize /= numHandlers;
    }
    for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
    {
        if(r != this->rank_world and this->rankHandlers[r] != nullptr and
           this->rankHandlers[r]->SupportsShrinkingReallocation() and
           this->rankHandlers[r]->buffsize > this->config.minimalBuffSize)
        {
            bool isNeighbor = neighbors.find(r) != neighbors.end();
            if(!isNeighbor)
            {
                shrinkList.push_back(r);
            }
            else if(this->rankHandlers[r]->buffsize > 4 * std::max(avgBuffSize, this->config.minimalBuffSize))
            {
                shrinkList.push_back(r);
            }
        }
    }
    const size_t localOriginalRequested = shrinkList.size();
    double shrinkPercent = this->config.shrinkPercent;
    shrinkPercent = std::max(0.0, std::min(1.0, shrinkPercent));
    if(not shrinkList.empty() and shrinkPercent < 1.0)
    {
        size_t shrinkBudget = 0;
        if(shrinkPercent > 0.0)
        {
            shrinkBudget = std::max<size_t>(
                1,
                static_cast<size_t>(std::ceil(shrinkPercent * static_cast<double>(shrinkList.size()))));
        }
        if(shrinkBudget < shrinkList.size())
        {
            shrinkList.resize(shrinkBudget);
        }
    }

    std::vector<std::vector<rank_t>> shrinkRequests(this->size_world);
    for(rank_t r : shrinkList)
    {
        shrinkRequests[r].push_back(this->rank_world);
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkRequests =
        MPI_Exchange_sparse_by_rank(shrinkRequests, this->comm_world, MPI_EXCHANGE_SPARSE_TAG + 20);

    boost::container::flat_set<rank_t> shrinkCandidates(shrinkList.cbegin(), shrinkList.cend());
    for(const auto &[requestingRank, ignoredPayload] : incomingShrinkRequests)
    {
        (void)ignoredPayload;
        shrinkCandidates.insert(requestingRank);
    }

    std::vector<std::vector<rank_t>> shrinkConfirmations(this->size_world);
    size_t candidatesWithoutHandler = 0;
    for(rank_t r : shrinkCandidates)
    {
        if(r == this->rank_world)
        {
            continue;
        }

        if(this->rankHandlers[r] != nullptr and
           this->rankHandlers[r]->SupportsShrinkingReallocation())
        {
            shrinkConfirmations[r].push_back(this->rank_world);
        }
        else
        {
            candidatesWithoutHandler++;
        }
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkConfirmations =
        MPI_Exchange_sparse_by_rank(shrinkConfirmations, this->comm_world, MPI_EXCHANGE_SPARSE_TAG + 21);

    boost::container::flat_set<rank_t> confirmedByPeer;
    for(const auto &[confirmingRank, ignoredPayload] : incomingShrinkConfirmations)
    {
        (void)ignoredPayload;
        confirmedByPeer.insert(confirmingRank);
    }

    std::vector<rank_t> shrinkPartners;
    shrinkPartners.reserve(shrinkCandidates.size());
    size_t missingPeerConfirmation = 0;
    for(rank_t r : shrinkCandidates)
    {
        if(r == this->rank_world)
        {
            continue;
        }

        if(this->rankHandlers[r] == nullptr)
        {
            continue;
        }

        if(confirmedByPeer.find(r) == confirmedByPeer.end())
        {
            missingPeerConfirmation++;
            continue;
        }

        shrinkPartners.push_back(r);
    }

    std::sort(shrinkPartners.begin(), shrinkPartners.end());

    (void)localOriginalRequested;
    (void)candidatesWithoutHandler;
    (void)missingPeerConfirmation;

    auto shrinkBuffer = [this](rank_t _rank)
    {
        double factor;
        if(std::find(this->neighbors.cbegin(), this->neighbors.cend(), _rank) != this->neighbors.cend())
        {
            factor = this->config.bufferShrinkNeighborFactor;
        }
        else
        {
            factor = this->config.bufferShrinkFactor;
        }
        this->rankHandlers[_rank]->requestedFactor = factor;
        this->rankHandlers[_rank]->Reallocate(factor);
    };

    for(rank_t r : shrinkPartners)
    {
        shrinkBuffer(r);
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum)
{
    const size_t bytesPerSlot = sizeof(MCParticle);
    size_t localHandlerMemory = 0;
    for(const RankHandler_t *h : this->rankHandlers)
    {
        if(h == nullptr)
        {
            continue;
        }
        localHandlerMemory += h->buffsize * bytesPerSlot;
    }
    size_t localSendBufferMemory = 0;
    size_t localDetachedMemory = 0;
    for(const RegisteredSendBuffer_t &buffer : this->sendBuffers)
    {
        localSendBufferMemory += buffer.capacity() * bytesPerSlot;
    }
    for(const std::vector<MCParticle> &particles : this->detachedRankParticles)
    {
        localDetachedMemory += particles.capacity() * bytesPerSlot;
    }

    struct { double val; int rank; } myMem, maxMem;
    myMem.val = static_cast<double>(localHandlerMemory);
    myMem.rank = this->rank_world;
    MPI_Allreduce(&myMem, &maxMem, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm_world);

    double avgMem = static_cast<double>(localHandlerMemory);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &avgMem, &avgMem, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    size_t preLoopInitial = initialParticlesNum;
    size_t preLoopPreStep = preStepParticlesNum;
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopInitial, &preLoopInitial, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopPreStep, &preLoopPreStep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    std::string handlerBreakdown;
    if(this->rank_world == maxMem.rank)
    {
        boost::container::flat_set<rank_t> neighborSet(this->neighbors.begin(), this->neighbors.end());
        std::ostringstream ss;
        double selfTotal = 0, neighborTotal = 0, nonNeighborTotal = 0;
        for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
        {
            const RankHandler_t *h = this->rankHandlers[r];
            if(h == nullptr)
            {
                continue;
            }
            double handlerMB = h->buffsize * bytesPerSlot / (1024.0 * 1024.0);
            bool isSelf = (h->peer_rank_world == this->rank_world);
            bool isNeighbor = !isSelf && neighborSet.count(r) > 0;
            (isSelf ? selfTotal : isNeighbor ? neighborTotal : nonNeighborTotal) += handlerMB;
            // const char *tag = isSelf ? " [self]" : isNeighbor ? " [neighbor]" : " [non-neighbor]";
            // ss << "  [" << h->peer_rank_world << "]: "
            //    << handlerMB << " MB (buffsize=" << h->buffsize << ")"
            //    << tag << "\n";
        }
        ss << "  Totals: self=" << selfTotal << " MB, neighbors=" << neighborTotal << " MB, non-neighbors=" << nonNeighborTotal << " MB\n";
        handlerBreakdown = ss.str();
    }

    if(maxMem.rank != 0)
    {
        if(this->rank_world == maxMem.rank)
        {
            int strLen = static_cast<int>(handlerBreakdown.size());
            MPI_Send(&strLen, 1, MPI_INT, 0, HANDLER_BREAKDOWN_LEN_TAG, this->comm_world);
            MPI_Send(handlerBreakdown.data(), strLen, MPI_CHAR, 0, HANDLER_BREAKDOWN_DATA_TAG, this->comm_world);
        }
        else if(this->rank_world == 0)
        {
            int strLen = 0;
            MPI_Recv(&strLen, 1, MPI_INT, maxMem.rank, HANDLER_BREAKDOWN_LEN_TAG, this->comm_world, MPI_STATUS_IGNORE);
            handlerBreakdown.resize(strLen);
            MPI_Recv(&handlerBreakdown[0], strLen, MPI_CHAR, maxMem.rank, HANDLER_BREAKDOWN_DATA_TAG, this->comm_world, MPI_STATUS_IGNORE);
        }
    }

    if(this->rank_world == 0)
    {
        avgMem /= this->size_world;
        std::cout << "RankHandler memory: max=" << maxMem.val / (1 << 20) << " MB (rank " << maxMem.rank << "), avg=" << avgMem / (1 << 20) << " MB" << std::endl;
        std::cout << handlerBreakdown;
        std::cout << "Local auxiliary particle capacity on rank 0: send="
                  << static_cast<double>(localSendBufferMemory) / (1 << 20)
                  << " MB, detached="
                  << static_cast<double>(localDetachedMemory) / (1 << 20)
                  << " MB" << std::endl;
        std::cout << "Starting with " << (preLoopInitial + preLoopPreStep) << ". Came with " << preLoopInitial << ". Generated " << preLoopPreStep << " particles in preStep." << std::endl;
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::RetireStaleHandlers(void)
{
    if(not this->config.retireStaleHandlers)
    {
        return;
    }

    boost::container::flat_set<rank_t> currentNeighbors(
        this->neighbors.cbegin(), this->neighbors.cend());
    std::vector<rank_t> staleCandidates;
    for(rank_t rank = 0; rank < static_cast<rank_t>(this->rankHandlers.size()); rank++)
    {
        if(rank != this->rank_world and this->rankHandlers[rank] != nullptr and
           currentNeighbors.find(rank) == currentNeighbors.end())
        {
            staleCandidates.push_back(rank);
        }
    }

    int localCandidates = static_cast<int>(staleCandidates.size());
    int globalCandidates = 0;
    MPI_Allreduce(&localCandidates, &globalCandidates, 1, MPI_INT, MPI_SUM,
                  this->comm_world);
    if(globalCandidates == 0)
    {
        return;
    }

    // A request or metadata message can outlive the transport operation that
    // triggered a resize. Drain those messages globally before deleting any
    // handler referenced by ReallocationAgent callbacks.
    int globalPending = 0;
    do
    {
        RMAFactory::MakeProgress(this->rdma_type);
        this->reallocationAgent->ProgressAsyncReallocations();
        int localPending = this->reallocationAgent->HasPendingAsyncReallocations() ? 1 : 0;
        MPI_Allreduce(&localPending, &globalPending, 1, MPI_INT, MPI_MAX,
                      this->comm_world);
    } while(globalPending != 0);
    MPI_Barrier(this->comm_world);

    size_t retiredHandlers = 0;
    size_t retiredParticleBytes = 0;
    auto retirePair = [this, &currentNeighbors, &retiredHandlers,
                       &retiredParticleBytes](rank_t rank, MPI_Comm coordinationComm)
    {
        const size_t rankIndex = static_cast<size_t>(rank);
        RankHandler_t *handler = this->rankHandlers[rankIndex];
        int localSafe =
            handler != nullptr and
            currentNeighbors.find(rank) == currentNeighbors.end() and
            handler->LocalEmpty() and
            this->sendBuffers[rankIndex].empty() and
            this->detachedRankParticles[rankIndex].empty() and
            not this->reallocationAgent->IsPendingReallocation(rank);
        int pairSafe = 0;
        MPI_Allreduce(&localSafe, &pairSafe, 1, MPI_INT, MPI_MIN,
                      coordinationComm);

        if(pairSafe)
        {
            this->sendBuffers[rankIndex].ReleaseStorage();
            std::vector<MCParticle>().swap(this->detachedRankParticles[rankIndex]);

            retiredParticleBytes += handler->buffsize * sizeof(MCParticle);
            handler->Destroy();
            delete handler;
            this->rankHandlers[rankIndex] = nullptr;

            MPI_Comm &oldComm = this->communicators[rankIndex];
            if(oldComm != MPI_COMM_NULL)
            {
                MPI_Comm_free(&oldComm);
            }
            retiredHandlers++;
        }

        MPI_Comm_free(&coordinationComm);
    };
    ForEachRankSyncByList(this->comm_world, staleCandidates, retirePair);

    unsigned long long localRetiredHandlers =
        static_cast<unsigned long long>(retiredHandlers);
    unsigned long long globalRetiredHandlers = 0;
    unsigned long long localRetiredBytes =
        static_cast<unsigned long long>(retiredParticleBytes);
    unsigned long long globalRetiredBytes = 0;
    MPI_Reduce(&localRetiredHandlers, &globalRetiredHandlers, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(&localRetiredBytes, &globalRetiredBytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0 and globalRetiredHandlers > 0)
    {
        std::cout << "Retired " << globalRetiredHandlers
                  << " stale peer handlers, releasing "
                  << static_cast<double>(globalRetiredBytes) / (1024.0 * 1024.0)
                  << " MiB of particle queue capacity." << std::endl;
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::PrepareHandlers(void)
{

    this->neighbors = GetNeighborList2(this->grid, this->ranks_ghost_map);

    // Self handler: 1-process communicator, no coordination needed
    if(this->rankHandlers[this->rank_world] == nullptr)
    {
        MPI_Comm_dup(MPI_COMM_SELF, &this->communicators[this->rank_world]);
        this->rankHandlers[this->rank_world] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, this->communicators[this->rank_world], this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
    }

    // Retire old peer state before allocating handlers for replacement
    // neighbors. The C++ allocator can reuse the released MCParticle[]
    // storage, while each backend creates fresh peer-specific registrations,
    // windows, addresses, keys, and mutexes.
    this->RetireStaleHandlers();

    std::vector<rank_t> newNeighbors;
    for(rank_t rank : this->neighbors)
    {
        if(this->rankHandlers[rank] == nullptr)
        {
            newNeighbors.push_back(rank);
        }
    }

    int numNewNeighbors = newNeighbors.size();
    MPI_Allreduce(MPI_IN_PLACE, &numNewNeighbors, 1, MPI_INT, MPI_SUM, this->comm_world);

    if(numNewNeighbors > 0)
    {
        auto createHandler = [this](rank_t rank, MPI_Comm pair_comm)
        {
            if(this->rankHandlers[rank] != nullptr)
            {
                return;
            }

            this->communicators[rank] = pair_comm;
            this->rankHandlers[rank] = new RankHandler_t(this->config.initialBufferSize, this->comm_world, pair_comm, this->reallocationAgent, this->rdma_type, this->config.minimalBuffSize);
            if(this->rankHandlers[rank]->peer_rank_world != rank)
            {
                STORMError eo("Peer rank world does not match");
                eo.addEntry("Rank", rank);
                eo.addEntry("Peer Rank World", this->rankHandlers[rank]->peer_rank_world);
                throw eo;
            }
        };
        ForEachRankSyncByList(this->comm_world, newNeighbors, createHandler);
    }

    this->ResetAllBuffers();
}

#endif // STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP
