#ifndef STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP
#define STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP

#define HANDLER_BREAKDOWN_LEN_TAG 999
#define HANDLER_BREAKDOWN_DATA_TAG 1000

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::ShrinkBuffers(void)
{
    if(this->fixedTransportActive)
    {
        throw std::runtime_error("RDMA engine: shrinking during fixed transport");
    }
    if(this->rankWorld == 0)
    {
        std::cout << "Shrinking buffers." << std::endl;
    }
    std::vector<rank_t> shrinkList;
    boost::container::flat_set<rank_t> neighbors(this->neighbors.cbegin(), this->neighbors.cend());
    size_t avgBuffSize = 0;
    size_t numHandlers = 0;
    for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
    {
        if(r != this->rankWorld and this->rankHandlers[r] != nullptr)
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
        if(r != this->rankWorld and this->rankHandlers[r] != nullptr and
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

    std::vector<std::vector<rank_t>> shrinkRequests(this->sizeWorld);
    for(rank_t r : shrinkList)
    {
        shrinkRequests[r].push_back(this->rankWorld);
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkRequests =
        MPI_Exchange_sparse_by_rank(shrinkRequests, this->commWorld, MPI_EXCHANGE_SPARSE_TAG + 20);

    boost::container::flat_set<rank_t> shrinkCandidates(shrinkList.cbegin(), shrinkList.cend());
    for(const auto &[requestingRank, ignoredPayload] : incomingShrinkRequests)
    {
        (void)ignoredPayload;
        shrinkCandidates.insert(requestingRank);
    }

    std::vector<std::vector<rank_t>> shrinkConfirmations(this->sizeWorld);
    size_t candidatesWithoutHandler = 0;
    for(rank_t r : shrinkCandidates)
    {
        if(r == this->rankWorld)
        {
            continue;
        }

        if(this->rankHandlers[r] != nullptr and
           this->rankHandlers[r]->SupportsShrinkingReallocation())
        {
            shrinkConfirmations[r].push_back(this->rankWorld);
        }
        else
        {
            candidatesWithoutHandler++;
        }
    }

    std::vector<std::pair<rank_t, std::vector<rank_t>>> incomingShrinkConfirmations =
        MPI_Exchange_sparse_by_rank(shrinkConfirmations, this->commWorld, MPI_EXCHANGE_SPARSE_TAG + 21);

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
        if(r == this->rankWorld)
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

    auto shrinkBuffer = [this](rank_t rank)
    {
        double factor;
        if(std::find(this->neighbors.cbegin(), this->neighbors.cend(), rank) != this->neighbors.cend())
        {
            factor = this->config.bufferShrinkNeighborFactor;
        }
        else
        {
            factor = this->config.bufferShrinkFactor;
        }
        this->rankHandlers[rank]->requestedFactor = factor;
        this->rankHandlers[rank]->Reallocate(factor);
    };

    for(rank_t r : shrinkPartners)
    {
        shrinkBuffer(r);
    }
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::RetireStaleHandlers(void)
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
        if(rank != this->rankWorld and this->rankHandlers[rank] != nullptr and
           currentNeighbors.find(rank) == currentNeighbors.end())
        {
            staleCandidates.push_back(rank);
        }
    }

    int localCandidates = static_cast<int>(staleCandidates.size());
    int globalCandidates = 0;
    MPI_Allreduce(&localCandidates, &globalCandidates, 1, MPI_INT, MPI_SUM,
                  this->commWorld);
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
        RMAFactory::MakeProgress(this->rdmaType);
        this->reallocationAgent->ProgressAsyncReallocations();
        int localPending = this->reallocationAgent->HasPendingAsyncReallocations() ? 1 : 0;
        MPI_Allreduce(&localPending, &globalPending, 1, MPI_INT, MPI_MAX,
                      this->commWorld);
    } while(globalPending != 0);
    MPI_Barrier(this->commWorld);

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
            not this->reallocationAgent->IsPendingReallocation(rank);
        int pairSafe = 0;
        MPI_Allreduce(&localSafe, &pairSafe, 1, MPI_INT, MPI_MIN,
                      coordinationComm);

        if(pairSafe)
        {
            this->sendBuffers[rankIndex].ReleaseStorage();

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
    ForEachRankSyncByList(this->commWorld, staleCandidates, retirePair);

    unsigned long long localRetiredHandlers =
        static_cast<unsigned long long>(retiredHandlers);
    unsigned long long globalRetiredHandlers = 0;
    unsigned long long localRetiredBytes =
        static_cast<unsigned long long>(retiredParticleBytes);
    unsigned long long globalRetiredBytes = 0;
    MPI_Reduce(&localRetiredHandlers, &globalRetiredHandlers, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->commWorld);
    MPI_Reduce(&localRetiredBytes, &globalRetiredBytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->commWorld);
    if(this->rankWorld == 0 and globalRetiredHandlers > 0)
    {
        std::cout << "Retired " << globalRetiredHandlers
                  << " stale peer handlers, releasing "
                  << static_cast<double>(globalRetiredBytes) / (1024.0 * 1024.0)
                  << " MiB of particle queue capacity." << std::endl;
    }
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::PrepareHandlers(void)
{

    this->neighbors = GetNeighborList2(this->grid, this->ranksGhostMap);

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
    MPI_Allreduce(MPI_IN_PLACE, &numNewNeighbors, 1, MPI_INT, MPI_SUM, this->commWorld);

    if(numNewNeighbors > 0)
    {
        auto createHandler = [this](rank_t rank, MPI_Comm pairComm)
        {
            if(this->rankHandlers[rank] != nullptr)
            {
                return;
            }

            this->communicators[rank] = pairComm;
            this->rankHandlers[rank] = new RankHandler_t(this->config.initialBufferSize, this->commWorld, pairComm, this->reallocationAgent, this->rdmaType, this->config.minimalBuffSize);
            if(this->rankHandlers[rank]->peer_rank_world != rank)
            {
                STORMError eo("Peer rank world does not match");
                eo.addEntry("Rank", rank);
                eo.addEntry("Peer Rank World", this->rankHandlers[rank]->peer_rank_world);
                throw eo;
            }
        };
        ForEachRankSyncByList(this->commWorld, newNeighbors, createHandler);
    }

    if(this->config.rdmaFixedStepQueues) this->AdaptFixedQueueCapacities();
    this->ResetAllBuffers();
}

template<typename T, typename Grid>
void RDMACommunicationEngine<T, Grid>::AdaptFixedQueueCapacities()
{
    // Demand is recorded by the producer, but allocation belongs to the
    // consumer. Exchange it collectively while all rings are quiescent.
    std::vector<unsigned long long> outgoing(this->sizeWorld, 0), incoming(this->sizeWorld, 0);
    for(size_t rank = 0; rank < this->rankHandlers.size(); ++rank)
    {
        if(this->rankHandlers[rank])
        {
            outgoing[rank] = std::min(this->rankHandlers[rank]->NextPeerCapacity(), this->config.rdmaFixedQueueMaxSize);

        }
    }
    MPI_Alltoall(outgoing.data(), 1, MPI_UNSIGNED_LONG_LONG, incoming.data(), 1, MPI_UNSIGNED_LONG_LONG, this->commWorld);
    // Both endpoints compute the same required capacity. Sorted peer order
    // gives the pair collectives an acyclic order (also used by shrinking).
    for(size_t rank = 0; rank < this->rankHandlers.size(); ++rank)
    {
        RankHandler_t *handler = this->rankHandlers[rank];
        if(not handler) continue;
        const size_t required = std::max<size_t>(outgoing[rank], incoming[rank]);
        if(required > std::min(handler->buffsize, handler->peer_buffsize))
        {
            handler->Reallocate(1.0, required);
        }
    }
}

#endif // STORM_RDMA_RANK_HANDLER_LIFECYCLE_HPP
