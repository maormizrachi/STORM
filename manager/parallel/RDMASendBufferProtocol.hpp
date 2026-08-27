#ifndef STORM_RDMA_SEND_BUFFER_PROTOCOL_HPP
#define STORM_RDMA_SEND_BUFFER_PROTOCOL_HPP

template<typename T, typename Grid, typename Physics>
typename RDMAMonteCarloManager<T, Grid, Physics>::RegisteredSendBuffer_t &RDMAMonteCarloManager<T, Grid, Physics>::GetSendBuffer(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBuffers.size()));
    return this->sendBuffers[static_cast<size_t>(rank)];
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::QueueReadySendBuffer(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBufferReadyQueued.size()));
    size_t rankIndex = static_cast<size_t>(rank);
    if(this->sendBufferReadyQueued[rankIndex])
    {
        return;
    }
    this->sendBufferReadyQueued[rankIndex] = 1;
    this->readySendBufferRanks.push_back(rank);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::MarkSendBufferEmpty(rank_t rank)
{
    assert(rank >= 0);
    assert(rank < static_cast<rank_t>(this->sendBuffers.size()));
    size_t rankIndex = static_cast<size_t>(rank);
    this->sendBuffers[rankIndex].clear();
    this->sendBufferReadyQueued[rankIndex] = 0;
    if(this->sendBufferActive[rankIndex])
    {
        this->sendBufferActive[rankIndex] = 0;
        assert(this->sendBufferPendingRanks > 0);
        this->sendBufferPendingRanks--;
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ResetSendBuffers(void)
{
    for(rank_t rank : this->sendBufferActiveRanks)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            continue;
        }
        this->sendBuffers[static_cast<size_t>(rank)].clear();
    }
    std::fill(this->sendBufferActive.begin(), this->sendBufferActive.end(), 0);
    std::fill(this->sendBufferListed.begin(), this->sendBufferListed.end(), 0);
    std::fill(this->sendBufferReadyQueued.begin(), this->sendBufferReadyQueued.end(), 0);
    this->sendBufferActiveRanks.clear();
    this->readySendBufferRanks.clear();
    this->readySendBufferCursor = 0;
    this->sendBufferPendingRanks = 0;
    this->sendBufferCycleCounter = 0;
    this->sendBufferPendingParticles = 0;
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::ReleaseSendBufferRegistrations(void)
{
    for(RegisteredSendBuffer_t &buffer : this->sendBuffers)
    {
        buffer.ReleaseRegistration();
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::NoteSendBufferGrowth(rank_t rank, size_t previousSize, const RegisteredSendBuffer_t &buffer, size_t addedParticles)
{
    this->sendBufferPendingParticles += addedParticles;
    if(addedParticles > 0 and previousSize == 0 and not buffer.empty())
    {
        assert(rank >= 0);
        assert(rank < static_cast<rank_t>(this->sendBufferActive.size()));
        size_t rankIndex = static_cast<size_t>(rank);
        if(not this->sendBufferActive[rankIndex])
        {
            this->sendBufferActive[rankIndex] = 1;
            this->sendBufferPendingRanks++;
        }
        if(not this->sendBufferListed[rankIndex])
        {
            this->sendBufferListed[rankIndex] = 1;
            this->sendBufferActiveRanks.push_back(rank);
        }
    }
    if(previousSize < this->config.sendBufferMinSize and buffer.size() >= this->config.sendBufferMinSize)
    {
        this->QueueReadySendBuffer(rank);
    }
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::NoteSendBufferFlush(rank_t rank, size_t flushedParticles)
{
    assert(this->sendBufferPendingParticles >= flushedParticles);
    this->sendBufferPendingParticles -= flushedParticles;
    this->MarkSendBufferEmpty(rank);
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::FlushSendBuffers(bool flushSmallBuffers)
{
    size_t pendingRanks = this->sendBufferPendingRanks;

    if(pendingRanks > 0)
    {
        this->sendBufferCycleCounter++;
    }
    else
    {
        this->sendBufferCycleCounter = 0;
    }

    bool allowIdleDrain = flushSmallBuffers;
    bool heldIdleDrain = false;
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    if(flushSmallBuffers and this->config.holdSmallIdleFlushes)
    {
        size_t holdoffCycles = std::max<size_t>(1, this->config.GetSmallIdleFlushHoldoffCycles());
        allowIdleDrain = this->sendBufferCycleCounter >= holdoffCycles;
    }

    if(pendingRanks == 0)
    {
        return;
    }

    auto flushRankIfReady = [&](rank_t toRank, bool allowIdleFlush)
    {
        if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            return;
        }
        size_t rankIndex = static_cast<size_t>(toRank);
        if(not this->sendBufferActive[rankIndex])
        {
            return;
        }
        RegisteredSendBuffer_t &particles = this->sendBuffers[rankIndex];
        if(particles.empty())
        {
            this->MarkSendBufferEmpty(toRank);
            return;
        }
        bool thresholdFlush = particles.size() >= this->config.sendBufferMinSize;
        bool idleFlush = allowIdleFlush &&
            (particles.size() >= this->config.sendBufferMinIdleDrainSize ||
             this->sendBufferCycleCounter >= this->config.sendBufferIdleDrainPatienceCycles);
        if(not thresholdFlush and not idleFlush)
        {
            return;
        }
        if(usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank))
        {
            if(thresholdFlush)
            {
                this->QueueReadySendBuffer(toRank);
            }
            return;
        }
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        size_t flushedParticles = particles.size();
        uint32_t sourceLkey = particles.SourceLkey(
            remoteHandler, remoteHandler->SupportsPersistentSendSourceRegistration());
        bool transferred = remoteHandler->TransferParticles(particles.data(), particles.size(), sourceLkey);
        if(not transferred)
        {
            this->ProgressReallocations();
            if(thresholdFlush)
            {
                this->QueueReadySendBuffer(toRank);
            }
            return;
        }
        this->transfersCounter++;
        this->NoteSendBufferFlush(toRank, flushedParticles);
    };

    size_t readyEntries = this->readySendBufferRanks.size() - this->readySendBufferCursor;
    for(size_t readyIndex = 0; readyIndex < readyEntries; readyIndex++)
    {
        rank_t toRank = this->readySendBufferRanks[this->readySendBufferCursor++];
        if(toRank >= 0 and toRank < static_cast<rank_t>(this->sendBufferReadyQueued.size()))
        {
            this->sendBufferReadyQueued[static_cast<size_t>(toRank)] = 0;
        }
        flushRankIfReady(toRank, false);
    }
    if(this->readySendBufferCursor >= this->readySendBufferRanks.size())
    {
        this->readySendBufferRanks.clear();
        this->readySendBufferCursor = 0;
    }
    else if(this->readySendBufferCursor > 1024 and this->readySendBufferCursor * 2 > this->readySendBufferRanks.size())
    {
        this->readySendBufferRanks.erase(this->readySendBufferRanks.begin(),
                                         this->readySendBufferRanks.begin() + static_cast<std::ptrdiff_t>(this->readySendBufferCursor));
        this->readySendBufferCursor = 0;
    }

    if(allowIdleDrain)
    {
        for(size_t index = 0; index < this->sendBufferActiveRanks.size();)
        {
            rank_t toRank = this->sendBufferActiveRanks[index];
            if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
            {
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            size_t rankIndex = static_cast<size_t>(toRank);
            if(not this->sendBufferActive[rankIndex])
            {
                this->sendBufferListed[rankIndex] = 0;
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            flushRankIfReady(toRank, true);
            if(not this->sendBufferActive[rankIndex])
            {
                this->sendBufferListed[rankIndex] = 0;
                this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
                this->sendBufferActiveRanks.pop_back();
                continue;
            }
            if(flushSmallBuffers)
            {
                heldIdleDrain = true;
            }
            index++;
        }
    }
    else if(flushSmallBuffers and this->sendBufferPendingRanks > 0)
    {
        heldIdleDrain = true;
    }

    if(this->sendBufferPendingRanks == 0)
    {
        this->sendBufferCycleCounter = 0;
    }
    (void)heldIdleDrain;
}

template<typename T, typename Grid, typename Physics>
void RDMAMonteCarloManager<T, Grid, Physics>::FlushAllSendBuffers(void)
{
    const bool usesAsyncReallocation = this->UsesAsyncReallocation();
    for(size_t index = 0; index < this->sendBufferActiveRanks.size();)
    {
        rank_t toRank = this->sendBufferActiveRanks[index];
        if(toRank < 0 or toRank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        size_t rankIndex = static_cast<size_t>(toRank);
        if(not this->sendBufferActive[rankIndex])
        {
            this->sendBufferListed[rankIndex] = 0;
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        RegisteredSendBuffer_t &particles = this->sendBuffers[rankIndex];
        if(particles.empty())
        {
            this->MarkSendBufferEmpty(toRank);
            this->sendBufferListed[rankIndex] = 0;
            this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
            this->sendBufferActiveRanks.pop_back();
            continue;
        }
        size_t flushedParticles = particles.size();
        if(usesAsyncReallocation and this->reallocationAgent->IsPendingReallocation(toRank))
        {
            index++;
            continue;
        }
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        uint32_t sourceLkey = particles.SourceLkey(
            remoteHandler, remoteHandler->SupportsPersistentSendSourceRegistration());
        bool transferred = remoteHandler->TransferParticles(particles.data(), particles.size(), sourceLkey);
        if(not transferred)
        {
            this->ProgressReallocations();
            index++;
            continue;
        }
        this->transfersCounter++;
        this->NoteSendBufferFlush(toRank, flushedParticles);
        this->sendBufferListed[rankIndex] = 0;
        this->sendBufferActiveRanks[index] = this->sendBufferActiveRanks.back();
        this->sendBufferActiveRanks.pop_back();
    }
    if(this->sendBufferPendingRanks == 0)
    {
        this->sendBufferCycleCounter = 0;
    }
}

template<typename T, typename Grid, typename Physics>
bool RDMAMonteCarloManager<T, Grid, Physics>::AllSendBuffersEmpty(void) const
{
    if(this->sendBufferPendingParticles == 0)
    {
        return true;
    }
    for(rank_t rank : this->sendBufferActiveRanks)
    {
        if(rank < 0 or rank >= static_cast<rank_t>(this->sendBuffers.size()))
        {
            continue;
        }
        size_t rankIndex = static_cast<size_t>(rank);
        if(this->sendBufferActive[rankIndex] and not this->sendBuffers[rankIndex].empty())
        {
            return false;
        }
    }
    return false;
}

#endif // STORM_RDMA_SEND_BUFFER_PROTOCOL_HPP
