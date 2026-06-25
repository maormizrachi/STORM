#ifdef RICH_MPI

#include "ReallocationAgent.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>

#define ASK_REALLOCATION_TAG 553
#define ANSWER_REALLOCATION_TAG 554
#define ASYNC_REALLOCATION_REQUEST_TAG 555
#define ASYNC_REALLOCATION_METADATA_TAG 556

static const rank_t NO_RANK = -1;

ReallocationAgent::ReallocationAgent(const MPI_Comm &comm, const ReallocationFunction &reallocationFunction,
                                     const LocalReallocationFunction &localReallocationFunction,
                                     const MetadataUpdateFunction &metadataUpdateFunction)
    : comm(comm), reallocationFunction(reallocationFunction),
      localReallocationFunction(localReallocationFunction), metadataUpdateFunction(metadataUpdateFunction),
      incomingAsyncRequest(MPI_REQUEST_NULL), asyncProgressCalls(0),
      asyncSendPollMinCycles(8), asyncIncomingPollActiveCycles(8),
      asyncIncomingPollIdleCycles(8), asyncMaxIncomingRequestsPerPoll(4)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    MPI_Barrier(this->comm);
    this->waitingFor = NO_RANK;
    MPI_Irecv(&this->incomingData, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &this->incomingRequest);
    if(this->AsyncEnabled())
    {
        MPI_Irecv(&this->incomingAsyncFactor, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASYNC_REALLOCATION_REQUEST_TAG, this->comm, &this->incomingAsyncRequest);
    }
}

ReallocationAgent::~ReallocationAgent()
{
    if(this->incomingRequest != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->incomingRequest);
    }
    if(this->incomingAsyncRequest != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->incomingAsyncRequest);
    }
}

void ReallocationAgent::GetIncoming(void)
{
    while(true)
    {
        int flag;
        MPI_Status status;
        MPI_Test(&this->incomingRequest, &flag, &status);
        if(__glibc_unlikely(flag))
        {
            this->incoming.push_back({status.MPI_SOURCE, this->incomingData});
            MPI_Irecv(&this->incomingData, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &this->incomingRequest);
        }
        else
        {
            break;
        }
    }

    std::sort(this->incoming.begin(), this->incoming.end(), [](const std::pair<rank_t, double> &a, const std::pair<rank_t, double> &b)
    {
        return a.second < b.second;
    });
}

rank_t ReallocationAgent::ShouldReallocate(void)
{
    this->GetIncoming();

    if(not this->incoming.empty())
    {
        rank_t toHandle = NO_RANK;
        auto it = this->incoming.end();

        if(this->waitingFor != NO_RANK)
        {
            it = std::find_if(this->incoming.begin(), this->incoming.end(), [this](const std::pair<rank_t, double> &p)
            {
                return p.first == this->waitingFor;
            });
            if(it == this->incoming.end())
            {
                // While waiting for a specific peer, only preempt that wait for a
                // lower-priority peer. This rank ordering breaks reallocation
                // cycles such as A waits for B, B waits for C, C waits for A.
                it = std::find_if(this->incoming.begin(), this->incoming.end(), [this](const std::pair<rank_t, double> &p)
                {
                    return p.first < this->waitingFor;
                });
                if(it == this->incoming.end())
                {
                    return NO_RANK;
                }
            }
        }
        else
        {
            it = this->incoming.begin();
        }
        
        toHandle = it->first;
        this->incoming.erase(it);
        
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, toHandle, ANSWER_REALLOCATION_TAG, this->comm);
        return toHandle;
    }

    return NO_RANK;
}

rank_t ReallocationAgent::HandleWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    if(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        this->reallocationsWhileWaiting++;
    }
    return handle;
}

void ReallocationAgent::HandleAllWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    while(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        handle = this->ShouldReallocate();
        this->reallocationsWhileWaiting++;
    }
}

void ReallocationAgent::RequestReallocation(rank_t fromRank)
{
    rank_t r;
    do
    {
        r = this->HandleWaitingReallocations();
        if(r == fromRank)
        {
            // the peer asked before
            return; // no need to ask again
        }
    } while(r != NO_RANK);
    
    this->waitingFor = fromRank;
    MPI_Request request1, request2;
    MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, fromRank, ANSWER_REALLOCATION_TAG, this->comm, &request1);
    double time = MPI_Wtime();
    MPI_Issend(&time, 1, MPI_DOUBLE, fromRank, ASK_REALLOCATION_TAG, this->comm, &request2);
    this->reallocationsWhileWaiting = 0;
    
    // bool alreadyJoinedWithPeer = false;
    // double alreadyJoinedTime = 0;

    // bool printed = false;

    while(true)
    {
        int flag;
        MPI_Test(&request1, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            // finally!
            MPI_Wait(&request2, MPI_STATUS_IGNORE);
            this->reallocationFunction(fromRank);
            this->waitingFor = NO_RANK;
            return;
        }
        else
        {
            this->HandleWaitingReallocations();
        }
    }
}

bool ReallocationAgent::AsyncEnabled(void) const
{
    return static_cast<bool>(this->localReallocationFunction) and static_cast<bool>(this->metadataUpdateFunction);
}

void ReallocationAgent::ConfigureAsyncPolling(size_t sendPollMinCycles,
                                             size_t incomingPollActiveCycles,
                                             size_t incomingPollIdleCycles,
                                             size_t maxIncomingRequestsPerPoll)
{
    this->asyncSendPollMinCycles = std::max<size_t>(1, sendPollMinCycles);
    this->asyncIncomingPollActiveCycles = std::max<size_t>(1, incomingPollActiveCycles);
    this->asyncIncomingPollIdleCycles = std::max<size_t>(1, incomingPollIdleCycles);
    this->asyncMaxIncomingRequestsPerPoll = maxIncomingRequestsPerPoll;
}

bool ReallocationAgent::ProgressAsyncSends(void)
{
    bool madeProgress = false;
    for(auto it = this->pendingFactorSends.begin(); it != this->pendingFactorSends.end();)
    {
        int flag = 0;
        MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            it = this->pendingFactorSends.erase(it);
            madeProgress = true;
        }
        else
        {
            ++it;
        }
    }

    for(auto it = this->pendingMetadataSends.begin(); it != this->pendingMetadataSends.end();)
    {
        int flag = 0;
        MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            it = this->pendingMetadataSends.erase(it);
            madeProgress = true;
        }
        else
        {
            ++it;
        }
    }
    return madeProgress;
}

void ReallocationAgent::RequestReallocationAsync(rank_t toRank, double factor)
{
    if(not this->AsyncEnabled())
    {
        throw std::runtime_error("ReallocationAgent::RequestReallocationAsync called without async callbacks");
    }
    if(this->pendingReallocRanks.find(toRank) != this->pendingReallocRanks.end())
    {
        return;
    }

    this->pendingReallocRanks.insert(toRank);
    this->pendingFactorSends.push_back({toRank, factor, MPI_REQUEST_NULL});
    PendingFactorSend &send = this->pendingFactorSends.back();
    MPI_Isend(&send.factor, 1, MPI_DOUBLE, toRank, ASYNC_REALLOCATION_REQUEST_TAG, this->comm, &send.request);
}

size_t ReallocationAgent::HandleIncomingAsyncRequests(size_t maxRequests)
{
    if(not this->AsyncEnabled() or this->incomingAsyncRequest == MPI_REQUEST_NULL)
    {
        return 0;
    }

    size_t handled = 0;
    while(maxRequests == 0 or handled < maxRequests)
    {
        int flag = 0;
        MPI_Status status;
        MPI_Test(&this->incomingAsyncRequest, &flag, &status);
        if(not flag)
        {
            break;
        }

        rank_t fromRank = status.MPI_SOURCE;
        double factor = this->incomingAsyncFactor;

        MPI_Irecv(&this->incomingAsyncFactor, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASYNC_REALLOCATION_REQUEST_TAG, this->comm, &this->incomingAsyncRequest);

        ReallocationMetadata metadata = this->localReallocationFunction(fromRank, factor);
        this->pendingMetadataSends.push_back({fromRank, metadata, MPI_REQUEST_NULL});
        PendingMetadataSend &send = this->pendingMetadataSends.back();
        MPI_Isend(&send.metadata, static_cast<int>(sizeof(ReallocationMetadata)), MPI_BYTE, fromRank, ASYNC_REALLOCATION_METADATA_TAG, this->comm, &send.request);
        handled++;
    }
    return handled;
}

bool ReallocationAgent::CheckMetadataUpdates(void)
{
    if(not this->AsyncEnabled() or this->pendingReallocRanks.empty())
    {
        return false;
    }

    bool madeProgress = false;
    while(true)
    {
        int flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, ASYNC_REALLOCATION_METADATA_TAG, this->comm, &flag, &status);
        if(not flag)
        {
            break;
        }

        ReallocationMetadata metadata;
        rank_t fromRank = status.MPI_SOURCE;
        MPI_Recv(&metadata, static_cast<int>(sizeof(ReallocationMetadata)), MPI_BYTE, fromRank, ASYNC_REALLOCATION_METADATA_TAG, this->comm, MPI_STATUS_IGNORE);
        this->metadataUpdateFunction(fromRank, metadata);
        this->pendingReallocRanks.erase(fromRank);
        madeProgress = true;
    }
    return madeProgress;
}

void ReallocationAgent::ProgressAsyncReallocations(void)
{
    if(not this->AsyncEnabled())
    {
        return;
    }

    this->asyncProgressCalls++;
    bool hasOutgoingSends = (not this->pendingFactorSends.empty()) or
                            (not this->pendingMetadataSends.empty());
    bool hasPendingRanks = not this->pendingReallocRanks.empty();
    bool hasLocalAsyncWork = hasOutgoingSends or hasPendingRanks;
    bool pollSends = hasOutgoingSends and
                     (this->asyncProgressCalls % this->asyncSendPollMinCycles == 0);
    if(pollSends)
    {
        this->ProgressAsyncSends();
    }
    this->CheckMetadataUpdates();
    size_t incomingPollCycles = hasLocalAsyncWork
        ? this->asyncIncomingPollActiveCycles
        : this->asyncIncomingPollIdleCycles;
    if(this->asyncProgressCalls % incomingPollCycles == 0)
    {
        this->HandleIncomingAsyncRequests(this->asyncMaxIncomingRequestsPerPoll);
    }
}

bool ReallocationAgent::IsPendingReallocation(rank_t rank) const
{
    return this->pendingReallocRanks.find(rank) != this->pendingReallocRanks.end();
}

bool ReallocationAgent::HasPendingAsyncReallocations(void) const
{
    return (not this->pendingReallocRanks.empty()) or
           (not this->pendingFactorSends.empty()) or
           (not this->pendingMetadataSends.empty());
}

#endif // RICH_MPI
