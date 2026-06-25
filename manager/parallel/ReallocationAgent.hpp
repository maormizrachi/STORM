#ifndef REALLOCATION_AGENT_HPP
#define REALLOCATION_AGENT_HPP

#ifdef RICH_MPI

#include <vector>
#include <list>
#include <set>
#include <functional>
#include <cstddef>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "utils/rma/RemoteMemoryAgent.hpp"

struct ReallocationMetadata
{
    RemoteBufferInfo particles;
    RemoteBufferInfo av;
    RemoteBufferInfo th;
    RemoteBufferInfo lengths;
    size_t new_buffsize = 0;
};

class ReallocationAgent
{
public:
    using ReallocationFunction = std::function<void(rank_t)>;
    using LocalReallocationFunction = std::function<ReallocationMetadata(rank_t, double)>;
    using MetadataUpdateFunction = std::function<void(rank_t, const ReallocationMetadata&)>;

    ReallocationAgent(const MPI_Comm &comm, const ReallocationFunction &reallocationFunction,
                      const LocalReallocationFunction &localReallocationFunction = LocalReallocationFunction(),
                      const MetadataUpdateFunction &metadataUpdateFunction = MetadataUpdateFunction());

    ~ReallocationAgent();

    void RequestReallocation(rank_t fromRank);

    void RequestReallocationAsync(rank_t toRank, double factor);

    rank_t HandleWaitingReallocations(void);

    void HandleAllWaitingReallocations(void);

    void ProgressAsyncReallocations(void);

    bool IsPendingReallocation(rank_t rank) const;

    bool HasPendingAsyncReallocations(void) const;

    void ConfigureAsyncPolling(size_t sendPollMinCycles,
                               size_t incomingPollActiveCycles,
                               size_t incomingPollIdleCycles,
                               size_t maxIncomingRequestsPerPoll);

private:
    MPI_Comm comm;
    rank_t rank;
    rank_t size;
    rank_t waitingFor;
    size_t reallocationsWhileWaiting;
    std::vector<std::pair<rank_t, double>> incoming;
    ReallocationFunction reallocationFunction;
    LocalReallocationFunction localReallocationFunction;
    MetadataUpdateFunction metadataUpdateFunction;
    double incomingData;
    MPI_Request incomingRequest;

    double incomingAsyncFactor;
    MPI_Request incomingAsyncRequest;
    std::set<rank_t> pendingReallocRanks;
    size_t asyncProgressCalls;
    size_t asyncSendPollMinCycles;
    size_t asyncIncomingPollActiveCycles;
    size_t asyncIncomingPollIdleCycles;
    size_t asyncMaxIncomingRequestsPerPoll;

    struct PendingFactorSend
    {
        rank_t rank;
        double factor;
        MPI_Request request;
    };

    struct PendingMetadataSend
    {
        rank_t rank;
        ReallocationMetadata metadata;
        MPI_Request request;
    };

    std::list<PendingFactorSend> pendingFactorSends;
    std::list<PendingMetadataSend> pendingMetadataSends;
    
    rank_t ShouldReallocate(void);

    void GetIncoming(void);

    bool AsyncEnabled(void) const;

    bool ProgressAsyncSends(void);

    size_t HandleIncomingAsyncRequests(size_t maxRequests);

    bool CheckMetadataUpdates(void);
};

#endif // RICH_MPI
#endif // REALLOCATION_AGENT_HPP
