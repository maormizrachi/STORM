#include <exception>
#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include "manager/parallel/ReallocationAgent.hpp"

namespace {

int runTest()
{
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != 4)
        throw std::runtime_error("reallocation test requires exactly four MPI ranks");

    int reallocations = 0;
    const bool requestsSynchronousReallocation = rank % 2 == 0;
    const int peer = requestsSynchronousReallocation ? rank + 1 : rank - 1;
    STORM::ReallocationAgent agent(
        MPI_COMM_WORLD,
        [&](rank_t callbackPeer)
        {
            int receivedRank = -1;
            MPI_Sendrecv(&rank, 1, MPI_INT, callbackPeer, 0,
                         &receivedRank, 1, MPI_INT, callbackPeer, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(receivedRank != callbackPeer)
                throw std::runtime_error("reallocation callback paired with the wrong peer");
            ++reallocations;
        },
        [](rank_t, double)
        {
            return STORM::ReallocationMetadata{};
        },
        [](rank_t, const STORM::ReallocationMetadata&) {});

    MPI_Barrier(MPI_COMM_WORLD);
    if(requestsSynchronousReallocation)
    {
        agent.RequestReallocation(peer);
    }
    else
    {
        while(reallocations == 0)
            agent.ProgressAsyncReallocations();
    }

    MPI_Request allRequestsCompleted;
    MPI_Ibarrier(MPI_COMM_WORLD, &allRequestsCompleted);
    int allCompleted = 0;
    while(not allCompleted)
    {
        agent.ProgressAsyncReallocations();
        MPI_Test(&allRequestsCompleted, &allCompleted, MPI_STATUS_IGNORE);
    }

    if(reallocations != 1)
    {
        std::cerr << "rank " << rank << ": expected 1 reallocation callback, got "
                  << reallocations << '\n';
    }
    const int localSuccess = (reallocations == 1) ? 1 : 0;
    int globalSuccess = 0;
    MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalSuccess == 1 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int result = 1;
    try
    {
        result = runTest();
    }
    catch(const std::exception &error)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
    }

    MPI_Finalize();
    return result;
}
