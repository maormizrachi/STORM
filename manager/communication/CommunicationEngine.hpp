#ifndef STORM_COMMUNICATION_ENGINE_HPP
#define STORM_COMMUNICATION_ENGINE_HPP

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include "../../particle/Particle.hpp"
#include "../MonteCarloConfig.hpp"
#ifdef STORM_WITH_MPI
#include <mpi.h>
#endif

namespace STORM
{

// Engines own queued and in-flight packets. A detached batch transfers ownership
// to the caller; Send copies into engine-owned storage before returning.
template<class T>
class CommunicationEngine
{
public:
    using MCParticle = Particle<T>;
    virtual ~CommunicationEngine() = default;
    virtual void Prepare() = 0;
    virtual void BeginTransport() {}
    virtual void EndTransport() {}
    virtual void Progress() = 0;
    // Cheap network polling between compute slices; full maintenance stays in Progress.
    virtual void Poll()
    {
        Progress();
    }
    virtual void Flush(bool idle) = 0;
    virtual void FlushAll() = 0;
    virtual bool Pending() const = 0;
    virtual void Send(rank_t destination, const MCParticle &particle) = 0;
    virtual void AppendLocal(const std::vector<MCParticle> &particles) = 0;
    virtual void Detach(rank_t source, std::vector<MCParticle> &particles) = 0;
    virtual size_t LocalSize(rank_t source) const = 0;
    virtual const std::vector<rank_t> &Neighbors() const = 0;
    virtual void VisitLocal(const std::function<void(MCParticle &)> &visit) = 0;
    virtual void ShrinkBuffers()
    {
    }
    virtual size_t MemoryBytes() const = 0;
    virtual size_t Transfers() const = 0;
    virtual rank_t Rank() const
    {
        return 0;
    }
    virtual rank_t Size() const
    {
        return 1;
    }
#ifdef STORM_WITH_MPI
    virtual MPI_Comm Communicator() const
    {
        return MPI_COMM_NULL;
    }
#endif

    enum class Reduction
    {
        Sum,
        Min,
        Max
    };
    // Diagnostics and collective votes share the engine's communicator. Serial
    // engines reduce locally even when STORM was built with MPI support.
    template<class Value>
    void Reduce(const Value *local, Value *result, size_t count, Reduction operation, bool all = false) const
    {
#ifdef STORM_WITH_MPI
        if(Communicator() != MPI_COMM_NULL)
        {
            MPI_Datatype type;
            if constexpr(std::is_same_v<Value, double>)
            {
                type = MPI_DOUBLE;
            }
            else if constexpr(std::is_same_v<Value, int>)
            {
                type = MPI_INT;
            }
            else
            {
                static_assert(std::is_same_v<Value, unsigned long long>);
                type = MPI_UNSIGNED_LONG_LONG;
            }
            MPI_Op op = operation == Reduction::Sum ? MPI_SUM : operation == Reduction::Min ? MPI_MIN
                                                                                            : MPI_MAX;
            const void *input = local == result ? MPI_IN_PLACE : static_cast<const void *>(local);
            if(all)
            {
                MPI_Allreduce(input, result, count, type, op, Communicator());
            }
            else
            {
                MPI_Reduce(local == result && Rank() != 0 ? local : input,
                           result, count, type, op, 0, Communicator());
            }
            return;
        }
#endif
        if(local != result)
        {
            std::copy_n(local, count, result);
        }
    }
    bool Agree(bool local) const
    {
        int vote = local;
        Reduce(&vote, &vote, 1, Reduction::Min, true);
        return vote != 0;
    }
    void Barrier() const
    {
#ifdef STORM_WITH_MPI
        if(Communicator() != MPI_COMM_NULL)
        {
            MPI_Barrier(Communicator());
        }
#endif
    }

    // Nonblocking diagnostic snapshots. Counter meanings belong to the caller.
    void ResetCounterSnapshots(void)
    {
        counterSnapshots.clear();
        lastCounterPublish = 0;
    }
    void PublishCounters(const unsigned long long *values, size_t count, double elapsed)
    {
        if(counterSnapshots.empty())
        {
            counterSnapshots.assign(Size(), std::vector<unsigned long long>(count));
        }
        if(Rank() == 0)
        {
            std::copy_n(values, count, counterSnapshots[0].begin());
#ifdef STORM_WITH_MPI
            if(Communicator() != MPI_COMM_NULL)
            {
                int ready = 0;
                MPI_Status status;
                for(;;)
                {
                    MPI_Iprobe(MPI_ANY_SOURCE, 9941, Communicator(), &ready, &status);
                    if(!ready)
                    {
                        break;
                    }
                    MPI_Recv(counterSnapshots[status.MPI_SOURCE].data(), count,
                             MPI_UNSIGNED_LONG_LONG, status.MPI_SOURCE, 9941, Communicator(), MPI_STATUS_IGNORE);
                }
            }
#endif
        }
#ifdef STORM_WITH_MPI
        else if(elapsed - lastCounterPublish >= 5.0)
        {
            int done = 1;
            if(counterRequest != MPI_REQUEST_NULL)
            {
                MPI_Test(&counterRequest, &done, MPI_STATUS_IGNORE);
            }
            if(done)
            {
                outgoingCounters.assign(values, values + count);
                MPI_Isend(outgoingCounters.data(), count, MPI_UNSIGNED_LONG_LONG,
                          0, 9941, Communicator(), &counterRequest);
                lastCounterPublish = elapsed;
            }
        }
#endif
    }
    std::vector<unsigned long long> CounterTotals() const
    {
        if(counterSnapshots.empty())
        {
            return {};
        }
        std::vector<unsigned long long> totals(counterSnapshots[0].size(), 0);
        for(const std::vector<unsigned long long> &snapshot : counterSnapshots)
        {
            for(size_t i = 0; i < totals.size(); ++i)
            {
                totals[i] += snapshot[i];
            }
        }
        return totals;
    }
    void FinishCounters()
    {
#ifdef STORM_WITH_MPI
        if(counterRequest != MPI_REQUEST_NULL)
        {
            MPI_Wait(&counterRequest, MPI_STATUS_IGNORE);
        }
#endif
    }

private:
    std::vector<std::vector<unsigned long long>> counterSnapshots;
    double lastCounterPublish = 0;
#ifdef STORM_WITH_MPI
    std::vector<unsigned long long> outgoingCounters;
    MPI_Request counterRequest = MPI_REQUEST_NULL;
#endif
};

} // namespace STORM
#endif
