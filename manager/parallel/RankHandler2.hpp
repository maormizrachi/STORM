#ifndef MONTECARLO_RANK_HANDLER2_HPP
#define MONTECARLO_RANK_HANDLER2_HPP

#include <cassert>

#ifdef RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "utils/rma-helpers/DistributedMutex.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/manager/ReallocationAgent.hpp"
#include "utils/rma/RMAFactory.hpp"
#include "misc/memory_debug.hpp"
#include "misc/universal_error.hpp"
#include "monte/manager/MonteCarloConfig.hpp"

template<typename T, typename Grid>
class RankHandler2
{
public:
    using index_t = uint32_t;
    using MCParticle = MonteCarloParticle<T, Grid>;

    RankHandler2(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm,
                 std::shared_ptr<ReallocationAgent> &reallocationAgent,
                 RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA,
                 size_t minimalBuffSize = 50);

    ~RankHandler2();

    bool TransferParticles(const std::vector<MCParticle> &particles);

    bool TransferParticles(const MCParticle *particles, size_t particlesNum, uint32_t source_lkey = 0);

    size_t LocalSize(void) const;

    bool LocalEmpty(void) const;

    void DetachLocalParticles(std::vector<MCParticle> &result);

    void AppendLocalParticles(const MCParticle *particles, size_t particlesNum);

    template<typename Writer>
    void AppendLocalParticles(size_t particlesNum, const Writer &writer);

    MCParticle &LocalParticleAt(size_t logicalIndex);

    const MCParticle &LocalParticleAt(size_t logicalIndex) const;

    template<typename Func>
    void ForEachLocalParticle(const Func &func);

    void Reset(void);

    void Destroy(void);

    MPI_Comm comm_world, comm;
    rank_t rank_world, rank_internal;
    rank_t size_world, size_internal;
    rank_t other_rank, peer_rank_world;

    size_t buffsize, peer_buffsize;
    MCParticle *particles;

private:
    static constexpr size_t HEAD_INDEX = 0;
    static constexpr size_t TAIL_INDEX = 1;
    uint64_t queue_storage[2]; // [head, tail], monotonic SPSC counters

public:
    volatile uint64_t &head;
    volatile uint64_t &tail;

    std::shared_ptr<ReallocationAgent> &reallocationAgent;

    void Reallocate(double factor);

    ReallocationMetadata LocalReallocate(double factor);

    void UpdatePeerRemoteInfo(const ReallocationMetadata &metadata);

    bool UsesAsyncReallocation(void) const;

    typename RemoteMemoryAgent<MCParticle>::SourceRegistration RegisterSendSource(const MCParticle *particles, size_t particlesNum);

    void DeregisterSendSource(uint64_t handle);

    inline void LockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localListMutex->Lock();
        }
    }

    inline void UnlockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localListMutex->Unlock();
        }
    }

    double requestedFactor;
    #ifdef TIMING
    double reallocationTime;
    size_t reallocationsThisStep;
    size_t reallocationsTotal;
    size_t peakBufferUsage;
    size_t transferCallsThisStep;
    size_t contiguousParticlePutsThisStep;
    size_t contiguousParticlesThisStep;
    size_t scatterParticlePutsThisStep;
    size_t scatterParticlesThisStep;
    size_t transferCallsWithContiguousAllocationThisStep;
    size_t transferCallsWithoutContiguousAllocationThisStep;
    size_t transferReallocationRequestsThisStep;
    size_t transferCallsWithReallocationThisStep;
    size_t remoteLockCallsThisStep;
    double transferTotalTimeThisStep;
    double transferLockWaitTimeThisStep;
    double transferReallocationWaitTimeThisStep;
    double transferAvailReserveTimeThisStep;
    double transferAvailIndexGetTimeThisStep;
    double transferParticlePutTimeThisStep;
    double transferTHLengthGetTimeThisStep;
    double transferTHPutTimeThisStep;
    double transferTHLengthPublishTimeThisStep;
    double transferAVLengthFlushTimeThisStep;
    double transferUnlockTimeThisStep;
    #endif // TIMING
    size_t minimalBuffSize;
    #ifdef TIMING
    double constructionRmaTime;
    double constructionMutexTime;
    double constructionResetTime;
    double constructionPeerInfoTime;
    double constructionTotalTime;
    #endif // TIMING

private:
    std::unique_ptr<RemoteMemoryAgent<MCParticle>> particles_agent;
    std::unique_ptr<RemoteMemoryAgent<uint64_t>> lengths_agent;
    std::shared_ptr<DistributedMutex> localListMutex;
    std::shared_ptr<DistributedMutex> remoteListMutex;
    RDMA_Type rdma_type;
    bool destroyed;
    MPI_Group group_world, group_internal;
};

template<typename T, typename Grid>
RankHandler2<T, Grid>::RankHandler2(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm,
                                    std::shared_ptr<ReallocationAgent> &reallocationAgent,
                                    RDMA_Type rdma_type, size_t minimalBuffSize):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize),
    particles(nullptr), queue_storage{0, 0}, head(queue_storage[HEAD_INDEX]), tail(queue_storage[TAIL_INDEX]),
    reallocationAgent(reallocationAgent), rdma_type(rdma_type), destroyed(false),
    group_world(MPI_GROUP_NULL), group_internal(MPI_GROUP_NULL),
    minimalBuffSize(minimalBuffSize)
{
    #ifdef TIMING
    auto constructorStart = std::chrono::high_resolution_clock::now();
    auto secondsSince = [](const std::chrono::high_resolution_clock::time_point &start)
    {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    };

    this->constructionRmaTime = 0;
    this->constructionMutexTime = 0;
    this->constructionResetTime = 0;
    this->constructionPeerInfoTime = 0;
    this->constructionTotalTime = 0;
    #endif // TIMING

    assert(private_comm != MPI_COMM_NULL);

    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);
    MPI_Comm_rank(this->comm, &this->rank_internal);
    MPI_Comm_size(this->comm, &this->size_internal);

    assert(this->size_internal == 2 or this->size_internal == 1);
    assert(this->rank_internal == 0 or this->rank_internal == 1);

    this->requestedFactor = 1;
    #ifdef TIMING
    this->reallocationTime = 0;
    this->reallocationsThisStep = 0;
    this->reallocationsTotal = 0;
    this->peakBufferUsage = 0;
    this->transferCallsThisStep = 0;
    this->contiguousParticlePutsThisStep = 0;
    this->contiguousParticlesThisStep = 0;
    this->scatterParticlePutsThisStep = 0;
    this->scatterParticlesThisStep = 0;
    this->transferCallsWithContiguousAllocationThisStep = 0;
    this->transferCallsWithoutContiguousAllocationThisStep = 0;
    this->transferReallocationRequestsThisStep = 0;
    this->transferCallsWithReallocationThisStep = 0;
    this->remoteLockCallsThisStep = 0;
    this->transferTotalTimeThisStep = 0;
    this->transferLockWaitTimeThisStep = 0;
    this->transferReallocationWaitTimeThisStep = 0;
    this->transferAvailReserveTimeThisStep = 0;
    this->transferAvailIndexGetTimeThisStep = 0;
    this->transferParticlePutTimeThisStep = 0;
    this->transferTHLengthGetTimeThisStep = 0;
    this->transferTHPutTimeThisStep = 0;
    this->transferTHLengthPublishTimeThisStep = 0;
    this->transferAVLengthFlushTimeThisStep = 0;
    this->transferUnlockTimeThisStep = 0;
    #endif // TIMING

    if(this->size_internal > 1)
    {
        this->other_rank = 1 - this->rank_internal;

        #ifdef TIMING
        auto sectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        this->particles_agent = RMAFactory::Create<MCParticle>(this->rdma_type, this->buffsize, this->comm);
        this->lengths_agent = RMAFactory::CreateOver<uint64_t>(this->rdma_type, this->queue_storage, 2, this->comm);
        #ifdef TIMING
        this->constructionRmaTime = secondsSince(sectionStart);
        #endif // TIMING

        this->particles = this->particles_agent->GetLocalPointer();

        #ifdef TIMING
        sectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0, this->rdma_type);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1, this->rdma_type);
        this->localListMutex = (this->rank_internal == 0)? rank0Mutex : rank1Mutex;
        this->remoteListMutex = (this->rank_internal == 0)? rank1Mutex : rank0Mutex;
        #ifdef TIMING
        this->constructionMutexTime = secondsSince(sectionStart);

        sectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        this->Reset();
        MPI_Barrier(this->comm);
        #ifdef TIMING
        this->constructionResetTime = secondsSince(sectionStart);
        #endif // TIMING
    }
    else
    {
        this->other_rank = 0;
        #ifdef TIMING
        auto sectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        this->particles = new MCParticle[this->buffsize];
        this->head = 0;
        this->tail = 0;
        #ifdef TIMING
        this->constructionRmaTime = secondsSince(sectionStart);
        #endif // TIMING
    }

    #ifdef TIMING
    auto peerInfoStart = std::chrono::high_resolution_clock::now();
    #endif // TIMING
    if(this->size_internal > 1)
    {
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0,
                     &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0,
                     this->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&this->rank_world, 1, MPI_INT, this->other_rank, 0,
                     &this->peer_rank_world, 1, MPI_INT, this->other_rank, 0,
                     this->comm, MPI_STATUS_IGNORE);

        MPI_Comm_group(this->comm_world, &this->group_world);
        MPI_Comm_group(this->comm, &this->group_internal);
    }
    else
    {
        this->peer_buffsize = this->buffsize;
        this->peer_rank_world = this->rank_world;
    }

    MPI_Barrier(this->comm);
    #ifdef TIMING
    this->constructionPeerInfoTime = secondsSince(peerInfoStart);
    this->constructionTotalTime = secondsSince(constructorStart);
    #endif // TIMING
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::Reset(void)
{
    this->head = 0;
    this->tail = 0;
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }

    if(this->size_internal > 1)
    {
        this->particles_agent->Free();
        this->lengths_agent->Free();
        MPI_Group_free(&this->group_world);
        MPI_Group_free(&this->group_internal);
        DistributedMutex *mutex1 = (this->rank_internal == 0)? this->localListMutex.get() : this->remoteListMutex.get();
        DistributedMutex *mutex2 = (this->rank_internal == 1)? this->localListMutex.get() : this->remoteListMutex.get();
        mutex1->Destroy();
        mutex2->Destroy();
    }
    else
    {
        delete[] this->particles;
    }
    this->destroyed = true;
}

template<typename T, typename Grid>
RankHandler2<T, Grid>::~RankHandler2()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

template<typename T, typename Grid>
size_t RankHandler2<T, Grid>::LocalSize(void) const
{
    uint64_t localHead = this->head;
    uint64_t localTail = this->tail;
    if(localTail < localHead or localTail - localHead > this->buffsize)
    {
        UniversalError eo("RankHandler2::LocalSize: invalid SPSC queue counters");
        eo.addEntry("My Rank", this->rank_world);
        eo.addEntry("Peer Rank", this->peer_rank_world);
        eo.addEntry("Head", static_cast<size_t>(localHead));
        eo.addEntry("Tail", static_cast<size_t>(localTail));
        eo.addEntry("Buffer Size", this->buffsize);
        throw eo;
    }
    return static_cast<size_t>(localTail - localHead);
}

template<typename T, typename Grid>
bool RankHandler2<T, Grid>::LocalEmpty(void) const
{
    return this->LocalSize() == 0;
}

template<typename T, typename Grid>
typename RankHandler2<T, Grid>::MCParticle &RankHandler2<T, Grid>::LocalParticleAt(size_t logicalIndex)
{
    assert(logicalIndex < this->LocalSize());
    return this->particles[(static_cast<size_t>(this->head) + logicalIndex) % this->buffsize];
}

template<typename T, typename Grid>
const typename RankHandler2<T, Grid>::MCParticle &RankHandler2<T, Grid>::LocalParticleAt(size_t logicalIndex) const
{
    assert(logicalIndex < this->LocalSize());
    return this->particles[(static_cast<size_t>(this->head) + logicalIndex) % this->buffsize];
}

template<typename T, typename Grid>
template<typename Func>
void RankHandler2<T, Grid>::ForEachLocalParticle(const Func &func)
{
    size_t count = this->LocalSize();
    size_t start = static_cast<size_t>(this->head % this->buffsize);
    size_t first = std::min(count, this->buffsize - start);
    for(size_t i = 0; i < first; i++)
    {
        func(this->particles[start + i], i);
    }
    for(size_t i = first; i < count; i++)
    {
        func(this->particles[i - first], i);
    }
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::DetachLocalParticles(std::vector<MCParticle> &result)
{
    uint64_t localHead = this->head;
    uint64_t localTail = this->tail;
    if(localTail < localHead or localTail - localHead > this->buffsize)
    {
        UniversalError eo("RankHandler2::DetachLocalParticles: invalid SPSC queue counters");
        eo.addEntry("My Rank", this->rank_world);
        eo.addEntry("Peer Rank", this->peer_rank_world);
        eo.addEntry("Head", static_cast<size_t>(localHead));
        eo.addEntry("Tail", static_cast<size_t>(localTail));
        eo.addEntry("Buffer Size", this->buffsize);
        throw eo;
    }

    size_t count = static_cast<size_t>(localTail - localHead);
    result.resize(count);
    if(count == 0)
    {
        return;
    }

    size_t start = static_cast<size_t>(localHead % this->buffsize);
    size_t first = std::min(count, this->buffsize - start);
    std::memcpy(result.data(), this->particles + start, first * sizeof(MCParticle));
    if(first < count)
    {
        std::memcpy(result.data() + first, this->particles, (count - first) * sizeof(MCParticle));
    }
    this->head = localTail;
}

template<typename T, typename Grid>
template<typename Writer>
void RankHandler2<T, Grid>::AppendLocalParticles(size_t particlesNum, const Writer &writer)
{
    if(particlesNum == 0)
    {
        return;
    }

    size_t currentSize = this->LocalSize();
    if(currentSize + particlesNum > this->buffsize)
    {
        double factor = std::max<double>(
            1.5,
            static_cast<double>(currentSize + particlesNum) /
                static_cast<double>(std::max<size_t>(1, this->buffsize)) * 1.5);
        this->Reallocate(factor);
        currentSize = this->LocalSize();
        assert(currentSize + particlesNum <= this->buffsize);
    }

    uint64_t localTail = this->tail;
    size_t start = static_cast<size_t>(localTail % this->buffsize);
    size_t first = std::min(particlesNum, this->buffsize - start);
    for(size_t i = 0; i < first; i++)
    {
        writer(this->particles[start + i], i);
    }
    for(size_t i = first; i < particlesNum; i++)
    {
        writer(this->particles[i - first], i);
    }
    this->tail = localTail + particlesNum;
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::AppendLocalParticles(const MCParticle *particles, size_t particlesNum)
{
    this->AppendLocalParticles(particlesNum, [particles](MCParticle &destination, size_t index)
    {
        std::memcpy(&destination, particles + index, sizeof(MCParticle));
    });
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::Reallocate(double factor)
{
    memory_debug::check_system_memory("RankHandler2::Reallocate");

    #ifdef TIMING
    auto reallocationStart = std::chrono::high_resolution_clock::now();
    this->reallocationsThisStep++;
    this->reallocationsTotal++;
    #endif // TIMING

    if(this->size_internal > 1)
    {
        double requestedFactorSelf;
        MPI_Sendrecv(&this->requestedFactor, 1, MPI_DOUBLE, this->other_rank, 0,
                     &requestedFactorSelf, 1, MPI_DOUBLE, this->other_rank, 0,
                     this->comm, MPI_STATUS_IGNORE);
        factor = std::max(factor, requestedFactorSelf);
    }

    size_t oldBuffSize = this->buffsize;
    size_t newBuffSize = std::ceil(static_cast<double>(this->buffsize) * factor);
    newBuffSize = std::max<size_t>(newBuffSize, this->minimalBuffSize);

    size_t localCount = this->LocalSize();
    bool noParticles = (localCount == 0);
    if(newBuffSize < oldBuffSize and not noParticles)
    {
        UniversalError eo("RankHandler2: can not shrink memory when there are particles");
        eo.addEntry("My Rank", this->rank_world);
        eo.addEntry("Peer Rank", this->peer_rank_world);
        eo.addEntry("Particles", localCount);
        eo.addEntry("Old Buffer Size", oldBuffSize);
        eo.addEntry("New Buffer Size", newBuffSize);
        throw eo;
    }

    std::vector<MCParticle> activeParticles;
    if(not noParticles)
    {
        activeParticles.resize(localCount);
        this->ForEachLocalParticle([&activeParticles](const MCParticle &particle, size_t index)
        {
            std::memcpy(activeParticles.data() + index, &particle, sizeof(MCParticle));
        });
    }

    this->buffsize = newBuffSize;

    if(this->size_internal > 1)
    {
        int localNoParticles = noParticles ? 1 : 0;
        int peerNoParticles = 0;
        MPI_Sendrecv(&localNoParticles, 1, MPI_INT, this->other_rank, 0,
                     &peerNoParticles, 1, MPI_INT, this->other_rank, 0,
                     this->comm, MPI_STATUS_IGNORE);
        bool bothEmpty = (localNoParticles and peerNoParticles);

        if(bothEmpty)
        {
            this->particles_agent->Replace(this->buffsize);
        }
        else
        {
            this->particles_agent->Resize(this->buffsize);
        }

        this->particles = this->particles_agent->GetLocalPointer();
        if(noParticles)
        {
            this->head = 0;
            this->tail = 0;
        }
        else
        {
            std::memcpy(this->particles, activeParticles.data(), localCount * sizeof(MCParticle));
            this->head = 0;
            this->tail = static_cast<uint64_t>(localCount);
        }

        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0,
                     &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0,
                     this->comm, MPI_STATUS_IGNORE);
    }
    else
    {
        if(noParticles)
        {
            delete[] this->particles;
            this->particles = new MCParticle[this->buffsize];
            this->head = 0;
            this->tail = 0;
        }
        else
        {
            MCParticle *new_particles = new MCParticle[this->buffsize];
            std::memcpy(new_particles, activeParticles.data(), localCount * sizeof(MCParticle));
            delete[] this->particles;
            this->particles = new_particles;
            this->head = 0;
            this->tail = static_cast<uint64_t>(localCount);
        }
        this->peer_buffsize = this->buffsize;
    }

    this->requestedFactor = 1;
    #ifdef TIMING
    this->reallocationTime += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - reallocationStart).count();
    #endif // TIMING
}

template<typename T, typename Grid>
bool RankHandler2<T, Grid>::UsesAsyncReallocation(void) const
{
    if(this->size_internal <= 1)
    {
        return false;
    }
    RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                             ? RMAFactory::ResolveAutoRDMA()
                             : this->rdma_type;
    return resolved == RDMA_Type::IBV_RDMA;
}

template<typename T, typename Grid>
ReallocationMetadata RankHandler2<T, Grid>::LocalReallocate(double factor)
{
    if(not this->UsesAsyncReallocation())
    {
        throw std::runtime_error("RankHandler2::LocalReallocate is only supported for the IBV RMA backend");
    }

    memory_debug::check_system_memory("RankHandler2::LocalReallocate");

    #ifdef TIMING
    auto reallocationStart = std::chrono::high_resolution_clock::now();
    this->reallocationsThisStep++;
    this->reallocationsTotal++;
    #endif // TIMING

    this->LockSelfBuffer();
    bool locked = true;
    try
    {
        factor = std::max(factor, 1.0);
        size_t newBuffSize = std::ceil(static_cast<double>(this->buffsize) * factor);
        newBuffSize = std::max<size_t>(newBuffSize, this->minimalBuffSize);
        if(newBuffSize <= this->buffsize)
        {
            newBuffSize = this->buffsize + 1;
        }

        size_t localCount = this->LocalSize();
        std::vector<MCParticle> activeParticles;
        if(localCount > 0)
        {
            activeParticles.resize(localCount);
            this->ForEachLocalParticle([&activeParticles](const MCParticle &particle, size_t index)
            {
                std::memcpy(activeParticles.data() + index, &particle, sizeof(MCParticle));
            });
        }

        this->buffsize = newBuffSize;
        RemoteBufferInfo particlesInfo = this->particles_agent->LocalResize(this->buffsize);
        RemoteBufferInfo lengthsInfo = this->lengths_agent->GetLocalRemoteInfo();
        this->particles = this->particles_agent->GetLocalPointer();
        if(localCount > 0)
        {
            std::memcpy(this->particles, activeParticles.data(), localCount * sizeof(MCParticle));
        }
        this->head = 0;
        this->tail = static_cast<uint64_t>(localCount);

        this->UnlockSelfBuffer();
        locked = false;

        #ifdef TIMING
        this->reallocationTime += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - reallocationStart).count();
        #endif // TIMING

        ReallocationMetadata metadata;
        metadata.particles = particlesInfo;
        metadata.lengths = lengthsInfo;
        metadata.new_buffsize = this->buffsize;
        return metadata;
    }
    catch(...)
    {
        if(locked)
        {
            this->UnlockSelfBuffer();
        }
        throw;
    }
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::UpdatePeerRemoteInfo(const ReallocationMetadata &metadata)
{
    if(this->size_internal <= 1)
    {
        return;
    }
    this->particles_agent->UpdateRemoteInfo(this->other_rank, metadata.particles);
    this->lengths_agent->UpdateRemoteInfo(this->other_rank, metadata.lengths);
    this->peer_buffsize = metadata.new_buffsize;
}

template<typename T, typename Grid>
typename RemoteMemoryAgent<typename RankHandler2<T, Grid>::MCParticle>::SourceRegistration
RankHandler2<T, Grid>::RegisterSendSource(const MCParticle *particles, size_t particlesNum)
{
    if(particles == nullptr or particlesNum == 0)
    {
        return {};
    }
    return this->particles_agent->RegisterExternalSource(particles, particlesNum * sizeof(MCParticle));
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::DeregisterSendSource(uint64_t handle)
{
    this->particles_agent->DeregisterExternalSource(handle);
}

template<typename T, typename Grid>
bool RankHandler2<T, Grid>::TransferParticles(const std::vector<MCParticle> &particles)
{
    return this->TransferParticles(particles.data(), particles.size(), 0);
}

template<typename T, typename Grid>
bool RankHandler2<T, Grid>::TransferParticles(const MCParticle *particles, size_t particlesNum, uint32_t source_lkey)
{
    size_t Np = particlesNum;
    if(particles == nullptr or particlesNum == 0)
    {
        return true;
    }

    #ifdef TIMING
    this->transferCallsThisStep++;
    auto transferTotalStart = std::chrono::high_resolution_clock::now();
    auto secondsSince = [](const std::chrono::high_resolution_clock::time_point &start)
    {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    };
    #endif // TIMING

    if(this->size_internal > 1)
    {
        uint64_t remoteCounters[2] = {0, 0};
        auto getRemoteCounters = [&]()
        {
            this->lengths_agent->Get(remoteCounters, 2, this->other_rank, 0);
            uint64_t remoteHead = remoteCounters[HEAD_INDEX];
            uint64_t remoteTail = remoteCounters[TAIL_INDEX];
            if(remoteTail < remoteHead or remoteTail - remoteHead > this->peer_buffsize)
            {
                UniversalError eo("RankHandler2::TransferParticles: remote queue counters are out of bounds");
                eo.addEntry("Remote Head", static_cast<size_t>(remoteHead));
                eo.addEntry("Remote Tail", static_cast<size_t>(remoteTail));
                eo.addEntry("Peer Buffer Size", this->peer_buffsize);
                eo.addEntry("My Rank", this->rank_world);
                eo.addEntry("Peer Rank", this->peer_rank_world);
                throw eo;
            }
        };

        #ifdef TIMING
        auto transferSectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        getRemoteCounters();
        #ifdef TIMING
        this->transferAvailReserveTimeThisStep += secondsSince(transferSectionStart);
        auto reallocationStart = std::chrono::high_resolution_clock::now();
        size_t reallocationRequestsForThisTransfer = 0;
        #endif // TIMING

        while(static_cast<size_t>(remoteCounters[TAIL_INDEX] - remoteCounters[HEAD_INDEX]) + Np > this->peer_buffsize)
        {
            size_t remoteCount = static_cast<size_t>(remoteCounters[TAIL_INDEX] - remoteCounters[HEAD_INDEX]);
            size_t requiredSize = remoteCount + Np;
            size_t denominator = std::max<size_t>(1, this->peer_buffsize);
            this->requestedFactor = static_cast<double>(requiredSize) /
                                    static_cast<double>(denominator) * 1.5;

            #ifdef TIMING
            this->transferReallocationRequestsThisStep++;
            reallocationRequestsForThisTransfer++;
            #endif // TIMING

            if(this->UsesAsyncReallocation())
            {
                this->reallocationAgent->RequestReallocationAsync(this->peer_rank_world, this->requestedFactor);
                this->requestedFactor = 1;
                #ifdef TIMING
                this->transferCallsWithReallocationThisStep++;
                this->transferTotalTimeThisStep += secondsSince(transferTotalStart);
                #endif // TIMING
                return false;
            }

            this->reallocationAgent->RequestReallocation(this->peer_rank_world);

            #ifdef TIMING
            transferSectionStart = std::chrono::high_resolution_clock::now();
            #endif // TIMING
            getRemoteCounters();
            #ifdef TIMING
            this->transferAvailReserveTimeThisStep += secondsSince(transferSectionStart);
            #endif // TIMING
        }

        #ifdef TIMING
        if(reallocationRequestsForThisTransfer > 0)
        {
            double reallocationSeconds = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - reallocationStart).count();
            this->reallocationTime += reallocationSeconds;
            this->transferReallocationWaitTimeThisStep += reallocationSeconds;
            this->transferCallsWithReallocationThisStep++;
        }
        #endif // TIMING

        #ifdef RDMONT_DEBUG
        for(size_t i = 0; i < Np; i++)
        {
            const MCParticle &particle = particles[i];
            if(particle.nextRank != this->peer_rank_world)
            {
                UniversalError eo("RankHandler2::TransferParticles: particle will not be sent to the expected rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", this->rank_world);
                eo.addEntry("Expected Rank", particle.nextRank);
                eo.addEntry("Next Rank", this->peer_rank_world);
                throw eo;
            }
        }
        #endif // RDMONT_DEBUG

        RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                                 ? RMAFactory::ResolveAutoRDMA()
                                 : this->rdma_type;
        bool is_mpi = (resolved == RDMA_Type::MPI_RMA);

        uint64_t remoteTail = remoteCounters[TAIL_INDEX];
        size_t start = static_cast<size_t>(remoteTail % this->peer_buffsize);
        size_t first = std::min(Np, this->peer_buffsize - start);
        #ifdef TIMING
        transferSectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        this->particles_agent->Put(particles, first, this->other_rank, start, is_mpi and first == Np, source_lkey);
        if(first < Np)
        {
            this->particles_agent->Put(particles + first, Np - first, this->other_rank, 0, is_mpi, source_lkey);
        }
        #ifdef TIMING
        this->transferParticlePutTimeThisStep += secondsSince(transferSectionStart);
        this->contiguousParticlePutsThisStep += (first < Np)? 2 : 1;
        this->contiguousParticlesThisStep += Np;
        this->transferCallsWithContiguousAllocationThisStep++;
        #endif // TIMING

        uint64_t tailIncrement = static_cast<uint64_t>(Np);
        #ifdef TIMING
        transferSectionStart = std::chrono::high_resolution_clock::now();
        #endif // TIMING
        uint64_t observedTail = this->lengths_agent->FetchAndAdd(tailIncrement, this->other_rank, TAIL_INDEX, true);
        #ifdef TIMING
        this->transferTHLengthPublishTimeThisStep += secondsSince(transferSectionStart);
        #endif // TIMING
        if(observedTail != remoteTail)
        {
            UniversalError eo("RankHandler2::TransferParticles: producer tail changed unexpectedly");
            eo.addEntry("Observed Tail", static_cast<size_t>(observedTail));
            eo.addEntry("Expected Tail", static_cast<size_t>(remoteTail));
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
    }
    else
    {
        this->AppendLocalParticles(particles, Np);
        #ifdef TIMING
        this->contiguousParticlePutsThisStep++;
        this->contiguousParticlesThisStep += Np;
        this->transferCallsWithContiguousAllocationThisStep++;
        this->peakBufferUsage = std::max(this->peakBufferUsage, this->LocalSize());
        #endif // TIMING
    }
    #ifdef TIMING
    this->transferTotalTimeThisStep += secondsSince(transferTotalStart);
    #endif // TIMING
    return true;
}

#endif // RICH_MPI
#endif // MONTECARLO_RANK_HANDLER2_HPP
