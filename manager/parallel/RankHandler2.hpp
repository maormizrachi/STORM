#ifndef MONTECARLO_RANK_HANDLER2_HPP
#define MONTECARLO_RANK_HANDLER2_HPP

#include <cassert>
#include <atomic>

#ifdef STORM_WITH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <mpi_utils/mpi_commands.hpp>
#include <rma/DistributedMutex.hpp>
#include "../../particle/Particle.hpp"
#include "ReallocationAgent.hpp"
#include <rma/RMAFactory.hpp>
#include "../../StormError.hpp"
#ifdef MEMORY_DEBUG
#include "misc/memory_debug.hpp"
#endif
#include "../MonteCarloConfig.hpp"

namespace STORM {

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

    void SynchronizeLocalQueueForRead(void) const;
    void SynchronizeLocalParticlesForRead(void) const;
    void PublishLocalQueueCounter(void);
    static uint64_t AuditParticleHash(const MCParticle &particle);
    void AuditParticles(const char *event, rank_t sourceRank, rank_t destinationRank,
                        uint64_t firstTicket, const MCParticle *particleData,
                        size_t count, size_t ringSize) const;
    uint64_t audit_epoch;

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
    size_t minimalBuffSize;

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
    particles(nullptr), queue_storage{0, 0}, audit_epoch(0),
    head(queue_storage[HEAD_INDEX]), tail(queue_storage[TAIL_INDEX]),
    reallocationAgent(reallocationAgent), rdma_type(rdma_type), destroyed(false),
    group_world(MPI_GROUP_NULL), group_internal(MPI_GROUP_NULL),
    minimalBuffSize(minimalBuffSize)
{

    assert(private_comm != MPI_COMM_NULL);

    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);
    MPI_Comm_rank(this->comm, &this->rank_internal);
    MPI_Comm_size(this->comm, &this->size_internal);

    assert(this->size_internal == 2 or this->size_internal == 1);
    assert(this->rank_internal == 0 or this->rank_internal == 1);

    this->requestedFactor = 1;

    if(this->size_internal > 1)
    {
        this->other_rank = 1 - this->rank_internal;

        this->particles_agent = RMAFactory::Create<MCParticle>(this->rdma_type, this->buffsize, this->comm);
        this->lengths_agent = RMAFactory::CreateOver<uint64_t>(this->rdma_type, this->queue_storage, 2, this->comm);

        this->particles = this->particles_agent->GetLocalPointer();

        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0, this->rdma_type);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1, this->rdma_type);
        this->localListMutex = (this->rank_internal == 0)? rank0Mutex : rank1Mutex;
        this->remoteListMutex = (this->rank_internal == 0)? rank1Mutex : rank0Mutex;
        this->Reset();
        MPI_Barrier(this->comm);
    }
    else
    {
        this->other_rank = 0;
        this->particles = new MCParticle[this->buffsize];
        this->Reset();
    }

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
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::Reset(void)
{
    this->audit_epoch++;
    this->head = 0;
    this->tail = 0;
    this->PublishLocalQueueCounter();
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
void RankHandler2<T, Grid>::SynchronizeLocalQueueForRead(void) const
{
    if(this->size_internal > 1 and this->lengths_agent)
    {
        this->lengths_agent->SyncLocal();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::SynchronizeLocalParticlesForRead(void) const
{
    if(this->size_internal > 1 and this->particles_agent)
    {
        this->particles_agent->SyncLocal();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::PublishLocalQueueCounter(void)
{
    std::atomic_thread_fence(std::memory_order_release);
    if(this->size_internal > 1 and this->lengths_agent)
    {
        this->lengths_agent->SyncLocal();
    }
}

template<typename T, typename Grid>
uint64_t RankHandler2<T, Grid>::AuditParticleHash(const MCParticle &particle)
{
    uint64_t hash = 1469598103934665603ULL;
    auto append = [&hash](const auto &value)
    {
        using value_t = typename std::decay<decltype(value)>::type;
        static_assert(std::is_trivially_copyable<value_t>::value,
                      "RMA audit fields must be trivially copyable");
        const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&value);
        for(size_t i = 0; i < sizeof(value_t); i++)
        {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= 1099511628211ULL;
        }
    };

    append(particle.rank);
    append(particle.id);
    append(particle.cellID);
    append(particle.sourceCellID);
    append(particle.location.x);
    append(particle.location.y);
    append(particle.location.z);
    append(particle.velocity.x);
    append(particle.velocity.y);
    append(particle.velocity.z);
    append(particle.cellIndex);
    append(particle.timeLeft);
    append(particle.frequency);
    append(particle.weight);
    append(particle.initialWeight);
    append(particle.steps);
    append(particle.on_track);
    append(particle.sent);
#ifdef STORM_DEBUG
    append(particle.nextRank);
    append(particle.sentByRank);
    append(particle.removedFromRank);
#endif
    return hash;
}

template<typename T, typename Grid>
void RankHandler2<T, Grid>::AuditParticles(const char *event, rank_t sourceRank,
                                            rank_t destinationRank, uint64_t firstTicket,
                                            const MCParticle *particleData, size_t count,
                                            size_t ringSize) const
{
    const char *enabled = std::getenv("STORM_RMA_AUDIT");
    if(not enabled or enabled[0] == '\0' or std::strcmp(enabled, "0") == 0 or count == 0)
    {
        return;
    }

    auto matchesFilter = [](const char *name, long long value)
    {
        const char *text = std::getenv(name);
        if(not text or text[0] == '\0')
        {
            return true;
        }
        char *end = nullptr;
        long long filter = std::strtoll(text, &end, 10);
        return end == text or filter < 0 or filter == value;
    };

    if(not matchesFilter("STORM_RMA_AUDIT_SRC", sourceRank) or
       not matchesFilter("STORM_RMA_AUDIT_DST", destinationRank) or
       not matchesFilter("STORM_RMA_AUDIT_EPOCH", static_cast<long long>(this->audit_epoch)))
    {
        return;
    }

    static uint64_t emittedLines = 0;
    uint64_t maxLines = 2000000;
    if(const char *text = std::getenv("STORM_RMA_AUDIT_MAX_LINES"))
    {
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(text, &end, 10);
        if(end != text)
        {
            maxLines = static_cast<uint64_t>(parsed);
        }
    }

    static std::ofstream stream;
    if(not stream.is_open())
    {
        const char *prefix = std::getenv("STORM_RMA_AUDIT_PREFIX");
        std::ostringstream filename;
        filename << ((prefix and prefix[0] != '\0') ? prefix : "storm_rma_audit")
                 << "_rank" << this->rank_world << ".csv";
        stream.open(filename.str(), std::ios::out | std::ios::app);
        if(not stream)
        {
            throw std::runtime_error("RankHandler2: failed to open RMA audit log " + filename.str());
        }
        if(stream.tellp() == std::streampos(0))
        {
            stream << "event,epoch,logger_rank,src,dst,ticket,slot,hash,particle_origin,id,cell_index,x,y,z,weight,frequency,time_left\n";
        }
        stream << std::setprecision(17);
    }

    for(size_t i = 0; i < count; i++)
    {
        if(maxLines != 0 and emittedLines >= maxLines)
        {
            break;
        }
        const MCParticle &particle = particleData[i];
        uint64_t ticket = firstTicket + static_cast<uint64_t>(i);
        size_t slot = ringSize == 0 ? 0 : static_cast<size_t>(ticket % ringSize);
        stream << event << ',' << this->audit_epoch << ',' << this->rank_world << ','
               << sourceRank << ',' << destinationRank << ',' << ticket << ',' << slot << ','
               << std::hex << AuditParticleHash(particle) << std::dec << ','
               << particle.rank << ',' << particle.id << ',' << particle.cellIndex << ','
               << particle.location.x << ',' << particle.location.y << ',' << particle.location.z << ','
               << particle.weight << ',' << particle.frequency << ',' << particle.timeLeft << '\n';
        emittedLines++;
    }
    stream.flush();
}

template<typename T, typename Grid>
size_t RankHandler2<T, Grid>::LocalSize(void) const
{
    this->SynchronizeLocalQueueForRead();
    uint64_t localHead = this->head;
    uint64_t localTail = this->tail;
    if(localTail < localHead or localTail - localHead > this->buffsize)
    {
        STORMError eo("RankHandler2::LocalSize: invalid SPSC queue counters");
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
    this->SynchronizeLocalQueueForRead();
    uint64_t localHead = this->head;
    uint64_t localTail = this->tail;
    if(localTail < localHead or localTail - localHead > this->buffsize)
    {
        STORMError eo("RankHandler2::DetachLocalParticles: invalid SPSC queue counters");
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

    this->SynchronizeLocalParticlesForRead();
    size_t start = static_cast<size_t>(localHead % this->buffsize);
    size_t first = std::min(count, this->buffsize - start);
    std::memcpy(result.data(), this->particles + start, first * sizeof(MCParticle));
    if(first < count)
    {
        std::memcpy(result.data() + first, this->particles, (count - first) * sizeof(MCParticle));
    }
    this->AuditParticles("RECV", this->peer_rank_world, this->rank_world,
                         localHead, result.data(), count, this->buffsize);
    this->head = localTail;
    this->PublishLocalQueueCounter();
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
    std::atomic_thread_fence(std::memory_order_release);
    this->tail = localTail + particlesNum;
    this->PublishLocalQueueCounter();
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
#ifdef MEMORY_DEBUG
    memory_debug::check_system_memory("RankHandler2::Reallocate");
#endif


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
        STORMError eo("RankHandler2: can not shrink memory when there are particles");
        eo.addEntry("My Rank", this->rank_world);
        eo.addEntry("Peer Rank", this->peer_rank_world);
        eo.addEntry("Particles", localCount);
        eo.addEntry("Old Buffer Size", oldBuffSize);
        eo.addEntry("New Buffer Size", newBuffSize);
        throw eo;
    }

    std::vector<unsigned char> activeParticles;
    if(not noParticles)
    {
        activeParticles.resize(localCount * sizeof(MCParticle));
        this->ForEachLocalParticle([&activeParticles](const MCParticle &particle, size_t index)
        {
            std::memcpy(activeParticles.data() + index * sizeof(MCParticle), &particle, sizeof(MCParticle));
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

    this->audit_epoch++;
    this->PublishLocalQueueCounter();
    this->requestedFactor = 1;
}

template<typename T, typename Grid>
bool RankHandler2<T, Grid>::UsesAsyncReallocation(void) const
{
    if(this->size_internal <= 1)
    {
        return false;
    }
    return this->particles_agent->SupportsAsyncReallocation() and
           this->lengths_agent->SupportsAsyncReallocation();
}

template<typename T, typename Grid>
ReallocationMetadata RankHandler2<T, Grid>::LocalReallocate(double factor)
{
    if(not this->UsesAsyncReallocation())
    {
        throw std::runtime_error("RankHandler2::LocalReallocate is only supported for native RDMA backends");
    }

#ifdef MEMORY_DEBUG
    memory_debug::check_system_memory("RankHandler2::LocalReallocate");
#endif


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
        std::vector<unsigned char> activeParticles;
        if(localCount > 0)
        {
            activeParticles.resize(localCount * sizeof(MCParticle));
            this->ForEachLocalParticle([&activeParticles](const MCParticle &particle, size_t index)
            {
                std::memcpy(activeParticles.data() + index * sizeof(MCParticle), &particle, sizeof(MCParticle));
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
        this->audit_epoch++;
        this->PublishLocalQueueCounter();

        this->UnlockSelfBuffer();
        locked = false;


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
    this->audit_epoch++;
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


    if(this->size_internal > 1)
    {
        // The peer may resize/replace its receive queue asynchronously. Hold
        // the peer-owned mutex for the complete remote queue transaction so a
        // reallocation cannot invalidate the remote particle window between
        // reading its counters and publishing the new tail. Drop the lock
        // while asking the peer to reallocate; its callback acquires this
        // same mutex on its local side.
        this->remoteListMutex->Lock();

        uint64_t remoteHead = 0;
        uint64_t remoteTail = 0;
        auto getRemoteCounters = [&]()
        {
            // Tail is written only by this producer. Read it separately from head
            // so the pair cannot be a torn 16-byte snapshot while the consumer
            // advances head.
            this->lengths_agent->Get(&remoteTail, 1, this->other_rank, TAIL_INDEX);
            this->lengths_agent->Get(&remoteHead, 1, this->other_rank, HEAD_INDEX);
            if(remoteTail < remoteHead or remoteTail - remoteHead > this->peer_buffsize)
            {
                STORMError eo("RankHandler2::TransferParticles: remote queue counters are out of bounds");
                eo.addEntry("Remote Head", static_cast<size_t>(remoteHead));
                eo.addEntry("Remote Tail", static_cast<size_t>(remoteTail));
                eo.addEntry("Peer Buffer Size", this->peer_buffsize);
                eo.addEntry("My Rank", this->rank_world);
                eo.addEntry("Peer Rank", this->peer_rank_world);
                throw eo;
            }
        };

        getRemoteCounters();

        size_t remoteCount = static_cast<size_t>(remoteTail - remoteHead);
        while(Np > this->peer_buffsize - remoteCount)
        {
            if(Np > std::numeric_limits<size_t>::max() - remoteCount)
            {
                STORMError eo("RankHandler2::TransferParticles: transfer size overflow");
                eo.addEntry("Remote Count", remoteCount);
                eo.addEntry("Particles", Np);
                eo.addEntry("My Rank", this->rank_world);
                eo.addEntry("Peer Rank", this->peer_rank_world);
                throw eo;
            }

            size_t requiredSize = remoteCount + Np;
            size_t denominator = std::max<size_t>(1, this->peer_buffsize);
            this->requestedFactor = static_cast<double>(requiredSize) /
                                    static_cast<double>(denominator) * 1.5;


            if(this->UsesAsyncReallocation())
            {
                this->particles_agent->QuiesceTarget(this->other_rank);
                this->lengths_agent->QuiesceTarget(this->other_rank);
                this->remoteListMutex->Unlock();
                this->reallocationAgent->RequestReallocationAsync(this->peer_rank_world, this->requestedFactor);
                this->requestedFactor = 1;
                return false;
            }

            this->remoteListMutex->Unlock();
            this->reallocationAgent->RequestReallocation(this->peer_rank_world);
            this->remoteListMutex->Lock();

            getRemoteCounters();
            remoteCount = static_cast<size_t>(remoteTail - remoteHead);
        }


        #ifdef STORM_DEBUG
        for(size_t i = 0; i < Np; i++)
        {
            const MCParticle &particle = particles[i];
            if(particle.nextRank != this->peer_rank_world)
            {
                STORMError eo("RankHandler2::TransferParticles: particle will not be sent to the expected rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", this->rank_world);
                eo.addEntry("Expected Rank", particle.nextRank);
                eo.addEntry("Next Rank", this->peer_rank_world);
                throw eo;
            }
        }
        #endif // STORM_DEBUG

        if(Np > std::numeric_limits<uint64_t>::max() - remoteTail)
        {
            STORMError eo("RankHandler2::TransferParticles: tail counter overflow");
            eo.addEntry("Remote Tail", static_cast<size_t>(remoteTail));
            eo.addEntry("Particles", Np);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }

        size_t start = static_cast<size_t>(remoteTail % this->peer_buffsize);
        size_t first = std::min(Np, this->peer_buffsize - start);
        this->particles_agent->Put(particles, first, this->other_rank, start, false, source_lkey);
        if(first < Np)
        {
            this->particles_agent->Put(particles + first, Np - first, this->other_rank, 0,
                                       false, source_lkey);
        }

        // This queue is SPSC: this rank is the sole producer of the peer's tail.
        // QuiesceTarget confirms all particle writes are visible at the remote.
        // The flushed Put publishes the new tail with a signaled write + drain-all,
        // which guarantees tail visibility and progresses pending MPI traffic.
        this->particles_agent->QuiesceTarget(this->other_rank);

        uint64_t newTail = remoteTail + static_cast<uint64_t>(Np);
        this->lengths_agent->Put(&newTail, 1, this->other_rank, TAIL_INDEX, true);
        this->AuditParticles("SEND", this->rank_world, this->peer_rank_world,
                             remoteTail, particles, Np, this->peer_buffsize);
        this->remoteListMutex->Unlock();
    }
    else
    {
        this->AppendLocalParticles(particles, Np);
    }
    return true;
}

} // namespace STORM

#endif // STORM_WITH_MPI
#endif // MONTECARLO_RANK_HANDLER2_HPP
