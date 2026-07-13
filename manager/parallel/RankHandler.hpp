#ifndef MONTECARLO_RANK_HANDLER_HPP
#define MONTECARLO_RANK_HANDLER_HPP

#include <cassert>

#ifdef STORM_WITH_MPI

#include <vector>
#include <memory>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <mpi.h>
#include <mpi_utils/mpi_commands.hpp>
#include <rma/DistributedMutex.hpp>
#include "../../particle/Particle.hpp"
#include "ReallocationAgent.hpp"
#include <rma/RMAFactory.hpp>
#ifdef MEMORY_DEBUG
#include "misc/memory_debug.hpp"
#endif
#include "../MonteCarloConfig.hpp"

#define MPI_INDEX_T MPI_UINT32_T

namespace STORM {

template<typename T, typename Grid>
class RankHandler
{
public:
    using index_t = uint32_t;
    using MCParticle = MonteCarloParticle<T, Grid>;

    RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent, RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA,
                size_t minimalBuffSize = 50);
    
    ~RankHandler();
    
    bool TransferParticles(const std::vector<MCParticle> &particles);

    void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);

    void Reset(void);

    void Destroy(void);

    MPI_Comm comm_world, comm;
    rank_t rank_world, rank_internal;
    rank_t size_world, size_internal;
    rank_t other_rank, peer_rank_world;

    size_t buffsize, peer_buffsize;
    MCParticle *particles;
    index_t *av;
    index_t *th;

private:
    int lengths_storage[2]; // [0] = av_length, [1] = th_length
public:
    volatile int &av_length;
    volatile int &th_length;

    std::shared_ptr<ReallocationAgent> &reallocationAgent;
    
    void Reallocate(double factor);

    ReallocationMetadata LocalReallocate(double factor);

    void UpdatePeerRemoteInfo(const ReallocationMetadata &metadata);

    bool UsesAsyncReallocation(void) const;

    // todo: necessary?
    inline void LockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localTHMutex->Lock();
        }
    }

    inline void UnlockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localTHMutex->Unlock();
        }
    }

    inline void MakeProgress(void)
    {
        if(this->size_internal > 1)
        {
            this->particles_agent->MakeProgress();
            this->av_agent->MakeProgress();
            this->th_agent->MakeProgress();
            this->lengths_agent->MakeProgress();
            this->localTHMutex->MakeProgress();
            this->remoteTHMutex->MakeProgress();
        }
    }

    double requestedFactor;
    size_t minimalBuffSize;

private:
    std::unique_ptr<RemoteMemoryAgent<MCParticle>> particles_agent;
    std::unique_ptr<RemoteMemoryAgent<index_t>> av_agent;
    std::unique_ptr<RemoteMemoryAgent<index_t>> th_agent;
    std::unique_ptr<RemoteMemoryAgent<int>> lengths_agent;
    std::shared_ptr<DistributedMutex> localTHMutex;
    std::shared_ptr<DistributedMutex> remoteTHMutex;
    RDMA_Type rdma_type;
    bool destroyed;
    MPI_Group group_world, group_internal;

    #ifdef ADVANCED_STORM_DEBUG
    void ValidateArraysContents(void) const;

    void ValidateRemoteArraysContents(void);
    #endif // ADVANCED_STORM_DEBUG
};

template<typename T, typename Grid>
RankHandler<T, Grid>::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent, RDMA_Type rdma_type,
                                  size_t minimalBuffSize):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize),
    lengths_storage{0, 0},
    av_length(lengths_storage[0]), th_length(lengths_storage[1]),
    rdma_type(rdma_type), destroyed(false), reallocationAgent(reallocationAgent),
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
        this->av_agent = RMAFactory::Create<index_t>(this->rdma_type, this->buffsize, this->comm);
        this->th_agent = RMAFactory::Create<index_t>(this->rdma_type, this->buffsize, this->comm);
        this->lengths_agent = RMAFactory::CreateOver<int>(this->rdma_type, this->lengths_storage, 2, this->comm);

        this->particles = this->particles_agent->GetLocalPointer();
        this->av = this->av_agent->GetLocalPointer();
        this->th = this->th_agent->GetLocalPointer();
        
        // initialize mutexes
        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0, this->rdma_type);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1, this->rdma_type);
        this->localTHMutex = (this->rank_internal == 0)? rank0Mutex : rank1Mutex;
        this->remoteTHMutex = (this->rank_internal == 0)? rank1Mutex : rank0Mutex;
        this->Reset();
        MPI_Barrier(this->comm);
    }
    else
    {
        this->other_rank = 0;
        this->particles = new MCParticle[this->buffsize];
        this->av = new index_t[this->buffsize];
        this->th = new index_t[this->buffsize];
        this->lengths_storage[0] = static_cast<int>(this->buffsize);
        std::iota(this->av, this->av + this->buffsize, 0);
    }

    if(this->size_internal > 1)
    {
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&this->rank_world, 1, MPI_INT, this->other_rank, 0, &this->peer_rank_world, 1, MPI_INT, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);

        MPI_Comm_group(this->comm_world, &this->group_world);
        MPI_Comm_group(this->comm, &this->group_internal);

        // int ranks_in_group[2] = {this->rank_internal, this->other_rank};
        // int ranks_in_world[2];
        // MPI_Group_translate_ranks(this->group_internal, 2, ranks_in_group, this->group_world, ranks_in_world);
        // if(ranks_in_world[0] != this->rank_world)
        // {
        //     // STORMError eo("RankHandler constructor: ranks translation failed");
        //     // eo.addEntry("Real rank", ranks_in_world[0]);
        //     // eo.addEntry("Expected rank", this->rank_world);
        //     // eo.addEntry("Peer rank", this->peer_rank_world);
        //     // throw eo;
        //     std::cout << "RankHandler constructor: ranks translation failed" << std::endl;
        //     std::cout << "Real rank " << ranks_in_world[0] << std::endl;
        //     std::cout << "Real other rank " << ranks_in_world[1] << std::endl;
        //     std::cout << "Expected rank " << this->rank_world << std::endl;
        //     std::cout << "Peer rank " << this->peer_rank_world << std::endl;
        // }
        // assert(ranks_in_world[1] == this->peer_rank_world);
    }
    else
    {
        this->peer_buffsize = this->buffsize;
        this->peer_rank_world = this->rank_world;
    }

    MPI_Barrier(this->comm);
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reset(void)
{
    this->av_length = static_cast<int>(this->buffsize);
    this->th_length = 0;
    std::fill(this->th, this->th + this->buffsize, std::numeric_limits<index_t>::max());
    std::iota(this->av, this->av + this->buffsize, 0);
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }

    #ifdef STORM_WITH_MPI
    if(this->size_internal > 1)
    {
        this->particles_agent->Free();
        this->av_agent->Free();
        this->th_agent->Free();
        this->lengths_agent->Free();
        MPI_Group_free(&this->group_world);
        MPI_Group_free(&this->group_internal);
        DistributedMutex *mutex1 = (this->rank_internal == 0)? this->localTHMutex.get() : this->remoteTHMutex.get();
        DistributedMutex *mutex2 = (this->rank_internal == 1)? this->localTHMutex.get() : this->remoteTHMutex.get();
        mutex1->Destroy();
        mutex2->Destroy();
    }
    else
    {
    #endif // STORM_WITH_MPI
        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
    #ifdef STORM_WITH_MPI
    }
    #endif // STORM_WITH_MPI
    this->destroyed = true;
}

template<typename T, typename Grid>
RankHandler<T, Grid>::~RankHandler()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

#ifdef ADVANCED_STORM_DEBUG
template<typename T, typename Grid>
void RankHandler<T, Grid>::ValidateArraysContents(void) const
{
    // if(this->localTHMutex != nullptr)
    // {
    //     this->localTHMutex->Lock();
    // }
        
    boost::container::flat_map<index_t, size_t> avMap;
    int av_length = this->av_length;
    for(int i = 0; i < av_length; i++)
    {
        index_t avValue = this->av[i];
        if(avValue >= this->buffsize)
        {
            STORMError eo("RankHandler::ValidateArraysContents: AV value is out of bounds");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(avValue) != avMap.end())
        {
            STORMError eo("RankHandler::ValidateArraysContents: AV value is duplicated");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("Previously In Index", avMap[avValue]);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        avMap[avValue] = i;
    }

    boost::container::flat_map<index_t, size_t> thMap;
    int th_length = this->th_length;
    for(int i = 0; i < th_length; i++)
    {
        index_t thValue = this->th[i];
        if(thValue >= this->buffsize)
        {
            STORMError eo("RankHandler::ValidateArraysContents: TH value is out of bounds");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(thMap.find(thValue) != thMap.end())
        {
            STORMError eo("RankHandler::ValidateArraysContents: TH value is duplicated");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("Previously In Index", thMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(thValue) != avMap.end())
        {
            STORMError eo("RankHandler::ValidateArraysContents: TH value is in AV");
            eo.addEntry("Value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("AV index", avMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("AV length", av_length);
            eo.addEntry("The Particle Index", this->th[i]);
            eo.addEntry("The Particle", this->particles[this->th[i]]);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        thMap[thValue] = i;
    }

    // if(this->localTHMutex != nullptr)
    // {
    //     this->localTHMutex->Unlock();
    // }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::ValidateRemoteArraysContents(void)
{
    static std::vector<index_t> remoteAV;
    static std::vector<index_t> remoteTH;

    int lengths[2];
    this->lengths_agent->Get(lengths, 2, this->other_rank, 0);
    int av_length = lengths[0];
    int th_length = lengths[1];

    if(remoteAV.size() < static_cast<size_t>(av_length))
    {
        remoteAV.resize(av_length);
    }
    this->av_agent->Get(remoteAV.data(), av_length, this->other_rank, 0);

    if(remoteTH.size() < static_cast<size_t>(th_length))
    {
        remoteTH.resize(th_length);
    }
    this->th_agent->Get(remoteTH.data(), th_length, this->other_rank, 0);

    // get AV and TH
    boost::container::flat_map<index_t, size_t> avMap;
    for(int i = 0; i < av_length; i++)
    {
        index_t avValue = remoteAV[i];
        if(avValue >= this->peer_buffsize)
        {
            STORMError eo("RankHandler::ValidateRemoteArraysContents: Remote AV value is out of bounds");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(avValue) != avMap.end())
        {
            STORMError eo("RankHandler::ValidateRemoteArraysContents: Remote AV value is duplicated");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("Previously In Index", avMap[avValue]);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        avMap[avValue] = i;
    }

    boost::container::flat_map<index_t, size_t> thMap;
    for(int i = 0; i < th_length; i++)
    {
        index_t thValue = remoteTH[i];
        if(thValue >= this->peer_buffsize)
        {
            STORMError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is out of bounds");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(thMap.find(thValue) != thMap.end())
        {
            STORMError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is duplicated");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("Previously In Index", thMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(thValue) != avMap.end())
        {
            STORMError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is in AV");
            eo.addEntry("Value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("AV index", avMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        thMap[thValue] = i;
    }
}
#endif // ADVANCED_STORM_DEBUG

template<typename T, typename Grid>
void RankHandler<T, Grid>::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
{
    static constexpr index_t inf = std::numeric_limits<index_t>::max();

    if(indicesInToHandle.empty())
    {
        return;
    }
    if(this->size_internal > 1)
    {
        // lock self mutex
        this->localTHMutex->Lock();
    }
    
    // std::cout << "Rank " << this->rank_world << " removes particles " << indicesInToHandle << " from rank " << this->peer_rank_world << "'s buffer" << std::endl;
    volatile int &th_length = this->th_length;
    volatile int &av_length = this->av_length;
    
    #ifdef ADVANCED_STORM_DEBUG
    try
    {
        this->ValidateArraysContents();
    }
    catch(STORMError &e)
    {
        e.addEntry("Where", std::string("RankHandler::RemoveParticles - beginning"));
        e.addEntry("To Remove", num);
        throw e;
    }
    #endif // ADVANCED_STORM_DEBUG

    #ifdef STORM_DEBUG
    boost::container::flat_map<size_t, size_t> indicesMap;
    #endif // STORM_DEBUG
    static thread_local std::vector<index_t> freedIndices;
    freedIndices.clear();
    freedIndices.reserve(num);

    for(int i = static_cast<int>(num) - 1; i >= 0; i--)
    {
        const size_t &toHandleIndex = indicesInToHandle[i];
        assert(i == 0 or indicesInToHandle[i] > indicesInToHandle[i-1]); // should be in a descending order
        // std::cout << "Rank " << this->rank_world << " removes particle " << toHandleIndex << " from handler of rank " << this->peer_rank_world << std::endl;
        assert(toHandleIndex < th_length);
        index_t particleIdx = this->th[toHandleIndex];
        #ifdef STORM_DEBUG
        if(indicesMap.find(particleIdx) != indicesMap.end())
        {
            STORMError eo("RankHandler::RemoveParticles: trying to remove the same particle twice");
            eo.addEntry("Particle index", particleIdx);
            eo.addEntry("TH 1", indicesMap[particleIdx]);
            eo.addEntry("TH 2", toHandleIndex);
            eo.addEntry("Rank", this->rank_world);
            throw eo;
        }
        indicesMap.insert({particleIdx, toHandleIndex});
        #endif // STORM_DEBUG
        assert(av_length < this->buffsize);
        #ifdef STORM_DEBUG
        auto it = std::find(this->av, this->av + av_length, particleIdx);
        if(it != this->av + av_length)
        {
            STORMError eo("RankHandler::RemoveParticles: trying to remove an already available particle");
            eo.addEntry("Particle index", particleIdx);
            eo.addEntry("Already found in index", std::distance(this->av, it));
            eo.addEntry("AV Length", av_length);
            eo.addEntry("Rank", this->rank_world);
            throw eo;
        }
        #endif // STORM_DEBUG
        freedIndices.push_back(particleIdx);
        this->th[toHandleIndex] = this->th[--th_length];
        this->th[th_length] = inf;
        assert(th_length >= 0);
    }

    std::sort(freedIndices.begin(), freedIndices.end());
    if(static_cast<size_t>(av_length) + freedIndices.size() > this->buffsize)
    {
        STORMError eo("RankHandler::RemoveParticles: insufficient AV capacity while appending freed slots");
        eo.addEntry("My Rank", this->rank_world);
        eo.addEntry("Peer Rank", this->peer_rank_world);
        eo.addEntry("AV Length", av_length);
        eo.addEntry("Freed Slots", freedIndices.size());
        eo.addEntry("Buffer Size", this->buffsize);
        throw eo;
    }
    for(index_t particleIdx : freedIndices)
    {
        this->av[av_length++] = particleIdx;
    }

    #ifdef ADVANCED_STORM_DEBUG
    try
    {
        this->ValidateArraysContents();
    }
    catch(STORMError &e)
    {
        e.addEntry("Where", std::string("RankHandler::RemoveParticles - end"));
        e.addEntry("To Remove", num);
        throw e;
    }
    #endif // ADVANCED_STORM_DEBUG

    if(this->size_internal > 1)
    {
        // release self mutex
        this->localTHMutex->Unlock();
    }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reallocate(double factor)
{
#ifdef MEMORY_DEBUG
    memory_debug::check_system_memory("RankHandler::Reallocate");
#endif

    static constexpr index_t inf = std::numeric_limits<index_t>::max();
    

    double requestedFactorSelf;
    MPI_Sendrecv(&this->requestedFactor, 1, MPI_DOUBLE, this->other_rank, 0, &requestedFactorSelf, 1, MPI_DOUBLE, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    factor = std::max(factor, requestedFactorSelf);
    
    size_t newBuffSize = std::ceil(this->buffsize * factor);
    size_t oldBuffSize = this->buffsize;

    bool noParticles = (this->th_length == 0);
    if(oldBuffSize > newBuffSize)
    {
        if(not noParticles)
        {
            STORMError eo("Can not shrink memory when there are particles (there are " + std::to_string(this->th_length) + " particles)");
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            eo.addEntry("TH Length", this->th_length);
            throw eo;
        }
    }
    size_t peerNewBuffSize = std::ceil(this->peer_buffsize * factor);
    newBuffSize = std::max<size_t>(newBuffSize, this->minimalBuffSize);
    peerNewBuffSize = std::max<size_t>(peerNewBuffSize, this->minimalBuffSize);

    this->buffsize = newBuffSize;

    if(this->size_internal > 1)
    {
        assert(this->size_internal == 2);

        int localNoParticles = noParticles ? 1 : 0;
        int peerNoParticles = 0;
        MPI_Sendrecv(&localNoParticles, 1, MPI_INT, this->other_rank, 0, &peerNoParticles, 1, MPI_INT, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
        bool bothEmpty = (localNoParticles and peerNoParticles);

        if(bothEmpty)
        {
            this->particles_agent->Replace(this->buffsize);
            this->av_agent->Replace(this->buffsize);
            this->th_agent->Replace(this->buffsize);
        }
        else
        {
            this->particles_agent->Resize(this->buffsize);
            this->av_agent->Resize(this->buffsize);
            this->th_agent->Resize(this->buffsize);
        }

        this->particles = this->particles_agent->GetLocalPointer();
        this->av = this->av_agent->GetLocalPointer();
        this->th = this->th_agent->GetLocalPointer();

        if(noParticles)
        {
            std::iota(this->av, this->av + this->buffsize, 0);
            this->av_length = static_cast<int>(this->buffsize);
            this->th_length = 0;
            std::fill(this->th, this->th + this->buffsize, inf);
        }
        else if(this->buffsize >= oldBuffSize)
        {
            size_t difference = this->buffsize - oldBuffSize;
            std::memmove(this->av + difference, this->av, oldBuffSize * sizeof(index_t));
            std::iota(this->av, this->av + difference, static_cast<index_t>(oldBuffSize));

            std::fill(this->th + oldBuffSize, this->th + this->buffsize, inf);

            int difference_int = static_cast<int>(difference);
            this->av_length += difference_int;
        }
        else
        {
            std::iota(this->av, this->av + this->buffsize, 0);
            this->av_length = static_cast<int>(this->buffsize);
        }
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);

        #ifdef ADVANCED_STORM_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_STORM_DEBUG
    }
    else
    {
        if(noParticles)
        {
            delete[] this->particles;
            delete[] this->av;
            delete[] this->th;

            this->particles = new MCParticle[this->buffsize];
            this->av = new index_t[this->buffsize];
            this->th = new index_t[this->buffsize];

            std::iota(this->av, this->av + this->buffsize, 0);
            this->av_length = static_cast<int>(this->buffsize);
            this->th_length = 0;
            std::fill(this->th, this->th + this->buffsize, inf);
        }
        else
        {
            MCParticle *new_particles = new MCParticle[this->buffsize];
            index_t *new_av = new typename RankHandler::index_t[this->buffsize];
            index_t *new_th = new typename RankHandler::index_t[this->buffsize];

            if(this->buffsize >= oldBuffSize)
            {
                std::memcpy(new_particles, this->particles, oldBuffSize * sizeof(MCParticle));
                std::memcpy(new_th, this->th, this->th_length * sizeof(index_t));
                size_t difference = this->buffsize - oldBuffSize;
                std::memcpy(new_av + difference, this->av, oldBuffSize * sizeof(index_t));
                std::iota(new_av, new_av + difference, oldBuffSize);
                int difference_int = difference;
                this->av_length += difference_int;
            }
            else
            {
                std::memcpy(new_particles, this->particles, this->buffsize * sizeof(MCParticle));
                std::memcpy(new_th, this->th, this->buffsize * sizeof(index_t));
                std::iota(new_av, new_av + this->buffsize, 0);
                this->av_length = static_cast<int>(this->buffsize);
            }
            std::fill(new_th + this->th_length, new_th + this->buffsize, inf);

            delete[] this->particles;
            delete[] this->av;
            delete[] this->th;
            this->particles = new_particles;
            this->av = new_av;
            this->th = new_th;
        }

        this->peer_buffsize = this->buffsize;

        #ifdef ADVANCED_STORM_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_STORM_DEBUG
    }

    this->requestedFactor = 1;
}

template<typename T, typename Grid>
bool RankHandler<T, Grid>::UsesAsyncReallocation(void) const
{
    if(this->size_internal <= 1)
    {
        return false;
    }
    RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                             ? RMAFactory::ResolveAutoRDMA()
                             : this->rdma_type;
    return resolved == RDMA_Type::OFI_RDMA or resolved == RDMA_Type::IBV_RDMA;
}

template<typename T, typename Grid>
ReallocationMetadata RankHandler<T, Grid>::LocalReallocate(double factor)
{
    if(not this->UsesAsyncReallocation())
    {
        throw std::runtime_error("RankHandler::LocalReallocate is only supported for native RDMA backends");
    }

#ifdef MEMORY_DEBUG
    memory_debug::check_system_memory("RankHandler::LocalReallocate");
#endif

    static constexpr index_t inf = std::numeric_limits<index_t>::max();


    this->LockSelfBuffer();
    bool locked = true;
    try
    {
        factor = std::max(factor, 1.0);
        size_t oldBuffSize = this->buffsize;
        size_t newBuffSize = std::ceil(static_cast<double>(this->buffsize) * factor);
        newBuffSize = std::max<size_t>(newBuffSize, this->minimalBuffSize);
        if(newBuffSize <= this->buffsize)
        {
            newBuffSize = this->buffsize + 1;
        }

        bool noParticles = (this->th_length == 0);
        this->buffsize = newBuffSize;

        RemoteBufferInfo particlesInfo = this->particles_agent->LocalResize(this->buffsize);
        RemoteBufferInfo avInfo = this->av_agent->LocalResize(this->buffsize);
        RemoteBufferInfo thInfo = this->th_agent->LocalResize(this->buffsize);
        RemoteBufferInfo lengthsInfo = this->lengths_agent->GetLocalRemoteInfo();

        this->particles = this->particles_agent->GetLocalPointer();
        this->av = this->av_agent->GetLocalPointer();
        this->th = this->th_agent->GetLocalPointer();

        if(noParticles)
        {
            std::iota(this->av, this->av + this->buffsize, 0);
            this->av_length = static_cast<int>(this->buffsize);
            this->th_length = 0;
            std::fill(this->th, this->th + this->buffsize, inf);
        }
        else
        {
            size_t difference = this->buffsize - oldBuffSize;
            std::memmove(this->av + difference, this->av, oldBuffSize * sizeof(index_t));
            std::iota(this->av, this->av + difference, static_cast<index_t>(oldBuffSize));
            std::fill(this->th + oldBuffSize, this->th + this->buffsize, inf);
            this->av_length += static_cast<int>(difference);
        }

        #ifdef ADVANCED_STORM_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_STORM_DEBUG

        this->UnlockSelfBuffer();
        locked = false;


        return {particlesInfo, avInfo, thInfo, lengthsInfo, this->buffsize};
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
void RankHandler<T, Grid>::UpdatePeerRemoteInfo(const ReallocationMetadata &metadata)
{
    if(this->size_internal <= 1)
    {
        return;
    }
    this->particles_agent->UpdateRemoteInfo(this->other_rank, metadata.particles);
    this->av_agent->UpdateRemoteInfo(this->other_rank, metadata.av);
    this->th_agent->UpdateRemoteInfo(this->other_rank, metadata.th);
    this->lengths_agent->UpdateRemoteInfo(this->other_rank, metadata.lengths);
    this->peer_buffsize = metadata.new_buffsize;
}

template<typename T, typename Grid>
bool RankHandler<T, Grid>::TransferParticles(const std::vector<MCParticle> &particles)
{
    size_t Np = particles.size();
    if(particles.empty())
    {
        return true;
    }

    if(this->size_internal > 1)
    {
        this->remoteTHMutex->Lock();

        #ifdef ADVANCED_STORM_DEBUG
            try
            {
                ValidateRemoteArraysContents();
            }
            catch(STORMError &eo)
            {
                eo.addEntry("Where", std::string("RankHandler<T, Grid>::TransferParticles - before transfer"));
                throw eo;
            }
        #endif // ADVANCED_STORM_DEBUG

        #ifdef ADVANCED_STORM_DEBUG
        size_t reallocationsCounter = 0;
        #endif // ADVANCED_STORM_DEBUG
        int remoteLengths[2];
        auto getAvailableLength = [&](void)
        {
            this->lengths_agent->Get(remoteLengths, 2, this->other_rank, 0);
            int availLength = remoteLengths[0];
            assert(availLength <= static_cast<int>(this->peer_buffsize));
            if(availLength < static_cast<int>(Np))
            {
                #ifdef ADVANCED_STORM_DEBUG
                reallocationsCounter++;
                #endif // ADVANCED_STORM_DEBUG
            }
            assert(availLength >= 0);
            return availLength;
        };

        int availLength = getAvailableLength();
        while(availLength < Np)
        {
            size_t peerUsed = this->peer_buffsize - static_cast<size_t>(availLength);
            this->requestedFactor = static_cast<double>(peerUsed + Np) /
                                    static_cast<double>(this->peer_buffsize) * 1.5;

            this->remoteTHMutex->Unlock();
            if(this->UsesAsyncReallocation())
            {
                this->reallocationAgent->RequestReallocationAsync(this->peer_rank_world, this->requestedFactor);
                this->requestedFactor = 1;
                return false;
            }
            this->reallocationAgent->RequestReallocation(this->peer_rank_world);
            this->remoteTHMutex->Lock();
            availLength = getAvailableLength();
        }
        assert(availLength >= Np);

        static thread_local std::vector<index_t> availIndices;
        static thread_local uint32_t availIndices_lkey = 0;
        static thread_local uint64_t availIndices_reg_handle = 0;
        static thread_local const index_t *availIndices_reg_ptr = nullptr;
        availIndices.resize(Np);
        this->av_agent->Get(availIndices.data(), Np, this->other_rank, availLength - Np);

        #ifdef STORM_DEBUG
        boost::container::flat_map<index_t, size_t> availIndicesMap;
        for(size_t i = 0; i < Np; i++)
        {
            index_t availIndex = availIndices[i];
            if(availIndicesMap.find(availIndex) != availIndicesMap.end())
            {
                STORMError eo("RankHandler<T, Grid>::TransferParticles: duplication in available Index");
                eo.addEntry("Available Index", availIndex);
                eo.addEntry("Already in Index", availIndicesMap[availIndex]);
                eo.addEntry("Index", i);
                eo.addEntry("Rank", this->rank_world);
                eo.addEntry("Peer Rank", this->peer_rank_world);
                throw eo;
            }
            availIndicesMap.insert({availIndex, i});
            assert(availIndex < this->peer_buffsize);
            if(particles[i].nextRank != this->peer_rank_world)
            {
                STORMError eo("RankHandler<T, Grid>::TransferParticles: Particle will not be sent to the expected rank");
                eo.addEntry("Particle", particles[i]);
                eo.addEntry("Origin", this->rank_world);
                eo.addEntry("Expected Rank", particles[i].nextRank);
                eo.addEntry("Next Rank", this->peer_rank_world);
                throw eo;
            }
        }
        #endif // STORM_DEBUG

        assert(this->other_rank != this->rank_internal);
        static thread_local std::vector<std::pair<index_t, size_t>> orderedTargets;
        static thread_local std::vector<MCParticle> orderedParticles;
        static thread_local std::vector<index_t> orderedIndices;
        using PutBatchEntry = typename RemoteMemoryAgent<MCParticle>::PutBatchEntry;
        static thread_local std::vector<PutBatchEntry> batchEntries;
        static thread_local uint32_t orderedParticles_lkey = 0;
        static thread_local uint64_t orderedParticles_reg_handle = 0;
        static thread_local const MCParticle *orderedParticles_reg_ptr = nullptr;

        orderedTargets.resize(Np);
        orderedParticles.clear();
        orderedIndices.clear();
        batchEntries.clear();
        orderedParticles.reserve(Np);
        orderedIndices.reserve(Np);

        for(size_t i = 0; i < Np; i++)
        {
            orderedTargets[i] = std::make_pair(availIndices[i], i);
        }
        std::sort(orderedTargets.begin(), orderedTargets.end());

        for(size_t i = 0; i < Np; i++)
        {
            orderedIndices.push_back(orderedTargets[i].first);
            orderedParticles.push_back(particles[orderedTargets[i].second]);
        }

        size_t runStart = 0;
        bool usedContiguousAllocation = false;
        while(runStart < Np)
        {
            size_t runEnd = runStart + 1;
            while(runEnd < Np and orderedIndices[runEnd] == orderedIndices[runEnd - 1] + 1)
            {
                runEnd++;
            }

            size_t runLength = runEnd - runStart;
            batchEntries.push_back({runStart, orderedIndices[runStart], runLength});

            if(runLength > 1)
            {
                usedContiguousAllocation = true;
            }
            else
            {
            }

            runStart = runEnd;
        }


        if(orderedParticles.data() != orderedParticles_reg_ptr)
        {
            if(orderedParticles_reg_handle)
            {
                this->particles_agent->DeregisterExternalSource(orderedParticles_reg_handle);
            }
            auto reg = this->particles_agent->RegisterExternalSource(
                orderedParticles.data(), orderedParticles.capacity() * sizeof(MCParticle));
            orderedParticles_lkey = reg.lkey;
            orderedParticles_reg_handle = reg.handle;
            orderedParticles_reg_ptr = orderedParticles.data();
        }

        this->particles_agent->PutBatch(orderedParticles.data(), Np,
                                        batchEntries.data(), batchEntries.size(),
                                        this->other_rank, false, orderedParticles_lkey);

        (void)usedContiguousAllocation;


        int toHandleLength = remoteLengths[1];
        assert(toHandleLength >= 0);
        assert(toHandleLength < static_cast<int>(this->peer_buffsize));

        // MPI RMA uses separate MPI_Win objects, and OFI providers do not
        // guarantee the IBV-style RC FIFO ordering that this publication path
        // relies on. Flush payload writes before exposing queue metadata.
        RDMA_Type resolved = (this->rdma_type == RDMA_Type::AUTO_RDMA)
                                 ? RMAFactory::ResolveAutoRDMA()
                                 : this->rdma_type;
        bool is_mpi = (resolved == RDMA_Type::MPI_RMA);
        bool is_ofi = (resolved == RDMA_Type::OFI_RDMA);

        if(is_mpi or is_ofi)
        {
            this->particles_agent->Flush(this->other_rank);
        }

        if(availIndices.data() != availIndices_reg_ptr)
        {
            if(availIndices_reg_handle)
            {
                this->th_agent->DeregisterExternalSource(availIndices_reg_handle);
            }
            auto reg = this->th_agent->RegisterExternalSource(
                availIndices.data(), availIndices.capacity() * sizeof(index_t));
            availIndices_lkey = reg.lkey;
            availIndices_reg_handle = reg.handle;
            availIndices_reg_ptr = availIndices.data();
        }
        this->th_agent->Put(availIndices.data(), Np, this->other_rank, toHandleLength, true, availIndices_lkey);
        int newLengths[2] = {availLength - static_cast<int>(Np),
                             toHandleLength + static_cast<int>(Np)};
        this->lengths_agent->Put(newLengths, 2, this->other_rank, 0, is_mpi or is_ofi);

        #ifdef ADVANCED_STORM_DEBUG
        try
        {
            ValidateRemoteArraysContents();
        }
        catch(STORMError &eo)
        {
            eo.addEntry("Where", std::string("RankHandler<T, Grid>::TransferParticles - after transfer"));
            eo.addEntry("Transfer Amount", Np);
            eo.addEntry("AV Indices", availIndices);
            eo.addEntry("Expected TH length", newLengths[1]);
            eo.addEntry("Expected AV length", newLengths[0]);
            eo.addEntry("Reallocations Counter", reallocationsCounter);
            throw eo;
        }
        #endif // ADVANCED_STORM_DEBUG

        // release remote mutex
        this->remoteTHMutex->Unlock();
    }
    else
    {
        for(size_t i = 0; i < Np; i++)
        {
            assert(this->av_length > 0);
            size_t availIndex = this->av[--this->av_length];
            this->particles[availIndex] = particles[i];
            #ifdef STORM_DEBUG
            if(particles[i].nextRank != this->rank_world)
            {
                STORMError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particles[i]);
                eo.addEntry("Origin", particles[i].sentByRank);
                eo.addEntry("Next Rank", particles[i].nextRank);
                throw eo;
            }
            #endif // STORM_DEBUG
            this->th[this->th_length++] = availIndex;
            assert(this->th_length < static_cast<int>(this->buffsize));
        }
    }
    return true;
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // MONTECARLO_RANK_HANDLER_HPP
