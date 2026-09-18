#ifndef STORM_P2P_COMMUNICATION_ENGINE_HPP
#define STORM_P2P_COMMUNICATION_ENGINE_HPP
#ifdef STORM_WITH_MPI
#include <boost/container/flat_set.hpp>
#include <mpi_utils/BuffersManager.hpp>
#include "CommunicationEngine.hpp"
#include "../../gpu/DeviceParticle.hpp"
#include <type_traits>
#include <stdexcept>

namespace STORM
{
template<class T, class Grid>
class P2PCommunicationEngine : public CommunicationEngine<T>
{
    using MCParticle = Particle<T>;

    // Exclude Serializable vtables and copy the transport payload in one block.
    struct PackedParticle
    {
        gpu::DeviceParticle particle;
        gpu::DeviceParticleCold cold;
        std::uint8_t sent = 0;
    };
    static_assert(std::is_trivially_copyable_v<PackedParticle>);
#if !defined(STORM_DEBUG) && !defined(STORM_WITH_TRACING_HISTORY)
    static constexpr bool packedWire = std::is_same_v<typename T::coord_type, double>;
#else
    static constexpr bool packedWire = false;
#endif
    using WireParticle = std::conditional_t<packedWire, PackedParticle, MCParticle>;

public:
    P2PCommunicationEngine(const Grid &grid, const MonteCarloConfig &config, MPI_Comm comm)
        : grid(grid), config(config), comm(comm)
    {
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &size);
        incoming.resize(size);
    }

    void Prepare() override
    {
        buffers.reset();
        lastTransfers = 0;
        neighbors = grid.GetDuplicatedProcs();
        for(std::vector<MCParticle> &queue : incoming)
        {
            queue.clear();
        }
        buffers = std::make_unique<BuffersManager<WireParticle>>(
            comm, [this](const WireParticle *particles, size_t count, rank_t source)
            {
                std::vector<MCParticle> &queue = incoming[source];
                if constexpr(packedWire)
                {
                    const auto offset = queue.size();
                    queue.resize(offset + count);
                    for(std::size_t i = 0; i < count; ++i)
                    {
                        gpu::UnpackParticle(particles[i].particle, particles[i].cold, queue[offset + i]);
                        queue[offset + i].sent = particles[i].sent;
                    }
                }
                else
                    queue.insert(queue.end(), particles, particles + count);
            },
            8817, std::max<size_t>(1000, 2 * config.sendBufferMinSize) * sizeof(WireParticle), config.sendBufferMinSize * sizeof(WireParticle), dispatchCycles, size, neighbors);
    }

    void Progress() override
    {
        if(buffers)
        {
            buffers->HandleIncomingOutcoming();
        }
    }

    // Tear the particle buffers down as soon as the step's transport loop
    // ends (collective: every rank reaches EndTransport). Between steps this
    // engine then holds no posted MPI receives, so a manager that is kept
    // alive but no longer stepped cannot match messages meant for a newer
    // manager on the same communicator and tag.
    void EndTransport() override
    {
        if(buffers)
        {
            lastTransfers += buffers->GetSentCounter();
            buffers.reset();
        }
    }

    void Flush(bool) override
    {
        this->Progress();
    }

    void FlushAll() override
    {
        if(!buffers)
        {
            return;
        }
        // BuffersManager advances its dispatch age on each progress call.
        // Drain short batches without changing the shared MPI utility API.
        for(size_t cycle = 0; cycle <= dispatchCycles && buffers->GetPendingNumber(); ++cycle)
        {
            Progress();
        }
    }

    bool Pending() const override
    {
        if(buffers && (buffers->GetPendingNumber() || buffers->GetActiveSendsNumber()))
        {
            return true;
        }
        for(const std::vector<MCParticle> &queue : incoming)
        {
            if(!queue.empty())
            {
                return true;
            }
        }
        return false;
    }
    void Send(rank_t destination, const MCParticle &particle) override
    {
        if(!buffers)
        {
            throw std::runtime_error("P2PCommunicationEngine::Send called outside a transport step");
        }
        if constexpr(packedWire)
        {
            PackedParticle packed;
            gpu::PackParticle(particle, packed.particle, packed.cold);
            packed.sent = particle.sent;
            buffers->Add(destination, packed);
        }
        else
            buffers->Add(destination, particle);
    }

    void AppendLocal(const std::vector<MCParticle> &particles) override
    {
        std::vector<MCParticle> &queue = incoming[rank];
        queue.insert(queue.end(), particles.begin(), particles.end());
    }

    void Detach(rank_t source, std::vector<MCParticle> &particles) override
    {
        particles.swap(incoming[source]);
    }

    size_t LocalSize(rank_t source) const override
    {
        return incoming[source].size();
    }

    const std::vector<rank_t> &Neighbors() const override
    {
        return neighbors;
    }

    void VisitLocal(const std::function<void(MCParticle &)> &visit) override
    {
        for(std::vector<MCParticle> &queue : incoming)
        {
            for(MCParticle &particle : queue)
            {
                visit(particle);
            }
        }
    }

    size_t MemoryBytes() const override
    {
        size_t bytes = this->buffers ? this->buffers->GetTotalMemoryBytes() : 0;
        for(const std::vector<MCParticle> &queue : this->incoming)
        {
            bytes += queue.capacity() * sizeof(MCParticle);
        }
        return bytes;
    }

    size_t Transfers() const override
    {
        return lastTransfers + (this->buffers ? this->buffers->GetSentCounter() : 0);
    }

    std::vector<unsigned long long> MessageCountsSent() const override
    {
        if(!this->buffers) return {};
        const std::vector<size_t> &c = this->buffers->GetAllSendCounters();
        return std::vector<unsigned long long>(c.begin(), c.end());
    }

    std::vector<unsigned long long> MessageCountsReceived() const override
    {
        if(!this->buffers) return {};
        const std::vector<size_t> &c = this->buffers->GetAllRecvCounters();
        return std::vector<unsigned long long>(c.begin(), c.end());
    }

    rank_t Rank() const override
    {
        return this->rank;
    }

    rank_t Size() const override
    {
        return this->size;
    }

    MPI_Comm Communicator() const override
    {
        return this->comm;
    }

private:
    static constexpr size_t dispatchCycles = 500;
    const Grid &grid;
    MonteCarloConfig config;
    MPI_Comm comm;
    rank_t rank;
    rank_t size;
    std::vector<rank_t> neighbors;
    std::vector<std::vector<MCParticle>> incoming;
    std::unique_ptr<BuffersManager<WireParticle>> buffers;
    size_t lastTransfers = 0; // messages sent by buffers released at EndTransport
};
} // namespace STORM

#endif // STORM_WITH_MPI

#endif // STORM_P2P_COMMUNICATION_ENGINE_HPP
