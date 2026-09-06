#ifndef STORM_SERIAL_COMMUNICATION_ENGINE_HPP
#define STORM_SERIAL_COMMUNICATION_ENGINE_HPP

#include "CommunicationEngine.hpp"

namespace STORM
{

// Serial has ordinary local storage and no MPI or network resources.
template<class T>
class SerialCommunicationEngine : public CommunicationEngine<T>
{
    using MCParticle = Particle<T>;

public:
    void Prepare() override
    {
        this->particles.clear();
    }

    void Progress() override
    {
    }

    void Flush(bool) override
    {
    }

    void FlushAll() override
    {
    }

    bool Pending() const override
    {
        return not this->particles.empty();
    }

    void Send(rank_t, const MCParticle &) override
    {
        throw std::runtime_error("Serial transport cannot send to another rank");
    }

    void AppendLocal(const std::vector<MCParticle> &particles) override
    {
        this->particles.insert(this->particles.end(), particles.begin(), particles.end());
    }

    void Detach(rank_t, std::vector<MCParticle> &particles) override
    {
        particles.swap(this->particles);
    }

    size_t LocalSize(rank_t) const override
    {
        return this->particles.size();
    }

    const std::vector<rank_t> &Neighbors() const override
    {
        return this->neighbors;
    }

    void VisitLocal(const std::function<void(MCParticle &)> &visit) override
    {
        for(MCParticle &particle : this->particles)
        {
            visit(particle);
        }
    }

    size_t MemoryBytes() const override
    {
        return this->particles.capacity() * sizeof(MCParticle);
    }

    size_t Transfers() const override
    {
        return 0;
    }

private:
    std::vector<MCParticle> particles;
    std::vector<rank_t> neighbors;
};

} // namespace STORM

#endif // STORM_SERIAL_COMMUNICATION_ENGINE_HPP
