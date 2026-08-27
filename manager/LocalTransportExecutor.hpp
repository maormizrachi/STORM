#ifndef STORM_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace STORM {

// Host implementation of the local-compute side of the transport/communication
// boundary.  The manager supplies a contiguous batch and a one-particle kernel;
// an accelerator implementation can retain the same batch contract while
// replacing Execute() with pack/copy/launch/copy-back operations.
template<typename ParticleT>
class HostLocalTransportExecutor
{
public:
    template<typename TransportOne>
    std::size_t Execute(std::vector<ParticleT> &particles, std::size_t maximumParticles, TransportOne &&transportOne) const
    {
        const std::size_t count = std::min(std::max<std::size_t>(1, maximumParticles), particles.size());
        for(std::size_t processed = 0; processed < count; ++processed)
        {
            const std::size_t particleIndex = particles.size() - 1;
            transportOne(particles.back(), particleIndex);
            particles.pop_back();
        }
        return count;
    }

    template<typename ParticleStore, typename TransportOne>
    std::size_t Execute(ParticleStore &particles, std::size_t maximumParticles, TransportOne &&transportOne) const
    {
        const std::size_t count = std::min(std::max<std::size_t>(1, maximumParticles), particles.Size());
        for(std::size_t processed = 0; processed < count; ++processed)
        {
            transportOne(particles.Front(), processed);
            particles.Consume();
        }
        return count;
    }

    template<typename TransportOne>
    std::size_t execute(std::vector<ParticleT> &particles, std::size_t maximumParticles, TransportOne &&transportOne) const
    {
        return this->Execute(particles, maximumParticles, std::forward<TransportOne>(transportOne));
    }

    template<typename ParticleStore, typename TransportOne>
    std::size_t execute(ParticleStore &particles, std::size_t maximumParticles, TransportOne &&transportOne) const
    {
        return this->Execute(particles, maximumParticles, std::forward<TransportOne>(transportOne));
    }
};

} // namespace STORM

#endif // STORM_LOCAL_TRANSPORT_EXECUTOR_HPP
