#ifndef STORM_LOCAL_TRANSPORT_EXECUTOR_HPP
#define STORM_LOCAL_TRANSPORT_EXECUTOR_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace STORM {

// Host implementation of the local-compute side of the transport/communication
// boundary.  The manager supplies a contiguous batch and a one-particle kernel;
// an accelerator implementation can retain the same batch contract while
// replacing execute() with pack/copy/launch/copy-back operations.
template<typename ParticleT>
class HostLocalTransportExecutor
{
public:
    template<typename TransportOne>
    std::size_t execute(std::vector<ParticleT> &particles,
                        std::size_t maximumParticles,
                        TransportOne &&transportOne) const
    {
        const std::size_t count = std::min(
            std::max<std::size_t>(1, maximumParticles), particles.size());
        for(std::size_t processed = 0; processed < count; ++processed)
        {
            const std::size_t particleIndex = particles.size() - 1;
            transportOne(particles.back(), particleIndex);
            particles.pop_back();
        }
        return count;
    }
};

} // namespace STORM

#endif // STORM_LOCAL_TRANSPORT_EXECUTOR_HPP
