#ifndef STORM_MONTE_CARLO_TRACKER_HPP
#define STORM_MONTE_CARLO_TRACKER_HPP

#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <cstddef>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi.h>
#include <mpi_utils/mpi_commands.hpp>
#endif // STORM_WITH_MPI

namespace STORM {

template<typename ParticleT>
class MonteCarloTracker
{
public:
    MonteCarloTracker(void) = default;

#ifdef STORM_WITH_MPI
    explicit MonteCarloTracker(const MPI_Comm &comm) : comm(comm)
    {}
#endif // STORM_WITH_MPI

    void Reset(void)
    {
        this->track.clear();
    }

    std::vector<ParticleT> GetLocalTrackParticleRoute(size_t id) const
    {
        typename boost::container::flat_map<size_t, std::vector<ParticleT>>::const_iterator it = this->track.find(id);
        if(it == this->track.end())
        {
            return std::vector<ParticleT>();
        }
        return it->second;
    }

    std::vector<ParticleT> GetTrackParticleRoute(size_t id) const
    {
        std::vector<ParticleT> local = this->GetLocalTrackParticleRoute(id);
#ifdef STORM_WITH_MPI
        std::vector<ParticleT> global = MPI_All_cast(local, this->comm);
        std::sort(global.begin(), global.end(), [](const ParticleT &first, const ParticleT &second)
        {
            return first.steps < second.steps;
        });
        return global;
#else
        return local;
#endif // STORM_WITH_MPI
    }

    void ReportParticle(ParticleT &particle)
    {
        if(this->track.find(particle.id) == this->track.end())
        {
            this->track[particle.id] = std::vector<ParticleT>();
        }
        this->track[particle.id].push_back(particle);
    }

private:
#ifdef STORM_WITH_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#endif // STORM_WITH_MPI
    boost::container::flat_map<size_t, std::vector<ParticleT>> track;
};

} // namespace STORM

#endif // STORM_MONTE_CARLO_TRACKER_HPP
