#ifndef STORM_MONTE_CARLO_PARTICLE_INITIALIZATION_HPP
#define STORM_MONTE_CARLO_PARTICLE_INITIALIZATION_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../particle/Particle.hpp"

namespace STORM {

class MonteCarloParticleInitializer
{
public:
    template<typename ParticleT>
    static void Initialize(ParticleT &particle, dt_t fullDt)
    {
#if defined(STORM_DEBUG) && defined(STORM_WITH_MPI)
    particle.checkedHere = true;
    particle.nextRank = std::numeric_limits<rank_t>::max();
    particle.removedFromRank = false;
    particle.sentByRank = std::numeric_limits<rank_t>::max();
    particle.lastSeen = 0;
    particle.lastSeenRank = std::numeric_limits<rank_t>::max();
    particle.lastSeenRankBuf = std::numeric_limits<rank_t>::max();
    particle.lastSeenIndex = std::numeric_limits<size_t>::max();
#endif
#ifdef STORM_WITH_TRACING_HISTORY
    particle.tracingHistoryIndex = 0;
    particle.tracingHistoryCount = 0;
#endif
    particle.timeLeft = fullDt;
    particle.initialWeight = std::abs(particle.weight);
    particle.steps = 0;
    }

    template<typename ParticleT>
    static void Initialize(std::vector<ParticleT> &particles, dt_t fullDt)
    {
        for(ParticleT &particle : particles)
        {
            MonteCarloParticleInitializer::Initialize(particle, fullDt);
        }
    }

    template<typename ParticleStore>
    static void InitializeStore(ParticleStore &store, dt_t fullDt)
    {
        store.ForEachActive([fullDt](typename ParticleStore::value_type &particle, size_t)
        {
            MonteCarloParticleInitializer::Initialize(particle, fullDt);
        });
    }
};

template<typename ParticleT>
void InitializeMonteCarloParticle(ParticleT &particle, dt_t fullDt)
{
    MonteCarloParticleInitializer::Initialize(particle, fullDt);
}

template<typename ParticleT>
void InitializeMonteCarloParticles(std::vector<ParticleT> &particles, dt_t fullDt)
{
    MonteCarloParticleInitializer::Initialize(particles, fullDt);
}

template<typename ParticleStore>
void InitializeMonteCarloParticleStore(ParticleStore &store, dt_t fullDt)
{
    MonteCarloParticleInitializer::InitializeStore(store, fullDt);
}

} // namespace STORM

#endif // STORM_MONTE_CARLO_PARTICLE_INITIALIZATION_HPP
