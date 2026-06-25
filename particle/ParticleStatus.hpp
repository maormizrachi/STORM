#ifndef STORM_PARTICLE_STATUS_HPP
#define STORM_PARTICLE_STATUS_HPP

#include <string>

namespace STORM {

enum ParticleStatus
{
    NO_CELL_MOVE,
    CELL_MOVE,
    DONE,
    REMOVE,
    REFLECT
};

inline std::string ParticleStatusToString(int status)
{
    switch(status)
    {
        case NO_CELL_MOVE: return "NO_CELL_MOVE";
        case CELL_MOVE:    return "CELL_MOVE";
        case DONE:         return "DONE";
        case REMOVE:       return "REMOVE";
        case REFLECT:      return "REFLECT";
        default:           return "UNKNOWN(" + std::to_string(status) + ")";
    }
}

} // namespace STORM

// Back-compat aliases used throughout existing code
using MonteCarloParticleStatus = STORM::ParticleStatus;
inline std::string MonteCarloParticleStatusToString(int s) { return STORM::ParticleStatusToString(s); }

#endif // STORM_PARTICLE_STATUS_HPP
