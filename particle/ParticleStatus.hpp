#ifndef RDMONT_PARTICLE_STATUS_HPP
#define RDMONT_PARTICLE_STATUS_HPP

#include <string>

namespace RDMont {

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

} // namespace RDMont

// Back-compat aliases used throughout existing code
using MonteCarloParticleStatus = RDMont::ParticleStatus;
inline std::string MonteCarloParticleStatusToString(int s) { return RDMont::ParticleStatusToString(s); }

#endif // RDMONT_PARTICLE_STATUS_HPP
