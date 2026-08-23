#ifndef STORM_STEP_RESULT_HPP
#define STORM_STEP_RESULT_HPP

#include <limits>
#include "ParticleStatus.hpp"
#include "../types.hpp"

namespace STORM {

struct StepResult
{
    ParticleStatus change = ParticleStatus::NO_CELL_MOVE;
    cell_index_t nextCellIndex = std::numeric_limits<cell_index_t>::max();
    // Set when nextCellIndex is an actual outside-box boundary.  The manager
    // combines this with the boundary-condition outcome before accounting
    // escape energy.
    std::uint8_t boundaryCrossing = 0;
};

} // namespace STORM

// Back-compat alias
using MonteCarloFunctionality = STORM::StepResult;

#endif // STORM_STEP_RESULT_HPP
