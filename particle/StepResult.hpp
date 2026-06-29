#ifndef STORM_STEP_RESULT_HPP
#define STORM_STEP_RESULT_HPP

#include <limits>
#include "ParticleStatus.hpp"

namespace STORM {

template<typename T, typename Grid>
struct StepResult
{
    ParticleStatus change = ParticleStatus::NO_CELL_MOVE;
    size_t nextCellIndex = std::numeric_limits<size_t>::max();
};

} // namespace STORM

// Back-compat alias
template<typename T, typename Grid>
using MonteCarloFunctionality = STORM::StepResult<T, Grid>;

#endif // STORM_STEP_RESULT_HPP
