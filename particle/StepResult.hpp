#ifndef RDMONT_STEP_RESULT_HPP
#define RDMONT_STEP_RESULT_HPP

#include <limits>
#include "monte/particle/ParticleStatus.hpp"

namespace RDMont {

template<typename T, typename Grid>
struct StepResult
{
    ParticleStatus change = ParticleStatus::NO_CELL_MOVE;
    size_t nextCellIndex = std::numeric_limits<size_t>::max();
};

} // namespace RDMont

// Back-compat alias
template<typename T, typename Grid>
using MonteCarloFunctionality = RDMont::StepResult<T, Grid>;

#endif // RDMONT_STEP_RESULT_HPP
