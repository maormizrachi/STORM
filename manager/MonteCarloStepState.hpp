#ifndef STORM_MONTE_CARLO_STEP_STATE_HPP
#define STORM_MONTE_CARLO_STEP_STATE_HPP

#include <cstddef>
#include <vector>

namespace STORM {

template<typename ParticleT>
struct MonteCarloStepState
{
    std::vector<ParticleT> remaining;
    size_t leavingCount = 0;
};

} // namespace STORM

#endif // STORM_MONTE_CARLO_STEP_STATE_HPP
