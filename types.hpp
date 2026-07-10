#ifndef STORM_TYPES_HPP
#define STORM_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace STORM {

using dt_t = double;

#ifdef STORM_WITH_MPI
using rank_t = int;
#endif

} // namespace STORM

#endif // STORM_TYPES_HPP
