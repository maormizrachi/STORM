#ifndef RDMONT_TYPES_HPP
#define RDMONT_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace RDMont {

using dt_t = double;
using distance_t = double;

#ifdef RDMONT_WITH_MPI
using rank_t = int;
#endif

} // namespace RDMont

#endif // RDMONT_TYPES_HPP
