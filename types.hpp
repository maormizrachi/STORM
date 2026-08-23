#ifndef STORM_TYPES_HPP
#define STORM_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace STORM {

using dt_t = double;
using particle_id_t = std::uint64_t;
using cell_id_t = std::uint64_t;
using cell_index_t = std::uint64_t;
using particle_step_t = std::uint64_t;

static_assert(sizeof(particle_id_t) == 8, "particle_id_t is part of the packet ABI");
static_assert(sizeof(cell_id_t) == 8, "cell_id_t is part of the packet ABI");
static_assert(sizeof(cell_index_t) == 8, "cell_index_t is part of the packet ABI");
static_assert(sizeof(particle_step_t) == 8, "particle_step_t is part of the packet ABI");

#ifdef STORM_WITH_MPI
using rank_t = std::int32_t;
static_assert(sizeof(rank_t) == 4, "rank_t is part of the packet ABI");
#endif

} // namespace STORM

#endif // STORM_TYPES_HPP
