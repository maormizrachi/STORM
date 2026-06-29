#ifndef RANK_SYNC_HPP
#define RANK_SYNC_HPP

#ifdef STORM_WITH_MPI

#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <functional>
#include <utility>
#include <set>

using rank_t = int;

namespace STORM {

std::vector<rank_t> GetRanksOrder(const MPI_Comm &comm);

void ForEachRankSync(const MPI_Comm &comm, const std::vector<rank_t> &order, const std::function<void(rank_t)> &func, bool use_barrier = true);

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t, MPI_Comm)> &func, bool withBarrier = true);

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t)> &func, bool withBarrier = true);

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // RANK_SYNC_HPP