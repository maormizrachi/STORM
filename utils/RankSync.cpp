#ifdef STORM_WITH_MPI

#include "RankSync.hpp"

namespace STORM {

std::vector<rank_t> GetRanksOrder(const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    rank_t even_comm_size = (size / 2) * 2;

    bool sizeIsEven = (even_comm_size == size);
    std::vector<rank_t> ranks(even_comm_size);
    std::iota(ranks.begin(), ranks.end(), 0);
        
    rank_t lastRank = size - 1;

    std::vector<rank_t> myOrder;
    if(not sizeIsEven)
    {
        if(rank != lastRank)
        {
            myOrder.push_back(rank);
        }
    }
    else
    {
        myOrder.push_back(rank);
    }
    
    rank_t P = even_comm_size / 2;

    for(rank_t i = 0; i < (even_comm_size - 1); i++)
    {
        for(rank_t j = 0; j < P; j++)
        {
            rank_t a = ranks[j];
            rank_t b = ranks[even_comm_size - 1 - j];
            if(rank == a)
            {
                myOrder.push_back(b);
            }
            if(rank == b)
            {
                myOrder.push_back(a);
            }
        }

        std::vector<rank_t> new_ranks = {ranks[0], ranks.back()};
        new_ranks.insert(new_ranks.end(), ranks.begin() + 1, ranks.end() - 1);
        ranks = std::move(new_ranks);
    }

    if(sizeIsEven or (not sizeIsEven and rank != lastRank))
    {
        assert(myOrder.size() == even_comm_size);
    }
    return myOrder;
}

void ForEachRankSync(const MPI_Comm &comm, const std::vector<rank_t> &order, const std::function<void(rank_t)> &func, bool use_barrier)
{
    rank_t rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(const rank_t &_rank : order)
    {
        if(use_barrier) MPI_Barrier(comm);
        func(_rank);
    }

    rank_t lastRank = size - 1;
    rank_t even_comm_size = (size / 2) * 2;
    bool sizeIsEven = (even_comm_size == size);

    if(not sizeIsEven)
    {
        if(use_barrier)
        {
            if(rank == lastRank)
            {
                for(rank_t i = 0; i < even_comm_size; i++)
                {
                    MPI_Barrier(comm); // match with their barrier
                }
            }
        }
        // now everybody need to synchronize with rank `N-1`
        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            if(use_barrier) MPI_Barrier(comm);
            if(rank == lastRank or rank == _rank)
            {
                rank_t otherRank = (rank == _rank)? lastRank : _rank;
                func(otherRank);
            }
        }
    }
    MPI_Barrier(comm);
}

static std::pair<std::vector<rank_t>, int> BuildEdgeSchedule(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int send_count = static_cast<int>(new_neighbors.size());
    std::vector<int> all_counts(static_cast<size_t>(size));
    MPI_Allgather(&send_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<size_t>(size));
    displs[0] = 0;
    for(rank_t i = 1; i < size; i++)
        displs[static_cast<size_t>(i)] = displs[static_cast<size_t>(i - 1)] + all_counts[static_cast<size_t>(i - 1)];
    int total = displs[static_cast<size_t>(size - 1)] + all_counts[static_cast<size_t>(size - 1)];

    std::vector<rank_t> all_neighbors(static_cast<size_t>(total));
    MPI_Allgatherv(new_neighbors.data(), send_count, MPI_INT,
                   all_neighbors.data(), all_counts.data(), displs.data(), MPI_INT, comm);

    std::set<std::pair<rank_t, rank_t>> edge_set;
    for(rank_t r = 0; r < size; r++)
    {
        int offset = displs[static_cast<size_t>(r)];
        int count = all_counts[static_cast<size_t>(r)];
        for(int j = 0; j < count; j++)
        {
            rank_t nbr = all_neighbors[static_cast<size_t>(offset + j)];
            if(r != nbr)
            {
                edge_set.insert({std::min(r, nbr), std::max(r, nbr)});
            }
        }
    }

    std::vector<std::pair<rank_t, rank_t>> edges(edge_set.begin(), edge_set.end());
    size_t num_edges = edges.size();
    if(num_edges == 0)
        return {{}, 0};

    std::vector<std::vector<bool>> vertex_used(static_cast<size_t>(size));
    std::vector<int> edge_color(num_edges);
    int num_colors = 0;

    for(size_t i = 0; i < num_edges; i++)
    {
        auto [u, v] = edges[i];
        auto &used_u = vertex_used[static_cast<size_t>(u)];
        auto &used_v = vertex_used[static_cast<size_t>(v)];
        size_t max_len = std::max(used_u.size(), used_v.size());

        int c = 0;
        for(; static_cast<size_t>(c) < max_len; c++)
        {
            bool taken_u = static_cast<size_t>(c) < used_u.size() && used_u[static_cast<size_t>(c)];
            bool taken_v = static_cast<size_t>(c) < used_v.size() && used_v[static_cast<size_t>(c)];
            if(!taken_u && !taken_v)
                break;
        }

        edge_color[i] = c;
        if(static_cast<size_t>(c) >= used_u.size()) used_u.resize(static_cast<size_t>(c + 1), false);
        if(static_cast<size_t>(c) >= used_v.size()) used_v.resize(static_cast<size_t>(c + 1), false);
        used_u[static_cast<size_t>(c)] = true;
        used_v[static_cast<size_t>(c)] = true;
        num_colors = std::max(num_colors, c + 1);
    }

    std::vector<rank_t> my_rounds(static_cast<size_t>(num_colors), -1);
    for(size_t i = 0; i < num_edges; i++)
    {
        auto [u, v] = edges[i];
        int c = edge_color[i];
        if(rank == u)
            my_rounds[static_cast<size_t>(c)] = v;
        else if(rank == v)
            my_rounds[static_cast<size_t>(c)] = u;
    }

    return {std::move(my_rounds), num_colors};
}

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t, MPI_Comm)> &func, bool /*withBarrier*/)
{
    auto [my_rounds, num_colors] = BuildEdgeSchedule(comm, new_neighbors);

    if(num_colors == 0)
        return;

    rank_t rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Group world_group;
    MPI_Comm_group(comm, &world_group);

    // Edge coloring guarantees each rank has at most one pair per round,
    // so using the round number as tag satisfies MPI's requirement that
    // concurrent MPI_Comm_create_group calls involving the same process
    // use different tags.  No global barrier is needed: the blocking
    // MPI_Comm_create_group call itself synchronizes the two endpoints.
    for(int c = 0; c < num_colors; c++)
    {
        rank_t partner = my_rounds[static_cast<size_t>(c)];
        if(partner >= 0)
        {
            rank_t pair_ranks[2] = {std::min(rank, partner), std::max(rank, partner)};
            MPI_Group pair_group;
            MPI_Group_incl(world_group, 2, pair_ranks, &pair_group);

            MPI_Comm pair_comm;
            MPI_Comm_create_group(comm, pair_group, c, &pair_comm);
            MPI_Group_free(&pair_group);

            if(pair_comm != MPI_COMM_NULL)
            {
                func(partner, pair_comm);
            }
        }
    }

    MPI_Group_free(&world_group);
}

void ForEachRankSyncByList(const MPI_Comm &comm, const std::vector<rank_t> &new_neighbors,
                           const std::function<void(rank_t)> &func, bool /*withBarrier*/)
{
    auto [my_rounds, num_colors] = BuildEdgeSchedule(comm, new_neighbors);

    for(int c = 0; c < num_colors; c++)
    {
        rank_t partner = my_rounds[static_cast<size_t>(c)];
        if(partner >= 0)
        {
            func(partner);
        }
    }
}

} // namespace STORM

#endif // STORM_WITH_MPI