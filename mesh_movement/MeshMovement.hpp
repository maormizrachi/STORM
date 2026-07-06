#ifndef STORM_MESH_MOVEMENT_HPP
#define STORM_MESH_MOVEMENT_HPP

#include <vector>
#include "../particle/Particle.hpp"

#ifdef STORM_WITH_MPI
#include <mpi.h>
#include <mpi_utils/serialize/Serializer.hpp>
#include <mpi_utils/mpi_alltoall.hpp>
#endif // STORM_WITH_MPI

namespace STORM {

template<typename PointT, typename Grid>
struct MeshMovement
{
    using ParticleT = STORM::Particle<PointT, Grid>;

    static void UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles);

#ifdef STORM_WITH_MPI
    static void UpdateNewCellsAfterExchange(const Grid &grid, std::vector<ParticleT> &particles);
#endif // STORM_WITH_MPI
};

// ============================================================
// Generic default: UpdateNewCells
// ============================================================

template<typename PointT, typename Grid>
void MeshMovement<PointT, Grid>::UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles)
{
    size_t N = grid.GetPointNo();
    if(N == 0)
    {
#ifndef STORM_WITH_MPI
        particles.clear();
        return;
#endif
    }

#ifdef STORM_WITH_MPI
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<ParticleT> resolvedLocal;
    std::vector<Serializer> senders(worldSize);

    for(ParticleT &p : particles)
    {
        if(p.cellIndex < N && grid.IsPointInCell(p.location, p.cellIndex))
        {
            resolvedLocal.push_back(p);
            continue;
        }

        if(grid.IsPointOutsideBox(p.location))
        {
            continue;
        }

        size_t candidate = grid.GetContainingCell(p.location);
        if(candidate < N && grid.IsPointInCell(p.location, candidate))
        {
            p.cellIndex = candidate;
            resolvedLocal.push_back(p);
            continue;
        }

        if(!grid.PointInMyDomain(p.location))
        {
            rank_t owner = static_cast<rank_t>(grid.GetOwner(p.location));
            if(owner >= 0 && owner < worldSize && owner != rank)
            {
                senders[owner].insert(p);
                continue;
            }
        }

        if(candidate < N)
        {
            p.cellIndex = candidate;
            resolvedLocal.push_back(p);
        }
    }

    std::vector<std::vector<ParticleT>> received = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);

    for(rank_t r = 0; r < worldSize; ++r)
    {
        for(ParticleT &p : received[r])
        {
            if(N == 0)
            {
                continue;
            }

            size_t candidate = grid.GetContainingCell(p.location);
            if(candidate < N && grid.IsPointInCell(p.location, candidate))
            {
                p.cellIndex = candidate;
                resolvedLocal.push_back(p);
            }
        }
    }

    particles = std::move(resolvedLocal);

#else // serial
    for(ParticleT &p : particles)
    {
        if(p.cellIndex < N && grid.IsPointInCell(p.location, p.cellIndex))
        {
            continue;
        }

        size_t candidate = grid.GetContainingCell(p.location);
        p.cellIndex = candidate;
    }
#endif // STORM_WITH_MPI
}

// ============================================================
// Generic default: UpdateNewCellsAfterExchange
// ============================================================

#ifdef STORM_WITH_MPI

template<typename PointT, typename Grid>
void MeshMovement<PointT, Grid>::UpdateNewCellsAfterExchange(const Grid &grid, std::vector<ParticleT> &particles)
{
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const std::vector<int> sentProcs = grid.GetSentProcs();
    const std::vector<std::vector<size_t>> &sentPoints = grid.GetSentPoints();
    const std::vector<size_t> &selfIndex = grid.GetSelfIndex();

    size_t totalOriginal = selfIndex.size();
    for(const std::vector<size_t> &sent : sentPoints)
    {
        totalOriginal += sent.size();
    }

    // Build translation map: oldCellIndex -> (destRank, newCellIndex)
    // For local cells: selfIndex[newLocalIdx] = oldIdx, new index is known
    // For sent cells: new index on the remote rank is unknown, resolved on arrival
    std::vector<std::pair<rank_t, size_t>> translationMap(totalOriginal, {-1, std::numeric_limits<size_t>::max()});

    for(size_t newLocalIdx = 0; newLocalIdx < selfIndex.size(); ++newLocalIdx)
    {
        size_t oldIdx = selfIndex[newLocalIdx];
        if(oldIdx < translationMap.size())
        {
            translationMap[oldIdx] = {rank, newLocalIdx};
        }
    }

    for(size_t procIdx = 0; procIdx < sentProcs.size(); ++procIdx)
    {
        rank_t destRank = sentProcs[procIdx];
        for(size_t j = 0; j < sentPoints[procIdx].size(); ++j)
        {
            size_t oldIdx = sentPoints[procIdx][j];
            if(oldIdx < translationMap.size())
            {
                translationMap[oldIdx] = {destRank, std::numeric_limits<size_t>::max()};
            }
        }
    }

    std::vector<Serializer> senders(worldSize);
    std::vector<ParticleT> selfParticles;

    for(ParticleT &p : particles)
    {
        size_t oldIdx = p.cellIndex;
        if(oldIdx < translationMap.size())
        {
            auto [destRank, newIdx] = translationMap[oldIdx];
            if(newIdx != std::numeric_limits<size_t>::max())
            {
                p.cellIndex = newIdx;
            }
            if(destRank == rank)
            {
                selfParticles.push_back(p);
            }
            else if(destRank >= 0 && destRank < worldSize)
            {
                senders[destRank].insert(p);
            }
        }
    }

    std::vector<std::vector<ParticleT>> received = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);

    particles = std::move(selfParticles);

    size_t N = grid.GetPointNo();
    for(rank_t r = 0; r < worldSize; ++r)
    {
        for(ParticleT &p : received[r])
        {
            if(p.cellIndex >= N || p.cellIndex == std::numeric_limits<size_t>::max())
            {
                size_t candidate = grid.GetContainingCell(p.location);
                if(candidate < N)
                {
                    p.cellIndex = candidate;
                }
            }
            particles.push_back(p);
        }
    }
}

#endif // STORM_WITH_MPI

// ============================================================
// Free function wrappers
// ============================================================

template<typename PointT, typename Grid>
void UpdateNewCells(const Grid &grid, std::vector<Particle<PointT, Grid>> &particles)
{
    MeshMovement<PointT, Grid>::UpdateNewCells(grid, particles);
}

#ifdef STORM_WITH_MPI

template<typename PointT, typename Grid>
void UpdateNewCellsAfterExchange(const Grid &grid, std::vector<Particle<PointT, Grid>> &particles)
{
    MeshMovement<PointT, Grid>::UpdateNewCellsAfterExchange(grid, particles);
}

#endif // STORM_WITH_MPI

} // namespace STORM

#endif // STORM_MESH_MOVEMENT_HPP
