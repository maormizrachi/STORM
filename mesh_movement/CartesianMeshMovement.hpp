#ifndef STORM_CARTESIAN_MESH_MOVEMENT_HPP
#define STORM_CARTESIAN_MESH_MOVEMENT_HPP

#if __has_include(<MadCart/CartesianMesh3D.hpp>)

#include "MeshMovement.hpp"
#include <MadCart/CartesianMesh3D.hpp>

namespace STORM {

// ============================================================
// Partial specialization for CartesianMesh3D
// ============================================================

template<typename PointT>
struct MeshMovement<PointT, MadCart::CartesianMesh3D<PointT>>
{
    using Grid = MadCart::CartesianMesh3D<PointT>;
    using ParticleT = STORM::Particle<PointT, Grid>;

    static void UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles);

#ifdef STORM_WITH_MPI
    static void UpdateNewCellsAfterExchange(const Grid &grid, std::vector<ParticleT> &particles);
#endif // STORM_WITH_MPI
};

// ============================================================
// UpdateNewCells -- Cartesian floor indexing is exact
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadCart::CartesianMesh3D<PointT>>::UpdateNewCells(
    const Grid &grid, std::vector<ParticleT> &particles)
{
    size_t N = grid.GetPointNo();

#ifdef STORM_WITH_MPI
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<ParticleT> resolvedLocal;
    std::vector<Serializer> senders(worldSize);

    for(ParticleT &p : particles)
    {
        if(grid.IsPointOutsideBox(p.location))
        {
            continue;
        }

        if(grid.PointInMyDomain(p.location))
        {
            size_t cell = grid.GetContainingCell(p.location);
            if(cell < N)
            {
                p.cellIndex = cell;
                resolvedLocal.push_back(p);
            }
        }
        else
        {
            rank_t owner = static_cast<rank_t>(grid.GetOwner(p.location));
            if(owner >= 0 && owner < worldSize && owner != rank)
            {
                senders[owner].insert(p);
            }
        }
    }

    std::vector<std::vector<ParticleT>> received = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);

    for(rank_t r = 0; r < worldSize; ++r)
    {
        for(ParticleT &p : received[r])
        {
            size_t cell = grid.GetContainingCell(p.location);
            if(cell < N)
            {
                p.cellIndex = cell;
                resolvedLocal.push_back(p);
            }
        }
    }

    particles = std::move(resolvedLocal);

#else // serial
    auto it = particles.begin();
    while(it != particles.end())
    {
        if(grid.IsPointOutsideBox(it->location))
        {
            it = particles.erase(it);
            continue;
        }
        it->cellIndex = grid.GetContainingCell(it->location);
        ++it;
    }
#endif // STORM_WITH_MPI
}

// ============================================================
// UpdateNewCellsAfterExchange
// Uses inlined ExchangeChain logic (same as Voronoi version)
// ============================================================

#ifdef STORM_WITH_MPI

template<typename PointT>
void MeshMovement<PointT, MadCart::CartesianMesh3D<PointT>>::UpdateNewCellsAfterExchange(
    const Grid &grid, std::vector<ParticleT> &particles)
{
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const std::vector<int> sentProcs = grid.GetSentProcs();
    const std::vector<std::vector<size_t>> &sentPoints = grid.GetSentPoints();
    const std::vector<size_t> &selfIndex = grid.GetSelfIndex();

    // Build translation map using ExchangeChain algorithm
    std::vector<size_t> sendAmounts(worldSize, 0);
    for(size_t i = 0; i < sentProcs.size(); i++)
    {
        rank_t destRank = sentProcs[i];
        sendAmounts[destRank] = sentPoints[i].size();
    }

    std::vector<size_t> recvAmounts(worldSize);
    MPI_Alltoall(sendAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, recvAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    std::vector<size_t> offsets(worldSize, 0);
    size_t currentOffset = selfIndex.size();
    for(size_t i = 0; i < sentProcs.size(); i++)
    {
        rank_t destRank = sentProcs[i];
        offsets[destRank] = currentOffset;
        currentOffset += recvAmounts[destRank];
    }

    std::vector<size_t> otherRanksOffsets(worldSize, 0);
    MPI_Alltoall(offsets.data(), 1, MPI_UNSIGNED_LONG_LONG, otherRanksOffsets.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> translationMap;
    for(size_t i = 0; i < selfIndex.size(); i++)
    {
        size_t index = selfIndex[i];
        translationMap[index] = {rank, i};
    }
    for(size_t i = 0; i < sentProcs.size(); i++)
    {
        rank_t destRank = sentProcs[i];
        size_t rankOffset = otherRanksOffsets[destRank];
        for(size_t j = 0; j < sentPoints[i].size(); j++)
        {
            size_t index = sentPoints[i][j];
            translationMap[index] = {destRank, rankOffset + j};
        }
    }

    // Transfer particles using translation map
    std::vector<Serializer> senders(worldSize);
    std::vector<ParticleT> selfParticles;
    for(ParticleT &p : particles)
    {
        size_t particleCellIdx = p.cellIndex;
        auto it = translationMap.find(particleCellIdx);
        if(it != translationMap.end())
        {
            auto [newRank, newCellIdx] = it->second;
            p.cellIndex = newCellIdx;
            if(newRank == rank)
            {
                selfParticles.push_back(p);
            }
            else
            {
                senders[newRank].insert(p);
            }
        }
    }

    std::vector<std::vector<ParticleT>> allNewParticles = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);

    particles = std::move(selfParticles);
    for(const std::vector<ParticleT> &procParticles : allNewParticles)
    {
        particles.insert(particles.end(), procParticles.cbegin(), procParticles.cend());
    }
    particles.shrink_to_fit();
}

#endif // STORM_WITH_MPI

} // namespace STORM

#endif // __has_include(<MadCart/CartesianMesh3D.hpp>)

#endif // STORM_CARTESIAN_MESH_MOVEMENT_HPP
