#ifndef STORM_VORONOI_MESH_MOVEMENT_HPP
#define STORM_VORONOI_MESH_MOVEMENT_HPP

#if __has_include(<MadVoro/Voronoi3D.hpp>)

#include "MeshMovement.hpp"
#include <MadVoro/Voronoi3D.hpp>
#include <spatial_ds/OctTree/OctTree.hpp>
#include <spatial_ds/utils/BoundingBox.hpp>
#include <MadVoro/range/finders/utils/IndexedVector.hpp>
#include <cassert>
#include <iostream>
#include <chrono>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>

#ifdef STORM_WITH_MPI
#include <MeshDecomposer3D/environment/EnvironmentAgent.hpp>
#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>
#endif // STORM_WITH_MPI

#ifndef START_TIMER
#define START_TIMER(msg) (void)0
#define STORM_DEFINED_START_TIMER
#endif
#ifndef START_TIMER_PREEMPTIVE
#define START_TIMER_PREEMPTIVE(msg) (void)0
#define STORM_DEFINED_START_TIMER_PREEMPTIVE
#endif

namespace STORM {

namespace mesh_movement_detail {

constexpr double UPDATE_NEW_CELLS_BOX_EPS_FACTOR = 16.0;
constexpr int RADIUSES_FACTOR = 2;

} // namespace mesh_movement_detail

// ============================================================
// Partial specialization for Voronoi3D
// ============================================================

template<typename PointT>
struct MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>
{
    using Grid = MadVoro::Voronoi3D<PointT>;
    using ParticleT = STORM::Particle<PointT, Grid>;
    using IVec = IndexedVector<PointT>;

    static void UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles, const std::vector<size_t> &cellIDs);
    static void UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles);

    static void AssertLocations(const Grid &grid, const std::vector<ParticleT> &particles);

#ifdef STORM_WITH_MPI
    static void UpdateNewCellsAfterExchange(const Grid &grid, std::vector<ParticleT> &particles);

private:
    static void InternalMovements(const Grid &grid, std::vector<ParticleT> &particles, const std::vector<size_t> &cellIDs);
    static void FirstInaccurateMovements(const Grid &grid, std::vector<ParticleT> &particles);
    static size_t ResolveRemainingParticles(const Grid &grid, std::vector<ParticleT> &particles, const OctTree<IVec> &octTree);
    static void TransferParticlesWithTranslationMap(const Grid &grid, std::vector<ParticleT> &particles,
                                                    const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &cellsTranslation);
#endif // STORM_WITH_MPI
};

// ============================================================
// AssertLocations
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::AssertLocations(const Grid &grid, const std::vector<ParticleT> &particles)
{
    START_TIMER("Assert Locations");
    size_t N = grid.GetPointNo();
    size_t Nparticles = particles.size();
    for(size_t i = 0; i < Nparticles; i++)
    {
        const ParticleT &particle = particles[i];
        size_t cellIndex = particle.cellIndex;
        if(cellIndex >= N)
        {
            StormError eo("AssertLocations: Particle cell index is out of range");
            eo.addEntry("Particle Index", i);
            eo.addEntry("Cell Index", cellIndex);
            eo.addEntry("N", N);
            throw eo;
        }
        if(not grid.IsPointInCell(particle.location, cellIndex))
        {
            StormError eo("AssertLocations: Particle is not in its cell");
            eo.addEntry("Particle Index", i);
            eo.addEntry("Cell Index", cellIndex);
            eo.addEntry("Cell Point", grid.GetMeshPoint(cellIndex));
            throw eo;
        }
    }
}

// ============================================================
// UpdateNewCells -- serial + MPI paths
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles, const std::vector<size_t> &cellIDs)
{
    START_TIMER("Update New Cells");

    try
    {
        size_t N = grid.GetPointNo();
#ifndef STORM_WITH_MPI
        if(N == 0)
        {
            if(particles.empty())
            {
                return;
            }
            StormError eo("UpdateNewCells: particles remain on a rank with no local cells");
            eo.addEntry("Particle count", particles.size());
            throw eo;
        }
        for(ParticleT &p : particles)
        {
            if(p.cellIndex < N)
            {
                if(grid.IsPointInCell(p.location, p.cellIndex))
                {
                    continue;
                }
            }
            p.cellIndex = grid.GetContainingCell(p.location);
        }
#else // STORM_WITH_MPI
        rank_t rank, worldSize;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        InternalMovements(grid, particles, cellIDs);

        START_TIMER_PREEMPTIVE("Local Trees Construction");

        auto [tess_ll, tess_ur] = grid.GetBoxCoordinates();
        PointT ll = tess_ll;
        PointT ur = tess_ur;
        if(N > 0)
        {
            ll = PointT(std::numeric_limits<double>::max());
            ur = PointT(std::numeric_limits<double>::lowest());

            const std::vector<PointT> &vertices = grid.GetFacePoints();
            for(const auto &vec : grid.GetAllPointsInFace())
            {
                for(size_t pointIdx : vec)
                {
                    const PointT &p = vertices[pointIdx];
                    ll.x = std::min(ll.x, p.x);
                    ll.y = std::min(ll.y, p.y);
                    ll.z = std::min(ll.z, p.z);
                    ur.x = std::max(ur.x, p.x);
                    ur.y = std::max(ur.y, p.y);
                    ur.z = std::max(ur.z, p.z);
                }
            }

            PointT boxsize = ur - ll;
            ll -= EPSILON * boxsize;
            ur += EPSILON * boxsize;
        }

        PointT tess_boxsize = tess_ur - tess_ll;
        tess_ll -= mesh_movement_detail::UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
        tess_ur += mesh_movement_detail::UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
        BoundingBox<PointT> bb(tess_ll, tess_ur);
        BoundingBox<PointT> subBox(ll, ur);

        if(not bb.contains(subBox))
        {
            StormError eo("UpdateNewCells: Sub-box is not contained within the main bounding box");
            throw eo;
        }

        OctTree<IVec> octTree(IVec(ll, std::numeric_limits<size_t>::max()), IVec(ur, std::numeric_limits<size_t>::max()));

        for(size_t i = 0; i < N; i++)
        {
            octTree.insert(IVec(grid.GetMeshPoint(i), i));
        }
        assert(octTree.getSize() == N);

        std::vector<ParticleT> shouldExchangeParticles;

        std::vector<double> halfMinNeighborDist2(N);
        for(size_t i = 0; i < N; i++)
        {
            double minDist2 = std::numeric_limits<double>::max();
            const PointT &gi = grid.GetMeshPoint(i);
            for(size_t faceIdx : grid.GetCellFaces(i))
            {
                const auto &[n1, n2] = grid.GetFaceNeighbors(faceIdx);
                size_t neighbor = (n1 == i) ? n2 : n1;
                PointT diff = gi - grid.GetMeshPoint(neighbor);
                double d2 = ScalarProd(diff, diff);
                if(d2 < minDist2)
                {
                    minDist2 = d2;
                }
            }
            halfMinNeighborDist2[i] = minDist2 * 0.25;
        }

        START_TIMER_PREEMPTIVE("Self Update");
        size_t writeIdx = 0;
        if(octTree.getSize() > 0)
        {
            for(size_t i = 0; i < particles.size(); i++)
            {
                ParticleT &p = particles[i];
                bool keep = false;

                if(p.cellIndex < N)
                {
                    PointT diff = p.location - grid.GetMeshPoint(p.cellIndex);
                    double d2 = ScalarProd(diff, diff);
                    if(d2 < halfMinNeighborDist2[p.cellIndex])
                    {
                        keep = true;
                    }
                    else if(grid.IsPointInCell(p.location, p.cellIndex))
                    {
                        keep = true;
                    }
                }

                if(not keep)
                {
                    if(grid.IsPointOutsideBox(p.location))
                    {
                        StormError eo("Particle location is outside of the bounding box");
                        eo.addEntry("Particle id", p.id);
                        eo.addEntry("Location", p.location);
                        throw eo;
                    }

                    auto twoClosest = octTree.getKClosestPoints(p.location, 2);
                    for(const auto &[cell, dist] : twoClosest)
                    {
                        size_t index = cell.getIndex();
                        if(grid.IsPointInCell(p.location, index))
                        {
                            p.cellIndex = index;
                            keep = true;
                            break;
                        }
                    }
                }

                if(keep)
                {
                    if(writeIdx != i)
                    {
                        particles[writeIdx] = std::move(p);
                    }
                    writeIdx++;
                }
                else
                {
                    shouldExchangeParticles.push_back(std::move(p));
                }
            }
        }
        else
        {
            shouldExchangeParticles = std::move(particles);
            writeIdx = 0;
        }

        size_t keptCount = writeIdx;
        particles.resize(keptCount);

        FirstInaccurateMovements(grid, shouldExchangeParticles);
        MPI_Barrier(MPI_COMM_WORLD);

        START_TIMER_PREEMPTIVE("Main Loop");

        size_t iterations = ResolveRemainingParticles(grid, shouldExchangeParticles, octTree);

        if(rank == 0)
        {
            std::cout << "Rank " << rank << ", done UpdateNewCells, iterations is " << iterations << std::endl;
        }

        particles.insert(particles.end(),
            std::make_move_iterator(shouldExchangeParticles.begin()),
            std::make_move_iterator(shouldExchangeParticles.end()));
#endif // STORM_WITH_MPI

#ifndef STORM_WITH_MPI
        AssertLocations(grid, particles);
#endif
    }
    catch(const StormError &eo)
    {
        throw;
    }
}

// ============================================================
// UpdateNewCells -- convenience overload without cellIDs
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::UpdateNewCells(const Grid &grid, std::vector<ParticleT> &particles)
{
    size_t N = grid.GetPointNo();
    std::vector<size_t> cellIDs(N);
    for(size_t i = 0; i < N; i++)
    {
        cellIDs[i] = i;
    }
    UpdateNewCells(grid, particles, cellIDs);
}

// ============================================================
// MPI-only helpers
// ============================================================

#ifdef STORM_WITH_MPI

// ============================================================
// InternalMovements -- fast remap via persistent cellID
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::InternalMovements(const Grid &grid, std::vector<ParticleT> &particles, const std::vector<size_t> &cellIDs)
{
    START_TIMER("Internal Movement");
    size_t count = 0;
    boost::container::flat_map<size_t, size_t> cellIDtoIndex;
    size_t N = cellIDs.size();
    for(size_t i = 0; i < N; i++)
    {
        cellIDtoIndex[cellIDs.at(i)] = i;
    }

    for(ParticleT &particle : particles)
    {
        size_t cellID = particle.cellID;

        auto it = cellIDtoIndex.find(cellID);
        if(it == cellIDtoIndex.cend())
        {
            continue;
        }

        count++;
        size_t newCellIndex = (*it).second;
        if(particle.cellIndex != newCellIndex)
        {
            particle.cellIndex = newCellIndex;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    rank_t r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if(r == 0)
    {
        std::cout << "Internal movements (determined by cell ID) counted for " << count << " particles" << std::endl;
    }
}

// ============================================================
// FirstInaccurateMovements
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::FirstInaccurateMovements(const Grid &grid, std::vector<ParticleT> &particles)
{
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const std::shared_ptr<EnvironmentAgent<PointT>> &envAgent = grid.GetEnvironmentAgent();
    std::vector<ParticleT> newParticles;
    std::vector<Serializer> senders(worldSize);

    auto start = std::chrono::high_resolution_clock::now();
    size_t sentCounter = 0;
    for(ParticleT &p : particles)
    {
        rank_t approxOwner = envAgent->getOwner(p.location);
        if(approxOwner == rank)
        {
            newParticles.push_back(p);
        }
        else
        {
            senders[approxOwner].insert(p);
            sentCounter++;
        }
    }

    size_t Nparticles = particles.size();
    (void)Nparticles;
    particles.clear();
    auto end = std::chrono::high_resolution_clock::now();
    double timeInLoop1 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    (void)timeInLoop1;
    MPI_Barrier(MPI_COMM_WORLD);

    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ParticleT>> receiveValues = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    double timeInExchange = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    (void)timeInExchange;
    MPI_Barrier(MPI_COMM_WORLD);

    start = std::chrono::high_resolution_clock::now();
    for(rank_t _rank = 0; _rank < worldSize; _rank++)
    {
        const std::vector<ParticleT> &particlesFromRank = receiveValues[_rank];
        newParticles.insert(newParticles.end(), particlesFromRank.cbegin(), particlesFromRank.cend());
    }
    particles = std::move(newParticles);

    MPI_Allreduce(MPI_IN_PLACE, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    double timeInLoop2 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    (void)timeInLoop2;

    if(rank == 0)
    {
        std::cout << "First inaccurate movements sent for " << sentCounter << " particles" << std::endl;
    }
}

// ============================================================
// ResolveRemainingParticles
// ============================================================

template<typename PointT>
size_t MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::ResolveRemainingParticles(
    const Grid &grid, std::vector<ParticleT> &particles, const OctTree<IVec> &octTree)
{
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t N = grid.GetPointNo();

    auto [tess_ll, tess_ur] = grid.GetBoxCoordinates();
    PointT tess_boxsize = tess_ur - tess_ll;
    tess_ll -= mesh_movement_detail::UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
    tess_ur += mesh_movement_detail::UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
    OctTree<IVec> wideTree(IVec(tess_ll, std::numeric_limits<size_t>::max()), IVec(tess_ur, std::numeric_limits<size_t>::max()));
    for(size_t i = 0; i < N; i++)
    {
        wideTree.insert(IVec(grid.GetMeshPoint(i), i));
    }
    DistributedOctTree<IVec> distributedOctTree(&wideTree);

    double avgCellSize = 0;
    if(N > 0)
    {
        for(size_t i = 0; i < N; i++)
        {
            avgCellSize += grid.GetWidth(i);
        }
        avgCellSize /= N;
    }
    double avgOfAvgCellSize = avgCellSize;
    MPI_Allreduce(MPI_IN_PLACE, &avgOfAvgCellSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avgOfAvgCellSize /= worldSize;
    double initialRadius = avgCellSize * mesh_movement_detail::RADIUSES_FACTOR;
    if(N == 0)
    {
        initialRadius = avgOfAvgCellSize * mesh_movement_detail::RADIUSES_FACTOR;
    }

    std::vector<ParticleT> resolvedParticles;
    boost::container::flat_set<size_t> particlesLeft;
    std::vector<boost::container::flat_set<rank_t>> ranksTested;
    for(size_t i = 0; i < particles.size(); i++)
    {
        ParticleT &p = particles[i];
        size_t closestCell = std::numeric_limits<size_t>::max();
        if(N > 0)
        {
            closestCell = grid.GetContainingCell(p.location);
        }
        if(closestCell < N and grid.IsPointInCell(p.location, closestCell))
        {
            p.cellIndex = closestCell;
            resolvedParticles.push_back(p);
            ranksTested.push_back({});
        }
        else
        {
            particlesLeft.insert(i);
            ranksTested.push_back({rank});
        }
    }
    {
        size_t numParticlesLeft = particlesLeft.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesLeft, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0)
        {
            std::cout << "Number of particles left to determination: " << numParticlesLeft << std::endl;
        }
    }

    std::vector<double> radiuses(particles.size(), initialRadius);
    std::vector<std::vector<ParticleT>> sendValues(worldSize);
    std::vector<std::vector<size_t>> sendIndicesCpy(worldSize);
    std::vector<std::vector<size_t>> acknowledgementValues(worldSize);

    size_t iterations = 0;

    while(true)
    {
        iterations++;
        size_t localLeftParticles = particlesLeft.size();
        size_t globalLeftParticles;
        MPI_Allreduce(&localLeftParticles, &globalLeftParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(globalLeftParticles == 0)
        {
            break;
        }

        for(rank_t _rank = 0; _rank < worldSize; _rank++)
        {
            sendValues[_rank].clear();
            sendIndicesCpy[_rank].clear();
            acknowledgementValues[_rank].clear();
        }

        for(size_t i : particlesLeft)
        {
            ParticleT &p = particles[i];
            bool atLeastOneNew = false;
            while(not atLeastOneNew)
            {
                if(static_cast<rank_t>(ranksTested[i].size()) == worldSize)
                {
                    StormError eo("UpdateNewCells: All ranks were tested for particle");
                    eo.addEntry("Cell Index", p.cellIndex);
                    eo.addEntry("N", N);
                    size_t closestPointIdx = grid.GetContainingCell(p.location);
                    eo.addEntry("Local Closest Local Point", closestPointIdx);
                    eo.addEntry("Local Closest Point Value", grid.GetMeshPoint(closestPointIdx));
                    eo.addEntry("Rank", rank);
                    throw eo;
                }

                auto intersectingRanks = distributedOctTree.getIntersectingRanks(p.location, radiuses[i]);
                for(rank_t _rank : intersectingRanks)
                {
                    if(ranksTested[i].find(_rank) == ranksTested[i].end())
                    {
                        atLeastOneNew = true;
                        sendValues[_rank].push_back(p);
                        sendIndicesCpy[_rank].push_back(i);
                        ranksTested[i].insert(_rank);
                    }
                }
                radiuses[i] *= mesh_movement_detail::RADIUSES_FACTOR;
            }
        }

        std::vector<std::vector<ParticleT>> receiveValues = MPI_Exchange_all_to_all_sparse(sendValues, MPI_COMM_WORLD);
        assert(static_cast<rank_t>(receiveValues.size()) == worldSize);

        if(octTree.getSize() > 0)
        {
            for(rank_t _rank = 0; _rank < worldSize; _rank++)
            {
                std::vector<ParticleT> &particlesFromRank = receiveValues[_rank];
                size_t Np = particlesFromRank.size();
                for(size_t i = 0; i < Np; i++)
                {
                    ParticleT &p = particlesFromRank[i];
                    auto candidates = octTree.getKClosestPoints(p.location, 2);
                    for(const auto &[cell, dist] : candidates)
                    {
                        size_t index = cell.getIndex();
                        if(index < N and grid.IsPointInCell(p.location, index))
                        {
                            p.cellIndex = index;
                            resolvedParticles.push_back(p);
                            acknowledgementValues[_rank].push_back(i);
                            break;
                        }
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> acknowledgements = MPI_Exchange_all_to_all_sparse(acknowledgementValues, MPI_COMM_WORLD);
        assert(static_cast<rank_t>(acknowledgements.size()) == worldSize);

        std::vector<size_t> toErase;
        for(rank_t _rank = 0; _rank < worldSize; _rank++)
        {
            for(size_t i : acknowledgements[_rank])
            {
                toErase.push_back(sendIndicesCpy[_rank][i]);
            }
        }
        std::sort(toErase.begin(), toErase.end());
        toErase.erase(std::unique(toErase.begin(), toErase.end()), toErase.end());

        std::vector<size_t> remainingVec;
        remainingVec.reserve(particlesLeft.size());
        std::set_difference(particlesLeft.begin(), particlesLeft.end(), toErase.begin(), toErase.end(), std::back_inserter(remainingVec));
        particlesLeft = boost::container::flat_set<size_t>(boost::container::ordered_unique_range_t{}, remainingVec.begin(), remainingVec.end());
    }

    particles = std::move(resolvedParticles);
    return iterations;
}

// ============================================================
// TransferParticlesWithTranslationMap
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::TransferParticlesWithTranslationMap(
    const Grid &grid, std::vector<ParticleT> &particles,
    const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &cellsTranslation)
{
    rank_t rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    START_TIMER("Prepare Transfer Data");

    std::vector<Serializer> senders(worldSize);
    std::vector<ParticleT> selfParticles;
    size_t sentCounter = 0;
    for(ParticleT &p : particles)
    {
        size_t particleCellIdx = p.cellIndex;
        auto [newRank, newCellIdx] = cellsTranslation.at(particleCellIdx);
        p.cellIndex = newCellIdx;
        if(newRank == rank)
        {
            selfParticles.push_back(p);
        }
        else
        {
            senders[newRank].insert(p);
            sentCounter++;
        }
    }

    particles.clear();

    std::vector<std::vector<ParticleT>> allNewParticles;
    {
        START_TIMER_PREEMPTIVE("Particles Exchange");
        allNewParticles = MPI_Exchange_all_to_all_serializers<ParticleT>(senders, MPI_COMM_WORLD);
    }

    size_t receivedCounter = 0;
    particles = std::move(selfParticles);
    std::for_each(allNewParticles.cbegin(), allNewParticles.cend(), [&particles, &receivedCounter](const std::vector<ParticleT> &procParticles)
    {
        receivedCounter += procParticles.size();
        particles.insert(particles.end(), procParticles.cbegin(), procParticles.cend());
    });

    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &sentCounter, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &receivedCounter, &receivedCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    particles.shrink_to_fit();
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// UpdateNewCellsAfterExchange
// Inlines the ExchangeChain logic from RICH since STORM
// does not have ExchangeChain.
// ============================================================

template<typename PointT>
void MeshMovement<PointT, MadVoro::Voronoi3D<PointT>>::UpdateNewCellsAfterExchange(
    const Grid &grid, std::vector<ParticleT> &particles)
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

    // Build the translation map using the same algorithm as
    // ExchangeChain::UpdateTransferMap in RICH.
    // Step 1: exchange send counts
    std::vector<size_t> sendAmounts(worldSize, 0);
    for(size_t i = 0; i < sentProcs.size(); i++)
    {
        rank_t destRank = sentProcs[i];
        sendAmounts[destRank] = sentPoints[i].size();
    }

    std::vector<size_t> recvAmounts(worldSize);
    MPI_Alltoall(sendAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, recvAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    // Step 2: compute where my sent items land on each destination rank
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

    // Step 3: build translation map oldIndex -> (destRank, newIndex)
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> translationMap;
    translationMap.reserve(totalOriginal);

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

    TransferParticlesWithTranslationMap(grid, particles, translationMap);
}

#endif // STORM_WITH_MPI

} // namespace STORM

#ifdef STORM_DEFINED_START_TIMER
#undef START_TIMER
#undef STORM_DEFINED_START_TIMER
#endif
#ifdef STORM_DEFINED_START_TIMER_PREEMPTIVE
#undef START_TIMER_PREEMPTIVE
#undef STORM_DEFINED_START_TIMER_PREEMPTIVE
#endif

#endif // __has_include(<MadVoro/Voronoi3D.hpp>)

#endif // STORM_VORONOI_MESH_MOVEMENT_HPP
