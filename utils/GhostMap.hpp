#ifndef GHOST_MAP_HPP
#define GHOST_MAP_HPP

#include <cassert>
#include <utility>
#include "../manager/MonteCarloConfig.hpp"
#include "../elementary/PointOps.hpp"

#ifdef STORM_WITH_MPI

#include <mpi_utils/mpi_commands.hpp>
#include <boost/container/flat_map.hpp>
#include "../StormError.hpp"

#endif // STORM_WITH_MPI
namespace STORM {

template<typename Grid, typename Point>
void ApplyPeriodicCellMove(const Grid &grid, Point &location, size_t imageIndex, const Point &velocity)
{
    double speed = fastabs(velocity);
    if(speed > 0.0)
    {
        double push = MONTECARLO_EPSILON / speed;
        location.x += velocity.x * push;
        location.y += velocity.y * push;
        location.z += velocity.z * push;
    }
    else
    {
        location = (1 - MONTECARLO_EPSILON) * location + MONTECARLO_EPSILON * grid.GetMeshPoint(imageIndex);
    }
    grid.WrapPeriodicPoint(location);
}

template<typename Grid>
size_t ResolvePhysicalCellIndex(const Grid &grid, size_t meshIndex, size_t cellNumber)
{
    if(meshIndex < cellNumber)
    {
        return meshIndex;
    }
    if(!grid.IsPeriodicImage(meshIndex))
    {
        return meshIndex;
    }
    const size_t physical = grid.ResolvePeriodicImageIndex(meshIndex);
    if(physical < cellNumber)
    {
        return physical;
    }
    return meshIndex;
}

#ifdef STORM_WITH_MPI
template<typename Grid>
boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap(const Grid &grid)
{
    static const Grid *cachedGrid = nullptr;
    static size_t cachedBuildGeneration = SIZE_MAX;
    static boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    
    if(grid.GetBuildGeneration() == cachedBuildGeneration and cachedGrid == &grid)
    {
        return ranks_ghost_map;
    }
    cachedGrid = &grid;
    cachedBuildGeneration = grid.GetBuildGeneration();
    ranks_ghost_map.clear();
    const std::vector<int> &dupProcs = grid.GetDuplicatedProcs();
    std::vector<std::vector<size_t>> incoming = ::MPI_exchange_data(dupProcs, grid.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = grid.GetGhostIndeces();
    for(size_t i = 0; i < incoming.size(); i++)
    {
        int _rank = dupProcs[i];
        for(size_t j = 0; j < incoming[i].size(); j++)
        {
            assert(incoming[i].size() == ghosts[i].size());
            if(ranks_ghost_map.find(ghosts[i][j]) != ranks_ghost_map.end())
            {
                STORMError eo("Duplicate in ranks ghost map");
                eo.addEntry("Index", ghosts[i][j]);
                eo.addEntry("Rank", _rank);
                eo.addEntry("Incoming", incoming[i][j]);
                throw eo;
            }
            ranks_ghost_map.emplace(ghosts[i][j], std::make_pair(_rank, incoming[i][j]));
        }
    }
    return ranks_ghost_map;
}

#endif // STORM_WITH_MPI

} // namespace STORM

#endif // GHOST_MAP_HPP