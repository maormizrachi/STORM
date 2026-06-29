#ifndef GHOST_MAP_HPP
#define GHOST_MAP_HPP

#include <cassert>

#ifdef STORM_WITH_MPI

#include <mpi_utils/mpi_commands.hpp>
#include <boost/container/flat_map.hpp>
#include "../StormError.hpp"

namespace STORM {

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
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(dupProcs, grid.GetDuplicatedPoints());
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
            ranks_ghost_map.insert({ghosts[i][j], {_rank, incoming[i][j]}});
        }
    }
    return ranks_ghost_map;
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // GHOST_MAP_HPP