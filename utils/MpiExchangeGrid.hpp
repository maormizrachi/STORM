#ifndef MPI_EXCHANGE_GRID_HPP
#define MPI_EXCHANGE_GRID_HPP

#ifdef STORM_WITH_MPI

#include <vector>
#include <mpi_utils/mpi_exchange.hpp>

namespace STORM {

template<typename T, typename GridT>
void MPI_exchange_data(const GridT &grid, std::vector<T> &cells, bool ghost_or_sent)
{
	const std::vector<rank_t> &correspondents = grid.GetDuplicatedProcs();
	const std::vector<std::vector<size_t>> &indices = grid.GetDuplicatedPoints();
	std::vector<std::vector<T>> exchange = MPI_exchange_data_indexed(correspondents, cells, indices);
	const std::vector<std::vector<size_t>> &ghostIndices = grid.GetGhostIndeces();
	if(ghost_or_sent)
	{
		T default_val{};
		cells.resize(grid.GetTotalPointNumber(), default_val);
		for(size_t i = 0; i < correspondents.size(); ++i)
		{
			for(size_t j = 0; j < exchange[i].size(); ++j)
			{
				cells[ghostIndices[i][j]] = exchange[i][j];
			}
		}
	}
	else
	{
		for(size_t i = 0; i < correspondents.size(); ++i)
		{
			for(size_t j = 0; j < exchange[i].size(); ++j)
			{
				cells.push_back(exchange[i][j]);
			}
		}
	}
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // MPI_EXCHANGE_GRID_HPP
