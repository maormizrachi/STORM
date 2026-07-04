#ifndef MPI_EXCHANGE_GRID_HPP
#define MPI_EXCHANGE_GRID_HPP

#ifdef STORM_WITH_MPI

#include <vector>
#include <mpi_utils/mpi_exchange.hpp>

namespace STORM {

template<typename T, typename GridT>
void MPI_exchange_data(const GridT &grid, std::vector<T> &cells, bool ghost_or_sent)
{
	const std::vector<rank_t> &correspondents = ghost_or_sent ? grid.GetDuplicatedProcs() : grid.GetSentProcs();
	const std::vector<std::vector<size_t>> &indices = ghost_or_sent ? grid.GetDuplicatedPoints() : grid.GetSentPoints();
	std::vector<std::vector<T>> exchange = MPI_exchange_data_indexed(correspondents, cells, indices);
	if(ghost_or_sent)
	{
		const std::vector<std::vector<size_t>> &ghostIndices = grid.GetGhostIndeces();
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
		const std::vector<size_t> &selfIndex = grid.GetSelfIndex();
		std::vector<T> result;
		result.reserve(selfIndex.size());
		for(size_t idx : selfIndex)
		{
			result.push_back(std::move(cells[idx]));
		}
		for(size_t i = 0; i < exchange.size(); ++i)
		{
			for(size_t j = 0; j < exchange[i].size(); ++j)
			{
				result.push_back(std::move(exchange[i][j]));
			}
		}
		cells = std::move(result);
	}
}

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // MPI_EXCHANGE_GRID_HPP
