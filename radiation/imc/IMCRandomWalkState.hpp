#ifndef STORM_RADIATION_IMC_RANDOM_WALK_STATE_HPP
#define STORM_RADIATION_IMC_RANDOM_WALK_STATE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace STORM::radiation_imc_detail {

/// Per-cell state for the optional random-walk accelerator.
template<typename RandomWalkT, typename PGRWCellDataT>
struct IMCRandomWalkState
{
    std::unique_ptr<RandomWalkT> randomWalk_;
    std::vector<std::uint8_t> rwCellEligible_;
    std::vector<double> rwCellTotalOpacity_;
    std::vector<PGRWCellDataT> rwCellData_;
    std::size_t rwStepCount_ = 0;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_RANDOM_WALK_STATE_HPP
