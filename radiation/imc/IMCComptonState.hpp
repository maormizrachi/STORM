#ifndef STORM_RADIATION_IMC_COMPTON_STATE_HPP
#define STORM_RADIATION_IMC_COMPTON_STATE_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace STORM::radiation_imc_detail {

/// Cache and group metadata used by the Compton correction phase.
template<typename ComptonCellDataT, typename ComptonBackendT, std::size_t NumGroups>
struct IMCComptonState
{
    using GroupArray = std::array<double, NumGroups>;

    std::vector<ComptonCellDataT> comptonData_;
    GroupArray comptonGroupCenters_{};
    GroupArray comptonGroupWidths_{};
    std::unique_ptr<ComptonBackendT> comptonMatrixGen_;
    bool comptonGroupsInitialized_ = false;
    bool comptonDataReusableInPreStep_ = false;
    double comptonRiskPrecomputeDt_ = -1.0;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_COMPTON_STATE_HPP
