#ifndef STORM_RADIATION_IMC_COMPTON_STATE_HPP
#define STORM_RADIATION_IMC_COMPTON_STATE_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "../compton/ComptonThinned.hpp"

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

    // Thinned event mode (ComptonThinned.hpp, plan v10).
    compton_thinned::Grid comptonThinnedGrid_{};
    bool comptonThinnedGridChecked_ = false;
    // Set by the census preparation (adjustExistingParticles) and consumed by
    // the next preStep, so a step without preparation fails loudly.
    bool comptonCensusPrepared_ = false;
    // Comoving energy the packets of a cell carry beyond the expected
    // exchange, accumulated over the step's events and settled in postStep.
    std::vector<double> comptonLedger_;
    // Hazard-weighted path-length tallies of this step per cell and group:
    // integral of D w_c dt (the drive) and of D w_c^2 dt (the noise predictor).
    std::vector<GroupArray> comptonHazardEnergy_;
    std::vector<GroupArray> comptonHazardWeightSquared_;
    // Previous step's hazard-weighted sum of w_c^2 per unit time (erg^2).
    std::vector<GroupArray> comptonPreviousWeightSquared_;
    // Settlement results per cell, used by the correction and the closure:
    // actual census photons before settlement, end-of-step h = N/E per group,
    // the comoving energy the gas paid (-X or 0), and the gas change of the
    // correction.
    std::vector<double> comptonRawPhotons_;
    std::vector<GroupArray> comptonEndH_;
    std::vector<double> comptonGasSettlement_;
    std::vector<double> comptonCorrectionGasChange_;
    compton_thinned::Diagnostics comptonThinnedDiagnostics_{};
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_COMPTON_STATE_HPP
