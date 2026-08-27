#ifndef STORM_RADIATION_IMC_SOURCE_CONTROL_HPP
#define STORM_RADIATION_IMC_SOURCE_CONTROL_HPP

#include <array>
#include <cstddef>
#include <unordered_map>

namespace STORM::radiation_imc_detail {

/// Adaptive source allocation and learned group-emission controls.
template<std::size_t NumGroups>
struct IMCSourceControl
{
    using GroupArray = std::array<double, NumGroups>;

    std::unordered_map<std::size_t, double> adaptiveSourceScores_;
    bool adaptiveSourceScoresEnabled_ = false;
    double adaptiveSourceStrength_ = 0.0;
    double adaptiveSourceMaxFactor_ = 1.0;
    double adaptiveSourceLearnedReserveFrac_ = 0.0;
    double adaptiveSourceLearnedMinFactor_ = 1.0;
    double adaptiveSourceObserverBudgetMultiplier_ = 1.0;
    std::size_t adaptiveSourceLearnedMinPhotons_ = 0;
    std::size_t adaptiveSourceLearnedMaxPhotons_ = 0;
    double adaptiveSourceScorePower_ = 1.0;
    std::unordered_map<std::size_t, GroupArray> adaptiveSourceCellGroupScores_;
    bool adaptiveSourceCellGroupScoresEnabled_ = false;
    double adaptiveGroupStrength_ = 0.0;
    double adaptiveGroupPdfFloor_ = 0.0;
    double adaptiveGroupMaxBias_ = 1.0;
    double adaptiveGroupMaxWeightCorrection_ = 1.0;
    bool sourceEmissionControlEnabled_ = false;
    bool sourceEmissionUseLearnedScores_ = false;
    bool sourceEmissionIncludeUniformBase_ = true;
    std::size_t sourceEmissionBaseMultiplier_ = 1;
    std::size_t sourceEmissionLearnedBoostFactor_ = 20;
    std::size_t sourceEmissionLearnedExtraBudget_ = 0;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_SOURCE_CONTROL_HPP
