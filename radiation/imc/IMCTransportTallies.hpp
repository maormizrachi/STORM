#ifndef STORM_RADIATION_IMC_TRANSPORT_TALLIES_HPP
#define STORM_RADIATION_IMC_TRANSPORT_TALLIES_HPP

#include <array>
#include <cstddef>
#include <vector>

namespace STORM::radiation_imc_detail {

/// Per-cell opacity snapshots and transport tallies accumulated during a step.
template<typename PointT, std::size_t NumGroups>
struct IMCTransportTallies
{
    using GroupArray = std::array<double, NumGroups>;

    std::array<double, NumGroups + 1> energyBoundaries_{};
    std::vector<double> factorFleck_;
    std::vector<double> planckOpacities_;
    std::vector<double> scatteringOpacities_;
    std::vector<double> Erad_time_avg_;
    std::vector<GroupArray> Eg_time_avg_;
    // Lab-frame energy and momentum the radiation field handed to the
    // material this step.  The internal-energy share is derived from both in
    // applyMaterialExchange(); nothing tallies internal energy directly.
    std::vector<double> pendingMaterialEnergy_;
    std::vector<PointT> pendingMomentum_;
    std::vector<PointT> transportCellVelocities_;
    std::vector<double> spectralAbsorptionScale_;
    std::vector<double> groupAbsorptionOpacities_;
    std::vector<double> thermalKT_;
    std::vector<double> thermalEmissionCdf_;
    std::vector<double> pendingRadiationEnergy_;
    std::vector<double> pendingGroupRadiationEnergy_;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_TRANSPORT_TALLIES_HPP
