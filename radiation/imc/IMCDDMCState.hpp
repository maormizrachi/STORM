#ifndef STORM_RADIATION_IMC_DDMC_STATE_HPP
#define STORM_RADIATION_IMC_DDMC_STATE_HPP

#include <cstddef>
#include <vector>

#include "../ddmc/DDMCDiagnostics.hpp"

namespace STORM::radiation_imc_detail {

/// Geometry and point metadata used by the DDMC accelerator.
template<typename DDMCCellDataT, typename PointT>
struct IMCDDMCState
    : IMCDDMCDiagnostics
{
    std::vector<DDMCCellDataT> ddmcCellData_;
    std::vector<int> ddmcPointEligible_;
    std::vector<double> ddmcPointDiffusionCoefficient_;
    std::vector<double> ddmcPointSigmaDiffusion_;
    std::vector<double> ddmcPointSigmaParticleGate_;
    std::vector<std::size_t> ddmcPointGroupCutoff_;
    std::vector<PointT> ddmcPointVelocity_;
    std::vector<std::size_t> ddmcPointCellID_;
    std::vector<PointT> ddmcFluxRhsIntegrated_;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_DDMC_STATE_HPP
