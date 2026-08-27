#ifndef STORM_RADIATION_DDMC_DIAGNOSTICS_HPP
#define STORM_RADIATION_DDMC_DIAGNOSTICS_HPP

#include <cstddef>
#include <limits>
#include <map>
#include <tuple>

namespace STORM::radiation_imc_detail {

enum class DDMCDiagnosticEventKind : unsigned char
{
    IMCCandidate,
    IMCFrequencyReject,
    IMCIncident,
    IMCAdmitted,
    IMCReflected,
    IMCBypass,
    DDMCToDDMC,
    DDMCToIMC
};

static constexpr std::size_t DDMC_DIAGNOSTIC_GREY_GROUP = std::numeric_limits<std::size_t>::max();

struct DDMCDiagnosticEventKey
{
    DDMCDiagnosticEventKind kind = DDMCDiagnosticEventKind::IMCCandidate;
    std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
    std::size_t sourceCellID = std::numeric_limits<std::size_t>::max();
    std::size_t targetCellID = std::numeric_limits<std::size_t>::max();
    std::size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;

    bool operator<(DDMCDiagnosticEventKey const &other) const
    {
        return std::tie(kind, faceIndex, sourceCellID, targetCellID, group) <
               std::tie(other.kind, other.faceIndex, other.sourceCellID, other.targetCellID, other.group);
    }
};

struct DDMCDiagnosticEventAccumulator
{
    std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
    std::size_t sourceCellID = std::numeric_limits<std::size_t>::max();
    std::size_t targetCellID = std::numeric_limits<std::size_t>::max();
    std::size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;
    std::size_t sourceGroupCutoff = 0;
    std::size_t targetGroupCutoff = 0;
    double faceX = std::numeric_limits<double>::quiet_NaN();
    double sourceGeneratorX = std::numeric_limits<double>::quiet_NaN();
    double targetGeneratorX = std::numeric_limits<double>::quiet_NaN();
    std::size_t count = 0;
    double signedEnergy = 0.0;
    double absoluteEnergy = 0.0;
    double muSum = 0.0;
    std::size_t muCount = 0;
    double admissionProbabilitySum = 0.0;
    std::size_t admissionProbabilityCount = 0;
};

/// Counters and event records emitted by the DDMC interface.
struct IMCDDMCDiagnostics
{
    double ddmcLeakReciprocityResidualMax_ = 0.0;
    std::size_t ddmcLeakReciprocityCheckCount_ = 0;
    std::size_t ddmcMomentumFeedbackCount_ = 0;
    std::size_t ddmcMomentumMatrixFallbackCount_ = 0;
    std::size_t ddmcResidentLeakCount_ = 0;
    std::size_t ddmcTransportLeakCount_ = 0;
    std::size_t ddmcRemoteResidentLeakCount_ = 0;
    std::size_t ddmcMPIFaceFluxReductionCount_ = 0;
    std::size_t ddmcLeakInvalidGeometryCount_ = 0;
    std::size_t ddmcUnsupportedBoundaryFaceCount_ = 0;
    std::size_t ddmcInterfaceIncidentCount_ = 0;
    std::size_t ddmcInterfaceAdmittedCount_ = 0;
    std::size_t ddmcInterfaceReflectedCount_ = 0;
    std::size_t ddmcInterfaceGuAppliedCount_ = 0;
    std::size_t ddmcInterfaceGuFallbackCount_ = 0;
    std::size_t ddmcInterfaceBypassCount_ = 0;
    std::size_t ddmcInterfaceSplitPacketCount_ = 0;
    std::size_t ddmcInterfaceFluxTallyCount_ = 0;
    double ddmcInterfaceMinimumMu_ = std::numeric_limits<double>::infinity();
    std::map<DDMCDiagnosticEventKey, DDMCDiagnosticEventAccumulator> ddmcDiagnosticEvents_;
    std::size_t ddmcExternalSourceCandidateFaceCount_ = 0;
    std::size_t ddmcExternalSourceAcceleratedFaceCount_ = 0;
    std::size_t ddmcExternalSourceExplicitFallbackFaceCount_ = 0;
    std::size_t ddmcExternalSourceInteriorExcludedCellCount_ = 0;
    std::size_t ddmcExternalSourceThermalizationCount_ = 0;
    std::size_t ddmcExternalSourceStayDDMCCount_ = 0;
    std::size_t ddmcExternalSourceToIMCCount_ = 0;
    double ddmcExternalSourceThermalizedEnergy_ = 0.0;
    double ddmcExternalSourceToIMCEnergy_ = 0.0;
    double ddmcExternalSourceMinimumFaceOpticalDepth_ = std::numeric_limits<double>::infinity();
    std::size_t ddmcStepCount_ = 0;
    std::size_t ddmcLeakCount_ = 0;
    std::size_t ddmcCensusCount_ = 0;
    std::size_t ddmcUpscatterCount_ = 0;
    std::size_t ddmcFallbackCount_ = 0;
    std::size_t ddmcMovingInterfaceBypassCount_ = 0;
    double ddmcMovingInterfaceMaxFactor_ = 0.0;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_DDMC_DIAGNOSTICS_HPP
