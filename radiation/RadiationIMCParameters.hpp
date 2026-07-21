#ifndef STORM_RADIATION_IMC_PARAMETERS_HPP
#define STORM_RADIATION_IMC_PARAMETERS_HPP

#include <array>
#include <cstddef>
#include <iosfwd>
#include <ostream>
#include <string>

namespace STORM {

enum class ComptonInducedMode
{
    RadiationField,
    AdaptivePlanckFallback
};

enum class ComptonOccupationMode
{
    Zero,
    RadiationField,
    PlanckFunction
};

template<std::size_t NumGroups>
struct RadiationIMCParameters
{
    std::size_t newPhotonsPerCell = 0;
    bool withHydro = false;
    bool diffusionPressureGradient = false;
    bool MMC = false;
    bool withMultigroupOpacity = false;
    bool withRandomWalk = false;
    double rwMinCellOpticalDepth = 25.0;
    double rwMinParticleOpticalDepth = 5.0;
    bool withDDMC = false;
    double ddmcMinCellOpticalDepth = 15.0;
    double ddmcMinParticleOpticalDepth = 5.0;
    // Moving IMC-to-DDMC interface corrections above this bound bypass DDMC
    // for that crossing and remain unbiased in ordinary IMC transport.
    double ddmcMaxMovingInterfaceWeightCorrection = 10.0;
    bool ddmcUseMultigroupPGRW = false;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    // Canonical transport-level switch for polarized Thomson scattering.
    // postProcess.polarization.enabled remains a compatibility alias.
    bool withPolarization = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    ComptonInducedMode comptonInducedMode =
        ComptonInducedMode::AdaptivePlanckFallback;
    bool comptonAllowNZeroFallback = true;
    bool comptonAngleDependent = true;
    std::size_t comptonMatrixSamples = 200000;
    std::array<double, NumGroups + 1> energyBoundaries{};
    bool energyBoundariesProvided = false;

    struct PostProcessParameters
    {
        bool enabled = false;
        double sourceDt = 0.0;
        double transportTime = 0.0;
        bool useCellVelocities = true;

        struct PolarizationParameters
        {
            bool enabled = false;
            int manualScatteringsAfterAcceleration = 4;
            double depolarizationScatterings = 2.0;
            std::string acceleratedClosure = "damped_last_scatterings";
        } polarization;
    } postProcess;
};

template<std::size_t NumGroups>
std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters<NumGroups> &parameters)
{
    os << "STORM IMC, with parameters:\n";
    os << "\tnew photons per cell: " << parameters.newPhotonsPerCell << '\n';
    os << "\twith hydro: " << parameters.withHydro << '\n';
    os << "\tdiffusion pressure gradient: " << parameters.diffusionPressureGradient << '\n';
    os << "\tMMC: " << parameters.MMC << '\n';
    os << "\twith multigroup opacity: " << parameters.withMultigroupOpacity << '\n';
    os << "\twith random walk: " << parameters.withRandomWalk << '\n';
    os << "\twith DDMC: " << parameters.withDDMC << '\n';
    os << "\tno hydro feedback: " << parameters.noHydroFeedback << '\n';
    os << "\twith group time averages: " << parameters.withEgTimeAvg << '\n';
    os << "\twith polarization: " << parameters.withPolarization << '\n';
    os << "\twith Compton: " << parameters.withCompton << '\n';
    if(parameters.withRandomWalk)
    {
        os << "\tRW min cell optical depth: " << parameters.rwMinCellOpticalDepth << '\n';
        os << "\tRW min particle optical depth: " << parameters.rwMinParticleOpticalDepth << '\n';
    }
    if(parameters.withDDMC)
    {
        os << "\tDDMC min cell optical depth: " << parameters.ddmcMinCellOpticalDepth << '\n';
        os << "\tDDMC min particle optical depth: " << parameters.ddmcMinParticleOpticalDepth << '\n';
        os << "\tDDMC multigroup PGRW: " << parameters.ddmcUseMultigroupPGRW << '\n';
        os << "\tDDMC max moving interface weight correction: "
           << parameters.ddmcMaxMovingInterfaceWeightCorrection << '\n';
    }
    if(parameters.withCompton)
    {
        os << "\tCompton induced terms: " << parameters.comptonUseInduced << '\n';
        os << "\tCompton induced mode: "
           << (parameters.comptonInducedMode ==
                       ComptonInducedMode::RadiationField
                   ? "radiation-field"
                   : "adaptive-planck-fallback")
           << '\n';
        os << "\tCompton n=0 fallback: " << parameters.comptonAllowNZeroFallback << '\n';
        os << "\tCompton angle dependent: " << parameters.comptonAngleDependent << '\n';
        os << "\tCompton matrix samples: " << parameters.comptonMatrixSamples << '\n';
    }
    if(parameters.postProcess.enabled)
    {
        os << "\tpost-process source dt: " << parameters.postProcess.sourceDt << '\n';
        os << "\tpost-process transport time: " << parameters.postProcess.transportTime << '\n';
        os << "\tpost-process use cell velocities: " << parameters.postProcess.useCellVelocities << '\n';
        os << "\tpost-process polarization: " << parameters.postProcess.polarization.enabled << '\n';
    }
    return os;
}

} // namespace STORM

#endif // STORM_RADIATION_IMC_PARAMETERS_HPP
