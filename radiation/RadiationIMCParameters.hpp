#ifndef STORM_RADIATION_IMC_PARAMETERS_HPP
#define STORM_RADIATION_IMC_PARAMETERS_HPP

#include <array>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <string>

#include <units/units.hpp>

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

// Energy update of an out-of-group Compton event.  Mean: every event moves the
// packet to the sampled group and multiplies its weight by the source group's
// mean energy multiplier.  Exact: the weight takes the target-to-source centre
// ratio.  Thinned: with probability f the packet jumps, frequency and weight
// both scaled by the centre ratio (photons kept), otherwise it only changes
// direction; the gas gets the expected exchange and the zero-mean remainder is
// settled photon-neutrally at the end of the step.  Thinned (the default)
// needs a log-uniform group grid and the census preparation
// (adjustExistingParticles) before every step.
enum class ComptonEventMode
{
    Mean,
    Exact,
    Thinned
};

// Source of the Compton group-to-group tables.  MACE (the default):
// deterministic number-weighted effective-rate tables (MACE library,
// deps/MACE), read from a cache file in comptonTableCacheDir (or
// $STORM_COMPTON_TABLE_DIR, or the working directory) or built with all MPI
// ranks and cached there; build_compton_table prebuilds them for a grid.
// CMMC: Monte Carlo sampled tables built at startup.
enum class ComptonTableSource
{
    CMMC,
    MACE
};

// MACE kernel solver used to build tables.
enum class ComptonTableSolver
{
    Exact,
    Approximate
};

template<std::size_t NumGroups>
struct RadiationIMCParameters
{
    std::size_t newPhotonsPerCell = 0;
    // Per-cell floor on emission packets. Zero keeps it equal to
    // newPhotonsPerCell, which also sets the global emission budget and the
    // per-cell cap; set it lower to stop cells that emit a negligible share of
    // the total from being padded up to the budget-setting count.
    std::size_t emissionFloorPhotonsPerCell = 0;
    // Diagnostic only: multiplies the beta*sigma*c*dt argument of the Fleck factor
    // (f = 1 / (1 + scale * beta*sigma*c*dt)). 1 is Fleck-Cummings; used to measure
    // how strongly a result depends on the linearisation. Not for production.
    double fleckArgumentScale = 1.0;
    // A packet is killed once its weight falls below this fraction of the weight it
    // was created with.  0.001 is STORM's long-standing value; Branson's equivalent
    // (Constants::cutoff_fraction) is 0.01, and since the number of events a packet
    // survives goes as ln(1/cutoff)/fleck, the choice changes transport cost by that
    // ratio.  Exposed so the two can be matched when comparing run times.
    double weightCutoffFraction = 1e-3;
    double lightSpeed = units::clight;
    bool withHydro = false;
    // For planar x-symmetric material/boundaries and gas velocity along x.
    // A packet represents mirrored rays with zero net transverse momentum;
    // retain its full angular direction for transport and project coupling
    // and momentum diagnostics with momentumForCoupling().
    bool planarMomentumX = false;
    bool diffusionPressureGradient = false;
    bool MMC = false;
    // Keep lab-frame transport and scattering while still tallying momentum
    // into hydrodynamics. This omits all material-velocity (O(v/c)) terms.
    bool staticScatterers = false;
    bool withMultigroupOpacity = false;
    bool withRandomWalk = false;
    // Opt in only for a static x slab with reflecting transverse box walls.
    // Fold y/z motion analytically instead of treating each wall as an event.
    bool withSlabTransport = false;
    double rwMinCellOpticalDepth = 25.0;
    double rwMinParticleOpticalDepth = 5.0;
    bool withDDMC = false;
#if defined(STORM_WITH_GPU)
    bool ddmcGpuEnable = true;
#else
    bool ddmcGpuEnable = false;
#endif
    double ddmcMinCellOpticalDepth = 15.0;
    double ddmcExternalSourceMinFaceOpticalDepth = 5.0;
    bool ddmcUseMovingInterfaceCorrection = true;
    // Advection of DDMC radiation with the gas, div(v E0) of the O(v/c)
    // comoving-frame diffusion equation. Packets keep lab positions on a mesh
    // frozen during transport, so the gas would otherwise not carry DDMC
    // radiation; it applies whenever DDMC works in the comoving frame.
    bool ddmcAdvection = true;
    // Cells whose advection Peclet number |v| (L/2) / D exceeds this stay in
    // IMC. Infinite (the default) never excludes a cell.
    double ddmcMaxCellAdvectionPeclet = std::numeric_limits<double>::infinity();
    double ddmcMaxInterfaceVelocityOverC = 0.1;
    double ddmcInterfaceTargetWeightRatio = 2.0;
    std::size_t ddmcMaxInterfaceSplits = 64;
    // Moving IMC-to-DDMC interface corrections above this bound bypass DDMC
    // for that crossing and remain unbiased in ordinary IMC transport.
    double ddmcMaxMovingInterfaceWeightCorrection = 10.0;
    bool withMultigroupDDMC = false;
    std::size_t ddmcMaxGroupCutoff = NumGroups;
    bool ddmcInterfaceDiagnostics = false;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    // Canonical transport-level switch for polarized Thomson scattering.
    // postProcess.polarization.enabled remains a compatibility alias.
    bool withPolarization = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    ComptonInducedMode comptonInducedMode = ComptonInducedMode::AdaptivePlanckFallback;
    bool comptonAllowNZeroFallback = true;
    bool comptonAngleDependent = true;
    // Linearize the temperature dependence of the Compton matrix in the Fleck
    // factor.  False treats it explicitly (for stability tests only).
    bool comptonLinearizeTemperature = true;
    // Multiply the packet energy in a Compton event by the ratio of the target to
    // the source group center (true) instead of the mean ratio of the source
    // group (false).  The exact ratio preserves a Planck spectrum during transport.
    bool comptonExactEventEnergy = false;
    ComptonEventMode comptonEventMode = ComptonEventMode::Thinned;
    // Thinned mode: the gas pays the ledger when its predicted noise is below
    // this fraction of its internal energy; otherwise a two-group stencil does.
    double comptonSettlementGasTolerance = 1e-3;
    // Thinned mode: the planned stencil's low group must hold this many
    // predicted ledger standard deviations.
    double comptonSettlementSigmaMargin = 10.0;
    // In-group energy-only term of the last group (CMMC up minus down scatter).
    bool comptonTopGroupTerm = false;
    // Runtime occupation cap max(100, factor * nPlanck(max(Te, Trad))) per group.
    double comptonOccupationCapFactor = 10.0;
    // Optional fixed occupation cap (0 = off); a cost option, see the plan.
    double comptonOccupationMax = 0.0;
    // Symmetric pair throttle n (<= 0 = off): both directions of a pair scaled by
    // min(1, (1 + n) / (1 + max(n_i, n_j))); bounds the stimulated event cost and keeps
    // every fixed point, see compton/ComptonPairThrottle.hpp.  Default 100 (scan of
    // 2026-11-02, runs/imc_compton_paper/pair_throttle_scan_2026-11-02).
    double comptonPairThrottleOccupation = 100.0;
    // Thinned mode: close lab energy and momentum after reconciliation.
    bool comptonLabClosure = true;
    // Thinned mode: largest factor by which reconciliation scales packets up.
    double comptonReconcileUpscaleCap = 10.0;
    // Thinned mode: largest fraction of a group's old energy the linearized
    // drive may withdraw in one step, in (0, 1].
    double comptonLimiterFraction = 0.5;
    // Thinned mode: optional non-jump throttle Q, events per packet per step
    // above which non-jump events are thinned (0 = off; a cost option, see
    // ComptonThinned.hpp ThrottleEvents).
    double comptonEventThrottle = 0.0;
    // Thinned mode: when the end-of-step correction fails, retry with the
    // complete residual correction (matrix and drive, including Bcorr) scaled by
    // alpha = 1/2, 1/4, ... down to 1/1024; f and transport unchanged.  An
    // explicit numerical limiter (reported in the Thinned diagnostics); off by default.
    // Reached only when neither the direct solve nor the census-bounded re-solve
    // gave a finite solution for the withdrawal limiter (an ill-conditioned or
    // non-finite system); a negative endpoint is handled by that limiter instead.
    bool comptonCorrectionBacktracking = false;
    std::size_t comptonMatrixSamples = 200000;
    ComptonTableSource comptonTableSource = ComptonTableSource::MACE;
    // MACE tables: cache directory ("" = $STORM_COMPTON_TABLE_DIR, else the working directory), angle bins,
    // kernel solver, and OpenMP threads per rank while building (0: one per
    // rank under MPI with more than one rank, else the OpenMP default).
    std::string comptonTableCacheDir;
    std::size_t comptonTableAngleBins = 32;
    ComptonTableSolver comptonTableSolver = ComptonTableSolver::Exact;
    int comptonTableThreads = 0;
    std::array<double, NumGroups + 1> energyBoundaries{};
    bool energyBoundariesProvided = false;

    // comptonExactEventEnergy (the older switch) selects Exact over the mode.
    bool comptonThinnedEvents() const
    {
        return comptonEventMode == ComptonEventMode::Thinned && !comptonExactEventEnergy;
    }

    template<typename PointT>
    PointT momentumForCoupling(PointT momentum) const
    {
        if(planarMomentumX)
        {
            momentum.y = 0.0;
            momentum.z = 0.0;
        }
        return momentum;
    }

    struct PostProcessParameters
    {
        bool enabled = false;
        double sourceDt = 0.0;
        double transportTime = 0.0;
        bool useCellVelocities = true;
        // Frozen-flow limit on the DDMC comoving-frame compression shift. Each
        // packet's clock advances by rate dt on every DDMC and IMC step, with
        // rate = max(|div v|, |v| / max(r, cell size)) of its cell and r the
        // distance from compressionCenter; once the clock reaches
        // compressionMaxDynamicalTimes, DDMC no longer shifts the packet's
        // frequency and weight. Infinity keeps the unlimited shift, 0 turns it
        // off.
        double compressionMaxDynamicalTimes = std::numeric_limits<double>::infinity();
        std::array<double, 3> compressionCenter{0.0, 0.0, 0.0};

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
    os << "\temission floor photons per cell: "
       << (parameters.emissionFloorPhotonsPerCell > 0 ? parameters.emissionFloorPhotonsPerCell
                                                      : parameters.newPhotonsPerCell)
       << '\n';
    os << "\tlight speed: " << parameters.lightSpeed << '\n';
    os << "\twith hydro: " << parameters.withHydro << '\n';
    os << "\tplanar x momentum: " << parameters.planarMomentumX << '\n';
    os << "\tdiffusion pressure gradient: " << parameters.diffusionPressureGradient << '\n';
    os << "\tMMC: " << parameters.MMC << '\n';
    os << "\tstatic scatterers: " << parameters.staticScatterers << '\n';
    os << "\twith multigroup opacity: " << parameters.withMultigroupOpacity << '\n';
    os << "\twith random walk: " << parameters.withRandomWalk << '\n';
    os << "\twith slab transport: " << parameters.withSlabTransport << '\n';
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
        os << "\tDDMC GPU transport: " << parameters.ddmcGpuEnable << '\n';
        os << "\tDDMC min cell optical depth: " << parameters.ddmcMinCellOpticalDepth << '\n';
        os << "\tDDMC external-source min face optical depth: "
           << parameters.ddmcExternalSourceMinFaceOpticalDepth << '\n';
        os << "\tDDMC moving-interface correction: "
           << parameters.ddmcUseMovingInterfaceCorrection << '\n';
        os << "\tDDMC advection: " << parameters.ddmcAdvection << '\n';
        os << "\tDDMC max cell advection Peclet: "
           << parameters.ddmcMaxCellAdvectionPeclet << '\n';
        os << "\tDDMC maximum interface velocity/c: "
           << parameters.ddmcMaxInterfaceVelocityOverC << '\n';
        os << "\tDDMC maximum group cutoff: "
           << parameters.ddmcMaxGroupCutoff << '\n';
        os << "\twith multigroup DDMC: " << parameters.withMultigroupDDMC << '\n';
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
        os << "\tCompton temperature linearized: " << parameters.comptonLinearizeTemperature << '\n';
        os << "\tCompton exact event energy: " << parameters.comptonExactEventEnergy << '\n';
        os << "\tCompton event mode: "
           << (parameters.comptonThinnedEvents() ? "thinned"
               : (parameters.comptonExactEventEnergy || parameters.comptonEventMode == ComptonEventMode::Exact)
                   ? "exact" : "mean")
           << '\n';
        os << "\tCompton top-group term: " << parameters.comptonTopGroupTerm << '\n';
        os << "\tCompton occupation cap factor: " << parameters.comptonOccupationCapFactor << '\n';
        os << "\tCompton pair throttle occupation: " << parameters.comptonPairThrottleOccupation << '\n';
        os << "\tCompton table source: "
           << (parameters.comptonTableSource == ComptonTableSource::MACE ? "MACE" : "CMMC") << '\n';
        if(parameters.comptonTableSource == ComptonTableSource::MACE)
        {
            os << "\tCompton table angle bins: " << parameters.comptonTableAngleBins << '\n';
            os << "\tCompton table solver: "
               << (parameters.comptonTableSolver == ComptonTableSolver::Exact ? "exact" : "approximate") << '\n';
            os << "\tCompton table cache dir: "
               << (parameters.comptonTableCacheDir.empty() ? "." : parameters.comptonTableCacheDir) << '\n';
        }
        else
        {
            os << "\tCompton matrix samples: " << parameters.comptonMatrixSamples << '\n';
        }
    }
    if(parameters.postProcess.enabled)
    {
        os << "\tpost-process source dt: " << parameters.postProcess.sourceDt << '\n';
        os << "\tpost-process transport time: " << parameters.postProcess.transportTime << '\n';
        os << "\tpost-process use cell velocities: " << parameters.postProcess.useCellVelocities << '\n';
        os << "\tpost-process compression max dynamical times: "
           << parameters.postProcess.compressionMaxDynamicalTimes << '\n';
        os << "\tpost-process polarization: " << parameters.postProcess.polarization.enabled << '\n';
    }
    return os;
}

} // namespace STORM

#endif // STORM_RADIATION_IMC_PARAMETERS_HPP
