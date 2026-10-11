/*
 * DDMC advection test 2: uniform radiation carried across an IMC-DDMC
 * interface by uniformly moving gas.
 *
 * Gas fills x in [0, 70] cm, one 1 x 2 x 2 cm Cartesian cell across, and
 * moves along +x at v = beta c, beta = 0.03.  It scatters isotropically and
 * does not absorb.  The left 30 cells are thick (sigma = 8 /cm, sigma 4V/A = 8,
 * DDMC eligible with ddmcMinCellOpticalDepth = 4); the right 40 cells are thin
 * (sigma = 1 /cm, sigma 4V/A = 1, IMC).  The gas therefore carries radiation
 * from the DDMC region into the IMC region through the interface at x = 30 cm.
 *
 * The radiation starts uniform and isotropic in the comoving frame,
 * E0 = 1e12 erg/cm^3 everywhere.  Uniform flow of uniform comoving radiation
 * is a steady state of the O(v/c) comoving equations, so the profile must stay
 * flat.  Inside the DDMC region the upwind advection leaks keep it flat; at
 * the interface the DDMC -> IMC leaks carry no advection and the O(v/c)
 * balance comes from the moving-interface correction of IMC -> DDMC packets.
 * A jump or dip at x = 30 cm is the failure this test looks for.
 *
 * The walls are rigid (RigidBoundary), which DDMC supports.  Uniform flow
 * through a rigid wall depletes the radiation at the upstream wall and piles
 * it up at the downstream one; during the run, v t = 5 cm, neither
 * disturbance gets within about 10 cm of the interface (see README.md).
 *
 * Variants:
 *   moving_correction     ddmcAdvection = true, ddmcUseMovingInterfaceCorrection
 *                         = true (the defaults).  Gated: flat to 3 %.
 *   no_moving_correction  moving-interface correction off.  Recorded.
 *   no_advection          ddmcAdvection = false.  Recorded (expected to dip).
 *   static_control        beta = 0: the interface alone.  Recorded.
 *
 * The energy profile is the time-averaged radiation energy density tally
 * (RadiationIMC::getEradTimeAvg) averaged over the steps after 40 % of the
 * run, normalised by the initial E0.  The comoving census energy density is
 * reported alongside.  The tally is comoving in DDMC cells and lab-frame in
 * IMC cells; the two differ at O(beta^2), about 0.1 %.
 *
 * Usage:
 *   ./ddmc_advection_interface [variants] [packetsPerCell] [steps]
 *     variants        "all" (default) or a comma-separated list of the names above
 *     packetsPerCell  initial packets in every cell (default 4000)
 *     steps           census steps over the run (default 20)
 *
 * Output (current directory):
 *   stdout                                   one DDMC_ADVECTION_INTERFACE_METRICS line per variant
 *   ddmc_advection_interface_metrics.txt     the same key=value lines
 *   ddmc_advection_interface_profiles.txt    final averaged profiles of every variant
 * The pass/fail criteria live in ../ddmc_advection/check_ddmc_advection.py.
 */

#ifdef STORM_WITH_MPI
#error "ddmc_advection_interface is serial: REGRESSION_INFO tags it serial, so CMake builds it without STORM_WITH_MPI"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include <units/units.hpp>

#include "StormError.hpp"
#include "boundary/RigidBoundary.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationIMCParameters.hpp"
#include "population/NoPopulationControl.hpp"
#include "manager/MonteCarloManager.hpp"
#include "examples/ddmc_advection/DDMCAdvectionCommon.hpp"

namespace
{

using STORM::StormError;
using STORM::examples::ddmc_advection::AdvectionCell;
using STORM::examples::ddmc_advection::AdvectionExtensives;
using STORM::examples::ddmc_advection::ComovingEnergy;
using STORM::examples::ddmc_advection::ConstantCvEOS;
using STORM::examples::ddmc_advection::IsotropicDirection;
using STORM::examples::ddmc_advection::MetricsLine;
using STORM::examples::ddmc_advection::OpenUnitStream;
using STORM::examples::ddmc_advection::ParsePositiveCount;
using STORM::examples::ddmc_advection::ScatteringOpacity;
using STORM::examples::ddmc_advection::SetLabPacketFromComoving;
using STORM::examples::ddmc_advection::SetPacketIdentity;
using STORM::examples::ddmc_advection::SplitList;

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using MCParticle = STORM::Particle<Vector3D>;
using Opacity = ScatteringOpacity<Grid>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, AdvectionCell, AdvectionExtensives,
                                ConstantCvEOS, 1, Opacity>;

// ---- Problem definition (CGS) ----------------------------------------------

constexpr std::size_t N_THICK = 30;                 // DDMC cells, x in [0, 30] cm
constexpr std::size_t N_THIN = 40;                  // IMC cells, x in [30, 70] cm
constexpr std::size_t NX = N_THICK + N_THIN;
constexpr double CELL_WIDTH = 1.0;                  // [cm] along x
constexpr double TRANSVERSE_WIDTH = 2.0;            // [cm]; one cell across y and z
// Mean chord 4V/A = 4 * 4 / 16 = 1 cm, so sigma 4V/A = sigma * 1 cm.
constexpr double THICK_OPACITY = 8.0;               // [1/cm]
constexpr double THIN_OPACITY = 1.0;                // [1/cm]
constexpr double DDMC_MIN_CELL_OPTICAL_DEPTH = 4.0;
constexpr double BETA = 0.03;                       // gas speed / c, along +x
constexpr double ADVECTED_CELLS = 5.0;              // v t / CELL_WIDTH at the end
constexpr double ENERGY_DENSITY = 1.0e12;           // [erg/cm^3], comoving
constexpr double DENSITY = 1.0;                     // [g/cm^3]
constexpr double TEMPERATURE = 1.0e4;               // [K]; any positive value
constexpr double CV_PER_MASS = 1.0e8;               // [erg/g/K]; unused without absorption
constexpr std::size_t WINDOW_HALF_WIDTH = 4;        // cells on each side of the interface
constexpr double AVERAGE_START_FRACTION = 0.4;      // average the steps after this fraction of the run
constexpr std::size_t THICK_BULK_BEGIN = 18;        // bulk reference cells, away from walls and interface
constexpr std::size_t THICK_BULK_END = 24;
constexpr std::size_t THIN_BULK_BEGIN = 40;
constexpr std::size_t THIN_BULK_END = 50;
constexpr std::uint64_t INITIAL_PACKET_SEED = 0x1f2026ULL;

constexpr const char *METRICS_PATH = "ddmc_advection_interface_metrics.txt";
constexpr const char *PROFILES_PATH = "ddmc_advection_interface_profiles.txt";
constexpr const char *METRICS_TAG = "DDMC_ADVECTION_INTERFACE_METRICS";

// ---- Variants ----------------------------------------------------------------

struct InterfaceVariant
{
    std::string name;
    double beta = BETA;
    bool advection = true;
    bool movingCorrection = true;
    bool gated = false;
};

const std::vector<InterfaceVariant> &AllVariants()
{
    static const std::vector<InterfaceVariant> variants = {
        {"moving_correction", BETA, true, true, true},
        {"no_moving_correction", BETA, true, false, false},
        {"no_advection", BETA, false, true, false},
        {"static_control", 0.0, true, true, false}};
    return variants;
}

struct InterfaceOptions
{
    std::vector<InterfaceVariant> variants = AllVariants();
    std::size_t packetsPerCell = 4000;
    std::size_t steps = 20;
};

InterfaceOptions ParseOptions(int argc, char *argv[])
{
    InterfaceOptions options;
    if(argc > 4)
    {
        StormError eo("Too many arguments; usage: ddmc_advection_interface [variants] [packetsPerCell] [steps]");
        eo.addEntry("Argument count", argc - 1);
        throw eo;
    }
    if(argc >= 2 && std::string(argv[1]) != "all")
    {
        options.variants.clear();
        for(const std::string &name : SplitList(argv[1], ','))
        {
            auto const known = std::find_if(AllVariants().begin(), AllVariants().end(),
                                            [&name](const InterfaceVariant &variant)
                                            {
                                                return variant.name == name;
                                            });
            if(known == AllVariants().end())
            {
                StormError eo("Unknown variant; use all, moving_correction, no_moving_correction, no_advection or static_control");
                eo.addEntry("Variant", name);
                throw eo;
            }
            bool const duplicate = std::any_of(options.variants.begin(), options.variants.end(),
                                               [&name](const InterfaceVariant &variant)
                                               {
                                                   return variant.name == name;
                                               });
            if(duplicate)
            {
                StormError eo("Variant listed twice");
                eo.addEntry("Variant", name);
                throw eo;
            }
            options.variants.push_back(*known);
        }
    }
    if(argc >= 3)
    {
        options.packetsPerCell = ParsePositiveCount(argv[2], "packetsPerCell");
    }
    if(argc >= 4)
    {
        options.steps = ParsePositiveCount(argv[3], "steps");
    }
    return options;
}

bool IsThickCell(double x)
{
    return x < static_cast<double>(N_THICK) * CELL_WIDTH;
}

// ---- Initial condition ------------------------------------------------------------

// packetsPerCell packets in every cell, uniform in the cell, isotropic in the
// comoving frame, each carrying ENERGY_DENSITY * V / packetsPerCell of
// comoving energy.
std::vector<MCParticle> MakeUniformPackets(const Grid &grid,
                                           const std::vector<AdvectionCell> &cells,
                                           std::size_t packetsPerCell,
                                           double lightSpeed)
{
    OpenUnitStream stream(INITIAL_PACKET_SEED);
    auto const [lower, upper] = grid.GetBoxCoordinates();
    std::vector<MCParticle> particles;
    particles.reserve(cells.size() * packetsPerCell);
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        double const centerX = grid.GetMeshPoint(i).x;
        double const comovingEnergy = ENERGY_DENSITY * grid.GetVolume(i) / static_cast<double>(packetsPerCell);
        for(std::size_t k = 0; k < packetsPerCell; ++k)
        {
            MCParticle particle;
            particle.location = Vector3D(centerX + CELL_WIDTH * (stream.Next() - 0.5),
                                         lower.y + (upper.y - lower.y) * stream.Next(),
                                         lower.z + (upper.z - lower.z) * stream.Next());
            std::size_t const containing = grid.GetContainingCell(particle.location);
            if(containing != i)
            {
                StormError eo("ddmc_advection_interface: an initial packet left its cell");
                eo.addEntry("Cell index", i);
                eo.addEntry("Containing cell", containing);
                eo.addEntry("Location", particle.location);
                throw eo;
            }
            particle.cellIndex = i;
            particle.cellID = cells[i].ID;
            particle.sourceCellID = particle.cellID;
            particle.frequency = units::kev;
            SetLabPacketFromComoving(particle, IsotropicDirection(stream), comovingEnergy,
                                     cells[i].velocity, lightSpeed);
            particle.initialWeight = std::abs(particle.weight);
            SetPacketIdentity(particle, particles.size(), INITIAL_PACKET_SEED);
            particles.push_back(particle);
        }
    }
    return particles;
}

double TotalComovingEnergy(const std::vector<MCParticle> &particles,
                           const std::vector<AdvectionCell> &cells,
                           double lightSpeed)
{
    long double total = 0.0L;
    for(const MCParticle &particle : particles)
    {
        if(particle.cellIndex >= cells.size())
        {
            StormError eo("ddmc_advection_interface: a census packet has no local cell");
            eo.addEntry("Particle", particle);
            throw eo;
        }
        total += ComovingEnergy(particle, cells[particle.cellIndex].velocity, lightSpeed);
    }
    return static_cast<double>(total);
}

// ---- One variant ----------------------------------------------------------------------------

struct ProfileSummary
{
    std::vector<double> window;      // E / E0 for the WINDOW_HALF_WIDTH cells on each side
    double interfaceRatio = 0.0;     // E(last DDMC cell) / E(first IMC cell)
    double maxWindowDeviation = 0.0; // max |E / E0 - 1| over the window
    double thickBulk = 0.0;          // mean E / E0 over the thick bulk reference cells
    double thinBulk = 0.0;           // mean E / E0 over the thin bulk reference cells
};

struct InterfaceResult
{
    InterfaceVariant variant;
    double time = 0.0;
    double speed = 0.0;
    std::size_t packetsPerCell = 0;
    std::size_t cells = 0;
    std::size_t averagedSteps = 0;
    double referenceEnergyDensity = 0.0;
    double initialEnergy = 0.0;
    double finalEnergy = 0.0;
    std::size_t ddmcEligibleCells = 0;
    std::size_t eligibilityMismatches = 0;  // cells whose eligibility differs from thick/thin
    std::size_t advectiveLeaks = 0;
    std::size_t ddmcLeaks = 0;
    std::size_t ddmcFallbacks = 0;
    std::size_t movingInterfaceBypasses = 0;
    double maxMovingInterfaceFactor = 0.0;
    double maxCellPeclet = 0.0;
    double wallSeconds = 0.0;
    std::vector<std::size_t> xOrder;        // cell indices sorted by x
    std::vector<double> tallyProfile;       // averaged time-averaged tally / E0, by cell index
    std::vector<double> censusProfile;      // averaged comoving census density / E0, by cell index
    std::vector<int> eligible;              // DDMC eligibility at the last step, by cell index
    ProfileSummary tally;
    ProfileSummary census;
};

ProfileSummary Summarize(const std::vector<double> &profile, const std::vector<std::size_t> &xOrder)
{
    ProfileSummary summary;
    for(std::size_t rank = N_THICK - WINDOW_HALF_WIDTH; rank < N_THICK + WINDOW_HALF_WIDTH; ++rank)
    {
        double const value = profile[xOrder[rank]];
        summary.window.push_back(value);
        summary.maxWindowDeviation = std::max(summary.maxWindowDeviation, std::abs(value - 1.0));
    }
    double const thin = profile[xOrder[N_THICK]];
    summary.interfaceRatio = thin > 0.0 ? profile[xOrder[N_THICK - 1]] / thin
                                        : std::numeric_limits<double>::infinity();
    auto mean = [&](std::size_t begin, std::size_t end)
    {
        double sum = 0.0;
        for(std::size_t rank = begin; rank < end; ++rank)
        {
            sum += profile[xOrder[rank]];
        }
        return sum / static_cast<double>(end - begin);
    };
    summary.thickBulk = mean(THICK_BULK_BEGIN, THICK_BULK_END);
    summary.thinBulk = mean(THIN_BULK_BEGIN, THIN_BULK_END);
    return summary;
}

InterfaceResult RunVariant(const InterfaceVariant &variant, const InterfaceOptions &options)
{
    auto const wallStart = std::chrono::steady_clock::now();

    STORM::RadiationIMCParameters<1> parameters;
    double const lightSpeed = parameters.lightSpeed;
    double const speed = variant.beta * lightSpeed;
    Vector3D const gasVelocity(speed, 0.0, 0.0);
    // The run length is set by the moving variants, so static_control covers
    // the same time.
    double const endTime = ADVECTED_CELLS * CELL_WIDTH / (BETA * lightSpeed);
    double const dt = endTime / static_cast<double>(options.steps);

    Grid grid(Vector3D(0.0, 0.0, 0.0),
              Vector3D(static_cast<double>(NX) * CELL_WIDTH, TRANSVERSE_WIDTH, TRANSVERSE_WIDTH),
              NX, 1, 1);
    std::size_t const cellCount = grid.GetPointNo();
    if(cellCount != NX)
    {
        StormError eo("ddmc_advection_interface: unexpected Cartesian cell count");
        eo.addEntry("Expected", NX);
        eo.addEntry("Built", cellCount);
        throw eo;
    }

    std::vector<AdvectionCell> cells(cellCount);
    std::vector<AdvectionExtensives> extensives(cellCount);
    std::size_t thickCells = 0;
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        double const volume = grid.GetVolume(i);
        bool const thick = IsThickCell(grid.GetMeshPoint(i).x);
        thickCells += thick ? 1 : 0;
        cells[i].ID = i;
        cells[i].temperature = TEMPERATURE;
        cells[i].density = DENSITY;
        cells[i].velocity = gasVelocity;
        cells[i].scatteringOpacity = thick ? THICK_OPACITY : THIN_OPACITY;
        cells[i].internalEnergy = CV_PER_MASS * TEMPERATURE * DENSITY * volume;
        extensives[i].mass = DENSITY * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }
    if(thickCells != N_THICK)
    {
        StormError eo("ddmc_advection_interface: the thick region does not have the expected cell count");
        eo.addEntry("Expected", N_THICK);
        eo.addEntry("Found", thickCells);
        throw eo;
    }

    InterfaceResult result;
    result.variant = variant;
    result.speed = speed;
    result.packetsPerCell = options.packetsPerCell;
    result.cells = cellCount;
    result.xOrder.resize(cellCount);
    std::iota(result.xOrder.begin(), result.xOrder.end(), std::size_t(0));
    std::sort(result.xOrder.begin(), result.xOrder.end(), [&grid](std::size_t a, std::size_t b)
              {
                  return grid.GetMeshPoint(a).x < grid.GetMeshPoint(b).x;
              });

    // RadiationIMC requires a positive emission budget; with zero absorption
    // every cell's emission energy is zero and no packet is emitted.
    parameters.newPhotonsPerCell = 1;
    parameters.withHydro = true;
    parameters.noHydroFeedback = true;
    parameters.withDDMC = true;
    parameters.ddmcMinCellOpticalDepth = DDMC_MIN_CELL_OPTICAL_DEPTH;
    parameters.ddmcAdvection = variant.advection;
    parameters.ddmcUseMovingInterfaceCorrection = variant.movingCorrection;
    parameters.energyBoundaries = {0.0, 1.0e30};
    parameters.energyBoundariesProvided = true;

    auto eos = std::make_shared<ConstantCvEOS>(CV_PER_MASS);
    auto opacity = std::make_shared<Opacity>();
    std::shared_ptr<STORM::BoundaryCondition<Vector3D, Grid>> boundary =
        std::make_shared<STORM::RigidBoundary<Vector3D, Grid>>(grid);
    auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacity, parameters);
    std::shared_ptr<STORM::PopulationControl<Vector3D, Grid>> populationControl =
        std::make_shared<STORM::NoPopulationControl<Vector3D, Grid>>(grid);
    STORM::MonteCarloManager<Vector3D, Grid, IMC> manager(grid, physics, populationControl, boundary);

    manager.getParticles() = MakeUniformPackets(grid, cells, options.packetsPerCell, lightSpeed);
    result.initialEnergy = TotalComovingEnergy(std::as_const(manager).getParticles(), cells, lightSpeed);
    double totalVolume = 0.0;
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        totalVolume += grid.GetVolume(i);
    }
    result.referenceEnergyDensity = result.initialEnergy / totalVolume;

    std::cout << "\n=== ddmc_advection_interface variant " << variant.name
              << ": beta=" << variant.beta << " advection=" << variant.advection
              << " movingCorrection=" << variant.movingCorrection
              << " packetsPerCell=" << options.packetsPerCell << " dt=" << dt
              << " s steps=" << options.steps << " ===" << std::endl;

    std::size_t const firstAveragedStep = std::min(
        options.steps - 1,
        static_cast<std::size_t>(std::floor(AVERAGE_START_FRACTION * static_cast<double>(options.steps))));
    std::vector<long double> tallySum(cellCount, 0.0L);
    std::vector<long double> censusSum(cellCount, 0.0L);
    double simTime = 0.0;
    for(std::size_t step = 0; step < options.steps; ++step)
    {
        manager.step(dt);
        simTime += dt;

        // RadiationIMC resets these counters at every preStep.
        result.advectiveLeaks += physics->getDDMCAdvectiveLeakCount();
        result.ddmcLeaks += physics->getDDMCLeakCount();
        result.ddmcFallbacks += physics->getDDMCFallbackCount();
        result.movingInterfaceBypasses += physics->getDDMCMovingInterfaceBypassCount();
        result.maxMovingInterfaceFactor = std::max(result.maxMovingInterfaceFactor,
                                                   physics->getDDMCMovingInterfaceMaxFactor());

        const std::vector<double> &tally = physics->getEradTimeAvg();
        if(tally.size() != cellCount)
        {
            StormError eo("ddmc_advection_interface: radiation energy tally has the wrong size");
            eo.addEntry("Tally size", tally.size());
            eo.addEntry("Cells", cellCount);
            throw eo;
        }
        std::vector<double> census(cellCount, 0.0);
        for(const MCParticle &particle : std::as_const(manager).getParticles())
        {
            if(particle.cellIndex >= cellCount)
            {
                StormError eo("ddmc_advection_interface: a census packet has no local cell");
                eo.addEntry("Particle", particle);
                throw eo;
            }
            census[particle.cellIndex] +=
                ComovingEnergy(particle, cells[particle.cellIndex].velocity, lightSpeed) /
                grid.GetVolume(particle.cellIndex);
        }

        double stepDeviation = 0.0;
        for(std::size_t rank = N_THICK - WINDOW_HALF_WIDTH; rank < N_THICK + WINDOW_HALF_WIDTH; ++rank)
        {
            std::size_t const i = result.xOrder[rank];
            stepDeviation = std::max(stepDeviation, std::abs(tally[i] / result.referenceEnergyDensity - 1.0));
        }
        if(step >= firstAveragedStep)
        {
            for(std::size_t i = 0; i < cellCount; ++i)
            {
                tallySum[i] += tally[i];
                censusSum[i] += census[i];
            }
            ++result.averagedSteps;
        }

        std::cout << "[" << variant.name << "] step " << step + 1 << "/" << options.steps
                  << std::scientific << std::setprecision(4)
                  << "  t=" << simTime << " s"
                  << "  packets=" << std::as_const(manager).getParticles().size()
                  << "  E(last DDMC)/E0="
                  << tally[result.xOrder[N_THICK - 1]] / result.referenceEnergyDensity
                  << "  E(first IMC)/E0="
                  << tally[result.xOrder[N_THICK]] / result.referenceEnergyDensity
                  << "  window max|E/E0-1|=" << stepDeviation
                  << (step >= firstAveragedStep ? "  (averaged)" : "")
                  << std::defaultfloat << std::endl;
    }
    result.time = simTime;
    result.finalEnergy = TotalComovingEnergy(std::as_const(manager).getParticles(), cells, lightSpeed);

    result.tallyProfile.assign(cellCount, 0.0);
    result.censusProfile.assign(cellCount, 0.0);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        long double const norm = static_cast<long double>(result.averagedSteps) * result.referenceEnergyDensity;
        result.tallyProfile[i] = static_cast<double>(tallySum[i] / norm);
        result.censusProfile[i] = static_cast<double>(censusSum[i] / norm);
    }
    result.tally = Summarize(result.tallyProfile, result.xOrder);
    result.census = Summarize(result.censusProfile, result.xOrder);

    const auto &ddmcData = physics->getDDMCCellData();
    result.eligible.assign(cellCount, 0);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        bool const eligible = i < ddmcData.size() && ddmcData[i].eligible;
        result.eligible[i] = eligible ? 1 : 0;
        result.ddmcEligibleCells += eligible ? 1 : 0;
        if(eligible != IsThickCell(grid.GetMeshPoint(i).x))
        {
            ++result.eligibilityMismatches;
        }
    }
    result.maxCellPeclet = physics->getDDMCMaxCellAdvectionPeclet();
    result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    return result;
}

// ---- Reporting ------------------------------------------------------------------------------

MetricsLine Metrics(const InterfaceResult &result)
{
    MetricsLine line(result.variant.name);
    line.AddFlag("advection", result.variant.advection)
        .AddFlag("moving_correction", result.variant.movingCorrection)
        .AddFlag("gated", result.variant.gated)
        .AddReal("beta", result.variant.beta)
        .AddReal("v_cm_per_s", result.speed)
        .AddReal("t_s", result.time)
        .AddReal("vt_cm", result.speed * result.time)
        .AddCount("cells", result.cells)
        .AddCount("thick_cells", N_THICK)
        .AddCount("window_half_width", WINDOW_HALF_WIDTH)
        .AddCount("packets_per_cell", result.packetsPerCell)
        .AddCount("averaged_steps", result.averagedSteps)
        .AddReal("e0_reference", result.referenceEnergyDensity)
        .AddCount("ddmc_eligible_cells", result.ddmcEligibleCells)
        .AddCount("expected_ddmc_cells", N_THICK)
        .AddCount("eligibility_mismatches", result.eligibilityMismatches)
        .AddReal("e_ratio_interface", result.tally.interfaceRatio)
        .AddReal("max_dev_window", result.tally.maxWindowDeviation)
        .AddReals("window_profile", result.tally.window)
        .AddReal("thick_bulk_rel", result.tally.thickBulk)
        .AddReal("thin_bulk_rel", result.tally.thinBulk)
        .AddReal("e_ratio_interface_census", result.census.interfaceRatio)
        .AddReal("max_dev_window_census", result.census.maxWindowDeviation)
        .AddReals("window_profile_census", result.census.window)
        .AddReal("thick_bulk_rel_census", result.census.thickBulk)
        .AddReal("thin_bulk_rel_census", result.census.thinBulk)
        .AddReal("e0_rel_change", (result.finalEnergy - result.initialEnergy) / result.initialEnergy)
        .AddCount("advective_leaks", result.advectiveLeaks)
        .AddCount("ddmc_leaks", result.ddmcLeaks)
        .AddCount("ddmc_fallbacks", result.ddmcFallbacks)
        .AddCount("moving_interface_bypasses", result.movingInterfaceBypasses)
        .AddReal("max_moving_interface_factor", result.maxMovingInterfaceFactor)
        .AddReal("max_cell_peclet", result.maxCellPeclet)
        .AddReal("wall_s", result.wallSeconds);
    return line;
}

void WriteOutputs(const std::vector<InterfaceResult> &results, const std::vector<std::string> &metricLines)
{
    {
        std::ofstream out(METRICS_PATH);
        out << "# ddmc_advection_interface metrics: one line per variant, space-separated key=value.\n";
        out << "# Checked by examples/ddmc_advection/check_ddmc_advection.py interface <this file>.\n";
        for(const std::string &line : metricLines)
        {
            out << line << "\n";
        }
        if(!out)
        {
            StormError eo("ddmc_advection_interface: could not write the metrics file");
            eo.addEntry("Path", METRICS_PATH);
            throw eo;
        }
    }
    {
        std::ofstream out(PROFILES_PATH);
        out << "# variant x_center_cm tally_E_over_E0 census_E_over_E0 thick ddmc_eligible\n";
        out << "# Profiles are averaged over the steps after "
            << AVERAGE_START_FRACTION << " of the run; the interface is at x = "
            << static_cast<double>(N_THICK) * CELL_WIDTH << " cm.\n";
        out << std::scientific << std::setprecision(8);
        for(const InterfaceResult &result : results)
        {
            for(std::size_t rank = 0; rank < result.xOrder.size(); ++rank)
            {
                std::size_t const i = result.xOrder[rank];
                double const x = (static_cast<double>(rank) + 0.5) * CELL_WIDTH;
                out << result.variant.name << " " << x << " " << result.tallyProfile[i] << " "
                    << result.censusProfile[i] << " " << (rank < N_THICK ? 1 : 0) << " "
                    << result.eligible[i] << "\n";
            }
        }
    }
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        InterfaceOptions const options = ParseOptions(argc, argv);
        std::cout << "DDMC advection interface: " << N_THICK << " thick (sigma=" << THICK_OPACITY
                  << " /cm, DDMC) + " << N_THIN << " thin (sigma=" << THIN_OPACITY
                  << " /cm, IMC) cells of " << CELL_WIDTH << " x " << TRANSVERSE_WIDTH << " x "
                  << TRANSVERSE_WIDTH << " cm, beta=" << BETA << ", v t=" << ADVECTED_CELLS * CELL_WIDTH
                  << " cm, uniform comoving E0=" << ENERGY_DENSITY << " erg/cm^3" << std::endl;

        std::vector<InterfaceResult> results;
        std::vector<std::string> metricLines;
        for(const InterfaceVariant &variant : options.variants)
        {
            results.push_back(RunVariant(variant, options));
            metricLines.push_back(Metrics(results.back()).str());
            std::cout << METRICS_TAG << " " << metricLines.back() << std::endl;
        }
        WriteOutputs(results, metricLines);

        std::cout << "\nSummary (expected: flat, E/E0 = 1, for moving_correction and static_control):\n";
        for(const InterfaceResult &result : results)
        {
            std::cout << "  " << std::left << std::setw(21) << result.variant.name << std::right
                      << std::fixed << std::setprecision(4)
                      << "  E(DDMC)/E(IMC) at interface=" << result.tally.interfaceRatio
                      << "  window max|E/E0-1|=" << result.tally.maxWindowDeviation
                      << "  DDMC cells=" << result.ddmcEligibleCells << "/" << N_THICK
                      << std::defaultfloat << "  wall=" << result.wallSeconds << " s" << std::endl;
        }
        std::cout << "Wrote " << METRICS_PATH << ", " << PROFILES_PATH << "\n"
                  << "Check: python3 ../ddmc_advection/check_ddmc_advection.py interface " << METRICS_PATH
                  << std::endl;
    }
    catch(const std::exception &error)
    {
        std::cerr << "ddmc_advection_interface failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
