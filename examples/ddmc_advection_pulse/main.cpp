/*
 * DDMC advection test 1: a radiation pulse carried by a uniformly moving,
 * optically thick, purely scattering medium.
 *
 * Gas fills x in [0, 60] cm (3 x 3 cells of 1 cm across) and moves along +x
 * at v = beta c, beta = 0.03.  It scatters isotropically with sigma = 8 /cm
 * (8 mean free paths per cell, sigma 4V/A = 5.3) and does not absorb, so it
 * does not emit either.  Transport is in the comoving frame (withHydro) and
 * the gas never feels the radiation (noHydroFeedback).  A Gaussian pulse of
 * comoving-isotropic radiation (centre 26 cm, sigma_0 = 2 cm) is the only
 * energy.  With div v = 0 the O(v/c) comoving diffusion equation reduces to
 *
 *   dE0/dt + v . grad E0 = D lap E0,      D = c / (3 sigma),
 *
 * so the comoving energy is conserved, its centroid moves with the gas,
 * x_c(t) = x_c(0) + v t, and its variance grows as var(0) + 2 D t.  The run
 * lasts t = 4 cm / v, while diffusion spreads the pulse by sqrt(2 D t) = 3.3 cm.
 * The pulse stays more than 6 standard deviations from the x walls, which are
 * rigid (RigidBoundary), a face behaviour DDMC supports.
 *
 * Four variants run on the same initial packets:
 *   imc                IMC only (withDDMC = false): the transport reference.
 *   ddmc_static        DDMC with ddmcAdvection = true on a static mesh.
 *   ddmc_moving_mesh   DDMC with ddmcAdvection = true; before every step the
 *                      mesh points and the box are translated to x + v t.
 *   ddmc_no_advection  DDMC with ddmcAdvection = false.  The centroid stays
 *                      put; recorded, not gated.
 *
 * Every packet keeps its comoving energy e0 = gamma e (1 - v.u/c^2) to
 * roundoff (see ComovingEnergy), so the total comoving energy is the
 * conserved quantity checked here, and the centroid and variance are
 * e0-weighted moments of the census packet positions.
 *
 * Usage:
 *   ./ddmc_advection_pulse [variants] [ddmcPackets] [imcPackets] [steps]
 *     variants     "all" (default) or a comma-separated list of the names above
 *     ddmcPackets  packets in each DDMC variant (default 100000)
 *     imcPackets   packets in the IMC variant (default 40000)
 *     steps        census steps over the run (default 20)
 *
 * Output (current directory):
 *   stdout                               one DDMC_ADVECTION_PULSE_METRICS line per variant
 *   ddmc_advection_pulse_metrics.txt     the same key=value lines
 *   ddmc_advection_pulse_history.txt     centroid, variance and energy after every step
 *   ddmc_advection_pulse_profiles.txt    final comoving energy fraction per 1 cm x bin
 * The pass/fail criteria live in ../ddmc_advection/check_ddmc_advection.py.
 */

#ifdef STORM_WITH_MPI
#error "ddmc_advection_pulse is serial: REGRESSION_INFO tags it serial, so CMake builds it without STORM_WITH_MPI"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include <units/units.hpp>

#include "StormError.hpp"
#include "boundary/RigidBoundary.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationIMCParameters.hpp"
#include "population/NoPopulationControl.hpp"
#include "manager/MonteCarloManager.hpp"
#include "mesh_movement/VoronoiMeshMovement.hpp"
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

using Grid = MadVoro::Voronoi3D<Vector3D>;
using MCParticle = STORM::Particle<Vector3D>;
using Opacity = ScatteringOpacity<Grid>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, AdvectionCell, AdvectionExtensives,
                                ConstantCvEOS, 1, Opacity>;

// ---- Problem definition (CGS) ----------------------------------------------

constexpr std::size_t NX = 60;                      // cells along x
constexpr std::size_t NYZ = 3;                      // cells along y and along z
constexpr double CELL_WIDTH = 1.0;                  // [cm], cubic lattice cells
constexpr double SCATTERING_OPACITY = 8.0;          // [1/cm]: 8 mean free paths per cell
// The gate is sigma * 4V/A = 8 * 2/3 = 5.3 for a cubic cell.
constexpr double DDMC_MIN_CELL_OPTICAL_DEPTH = 4.0;
constexpr double BETA = 0.03;                       // gas speed / c, along +x
constexpr double PULSE_CENTER = 26.0;               // [cm]
constexpr double PULSE_SIGMA = 2.0;                 // [cm]
constexpr double PULSE_TRUNCATION = 4.0;            // in units of PULSE_SIGMA
constexpr double ADVECTED_CELLS = 4.0;              // v t / CELL_WIDTH at the end
constexpr double PULSE_ENERGY = 1.0e12;             // [erg], comoving
constexpr double DENSITY = 1.0;                     // [g/cm^3]
constexpr double TEMPERATURE = 1.0e4;               // [K]; any positive value
constexpr double CV_PER_MASS = 1.0e8;               // [erg/g/K]; unused without absorption
constexpr std::uint64_t INITIAL_PACKET_SEED = 0x5eed2026ULL;

constexpr const char *METRICS_PATH = "ddmc_advection_pulse_metrics.txt";
constexpr const char *HISTORY_PATH = "ddmc_advection_pulse_history.txt";
constexpr const char *PROFILES_PATH = "ddmc_advection_pulse_profiles.txt";
constexpr const char *METRICS_TAG = "DDMC_ADVECTION_PULSE_METRICS";

// ---- Variants ----------------------------------------------------------------

struct PulseVariant
{
    std::string name;
    bool ddmc = false;
    bool advection = true;
    bool movingMesh = false;
    bool gated = true;
};

const std::vector<PulseVariant> &AllVariants()
{
    static const std::vector<PulseVariant> variants = {
        {"imc", false, true, false, true},
        {"ddmc_static", true, true, false, true},
        {"ddmc_moving_mesh", true, true, true, true},
        {"ddmc_no_advection", true, false, false, false}};
    return variants;
}

struct PulseOptions
{
    std::vector<PulseVariant> variants = AllVariants();
    std::size_t ddmcPackets = 100000;
    std::size_t imcPackets = 40000;
    std::size_t steps = 20;
};

PulseOptions ParseOptions(int argc, char *argv[])
{
    PulseOptions options;
    if(argc > 5)
    {
        StormError eo("Too many arguments; usage: ddmc_advection_pulse [variants] [ddmcPackets] [imcPackets] [steps]");
        eo.addEntry("Argument count", argc - 1);
        throw eo;
    }
    if(argc >= 2 && std::string(argv[1]) != "all")
    {
        options.variants.clear();
        for(const std::string &name : SplitList(argv[1], ','))
        {
            auto const known = std::find_if(AllVariants().begin(), AllVariants().end(),
                                            [&name](const PulseVariant &variant)
                                            {
                                                return variant.name == name;
                                            });
            if(known == AllVariants().end())
            {
                StormError eo("Unknown variant; use all, imc, ddmc_static, ddmc_moving_mesh or ddmc_no_advection");
                eo.addEntry("Variant", name);
                throw eo;
            }
            bool const duplicate = std::any_of(options.variants.begin(), options.variants.end(),
                                               [&name](const PulseVariant &variant)
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
        options.ddmcPackets = ParsePositiveCount(argv[2], "ddmcPackets");
    }
    if(argc >= 4)
    {
        options.imcPackets = ParsePositiveCount(argv[3], "imcPackets");
    }
    if(argc >= 5)
    {
        options.steps = ParsePositiveCount(argv[4], "steps");
    }
    return options;
}

// ---- Mesh ----------------------------------------------------------------------

Vector3D BoxLower(double shift)
{
    return Vector3D(shift, 0.0, 0.0);
}

Vector3D BoxUpper(double shift)
{
    return Vector3D(shift + static_cast<double>(NX) * CELL_WIDTH,
                    static_cast<double>(NYZ) * CELL_WIDTH,
                    static_cast<double>(NYZ) * CELL_WIDTH);
}

// Cubic lattice translated by `shift` along x.  Serial Build keeps the point
// order, so cell i is always lattice point i.
std::vector<Vector3D> LatticePoints(double shift)
{
    std::vector<Vector3D> points;
    points.reserve(NX * NYZ * NYZ);
    for(std::size_t ix = 0; ix < NX; ++ix)
    {
        for(std::size_t iy = 0; iy < NYZ; ++iy)
        {
            for(std::size_t iz = 0; iz < NYZ; ++iz)
            {
                points.emplace_back(shift + (static_cast<double>(ix) + 0.5) * CELL_WIDTH,
                                    (static_cast<double>(iy) + 0.5) * CELL_WIDTH,
                                    (static_cast<double>(iz) + 0.5) * CELL_WIDTH);
            }
        }
    }
    return points;
}

void BuildMesh(Grid &grid, double shift)
{
    grid.SetBox(BoxLower(shift), BoxUpper(shift));
    grid.Build(LatticePoints(shift));
    if(grid.GetPointNo() != NX * NYZ * NYZ)
    {
        StormError eo("ddmc_advection_pulse: the Voronoi build lost lattice cells");
        eo.addEntry("Expected cells", NX * NYZ * NYZ);
        eo.addEntry("Built cells", grid.GetPointNo());
        eo.addEntry("Shift [cm]", shift);
        throw eo;
    }
}

// Moves the mesh with the gas: the lattice and the box are translated to
// `shift` along x and the tessellation is rebuilt.  Packets keep their lab
// positions and are reassigned to the cells that now contain them, as in
// examples/moving_slab.  Packets left behind outside the box are dropped; the
// returned comoving energy of those is expected to be zero, since the pulse
// never reaches the walls.
double MoveMeshWithGas(Grid &grid, double shift,
                       const std::vector<AdvectionCell> &cells,
                       std::vector<AdvectionExtensives> &extensives,
                       std::vector<MCParticle> &particles,
                       const Vector3D &gasVelocity, double lightSpeed)
{
    BuildMesh(grid, shift);
    if(cells.size() != grid.GetPointNo() || extensives.size() != grid.GetPointNo())
    {
        StormError eo("ddmc_advection_pulse: cell arrays do not match the rebuilt mesh");
        eo.addEntry("Cells", cells.size());
        eo.addEntry("Mesh cells", grid.GetPointNo());
        throw eo;
    }
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        extensives[i].mass = cells[i].density * grid.GetVolume(i);
    }

    double removedEnergy = 0.0;
    particles.erase(std::remove_if(particles.begin(), particles.end(),
                                   [&](const MCParticle &particle)
                                   {
                                       if(!grid.IsPointOutsideBox(particle.location))
                                       {
                                           return false;
                                       }
                                       removedEnergy += ComovingEnergy(particle, gasVelocity, lightSpeed);
                                       return true;
                                   }),
                    particles.end());

    std::vector<std::size_t> cellIDs(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        cellIDs[i] = cells[i].ID;
    }
    STORM::MeshMovement<Vector3D, Grid>::UpdateNewCells(grid, particles, cellIDs);
    for(MCParticle &particle : particles)
    {
        if(particle.cellIndex >= cells.size())
        {
            StormError eo("ddmc_advection_pulse: a packet was not placed in a cell after the mesh moved");
            eo.addEntry("Cell index", particle.cellIndex);
            eo.addEntry("Location", particle.location);
            throw eo;
        }
        particle.cellID = cells[particle.cellIndex].ID;
    }
    return removedEnergy;
}

// ---- Initial condition ------------------------------------------------------------

// Gaussian pulse in x (truncated at PULSE_TRUNCATION sigma), uniform across
// y and z, isotropic in the comoving frame, every packet with the same
// comoving energy.  The same deterministic stream feeds every variant, so a
// smaller packet count takes a prefix of the same packets.
std::vector<MCParticle> MakePulsePackets(const Grid &grid,
                                         const std::vector<AdvectionCell> &cells,
                                         std::size_t count,
                                         const Vector3D &gasVelocity,
                                         double lightSpeed)
{
    OpenUnitStream stream(INITIAL_PACKET_SEED);
    auto const [lower, upper] = grid.GetBoxCoordinates();
    double const comovingEnergy = PULSE_ENERGY / static_cast<double>(count);

    std::vector<MCParticle> particles;
    particles.reserve(count);
    while(particles.size() < count)
    {
        double const offset = PULSE_SIGMA * stream.NextNormal();
        if(std::abs(offset) > PULSE_TRUNCATION * PULSE_SIGMA)
        {
            continue;
        }
        MCParticle particle;
        particle.location = Vector3D(PULSE_CENTER + offset,
                                     lower.y + (upper.y - lower.y) * stream.Next(),
                                     lower.z + (upper.z - lower.z) * stream.Next());
        std::size_t const cellIndex = grid.GetContainingCell(particle.location);
        if(cellIndex >= cells.size() || grid.IsPointOutsideBox(particle.location))
        {
            StormError eo("ddmc_advection_pulse: an initial packet is outside the mesh");
            eo.addEntry("Location", particle.location);
            eo.addEntry("Cell index", cellIndex);
            throw eo;
        }
        particle.cellIndex = cellIndex;
        particle.cellID = cells[cellIndex].ID;
        particle.sourceCellID = particle.cellID;
        particle.frequency = units::kev;
        SetLabPacketFromComoving(particle, IsotropicDirection(stream), comovingEnergy,
                                 gasVelocity, lightSpeed);
        particle.initialWeight = std::abs(particle.weight);
        SetPacketIdentity(particle, particles.size(), INITIAL_PACKET_SEED);
        particles.push_back(particle);
    }
    return particles;
}

// ---- Measurement ------------------------------------------------------------------------

struct PulseMoments
{
    double comovingEnergy = 0.0;   // [erg]
    double labEnergy = 0.0;        // [erg]
    double centroid = 0.0;         // [cm], comoving-energy weighted
    double variance = 0.0;         // [cm^2], comoving-energy weighted
    std::size_t packets = 0;
};

PulseMoments MeasurePulse(const std::vector<MCParticle> &particles,
                          const Vector3D &gasVelocity, double lightSpeed)
{
    PulseMoments moments;
    moments.packets = particles.size();
    std::vector<double> energies(particles.size());
    long double energy = 0.0L;
    long double labEnergy = 0.0L;
    long double firstMoment = 0.0L;
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        double const e0 = ComovingEnergy(particles[i], gasVelocity, lightSpeed);
        if(!std::isfinite(e0) || !std::isfinite(particles[i].location.x))
        {
            StormError eo("ddmc_advection_pulse: a census packet has a non-finite energy or position");
            eo.addEntry("Particle", particles[i]);
            throw eo;
        }
        energies[i] = e0;
        energy += e0;
        labEnergy += particles[i].weight;
        firstMoment += static_cast<long double>(e0) * particles[i].location.x;
    }
    if(!(energy > 0.0L))
    {
        StormError eo("ddmc_advection_pulse: no radiation energy left to measure");
        eo.addEntry("Packets", particles.size());
        throw eo;
    }
    long double const centroid = firstMoment / energy;
    long double secondMoment = 0.0L;
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        long double const offset = particles[i].location.x - centroid;
        secondMoment += energies[i] * offset * offset;
    }
    moments.comovingEnergy = static_cast<double>(energy);
    moments.labEnergy = static_cast<double>(labEnergy);
    moments.centroid = static_cast<double>(centroid);
    moments.variance = static_cast<double>(secondMoment / energy);
    return moments;
}

// Fraction of the comoving energy in lab-frame x bins of CELL_WIDTH from 0.
std::vector<double> EnergyProfile(const std::vector<MCParticle> &particles,
                                  const Vector3D &gasVelocity, double lightSpeed,
                                  std::size_t bins)
{
    std::vector<long double> sums(bins, 0.0L);
    long double total = 0.0L;
    for(const MCParticle &particle : particles)
    {
        double const e0 = ComovingEnergy(particle, gasVelocity, lightSpeed);
        total += e0;
        double const position = std::floor(particle.location.x / CELL_WIDTH);
        if(position >= 0.0 && position < static_cast<double>(bins))
        {
            sums[static_cast<std::size_t>(position)] += e0;
        }
    }
    std::vector<double> fractions(bins, 0.0);
    for(std::size_t bin = 0; bin < bins; ++bin)
    {
        fractions[bin] = total > 0.0L ? static_cast<double>(sums[bin] / total) : 0.0;
    }
    return fractions;
}

// ---- One variant ----------------------------------------------------------------------------

struct HistoryRow
{
    std::size_t step = 0;
    double time = 0.0;
    double centroid = 0.0;
    double variance = 0.0;
    double energyChange = 0.0;
};

struct PulseResult
{
    PulseVariant variant;
    double lightSpeed = 0.0;
    double speed = 0.0;
    double time = 0.0;
    std::size_t cells = 0;
    PulseMoments initialMoments;
    PulseMoments finalMoments;
    double removedEnergy = 0.0;
    std::size_t ddmcEligibleCells = 0;
    std::size_t advectiveLeaks = 0;
    std::size_t ddmcLeaks = 0;
    std::size_t ddmcEvents = 0;
    std::size_t ddmcFallbacks = 0;
    double maxCellPeclet = 0.0;
    std::array<std::size_t, 6> pecletHistogram{};
    // max |div v| h / v over cells that do not touch an x wall; zero up to
    // roundoff for a uniform gas velocity.
    double maxInteriorDivergence = 0.0;
    double wallSeconds = 0.0;
    std::vector<HistoryRow> history;
    std::vector<double> profile;
};

PulseResult RunVariant(const PulseVariant &variant, const PulseOptions &options)
{
    auto const wallStart = std::chrono::steady_clock::now();

    STORM::RadiationIMCParameters<1> parameters;
    double const lightSpeed = parameters.lightSpeed;
    double const speed = BETA * lightSpeed;
    Vector3D const gasVelocity(speed, 0.0, 0.0);
    double const endTime = ADVECTED_CELLS * CELL_WIDTH / speed;
    double const dt = endTime / static_cast<double>(options.steps);

    Grid grid(BoxLower(0.0), BoxUpper(0.0));
    BuildMesh(grid, 0.0);
    std::size_t const cellCount = grid.GetPointNo();

    std::vector<AdvectionCell> cells(cellCount);
    std::vector<AdvectionExtensives> extensives(cellCount);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        double const volume = grid.GetVolume(i);
        cells[i].ID = i;
        cells[i].temperature = TEMPERATURE;
        cells[i].density = DENSITY;
        cells[i].velocity = gasVelocity;
        cells[i].scatteringOpacity = SCATTERING_OPACITY;
        cells[i].internalEnergy = CV_PER_MASS * TEMPERATURE * DENSITY * volume;
        extensives[i].mass = DENSITY * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    // RadiationIMC requires a positive emission budget; with zero absorption
    // every cell's emission energy is zero and no packet is emitted.
    parameters.newPhotonsPerCell = 1;
    // Comoving-frame transport (Lorentz-transformed scattering for IMC,
    // comoving residents for DDMC) with the gas velocity held fixed.
    parameters.withHydro = true;
    parameters.noHydroFeedback = true;
    parameters.withDDMC = variant.ddmc;
    parameters.ddmcMinCellOpticalDepth = DDMC_MIN_CELL_OPTICAL_DEPTH;
    parameters.ddmcAdvection = variant.advection;
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

    std::size_t const packets = variant.ddmc ? options.ddmcPackets : options.imcPackets;
    manager.getParticles() = MakePulsePackets(grid, cells, packets, gasVelocity, lightSpeed);

    PulseResult result;
    result.variant = variant;
    result.lightSpeed = lightSpeed;
    result.speed = speed;
    result.cells = cellCount;
    result.initialMoments = MeasurePulse(std::as_const(manager).getParticles(), gasVelocity, lightSpeed);
    result.finalMoments = result.initialMoments;

    std::cout << "\n=== ddmc_advection_pulse variant " << variant.name
              << ": ddmc=" << variant.ddmc << " advection=" << variant.advection
              << " movingMesh=" << variant.movingMesh << " packets=" << packets
              << " cells=" << cellCount << " dt=" << dt << " s steps=" << options.steps
              << " ===" << std::endl;

    double simTime = 0.0;
    for(std::size_t step = 0; step < options.steps; ++step)
    {
        if(variant.movingMesh && step > 0)
        {
            result.removedEnergy += MoveMeshWithGas(grid, speed * simTime, cells, extensives,
                                                    manager.getParticles(), gasVelocity, lightSpeed);
        }

        manager.step(dt);
        simTime += dt;

        // RadiationIMC resets these counters at every preStep.
        result.advectiveLeaks += physics->getDDMCAdvectiveLeakCount();
        result.ddmcLeaks += physics->getDDMCLeakCount();
        result.ddmcEvents += physics->getDDMCStepCount();
        result.ddmcFallbacks += physics->getDDMCFallbackCount();

        result.finalMoments = MeasurePulse(std::as_const(manager).getParticles(), gasVelocity, lightSpeed);
        HistoryRow row;
        row.step = step + 1;
        row.time = simTime;
        row.centroid = result.finalMoments.centroid;
        row.variance = result.finalMoments.variance;
        row.energyChange = (result.finalMoments.comovingEnergy + result.removedEnergy - result.initialMoments.comovingEnergy) /
                           result.initialMoments.comovingEnergy;
        result.history.push_back(row);

        std::cout << "[" << variant.name << "] step " << row.step << "/" << options.steps
                  << std::scientific << std::setprecision(6)
                  << "  t=" << row.time << " s"
                  << "  packets=" << result.finalMoments.packets
                  << "  dx/(v t)=" << (row.centroid - result.initialMoments.centroid) / (speed * simTime)
                  << "  var=" << row.variance << " cm^2"
                  << "  dE0/E0=" << row.energyChange
                  << std::defaultfloat << std::endl;
    }
    result.time = simTime;

    if(variant.ddmc)
    {
        const auto &ddmcData = physics->getDDMCCellData();
        auto const [lower, upper] = grid.GetBoxCoordinates();
        for(std::size_t i = 0; i < ddmcData.size() && i < cellCount; ++i)
        {
            if(ddmcData[i].eligible)
            {
                ++result.ddmcEligibleCells;
            }
            double const x = grid.GetMeshPoint(i).x;
            bool const touchesXWall = (x - lower.x < CELL_WIDTH) || (upper.x - x < CELL_WIDTH);
            if(!touchesXWall)
            {
                result.maxInteriorDivergence = std::max(result.maxInteriorDivergence,
                    std::abs(ddmcData[i].velocityDivergence) * CELL_WIDTH / speed);
            }
        }
        result.maxCellPeclet = physics->getDDMCMaxCellAdvectionPeclet();
        result.pecletHistogram = physics->getDDMCAdvectionPecletHistogram();
    }

    std::size_t const bins = static_cast<std::size_t>(
        std::ceil((static_cast<double>(NX) * CELL_WIDTH + speed * simTime) / CELL_WIDTH));
    result.profile = EnergyProfile(std::as_const(manager).getParticles(), gasVelocity, lightSpeed, bins);
    result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    return result;
}

// ---- Reporting ------------------------------------------------------------------------------

MetricsLine Metrics(const PulseResult &result)
{
    double const vt = result.speed * result.time;
    double const diffusion = result.lightSpeed / (3.0 * SCATTERING_OPACITY);
    double const dx = result.finalMoments.centroid - result.initialMoments.centroid;
    double const varianceDiffusion = result.initialMoments.variance + 2.0 * diffusion * result.time;
    // First-order upwind advection adds the numerical diffusion v h / 2.
    double const varianceUpwind = result.initialMoments.variance +
                                  2.0 * (diffusion + 0.5 * result.speed * CELL_WIDTH) * result.time;
    double const centroidError = std::sqrt(result.finalMoments.variance /
                                           static_cast<double>(std::max<std::size_t>(1, result.finalMoments.packets)));
    double const energyChange = (result.finalMoments.comovingEnergy + result.removedEnergy -
                                 result.initialMoments.comovingEnergy) / result.initialMoments.comovingEnergy;
    double const labEnergyChange = (result.finalMoments.labEnergy - result.initialMoments.labEnergy) / result.initialMoments.labEnergy;

    std::string histogram;
    for(std::size_t bin = 0; bin < result.pecletHistogram.size(); ++bin)
    {
        histogram += (bin > 0 ? "," : "") + std::to_string(result.pecletHistogram[bin]);
    }

    MetricsLine line(result.variant.name);
    line.AddFlag("ddmc", result.variant.ddmc)
        .AddFlag("advection", result.variant.advection)
        .AddFlag("moving_mesh", result.variant.movingMesh)
        .AddFlag("gated", result.variant.gated)
        .AddReal("beta", BETA)
        .AddReal("v_cm_per_s", result.speed)
        .AddReal("t_s", result.time)
        .AddReal("vt_cm", vt)
        .AddReal("cell_width_cm", CELL_WIDTH)
        .AddReal("sigma_per_cm", SCATTERING_OPACITY)
        .AddReal("diffusion_cm2_per_s", diffusion)
        .AddCount("cells", result.cells)
        .AddCount("packets", result.initialMoments.packets)
        .AddCount("final_packets", result.finalMoments.packets)
        .AddReal("x0_cm", result.initialMoments.centroid)
        .AddReal("x1_cm", result.finalMoments.centroid)
        .AddReal("dx_cm", dx)
        .AddReal("dx_over_vt", dx / vt)
        .AddReal("centroid_err_over_vt", centroidError / vt)
        .AddReal("var0_cm2", result.initialMoments.variance)
        .AddReal("var1_cm2", result.finalMoments.variance)
        .AddReal("var_diffusion_cm2", varianceDiffusion)
        .AddReal("var_ratio_diffusion", result.finalMoments.variance / varianceDiffusion)
        .AddReal("var_upwind_cm2", varianceUpwind)
        .AddReal("var_ratio_upwind", result.finalMoments.variance / varianceUpwind)
        .AddReal("e0_initial", result.initialMoments.comovingEnergy)
        .AddReal("e0_final", result.finalMoments.comovingEnergy)
        .AddReal("e0_removed", result.removedEnergy)
        .AddReal("e0_rel_change", energyChange)
        .AddReal("lab_energy_rel_change", labEnergyChange)
        .AddCount("ddmc_eligible_cells", result.ddmcEligibleCells)
        .AddCount("advective_leaks", result.advectiveLeaks)
        .AddCount("ddmc_leaks", result.ddmcLeaks)
        .AddCount("ddmc_events", result.ddmcEvents)
        .AddCount("ddmc_fallbacks", result.ddmcFallbacks)
        .AddReal("max_cell_peclet", result.maxCellPeclet)
        .AddText("peclet_histogram", histogram)
        .AddReal("max_interior_divv_h_over_v", result.maxInteriorDivergence)
        .AddReal("wall_s", result.wallSeconds);
    return line;
}

void WriteOutputs(const std::vector<PulseResult> &results, const std::vector<std::string> &metricLines)
{
    {
        std::ofstream out(METRICS_PATH);
        out << "# ddmc_advection_pulse metrics: one line per variant, space-separated key=value.\n";
        out << "# Checked by examples/ddmc_advection/check_ddmc_advection.py pulse <this file>.\n";
        for(const std::string &line : metricLines)
        {
            out << line << "\n";
        }
        if(!out)
        {
            StormError eo("ddmc_advection_pulse: could not write the metrics file");
            eo.addEntry("Path", METRICS_PATH);
            throw eo;
        }
    }
    {
        std::ofstream out(HISTORY_PATH);
        out << "# variant step t_s centroid_cm variance_cm2 comoving_energy_rel_change\n";
        out << std::scientific << std::setprecision(10);
        for(const PulseResult &result : results)
        {
            out << result.variant.name << " 0 0 " << result.initialMoments.centroid << " "
                << result.initialMoments.variance << " 0\n";
            for(const HistoryRow &row : result.history)
            {
                out << result.variant.name << " " << row.step << " " << row.time << " " << row.centroid
                    << " " << row.variance << " " << row.energyChange << "\n";
            }
        }
    }
    {
        std::ofstream out(PROFILES_PATH);
        out << "# variant x_center_cm comoving_energy_fraction (final census, 1 cm lab-frame bins)\n";
        out << std::scientific << std::setprecision(8);
        for(const PulseResult &result : results)
        {
            for(std::size_t bin = 0; bin < result.profile.size(); ++bin)
            {
                out << result.variant.name << " " << (static_cast<double>(bin) + 0.5) * CELL_WIDTH << " "
                    << result.profile[bin] << "\n";
            }
        }
    }
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        PulseOptions const options = ParseOptions(argc, argv);
        std::cout << "DDMC advection pulse: " << NX << "x" << NYZ << "x" << NYZ << " Voronoi lattice of "
                  << CELL_WIDTH << " cm cells, sigma_s=" << SCATTERING_OPACITY << " /cm, beta=" << BETA
                  << ", pulse x0=" << PULSE_CENTER << " cm sigma0=" << PULSE_SIGMA
                  << " cm, v t=" << ADVECTED_CELLS * CELL_WIDTH << " cm" << std::endl;

        std::vector<PulseResult> results;
        std::vector<std::string> metricLines;
        for(const PulseVariant &variant : options.variants)
        {
            results.push_back(RunVariant(variant, options));
            metricLines.push_back(Metrics(results.back()).str());
            std::cout << METRICS_TAG << " " << metricLines.back() << std::endl;
        }
        WriteOutputs(results, metricLines);

        std::cout << "\nSummary (expected: dx/(v t) = 1 except ddmc_no_advection; dE0/E0 at roundoff):\n";
        for(const PulseResult &result : results)
        {
            double const vt = result.speed * result.time;
            double const diffusion = result.lightSpeed / (3.0 * SCATTERING_OPACITY);
            std::cout << "  " << std::left << std::setw(18) << result.variant.name << std::right
                      << std::fixed << std::setprecision(4)
                      << "  dx/(v t)=" << (result.finalMoments.centroid - result.initialMoments.centroid) / vt
                      << "  var/(var0+2Dt)="
                      << result.finalMoments.variance / (result.initialMoments.variance + 2.0 * diffusion * result.time)
                      << std::scientific << std::setprecision(3)
                      << "  dE0/E0="
                      << (result.finalMoments.comovingEnergy + result.removedEnergy - result.initialMoments.comovingEnergy) /
                             result.initialMoments.comovingEnergy
                      << std::defaultfloat << "  wall=" << result.wallSeconds << " s" << std::endl;
        }
        std::cout << "Wrote " << METRICS_PATH << ", " << HISTORY_PATH << ", " << PROFILES_PATH << "\n"
                  << "Check: python3 ../ddmc_advection/check_ddmc_advection.py pulse " << METRICS_PATH
                  << std::endl;
    }
    catch(const std::exception &error)
    {
        std::cerr << "ddmc_advection_pulse failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
