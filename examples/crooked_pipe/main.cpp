#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <mpi.h>
#include <mpi_utils/mpi_collectives.hpp>
#include <units/units.hpp>

#include "MadVoro/Voronoi3D.hpp"
#include "CrookedPipeBoundary.hpp"
#include "CrookedPipeEOS.hpp"
#include "CrookedPipeOpacity.hpp"
#include "examples/MPI_ParticleDtype.hpp"
#include "examples/Vector3D.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "mesh_movement/VoronoiMeshMovement.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationIMC.hpp"
#include "utils/MpiExchangeGrid.hpp"
#ifdef MADVORO_WITH_VTK
#include "MadVoro/io/vtk/write_vtu_3d.hpp"
#endif
#ifdef MADVORO_WITH_HDF5
#include "MadVoro/io/hdf5/WriteVoronoiHDF5.hpp"
#endif

using Grid = MadVoro::Voronoi3D<Vector3D>;
using Particle = STORM::Particle<Vector3D>;

namespace {

constexpr std::size_t probeCount = 5;
const double probeX[probeCount] = {0.25, 2.75, 3.5, 4.25, 6.75};
const double probeR[probeCount] = {0.0, 0.0, 1.25, 0.0, 0.0};

struct Options
{
    std::size_t backgroundPoints = 20000;
    std::size_t wallLayers = 4;
    std::size_t wallPoints = 0;
    double wallLayerWidth = 0.01;
    std::size_t newPhotonsPerCell = 25;
    std::size_t minPhotonsPerCell = 100;
    std::size_t channelPoints = 0;
    std::size_t wallThinLayers = 3;
    double wallTangential = 0.04;
    double wallGrowth = 2.0;
    bool randomWalls = false;
    bool ddmc = false;
    double ddmcMinTau = 15.0;
    // Bounding experiments: override the benchmark heat capacities (erg/keV/cm^3).
    double thickCvPerVolume = 1.0e16;
    double thinCvPerVolume = 1.0e13;
    double thickOpacity = 2000.0;
    double thinOpacity = 0.2;
    bool reflectSource = false;
    bool reflectExit = false;
    double sourceScale = 1.0;
    bool energyLedger = false;
    double initialDt = 1.0e-11;
    double dtGrowth = 1.1;
    double fleckScale = 1.0;
    double probeRadius = 0.1;
    std::size_t censusFloorPerCell = 0;
    std::size_t censusCapPerCell = 0;
    std::size_t emissionFloorPerCell = 0;
    std::size_t boundaryPhotonsPerFace = 100;
    double transportWorkScale = 1.0;
    std::size_t transportInnerSteps = 0;
    std::size_t transportMinLaunch = std::numeric_limits<std::size_t>::max();
    std::size_t transportHoldSkips = 0;
    std::size_t rebalanceInterval = 10;
    std::size_t maxSteps = std::numeric_limits<std::size_t>::max();
    double finalTime = 1.0e-6;
    double maximumDt = 1.0e-9;
    bool withRandomWalk = true;
    std::string outputProbes;
    std::string outputDir;
    std::size_t snapshotInterval = 50;
    STORM::ManagerType managerType = STORM::ManagerType::Auto;
    STORM::RDMAEngine rdmaEngine = STORM::RDMAEngine::OFI;
    STORM::MonteCarloConfig transportConfig;
};

void PrintUsage(const char *program)
{
    std::cerr << "Transport experiments: --optimized-rdma (fixed queues, fair 65536-event slices, 5 ms flush age),\n"
              << "  --rdma-fixed-queues-off, --receive-scheduling-off,\n"
              << "  --rdma-ring-size N, --rdma-ring-limit N, --transport-event-budget N, --send-max-age-us N\n";
    std::cerr << "Usage: mpirun -np <N> " << program << " [points] [new_photons] [min_photons] [options]\n"
              << "  --points <N>                 Background mesh points (default: 20000)\n"
              << "  --wall-layers <N>            Graded refinement layers on the thick side of each interface (default: 4)\n"
              << "  --wall-width <cm>            Width of the first refinement layer (default: 0.01)\n"
              << "  --wall-points <N>            Points in the first refinement layer of each interface; --random-walls only (default: 4 x points)\n"
              << "  --wall-thin-layers <N>       Graded slabs mirrored on the thin side of each interface (default: 3)\n"
              << "  --wall-tangential <cm>       Lattice spacing along each interface for structured shells (default: 0.04)\n"
              << "  --random-walls               Fill interface shells with isotropic random points instead of structured slabs\n"
              << "  --wall-growth <f>            Thickness ratio between successive structured slabs (default: 2)\n"
              << "  --ddmc                       Transport cells above --ddmc-min-tau with DDMC instead of IMC\n"
              << "  --ddmc-min-tau <tau>         sigma x mean chord above which a cell is DDMC-eligible (default: 15)\n"
              << "  --thick-cv <erg/keV/cm3>     Wall heat capacity per volume; benchmark value 1e16 (bounding tests only)\n"
              << "  --thin-cv <erg/keV/cm3>      Channel heat capacity per volume; benchmark value 1e13 (bounding tests only)\n"
              << "  --thick-opacity <1/cm>       Wall absorption opacity; benchmark 2000 (diagnostic)\n"
              << "  --thin-opacity <1/cm>        Channel absorption opacity; benchmark 0.2 (diagnostic)\n"
              << "  --reflect-source             Mirror outgoing packets at the source disc instead of absorbing them\n"
              << "  --reflect-exit               Mirror outgoing packets at the pipe exit (x = 7) instead of losing them\n"
              << "  --source-scale <f>           Multiply the injected source flux (diagnostic; default: 1)\n"
              << "  --energy-ledger              Print a per-cycle global energy balance\n"
              << "  --initial-dt <s>             First time step; the benchmark ramps from 1e-11 by 1.1x per step (default: 1e-11)\n"
              << "  --fleck-scale <f>            Diagnostic multiplier on the Fleck-factor argument (default: 1)\n"
              << "  --dt-growth <f>              Time-step growth factor per cycle until --max-dt (default: 1.1, the benchmark's)\n"
              << "  --channel-points <N>         Extra points filling the optically thin channel (default: 0)\n"
              << "  --probe-radius <cm>          Probe over the axisymmetric ring within this radius; 0 = single cell (default: 0.1)\n"
              << "  --new-photons <N>            Thermal packets emitted per cell per step (default: 25)\n"
              << "  --min-photons <N>            Population-control target per cell (default: 100)\n"
              << "  --census-floor <N>           Census packets a cell is padded up to; 0 uses --min-photons\n"
              << "  --census-cap <N>             Census packets a cell may hold; 0 uses 20 x --min-photons\n"
              << "  --emission-floor <N>         Emission packets a cell is padded up to; 0 uses --new-photons\n"
              << "  --output-probes <file>        Write five probe temperature histories\n"
              << "  --output-dir <dir>            Production output directory (profile, VTK, latest.h5)\n"
              << "  --snapshot-interval <N>       VTK/HDF5 snapshot every N cycles; 0 writes only at the end (default: 50)\n"
              << "  --final-time <seconds>        End time (default: 1e-6)\n"
              << "  --max-dt <seconds>            Largest time step (default: 1e-9)\n"
              << "  --transport-work-scale <x>    Positive transport weight for mesh balancing\n"
              << "  --transport-inner-steps <N>   Kokkos events per particle per wave (positive)\n"
              << "  --transport-min-launch <N>    Kokkos target batch size; 0 disables holding\n"
              << "  --transport-hold-skips <N>    Maximum polls before launching a small batch (positive)\n"
              << "  --rebalance-interval <N>      Check mesh balance every N cycles after startup (default: 10)\n"
              << "  --max-steps <N>               Stop after at most N steps\n"
              << "  --random-walk                 Enable random-walk acceleration (default)\n"
              << "  --no-random-walk              Disable random-walk acceleration\n"
              << "  --manager <auto|p2p|rdma|legacy>\n"
              << "  --rdma-engine <auto|ofi|ibv|mpi>\n"
              << "  --boundary-photons <N>        Drive packets emitted per boundary face (default: 100)\n"
              << "  --help                        Show this message\n";
}

std::string RequireValue(int argc, char *argv[], int &index)
{
    if(index + 1 >= argc)
    {
        throw std::runtime_error(std::string(argv[index]) + " requires a value");
    }
    ++index;
    return argv[index];
}

STORM::ManagerType ParseManager(const std::string &name)
{
    if(name == "auto")
    {
        return STORM::ManagerType::Auto;
    }
    if(name == "p2p")
    {
        return STORM::ManagerType::P2P;
    }
    if(name == "rdma")
    {
        return STORM::ManagerType::RDMA;
    }
    if(name == "legacy")
    {
        return STORM::ManagerType::Legacy;
    }
    throw std::runtime_error("Unknown manager: " + name);
}

STORM::RDMAEngine ParseRDMAEngine(const std::string &name)
{
    if(name == "auto")
    {
        return STORM::RDMAEngine::Auto;
    }
    if(name == "ofi")
    {
        return STORM::RDMAEngine::OFI;
    }
    if(name == "ibv")
    {
        return STORM::RDMAEngine::IBV;
    }
    if(name == "mpi")
    {
        return STORM::RDMAEngine::MPI;
    }
    throw std::runtime_error("Unknown RDMA engine: " + name);
}

Options ParseOptions(int argc, char *argv[], bool &showHelp)
{
    Options options;
    std::vector<std::string> positional;
    showHelp = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string argument(argv[i]);
        if(argument == "--help" or argument == "-h")
        {
            showHelp = true;
        }
        else if(argument == "--points")
        {
            options.backgroundPoints = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--wall-layers")
        {
            options.wallLayers = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--channel-points")
        {
            options.channelPoints = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--probe-radius")
        {
            options.probeRadius = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.probeRadius) || options.probeRadius < 0.0)
                throw std::runtime_error("--probe-radius must be finite and nonnegative");
        }
        else if(argument == "--wall-points")
        {
            options.wallPoints = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--wall-thin-layers")
        {
            options.wallThinLayers = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--wall-tangential")
        {
            options.wallTangential = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.wallTangential) || options.wallTangential <= 0.0)
                throw std::runtime_error("--wall-tangential must be finite and positive");
        }
        else if(argument == "--random-walls")
        {
            options.randomWalls = true;
        }
        else if(argument == "--wall-growth")
        {
            options.wallGrowth = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.wallGrowth) || options.wallGrowth < 1.0)
                throw std::runtime_error("--wall-growth must be finite and at least 1");
        }
        else if(argument == "--ddmc")
        {
            options.ddmc = true;
        }
        else if(argument == "--dt-growth")
        {
            options.dtGrowth = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.dtGrowth) || options.dtGrowth < 1.0)
                throw std::runtime_error("--dt-growth must be finite and at least 1");
        }
        else if(argument == "--fleck-scale")
        {
            options.fleckScale = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.fleckScale) || options.fleckScale <= 0.0)
                throw std::runtime_error("--fleck-scale must be finite and positive");
        }
        else if(argument == "--initial-dt")
        {
            options.initialDt = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.initialDt) || options.initialDt <= 0.0)
                throw std::runtime_error("--initial-dt must be finite and positive");
        }
        else if(argument == "--energy-ledger")
        {
            options.energyLedger = true;
        }
        else if(argument == "--source-scale")
        {
            options.sourceScale = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.sourceScale) || options.sourceScale <= 0.0)
                throw std::runtime_error("--source-scale must be finite and positive");
        }
        else if(argument == "--reflect-source")
        {
            options.reflectSource = true;
        }
        else if(argument == "--reflect-exit")
        {
            options.reflectExit = true;
        }
        else if(argument == "--thick-opacity")
        {
            options.thickOpacity = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.thickOpacity) || options.thickOpacity <= 0.0)
                throw std::runtime_error("--thick-opacity must be finite and positive");
        }
        else if(argument == "--thin-opacity")
        {
            options.thinOpacity = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.thinOpacity) || options.thinOpacity <= 0.0)
                throw std::runtime_error("--thin-opacity must be finite and positive");
        }
        else if(argument == "--thick-cv")
        {
            options.thickCvPerVolume = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.thickCvPerVolume) || options.thickCvPerVolume <= 0.0)
                throw std::runtime_error("--thick-cv must be finite and positive");
        }
        else if(argument == "--thin-cv")
        {
            options.thinCvPerVolume = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.thinCvPerVolume) || options.thinCvPerVolume <= 0.0)
                throw std::runtime_error("--thin-cv must be finite and positive");
        }
        else if(argument == "--ddmc-min-tau")
        {
            options.ddmcMinTau = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.ddmcMinTau) || options.ddmcMinTau <= 0.0)
                throw std::runtime_error("--ddmc-min-tau must be finite and positive");
        }
        else if(argument == "--wall-width")
        {
            options.wallLayerWidth = std::stod(RequireValue(argc, argv, i));
        }
        else if(argument == "--new-photons")
        {
            options.newPhotonsPerCell = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--min-photons")
        {
            options.minPhotonsPerCell = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--census-floor")
        {
            options.censusFloorPerCell = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--census-cap")
        {
            options.censusCapPerCell = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--emission-floor")
        {
            options.emissionFloorPerCell = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--boundary-photons")
        {
            options.boundaryPhotonsPerFace = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--output-probes")
        {
            options.outputProbes = RequireValue(argc, argv, i);
        }
        else if(argument == "--output-dir")
        {
            options.outputDir = RequireValue(argc, argv, i);
        }
        else if(argument == "--snapshot-interval")
        {
            options.snapshotInterval = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--final-time")
        {
            options.finalTime = std::stod(RequireValue(argc, argv, i));
        }
        else if(argument == "--max-dt")
        {
            options.maximumDt = std::stod(RequireValue(argc, argv, i));
        }
        else if(argument == "--optimized-rdma")
        {
            options.transportConfig.rdmaFixedStepQueues = true;
            options.transportConfig.fairReceiveScheduling = true;
            options.transportConfig.localTransportEventBudget = 65536;
            options.transportConfig.sendBufferMaxAgeMicroseconds = 5000;
        }
        else if(argument == "--rdma-fixed-queues-off")
        {
            options.transportConfig.rdmaFixedStepQueues = false;
        }
        else if(argument == "--receive-scheduling-off")
        {
            options.transportConfig.fairReceiveScheduling = false;
        }
        else if(argument == "--rdma-ring-size" || argument == "--rdma-ring-limit" ||
                argument == "--transport-event-budget" || argument == "--send-max-age-us")
        {
            const auto value = RequireValue(argc, argv, i);
            if(value.empty() || value[0] == '-') throw std::runtime_error(argument + " must be nonnegative");
            size_t consumed = 0;
            const auto number = std::stoull(value, &consumed);
            if(consumed != value.size() || (number == 0 && argument != "--send-max-age-us"))
                throw std::runtime_error("Invalid value for " + argument);
            if(argument == "--rdma-ring-size") options.transportConfig.initialBufferSize = number;
            else if(argument == "--rdma-ring-limit") options.transportConfig.rdmaFixedQueueMaxSize = number;
            else if(argument == "--transport-event-budget") options.transportConfig.localTransportEventBudget = number;
            else options.transportConfig.sendBufferMaxAgeMicroseconds = number;
        }
        else if(argument == "--transport-work-scale")
        {
            options.transportWorkScale = std::stod(RequireValue(argc, argv, i));
            if(!std::isfinite(options.transportWorkScale) || options.transportWorkScale <= 0.0)
                throw std::runtime_error("--transport-work-scale must be finite and positive");
        }
        else if(argument == "--transport-inner-steps")
        {
            options.transportInnerSteps = std::stoull(RequireValue(argc, argv, i));
            if(options.transportInnerSteps == 0)
                throw std::runtime_error("--transport-inner-steps must be positive");
        }
        else if(argument == "--transport-min-launch")
        {
            const auto value = RequireValue(argc, argv, i);
            if(value.empty() || value[0] == '-')
                throw std::runtime_error("--transport-min-launch must be nonnegative");
            options.transportMinLaunch = std::stoull(value);
        }
        else if(argument == "--transport-hold-skips")
        {
            const auto value = RequireValue(argc, argv, i);
            if(value.empty() || value[0] == '-')
                throw std::runtime_error("--transport-hold-skips must be positive");
            options.transportHoldSkips = std::stoull(value);
            if(options.transportHoldSkips == 0)
                throw std::runtime_error("--transport-hold-skips must be positive");
        }
        else if(argument == "--rebalance-interval")
        {
            const auto value = RequireValue(argc, argv, i);
            if(value.empty() || value[0] == '-')
                throw std::runtime_error("--rebalance-interval must be positive");
            options.rebalanceInterval = std::stoull(value);
            if(options.rebalanceInterval == 0)
                throw std::runtime_error("--rebalance-interval must be positive");
        }
        else if(argument == "--max-steps")
        {
            options.maxSteps = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--random-walk")
        {
            options.withRandomWalk = true;
        }
        else if(argument == "--no-random-walk")
        {
            options.withRandomWalk = false;
        }
        else if(argument == "--manager")
        {
            options.managerType = ParseManager(RequireValue(argc, argv, i));
        }
        else if(argument == "--rdma-engine")
        {
            options.rdmaEngine = ParseRDMAEngine(RequireValue(argc, argv, i));
        }
        else if(!argument.empty() and argument[0] == '-')
        {
            throw std::runtime_error("Unknown option: " + argument);
        }
        else
        {
            positional.push_back(argument);
        }
    }
    if(positional.size() > 3)
    {
        throw std::runtime_error("Too many positional arguments");
    }
    if(positional.size() > 0)
    {
        options.backgroundPoints = std::stoull(positional[0]);
    }
    if(positional.size() > 1)
    {
        options.newPhotonsPerCell = std::stoull(positional[1]);
    }
    if(positional.size() > 2)
    {
        options.minPhotonsPerCell = std::stoull(positional[2]);
    }
    if(options.backgroundPoints == 0 or options.wallLayers == 0 or options.wallLayerWidth <= 0.0 or options.newPhotonsPerCell == 0 or options.minPhotonsPerCell == 0 or
       options.boundaryPhotonsPerFace == 0 or options.finalTime <= 0.0 or options.maximumDt <= 0.0)
    {
        throw std::runtime_error("Point, photon, time-step, and final-time values must be positive");
    }
    return options;
}

std::vector<Vector3D> RandomCylinder(std::size_t pointCount, double innerRadius, double outerRadius,
                                     double xMinimum, double xMaximum, boost::mt19937_64 &generator)
{
    boost::random::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Vector3D> points;
    points.reserve(pointCount);
    for(std::size_t i = 0; i < pointCount; ++i)
    {
        double radius = innerRadius + uniform(generator) * (outerRadius - innerRadius);
        double angle = 2.0 * std::acos(-1.0) * uniform(generator);
        double x = xMinimum + uniform(generator) * (xMaximum - xMinimum);
        points.emplace_back(x, radius * std::cos(angle), radius * std::sin(angle));
    }
    return points;
}

std::vector<Vector3D> RandomBox(std::size_t pointCount, const Vector3D &lower, const Vector3D &upper,
                               boost::mt19937_64 &generator)
{
    boost::random::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Vector3D> points;
    points.reserve(pointCount);
    for(std::size_t i = 0; i < pointCount; ++i)
    {
        points.emplace_back(lower.x + uniform(generator) * (upper.x - lower.x),
                            lower.y + uniform(generator) * (upper.y - lower.y),
                            lower.z + uniform(generator) * (upper.z - lower.z));
    }
    return points;
}

void AppendPoints(std::vector<Vector3D> &points, std::vector<Vector3D> &&additional)
{
    points.insert(points.end(), additional.begin(), additional.end());
}

// Radiation entering an optically thick wall is absorbed within a boundary layer far
// thinner than a background cell, so the wall is resolved with geometrically graded
// shells: each successive layer is twice as thick and holds half as many points.
const double domainRadius = 2.0;
const double domainXMinimum = 0.0;
const double domainXMaximum = 7.0;

void AppendGradedRadialShells(std::vector<Vector3D> &points, double interfaceRadius, double xMinimum, double xMaximum,
                              double firstWidth, std::size_t firstPoints, std::size_t layerCount, bool outward,
                              boost::mt19937_64 &generator)
{
    double offset = 0.0;
    for(std::size_t layer = 0; layer < layerCount; ++layer)
    {
        double width = firstWidth * static_cast<double>(1u << layer);
        std::size_t layerPoints = std::max<std::size_t>(1, firstPoints >> layer);
        double inner = outward ? interfaceRadius + offset : interfaceRadius - offset - width;
        double outer = std::min(inner + width, domainRadius);
        inner = std::max(inner, 0.0);
        offset += width;
        if(outer - inner < 1.0e-12)
        {
            break;
        }
        AppendPoints(points, RandomCylinder(layerPoints, inner, outer, xMinimum, xMaximum, generator));
    }
}

void AppendGradedAxialShells(std::vector<Vector3D> &points, double interfaceX, double innerRadius, double outerRadius,
                             double firstWidth, std::size_t firstPoints, std::size_t layerCount, bool outward,
                             double limit, boost::mt19937_64 &generator)
{
    double lowerLimit = outward ? domainXMinimum : limit;
    double upperLimit = outward ? limit : domainXMaximum;
    double offset = 0.0;
    for(std::size_t layer = 0; layer < layerCount; ++layer)
    {
        double width = firstWidth * static_cast<double>(1u << layer);
        std::size_t layerPoints = std::max<std::size_t>(1, firstPoints >> layer);
        double start = std::max(outward ? interfaceX + offset : interfaceX - offset - width, lowerLimit);
        double end = std::min(start + width, upperLimit);
        offset += width;
        if(end - start < 1.0e-12)
        {
            break;
        }
        AppendPoints(points, RandomCylinder(layerPoints, innerRadius, outerRadius, start, end, generator));
    }
}

// ---------------------------------------------------------------------------
// Structured, anisotropic interface shells.
//
// RandomCylinder fills a shell isotropically, so its cells are as wide as they are
// thick and a layer of thickness w costs A/w^2 cells over the ~63 cm^2 of interface:
// 3.9M cells at the 4e-3 cm first opaque cell Steinberg & Heizler use, which is what
// pinned --wall-width at 0.01 and left the first wall cell holding ~15x the heat
// capacity it should. Only the interface normal needs that resolution. Here points sit
// on a lattice with spacing `tangential` along the interface, at the mid-thickness of
// each graded layer, so each Voronoi cell is a slab w thick and ~tangential across and
// a layer costs A/tangential^2 instead. The same lattice is mirrored on the thin side
// so that the Voronoi face between each mirrored pair lies exactly on the interface --
// no labelling rule can then misplace it -- and is graded there too so the slabs hand
// over smoothly to the channel fill. Every point is jittered by a fraction of its
// spacing because a perfect lattice hands the Voronoi builder degenerate cospherical
// configurations.
// ---------------------------------------------------------------------------

struct StructuredShellSpec
{
    double firstWidth;
    double tangential;
    std::size_t thickLayers;
    std::size_t thinLayers;
    // Thickness ratio between successive slabs. Doubling reaches the background cell
    // size in few layers but leaves the second and third slabs 2-4x coarser than the
    // benchmark's ~1.1x logarithmic grid at the depths the Marshak wave reaches over
    // 10-100 ns, where IMC teleportation then runs ahead of the true diffusion front.
    double growth = 2.0;

    double StackThickness(std::size_t layers) const
    {
        if(layers == 0)
        {
            return 0.0;
        }
        if(std::abs(growth - 1.0) < 1.0e-12)
        {
            return firstWidth * static_cast<double>(layers);
        }
        return firstWidth * (std::pow(growth, static_cast<double>(layers)) - 1.0) / (growth - 1.0);
    }
};

// Calls place(side, distance, width) for each graded layer: side is +1 on the thick
// side and -1 on the thin side, distance is from the interface to the layer's
// mid-thickness, and widths double outward from the interface.
template<typename PlaceLayer>
void ForEachGradedLayer(const StructuredShellSpec &spec, PlaceLayer &&place)
{
    for(int side : {+1, -1})
    {
        std::size_t layers = side > 0 ? spec.thickLayers : spec.thinLayers;
        double offset = 0.0;
        for(std::size_t layer = 0; layer < layers; ++layer)
        {
            double width = spec.firstWidth * std::pow(spec.growth, static_cast<double>(layer));
            place(side, offset + 0.5 * width, width);
            offset += width;
        }
    }
}

void AppendStructuredRadialShells(std::vector<Vector3D> &points, double interfaceRadius, double xMinimum,
                                  double xMaximum, const StructuredShellSpec &spec, bool thickOutward,
                                  boost::mt19937_64 &generator)
{
    boost::random::uniform_real_distribution<double> jitter(-0.15, 0.15);
    const double twoPi = 2.0 * std::acos(-1.0);
    const double sign = thickOutward ? 1.0 : -1.0;
    // The tangential jitter must be drawn once per lattice site and shared by every slab
    // in that stack (both sides of the interface). Independent per-point jitter of
    // +-15% of the 0.04-0.06 cm spacing tilts the Voronoi bisector between a mirrored
    // thin/thick pair only 1e-4 cm apart radially, turning the interface into a fuzz
    // ~0.01 cm thick in which wall-labelled cells bulge into the channel (3.5% of the
    // channel volume at 1e-4 cm slabs). With a shared site, mirrored pairs bisect exactly
    // on the interface regardless of slab thickness. Radial jitter stays per point.
    std::size_t azimuthal = std::max<std::size_t>(
        8, static_cast<std::size_t>(std::llround(twoPi * interfaceRadius / spec.tangential)));
    std::size_t axial = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::llround((xMaximum - xMinimum) / spec.tangential)));
    const double dPhi = twoPi / static_cast<double>(azimuthal);
    const double dX = (xMaximum - xMinimum) / static_cast<double>(axial);
    std::vector<double> sitePhi(azimuthal * axial), siteX(azimuthal * axial);
    for(std::size_t i = 0; i < azimuthal; ++i)
    {
        for(std::size_t j = 0; j < axial; ++j)
        {
            sitePhi[i * axial + j] = (static_cast<double>(i) + 0.5 + jitter(generator)) * dPhi;
            siteX[i * axial + j] = xMinimum + (static_cast<double>(j) + 0.5 + jitter(generator)) * dX;
        }
    }
    ForEachGradedLayer(spec, [&](int side, double distance, double width) {
        double radius = interfaceRadius + sign * static_cast<double>(side) * distance;
        if(radius <= 0.5 * width || radius >= domainRadius - 0.5 * width)
        {
            return;
        }
        for(std::size_t site = 0; site < sitePhi.size(); ++site)
        {
            double r = radius + jitter(generator) * 0.5 * width;
            points.emplace_back(siteX[site], r * std::cos(sitePhi[site]), r * std::sin(sitePhi[site]));
        }
    });
}

void AppendStructuredAxialShells(std::vector<Vector3D> &points, double interfaceX, double innerRadius,
                                 double outerRadius, const StructuredShellSpec &spec, bool thickTowardPositiveX,
                                 boost::mt19937_64 &generator)
{
    boost::random::uniform_real_distribution<double> jitter(-0.15, 0.15);
    const double twoPi = 2.0 * std::acos(-1.0);
    const double sign = thickTowardPositiveX ? 1.0 : -1.0;
    // Shared (r, phi) sites for every slab of the stack; see AppendStructuredRadialShells.
    std::size_t rings = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::llround((outerRadius - innerRadius) / spec.tangential)));
    const double dR = (outerRadius - innerRadius) / static_cast<double>(rings);
    std::vector<double> siteR, sitePhi;
    for(std::size_t k = 0; k < rings; ++k)
    {
        double ringRadius = innerRadius + (static_cast<double>(k) + 0.5) * dR;
        std::size_t azimuthal = std::max<std::size_t>(
            6, static_cast<std::size_t>(std::llround(twoPi * ringRadius / spec.tangential)));
        double dPhi = twoPi / static_cast<double>(azimuthal);
        for(std::size_t i = 0; i < azimuthal; ++i)
        {
            siteR.push_back(ringRadius + jitter(generator) * dR);
            sitePhi.push_back((static_cast<double>(i) + 0.5 + jitter(generator)) * dPhi);
        }
    }
    ForEachGradedLayer(spec, [&](int side, double distance, double width) {
        double x = interfaceX + sign * static_cast<double>(side) * distance;
        if(x <= domainXMinimum + 0.5 * width || x >= domainXMaximum - 0.5 * width)
        {
            return;
        }
        for(std::size_t site = 0; site < siteR.size(); ++site)
        {
            double xx = x + jitter(generator) * 0.5 * width;
            points.emplace_back(xx, siteR[site] * std::cos(sitePhi[site]), siteR[site] * std::sin(sitePhi[site]));
        }
    });
}

bool IsThick(const Vector3D &point);

// True when (x, r) lies within `distance` of a material interface, found by probing
// IsThick at +-distance along both axes of the (x, r) plane. Every interface here is a
// cylinder about the x-axis or a plane of constant x, so this is exact except in the
// corner cells where two interfaces meet.
bool NearInterface(double x, double radius, double distance)
{
    auto thickAt = [](double xx, double rr) { return IsThick(Vector3D(xx, std::max(rr, 0.0), 0.0)); };
    bool centre = thickAt(x, radius);
    return thickAt(x + distance, radius) != centre || thickAt(x - distance, radius) != centre ||
           thickAt(x, radius + distance) != centre || thickAt(x, radius - distance) != centre;
}

// The interface shells resolve the wall, but the channel itself is only resolved by
// whatever share of the background points happens to land in it -- a few thousand cells
// for 15 cm^3, i.e. under three cells across the pipe radius. Steinberg & Heizler use a
// uniform 0.01 cm grid *in addition to* their interface refinement; this fills the thin
// region directly so the channel resolution can be set independently of the wall.
// Radial sampling is area-correct (r = R sqrt(u)), unlike RandomCylinder, which is only
// used for thin shells where the 1/r bias is negligible.
// `exclusion` keeps the fill clear of the structured thin-side slabs, which would
// otherwise be cut into by random points landing inside them.
std::vector<Vector3D> RandomThinRegion(std::size_t pointCount, double exclusion, boost::mt19937_64 &generator)
{
    boost::random::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Vector3D> points;
    points.reserve(pointCount);
    const double samplingRadius = 1.5;
    // Rejection sampling inside the bounding cylinder of the pipe; the thin region is
    // about 30% of it, so this terminates quickly. The cap only guards against a
    // geometry change that would make the thin region unreachable.
    const std::size_t attemptLimit = 200 * pointCount + 1000;
    std::size_t attempts = 0;
    while(points.size() < pointCount and attempts < attemptLimit)
    {
        ++attempts;
        double radius = samplingRadius * std::sqrt(uniform(generator));
        double angle = 2.0 * std::acos(-1.0) * uniform(generator);
        double x = domainXMinimum + uniform(generator) * (domainXMaximum - domainXMinimum);
        Vector3D candidate(x, radius * std::cos(angle), radius * std::sin(angle));
        if(IsThick(candidate))
        {
            continue;
        }
        if(exclusion > 0.0 and NearInterface(x, radius, exclusion))
        {
            continue;
        }
        points.push_back(candidate);
    }
    if(points.size() < pointCount)
    {
        throw std::runtime_error("--channel-points: could not sample the thin region");
    }
    return points;
}

std::vector<Vector3D> GeneratePoints(std::size_t backgroundPoints, std::size_t wallLayers, std::size_t wallPoints,
                                     double width, std::size_t channelPoints, std::size_t wallThinLayers,
                                     double wallTangential, double wallGrowth, bool randomWalls,
                                     const Vector3D &lower, const Vector3D &upper)
{
    boost::mt19937_64 generator(42);
    std::vector<Vector3D> points = RandomBox(backgroundPoints, lower, upper, generator);
    double channelExclusion = 0.0;
    // The plug is thick only for x in [3, 4]; the interface at r = 1 exists there alone.
    // Refining r < 1 over the whole elbow, as an earlier version did, dropped half of
    // those points into the thin elbow.
    if(randomWalls)
    {
        std::size_t refinedPoints = wallPoints > 0 ? wallPoints : 4 * backgroundPoints;
        AppendGradedRadialShells(points, 0.5, 0.0, 2.5, width, refinedPoints, wallLayers, true, generator);
        AppendGradedRadialShells(points, 0.5, 4.5, 7.0, width, refinedPoints, wallLayers, true, generator);
        AppendGradedRadialShells(points, 1.5, 2.5, 4.5, width, refinedPoints, wallLayers, true, generator);
        AppendGradedRadialShells(points, 1.0, 3.0, 4.0, width, refinedPoints, wallLayers, false, generator);
        AppendGradedAxialShells(points, 2.5, 0.5, 1.5, width, refinedPoints, wallLayers, false, 0.0, generator);
        AppendGradedAxialShells(points, 4.5, 0.5, 1.5, width, refinedPoints, wallLayers, true, 7.0, generator);
        AppendGradedAxialShells(points, 3.0, 0.0, 1.0, width, refinedPoints, wallLayers, true, 3.5, generator);
        AppendGradedAxialShells(points, 4.0, 0.0, 1.0, width, refinedPoints, wallLayers, false, 3.5, generator);
    }
    else
    {
        StructuredShellSpec spec{width, wallTangential, wallLayers, wallThinLayers, wallGrowth};
        // Only background points exist here. They must also keep clear of every interface. Lattice sites are
        // wallTangential apart, so a random point sitting mid-gap within wallTangential/2 of
        // the interface is closer to the channel (or wall) volume in front of it than any
        // slab point is: on the thick side it owns a fat wall cell that reaches through the
        // thin slabs into the channel, on the thin side a channel cell that reaches tens of
        // mean free paths into the wall. With 20000 background points this produced ~70 such
        // cells along the inlet pipe alone (1% of its volume). Excluding a band of
        // 0.75 wallTangential on both sides costs a few hundred background points.
        const double backgroundExclusion =
            std::max(spec.StackThickness(wallThinLayers) + 0.25 * wallTangential, 0.75 * wallTangential);
        std::size_t kept = 0;
        for(const Vector3D &point : points)
        {
            double radius = std::sqrt(point.y * point.y + point.z * point.z);
            if(!NearInterface(point.x, radius, backgroundExclusion))
            {
                points[kept++] = point;
            }
        }
        points.resize(kept);
        AppendStructuredRadialShells(points, 0.5, 0.0, 2.5, spec, true, generator);
        AppendStructuredRadialShells(points, 0.5, 4.5, 7.0, spec, true, generator);
        AppendStructuredRadialShells(points, 1.5, 2.5, 4.5, spec, true, generator);
        AppendStructuredRadialShells(points, 1.0, 3.0, 4.0, spec, false, generator);
        AppendStructuredAxialShells(points, 2.5, 0.5, 1.5, spec, false, generator);
        AppendStructuredAxialShells(points, 4.5, 0.5, 1.5, spec, true, generator);
        AppendStructuredAxialShells(points, 3.0, 0.0, 1.0, spec, true, generator);
        AppendStructuredAxialShells(points, 4.0, 0.0, 1.0, spec, false, generator);
        // Thin-side slab stack plus a quarter lattice spacing of clearance.
        channelExclusion = spec.StackThickness(wallThinLayers) + 0.25 * wallTangential;
    }
    if(channelPoints > 0)
    {
        AppendPoints(points, RandomThinRegion(channelPoints, channelExclusion, generator));
    }
    return points;
}

bool IsThick(const Vector3D &point)
{
    double radius = std::sqrt(point.y * point.y + point.z * point.z);
    if(radius > 1.5)
    {
        return true;
    }
    if(radius < 0.5)
    {
        return point.x > 3.0 and point.x < 4.0;
    }
    if(point.x < 2.5 or point.x > 4.5)
    {
        return true;
    }
    return point.x > 3.0 and point.x < 4.0 and radius < 1.0;
}

void WriteProbeHeader(const std::string &path)
{
    std::filesystem::path outputPath(path);
    if(!outputPath.parent_path().empty())
    {
        std::filesystem::create_directories(outputPath.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    output << "# Crooked-pipe probe temperatures, Steinberg & Heizler 2022 Fig. 8\n";
    output << "# Probes (r, z_axis=x): P1 (0, 0.25), P2 (0, 2.75), P3 (1.25, 3.5), P4 (0, 4.25), P5 (0, 6.75) cm\n";
    output << "# t_ns, cycle, T1_keV, T2_keV, T3_keV, T4_keV, T5_keV\n";
}

// Reading one Voronoi cell makes the probe as noisy as the mesh is fine: the thin
// channel has c_v = 1e13, a thousandth of the wall, so a single small cell swings by
// ~0.09 keV per cycle between absorption and emission, and refining the mesh shrinks
// both the packet count and the heat capacity of the cell being read. The problem is
// axisymmetric, so averaging over the ring within probeRadius of each (r, z) location
// costs no physics, samples far more cells, and -- because the averaging volume is
// fixed in cm -- keeps the diagnostic independent of resolution. Only thin cells
// contribute, so a ring that grazes an interface cannot mix in wall temperatures.
// probeRadius = 0 restores the original nearest-cell behaviour.
void AppendProbes(const Grid &grid, const std::vector<STORM::RadiationCell> &cells,
                  const std::vector<int> &materialFlags, double time, std::size_t step,
                  const std::string &path, int rank, double probeRadius)
{
    double temperatures[probeCount];
    std::vector<double> weightedSum(probeCount, 0.0), weightSum(probeCount, 0.0);
    const double radiusSquared = probeRadius * probeRadius;
    struct DistanceRank
    {
        double distance;
        int rank;
    };
    std::vector<DistanceRank> nearest(probeCount, {std::numeric_limits<double>::max(), rank});
    std::vector<double> nearestTemperature(probeCount, 0.0);

    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
    {
        if(materialFlags[i] != 1)
        {
            continue;
        }
        Vector3D center = grid.GetCellCM(i);
        double radius = std::sqrt(center.y * center.y + center.z * center.z);
        for(std::size_t probe = 0; probe < probeCount; ++probe)
        {
            double dx = center.x - probeX[probe];
            double dr = radius - probeR[probe];
            double distance = dx * dx + dr * dr;
            if(distance < nearest[probe].distance)
            {
                nearest[probe].distance = distance;
                nearestTemperature[probe] = cells[i].temperature;
            }
            if(probeRadius > 0.0 and distance <= radiusSquared)
            {
                double volume = grid.GetVolume(i);
                weightedSum[probe] += cells[i].temperature * volume;
                weightSum[probe] += volume;
            }
        }
    }

    if(probeRadius > 0.0)
    {
        MPI_Allreduce(MPI_IN_PLACE, weightedSum.data(), static_cast<int>(probeCount), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, weightSum.data(), static_cast<int>(probeCount), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    }
    for(std::size_t probe = 0; probe < probeCount; ++probe)
    {
        if(probeRadius > 0.0 and weightSum[probe] > 0.0)
        {
            // Volume-weighted mean over the ring; no nearest-cell reduction needed.
            temperatures[probe] = weightedSum[probe] / weightSum[probe];
            continue;
        }
        DistanceRank global;
        MPI_Allreduce(&nearest[probe], &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        if(rank == global.rank)
        {
            temperatures[probe] = nearestTemperature[probe];
        }
        MPI_Bcast(&temperatures[probe], 1, MPI_DOUBLE, global.rank, MPI_COMM_WORLD);
    }
    if(rank == 0)
    {
        std::ofstream output(path, std::ios::app);
        output << time * 1.0e9 << ", " << step;
        for(std::size_t probe = 0; probe < probeCount; ++probe)
        {
            output << ", " << temperatures[probe] / units::kev_kelvin;
        }
        output << '\n';
    }
}

void FillSnapshotFields(const Grid &grid, const std::vector<STORM::RadiationCell> &cells,
                        const std::vector<STORM::SimpleExtensives> &extensives,
                        const std::vector<int> &materialFlags,
                        std::vector<std::string> &names, std::vector<std::vector<double>> &values)
{
    const std::size_t cellCount = grid.GetPointNo();
    names = {"Temperature_K", "Temperature_keV", "Erad", "Density", "Material", "Volume"};
    values.assign(names.size(), std::vector<double>(cellCount));
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        const double volume = grid.GetVolume(i);
        values[0][i] = cells[i].temperature;
        values[1][i] = cells[i].temperature / units::kev_kelvin;
        values[2][i] = cells[i].Erad;
        values[3][i] = (volume > 0.0) ? extensives[i].mass / volume : 0.0;
        values[4][i] = static_cast<double>(materialFlags[i]);
        values[5][i] = volume;
    }
}

void WriteSpatialProfile(const Grid &grid, const std::vector<STORM::RadiationCell> &cells,
                         const std::vector<STORM::SimpleExtensives> &extensives,
                         const std::vector<int> &materialFlags, double time, std::size_t step,
                         const std::string &path, int rank)
{
    std::vector<double> localRows;
    localRows.reserve(9 * grid.GetPointNo());
    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
    {
        const Vector3D center = grid.GetCellCM(i);
        const double volume = grid.GetVolume(i);
        localRows.push_back(center.x);
        localRows.push_back(center.y);
        localRows.push_back(center.z);
        localRows.push_back(std::sqrt(center.y * center.y + center.z * center.z));
        localRows.push_back(cells[i].temperature);
        localRows.push_back(cells[i].Erad);
        localRows.push_back((volume > 0.0) ? extensives[i].mass / volume : 0.0);
        localRows.push_back(static_cast<double>(materialFlags[i]));
        localRows.push_back(volume);
    }
    const std::size_t fieldsPerCell = 9;
    std::vector<double> rows = MPI_Gatherv_serializable(localRows, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::filesystem::path outputPath(path);
        if(!outputPath.parent_path().empty())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        std::ofstream output(path, std::ios::trunc);
        output << "# Crooked-pipe spatial profile\n";
        output << "# t_s=" << time << " t_ns=" << time * 1.0e9 << " cycle=" << step << "\n";
        output << "# x_cm, y_cm, z_cm, r_cm, T_K, T_keV, Erad, density, material, volume_cm3\n";
        const std::size_t cellCount = rows.size() / fieldsPerCell;
        for(std::size_t cell = 0; cell < cellCount; ++cell)
        {
            const std::size_t offset = cell * fieldsPerCell;
            output << rows[offset] << ", " << rows[offset + 1] << ", " << rows[offset + 2] << ", "
                   << rows[offset + 3] << ", " << rows[offset + 4] << ", "
                   << rows[offset + 4] / units::kev_kelvin << ", " << rows[offset + 5] << ", "
                   << rows[offset + 6] << ", " << rows[offset + 7] << ", " << rows[offset + 8] << '\n';
        }
        std::cout << "Wrote spatial profile " << path << " (" << cellCount << " cells)" << std::endl;
    }
}

void WriteProductionSnapshots(const Grid &grid, const std::vector<STORM::RadiationCell> &cells,
                              const std::vector<STORM::SimpleExtensives> &extensives,
                              const std::vector<int> &materialFlags, double time, std::size_t step,
                              const std::string &outputDir, std::size_t snapshotInterval,
                              std::size_t &lastSnapshotStep, bool force, int rank)
{
    if(outputDir.empty())
    {
        return;
    }
    if(!force)
    {
        if(snapshotInterval == 0 || step == 0 || (step % snapshotInterval) != 0)
        {
            return;
        }
    }
    if(lastSnapshotStep == step)
    {
        return;
    }
    lastSnapshotStep = step;

    std::vector<std::string> names;
    std::vector<std::vector<double>> values;
    FillSnapshotFields(grid, cells, extensives, materialFlags, names, values);

#ifdef MADVORO_WITH_VTK
    {
        std::filesystem::path vtkDir = std::filesystem::path(outputDir) / "vtk";
        if(rank == 0)
        {
            std::filesystem::create_directories(vtkDir);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        char fileName[64];
        std::snprintf(fileName, sizeof(fileName), "crooked_pipe_%05zu.pvtu", step);
        std::filesystem::path vtkPath = vtkDir / fileName;
        std::vector<Vector3D> coordinates(grid.GetPointNo());
        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            coordinates[i] = grid.GetMeshPoint(i);
        }
        MadVoro::IO::write_vtu3d::write_vtu_3d(vtkPath, names, values, {"Coordinates"}, {coordinates},
                                               time, step, grid);
        if(rank == 0)
        {
            std::cout << "Wrote VTK snapshot " << vtkPath.string() << std::endl;
        }
    }
#else
    if(force and rank == 0)
    {
        std::cerr << "Warning: VTK snapshots requested but STORM was not built with STORM_WITH_VTK." << std::endl;
    }
#endif

#ifdef MADVORO_WITH_HDF5
    {
        const std::string hdf5Path = (std::filesystem::path(outputDir) / "latest.h5").string();
#ifdef MADVORO_WITH_MPI
        MadVoro::IO::WriteVoronoiHDF5_Parallel(grid, hdf5Path, values, names, false);
#else
        MadVoro::IO::WriteVoronoiHDF5(grid, hdf5Path, values, names, false);
#endif
        if(rank == 0)
        {
            std::cout << "Wrote HDF5 snapshot " << hdf5Path << std::endl;
        }
    }
#else
    if(force and rank == 0)
    {
        std::cerr << "Warning: latest.h5 requested but STORM was not built with STORM_WITH_HDF5." << std::endl;
    }
#endif
}

// The thin channel volume that the mesh actually resolves is the clearest measure of
// how much of the pipe is lost to cells whose centre of mass falls in the thick wall.
void ReportMeshQuality(const Grid &grid, const std::vector<int> &materialFlags, int rank)
{
    const double analyticThinVolume = std::acos(-1.0) * (0.25 * 2.5 + 1.5 * 1.5 * 2.0 - 1.0 * 1.0 + 0.25 * 2.5);
    double localThinVolume = 0.0;
    std::size_t localThinCells = 0;
    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
    {
        if(materialFlags[i] == 1)
        {
            localThinVolume += grid.GetVolume(i);
            ++localThinCells;
        }
    }
    double thinVolume = 0.0;
    std::size_t thinCells = 0;
    MPI_Allreduce(&localThinVolume, &thinVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localThinCells, &thinCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    for(std::size_t probe = 0; probe < probeCount; ++probe)
    {
        struct DistanceRank
        {
            double distance;
            int rank;
        };
        DistanceRank local = {std::numeric_limits<double>::max(), rank};
        double localData[4] = {0.0, 0.0, 0.0, 0.0};
        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            Vector3D center = grid.GetCellCM(i);
            double radius = std::sqrt(center.y * center.y + center.z * center.z);
            double dx = center.x - probeX[probe];
            double dr = radius - probeR[probe];
            double distance = dx * dx + dr * dr;
            if(distance < local.distance)
            {
                local.distance = distance;
                localData[0] = center.x;
                localData[1] = radius;
                localData[2] = static_cast<double>(materialFlags[i]);
                localData[3] = grid.GetVolume(i);
            }
        }
        DistanceRank global;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        MPI_Bcast(localData, 4, MPI_DOUBLE, global.rank, MPI_COMM_WORLD);
        if(rank == 0)
        {
            std::cout << "Probe " << probe + 1 << " target (r=" << probeR[probe] << ", z=" << probeX[probe]
                      << ") cell (r=" << localData[1] << ", z=" << localData[0] << "), material="
                      << (localData[2] > 0.5 ? "thin" : "thick") << ", volume=" << localData[3]
                      << ", cell size=" << std::cbrt(localData[3]) << std::endl;
        }
    }
    if(rank == 0)
    {
        std::cout << "Thin channel: " << thinCells << " cells, meshed volume=" << thinVolume
                  << " cm^3, analytic=" << analyticThinVolume << " cm^3 ("
                  << 100.0 * thinVolume / analyticThinVolume << "%)" << std::endl;
    }
}

template<typename Physics>
bool Rebalance(Grid &grid, STORM::MonteCarloManager<Vector3D, Grid, Physics> &manager,
               std::vector<STORM::RadiationCell> &cells, std::vector<STORM::SimpleExtensives> &extensives,
               std::vector<int> &materialFlags, double transportWorkScale,
               double thickOpacity, double thinOpacity)
{
    std::size_t cellCount = grid.GetPointNo();
    std::vector<double> weights(cellCount, 50.0);

    // TEMPORARY EXPERIMENT (CP_BRANSON_WEIGHTS=1): partition the way Branson does.
    // Branson feeds METIS a static per-cell vertex weight of max(1, (int)(ln sigma_a)^3)
    // computed once from the initial material (branson/src/decompose_mesh.h:135) and never
    // re-partitions.  Here that is 439 for the thick wall and 1 for the thin channel.
    // Note this is the opposite of what this problem wants: the wall absorbs in ~1 mfp
    // while channel packets cross many cells per cycle, so the expensive cells are the
    // transparent ones.  Kept behind an environment flag; default path is unchanged.
    static const bool bransonWeights = [] {
        const char *value = std::getenv("CP_BRANSON_WEIGHTS");
        return value != nullptr && std::string(value) != "0" && std::string(value) != "false";
    }();
    if(bransonWeights)
    {
        for(std::size_t i = 0; i < cellCount; ++i)
        {
            double const sigma = (materialFlags[i] == 0) ? thickOpacity : thinOpacity;
            weights[i] = std::max(1.0, static_cast<double>(static_cast<int>(std::pow(std::log(sigma), 3))));
        }
        if(!grid.ShouldRebalance(weights))
        {
            return false;
        }
        grid.Rebalance(weights);
        STORM::MPI_exchange_data(grid, cells, false);
        STORM::MPI_exchange_data(grid, extensives, false);
        STORM::MPI_exchange_data(grid, materialFlags, false);
        STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
        STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);
        STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, manager.getParticles());
        std::size_t const newCellCount = grid.GetPointNo();
        cells.resize(newCellCount);
        extensives.resize(newCellCount);
        materialFlags.resize(newCellCount);
        return true;
    }
    const std::vector<std::size_t> &stepCounters = manager.GetCellsStepsCounters();
    const std::vector<std::size_t> &particleCounts = manager.GetBeginningParticleCount();
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        if(stepCounters.size() == cellCount)
        {
            weights[i] += transportWorkScale * static_cast<double>(stepCounters[i]);
        }
        if(particleCounts.size() == cellCount)
        {
            weights[i] += 10.0 * static_cast<double>(particleCounts[i]);
        }
    }
    if(!grid.ShouldRebalance(weights))
    {
        return false;
    }

    grid.Rebalance(weights);
    STORM::MPI_exchange_data(grid, cells, false);
    STORM::MPI_exchange_data(grid, extensives, false);
    STORM::MPI_exchange_data(grid, materialFlags, false);
    STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
    STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);
    STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, manager.getParticles());

    std::size_t newCellCount = grid.GetPointNo();
    cells.resize(newCellCount);
    extensives.resize(newCellCount);
    materialFlags.resize(newCellCount);
    manager.GetCellsStepsCounters().resize(newCellCount, 0);
    manager.GetBeginningParticleCount().resize(newCellCount, 0);
    return true;
}

} // namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    try
    {
        bool showHelp = false;
        Options options = ParseOptions(argc, argv, showHelp);
        if(showHelp)
        {
            if(rank == 0)
            {
                PrintUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        }

        {
            Vector3D lower(0.0, -2.0, -2.0);
            Vector3D upper(7.0, 2.0, 2.0);
            std::vector<Vector3D> allPoints;
            if(rank == 0)
            {
                allPoints = GeneratePoints(options.backgroundPoints, options.wallLayers, options.wallPoints,
                                           options.wallLayerWidth, options.channelPoints, options.wallThinLayers,
                                           options.wallTangential, options.wallGrowth, options.randomWalls,
                                           lower, upper);
                std::cout << "Generated " << allPoints.size() << " crooked-pipe mesh points" << std::endl;
            }
            std::vector<Vector3D> localPoints = MPI_Spread(allPoints, 0, MPI_COMM_WORLD);
            Grid grid(lower, upper);
            grid.BuildParallel(localPoints);

            std::size_t localCellCount = grid.GetPointNo();
            std::size_t globalCellCount = 0;
            MPI_Allreduce(&localCellCount, &globalCellCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            const double initialTemperature = 0.05 * units::kev_kelvin;
            const double driveTemperature = 0.5 * units::kev_kelvin;
            const double thickDensity = 1.0;
            const double thinDensity = 2.0;
            const double thickCvPerVolume = options.thickCvPerVolume / units::kev_kelvin;
            const double thinCvPerVolume = options.thinCvPerVolume / units::kev_kelvin;
            std::vector<STORM::RadiationCell> cells(localCellCount);
            std::vector<STORM::SimpleExtensives> extensives(localCellCount);
            std::vector<int> materialFlags(localCellCount);
            for(std::size_t i = 0; i < localCellCount; ++i)
            {
                // Flag from the generating point, not the cell centre of mass. A first-layer
                // wall point sits 0.01 cm from the interface while its thin-side neighbours are
                // an order of magnitude coarser, so its Voronoi cell bulges into the channel and
                // its centre of mass lands on the wrong side: with --wall-width 0.01 that
                // mislabelled 256,010 of 1,220,000 cells as thin, giving the Marshak layer
                // kappa = 0.2 instead of 2000 and c_v = 1e13 instead of 1e16.
                bool thick = IsThick(grid.GetMeshPoint(i));
                materialFlags[i] = thick ? 0 : 1;
                double density = thick ? thickDensity : thinDensity;
                double cvPerVolume = thick ? thickCvPerVolume : thinCvPerVolume;
                double volume = grid.GetVolume(i);
                cells[i].temperature = initialTemperature;
                cells[i].internalEnergy = cvPerVolume * initialTemperature * volume;
                cells[i].Erad = units::arad * std::pow(initialTemperature, 4) / density;
                extensives[i].mass = density * volume;
                extensives[i].internal_energy = cells[i].internalEnergy;
            }

            using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
                                            STORM::examples::CrookedPipeEOS, 1>;
            STORM::RadiationIMCParameters<1> parameters;
            parameters.newPhotonsPerCell = options.newPhotonsPerCell;
            parameters.emissionFloorPhotonsPerCell = options.emissionFloorPerCell;
            parameters.fleckArgumentScale = options.fleckScale;
            parameters.withHydro = false;
            parameters.withMultigroupOpacity = false;
            parameters.withRandomWalk = options.withRandomWalk;
            parameters.rwMinCellOpticalDepth = 25.0;
            parameters.withDDMC = options.ddmc;
            parameters.ddmcMinCellOpticalDepth = options.ddmcMinTau;
            parameters.energyBoundaries = {0.0, 1.0e30};
            parameters.energyBoundariesProvided = true;

            std::shared_ptr<STORM::examples::CrookedPipeEOS> eos =
                std::make_shared<STORM::examples::CrookedPipeEOS>(thickCvPerVolume, thinCvPerVolume, thickDensity, thinDensity);
            std::shared_ptr<STORM::examples::CrookedPipeOpacity<Vector3D, Grid>> opacity =
                std::make_shared<STORM::examples::CrookedPipeOpacity<Vector3D, Grid>>(materialFlags, cells,
                                                                                      options.thickOpacity, options.thinOpacity);
            std::shared_ptr<STORM::examples::CrookedPipeBoundary<Vector3D, Grid>> boundary =
                std::make_shared<STORM::examples::CrookedPipeBoundary<Vector3D, Grid>>(
                    grid, materialFlags, driveTemperature, options.boundaryPhotonsPerFace, options.reflectSource,
                    options.reflectExit, options.sourceScale);
            std::shared_ptr<IMC> physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacity, parameters);
            std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> populationControl =
                std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(
                    grid, options.minPhotonsPerCell, 4.0, options.censusFloorPerCell, options.censusCapPerCell);
            STORM::MonteCarloConfig transportConfig = options.transportConfig;
#ifdef STORM_WITH_GPU
            if(options.transportInnerSteps != 0)
                transportConfig.gpuMaxInnerSteps = options.transportInnerSteps;
            if(options.transportMinLaunch != std::numeric_limits<std::size_t>::max())
                transportConfig.gpuMinLaunchSize = options.transportMinLaunch;
            if(options.transportHoldSkips != 0)
                transportConfig.gpuHoldMaxSkips = options.transportHoldSkips;
#else
            if(options.transportInnerSteps != 0 || options.transportHoldSkips != 0 ||
               options.transportMinLaunch != std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("Transport wave/batch options require a Kokkos build");
#endif
            STORM::MonteCarloManager<Vector3D, Grid, IMC> manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(
                grid, physics, populationControl, boundary, options.managerType, options.rdmaEngine, transportConfig);

            if(!options.outputProbes.empty() and rank == 0)
            {
                WriteProbeHeader(options.outputProbes);
            }
            if(!options.outputDir.empty() and rank == 0)
            {
                std::filesystem::create_directories(options.outputDir);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(!options.outputProbes.empty())
            {
                AppendProbes(grid, cells, materialFlags, 0.0, 0, options.outputProbes, rank,
                             options.probeRadius);
            }

            if(rank == 0)
            {
                std::cout << "Crooked pipe: " << globalCellCount << " cells on " << processCount
                          << " ranks, new/min photons=" << options.newPhotonsPerCell << "/"
                          << options.minPhotonsPerCell
                          << ", census floor/cap="
                          << (options.censusFloorPerCell > 0 ? options.censusFloorPerCell : options.minPhotonsPerCell)
                          << "/"
                          << (options.censusCapPerCell > 0 ? options.censusCapPerCell : options.minPhotonsPerCell * 20)
                          << ", emission floor="
                          << (options.emissionFloorPerCell > 0 ? options.emissionFloorPerCell : options.newPhotonsPerCell)
                          << ", channel points=" << options.channelPoints
                          << ", walls=" << (options.randomWalls ? "random" : "structured")
                          << " width=" << options.wallLayerWidth << " thick/thin layers=" << options.wallLayers
                          << "/" << options.wallThinLayers << " tangential=" << options.wallTangential
                          << " growth=" << options.wallGrowth
                          << ", ddmc=" << (options.ddmc ? "on" : "off") << " min tau=" << options.ddmcMinTau
                          << ", cv thick/thin=" << options.thickCvPerVolume << "/" << options.thinCvPerVolume
                          << ", opacity thick/thin=" << options.thickOpacity << "/" << options.thinOpacity
                          << ", reflect source/exit=" << options.reflectSource << "/" << options.reflectExit
                          << ", source scale=" << options.sourceScale << ", fleck scale=" << options.fleckScale
                          << ", dt growth=" << options.dtGrowth
                          << ", probe radius=" << options.probeRadius << " cm"
                          << ", random walk=" << options.withRandomWalk
                          << ", max dt=" << options.maximumDt << " s"
                          << ", final time=" << options.finalTime << " s";
                if(!options.outputDir.empty())
                {
                    std::cout << ", output dir=" << options.outputDir
                              << ", snapshot interval=" << options.snapshotInterval;
                }
                std::cout << std::endl;
            }
            ReportMeshQuality(grid, materialFlags, rank);

            double ledgerInitialMaterial = 0.0;
            {
                double localMaterial = 0.0;
                for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
                {
                    localMaterial += cells[i].internalEnergy;
                }
                MPI_Allreduce(&localMaterial, &ledgerInitialMaterial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            double time = 0.0;
            double dt = std::min(options.initialDt, options.maximumDt);
            const double maximumDt = options.maximumDt;
            std::size_t step = 0;
            std::size_t lastSnapshotStep = std::numeric_limits<std::size_t>::max();
            std::chrono::high_resolution_clock::time_point wallStart = std::chrono::high_resolution_clock::now();
            double rebalanceSeconds = 0.0, probeSeconds = 0.0;
            while(time < options.finalTime and step < options.maxSteps)
            {
                if(step > 0 and (step <= 2 or step % options.rebalanceInterval == 0))
                {
                    const auto balanceStart = std::chrono::high_resolution_clock::now();
                    if(Rebalance(grid, manager, cells, extensives, materialFlags, options.transportWorkScale,
                                 options.thickOpacity, options.thinOpacity) and rank == 0)
                    {
                        std::cout << "Rebalanced at cycle " << step << std::endl;
                    }
                    rebalanceSeconds += std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - balanceStart).count();
                }

                double stepDt = std::min(dt, options.finalTime - time);
                std::chrono::high_resolution_clock::time_point stepStart = std::chrono::high_resolution_clock::now();
                manager.step(stepDt);
                double stepSeconds = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - stepStart).count();
                if(options.energyLedger)
                {
                    double local[8] = {0.0};
                    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
                    {
                        local[0] += cells[i].internalEnergy;
                    }
                    for(const auto &packet : manager.getParticles())
                    {
                        local[1] += packet.weight;
                    }
                    const auto &ledger = boundary->ledger();
                    local[2] = ledger.injected;
                    local[3] = ledger.sourceDisc;
                    local[4] = ledger.exitDisc;
                    local[5] = ledger.sideWall;
                    local[6] = ledger.interior;
                    local[7] = static_cast<double>(ledger.interiorCount);
                    double global[8] = {0.0};
                    MPI_Reduce(local, global, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    if(rank == 0)
                    {
                        const double stored = global[0] - ledgerInitialMaterial + global[1];
                        const double removed = global[3] + global[4] + global[5] + global[6];
                        std::cout << "Ledger cycle " << step << ": injected=" << global[2] << " removed(source disc/exit/side/interior)="
                                  << global[3] << "/" << global[4] << "/" << global[5] << "/" << global[6]
                                  << " interior_count=" << static_cast<std::size_t>(global[7])
                                  << " dE_material=" << (global[0] - ledgerInitialMaterial) << " census=" << global[1]
                                  << " balance(inj-rem-stored)/inj=" << (global[2] - removed - stored) / std::max(global[2], 1.0e-300)
                                  << std::endl;
                    }
                }
                if(options.ddmc and (step % 10 == 0))
                {
                    // The physics counters are per rank; report the global totals.
                    unsigned long long ddmcCounters[2] = {physics->getDDMCStepCount(), physics->getDDMCLeakCount()};
                    MPI_Allreduce(MPI_IN_PLACE, ddmcCounters, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                    if(rank == 0)
                    {
                        std::cout << "DDMC counters after cycle " << step << " (all ranks): steps=" << ddmcCounters[0]
                                  << " leaks=" << ddmcCounters[1] << std::endl;
                    }
                }
                time += stepDt;
                ++step;
                dt = std::min(options.dtGrowth * dt, maximumDt);

                if(!options.outputProbes.empty())
                {
                    const auto probeStart = std::chrono::high_resolution_clock::now();
                    AppendProbes(grid, cells, materialFlags, time, step, options.outputProbes, rank,
                                 options.probeRadius);
                    probeSeconds += std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - probeStart).count();
                }
                WriteProductionSnapshots(grid, cells, extensives, materialFlags, time, step,
                                         options.outputDir, options.snapshotInterval, lastSnapshotStep,
                                         false, rank);
                // GetParticleCount() is this rank's census only; the cycle line reports the
                // global census so it matches the manager's active_after_prestep count.
                unsigned long long censusParticles = manager.GetParticleCount();
                MPI_Allreduce(MPI_IN_PLACE, &censusParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                              MPI_COMM_WORLD);
                if(rank == 0)
                {
                    std::cout << "Cycle " << step << ", t=" << time << " s, dt=" << stepDt
                              << " s, census_particles=" << censusParticles
                              << ", wall=" << stepSeconds << " s" << std::endl;
                }
            }

            WriteProductionSnapshots(grid, cells, extensives, materialFlags, time, step,
                                     options.outputDir, options.snapshotInterval, lastSnapshotStep,
                                     true, rank);
            if(!options.outputDir.empty())
            {
                const std::string profilePath =
                    (std::filesystem::path(options.outputDir) / "crooked_pipe_profile.csv").string();
                WriteSpatialProfile(grid, cells, extensives, materialFlags, time, step, profilePath, rank);
            }

            if(rank == 0)
            {
                double wallSeconds = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wallStart).count();
                std::cout << "Outside-cycle time on rank 0: rebalance=" << rebalanceSeconds
                          << " s, probes=" << probeSeconds << " s" << std::endl;
                std::cout << "Finished at t=" << time << " s after " << step
                          << " cycles in " << wallSeconds << " s" << std::endl;
            }
        }
    }
    catch(const std::exception &error)
    {
        std::cerr << "Crooked pipe failure on rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Release fabric and device resources while MPI is still available.
    RMAFactory::Finalize(RDMA_Type::AUTO_RDMA);
#ifdef STORM_WITH_GPU
    STORM::gpu::KokkosRuntime::Finalize();
#endif
    MPI_Finalize();
    return 0;
}
