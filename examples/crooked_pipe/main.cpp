#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
    std::size_t boundaryPhotonsPerFace = 100;
    std::size_t maxSteps = std::numeric_limits<std::size_t>::max();
    double finalTime = 1.0e-6;
    double maximumDt = 1.0e-9;
    bool withRandomWalk = true;
    std::string outputProbes;
    STORM::ManagerType managerType = STORM::ManagerType::Auto;
    STORM::RDMAEngine rdmaEngine = STORM::RDMAEngine::OFI;
};

void PrintUsage(const char *program)
{
    std::cerr << "Usage: mpirun -np <N> " << program << " [points] [new_photons] [min_photons] [options]\n"
              << "  --points <N>                 Background mesh points (default: 20000)\n"
              << "  --wall-layers <N>            Graded refinement layers on the thick side of each interface (default: 4)\n"
              << "  --wall-width <cm>            Width of the first refinement layer (default: 0.01)\n"
              << "  --wall-points <N>            Points in the first refinement layer of each interface (default: 4 x points)\n"
              << "  --new-photons <N>            Thermal packets emitted per cell per step (default: 25)\n"
              << "  --min-photons <N>            Population-control target per cell (default: 100)\n"
              << "  --output-probes <file>        Write five probe temperature histories\n"
              << "  --final-time <seconds>        End time (default: 1e-6)\n"
              << "  --max-dt <seconds>            Largest time step (default: 1e-9)\n"
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
        else if(argument == "--wall-points")
        {
            options.wallPoints = std::stoull(RequireValue(argc, argv, i));
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
        else if(argument == "--boundary-photons")
        {
            options.boundaryPhotonsPerFace = std::stoull(RequireValue(argc, argv, i));
        }
        else if(argument == "--output-probes")
        {
            options.outputProbes = RequireValue(argc, argv, i);
        }
        else if(argument == "--final-time")
        {
            options.finalTime = std::stod(RequireValue(argc, argv, i));
        }
        else if(argument == "--max-dt")
        {
            options.maximumDt = std::stod(RequireValue(argc, argv, i));
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

std::vector<Vector3D> GeneratePoints(std::size_t backgroundPoints, std::size_t wallLayers, std::size_t wallPoints,
                                     double width, const Vector3D &lower, const Vector3D &upper)
{
    boost::mt19937_64 generator(42);
    std::vector<Vector3D> points = RandomBox(backgroundPoints, lower, upper, generator);
    std::size_t refinedPoints = wallPoints > 0 ? wallPoints : 4 * backgroundPoints;
    AppendGradedRadialShells(points, 0.5, 0.0, 2.5, width, refinedPoints, wallLayers, true, generator);
    AppendGradedRadialShells(points, 0.5, 4.5, 7.0, width, refinedPoints, wallLayers, true, generator);
    AppendGradedRadialShells(points, 1.5, 2.5, 4.5, width, refinedPoints, wallLayers, true, generator);
    AppendGradedRadialShells(points, 1.0, 2.5, 4.5, width, refinedPoints, wallLayers, false, generator);
    AppendGradedAxialShells(points, 2.5, 0.5, 1.5, width, refinedPoints, wallLayers, false, 0.0, generator);
    AppendGradedAxialShells(points, 4.5, 0.5, 1.5, width, refinedPoints, wallLayers, true, 7.0, generator);
    AppendGradedAxialShells(points, 3.0, 0.0, 1.0, width, refinedPoints, wallLayers, true, 3.5, generator);
    AppendGradedAxialShells(points, 4.0, 0.0, 1.0, width, refinedPoints, wallLayers, false, 3.5, generator);
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

void AppendProbes(const Grid &grid, const std::vector<STORM::RadiationCell> &cells, double time, std::size_t step,
                  const std::string &path, int rank)
{
    double temperatures[probeCount];
    for(std::size_t probe = 0; probe < probeCount; ++probe)
    {
        struct DistanceRank
        {
            double distance;
            int rank;
        };
        DistanceRank local = {std::numeric_limits<double>::max(), rank};
        double localTemperature = 0.0;
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
                localTemperature = cells[i].temperature;
            }
        }
        DistanceRank global;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        if(rank == global.rank)
        {
            temperatures[probe] = localTemperature;
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

bool Rebalance(Grid &grid, STORM::MonteCarloManager<Vector3D, Grid> &manager,
               std::vector<STORM::RadiationCell> &cells, std::vector<STORM::SimpleExtensives> &extensives,
               std::vector<int> &materialFlags, std::vector<Particle> &particles)
{
    std::size_t cellCount = grid.GetPointNo();
    std::vector<double> weights(cellCount, 50.0);
    const std::vector<std::size_t> &stepCounters = manager.GetCellsStepsCounters();
    const std::vector<std::size_t> &particleCounts = manager.GetBeginningParticleCount();
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        if(stepCounters.size() == cellCount)
        {
            weights[i] += static_cast<double>(stepCounters[i]);
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
    STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, particles);

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
                                           options.wallLayerWidth, lower, upper);
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
            const double thickCvPerVolume = 1.0e16 / units::kev_kelvin;
            const double thinCvPerVolume = 1.0e13 / units::kev_kelvin;
            std::vector<STORM::RadiationCell> cells(localCellCount);
            std::vector<STORM::SimpleExtensives> extensives(localCellCount);
            std::vector<int> materialFlags(localCellCount);
            for(std::size_t i = 0; i < localCellCount; ++i)
            {
                bool thick = IsThick(grid.GetCellCM(i));
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
            parameters.withHydro = false;
            parameters.withMultigroupOpacity = false;
            parameters.withRandomWalk = options.withRandomWalk;
            parameters.rwMinCellOpticalDepth = 25.0;
            parameters.energyBoundaries = {0.0, 1.0e30};
            parameters.energyBoundariesProvided = true;

            std::shared_ptr<STORM::examples::CrookedPipeEOS> eos =
                std::make_shared<STORM::examples::CrookedPipeEOS>(thickCvPerVolume, thinCvPerVolume, thickDensity, thinDensity);
            std::shared_ptr<STORM::examples::CrookedPipeOpacity<Vector3D, Grid>> opacity =
                std::make_shared<STORM::examples::CrookedPipeOpacity<Vector3D, Grid>>(materialFlags, cells);
            std::shared_ptr<STORM::examples::CrookedPipeBoundary<Vector3D, Grid>> boundary =
                std::make_shared<STORM::examples::CrookedPipeBoundary<Vector3D, Grid>>(
                    grid, materialFlags, driveTemperature, options.boundaryPhotonsPerFace);
            std::shared_ptr<IMC> physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacity, parameters);
            std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> populationControl =
                std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, options.minPhotonsPerCell, 4.0);
            STORM::MonteCarloManager<Vector3D, Grid> manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(
                grid, physics, populationControl, boundary, options.managerType, options.rdmaEngine);
            std::vector<Particle> particles;

            if(!options.outputProbes.empty() and rank == 0)
            {
                WriteProbeHeader(options.outputProbes);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(!options.outputProbes.empty())
            {
                AppendProbes(grid, cells, 0.0, 0, options.outputProbes, rank);
            }

            if(rank == 0)
            {
                std::cout << "Crooked pipe: " << globalCellCount << " cells on " << processCount
                          << " ranks, new/min photons=" << options.newPhotonsPerCell << "/"
                          << options.minPhotonsPerCell << ", random walk=" << options.withRandomWalk
                          << ", max dt=" << options.maximumDt << " s"
                          << ", final time=" << options.finalTime << " s" << std::endl;
            }
            ReportMeshQuality(grid, materialFlags, rank);

            double time = 0.0;
            double dt = 1.0e-11;
            const double maximumDt = options.maximumDt;
            std::size_t step = 0;
            std::chrono::high_resolution_clock::time_point wallStart = std::chrono::high_resolution_clock::now();
            while(time < options.finalTime and step < options.maxSteps)
            {
                if(step > 0 and (step <= 2 or step % 10 == 0))
                {
                    if(Rebalance(grid, manager, cells, extensives, materialFlags, particles) and rank == 0)
                    {
                        std::cout << "Rebalanced at cycle " << step << std::endl;
                    }
                }

                double stepDt = std::min(dt, options.finalTime - time);
                std::chrono::high_resolution_clock::time_point stepStart = std::chrono::high_resolution_clock::now();
                particles = manager.step(std::move(particles), stepDt);
                double stepSeconds = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - stepStart).count();
                time += stepDt;
                ++step;
                dt = std::min(1.1 * dt, maximumDt);

                if(!options.outputProbes.empty())
                {
                    AppendProbes(grid, cells, time, step, options.outputProbes, rank);
                }
                if(rank == 0)
                {
                    std::cout << "Cycle " << step << ", t=" << time << " s, dt=" << stepDt
                              << " s, particles=" << particles.size() << ", wall=" << stepSeconds << " s" << std::endl;
                }
            }

            if(rank == 0)
            {
                double wallSeconds = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wallStart).count();
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

    MPI_Finalize();
    return 0;
}
