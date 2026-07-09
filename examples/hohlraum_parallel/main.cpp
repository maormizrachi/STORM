#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>
#include <numeric>
#include <filesystem>
#include <chrono>
#include <mpi.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <mpi_utils/mpi_collectives.hpp>
#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include "PhysicalConstants.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationCell.hpp"
#include "population/CombPopulationControl.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "HohlraumOpacity.hpp"
#include "HohlraumBoundary.hpp"
#include "HohlraumIMC.hpp"
#include "utils/MpiExchangeGrid.hpp"
#include "examples/MPI_ParticleDtype.hpp"
#include "mesh_movement/VoronoiMeshMovement.hpp"
#ifdef MADVORO_WITH_VTK
#include "MadVoro/io/vtk/write_vtu_3d.hpp"
#endif

#ifdef MADVORO_WITH_VTK
template <typename Grid>
static void DumpVTK(const Grid &grid, const std::vector<STORM::RadiationCell> &cells,
                    const std::vector<int> &materialFlags, const std::string &outputDir,
                    size_t printNumber, int rank)
{
    size_t Ncells = grid.GetPointNo();
    std::vector<std::vector<double>> cellScalars = {
        std::vector<double>(Ncells),
        std::vector<double>(Ncells),
        std::vector<double>(Ncells)
    };
    std::vector<std::string> scalarNames = {"Temperature_K", "Temperature_keV", "Material"};
    std::vector<Vector3D> coords(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        cellScalars[0][i] = cells[i].temperature;
        cellScalars[1][i] = cells[i].temperature / STORM::constants::kev_kelvin;
        cellScalars[2][i] = materialFlags[i];
        coords[i] = grid.GetMeshPoint(i);
    }
    scalarNames.push_back("Point Index");
    std::vector<double> pointIdx(Ncells);
    std::iota(pointIdx.begin(), pointIdx.end(), 0.0);
    cellScalars.push_back(pointIdx);

    std::filesystem::path vtkPath = std::filesystem::path(outputDir) / ("hohlraum_" + std::to_string(printNumber) + ".pvtu");
    MadVoro::IO::write_vtu3d::write_vtu_3d(vtkPath, scalarNames, cellScalars,
        {"Coordinates"}, {coords}, grid);
    if(rank == 0)
    {
        std::cout << "Wrote VTK to " << vtkPath.string() << std::endl;
    }
}
#endif

/*
 * Parallel (MPI) 2D Cylindrical Hohlraum benchmark.
 *
 * Same physics as examples/hohlraum (matching RICH/runs/Elad_paper_hohlraum)
 * but uses MadVoro's parallel Voronoi construction and STORM's
 * MonteCarloManager factory for distributed particle transport (default: RDMA
 * with IBV, falling back to MPI RMA).
 *
 * Usage:
 *   mpirun -np <N> ./hohlraum_parallel [N_base] [new_per_cell] [min_per_cell]
 *                                      [--output-profile <file>]
 *                                      [--output-vtk <dir>]
 */

static void PrintUsage(const char *progName)
{
    std::cerr << "Usage: mpirun -np <N> " << progName
              << " [N_base] [new_per_cell] [min_per_cell] [options]\n"
              << "\nPositional arguments:\n"
              << "  N_base           Number of base mesh generator points (default: 2000)\n"
              << "  new_per_cell     New photon packets emitted per cell per step (default: 5)\n"
              << "  min_per_cell     Target packets per cell after population control (default: 15)\n"
              << "\nOptions:\n"
              << "  --output-profile <file>   Write final temperature profile to file\n"
#ifdef MADVORO_WITH_VTK
              << "  --output-vtk <dir>        Write mesh + fields to .pvtu files in <dir> every "
              << "50 steps\n"
#endif
              << "  --help                    Show this message\n";
}

static bool IsMaterial(double x, double r)
{
    if(x >= 0.10 and x <= 0.15 and r <= 0.45)
        return true;
    if(x >= 0.55 and x <= 0.95 and r <= 0.45)
        return true;
    if(x >= 1.35 and x <= 1.40 and r <= 0.65)
        return true;
    if(x >= 0.10 and x <= 1.40 and r >= 0.60 and r <= 0.65)
        return true;
    return false;
}

using Grid = MadVoro::Voronoi3D<Vector3D>;
using ParticleT = STORM::Particle<Vector3D, Grid>;

static bool Rebalance(Grid &grid,
                      STORM::MonteCarloManager<Vector3D, Grid> &manager,
                      std::vector<STORM::RadiationCell> &cells,
                      std::vector<STORM::SimpleExtensives> &extensives,
                      std::vector<int> &materialFlags,
                      std::vector<ParticleT> &particles,
                      int rank)
{
    size_t N = grid.GetPointNo();

    std::vector<double> weights(N, 50.0);
    const std::vector<size_t> &counters = manager.GetCellsStepsCounters();
    const std::vector<size_t> &particleCounts = manager.GetBeginningParticleCount();

    bool useCounters = counters.size() == N;
    bool useParticleCounts = particleCounts.size() == N;

    if(!useCounters || !useParticleCounts)
    {
        if(rank == 0)
        {
            std::cout << "WARNING: Rebalance size mismatch: N=" << N
                      << " counters=" << counters.size()
                      << " particleCounts=" << particleCounts.size() << std::endl;
        }
    }

    constexpr double particleWeight = 10.0;
    constexpr double countersWeight = 1.0;

    for(size_t i = 0; i < N; i++)
    {
        if(useParticleCounts)
        {
            weights[i] += particleWeight * static_cast<double>(particleCounts[i]);
        }
        if(useCounters)
        {
            weights[i] += countersWeight * static_cast<double>(counters[i]);
        }
    }

    if(!grid.ShouldRebalance(weights))
    {
        if(rank == 0)
        {
            std::cout << "Load balanced, skipping rebalance" << std::endl;
        }
        return false;
    }

    if(rank == 0)
    {
        std::cout << "Rebalancing..." << std::endl;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    grid.Rebalance(weights);

    STORM::MPI_exchange_data(grid, cells, false);
    STORM::MPI_exchange_data(grid, extensives, false);
    STORM::MPI_exchange_data(grid, materialFlags, false);

    STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, particles);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
    if(rank == 0)
    {
        std::cout << "Rebalance done in " << elapsed << "s, new local cells: " << grid.GetPointNo() << std::endl;
    }

    return true;
}

static std::vector<Vector3D> RandCylinder(size_t pointNum, double Rin, double Rout, double xMin, double xMax, boost::mt19937_64 &gen)
{
    boost::random::uniform_real_distribution<> dist;
    std::vector<Vector3D> res;
    res.reserve(pointNum);
    for(size_t i = 0; i < pointNum; ++i)
    {
        double u = dist(gen);
        double v = dist(gen);
        double w = dist(gen);
        double r = std::sqrt(u * (Rout * Rout - Rin * Rin) + Rin * Rin);
        double theta = 2.0 * M_PI * v;
        double x = w * (xMax - xMin) + xMin;
        res.push_back(Vector3D(x, r * std::cos(theta), r * std::sin(theta)));
    }
    return res;
}

static std::vector<Vector3D> RandRectangular(size_t pointNum, Vector3D ll, Vector3D ur, boost::mt19937_64 &gen)
{
    boost::random::uniform_real_distribution<> dist;
    std::vector<Vector3D> res;
    res.reserve(pointNum);
    for(size_t i = 0; i < pointNum; ++i)
    {
        double x = ll.x + dist(gen) * (ur.x - ll.x);
        double y = ll.y + dist(gen) * (ur.y - ll.y);
        double z = ll.z + dist(gen) * (ur.z - ll.z);
        res.push_back(Vector3D(x, y, z));
    }
    return res;
}

static std::vector<Vector3D> GenerateAllPoints(size_t N_base, Vector3D lower, Vector3D upper)
{
    boost::mt19937_64 gen(42);
    std::vector<Vector3D> points = RandRectangular(N_base, lower, upper, gen);

    double dx = 0.02;
    size_t Np = 4 * N_base;

    std::vector<Vector3D> extra;
    extra = RandCylinder(Np, 0.45, 0.45 + dx, 0.10, 0.15, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0.45, 0.45 + dx, 0.55, 0.95, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(2 * Np, 0, 0.45, 0.55, 0.95, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0.60, 0.60 + dx, 0.10, 1.40, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0.65, 0.65 + dx, 0.10, 1.40, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.65, 0.10 - dx / 2, 0.10 + dx, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.45, 0.15 - dx, 0.15 + dx / 2, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.45, 0.55 - dx, 0.55 + dx / 2, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.45, 0.95 - dx / 2, 0.95 + dx, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.65, 1.35 - dx / 2, 1.35 + dx, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(Np, 0, 0.65, 1.40 - dx / 2, 1.40 + dx, gen);
    points.insert(points.end(), extra.begin(), extra.end());

    return points;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
                                    STORM::examples::HohlraumEOS, 1>;

    size_t N_base = 15000;
    size_t newPhotonsPerCell = 5;
    size_t minPhotonsPerCell = 15;
    std::string outputProfileFile;
    std::string outputVtkDir;

    std::vector<std::string> positionalArgs;
    for(int a = 1; a < argc; a++)
    {
        std::string arg(argv[a]);
        if(arg == "--help" or arg == "-h")
        {
            if(rank == 0)
            {
                PrintUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
        else if(arg == "--output-profile")
        {
            if(a + 1 < argc)
            {
                outputProfileFile = argv[++a];
            }
            else
            {
                if(rank == 0)
                {
                    std::cerr << "--output-profile requires a filename argument\n";
                }
                MPI_Finalize();
                return 1;
            }
        }
        else if(arg == "--output-vtk")
        {
            if(a + 1 < argc)
            {
                outputVtkDir = argv[++a];
            }
            else
            {
                if(rank == 0)
                {
                    std::cerr << "--output-vtk requires a directory argument\n";
                }
                MPI_Finalize();
                return 1;
            }
        }
        else
        {
            positionalArgs.push_back(arg);
        }
    }
    if(positionalArgs.size() >= 1)
    {
        N_base = std::stoul(positionalArgs[0]);
    }
    if(positionalArgs.size() >= 2)
    {
        newPhotonsPerCell = std::stoul(positionalArgs[1]);
    }
    if(positionalArgs.size() >= 3)
    {
        minPhotonsPerCell = std::stoul(positionalArgs[2]);
    }

#ifdef MADVORO_WITH_VTK
    if(not outputVtkDir.empty())
    {
        if(rank == 0)
        {
            std::filesystem::create_directories(outputVtkDir);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    {
    const double Lx = 1.4;
    const double Ly = 0.65;
    const double Lz = 0.65;
    const double pad = 2 * 0.03;
    Vector3D lower(0, -(Ly + pad), -(Lz + pad));
    Vector3D upper(Lx + pad, Ly + pad, Lz + pad);

    std::vector<Vector3D> allPoints;
    if(rank == 0)
    {
        allPoints = GenerateAllPoints(N_base, lower, upper);
        std::cout << "Generated " << allPoints.size() << " points (N_base=" << N_base << ")" << std::endl;
    }

    std::vector<Vector3D> localPoints = MPI_Spread(allPoints, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Grid grid(lower, upper);
    grid.BuildParallel(localPoints);

    size_t Ncells = grid.GetPointNo();
    size_t globalCells = 0;
    size_t localCellsSz = Ncells;
    MPI_Reduce(&localCellsSz, &globalCells, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        std::cout << "Hohlraum parallel Voronoi grid: " << globalCells << " cells across " << nprocs << " ranks" << std::endl;
    }

    double T_init = 300.0;
    double densityMaterial = 10.0;
    double densityVacuum = 0.01;
    double cvPerVolumeMaterial = 3e15 / STORM::constants::kev_kelvin;
    double cvPerVolumeVacuum = 1e15 / STORM::constants::kev_kelvin;

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<STORM::SimpleExtensives> extensives(Ncells);
    std::vector<int> materialFlags(Ncells, 0);

    size_t nMaterialLocal = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetCellCM(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        bool isMat = IsMaterial(center.x, r);
        materialFlags[i] = isMat ? 1 : 0;
        nMaterialLocal += isMat ? 1 : 0;

        double volume = grid.GetVolume(i);
        double density = isMat ? densityMaterial : densityVacuum;
        double cvPerVolume = isMat ? cvPerVolumeMaterial : cvPerVolumeVacuum;

        cells[i].temperature = T_init;
        cells[i].internalEnergy = cvPerVolume * T_init * volume;
        cells[i].Erad = STORM::constants::arad * std::pow(T_init, 4) / density;

        extensives[i].mass = density * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    size_t nMaterialGlobal = 0;
    MPI_Reduce(&nMaterialLocal, &nMaterialGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "Material cells: " << nMaterialGlobal << ", vacuum cells: " << (globalCells - nMaterialGlobal) << std::endl;
    }

    double T_drive = 1.0 * STORM::constants::kev_kelvin;
    constexpr size_t boundaryPhotonsPerCell = 1000;

    STORM::RadiationIMCParameters<1> params;
    params.newPhotonsPerCell = newPhotonsPerCell;
    params.withRandomWalk = true;
    params.energyBoundaries = {0.0, 1e30};
    params.energyBoundariesProvided = true;

    std::shared_ptr<STORM::examples::HohlraumEOS> eos =
        std::make_shared<STORM::examples::HohlraumEOS>(cvPerVolumeMaterial, cvPerVolumeVacuum, densityMaterial, densityVacuum);
    std::shared_ptr<STORM::examples::HohlraumOpacity<Vector3D, Grid>> opacityModel =
        std::make_shared<STORM::examples::HohlraumOpacity<Vector3D, Grid>>(materialFlags, cells);
    std::shared_ptr<STORM::examples::HohlraumBoundary<Vector3D, Grid>> boundary =
        std::make_shared<STORM::examples::HohlraumBoundary<Vector3D, Grid>>(grid, T_drive, boundaryPhotonsPerCell);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, params);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, minPhotonsPerCell, 6.0);

    STORM::MonteCarloManager<Vector3D, Grid> manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(
        grid, physics, popControl, boundary);

    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    double dt = 1e-11;
    const double t_final = 3e-9;
    const double max_dt = 5e-11;
    constexpr size_t dumpInterval = 25;
    size_t vtkPrintNumber = 0;

    if(rank == 0)
    {
        std::cout << "Running until t_final = " << t_final * 1e9 << " ns, initial dt = " << dt << " s" << std::endl;
        std::cout << "Drive temperature: " << T_drive << " K (" << T_drive / STORM::constants::kev_kelvin << " keV)" << std::endl;
        std::cout << "Photons: new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << ", min_per_cell=" << minPhotonsPerCell << std::endl;
        std::cout << std::endl;
    }

    constexpr size_t rebalanceInterval = 10;
    double simTime = 0;
    size_t step = 0;
    size_t stepsSinceLastDump = 0;
    auto startWall = std::chrono::high_resolution_clock::now();
    double computeTotal = 0;

    while(simTime < t_final)
    {
        if(step > 0 and (step % rebalanceInterval == 0 or step <= 2))
        {
            if(Rebalance(grid, manager, cells, extensives, materialFlags, particles, rank))
            {
                Ncells = grid.GetPointNo();
            }
        }

        auto stepStart = std::chrono::high_resolution_clock::now();
        particles = manager.step(std::move(particles), dt);
        double computeSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stepStart).count();
        computeTotal += computeSec;

        simTime += dt;
        stepsSinceLastDump++;

        if(simTime >= 1e-9)
        {
            dt = std::min(1.05 * dt, max_dt);
        }

        if(step % 10 == 0)
        {
            double localMaxT = 0;
            double localSumT = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                localSumT += cells[i].temperature;
                if(cells[i].temperature > localMaxT)
                {
                    localMaxT = cells[i].temperature;
                }
            }
            double globalMaxT = 0;
            double globalSumT = 0;
            size_t localParticles = particles.size();
            size_t globalParticles = 0;
            MPI_Reduce(&localMaxT, &globalMaxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localSumT, &globalSumT, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localParticles, &globalParticles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            if(rank == 0)
            {
                double fraction = simTime / t_final;
                double eta = (fraction > 0) ? computeTotal * (1.0 - fraction) / fraction : 0;
                double avgT = globalSumT / globalCells;
                std::cout << "Cycle " << step << "  t=" << simTime * 1e9 << " ns"
                          << " (" << static_cast<int>(fraction * 100) << "%)"
                          << "  step=" << computeSec << "s"
                          << "  " << globalParticles << " particles"
                          << "  max T = " << globalMaxT / STORM::constants::kev_kelvin << " keV"
                          << "  avg T = " << avgT / STORM::constants::kev_kelvin << " keV"
                          << "  ETA=" << static_cast<int>(eta) / 60 << "m" << static_cast<int>(eta) % 60 << "s"
                          << std::endl;
            }
        }

#ifdef MADVORO_WITH_VTK
        if(not outputVtkDir.empty() and stepsSinceLastDump >= dumpInterval)
        {
            stepsSinceLastDump = 0;
            DumpVTK(grid, cells, materialFlags, outputVtkDir, vtkPrintNumber++, rank);
        }
#endif

        step++;
    }

    if(rank == 0)
    {
        double wallSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startWall).count();
        std::cout << "Total wall time: " << wallSec << "s (compute: " << computeTotal << "s)" << std::endl;
    }

    double rTolerance = 0.05;
    std::vector<std::pair<double, double>> localProfile;
    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetCellCM(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        if(r < rTolerance)
        {
            localProfile.push_back({center.x, cells[i].temperature});
        }
    }

    std::vector<std::pair<double, double>> profile = MPI_Gatherv_serializable(localProfile, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::sort(profile.begin(), profile.end());
        std::cout << "\nFinal temperature profile along x-axis (r~0):" << std::endl;
        for(const std::pair<double, double> &entry : profile)
        {
            std::cout << "  x = " << entry.first * 10 << " mm: T = " << entry.second / STORM::constants::kev_kelvin << " keV" << std::endl;
        }

        std::string profileFile = outputProfileFile.empty() ? "profile.txt" : outputProfileFile;
        {
            std::ofstream out(profileFile);
            out << "# x(cm), T(K), T(keV)\n";
            for(const std::pair<double, double> &entry : profile)
            {
                out << entry.first << ", " << entry.second << ", " << entry.second / STORM::constants::kev_kelvin << "\n";
            }
            out.close();
            std::cout << "Wrote profile to " << profileFile << " (" << profile.size() << " cells)" << std::endl;
        }

        {
            std::string scriptDir = __FILE__;
            scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
            std::string cmd = "python3 " + scriptDir + "/plot_profile.py " + profileFile
                              + " --save hohlraum_profile.png";
            std::cout << "Running: " << cmd << std::endl;
            std::system(cmd.c_str());
        }

        std::cout << "\nDone." << std::endl;
    }

#ifdef MADVORO_WITH_VTK
    if(not outputVtkDir.empty() and stepsSinceLastDump > 0)
    {
        DumpVTK(grid, cells, materialFlags, outputVtkDir, vtkPrintNumber, rank);
    }
#else
    if(not outputVtkDir.empty() and rank == 0)
    {
        std::cerr << "Warning: --output-vtk requested but MadVoro was not built with VTK support (MADVORO_WITH_VTK). Skipping." << std::endl;
    }
#endif

    } // destroy grid, manager, physics, etc. before MPI_Finalize

    MPI_Finalize();
    return 0;
}
