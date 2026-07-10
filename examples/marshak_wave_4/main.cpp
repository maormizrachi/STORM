/*
 * Marshak wave problem 4: Derei et al. (2024) Test 3
 *
 * Divergent density on a stretched grid.
 * kappa_R = 2*(T/keV)^{-4.5}*rho^{1.9}, kappa_P = 5e-4*kappa_R
 * u(T,rho) = 1e14*(T/keV)^6*rho^{0.7} erg/cm^3
 * rho(x) = x^{-40/139}, geometrically stretched grid
 * T_bath(t) = 1.01008116*(t/ns)^{14/139} keV
 * Domain [1e-5, 1+1e-5] cm, t_final = 1 ns
 *
 * Uses CartesianMesh3D with non-uniform x-edges.
 * Supports both serial and MPI-parallel execution.
 *
 * Usage:
 *   Serial:          ./marshak_wave_4 [new_per_cell] [boundary_per_cell]
 *   MPI parallel:    mpirun -np N ./marshak_wave_4 [new_per_cell] [boundary_per_cell]
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#ifdef STORM_WITH_MPI
#include <mpi.h>
#include <mpi_utils/mpi_collectives.hpp>
#endif
#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include <units/units.hpp>
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationCell.hpp"
#include "population/CombPopulationControl.hpp"
#ifdef STORM_WITH_MPI
#include "manager/MonteCarloManagerFactory.hpp"
#else
#include "manager/MonteCarloManagerSerial.hpp"
#endif
#include "examples/marshak_wave/MarshakOpacity.hpp"
#include "examples/marshak_wave/MarshakBoundary.hpp"
#include "examples/marshak_wave/MarshakCommon.hpp"

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using ParticleT = STORM::Particle<Vector3D, Grid>;
using namespace STORM;
using namespace STORM::examples;
using namespace units;

int main(int argc, char *argv[])
{
#ifdef STORM_WITH_MPI
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    int rank = 0;
#endif

    using IMC = RadiationIMC<Vector3D, Grid, RadiationCell, SimpleExtensives, MarshakEOS, 1>;

    size_t newPhotonsPerCell = (argc >= 2) ? std::stoul(argv[1]) : 15;
    size_t boundaryPhotonsPerCell = (argc >= 3) ? std::stoul(argv[2]) : 100;

#ifdef STORM_WITH_MPI
    { // scope: MPI-dependent objects must be destroyed before MPI_Finalize
#endif
    ProblemParams params = GetProblemParams(4);
    double xMax = params.xOffset + params.domainLength;

    std::vector<double> xEdges = BuildGeometricMeshEdges(0.0, xMax);
    size_t globalNx = xEdges.size() - 1;
    double dy = xMax / static_cast<double>(globalNx);

    Grid grid(xEdges, 0.0, dy, 1, 0.0, dy, 1);

#ifdef STORM_WITH_MPI
    std::vector<double> uniformWeights(globalNx, 1.0);
    grid.BuildParallel(uniformWeights);
#endif

    size_t Ncells = grid.GetPointNo();

#ifdef STORM_WITH_MPI
    size_t globalCells = 0;
    {
        size_t localSz = Ncells;
        MPI_Reduce(&localSz, &globalCells, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif

    if(rank == 0)
    {
#ifdef STORM_WITH_MPI
        std::cout << "Marshak wave problem 4 (parallel): " << globalCells << " cells across "
                  << nprocs << " ranks (" << globalNx << " global), domain [0, " << xMax << "] cm" << std::endl;
#else
        std::cout << "Marshak wave problem 4: " << Ncells << " cells, domain [0, " << xMax << "] cm" << std::endl;
#endif
    }

    double keV_K = kev_kelvin;
    double T_init = 1e-3 * keV_K;

    std::vector<RadiationCell> cells(Ncells);
    std::vector<SimpleExtensives> extensives(Ncells);
    std::vector<double> densities(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        double x = grid.GetCellCM(i).x;
        double volume = grid.GetVolume(i);
        double rho = ComputeDensity(4, x);
        densities[i] = rho;
        cells[i].temperature = T_init;
        cells[i].internalEnergy = EOS_E_from_T(params, T_init, rho) * volume;
        extensives[i].mass = rho * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    double T_bath_init = BathTemperature(params, params.initialDt);

    RadiationIMCParameters<1> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.withRandomWalk = true;
    imcParams.energyBoundaries = {0.0, 1e30};
    imcParams.energyBoundariesProvided = true;

    std::shared_ptr<MarshakEOS> eos = std::make_shared<MarshakEOS>(params);
    std::shared_ptr<MarshakOpacity<Vector3D, Grid>> opacityModel =
        std::make_shared<MarshakOpacity<Vector3D, Grid>>(params.kappaP0, params.kappaR0, params.alpha, params.betaRho, densities, cells);
    std::shared_ptr<MarshakBoundary<Vector3D, Grid>> boundary =
        std::make_shared<MarshakBoundary<Vector3D, Grid>>(grid, T_bath_init, boundaryPhotonsPerCell);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    std::shared_ptr<CombPopulationControl<Vector3D, Grid>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, Grid>>(grid, 15, 6.0);

#ifdef STORM_WITH_MPI
    MonteCarloManager<Vector3D, Grid> manager = CreateMonteCarloManager<Vector3D, Grid>(
        grid, physics, popControl, boundary);
#else
    MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);
#endif

    std::vector<ParticleT> particles;

    double dt = params.initialDt;
    double maxDt = 5e-13;
    double simTime = 0;
    size_t cycle = 0;

    if(rank == 0)
    {
        std::cout << "T_bath(t_final) = " << BathTemperature(params, params.tf) / keV_K << " keV" << std::endl;
        std::cout << "new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << std::endl;
        std::cout << std::endl;
    }

    auto wallStart = std::chrono::high_resolution_clock::now();

    while(simTime < params.tf)
    {
        double t_now = std::max(simTime, params.initialDt);
        double T_bath = BathTemperature(params, t_now);
        boundary->SetTemperature(T_bath);

        dt = std::min(dt, params.tf - simTime);

        if(rank == 0)
        {
            double maxT_keV = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                maxT_keV = std::max(maxT_keV, cells[i].temperature / keV_K);
            }
            int pct = static_cast<int>(simTime / params.tf * 100);
            std::cout << "Cycle " << cycle << " (" << pct << "%)  dt=" << dt
                      << "  t=" << simTime * 1e9 << "/" << params.tf * 1e9 << " ns"
                      << "  maxT=" << maxT_keV << " keV  T_bath=" << T_bath / keV_K << " keV" << std::endl;
        }

        particles = manager.step(std::move(particles), dt);

        simTime += dt;
        cycle++;

        double newDt = std::max(params.initialDt, simTime * 1e-3);
        dt = std::min(newDt, maxDt);
    }

    double wallSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();

    const std::vector<double> &EradTimeAvg = physics->getEradTimeAvg();
    Ncells = grid.GetPointNo();

    std::vector<double> allX, allTgas, allTrad;

#ifdef STORM_WITH_MPI
    {
        std::vector<double> localX(Ncells), localTgas(Ncells), localTrad(Ncells);
        for(size_t i = 0; i < Ncells; i++)
        {
            localX[i] = grid.GetMeshPoint(i).x;
            localTgas[i] = cells[i].temperature;
            double Erad = std::max(EradTimeAvg[i], 0.0);
            localTrad[i] = std::pow(Erad / arad, 0.25);
        }

        int localCount = static_cast<int>(Ncells);
        std::vector<int> recvCounts(nprocs), displacements(nprocs);
        MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(rank == 0)
        {
            displacements[0] = 0;
            for(int r = 1; r < nprocs; r++)
            {
                displacements[r] = displacements[r - 1] + recvCounts[r - 1];
            }
            int total = displacements[nprocs - 1] + recvCounts[nprocs - 1];
            allX.resize(total);
            allTgas.resize(total);
            allTrad.resize(total);
        }
        MPI_Gatherv(localX.data(), localCount, MPI_DOUBLE, allX.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localTgas.data(), localCount, MPI_DOUBLE, allTgas.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localTrad.data(), localCount, MPI_DOUBLE, allTrad.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#else
    allX.resize(Ncells);
    allTgas.resize(Ncells);
    allTrad.resize(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        allX[i] = grid.GetMeshPoint(i).x;
        allTgas[i] = cells[i].temperature;
        double Erad = std::max(EradTimeAvg[i], 0.0);
        allTrad[i] = std::pow(Erad / arad, 0.25);
    }
#endif

    if(rank == 0)
    {
        size_t totalCells = allX.size();
        std::vector<size_t> idx(totalCells);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return allX[a] < allX[b];
        });

        std::vector<double> simX(totalCells), simT(totalCells), simTrad(totalCells);
        for(size_t i = 0; i < totalCells; i++)
        {
            simX[i] = allX[idx[i]];
            simT[i] = allTgas[idx[i]];
            simTrad[i] = allTrad[idx[i]];
        }

        std::string profilePath = "marshak_wave_4_profile.txt";
        {
            std::ofstream out(profilePath);
            out << std::scientific << std::setprecision(12);
            for(size_t i = 0; i < totalCells; i++)
            {
                out << simX[i] << " " << simT[i] << " " << simTrad[i] << "\n";
            }
            std::cout << "\nWrote " << profilePath << std::endl;
        }

        std::string refPath = std::string(STORM_DATA_DIR) + "/reference.txt";
        std::vector<ReferencePoint> ref = LoadReference(refPath);
        if(!ref.empty())
        {
            double l1 = ComputeL1(simX, simT, ref, true);
            std::cout << "TGAS_REL_L1 = " << std::scientific << l1 << std::endl;
            if(l1 >= 0 && l1 < 0.10)
            {
                std::cout << "PASS (L1 < 0.10)" << std::endl;
            }
            else if(l1 >= 0)
            {
                std::cout << "WARN: L1 = " << l1 << " >= 0.10" << std::endl;
            }
        }

        std::cout << "\nWall time: " << wallSec << " s, cycles: " << cycle << std::endl;

        std::string scriptDir = __FILE__;
        scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
        std::string cmd = "python3 " + scriptDir + "/../marshak_wave/plot_marshak.py 4";
        std::cout << "Running: " << cmd << std::endl;
        std::system(cmd.c_str());

        std::cout << "\nDone." << std::endl;
    }
#ifdef STORM_WITH_MPI
    } // end MPI scope
    MPI_Finalize();
#endif
    return 0;
}
