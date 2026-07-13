/*
 * Densmore et al. (2012), Figure 4: heterogeneous step-opacity benchmark.
 *
 * 1D slab, x in [0, 3] cm.
 * Left BC:  Planck source at T = 1 keV;  particles escape.
 * Right BC: reflecting.
 * All y/z BCs: reflecting.
 *
 * Opacity: sigma(E) = sigma0 / (sqrt(kT) * E^3)
 *   sigma0 = 10  keV^{3.5}/cm   for x < 2 cm
 *   sigma0 = 1000 keV^{3.5}/cm  for x >= 2 cm
 *
 * EOS: Cv = 1e15 / kev_kelvin  erg/K/cm^3   (constant, linear EOS)
 * Initial T = 1 eV.   density = 1.
 * dt = 5e-12 s,  t_final = 1e-9 s  (200 steps).
 *
 * 30-group frequency-dependent transport with opacity-weighted Planck
 * emission sampling.
 *
 * Supports both serial and MPI-parallel execution.
 *
 * Usage:
 *   Serial:       ./densmore2012 [Nx] [new_per_cell] [boundary_per_cell]
 *   MPI parallel: mpirun -np N ./densmore2012 [Nx] [new_per_cell] [boundary_per_cell]
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
#include "DensmoreOpacity.hpp"
#include "DensmoreBoundary.hpp"

using Grid = MadCart::CartesianMesh3D<Vector3D>;

class DensmoreEOS
{
public:
    DensmoreEOS(double cvPerVolume, double density)
        : cvPerMass_(cvPerVolume / density)
    {}

    double dT2cv(double /*density*/, double /*temperature*/,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return cvPerMass_;
    }

    double de2T(double /*density*/, double specificEnergy,
                const std::vector<double> &, const std::vector<std::string> &) const
    {
        return (cvPerMass_ > 0.0) ? specificEnergy / cvPerMass_ : 0.0;
    }

private:
    double cvPerMass_;
};

struct RefPoint
{
    double x, T_keV;
};

static std::vector<RefPoint> LoadCSV(const std::string &path)
{
    std::vector<RefPoint> ref;
    std::ifstream in(path);
    if(!in.is_open())
    {
        return ref;
    }
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() or line[0] == '#')
        {
            continue;
        }
        double x, T;
        char comma;
        std::istringstream iss(line);
        if(iss >> x >> comma >> T)
        {
            ref.push_back({x, T});
        }
    }
    return ref;
}

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

    constexpr size_t G = STORM::examples::N_DENSMORE_GROUPS;
    using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
                                    DensmoreEOS, G>;

    size_t Nx = (argc >= 2) ? std::stoul(argv[1]) : 512;
    size_t newPhotonsPerCell = (argc >= 3) ? std::stoul(argv[2]) : 16;
    size_t boundaryPhotonsPerCell = (argc >= 4) ? std::stoul(argv[3]) : 100;

#ifdef STORM_WITH_MPI
    {
#endif

    double keV_K = units::kev_kelvin;
    double eV_K = keV_K / 1000.0;

    double domainLength = 3.0;
    double xStep = 2.0;
    double T_init = eV_K;
    double T_boundary = keV_K;
    double density = 1.0;
    double cvPerVolume = 1e15 / keV_K;
    double tf = 1e-9;
    double dt = 5e-12;
    size_t iterations = static_cast<size_t>(tf / dt);

    double Emin = units::kev * 1e-4;
    double Emax = units::kev * 1e2;
    std::array<double, G + 1> energyBoundaries{};
    energyBoundaries[0] = Emin;
    double ratio = std::pow(Emax / Emin, 1.0 / G);
    for(size_t g = 0; g < G; ++g)
    {
        energyBoundaries[g + 1] = energyBoundaries[g] * ratio;
    }

    double dy = domainLength / Nx;
    Vector3D lower(0, 0, 0);
    Vector3D upper(domainLength, dy, dy);

    Grid grid(lower, upper, Nx, 1, 1);

#ifdef STORM_WITH_MPI
    std::vector<double> uniformWeights(Nx, 1.0);
    grid.BuildParallel(uniformWeights);
#endif

    size_t Ncells = grid.GetPointNo();

    if(rank == 0)
    {
#ifdef STORM_WITH_MPI
        size_t globalCells = 0;
        MPI_Reduce(&Ncells, &globalCells, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        std::cout << "Densmore 2012 heterogeneous step-opacity (" << G << "-group MC, "
                  << nprocs << " ranks, " << globalCells << " global cells)" << std::endl;
#else
        std::cout << "Densmore 2012 heterogeneous step-opacity (" << G << "-group MC)" << std::endl;
#endif
        std::cout << "Nx=" << Nx << ", domain=[0, " << domainLength << "] cm" << std::endl;
        std::cout << "new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << std::endl;
        std::cout << "dt=" << dt << " s, t_final=" << tf << " s, iterations=" << iterations << std::endl;
    }
#ifdef STORM_WITH_MPI
    else
    {
        MPI_Reduce(&Ncells, nullptr, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<STORM::SimpleExtensives> extensives(Ncells);
    std::vector<int> regionFlags(Ncells, 0);

    for(size_t i = 0; i < Ncells; i++)
    {
        double x = grid.GetCellCM(i).x;
        double volume = grid.GetVolume(i);
        regionFlags[i] = (x < xStep) ? 1 : 0;
        cells[i].temperature = T_init;
        cells[i].internalEnergy = cvPerVolume * T_init * volume;
        extensives[i].mass = density * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    STORM::RadiationIMCParameters<G> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.withRandomWalk = true;
    imcParams.withMultigroupOpacity = true;
    imcParams.energyBoundaries = energyBoundaries;
    imcParams.energyBoundariesProvided = true;

    std::shared_ptr<DensmoreEOS> eos = std::make_shared<DensmoreEOS>(cvPerVolume, density);
    std::shared_ptr<STORM::examples::DensmoreOpacity<Vector3D, Grid>> opacityModel =
        std::make_shared<STORM::examples::DensmoreOpacity<Vector3D, Grid>>(regionFlags, cells);
    opacityModel->setGroupBoundaries(energyBoundaries);
    std::shared_ptr<STORM::examples::DensmoreBoundary<Vector3D, Grid>> boundary =
        std::make_shared<STORM::examples::DensmoreBoundary<Vector3D, Grid>>(grid, T_boundary, boundaryPhotonsPerCell, energyBoundaries);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 200, 5.0);

#ifdef STORM_WITH_MPI
    STORM::MonteCarloManager<Vector3D, Grid> manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(
        grid, physics, popControl, boundary);
#else
    STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);
#endif
    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    for(size_t step = 0; step < iterations; step++)
    {
        particles = manager.step(std::move(particles), dt);

        if(rank == 0 && (step % 20 == 0 || step + 1 == iterations))
        {
            double maxT = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                maxT = std::max(maxT, cells[i].temperature);
            }
            double fraction = double(step + 1) / iterations;
            int pct = static_cast<int>(fraction * 100);
            std::cout << "Step " << step + 1 << "/" << iterations
                      << " (" << pct << "%)"
                      << "  particles=" << particles.size()
                      << "  maxT=" << maxT / keV_K << " keV" << std::endl;
        }
    }

    std::vector<double> allX, allT;

#ifdef STORM_WITH_MPI
    {
        std::vector<double> localX(Ncells), localT(Ncells);
        for(size_t i = 0; i < Ncells; i++)
        {
            localX[i] = grid.GetMeshPoint(i).x;
            localT[i] = cells[i].temperature;
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
            allT.resize(total);
        }
        MPI_Gatherv(localX.data(), localCount, MPI_DOUBLE,
                     allX.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localT.data(), localCount, MPI_DOUBLE,
                     allT.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#else
    allX.resize(Ncells);
    allT.resize(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        allX[i] = grid.GetMeshPoint(i).x;
        allT[i] = cells[i].temperature;
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

        std::vector<double> simX(totalCells), simT(totalCells);
        for(size_t i = 0; i < totalCells; i++)
        {
            simX[i] = allX[idx[i]];
            simT[i] = allT[idx[i]];
        }

        std::string profilePath = "densmore2012_profile.txt";
        {
            std::ofstream out(profilePath);
            out << "# Densmore2012 gray MC  t=" << tf << "  Nx=" << Nx << "\n";
            out << "# x(cm)  T(K)\n";
            for(size_t i = 0; i < totalCells; i++)
            {
                out << simX[i] << " " << simT[i] << "\n";
            }
            std::cout << "\nWrote " << profilePath << std::endl;
        }

        std::string refPath = std::string(STORM_DATA_DIR) + "/data/densmore2012_fig4_mc.csv";
        std::vector<RefPoint> ref = LoadCSV(refPath);
        if(!ref.empty())
        {
            double l1sum = 0;
            size_t count = 0;
            for(const RefPoint &rp : ref)
            {
                size_t j = 0;
                while(j + 1 < totalCells && simX[j + 1] < rp.x)
                {
                    j++;
                }
                if(j + 1 >= totalCells)
                {
                    continue;
                }
                double frac = (rp.x - simX[j]) / (simX[j + 1] - simX[j]);
                double T_interp = simT[j] + frac * (simT[j + 1] - simT[j]);
                double T_keV = T_interp / keV_K;
                l1sum += std::abs(T_keV - rp.T_keV);
                count++;
            }
            double l1 = (count > 0) ? l1sum / count : -1;
            std::cout << "DENSMORE2012_TGAS_L1 = " << std::scientific << l1 << " keV" << std::endl;
            if(l1 >= 0 && l1 < 0.10)
            {
                std::cout << "PASS (L1 < 0.10 keV)" << std::endl;
            }
            else
            {
                std::cout << "WARN: gray approximation may differ from multigroup reference" << std::endl;
            }
        }
        else
        {
            std::cout << "No reference data at " << refPath << " — skipping comparison" << std::endl;
        }

        {
            std::string scriptDir = __FILE__;
            scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
            std::string cmd = "python3 " + scriptDir + "/plot_densmore.py";
            std::cout << "Running: " << cmd << std::endl;
            std::system(cmd.c_str());
        }

        std::cout << "\nDone." << std::endl;
    }

#ifdef STORM_WITH_MPI
    }
    MPI_Finalize();
#endif
    return 0;
}
