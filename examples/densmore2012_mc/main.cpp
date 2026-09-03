#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include <mpi_utils/mpi_collectives.hpp>
#include <units/units.hpp>

#include "manager/MonteCarloManagerFactory.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationIMC.hpp"
#include "../densmore2012/DensmoreBoundary.hpp"
#include "../densmore2012/DensmoreOpacity.hpp"
#include "densmore2012_mesh.hpp"

namespace
{

using Grid = MadVoro::Voronoi3D<Vector3D>;
using Particle = STORM::Particle<Vector3D>;
constexpr size_t G = STORM::examples::N_DENSMORE_GROUPS;

class DensmoreEOS
{
public:
    DensmoreEOS(double cvPerVolume, double density)
        : cvPerMass_(cvPerVolume / density)
    {}

    double dT2cv(double, double, const std::vector<double> &,
                 const std::vector<std::string> &) const
    {
        return cvPerMass_;
    }

    double de2T(double, double specificEnergy, const std::vector<double> &,
                const std::vector<std::string> &) const
    {
        return specificEnergy / cvPerMass_;
    }

private:
    double cvPerMass_;
};

struct ProfilePoint
{
    double x;
    double temperature;
};

bool operator<(const ProfilePoint &left, const ProfilePoint &right)
{
    return left.x < right.x;
}

} // anonymous namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try
    {
        constexpr size_t newPhotonsPerCell = 50;
        constexpr size_t maxPhotonsPerCell = 200;
        constexpr size_t boundaryPhotonsPerCell = 100;
        constexpr double domainLength = densmore2012_mesh::domainLength;
        constexpr double transverseWidth = 3.0 / 256.0;
        constexpr double dt = 5e-12;
        constexpr double finalTime = 1e-9;
        constexpr size_t iterations =
            static_cast<size_t>(finalTime / dt);

        std::array<double, G + 1> energyBoundaries{};
        double const minimumEnergy = units::kev * 1e-4;
        double const maximumEnergy = units::kev * 1e2;
        energyBoundaries[0] = minimumEnergy;
        double const ratio = std::pow(maximumEnergy / minimumEnergy, 1.0 / G);
        for(size_t g = 0; g < G; ++g)
        {
            energyBoundaries[g + 1] = energyBoundaries[g] * ratio;
        }

        Vector3D lowerLeft(0, -0.5 * transverseWidth, -0.5 * transverseWidth);
        Vector3D upperRight(domainLength, 0.5 * transverseWidth,
                            0.5 * transverseWidth);
        std::vector<Vector3D> points;
        if(rank == 0)
        {
            points = densmore2012_mesh::BuildVoronoiSites();
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);

        Grid grid(lowerLeft, upperRight);
        grid.BuildParallel(points);
        size_t nCells = grid.GetPointNo();

        double const temperatureInitial = units::ev_kelvin;
        double const temperatureBoundary = units::kev_kelvin;
        double const density = 1.0;
        double const cvPerVolume = 1e15 / units::kev_kelvin;
        DensmoreEOS eosModel(cvPerVolume, density);
        std::vector<STORM::RadiationCell> cells(nCells);
        std::vector<STORM::SimpleExtensives> extensives(nCells);
        std::vector<int> regionFlags(nCells, 0);
        for(size_t i = 0; i < nCells; ++i)
        {
            double volume = grid.GetVolume(i);
            regionFlags[i] = grid.GetCellCM(i).x < 2.0 ? 1 : 0;
            cells[i].temperature = temperatureInitial;
            cells[i].internalEnergy = cvPerVolume * temperatureInitial * volume;
            extensives[i].mass = density * volume;
            extensives[i].internal_energy = cells[i].internalEnergy;
        }

        using IMC = STORM::RadiationIMC<
            Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
            DensmoreEOS, G>;
        STORM::RadiationIMCParameters<G> parameters;
        parameters.newPhotonsPerCell = newPhotonsPerCell;
        parameters.withMultigroupOpacity = true;
        parameters.withRandomWalk = false;
        parameters.withEgTimeAvg = true;
        parameters.energyBoundaries = energyBoundaries;
        parameters.energyBoundariesProvided = true;
#ifdef STORM_DENSMORE_MC_DDMC_REGRESSION
        parameters.withDDMC = true;
        parameters.withMultigroupDDMC = true;
#endif

        std::shared_ptr<DensmoreEOS> eos =
            std::make_shared<DensmoreEOS>(eosModel);
        std::shared_ptr<STORM::examples::DensmoreOpacity<Vector3D, Grid>> opacity =
            std::make_shared<STORM::examples::DensmoreOpacity<Vector3D, Grid>>(
                regionFlags, cells);
        opacity->setGroupBoundaries(energyBoundaries);
        std::shared_ptr<STORM::examples::DensmoreBoundary<Vector3D, Grid>> boundary =
            std::make_shared<STORM::examples::DensmoreBoundary<Vector3D, Grid>>(
                grid, temperatureBoundary, boundaryPhotonsPerCell,
                energyBoundaries);
        std::shared_ptr<IMC> physics = std::make_shared<IMC>(
            grid, boundary, cells, extensives, eos, opacity, parameters);
        std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> population =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(
                grid, maxPhotonsPerCell, 5.0);
        STORM::MonteCarloManager<Vector3D, Grid> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(
                grid, physics, population, boundary);
        std::vector<Particle> particles;

        if(rank == 0)
        {
            std::cout << "Densmore 2012 heterogeneous step-opacity"
#ifdef STORM_DENSMORE_MC_DDMC_REGRESSION
                      << " (MC regression, DDMC)"
#else
                      << " (MC regression)"
#endif
                      << "\n  Nx=" << densmore2012_mesh::cellCount
                      << ", G=" << G << ", new/cell=" << newPhotonsPerCell
                      << ", max/cell=" << maxPhotonsPerCell
                      << "\n  dt=" << dt << " s, t_final=" << finalTime
                      << " s, iterations=" << iterations << std::endl;
        }

        for(size_t step = 0; step < iterations; ++step)
        {
            particles = manager.step(std::move(particles), dt);
            if(rank == 0 && (step % 10 == 0 || step + 1 == iterations))
            {
                std::cout << "Cycle " << step + 1 << "/" << iterations
                          << " (" << static_cast<int>(
                              100.0 * (step + 1) / iterations)
                          << "%)" << std::endl;
            }
        }

        std::vector<double> localX(nCells);
        std::vector<double> localTemperature(nCells);
        for(size_t i = 0; i < nCells; ++i)
        {
            localX[i] = grid.GetCellCM(i).x;
            localTemperature[i] = cells[i].temperature;
        }
        int localCount = static_cast<int>(nCells);
        std::vector<int> counts(nprocs);
        std::vector<int> displacements(nprocs);
        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                   MPI_COMM_WORLD);

        std::vector<double> allX;
        std::vector<double> allTemperature;
        if(rank == 0)
        {
            displacements[0] = 0;
            for(int r = 1; r < nprocs; ++r)
            {
                displacements[r] = displacements[r - 1] + counts[r - 1];
            }
            int total = displacements[nprocs - 1] + counts[nprocs - 1];
            allX.resize(total);
            allTemperature.resize(total);
        }
        MPI_Gatherv(localX.data(), localCount, MPI_DOUBLE, allX.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(localTemperature.data(), localCount, MPI_DOUBLE,
                    allTemperature.data(), counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(rank == 0)
        {
            std::vector<ProfilePoint> profile(allX.size());
            for(size_t i = 0; i < profile.size(); ++i)
            {
                profile[i] = {allX[i], allTemperature[i]};
            }
            std::sort(profile.begin(), profile.end());
            std::ofstream output("densmore2012_mc"
#ifdef STORM_DENSMORE_MC_DDMC_REGRESSION
                                 "_ddmc"
#endif
                                 "_profile.txt");
            output << "# Densmore2012 MC regression  t=" << finalTime
                   << "  Nx=" << densmore2012_mesh::cellCount << "\n";
            output << "# x(cm)  T(K)\n";
            for(const ProfilePoint &point : profile)
            {
                output << std::setprecision(16) << point.x << " "
                       << point.temperature << "\n";
            }
            std::cout << "Wrote Densmore profile" << std::endl;
        }
    }
    catch(const std::exception &error)
    {
        std::cerr << "=== exception on rank " << rank << ": "
                  << error.what() << " ===" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
