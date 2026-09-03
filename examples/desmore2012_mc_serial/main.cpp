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

#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include <units/units.hpp>

#include "manager/MonteCarloManagerSerial.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationIMC.hpp"
#include "../densmore2012/DensmoreBoundary.hpp"
#include "../densmore2012/DensmoreOpacity.hpp"
#include "../desmore2012_mc/densmore2012_mesh.hpp"

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

} // anonymous namespace

int main()
{
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
        std::vector<Vector3D> points =
            densmore2012_mesh::BuildVoronoiSites();
        Grid grid(lowerLeft, upperRight);
        grid.Build(points);
        size_t const nCells = grid.GetPointNo();

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
        parameters.withRandomWalk = true;
        parameters.energyBoundaries = energyBoundaries;
        parameters.energyBoundariesProvided = true;

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
        STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(
            grid, physics, population, boundary);
        std::vector<Particle> particles;

        std::cout << "Densmore 2012 heterogeneous step-opacity"
                  << " (serial MC regression)"
                  << "\n  Nx=" << densmore2012_mesh::cellCount
                  << ", G=" << G << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << "\n  dt=" << dt << " s, t_final=" << finalTime
                  << " s, iterations=" << iterations << std::endl;

        for(size_t step = 0; step < iterations; ++step)
        {
            particles = manager.step(std::move(particles), dt);
            if(step % 10 == 0 || step + 1 == iterations)
            {
                std::cout << "Cycle " << step + 1 << "/" << iterations
                          << " (" << static_cast<int>(
                              100.0 * (step + 1) / iterations)
                          << "%)" << std::endl;
            }
        }

        std::vector<std::pair<double, double>> profile(nCells);
        for(size_t i = 0; i < nCells; ++i)
        {
            profile[i] = {grid.GetCellCM(i).x, cells[i].temperature};
        }
        std::sort(profile.begin(), profile.end());
        std::ofstream output("desmore2012_mc_serial_profile.txt");
        output << "# Densmore2012 serial MC+RW  t=" << finalTime
               << "  Nx=" << densmore2012_mesh::cellCount << "\n";
        output << "# x(cm)  T(K)\n";
        for(const std::pair<double, double> &point : profile)
        {
            output << std::setprecision(16) << point.first << " "
                   << point.second << "\n";
        }
        std::cout << "Wrote desmore2012_mc_serial_profile.txt" << std::endl;
    }
    catch(const std::exception &error)
    {
        std::cerr << "=== exception: " << error.what() << " ==="
                  << std::endl;
        return 1;
    }
    return 0;
}
