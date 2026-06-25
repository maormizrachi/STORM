#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "examples/Vector3D.hpp"
#include "CartesianMesh3D.hpp"
#include "monte/radiation/SimpleRadiationPhysics.hpp"
#include "monte/radiation/RadiationCell.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/manager/MonteCarloManagerSerial.hpp"
#include "HohlraumOpacity.hpp"

int main()
{
    using Grid = MadCart::CartesianMesh3D<Vector3D>;

    double Lx = 0.1;  // 1 mm hohlraum length
    double Ly = 0.05;
    double Lz = 0.05;
    size_t Nx = 20, Ny = 5, Nz = 5;

    Vector3D lower(0, 0, 0), upper(Lx, Ly, Lz);
    Grid grid(lower, upper, Nx, Ny, Nz);

    size_t Ncells = grid.GetPointNo();
    std::cout << "Hohlraum grid: " << Ncells << " cells" << std::endl;

    std::vector<RDMont::RadiationCell> cells(Ncells);
    std::vector<int> materialFlags(Ncells, 0);

    double T_init = 300.0;         // 300 K initial temperature
    double cv_material = 3e15;     // erg/K/cm^3 (representative)
    double cv_vacuum = 1e6;

    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetMeshPoint(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        bool isMaterial = (center.x < 0.01 or center.x > (Lx - 0.01) or r > 0.03);
        materialFlags[i] = isMaterial ? 1 : 0;
        cells[i].temperature = T_init;
        cells[i].cv = isMaterial ? cv_material : cv_vacuum;
        cells[i].internalEnergy = cells[i].cv * cells[i].temperature;
    }

    double T_drive = 1.0 * 1.16045e7;  // 1 keV in Kelvin

    auto opacityModel = std::make_shared<RDMont::examples::HohlraumOpacity>(materialFlags);
    auto boundary = std::make_shared<RDMont::SideTemperature<Vector3D, Grid>>(grid, T_drive, 5);
    auto physics = std::make_shared<RDMont::SimpleRadiationPhysics<Vector3D, Grid>>(grid, boundary, cells, opacityModel, 5);
    auto popControl = std::make_shared<RDMont::CombPopulationControl<Vector3D, Grid>>(grid, 15, 6.0);

    RDMont::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);

    std::vector<RDMont::Particle<Vector3D, Grid>> particles;

    double dt = 1e-11;  // 10 ps
    size_t Nsteps = 100;

    std::cout << "Running " << Nsteps << " steps with dt = " << dt << " s" << std::endl;
    std::cout << "Drive temperature: " << T_drive << " K (" << T_drive / 1.16045e7 << " keV)" << std::endl;
    std::cout << std::endl;

    for(size_t step = 0; step < Nsteps; step++)
    {
        particles = manager.step(std::move(particles), dt);

        if(step % 10 == 0 or step == Nsteps - 1)
        {
            double maxT = 0;
            double avgT = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                avgT += cells[i].temperature;
                if(cells[i].temperature > maxT)
                {
                    maxT = cells[i].temperature;
                }
            }
            avgT /= Ncells;
            std::cout << "Step " << step << ": " << particles.size() << " particles, "
                      << "max T = " << maxT / 1.16045e7 << " keV, "
                      << "avg T = " << avgT / 1.16045e7 << " keV" << std::endl;
        }
    }

    std::cout << "\nFinal temperature profile along x-axis (y=z=center):" << std::endl;
    for(size_t ix = 0; ix < Nx; ix++)
    {
        double xc = (ix + 0.5) * (Lx / Nx);
        Vector3D probe(xc, Ly / 2.0, Lz / 2.0);
        size_t cellIdx = grid.GetContainingCell(probe);
        std::cout << "  x = " << xc * 10 << " mm: T = " << cells[cellIdx].temperature / 1.16045e7 << " keV" << std::endl;
    }

    return 0;
}
