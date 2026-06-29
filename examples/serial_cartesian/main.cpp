#include <iostream>
#include <memory>
#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "particle/Particle.hpp"
#include "physics/NoPhysics.hpp"
#include "boundary/RigidBoundary.hpp"
#include "population/NoPopulationControl.hpp"
#include "manager/MonteCarloManagerSerial.hpp"

int main()
{
    using Grid = MadCart::CartesianMesh3D<Vector3D>;

    Vector3D lower(0, 0, 0), upper(1, 1, 1);
    Grid grid(lower, upper, 10, 10, 10);

    std::cout << "Grid: " << grid.GetPointNo() << " cells, " << grid.GetTotalFacesNumber() << " faces" << std::endl;

    auto boundary = std::make_shared<STORM::RigidBoundary<Vector3D, Grid>>(grid);
    auto physics = std::make_shared<STORM::NoPhysics<Vector3D, Grid>>(grid, boundary);
    auto popControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Grid>>(grid);

    STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);

    std::vector<STORM::Particle<Vector3D, Grid>> particles;
    for(size_t i = 0; i < 100; i++)
    {
        STORM::Particle<Vector3D, Grid> p;
        p.location = Vector3D(0.55, 0.55, 0.55);
        double theta = 2 * M_PI * i / 100.0;
        double phi = M_PI * (i % 50) / 50.0;
        p.velocity = Vector3D(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
        p.weight = 1.0;
        p.cellIndex = grid.GetContainingCell(p.location);
        particles.push_back(p);
    }

    double dt = 1e-3;
    for(int step = 0; step < 10; step++)
    {
        particles = manager.step(std::move(particles), dt);
        std::cout << "Step " << step << ": " << particles.size() << " particles remaining" << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
