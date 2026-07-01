#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include "radiation/SimpleRadiationPhysics.hpp"
#include "radiation/RadiationCell.hpp"
#include "boundary/SideTemperature.hpp"
#include "population/CombPopulationControl.hpp"
#include "manager/MonteCarloManagerSerial.hpp"
#include "HohlraumOpacity.hpp"

/*
 * 2D Cylindrical Hohlraum benchmark from:
 *   McClarren & Urbatsch (2009), as presented in
 *   Steinberg & Heizler (2021), arXiv:2108.13453, Section 4.2.
 *
 * Domain (3D):  x in [0, 1.4],  y in [-0.65, 0.65],  z in [-0.65, 0.65] cm
 * Symmetry axis is x; r = sqrt(y^2 + z^2).
 *
 * Material regions (absorbing):
 *   Left wall:    x in [0.10, 0.15], r <= 0.45
 *   Capsule:      x in [0.55, 0.95], r <= 0.45
 *   Right end cap: x in [1.35, 1.40], r <= 0.65
 *   Outer wall:   x in [0.10, 1.40], r in [0.60, 0.65]
 *
 * Material:   sigma_a = 300 * (T/keV)^{-3} cm^{-1},  Cv = 3e15 erg/keV/cm^3
 * Vacuum:     sigma_a ~ 0,  Cv ~ negligible
 *
 * BC:  x=0 (left) blackbody at T_drive = 1 keV;  all others reflecting.
 * Init: T = 300 K
 * dt = 1e-11 s
 *
 * Usage:
 *   ./hohlraum [N_base] [new_per_cell] [min_per_cell]
 *
 * N_base:        total number of Voronoi mesh generator points (default 2000)
 * new_per_cell:  new photon packets per cell per step (default 5)
 * min_per_cell:  target photon packets per cell after population control (default 15)
 */

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

int main(int argc, char *argv[])
{
    using Grid = MadVoro::Voronoi3D<Vector3D>;

    size_t N_base = 2000;
    size_t newPhotonsPerCell = 5;
    size_t minPhotonsPerCell = 15;

    if(argc >= 2)
    {
        N_base = std::stoul(argv[1]);
    }
    if(argc >= 3)
    {
        newPhotonsPerCell = std::stoul(argv[2]);
    }
    if(argc >= 4)
    {
        minPhotonsPerCell = std::stoul(argv[3]);
    }

    const double Lx = 1.4;
    const double Ly = 0.65;
    const double Lz = 0.65;
    const double pad = 0.03;
    Vector3D lower(0, -(Ly + pad), -(Lz + pad));
    Vector3D upper(Lx + pad, Ly + pad, Lz + pad);

    boost::mt19937_64 gen(42);

    std::vector<Vector3D> points = RandRectangular(N_base, lower, upper, gen);

    size_t refineFactor = N_base / 4;
    double dx = 0.02;

    std::vector<Vector3D> extra;
    extra = RandCylinder(refineFactor, 0.45, 0.45 + dx, 0.10, 0.15, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(refineFactor, 0.45, 0.45 + dx, 0.55, 0.95, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(refineFactor, 0.60, 0.60 + dx, 0.10, 1.40, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(refineFactor, 0.65, 0.65 + dx, 0.10, 1.40, gen);
    points.insert(points.end(), extra.begin(), extra.end());
    extra = RandCylinder(refineFactor, 0, 0.65, 1.35 - dx, 1.40 + dx, gen);
    points.insert(points.end(), extra.begin(), extra.end());

    Grid grid(lower, upper);
    grid.Build(points);

    size_t Ncells = grid.GetPointNo();
    std::cout << "Hohlraum Voronoi grid: " << Ncells << " cells"
              << " (N_base=" << N_base << ", total points=" << points.size() << ")" << std::endl;

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<int> materialFlags(Ncells, 0);

    double T_init = 300.0;
    double cv_material = 3e15;
    double cv_vacuum = 1e6;

    size_t nMaterial = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetCellCM(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        bool isMat = IsMaterial(center.x, r);
        materialFlags[i] = isMat ? 1 : 0;
        nMaterial += isMat ? 1 : 0;
        cells[i].temperature = T_init;
        cells[i].cv = isMat ? cv_material : cv_vacuum;
        cells[i].internalEnergy = cells[i].cv * cells[i].temperature;
    }
    std::cout << "Material cells: " << nMaterial << ", vacuum cells: " << (Ncells - nMaterial) << std::endl;

    double T_drive = 1.0 * 1.16045e7;  // 1 keV in Kelvin

    std::shared_ptr<STORM::examples::HohlraumOpacity> opacityModel = std::make_shared<STORM::examples::HohlraumOpacity>(materialFlags);
    std::shared_ptr<STORM::SideTemperature<Vector3D, Grid>> boundary = std::make_shared<STORM::SideTemperature<Vector3D, Grid>>(grid, T_drive, newPhotonsPerCell);
    std::shared_ptr<STORM::SimpleRadiationPhysics<Vector3D, Grid>> physics = std::make_shared<STORM::SimpleRadiationPhysics<Vector3D, Grid>>(grid, boundary, cells, opacityModel, newPhotonsPerCell);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl = std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, minPhotonsPerCell, 6.0);

    STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);

    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    double dt = 1e-11;
    size_t Nsteps = 100;

    std::cout << "Running " << Nsteps << " steps with dt = " << dt << " s" << std::endl;
    std::cout << "Drive temperature: " << T_drive << " K (" << T_drive / 1.16045e7 << " keV)" << std::endl;
    std::cout << "Photons: new_per_cell=" << newPhotonsPerCell << ", min_per_cell=" << minPhotonsPerCell << std::endl;
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

    std::cout << "\nFinal temperature profile along x-axis (r~0):" << std::endl;
    double rTolerance = 0.05;
    std::vector<std::pair<double, double>> profile;
    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetCellCM(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        if(r < rTolerance)
        {
            profile.push_back({center.x, cells[i].temperature});
        }
    }
    std::sort(profile.begin(), profile.end());
    for(const std::pair<double, double> &entry : profile)
    {
        std::cout << "  x = " << entry.first * 10 << " mm: T = " << entry.second / 1.16045e7 << " keV" << std::endl;
    }

    return 0;
}
