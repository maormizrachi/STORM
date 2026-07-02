#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>
#include <numeric>
#include <mpi.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include "PhysicalConstants.hpp"
#include "radiation/SimpleRadiationPhysics.hpp"
#include "radiation/RadiationCell.hpp"
#include "population/CombPopulationControl.hpp"
#include "manager/parallel/TwoSidedMonteCarloManager.hpp"
#include "HohlraumOpacity.hpp"
#include "HohlraumBoundary.hpp"

/*
 * Parallel (MPI) 2D Cylindrical Hohlraum benchmark.
 *
 * Same physics as examples/hohlraum (matching RICH/runs/Elad_paper_hohlraum)
 * but uses MadVoro's parallel Voronoi construction and STORM's
 * TwoSidedMonteCarloManager for distributed particle transport.
 *
 * Usage:
 *   mpirun -np <N> ./hohlraum_parallel [N_base] [new_per_cell] [min_per_cell]
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
    const double pad = 2 * 0.03;
    Vector3D lower(0, -(Ly + pad), -(Lz + pad));
    Vector3D upper(Lx + pad, Ly + pad, Lz + pad);

    std::vector<Vector3D> allPoints = GenerateAllPoints(N_base, lower, upper);

    Grid grid(lower, upper);
    std::vector<Vector3D> localPoints = grid.BuildParallel(allPoints);

    size_t Ncells = grid.GetPointNo();
    size_t globalCells = 0;
    size_t localCellsSz = Ncells;
    MPI_Reduce(&localCellsSz, &globalCells, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        std::cout << "Hohlraum parallel Voronoi grid: " << globalCells << " cells across " << nprocs << " ranks"
                  << " (N_base=" << N_base << ", total points=" << allPoints.size() << ")" << std::endl;
    }

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<int> materialFlags(Ncells, 0);

    double T_init = 300.0;
    double cv_material = 3e15 / STORM::constants::kev_kelvin;
    double cv_vacuum = 1e15 / STORM::constants::kev_kelvin;

    size_t nMaterialLocal = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        Vector3D center = grid.GetCellCM(i);
        double r = std::sqrt(center.y * center.y + center.z * center.z);
        bool isMat = IsMaterial(center.x, r);
        materialFlags[i] = isMat ? 1 : 0;
        nMaterialLocal += isMat ? 1 : 0;
        cells[i].temperature = T_init;
        cells[i].cv = isMat ? cv_material : cv_vacuum;
        cells[i].internalEnergy = cells[i].cv * cells[i].temperature;
    }

    size_t nMaterialGlobal = 0;
    MPI_Reduce(&nMaterialLocal, &nMaterialGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "Material cells: " << nMaterialGlobal << ", vacuum cells: " << (globalCells - nMaterialGlobal) << std::endl;
    }

    double T_drive = 1.0 * STORM::constants::kev_kelvin;
    constexpr size_t boundaryPhotonsPerCell = 1000;

    std::shared_ptr<STORM::examples::HohlraumOpacity> opacityModel = std::make_shared<STORM::examples::HohlraumOpacity>(materialFlags);
    std::shared_ptr<STORM::examples::HohlraumBoundary<Vector3D, Grid>> boundary = std::make_shared<STORM::examples::HohlraumBoundary<Vector3D, Grid>>(grid, T_drive, boundaryPhotonsPerCell);
    std::shared_ptr<STORM::SimpleRadiationPhysics<Vector3D, Grid>> physics = std::make_shared<STORM::SimpleRadiationPhysics<Vector3D, Grid>>(grid, boundary, cells, opacityModel, newPhotonsPerCell);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl = std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, minPhotonsPerCell, 6.0);

    STORM::TwoSidedMonteCarloManager<Vector3D, Grid> manager(grid, physics, popControl, boundary, MPI_COMM_WORLD);

    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    double dt = 1e-11;
    size_t Nsteps = 100;

    if(rank == 0)
    {
        std::cout << "Running " << Nsteps << " steps with dt = " << dt << " s" << std::endl;
        std::cout << "Drive temperature: " << T_drive << " K (" << T_drive / STORM::constants::kev_kelvin << " keV)" << std::endl;
        std::cout << "Photons: new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << ", min_per_cell=" << minPhotonsPerCell << std::endl;
        std::cout << std::endl;
    }

    for(size_t step = 0; step < Nsteps; step++)
    {
        particles = manager.step(std::move(particles), dt);

        if(step % 10 == 0 or step == Nsteps - 1)
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
                double avgT = globalSumT / globalCells;
                std::cout << "Step " << step << ": " << globalParticles << " particles, "
                          << "max T = " << globalMaxT / STORM::constants::kev_kelvin << " keV, "
                          << "avg T = " << avgT / STORM::constants::kev_kelvin << " keV" << std::endl;
            }
        }
    }

    if(rank == 0)
    {
        std::cout << "\nDone." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
