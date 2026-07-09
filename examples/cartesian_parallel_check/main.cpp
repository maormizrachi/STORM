#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"

namespace
{

using Grid = MadCart::CartesianMesh3D<Vector3D>;

std::size_t globalCellIndexFromPoint(const std::vector<double> &xEdges,
                                     double yLo,
                                     double dy,
                                     double zLo,
                                     double dz,
                                     std::size_t ny,
                                     std::size_t nz,
                                     const Vector3D &point)
{
    auto clampUniform = [](double val, double lo, double d, std::size_t n) {
        double frac = (val - lo) / d;
        if(frac < 0.0)
        {
            return std::size_t{0};
        }
        std::size_t idx = static_cast<std::size_t>(frac);
        return (idx >= n) ? n - 1 : idx;
    };

    auto it = std::upper_bound(xEdges.begin(), xEdges.end(), point.x);
    std::size_t i = 0;
    if(it == xEdges.begin())
    {
        i = 0;
    }
    else if(it == xEdges.end())
    {
        i = xEdges.size() - 2;
    }
    else
    {
        i = static_cast<std::size_t>(it - xEdges.begin()) - 1;
    }

    std::size_t j = clampUniform(point.y, yLo, dy, ny);
    std::size_t k = clampUniform(point.z, zLo, dz, nz);
    return (i * ny + j) * nz + k;
}

void globalCellIJK(std::size_t idx, std::size_t ny, std::size_t nz,
                   std::size_t &i, std::size_t &j, std::size_t &k)
{
    k = idx % nz;
    std::size_t rem = idx / nz;
    j = rem % ny;
    i = rem / ny;
}

void check(bool condition, int rank, int &failures, const std::string &message)
{
    if(condition)
    {
        return;
    }
    ++failures;
    std::cerr << "rank " << rank << ": " << message << std::endl;
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const std::size_t nx = 17;
    const std::size_t ny = 3;
    const std::size_t nz = 2;
    const double yLo = -0.25;
    const double yHi = 0.75;
    const double zLo = 0.1;
    const double zHi = 0.8;
    const double dy = (yHi - yLo) / static_cast<double>(ny);
    const double dz = (zHi - zLo) / static_cast<double>(nz);

    std::vector<double> xEdges(nx + 1);
    xEdges[0] = 0.02;
    for(std::size_t i = 1; i <= nx; ++i)
    {
        xEdges[i] = xEdges[i - 1] + 0.01 * std::pow(1.17, static_cast<double>(i - 1));
    }

    Grid grid(xEdges, yLo, yHi, ny, zLo, zHi, nz);

    std::vector<double> weights(nx * ny * nz);
    for(std::size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = 1.0 + static_cast<double>((i * 37) % 11);
    }
    grid.BuildParallel(weights);

    int failures = 0;
    const std::size_t localN = grid.GetPointNo();
    const std::size_t globalN = nx * ny * nz;

    unsigned long long localOwned = static_cast<unsigned long long>(localN);
    unsigned long long globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    check(globalOwned == globalN, rank, failures, "global owned cell count mismatch");

    unsigned long long localPhysicalBoundaryFaces = 0;
    unsigned long long localMpiGhostFaces = 0;

    for(std::size_t cell = 0; cell < localN; ++cell)
    {
        const Vector3D center = grid.GetMeshPoint(cell);
        std::size_t containing = grid.GetContainingCell(center);
        {
            std::ostringstream msg;
            msg << "owned center returned containing cell " << containing << " instead of " << cell;
            check(containing == cell, rank, failures, msg.str());
        }
        check(grid.IsPointInCell(center, cell), rank, failures, "owned center is not inside its local cell");
        check(grid.GetCellFaces(cell).size() == 6, rank, failures, "owned cell does not have six faces");

        std::size_t gi, gj, gk;
        globalCellIJK(globalCellIndexFromPoint(xEdges, yLo, dy, zLo, dz, ny, nz, center), ny, nz, gi, gj, gk);

        std::size_t expectedPhysical = 0;
        expectedPhysical += (gi == 0) ? 1 : 0;
        expectedPhysical += (gi == nx - 1) ? 1 : 0;
        expectedPhysical += (gj == 0) ? 1 : 0;
        expectedPhysical += (gj == ny - 1) ? 1 : 0;
        expectedPhysical += (gk == 0) ? 1 : 0;
        expectedPhysical += (gk == nz - 1) ? 1 : 0;

        std::size_t actualPhysical = 0;
        for(std::size_t faceIdx : grid.GetCellFaces(cell))
        {
            const auto &neighbors = grid.GetFaceNeighbors(faceIdx);
            std::size_t other = (neighbors.first == cell) ? neighbors.second : neighbors.first;
            bool physicalBoundary = grid.IsPointOutsideBox(other);
            if(physicalBoundary)
            {
                ++actualPhysical;
                ++localPhysicalBoundaryFaces;
                check(grid.BoundaryFace(faceIdx), rank, failures, "physical boundary face not reported as boundary");
            }
            else
            {
                if(other >= localN)
                {
                    ++localMpiGhostFaces;
                    check(grid.IsPointInCell(grid.GetMeshPoint(other), other), rank, failures,
                          "MPI ghost center is not inside its ghost cell");
                }
                check(!grid.BoundaryFace(faceIdx), rank, failures, "non-physical face reported as boundary");
            }
        }

        {
            std::ostringstream msg;
            msg << "physical boundary face count mismatch: expected " << expectedPhysical
                << ", got " << actualPhysical;
            check(actualPhysical == expectedPhysical, rank, failures, msg.str());
        }
    }

    for(const auto &ghostsFromRank : grid.GetGhostIndeces())
    {
        for(std::size_t ghost : ghostsFromRank)
        {
            check(!grid.IsPointOutsideBox(ghost), rank, failures, "MPI ghost reported as outside-box");
            check(grid.GetContainingCell(grid.GetMeshPoint(ghost)) == ghost, rank, failures,
                  "MPI ghost center did not map back to its local ghost index");
        }
    }

    unsigned long long globalPhysicalBoundaryFaces = 0;
    unsigned long long globalMpiGhostFaces = 0;
    MPI_Allreduce(&localPhysicalBoundaryFaces, &globalPhysicalBoundaryFaces, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMpiGhostFaces, &globalMpiGhostFaces, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const unsigned long long expectedPhysicalBoundaryFaces =
        2ULL * (ny * nz + nx * nz + nx * ny);
    check(globalPhysicalBoundaryFaces == expectedPhysicalBoundaryFaces, rank, failures,
          "global physical boundary face count mismatch");
    if(nprocs > 1)
    {
        check(globalMpiGhostFaces > 0, rank, failures, "parallel build produced no MPI ghost faces");
    }

    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(rank == 0)
    {
        if(globalFailures == 0)
        {
            std::cout << "cartesian_parallel_check PASS"
                      << " cells=" << globalOwned
                      << " physical_boundary_faces=" << globalPhysicalBoundaryFaces
                      << " mpi_ghost_faces=" << globalMpiGhostFaces
                      << std::endl;
        }
        else
        {
            std::cerr << "cartesian_parallel_check FAIL failures=" << globalFailures << std::endl;
        }
    }

    MPI_Finalize();
    return globalFailures == 0 ? 0 : 1;
}
