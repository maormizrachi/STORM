#ifndef STORM_GPU_FLAT_GRID_VIEW_HPP
#define STORM_GPU_FLAT_GRID_VIEW_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "KokkosTypes.hpp"
#include "../types.hpp"

namespace STORM
{
namespace gpu
{

template<typename PointT>
struct FlatGridView
{
    const std::size_t *cellFaceOffsets = nullptr;
    const PointT *cellCenters = nullptr;
    const PointT *normals = nullptr;
    const double *facePlaneOffsets = nullptr;
    const cell_index_t *nextCellIndices = nullptr;
    const std::uint8_t *boundaryCrossings = nullptr;
    const std::uint8_t *deviceBoundaryBehaviors = nullptr;
    std::size_t cellCount = 0;
};

struct Intersection
{
    double time = DBL_MAX;
    cell_index_t nextCellIndex = 0;
    std::size_t directedFace = 0;
    std::uint8_t boundaryCrossing = 0;
    std::uint8_t valid = 0;
};

template<typename ParticleT, typename PointT>
STORM_GPU_INLINE_FUNCTION
Intersection FindIntersection(const ParticleT &particle,
                              const FlatGridView<PointT> &grid,
                              const double speed)
{
    Intersection result;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= grid.cellCount)
    {
        return result;
    }

    const double velocityTolerance = 1.0e-12 * speed;
    const std::size_t begin = grid.cellFaceOffsets[cellIndex];
    const std::size_t end = grid.cellFaceOffsets[cellIndex + 1];

    for(std::size_t directedFace = begin; directedFace < end; ++directedFace)
    {
        const PointT &normal = grid.normals[directedFace];
        const double normalVelocity = normal.x * particle.velocity.x +
                                      normal.y * particle.velocity.y +
                                      normal.z * particle.velocity.z;
        if(normalVelocity >= -velocityTolerance)
        {
            continue;
        }

        const double locationPlane =
            particle.location.x * normal.x +
            particle.location.y * normal.y +
            particle.location.z * normal.z;
        const double time =
            (grid.facePlaneOffsets[directedFace] - locationPlane) /
            normalVelocity;
        if(time > 0.0 && time < result.time)
        {
            result.time = time;
            result.nextCellIndex = grid.nextCellIndices[directedFace];
            result.directedFace = directedFace;
            result.boundaryCrossing = grid.boundaryCrossings[directedFace];
            result.valid = 1;
        }
    }
    return result;
}

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_FLAT_GRID_VIEW_HPP
