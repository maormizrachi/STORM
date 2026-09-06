#ifndef STORM_GPU_FLAT_GRID_VIEW_HPP
#define STORM_GPU_FLAT_GRID_VIEW_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

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
    const cell_id_t *cellIDs = nullptr;
    const PointT *normals = nullptr;
    const double *facePlaneOffsets = nullptr;
    const cell_index_t *nextCellIndices = nullptr;
    const std::uint8_t *boundaryCrossings = nullptr;
    const std::uint8_t *deviceBoundaryBehaviors = nullptr;
    std::size_t cellCount = 0;
    std::uint8_t slabTransport = 0;
    double slabLowerY = 0.0, slabUpperY = 0.0;
    double slabLowerZ = 0.0, slabUpperZ = 0.0;
};

// Triangle-wave folding is the exact specular trajectory through any number
// of transverse reflections. At a wall, choose the inward outgoing velocity.
STORM_GPU_INLINE_FUNCTION
void FoldSlabCoordinate(double &position, double &velocity,
                        const double lower, const double upper)
{
    const double width = upper - lower;
    const double unfolded = position - lower;
    const double magnitude = unfolded < 0.0 ? -unfolded : unfolded;
    const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
                             (magnitude > width ? magnitude : width);
#ifdef STORM_WITH_GPU
    double offset = Kokkos::fmod(position - lower, 2.0 * width);
#else
    double offset = std::fmod(position - lower, 2.0 * width);
#endif
    if(offset < 0.0) offset += 2.0 * width;
    if(offset >= width)
    {
        position = upper - (offset - width);
        velocity = -velocity;
    }
    else
    {
        position = lower + offset;
    }
    if(position - lower <= tolerance)
    {
        position = lower;
        if(velocity < 0.0) velocity = -velocity;
    }
    if(upper - position <= tolerance)
    {
        position = upper;
        if(velocity > 0.0) velocity = -velocity;
    }
}

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
Intersection FindIntersection(const ParticleT &particle, const FlatGridView<PointT> &grid, const double speed)
{
    Intersection result;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= grid.cellCount)
    {
        return result;
    }
    // A direction parallel to the slab has no axial intersection. It still
    // has a valid scattering/census event, possibly at DBL_MAX face time.
    result.valid = grid.slabTransport;

    const double velocityTolerance = 1.0e-12 * speed;
    const std::size_t begin = grid.cellFaceOffsets[cellIndex];
    const std::size_t end = grid.cellFaceOffsets[cellIndex + 1];

    for(std::size_t directedFace = begin; directedFace < end; ++directedFace)
    {
        const PointT &normal = grid.normals[directedFace];
        if(grid.slabTransport && normal.x == 0.0)
        {
            continue;
        }
        const double normalVelocity = normal.x * particle.velocity.x +
                                      normal.y * particle.velocity.y +
                                      normal.z * particle.velocity.z;
        if(normalVelocity >= -velocityTolerance)
        {
            continue;
        }

        const double locationPlane = particle.location.x * normal.x + particle.location.y * normal.y + particle.location.z * normal.z;
        const double time = (grid.facePlaneOffsets[directedFace] - locationPlane) / normalVelocity;
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
