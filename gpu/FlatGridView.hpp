#ifndef STORM_GPU_FLAT_GRID_VIEW_HPP
#define STORM_GPU_FLAT_GRID_VIEW_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(STORM_CPU_VECTOR_INTERSECTION) && defined(__AVX2__) && !defined(STORM_WITH_GPU)
#include <immintrin.h>
#endif

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

    std::size_t directedFace = begin;
#if defined(STORM_CPU_VECTOR_INTERSECTION) && defined(__AVX2__) && !defined(STORM_WITH_GPU)
    if constexpr(std::is_same<decltype(particle.velocity.x), double>::value &&
                 std::is_same<decltype(particle.location.x), double>::value &&
                 std::is_same<decltype(PointT::x), double>::value)
    {
        const __m256d vx = _mm256_set1_pd(particle.velocity.x);
        const __m256d vy = _mm256_set1_pd(particle.velocity.y);
        const __m256d vz = _mm256_set1_pd(particle.velocity.z);
        const __m256d px = _mm256_set1_pd(particle.location.x);
        const __m256d py = _mm256_set1_pd(particle.location.y);
        const __m256d pz = _mm256_set1_pd(particle.location.z);
        const __m256d zero = _mm256_setzero_pd();
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d limit = _mm256_set1_pd(-velocityTolerance);
        const __m256d infinity = _mm256_set1_pd(DBL_MAX);
        for(; end - directedFace >= 4; directedFace += 4)
        {
            const PointT *n = grid.normals + directedFace;
            const __m256d nx = _mm256_setr_pd(n[0].x, n[1].x, n[2].x, n[3].x);
            const __m256d ny = _mm256_setr_pd(n[0].y, n[1].y, n[2].y, n[3].y);
            const __m256d nz = _mm256_setr_pd(n[0].z, n[1].z, n[2].z, n[3].z);
            // Keep the scalar arithmetic order; build with contraction off.
            const __m256d velocity = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(nx, vx),
                _mm256_mul_pd(ny, vy)), _mm256_mul_pd(nz, vz));
            __m256d outgoing = _mm256_cmp_pd(velocity, limit, _CMP_LT_OQ);
            if(grid.slabTransport)
                outgoing = _mm256_and_pd(outgoing, _mm256_cmp_pd(nx, zero, _CMP_NEQ_OQ));
            if(_mm256_movemask_pd(outgoing) == 0) continue;
            const __m256d plane = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(px, nx),
                _mm256_mul_pd(py, ny)), _mm256_mul_pd(pz, nz));
            // Inactive lanes use a nonzero denominator, including parallel
            // faces. They must not introduce divide-by-zero exceptions.
            const __m256d denominator = _mm256_blendv_pd(one, velocity, outgoing);
            const __m256d time = _mm256_div_pd(
                _mm256_sub_pd(_mm256_loadu_pd(grid.facePlaneOffsets + directedFace), plane), denominator);
            const __m256d valid = _mm256_and_pd(outgoing, _mm256_cmp_pd(time, zero, _CMP_GT_OQ));
            const __m256d candidates = _mm256_blendv_pd(infinity, time, valid);
            if(_mm256_movemask_pd(_mm256_cmp_pd(candidates, _mm256_set1_pd(result.time), _CMP_LT_OQ)) == 0)
                continue;
            alignas(32) double times[4];
            _mm256_store_pd(times, candidates);
            // Resolve equal-time edge/corner hits in original face order.
            for(std::size_t lane = 0; lane < 4; ++lane)
            {
                if(times[lane] < result.time)
                {
                    const std::size_t face = directedFace + lane;
                    result.time = times[lane];
                    result.nextCellIndex = grid.nextCellIndices[face];
                    result.directedFace = face;
                    result.boundaryCrossing = grid.boundaryCrossings[face];
                    result.valid = 1;
                }
            }
        }
    }
#endif
    for(; directedFace < end; ++directedFace)
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
