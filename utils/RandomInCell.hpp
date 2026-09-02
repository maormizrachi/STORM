#ifndef STORM_RANDOM_IN_CELL_HPP
#define STORM_RANDOM_IN_CELL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "../radiation/source/SourceCore.hpp"
#include "CounterRNG.hpp"

/// Volume-weighted tetrahedral decomposition of one cell, reusable across every
/// particle emitted from that cell. Depends only on the grid geometry.
struct CellVolumeDecomposition
{
    std::vector<double> cumVolumes;
    std::vector<std::array<std::size_t, 3>> tris;

    void clear()
    {
        cumVolumes.clear();
        tris.clear();
    }

    bool empty() const
    {
        return tris.empty();
    }
};

template<typename PointT, typename GridT>
struct RandomInCellPositionSampler
{
    using Decomposition = CellVolumeDecomposition;

    /// Retains `out`'s capacity so repeated calls across cells do not reallocate.
    void BuildDecomposition(const GridT &grid,
                            std::size_t cellIndex,
                            Decomposition &out) const
    {
        out.clear();
        PointT center = grid.GetMeshPoint(cellIndex);
        const auto &verts = grid.GetFacePoints();
        double totalVolume = 0;

        for(const std::size_t &faceIdx : grid.GetCellFaces(cellIndex))
        {
            const auto &fv = grid.GetPointsInFace(faceIdx);
            if(fv.size() < 3)
            {
                continue;
            }
            for(std::size_t i = 1; i + 1 < fv.size(); ++i)
            {
                PointT a = verts[fv[0]] - center;
                PointT b = verts[fv[i]] - center;
                PointT c = verts[fv[i + 1]] - center;
                double vol = std::abs(STORM::fallback::ScalarProd(a, STORM::fallback::CrossProduct(b, c)));
                totalVolume += vol;
                out.cumVolumes.push_back(totalVolume);
                out.tris.push_back({fv[0], fv[i], fv[i + 1]});
            }
        }
    }

    /// Draws exactly four values from `dist(rng)` when `decomp` is non-empty and
    /// none when it is, matching the draw order of `operator()`.
    PointT Sample(const GridT &grid,
                  std::size_t cellIndex,
                  const Decomposition &decomp,
                  std::mt19937_64 &rng,
                  std::uniform_real_distribution<double> &dist) const
    {
        PointT center = grid.GetMeshPoint(cellIndex);

        if(decomp.tris.empty())
        {
            return center;
        }

        const auto &verts = grid.GetFacePoints();

        double r = dist(rng) * decomp.cumVolumes.back();
        std::size_t idx = static_cast<std::size_t>(
            std::lower_bound(decomp.cumVolumes.begin(), decomp.cumVolumes.end(), r) -
            decomp.cumVolumes.begin());
        if(idx >= decomp.tris.size())
        {
            idx = decomp.tris.size() - 1;
        }

        const double s = dist(rng);
        const double t = dist(rng);
        const double u = dist(rng);
        const auto &tv = decomp.tris[idx];
        return STORM::source::SampleTetrahedronPosition(
            center, verts[tv[0]], verts[tv[1]], verts[tv[2]], s, t, u);
    }

    /// CounterRNG variant used by IMC host/device emission.
    PointT Sample(const GridT &grid,
                  std::size_t cellIndex,
                  const Decomposition &decomp,
                  std::uint64_t rngKey,
                  std::uint64_t &rngCounter) const
    {
        PointT center = grid.GetMeshPoint(cellIndex);
        if(decomp.tris.empty())
        {
            return center;
        }

        const auto &verts = grid.GetFacePoints();
        const double target =
            STORM::CounterRNG::unitOpen(rngKey, rngCounter++) *
            decomp.cumVolumes.back();
        std::size_t idx = static_cast<std::size_t>(
            std::lower_bound(decomp.cumVolumes.begin(), decomp.cumVolumes.end(), target) -
            decomp.cumVolumes.begin());
        if(idx >= decomp.tris.size())
        {
            idx = decomp.tris.size() - 1;
        }
        const double s = STORM::CounterRNG::unitOpen(rngKey, rngCounter++);
        const double t = STORM::CounterRNG::unitOpen(rngKey, rngCounter++);
        const double u = STORM::CounterRNG::unitOpen(rngKey, rngCounter++);
        const auto &tv = decomp.tris[idx];
        return STORM::source::SampleTetrahedronPosition(
            center, verts[tv[0]], verts[tv[1]], verts[tv[2]], s, t, u);
    }

    PointT operator()(const GridT &grid,
                      std::size_t cellIndex,
                      std::mt19937_64 &rng,
                      std::uniform_real_distribution<double> &dist) const
    {
        Decomposition decomp;
        this->BuildDecomposition(grid, cellIndex, decomp);
        return this->Sample(grid, cellIndex, decomp, rng, dist);
    }
};

namespace random_in_cell_detail {
inline std::mt19937_64 &GetRNG()
{
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng;
}
inline std::uniform_real_distribution<double> &GetDist()
{
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist;
}
} // namespace random_in_cell_detail

template<typename GridT>
typename GridT::Point_T RandomPointInCell(const GridT &grid, std::size_t cellIndex)
{
    auto &rng = random_in_cell_detail::GetRNG();
    auto &dist = random_in_cell_detail::GetDist();
    RandomInCellPositionSampler<typename GridT::Point_T, GridT> sampler;
    return sampler(grid, cellIndex, rng, dist);
}

inline void ReseedRandomInCell(uint64_t seed)
{
    random_in_cell_detail::GetRNG().seed(seed);
}

#endif // STORM_RANDOM_IN_CELL_HPP
