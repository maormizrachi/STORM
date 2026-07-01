#ifndef STORM_RANDOM_IN_CELL_HPP
#define STORM_RANDOM_IN_CELL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

template<typename PointT, typename GridT>
struct RandomInCellPositionSampler
{
    PointT operator()(const GridT &grid,
                      std::size_t cellIndex,
                      std::mt19937_64 &rng,
                      std::uniform_real_distribution<double> &dist) const
    {
        PointT center = grid.GetMeshPoint(cellIndex);

        std::vector<double> cumVolumes;
        std::vector<std::array<std::size_t, 3>> tris;
        double totalVolume = 0;

        for(const std::size_t &faceIdx : grid.GetCellFaces(cellIndex))
        {
            const auto &fv = grid.GetPointsInFace(faceIdx);
            if(fv.size() < 3)
            {
                continue;
            }
            const auto &verts = grid.GetFacePoints();
            for(std::size_t i = 1; i + 1 < fv.size(); ++i)
            {
                PointT a = verts[fv[0]] - center;
                PointT b = verts[fv[i]] - center;
                PointT c = verts[fv[i + 1]] - center;
                double vol = std::abs(ScalarProd(a, CrossProduct(b, c)));
                totalVolume += vol;
                cumVolumes.push_back(totalVolume);
                tris.push_back({fv[0], fv[i], fv[i + 1]});
            }
        }

        if(tris.empty())
        {
            return center;
        }

        const auto &verts = grid.GetFacePoints();

        double r = dist(rng) * totalVolume;
        std::size_t idx = static_cast<std::size_t>(
            std::lower_bound(cumVolumes.begin(), cumVolumes.end(), r) - cumVolumes.begin());
        if(idx >= tris.size())
        {
            idx = tris.size() - 1;
        }

        double s = dist(rng), t = dist(rng), u = dist(rng);
        if(s > t) std::swap(s, t);
        if(t > u) std::swap(t, u);
        if(s > t) std::swap(s, t);

        const auto &tv = tris[idx];
        return s * verts[tv[0]] + (t - s) * verts[tv[1]] + (u - t) * verts[tv[2]] + (1 - u) * center;
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
