#ifndef RDMONT_RANDOM_ON_FACE_HPP
#define RDMONT_RANDOM_ON_FACE_HPP

#include <random>
#include <vector>
#include <array>
#include <algorithm>

namespace RDMont {

template<typename T, typename Grid>
T RandomPointOnFace(const Grid &grid, size_t faceIndex)
{
    static std::mt19937 re(0);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    static size_t cachedFace = SIZE_MAX;
    static const Grid *cachedGrid = nullptr;
    static size_t cachedBuildGeneration = SIZE_MAX;
    static std::vector<double> cumAreas;
    static std::vector<std::array<size_t, 3>> tris;
    static double totalArea = 0;

    if(faceIndex != cachedFace or &grid != cachedGrid or grid.GetBuildGeneration() != cachedBuildGeneration)
    {
        cachedFace = faceIndex;
        cachedGrid = &grid;
        cachedBuildGeneration = grid.GetBuildGeneration();

        cumAreas.clear();
        tris.clear();
        totalArea = 0;

        const auto &fv = grid.GetPointsInFace(faceIndex);
        const auto &verts = grid.GetFacePoints();
        for(size_t i = 1; i + 1 < fv.size(); i++)
        {
            double area = abs(CrossProduct(verts[fv[i]] - verts[fv[0]], verts[fv[i + 1]] - verts[fv[0]]));
            totalArea += area;
            cumAreas.push_back(totalArea);
            tris.push_back({fv[0], fv[i], fv[i + 1]});
        }
    }

    const auto &verts = grid.GetFacePoints();

    double r = dist(re) * totalArea;
    size_t idx = static_cast<size_t>(std::lower_bound(cumAreas.begin(), cumAreas.end(), r) - cumAreas.begin());
    if(idx >= tris.size())
    {
        idx = tris.size() - 1;
    }

    double r1 = dist(re), r2 = dist(re);
    if(r1 + r2 > 1)
    {
        r1 = 1 - r1;
        r2 = 1 - r2;
    }

    const auto &tv = tris[idx];
    return (1 - r1 - r2) * verts[tv[0]] + r1 * verts[tv[1]] + r2 * verts[tv[2]];
}

} // namespace RDMont

#endif // RDMONT_RANDOM_ON_FACE_HPP
