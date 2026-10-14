#ifndef STORM_RANDOM_ON_FACE_HPP
#define STORM_RANDOM_ON_FACE_HPP

#include <random>
#include <vector>
#include <array>
#include <algorithm>
#include "../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

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
    static std::vector<T> cachedVertices;
    static double totalArea = 0;

    // The address and build generation alone do not identify a grid: a new grid
    // can reuse both.  The face's vertices confirm the cached triangulation.
    const auto &faceVertexIndices = grid.GetPointsInFace(faceIndex);
    const auto &facePoints = grid.GetFacePoints();
    bool sameVertices = cachedVertices.size() == faceVertexIndices.size();
    for(size_t i = 0; sameVertices && i < faceVertexIndices.size(); i++)
    {
        const T &v = facePoints[faceVertexIndices[i]];
        sameVertices = v.x == cachedVertices[i].x && v.y == cachedVertices[i].y && v.z == cachedVertices[i].z;
    }

    if(faceIndex != cachedFace or &grid != cachedGrid or grid.GetBuildGeneration() != cachedBuildGeneration or
       not sameVertices)
    {
        cachedFace = faceIndex;
        cachedGrid = &grid;
        cachedBuildGeneration = grid.GetBuildGeneration();
        cachedVertices.clear();
        for(size_t i = 0; i < faceVertexIndices.size(); i++)
            cachedVertices.push_back(facePoints[faceVertexIndices[i]]);

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

} // namespace STORM

#endif // STORM_RANDOM_ON_FACE_HPP
