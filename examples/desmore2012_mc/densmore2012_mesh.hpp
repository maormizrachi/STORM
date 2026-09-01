#ifndef STORM_REGRESSION_DENSMORE2012_MESH_HPP
#define STORM_REGRESSION_DENSMORE2012_MESH_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "examples/Vector3D.hpp"

namespace densmore2012_mesh
{

constexpr double domainLength = 3.0;
constexpr double interfacePosition = 2.0;
constexpr double thinCellWidth = 0.02;
constexpr double thickCellWidth = 0.005;
constexpr double refinementRatio = 1.47394;
constexpr std::size_t thinCellCount = 100;
constexpr std::size_t refinedCellCount = 10;
constexpr std::size_t regularThickCellCount = 199;
constexpr std::size_t cellCount =
    thinCellCount + refinedCellCount + regularThickCellCount;

inline std::vector<double> BuildCellEdges()
{
    std::vector<double> edges;
    edges.reserve(cellCount + 1);
    edges.push_back(0.0);
    for(std::size_t i = 1; i <= thinCellCount; ++i)
    {
        edges.push_back(static_cast<double>(i) * thinCellWidth);
    }

    double firstRefinedWidth = thickCellWidth * (refinementRatio - 1.0) /
        (std::pow(refinementRatio, static_cast<double>(refinedCellCount)) - 1.0);
    double x = interfacePosition;
    for(std::size_t i = 0; i < refinedCellCount; ++i)
    {
        x += firstRefinedWidth * std::pow(refinementRatio, static_cast<double>(i));
        edges.push_back(x);
    }
    edges.back() = interfacePosition + thickCellWidth;
    for(std::size_t i = 1; i <= regularThickCellCount; ++i)
    {
        edges.push_back(interfacePosition + thickCellWidth +
                        static_cast<double>(i) * thickCellWidth);
    }
    edges.back() = domainLength;
    if(edges.size() != cellCount + 1)
    {
        throw std::runtime_error("Densmore mesh has the wrong edge count");
    }
    return edges;
}

inline std::vector<Vector3D> BuildVoronoiSites()
{
    std::vector<double> edges = BuildCellEdges();
    std::vector<Vector3D> points;
    points.reserve(cellCount);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        points.emplace_back(0.5 * (edges[i] + edges[i + 1]), 0.0, 0.0);
    }

    points[thinCellCount - 1].x =
        2.0 * interfacePosition - points[thinCellCount].x;
    std::size_t firstRefined = thinCellCount;
    std::size_t lastRefined =
        thinCellCount + refinedCellCount - 1;
    for(std::size_t i = firstRefined + 1; i <= lastRefined; ++i)
    {
        points[i].x = 2.0 * edges[i] - points[i - 1].x;
    }
    std::size_t firstRegularThick = lastRefined + 1;
    points[firstRegularThick].x =
        2.0 * edges[firstRegularThick] - points[lastRefined].x;

    for(std::size_t i = 0; i < points.size(); ++i)
    {
        if(!(points[i].x > 0.0 && points[i].x < domainLength) ||
           (i > 0 && !(points[i].x > points[i - 1].x)))
        {
            throw std::runtime_error(
                "Densmore Voronoi generators are not strictly ordered");
        }
    }
    return points;
}

} // namespace densmore2012_mesh

#endif // STORM_REGRESSION_DENSMORE2012_MESH_HPP
