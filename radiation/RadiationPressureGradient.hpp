#ifndef STORM_RADIATION_PRESSURE_GRADIENT_HPP
#define STORM_RADIATION_PRESSURE_GRADIENT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace STORM::radiation_pressure_gradient_detail {

template<typename PointT>
inline double dot3(const PointT &left, const PointT &right)
{
    return left[0] * right[0] + left[1] * right[1] +
           left[2] * right[2];
}

template<typename PointT>
inline PointT zeroPoint()
{
    PointT result{};
    result[0] = 0.0;
    result[1] = 0.0;
    result[2] = 0.0;
    return result;
}

template<typename GridT, typename = void>
struct HasCellCentroid : std::false_type {};

template<typename GridT>
struct HasCellCentroid<GridT, std::void_t<decltype(
    std::declval<const GridT &>().GetCellCM(std::declval<std::size_t>()))>>
    : std::true_type {};

template<typename PointT, typename GridT>
inline PointT cellSamplePoint(const GridT &grid, std::size_t cellIndex)
{
    if constexpr(HasCellCentroid<GridT>::value)
        return grid.GetCellCM(cellIndex);
    else
        return grid.GetMeshPoint(cellIndex);
}

/*
 * Reconstruct a scalar gradient from directional differences without any
 * dynamic allocation.  The orthonormal basis makes the solve rank-aware, so
 * the same code handles full 3-D meshes and lower-dimensional slab/plane
 * meshes embedded at an arbitrary orientation in 3-D.
 */
template<typename PointT>
class DirectionalGradientAccumulator
{
public:
    DirectionalGradientAccumulator():
        rhs_(zeroPoint<PointT>()),
        basis_{{zeroPoint<PointT>(),
                zeroPoint<PointT>(),
                zeroPoint<PointT>()}}
    {}

    void addDifference(const PointT &displacement, double valueDifference)
    {
        double const distanceSquared = dot3(displacement, displacement);
        if(!(distanceSquared > 0.0) || !std::isfinite(distanceSquared) ||
           !std::isfinite(valueDifference))
            return;

        double const inverseDistance = 1.0 / std::sqrt(distanceSquared);
        PointT direction = zeroPoint<PointT>();
        for(std::size_t component = 0; component < 3; ++component)
            direction[component] = displacement[component] * inverseDistance;
        double const directionalSlope = valueDifference * inverseDistance;

        double const nx = direction[0];
        double const ny = direction[1];
        double const nz = direction[2];
        this->matrix_[0] += nx * nx;
        this->matrix_[1] += nx * ny;
        this->matrix_[2] += nx * nz;
        this->matrix_[3] += ny * ny;
        this->matrix_[4] += ny * nz;
        this->matrix_[5] += nz * nz;
        this->rhs_[0] += nx * directionalSlope;
        this->rhs_[1] += ny * directionalSlope;
        this->rhs_[2] += nz * directionalSlope;

        if(this->rank_ < 3)
            this->addBasisDirection(direction);
    }

    PointT solve() const
    {
        PointT gradient = zeroPoint<PointT>();
        if(this->rank_ == 0)
            return gradient;

        std::array<std::array<double, 3>, 3> projected{};
        std::array<double, 3> projectedRhs{};
        for(std::size_t row = 0; row < this->rank_; ++row)
        {
            PointT const matrixBasis = this->applyMatrix(this->basis_[row]);
            projectedRhs[row] = dot3(this->basis_[row], this->rhs_);
            for(std::size_t column = 0; column < this->rank_; ++column)
                projected[row][column] =
                    dot3(this->basis_[column], matrixBasis);
        }

        std::array<double, 3> coefficients{};
        bool solved = false;
        if(this->rank_ == 3)
            solved = solveThreeDimensional(projected, projectedRhs,
                                           coefficients);
        if(!solved && this->rank_ >= 2)
        {
            coefficients.fill(0.0);
            solved = solveBestPlane(projected, projectedRhs, this->rank_,
                                    coefficients);
        }
        if(!solved)
        {
            coefficients.fill(0.0);
            solveBestLine(projected, projectedRhs, this->rank_, coefficients);
        }

        for(std::size_t basisIndex = 0; basisIndex < this->rank_; ++basisIndex)
        {
            for(std::size_t component = 0; component < 3; ++component)
            {
                gradient[component] +=
                    coefficients[basisIndex] * this->basis_[basisIndex][component];
            }
        }
        return gradient;
    }

private:
    static constexpr double solveTolerance()
    {
        return 256.0 * std::numeric_limits<double>::epsilon();
    }

    void addBasisDirection(const PointT &direction)
    {
        PointT candidate = direction;
        // A second orthogonalization pass prevents loss of rank on skew meshes.
        for(int pass = 0; pass < 2; ++pass)
        {
            for(std::size_t basisIndex = 0; basisIndex < this->rank_; ++basisIndex)
            {
                double const projection =
                    dot3(candidate, this->basis_[basisIndex]);
                for(std::size_t component = 0; component < 3; ++component)
                {
                    candidate[component] -=
                        projection * this->basis_[basisIndex][component];
                }
            }
        }

        double const normSquared = dot3(candidate, candidate);
        constexpr double basisToleranceSquared = 1.0e-24;
        if(!(normSquared > basisToleranceSquared) ||
           !std::isfinite(normSquared))
            return;

        double const inverseNorm = 1.0 / std::sqrt(normSquared);
        for(std::size_t component = 0; component < 3; ++component)
        {
            this->basis_[this->rank_][component] =
                candidate[component] * inverseNorm;
        }
        ++this->rank_;
    }

    PointT applyMatrix(const PointT &vector) const
    {
        PointT result = zeroPoint<PointT>();
        result[0] = this->matrix_[0] * vector[0] +
                    this->matrix_[1] * vector[1] +
                    this->matrix_[2] * vector[2];
        result[1] = this->matrix_[1] * vector[0] +
                    this->matrix_[3] * vector[1] +
                    this->matrix_[4] * vector[2];
        result[2] = this->matrix_[2] * vector[0] +
                    this->matrix_[4] * vector[1] +
                    this->matrix_[5] * vector[2];
        return result;
    }

    static bool solveThreeDimensional(
        const std::array<std::array<double, 3>, 3> &matrix,
        const std::array<double, 3> &rhs,
        std::array<double, 3> &solution)
    {
        double const xx = matrix[0][0];
        double const xy = matrix[0][1];
        double const xz = matrix[0][2];
        double const yy = matrix[1][1];
        double const yz = matrix[1][2];
        double const zz = matrix[2][2];
        double const determinant = xx * (yy * zz - yz * yz) -
            xy * (xy * zz - yz * xz) +
            xz * (xy * yz - yy * xz);
        double const scale = std::max({std::abs(xx), std::abs(xy),
            std::abs(xz), std::abs(yy), std::abs(yz), std::abs(zz)});
        if(!(scale > 0.0) || !std::isfinite(determinant) ||
           std::abs(determinant) <= solveTolerance() * scale * scale * scale)
            return false;

        solution[0] = ((yy * zz - yz * yz) * rhs[0] +
                       (xz * yz - xy * zz) * rhs[1] +
                       (xy * yz - xz * yy) * rhs[2]) / determinant;
        solution[1] = ((xz * yz - xy * zz) * rhs[0] +
                       (xx * zz - xz * xz) * rhs[1] +
                       (xy * xz - xx * yz) * rhs[2]) / determinant;
        solution[2] = ((xy * yz - xz * yy) * rhs[0] +
                       (xy * xz - xx * yz) * rhs[1] +
                       (xx * yy - xy * xy) * rhs[2]) / determinant;
        return std::isfinite(solution[0]) && std::isfinite(solution[1]) &&
               std::isfinite(solution[2]);
    }

    static bool solveBestPlane(
        const std::array<std::array<double, 3>, 3> &matrix,
        const std::array<double, 3> &rhs,
        std::size_t rank,
        std::array<double, 3> &solution)
    {
        std::size_t bestFirst = 0;
        std::size_t bestSecond = 1;
        double bestQuality = -1.0;
        for(std::size_t first = 0; first < rank; ++first)
        {
            for(std::size_t second = first + 1; second < rank; ++second)
            {
                double const diagonalScale =
                    std::max(std::abs(matrix[first][first]),
                             std::abs(matrix[second][second]));
                if(!(diagonalScale > 0.0))
                    continue;
                double const determinant =
                    matrix[first][first] * matrix[second][second] -
                    matrix[first][second] * matrix[first][second];
                double const quality =
                    std::abs(determinant) / (diagonalScale * diagonalScale);
                if(quality > bestQuality)
                {
                    bestQuality = quality;
                    bestFirst = first;
                    bestSecond = second;
                }
            }
        }
        if(!(bestQuality > solveTolerance()))
            return false;

        double const determinant =
            matrix[bestFirst][bestFirst] * matrix[bestSecond][bestSecond] -
            matrix[bestFirst][bestSecond] * matrix[bestFirst][bestSecond];
        solution[bestFirst] =
            (rhs[bestFirst] * matrix[bestSecond][bestSecond] -
             rhs[bestSecond] * matrix[bestFirst][bestSecond]) / determinant;
        solution[bestSecond] =
            (rhs[bestSecond] * matrix[bestFirst][bestFirst] -
             rhs[bestFirst] * matrix[bestFirst][bestSecond]) / determinant;
        return std::isfinite(solution[bestFirst]) &&
               std::isfinite(solution[bestSecond]);
    }

    static void solveBestLine(
        const std::array<std::array<double, 3>, 3> &matrix,
        const std::array<double, 3> &rhs,
        std::size_t rank,
        std::array<double, 3> &solution)
    {
        std::size_t best = 0;
        for(std::size_t candidate = 1; candidate < rank; ++candidate)
        {
            if(matrix[candidate][candidate] > matrix[best][best])
                best = candidate;
        }
        if(matrix[best][best] > solveTolerance())
            solution[best] = rhs[best] / matrix[best][best];
    }

    // Symmetric matrix storage: xx, xy, xz, yy, yz, zz.
    std::array<double, 6> matrix_{};
    PointT rhs_;
    std::array<PointT, 3> basis_;
    std::size_t rank_ = 0;
};

template<typename PointT, typename GridT, typename ValuesT>
PointT reconstructRadiationEnergyGradient(const GridT &grid,
                                          const ValuesT &values,
                                          std::size_t cellIndex)
{
    DirectionalGradientAccumulator<PointT> accumulator;
    if(cellIndex >= values.size())
        return zeroPoint<PointT>();

    PointT const cellPoint = cellSamplePoint<PointT>(grid, cellIndex);
    for(std::size_t faceIndex : grid.GetCellFaces(cellIndex))
    {
        auto const &faceNeighbors = grid.GetFaceNeighbors(faceIndex);
        std::size_t neighborIndex = std::numeric_limits<std::size_t>::max();
        if(faceNeighbors.first == cellIndex)
            neighborIndex = faceNeighbors.second;
        else if(faceNeighbors.second == cellIndex)
            neighborIndex = faceNeighbors.first;

        if(neighborIndex >= values.size() ||
           grid.IsPointOutsideBox(neighborIndex))
            continue;

        accumulator.addDifference(
            cellSamplePoint<PointT>(grid, neighborIndex) - cellPoint,
            values[neighborIndex] - values[cellIndex]);
    }
    return accumulator.solve();
}

} // namespace STORM::radiation_pressure_gradient_detail

#endif // STORM_RADIATION_PRESSURE_GRADIENT_HPP
