#ifndef STORM_CROOKED_PIPE_BOUNDARY_HPP
#define STORM_CROOKED_PIPE_BOUNDARY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <boost/math/special_functions/pow.hpp>
#include <units/units.hpp>

#include "boundary/BoundaryCondition.hpp"
#include "elementary/PointOps.hpp"
#include "utils/RandomOnFace.hpp"

namespace STORM {
namespace examples {

using namespace STORM::fallback;

template<typename PointT, typename GridT>
class CrookedPipeBoundary : public BoundaryCondition<PointT, GridT>
{
public:
    CrookedPipeBoundary(const GridT &grid, const std::vector<int> &materialFlags, double driveTemperature, std::size_t photonsPerFace)
        : BoundaryCondition<PointT, GridT>(grid),
          materialFlags_(materialFlags),
          driveTemperature_(driveTemperature),
          photonsPerFace_(photonsPerFace)
    {}

    ParticleStatus apply(Particle<PointT> &particle) override
    {
        (void) particle;
        return ParticleStatus::REMOVE;
    }

    bool isEscape(ParticleStatus status) const override
    {
        return status == ParticleStatus::REMOVE;
    }

    std::vector<Particle<PointT>> generateNewBoundaryParticles(double fullDt) override
    {
        constexpr double boundaryNudge = 1.0e-8;
        const double temperatureFourth = boost::math::pow<4>(driveTemperature_);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        static std::mt19937_64 generator(0);
        std::vector<Particle<PointT>> particles;
        std::size_t cellCount = this->grid.GetPointNo();

        for(std::size_t i = 0; i < cellCount; ++i)
        {
            if(materialFlags_[i] != 1)
            {
                continue;
            }
            const PointT &point = this->grid.GetMeshPoint(i);
            const auto &cellFaces = this->grid.GetCellFaces(i);
            for(std::size_t faceIndex : cellFaces)
            {
                const std::pair<std::size_t, std::size_t> &neighbors = this->grid.GetFaceNeighbors(faceIndex);
                std::size_t neighborIndex = neighbors.first == i ? neighbors.second : neighbors.first;
                if(neighborIndex < cellCount or !this->grid.IsPointOutsideBox(neighborIndex))
                {
                    continue;
                }
                PointT normal = normalize(this->grid.GetMeshPoint(neighborIndex) - point);
                if(normal.x >= -0.99)
                {
                    continue;
                }

                double particleEnergy = units::sigma_sb * temperatureFourth * this->grid.GetArea(faceIndex) * fullDt /
                                        static_cast<double>(photonsPerFace_);
                for(std::size_t j = 0; j < photonsPerFace_; ++j)
                {
                    particles.emplace_back();
                    Particle<PointT> &particle = particles.back();
                    particle.location = RandomPointOnFace<PointT, GridT>(this->grid, faceIndex);
                    particle.location = particle.location * (1.0 - boundaryNudge) +
                                        boundaryNudge * this->grid.GetMeshPoint(i);
                    double mu = std::sqrt(uniform(generator));
                    double transverse = std::sqrt(1.0 - mu * mu);
                    double angle = 2.0 * std::acos(-1.0) * uniform(generator);
                    particle.velocity.x = mu;
                    particle.velocity.y = transverse * std::cos(angle);
                    particle.velocity.z = transverse * std::sin(angle);
                    particle.velocity *= units::clight;
                    particle.frequency = 0.0;
                    particle.weight = particleEnergy;
                    particle.initialWeight = particleEnergy;
                    particle.timeLeft = fullDt * uniform(generator);
                    particle.cellIndex = i;
                }
            }
        }
        return particles;
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(std::size_t, std::size_t insideCellIndex, std::size_t outsidePointIndex) const override
    {
        (void) insideCellIndex;
        (void) outsidePointIndex;
        return DDMCBoundaryFaceBehavior::Unsupported;
    }

private:
    const std::vector<int> &materialFlags_;
    double driveTemperature_;
    std::size_t photonsPerFace_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_CROOKED_PIPE_BOUNDARY_HPP
