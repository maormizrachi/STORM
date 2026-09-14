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

    // Energy ledger. Removed energy is classified by where the packet left: the source
    // disc, the pipe exit disc, any other box face, or nowhere near a box face at all --
    // the last would be a packet silently dropped at an interior face.
    struct Ledger
    {
        double injected = 0.0, sourceDisc = 0.0, exitDisc = 0.0, sideWall = 0.0, interior = 0.0;
        std::size_t interiorCount = 0;
    };
    const Ledger &ledger() const { return ledger_; }
    void resetLedger() { ledger_ = Ledger{}; }

    void tallyRemoval(const Particle<PointT> &particle)
    {
        const auto &[lower, upper] = this->grid.GetBoxCoordinates();
        const double radius = std::sqrt(particle.location.y * particle.location.y +
                                        particle.location.z * particle.location.z);
        constexpr double tolerance = 1.0e-5;
        const bool onLowerX = std::abs(particle.location.x - lower.x) < tolerance;
        const bool onUpperX = std::abs(particle.location.x - upper.x) < tolerance;
        const bool onSide = std::abs(std::abs(particle.location.y) - upper.y) < tolerance or
                            std::abs(std::abs(particle.location.z) - upper.z) < tolerance;
        if(onLowerX and radius < 0.5) ledger_.sourceDisc += particle.weight;
        else if(onUpperX and radius < 0.5) ledger_.exitDisc += particle.weight;
        else if(onLowerX or onUpperX or onSide) ledger_.sideWall += particle.weight;
        else { ledger_.interior += particle.weight; ++ledger_.interiorCount; }
    }

    ParticleStatus apply(Particle<PointT> &particle) override
    {
        tallyRemoval(particle);
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
                ledger_.injected += particleEnergy * static_cast<double>(photonsPerFace_);
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
    Ledger ledger_;
    const std::vector<int> &materialFlags_;
    double driveTemperature_;
    std::size_t photonsPerFace_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_CROOKED_PIPE_BOUNDARY_HPP
