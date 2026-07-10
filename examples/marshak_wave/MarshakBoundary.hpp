#ifndef STORM_MARSHAK_BOUNDARY_HPP
#define STORM_MARSHAK_BOUNDARY_HPP

#include <boost/math/special_functions/pow.hpp>
#include <random>
#include <cmath>
#include "boundary/BoundaryCondition.hpp"
#include <units/units.hpp>
#include "utils/RandomOnFace.hpp"
#include "elementary/PointOps.hpp"

namespace STORM {
namespace examples {

using namespace STORM::fallback;

/*
 * Marshak wave boundary condition for 1D slab problems:
 *   x = 0 face:   Planck source at T_bath (photons entering the domain) +
 *                  escape (photons hitting this face are removed)
 *   All other faces: specular reflection
 *
 * The drive temperature can be updated between steps via SetTemperature().
 */
template<typename T, typename Grid>
class MarshakBoundary : public BoundaryCondition<T, Grid>
{
public:
    MarshakBoundary(const Grid &grid, double temperature, size_t Npercell)
        : BoundaryCondition<T, Grid>(grid), temperature_(temperature), Npercell_(Npercell)
    {}

    void SetTemperature(double temp) { temperature_ = temp; }
    double GetTemperature() const { return temperature_; }

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx, size_t insideCellIndex, size_t outsidePointIndex) const override
    {
        (void)faceIdx; (void)insideCellIndex; (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

private:
    double temperature_;
    size_t Npercell_;
};

template<typename T, typename Grid>
ParticleStatus MarshakBoundary<T, Grid>::apply(Particle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    ParticleStatus status = ParticleStatus::DONE;
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    for(const typename Grid::Face_T &face : faces)
    {
        T normal;
        double faceScale = 0.0;
        if(this->getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
        {
            if(std::abs(normal.x) > 0.99)
            {
                if(std::abs(particle.location.x - ll.x) < std::abs(ur.x - particle.location.x))
                {
                    return ParticleStatus::REMOVE;
                }
            }
            if(this->reflectParticleOnBoxFace(particle, face))
            {
                status = ParticleStatus::REFLECT;
            }
        }
    }
    if(status == ParticleStatus::REFLECT)
    {
        return status;
    }

    std::cerr << "MarshakBoundary: particle not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> MarshakBoundary<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    double T4 = boost::math::pow<4>(this->temperature_);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re(0);

    std::vector<Particle<T, Grid>> newParticles;
    size_t N = this->grid.GetPointNo();

    for(size_t i = 0; i < N; i++)
    {
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(neighborIdx >= N and this->grid.IsPointOutsideBox(neighborIdx))
            {
                T normal = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if(normal.x < -0.99)
                {
                    double energyToProduce = units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell_;
                    for(size_t j = 0; j < this->Npercell_; j++)
                    {
                        newParticles.emplace_back();
                        Particle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = RandomPointOnFace<T, Grid>(this->grid, faceIdx);
                        double mu = std::sqrt(unif(re));
                        newParticle.velocity.x = mu;
                        double _1mmu = std::sqrt(1 - mu * mu);
                        double theta = 2 * M_PI * unif(re);
                        newParticle.velocity.y = _1mmu * std::cos(theta);
                        newParticle.velocity.z = _1mmu * std::sin(theta);
                        newParticle.velocity *= units::clight;
                        newParticle.frequency = 0;
                        newParticle.weight = energyToProduce;
                        newParticle.initialWeight = newParticle.weight;
                        newParticle.timeLeft = fullDt * unif(re);
                        newParticle.cellIndex = i;
                    }
                }
            }
        }
    }
    return newParticles;
}

} // namespace examples
} // namespace STORM

#endif // STORM_MARSHAK_BOUNDARY_HPP
