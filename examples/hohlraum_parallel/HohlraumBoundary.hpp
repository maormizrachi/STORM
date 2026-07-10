#ifndef STORM_HOHLRAUM_BOUNDARY_HPP
#define STORM_HOHLRAUM_BOUNDARY_HPP

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

template<typename T, typename Grid>
class HohlraumBoundary : public BoundaryCondition<T, Grid>
{
public:
    HohlraumBoundary(const Grid &grid, double temperature, size_t Npercell)
        : BoundaryCondition<T, Grid>(grid), temperature(temperature), Npercell(Npercell)
    {}

    ParticleStatus apply(Particle<T, Grid> &particle) override
    {
        return ParticleStatus::REMOVE;
    }

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

private:
    double temperature;
    size_t Npercell;
};

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> HohlraumBoundary<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    const double T4 = boost::math::pow<4>(this->temperature);
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
                if(normal.x < -0.99 and std::sqrt(point.y * point.y + point.z * point.z) < 0.65)
                {
                    double energyToProduce = units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
                    for(size_t j = 0; j < this->Npercell; j++)
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
    for(auto &p : newParticles)
    {
        if(this->grid.IsPointOutsideBox(p.location))
        {
            const T original = p.location;
            const T direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) and t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
        }
    }

    return newParticles;
}

} // namespace examples
} // namespace STORM

#endif // STORM_HOHLRAUM_BOUNDARY_HPP
