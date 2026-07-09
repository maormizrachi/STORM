#ifndef STORM_DENSMORE_BOUNDARY_HPP
#define STORM_DENSMORE_BOUNDARY_HPP

#include <array>
#include <cmath>
#include <random>
#include <vector>
#include <boost/math/special_functions/pow.hpp>
#include "boundary/BoundaryCondition.hpp"
#include "PhysicalConstants.hpp"
#include "utils/RandomOnFace.hpp"
#include "elementary/PointOps.hpp"
#include "DensmoreOpacity.hpp"
#include <planck_integral/planck_integral.hpp>

namespace STORM {
namespace examples {

using namespace STORM::fallback;

/*
 * Densmore boundary condition for 1D slab problems:
 *   x = 0 face: Planck source at T_drive with frequency sampling
 *               (photons hitting this face escape)
 *   All other faces: specular reflection
 */
template<typename T, typename Grid>
class DensmoreBoundary : public BoundaryCondition<T, Grid>
{
public:
    using GroupBoundaries = std::array<double, N_DENSMORE_GROUPS + 1>;

    DensmoreBoundary(const Grid &grid, double temperature, size_t Npercell,
                     const GroupBoundaries &boundaries)
        : BoundaryCondition<T, Grid>(grid),
          temperature_(temperature), Npercell_(Npercell),
          boundaries_(boundaries)
    {}

    void SetTemperature(double temp) { temperature_ = temp; }
    double GetTemperature() const { return temperature_; }

    ParticleStatus apply(Particle<T, Grid> &particle) override
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

        std::cerr << "DensmoreBoundary: particle not on any boundary" << std::endl;
        exit(1);
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx, size_t insideCellIndex, size_t outsidePointIndex) const override
    {
        (void)faceIdx; (void)insideCellIndex; (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override
    {
        double T4 = boost::math::pow<4>(temperature_);
        std::uniform_real_distribution<double> unif(0, 1);
        static std::mt19937_64 re(0);

        double kT = constants::k_boltz * temperature_;
        std::array<double, N_DENSMORE_GROUPS + 1> cdf{};
        cdf[0] = 0.0;
        for(size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double a = boundaries_[g] / kT;
            double b = boundaries_[g + 1] / kT;
            double bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 0.0;
            cdf[g + 1] = cdf[g] + bg;
        }
        double total = cdf[N_DENSMORE_GROUPS];

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
                        double energyToProduce = constants::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / Npercell_;
                        for(size_t j = 0; j < Npercell_; j++)
                        {
                            newParticles.emplace_back();
                            Particle<T, Grid> &p = newParticles.back();
                            p.location = RandomPointOnFace<T, Grid>(this->grid, faceIdx);
                            double mu = std::sqrt(unif(re));
                            p.velocity.x = mu;
                            double sinMu = std::sqrt(1 - mu * mu);
                            double theta = 2 * M_PI * unif(re);
                            p.velocity.y = sinMu * std::cos(theta);
                            p.velocity.z = sinMu * std::sin(theta);
                            p.velocity *= constants::clight;
                            if(total > 0.0)
                            {
                                double r = unif(re) * total;
                                size_t g = 0;
                                while(g + 1 < N_DENSMORE_GROUPS && cdf[g + 1] < r)
                                {
                                    ++g;
                                }
                                p.frequency = boundaries_[g] + unif(re) * (boundaries_[g + 1] - boundaries_[g]);
                            }
                            else
                            {
                                p.frequency = 0.5 * (boundaries_[0] + boundaries_[N_DENSMORE_GROUPS]);
                            }
                            p.weight = energyToProduce;
                            p.initialWeight = p.weight;
                            p.timeLeft = fullDt * unif(re);
                            p.cellIndex = i;
                        }
                    }
                }
            }
        }
        return newParticles;
    }

private:
    double temperature_;
    size_t Npercell_;
    GroupBoundaries boundaries_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_DENSMORE_BOUNDARY_HPP
