#ifndef STORM_TWO_SIDES_TEMPERATURE_HPP
#define STORM_TWO_SIDES_TEMPERATURE_HPP

#include <boost/math/special_functions/pow.hpp>
#include <random>
#include <cmath>
#include <vector>
#include "BoundaryCondition.hpp"
#include "monte/PhysicalConstants.hpp"
#include "monte/utils/PlanckIntegral.hpp"
#include "monte/utils/LinearInterpolation.hpp"
#include "monte/utils/RandomOnFace.hpp"
#include "monte/STORMError.hpp"

namespace STORM {

template<typename T, typename Grid>
class TwoSidesTemperature : public BoundaryCondition<T, Grid>
{
public:
    TwoSidesTemperature(const Grid &grid, double temperatureLeft, double temperatureRight, size_t Npercell, const std::vector<double> &energyBoundaries = {});

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

private:
    double temperatureLeft;
    double temperatureRight;
    size_t Npercell;
    std::vector<double> energyBoundaries;
    std::vector<double> cumulativePlanckFunctionLeft;
    std::vector<double> cumulativePlanckFunctionRight;
    bool multigroup;
};

template<typename T, typename Grid>
TwoSidesTemperature<T, Grid>::TwoSidesTemperature(const Grid &grid, double temperatureLeft, double temperatureRight, size_t Npercell, const std::vector<double> &energyBoundaries):
    BoundaryCondition<T, Grid>(grid), temperatureLeft(temperatureLeft), temperatureRight(temperatureRight), Npercell(Npercell), energyBoundaries(energyBoundaries), multigroup(!energyBoundaries.empty())
{
    if(this->multigroup)
    {
        size_t Ngroups = this->energyBoundaries.size() - 1;
        double const kTLeft = constants::k_boltz * temperatureLeft;
        double const kTRight = constants::k_boltz * temperatureRight;
        this->cumulativePlanckFunctionLeft.resize(Ngroups + 1);
        this->cumulativePlanckFunctionRight.resize(Ngroups + 1);
        this->cumulativePlanckFunctionLeft[0] = 0.0;
        this->cumulativePlanckFunctionRight[0] = 0.0;
        for(size_t g = 1; g <= Ngroups; g++)
        {
            double const aLeft = this->energyBoundaries[g - 1] / kTLeft;
            double const bLeft = this->energyBoundaries[g] / kTLeft;
            double const aRight = this->energyBoundaries[g - 1] / kTRight;
            double const bRight = this->energyBoundaries[g] / kTRight;
            this->cumulativePlanckFunctionLeft[g] = planck_integral::PlanckIntegral(aLeft, bLeft);
            this->cumulativePlanckFunctionLeft[g] += this->cumulativePlanckFunctionLeft[g - 1];
            this->cumulativePlanckFunctionRight[g] = planck_integral::PlanckIntegral(aRight, bRight);
            this->cumulativePlanckFunctionRight[g] += this->cumulativePlanckFunctionRight[g - 1];
        }
        if(std::abs(this->cumulativePlanckFunctionLeft.back() - 1.0) > 1e-8)
        {
            STORMError eo("Cumulative Planck function left does not sum to 1");
            eo.addEntry("Sum", this->cumulativePlanckFunctionLeft.back());
            throw eo;
        }
        if(std::abs(this->cumulativePlanckFunctionRight.back() - 1.0) > 1e-8)
        {
            STORMError eo("Cumulative Planck function right does not sum to 1");
            eo.addEntry("Sum", this->cumulativePlanckFunctionRight.back());
            throw eo;
        }
    }
}

template<typename T, typename Grid>
ParticleStatus TwoSidesTemperature<T, Grid>::apply(Particle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    size_t reflectsHad = 0;
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    for(const typename Grid::Face_T &face : faces)
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        T normal = CrossProduct(u, v);
        double absU = abs(u);
        if(std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
        {
            normal /= abs(normal);
            if(std::abs(normal.x) > 0.99)
            {
                return ParticleStatus::REMOVE;
            }
            reflectsHad++;
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
        }
    }

    if(reflectsHad >= 2)
    {
        const T &center = this->grid.GetMeshPoint(particle.cellIndex);
        constexpr double nudge = 1e-6;
        particle.location = particle.location + nudge * (center - particle.location);
    }
    if(reflectsHad > 0)
    {
        return ParticleStatus::REFLECT;
    }

    STORMError eo("Particle is not on any boundary");
    eo.addEntry("Particle", particle);
    throw eo;
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> TwoSidesTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    static const double T4_L = boost::math::pow<4>(this->temperatureLeft);
    static const double T4_R = boost::math::pow<4>(this->temperatureRight);
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
                if(std::abs(normal.x) > 0.99)
                {
                    double energyToProduce;
                    bool isLeft = false;
                    if(normal.x > 0)
                    {
                        isLeft = false;
                        energyToProduce = constants::sigma_sb * T4_R * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
                    }
                    else
                    {
                        isLeft = true;
                        energyToProduce = constants::sigma_sb * T4_L * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
                    }
                    for(size_t j = 0; j < this->Npercell; j++)
                    {
                        newParticles.emplace_back();
                        Particle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = RandomPointOnFace<T, Grid>(this->grid, faceIdx);
                        double mu = std::sqrt(unif(re));
                        newParticle.velocity.x = (normal.x > 0) ? -mu : mu;
                        double _1mmu = std::sqrt(1 - mu * mu);
                        double theta = 2 * M_PI * unif(re);
                        newParticle.steps = 0;
                        newParticle.velocity.y = _1mmu * std::cos(theta);
                        newParticle.velocity.z = _1mmu * std::sin(theta);
                        newParticle.velocity *= constants::clight;
                        newParticle.frequency = 0;
                        if(this->multigroup)
                        {
                            newParticle.frequency = LinearInterpolation((isLeft) ? this->cumulativePlanckFunctionLeft : this->cumulativePlanckFunctionRight, this->energyBoundaries, unif(re));
                        }
                        newParticle.weight = energyToProduce;
                        newParticle.initialWeight = newParticle.weight;
                        newParticle.timeLeft = fullDt * unif(re);
                        newParticle.cellIndex = i;
                        if(this->grid.IsPointOutsideBox(newParticle.location))
                        {
                            T original = newParticle.location;
                            T direction = point - original;
                            double t = 1e-8;
                            while(this->grid.IsPointOutsideBox(newParticle.location) && t < 1.0)
                            {
                                newParticle.location = original + t * direction;
                                t *= 2;
                            }
                            newParticle.location = newParticle.location + 1e-8 * (point - newParticle.location);
                        }
                    }
                }
            }
        }
    }
    return newParticles;
}

} // namespace STORM

#endif // STORM_TWO_SIDES_TEMPERATURE_HPP
