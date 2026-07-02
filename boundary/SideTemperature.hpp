#ifndef STORM_SIDE_TEMPERATURE_HPP
#define STORM_SIDE_TEMPERATURE_HPP

#include <boost/math/special_functions/pow.hpp>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include "BoundaryCondition.hpp"
#include "../PhysicalConstants.hpp"
#include <planck_integral/planck_integral.hpp>
#include "../utils/LinearInterpolation.hpp"
#include "../utils/RandomOnFace.hpp"
#include "../StormError.hpp"
#include "../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

template<typename T, typename Grid>
class SideTemperature : public BoundaryCondition<T, Grid>
{
public:
    SideTemperature(const Grid &grid, double temperature, size_t Npercell, const std::vector<double> &energyBoundaries = {});

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        T nOut;
        if(!this->getDDMCOrientedOutwardNormal(
                faceIdx, insideCellIndex, outsidePointIndex, nOut))
            return DDMCBoundaryFaceBehavior::Unsupported;

        // Left x boundary: thermal source / removal face.
        if(nOut.x < -0.99)
            return DDMCBoundaryFaceBehavior::Unsupported;

        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    void SetTemperature(double temp) { temperature = temp; }

private:
    double temperature;
    size_t Npercell;
    std::vector<double> energyBoundaries;
    std::vector<double> cumulativePlanckFunction;
    bool multigroup;
};

template<typename T, typename Grid>
SideTemperature<T, Grid>::SideTemperature(const Grid &grid, double temperature, size_t Npercell, const std::vector<double> &energyBoundaries):
    BoundaryCondition<T, Grid>(grid), temperature(temperature), Npercell(Npercell), energyBoundaries(energyBoundaries), multigroup(!energyBoundaries.empty())
{
    if(this->multigroup)
    {
        size_t Ngroups = this->energyBoundaries.size() - 1;
        double const kT = constants::k_boltz * temperature;
        this->cumulativePlanckFunction.resize(Ngroups + 1);
        this->cumulativePlanckFunction[0] = 0.0;
        for(size_t g = 1; g <= Ngroups; g++)
        {
            double const a = this->energyBoundaries[g - 1] / kT;
            double const b = this->energyBoundaries[g] / kT;
            this->cumulativePlanckFunction[g] = planck_integral::planck_integral(a, b);
            this->cumulativePlanckFunction[g] += this->cumulativePlanckFunction[g - 1];
        }
        if(std::abs(this->cumulativePlanckFunction.back() - 1.0) > 1e-8)
        {
            STORMError eo("Cumulative Planck function does not sum to 1");
            eo.addEntry("Sum", this->cumulativePlanckFunction.back());
            throw eo;
        }
    }
}

template<typename T, typename Grid>
ParticleStatus SideTemperature<T, Grid>::apply(Particle<T, Grid> &particle)
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
                status = ParticleStatus::REFLECT;
        }
    }
    if(status == ParticleStatus::REFLECT)
        return status;

    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> SideTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    static const double T4 = boost::math::pow<4>(this->temperature);
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
                    double energyToProduce = constants::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
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
                        newParticle.velocity *= constants::clight;
                        newParticle.frequency = 0;
                        if(this->multigroup)
                        {
                            newParticle.frequency = LinearInterpolation(this->cumulativePlanckFunction, this->energyBoundaries, unif(re));
                        }
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

} // namespace STORM

#endif // STORM_SIDE_TEMPERATURE_HPP
