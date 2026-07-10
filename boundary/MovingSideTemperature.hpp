#ifndef STORM_MOVING_SIDE_TEMPERATURE_HPP
#define STORM_MOVING_SIDE_TEMPERATURE_HPP

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <array>
#include <boost/math/special_functions/pow.hpp>
#ifdef STORM_WITH_MPI
#include <mpi.h>
#endif
#include "BoundaryCondition.hpp"
#include <units/units.hpp>
#include <planck_integral/planck_integral.hpp>
#include "utils/LinearInterpolation.hpp"
#include "utils/RandomOnFace.hpp"
#include "utils/LorentzTransformation.hpp"
#include "StormError.hpp"
#include "elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

template<typename T, typename Grid>
class MovingSideTemperature : public BoundaryCondition<T, Grid>
{
public:
    MovingSideTemperature(const Grid &grid, double temperature, size_t Npercell, const std::vector<double> &energyBoundaries = {});

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

        // Left x moving thermal source. Not implemented as a DDMC boundary yet.
        if(nOut.x < -0.99)
            return DDMCBoundaryFaceBehavior::Unsupported;

        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    void SetTemperature(double temperature);

private:
    double temperature_;
    size_t Npercell_;
    bool multigroup_;
    std::vector<double> energyBoundaries_;
    std::vector<double> cumulativePlanckFunction_;

    T leftFaceVelocity_;
    double prevLeftX_;

    void RecomputePlanckCDF();
    void updateLeftFaceVelocityFromBox(double fullDt);
};

template<typename T, typename Grid>
MovingSideTemperature<T, Grid>::MovingSideTemperature(
    const Grid &grid, double temperature, size_t Npercell, const std::vector<double> &energyBoundaries)
    : BoundaryCondition<T, Grid>(grid),
      temperature_(temperature), Npercell_(Npercell),
      multigroup_(!energyBoundaries.empty()),
      energyBoundaries_(energyBoundaries),
      leftFaceVelocity_(0, 0, 0)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    prevLeftX_ = ll.x;

    if(multigroup_)
        RecomputePlanckCDF();
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::SetTemperature(double temperature)
{
    temperature_ = temperature;
    if(multigroup_)
        RecomputePlanckCDF();
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::RecomputePlanckCDF()
{
    size_t const Ngroups = this->energyBoundaries_.size() - 1;
    double const kT = units::k_boltz * temperature_;
    this->cumulativePlanckFunction_.resize(Ngroups + 1);
    this->cumulativePlanckFunction_[0] = 0.0;
    for(size_t g = 1; g <= Ngroups; ++g)
    {
        double const a = this->energyBoundaries_[g - 1] / kT;
        double const b = this->energyBoundaries_[g] / kT;
        this->cumulativePlanckFunction_[g] = this->cumulativePlanckFunction_[g - 1]
                                           + planck_integral::planck_integral(a, b);
    }
    double const total = this->cumulativePlanckFunction_.back();
    if(!(total > 0.0) || !std::isfinite(total))
    {
        STORMError eo("MovingSideTemperature: invalid Planck CDF");
        eo.addEntry("temperature", temperature_);
        eo.addEntry("total", total);
        throw eo;
    }
    for(double &x : this->cumulativePlanckFunction_)
        x /= total;
    this->cumulativePlanckFunction_.back() = 1.0;
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::updateLeftFaceVelocityFromBox(double fullDt)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    double currentLeftX = ll.x;
    if(fullDt > 0.0)
        leftFaceVelocity_ = T((currentLeftX - prevLeftX_) / fullDt, 0, 0);
    prevLeftX_ = currentLeftX;
}

template<typename T, typename Grid>
ParticleStatus MovingSideTemperature<T, Grid>::apply(Particle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    ParticleStatus result = ParticleStatus::DONE;
    for(const typename Grid::Face_T &face : faces)
    {
        T normal;
        double faceScale = 0.0;
        if(this->getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
        {
            if(std::abs(normal.x) > 0.99)
            {
                if(std::abs(particle.location.x - ll.x) < std::abs(ur.x - particle.location.x))
                    return ParticleStatus::REMOVE;
            }
            if(this->reflectParticleOnBoxFace(particle, face))
                result = ParticleStatus::REFLECT;
        }
    }
    if(result == ParticleStatus::REFLECT)
        return result;
    std::cerr << "MovingSideTemperature: particle is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>>
MovingSideTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    updateLeftFaceVelocityFromBox(fullDt);

    double const T4 = boost::math::pow<4>(temperature_);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re([](){
        int seed = 0;
#ifdef STORM_WITH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &seed);
#endif
        return static_cast<std::mt19937_64::result_type>(seed);
    }());

    std::vector<Particle<T, Grid>> newParticles;
    size_t const N = this->grid.GetPointNo();

    for(size_t i = 0; i < N; ++i)
    {
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(neighborIdx >= N && this->grid.IsPointOutsideBox(neighborIdx))
            {
                T nOut = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if(nOut.x < -0.99)
                {
                    std::cout << "Left velocity: " << leftFaceVelocity_ << ", nOut: " << nOut << std::endl;
                    double const area = this->grid.GetArea(faceIdx);

                    double const v2 = ScalarProd(leftFaceVelocity_, leftFaceVelocity_);
                    double const gammaFace = 1.0 / std::sqrt(1.0 - v2 / (units::clight * units::clight));

                    double const dtFace = fullDt / gammaFace;
                    double const packetEnergyFace = units::sigma_sb * T4 * area * dtFace / Npercell_;
                    double const fluidEnergy = packetEnergyFace * Npercell_ * gammaFace;

                    T e1(0, 1, 0);
                    T e2(0, 0, 1);
                    double totalWeight = 0.0;
                    for(size_t j = 0; j < Npercell_; ++j)
                    {
                        Particle<T, Grid> p;
                        p.location = RandomPointOnFace<T, Grid>(this->grid, faceIdx);
                        p.location -= 1e-12 * std::sqrt(area) * nOut;
                        p.weight = packetEnergyFace;
                        p.timeLeft = fullDt * unif(re);
                        p.cellIndex = i;
                        p.frequency = 0;
                        if(multigroup_)
                            p.frequency = LinearInterpolation(this->cumulativePlanckFunction_, this->energyBoundaries_, unif(re));

                        do
                        {
                            double const mu = std::sqrt(unif(re));
                            double const sinTheta = std::sqrt(1.0 - mu * mu);
                            double const phi = 2.0 * M_PI * unif(re);

                            T dirFace = (-mu) * nOut
                                      + sinTheta * std::cos(phi) * e1
                                      + sinTheta * std::sin(phi) * e2;
                            dirFace = normalize(dirFace);

                            p.velocity = units::clight * dirFace;
                            LorentzTransformation(p, -1.0 * leftFaceVelocity_,
                                multigroup_ ? &this->energyBoundaries_ : nullptr);
                            p.initialWeight = p.weight;
                        } while(p.velocity.x < 0);
                        totalWeight += p.weight;
                        if(j == 0)
                            std::cout << "Fluid energy: " << packetEnergyFace << " end weight: " << p.weight << std::endl;
                        newParticles.push_back(p);
                    }
                    std::cout << "Total weight: " << totalWeight << ", fluidEnergy: " << fluidEnergy
                              << " expected lab weight: " << fluidEnergy * (1 + 2 * leftFaceVelocity_.x / (3 * units::clight))
                              << std::endl;
                }
            }
        }
    }
    return newParticles;
}

} // namespace STORM

#endif // STORM_MOVING_SIDE_TEMPERATURE_HPP
