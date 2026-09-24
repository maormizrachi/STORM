#ifndef STORM_BRANSON_HOHLRAUM_PHYSICS_HPP
#define STORM_BRANSON_HOHLRAUM_PHYSICS_HPP

// Material model and boundary for Branson's 3D hohlraum deck, reproduced in STORM.
//
// Branson (region.h, mesh.h):
//   sigma_a = opacA + opacB * T^opacC        [1/cm, T in keV]
//   material energy density = rho * CV * T   [jerk/cm^3]
//   f = 1 / (1 + dt * sigma_a * c * 4aT^3 / (CV rho))
// STORM's Fleck factor (IMCLifecycleProcess.hpp:805) is
//   f = 1 / (1 + 4 a T^3 sigma_P c dt / cv_volumetric)
// so the two agree exactly once cv_volumetric = rho * CV and the opacity matches.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <units/units.hpp>

#include "boundary/BoundaryCondition.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationOpacityModel.hpp"

namespace BransonHohlraum
{

// 1 shake = 1e-8 s, 1 jerk = 1e16 erg; keV -> K is units::kev_kelvin.
inline constexpr double shake = 1e-8;
inline constexpr double jerk = 1e16;

//! Branson's CV [jerk/(g keV)] as cgs specific heat [erg/(g K)].
inline double specificHeatCgs(double CV_branson)
{
    return CV_branson * jerk / units::kev_kelvin;
}

//! Constant-Cv gas. Branson gives every region the same CV and density, so one
//! constant covers the whole mesh; main.cpp refuses decks where that is false.
class EOS
{
  public:
    explicit EOS(double specificHeat) : specificHeat_(specificHeat) {}

    //! STORM divides the Fleck argument by this, so it must be volumetric.
    double dT2cv(double rho, double, const std::vector<double> &,
                 const std::vector<std::string> &) const
    {
        return rho * specificHeat_;
    }

    //! e is specific internal energy [erg/g].
    double de2T(double, double e, const std::vector<double> &,
                const std::vector<std::string> &) const
    {
        if(e < 0 || !std::isfinite(e))
        {
            throw std::runtime_error("Invalid material energy");
        }
        return e / specificHeat_;
    }

  private:
    double specificHeat_;
};

//! Per-region opacA + opacB * T^opacC, in 1/cm, with T converted back to keV.
template<typename PointT, typename GridT>
class Opacity : public STORM::RadiationOpacityModel<PointT, GridT, STORM::RadiationCell, 1>
{
  public:
    Opacity(const std::vector<int> &regionOfCell,
            const std::vector<STORM::RadiationCell> &cells,
            std::vector<double> opacA, std::vector<double> opacB,
            std::vector<double> opacC, std::vector<double> opacS)
        : regionOfCell_(regionOfCell), cells_(&cells), opacA_(std::move(opacA)),
          opacB_(std::move(opacB)), opacC_(std::move(opacC)), opacS_(std::move(opacS))
    {}

    // Overriding the one-argument forms would otherwise hide the frequency-taking
    // overloads the RadiationIMC concept check requires.
    using STORM::RadiationOpacityModel<PointT, GridT, STORM::RadiationCell, 1>::CalcScatteringOpacity;
    using STORM::RadiationOpacityModel<PointT, GridT, STORM::RadiationCell, 1>::CalcAbsorptionOpacity;

    double CalcPlanckOpacity(const STORM::RadiationCell &cell) override
    {
        const std::size_t r = region(cell);
        if(opacB_[r] == 0.0)
        {
            return opacA_[r];
        }
        const double keV = cell.temperature / units::kev_kelvin;
        return opacA_[r] + opacB_[r] * std::pow(keV, opacC_[r]);
    }

    double CalcAbsorptionOpacity(const STORM::RadiationCell &cell, double) override
    {
        return CalcPlanckOpacity(cell);
    }

    double CalcScatteringOpacity(const STORM::RadiationCell &cell) override
    {
        return opacS_[region(cell)];
    }

  private:
    std::size_t region(const STORM::RadiationCell &cell) const
    {
        return static_cast<std::size_t>(regionOfCell_[static_cast<std::size_t>(&cell - cells_->data())]);
    }

    const std::vector<int> &regionOfCell_;
    const std::vector<STORM::RadiationCell> *cells_;
    std::vector<double> opacA_, opacB_, opacC_, opacS_;
};

//! Branson's deck: REFLECT on bc_left (x=0) and bc_down (y=0), VACUUM elsewhere.
template<typename PointT, typename GridT>
class Boundary : public STORM::BoundaryCondition<PointT, GridT>
{
  public:
    explicit Boundary(const GridT &grid) : STORM::BoundaryCondition<PointT, GridT>(grid) {}

    STORM::ParticleStatus apply(STORM::Particle<PointT> &particle) override
    {
        for(const typename GridT::Face_T &face : this->grid.GetBoxFaces())
        {
            if(!isReflecting(face))
            {
                continue;
            }
            if(this->reflectParticleOnBoxFace(particle, face))
            {
                return STORM::ParticleStatus::REFLECT;
            }
        }
        escapedEnergy_ += particle.weight;
        return STORM::ParticleStatus::REMOVE;
    }

    bool isEscape(STORM::ParticleStatus status) const override
    {
        return status == STORM::ParticleStatus::REMOVE;
    }

    std::vector<STORM::Particle<PointT>> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t faceIdx, std::size_t, std::size_t) const override
    {
        const std::vector<typename GridT::Face_T> &faces = this->grid.GetBoxFaces();
        if(faceIdx < faces.size() && isReflecting(faces[faceIdx]))
        {
            return STORM::DDMCBoundaryFaceBehavior::ReflectingRigid;
        }
        return STORM::DDMCBoundaryFaceBehavior::Unsupported;
    }

    double getEscapedEnergy() const { return escapedEnergy_; }
    void resetEscapedEnergy() { escapedEnergy_ = 0.0; }

  private:
    //! A box face is reflecting when all of its vertices sit on x=0 or all on y=0.
    static bool isReflecting(const typename GridT::Face_T &face)
    {
        bool onX = true, onY = true;
        for(const PointT &v : face.vertices)
        {
            if(std::abs(v.x) > 1e-12) onX = false;
            if(std::abs(v.y) > 1e-12) onY = false;
        }
        return onX || onY;
    }

    double escapedEnergy_ = 0.0;
};

} // namespace BransonHohlraum

#endif
