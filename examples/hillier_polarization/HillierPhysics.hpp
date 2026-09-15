#ifndef STORM_HILLIER_POLARIZATION_PHYSICS_HPP
#define STORM_HILLIER_POLARIZATION_PHYSICS_HPP

// Material model for the Hillier (1994) polarized-scattering benchmark.
//
// A point source sits inside a detached, purely electron-scattering envelope with
// a prolate density distribution
//
//     sigma_e N_e(r,beta) = chi0 (R_min/r)^4 (1 + 10 cos^2 beta),
//     R_min = 2,  R_max = 30 R_min,
//
// chi0 carrying units of inverse length.  The solid-angle-averaged radial optical
// depth is then
//
//     tau_bar = chi0 <1+10cos^2 b> (R_min/3) (1 - (R_min/R_max)^3) = 2.8887 chi0,
//
// the tau_ave = 2.888 chi0 quoted by Bulla, Sim & Kromer (2015), whose Section 3.1
// reproduces this configuration.  (Their Eq. 23 prints the radial exponent as 2;
// Hillier uses 4, and only 4 reproduces their own tau_ave.)
//
// Only the source core absorbs, and therefore only the source core emits: the
// envelope is a pure scatterer, as the benchmark requires.
//
// STORM's RadiationCell carries no density, so the per-cell scattering coefficient
// is held alongside and looked up by the cell's offset into the cell vector -- the
// same idiom HohlraumOpacity uses.

#include <cmath>
#include <string>
#include <vector>

#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationOpacityModel.hpp"

namespace STORM {
namespace examples {

//! \brief Ideal-gas equation of state with a constant volumetric heat capacity.
//!
//! RadiationIMC feeds dT2cv into the Fleck factor beside a*T^4, so it must return
//! heat capacity per unit VOLUME; de2T receives energy per unit mass.  A constant
//! is enough here: the material is inert, and only the source core couples to the
//! radiation at all.
class HillierEOS
{
public:
    explicit HillierEOS(double cvPerVolume) : cvPerVolume_(cvPerVolume) {}

    double dT2cv(double, double, const std::vector<double> &,
                 const std::vector<std::string> &) const
    {
        return cvPerVolume_;
    }

    double de2T(double density, double specificEnergy, const std::vector<double> &,
                const std::vector<std::string> &) const
    {
        return specificEnergy * density / cvPerVolume_;
    }

private:
    double cvPerVolume_;
};

//! \brief Pure electron scattering in the envelope, absorption only in the core.
template<typename PointT, typename GridT>
class HillierOpacity final
    : public RadiationOpacityModel<PointT, GridT, RadiationCell, 1>
{
public:
    using RadiationOpacityModel<PointT, GridT, RadiationCell, 1>::CalcScatteringOpacity;

    //! \param scattering Per-cell sigma_e N_e [1/length], parallel to \a cells
    //! \param isCore Per-cell flag marking the emitting source core
    //! \param cells Cell vector the transport will pass back by reference
    //! \param coreAbsorption Absorption coefficient inside the source core
    HillierOpacity(const std::vector<double> &scattering,
                   const std::vector<char> &isCore,
                   const std::vector<RadiationCell> &cells,
                   double coreAbsorption)
        : scattering_(scattering), isCore_(isCore), cells_(&cells),
          coreAbsorption_(coreAbsorption) {}

    //! Emission and absorption: the core only.  A transparent envelope is what
    //! makes this a scattering benchmark rather than a thermal one.
    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t const i = cellIndex(cell);
        return (i < isCore_.size() && isCore_[i]) ? coreAbsorption_ : 0.0;
    }

    double CalcPlanckOpacityAtTemperature(const RadiationCell &cell, double) override
    {
        return CalcPlanckOpacity(cell);
    }

    double CalcAbsorptionOpacity(const RadiationCell &cell, double) override
    {
        return CalcPlanckOpacity(cell);
    }

    double CalcAbsorptionOpacityAtTemperature(const RadiationCell &cell, double,
                                              double) override
    {
        return CalcPlanckOpacity(cell);
    }

    //! Thomson scattering: sigma_e N_e straight from the tabulated envelope, so
    //! the geometry lives entirely in the initial conditions.
    double CalcScatteringOpacity(const RadiationCell &cell) override
    {
        std::size_t const i = cellIndex(cell);
        return (i < scattering_.size()) ? scattering_[i] : 0.0;
    }

    double CalcScatteringOpacity(const RadiationCell &cell, double) override
    {
        return CalcScatteringOpacity(cell);
    }

    double CalcScatteringOpacityAtTemperature(const RadiationCell &cell,
                                              double) override
    {
        return CalcScatteringOpacity(cell);
    }

    double CalcScatteringOpacityAtTemperature(const RadiationCell &cell, double,
                                              double) override
    {
        return CalcScatteringOpacity(cell);
    }

private:
    std::size_t cellIndex(const RadiationCell &cell) const
    {
        return static_cast<std::size_t>(&cell - cells_->data());
    }

    const std::vector<double> &scattering_;
    const std::vector<char> &isCore_;
    const std::vector<RadiationCell> *cells_;
    double coreAbsorption_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_HILLIER_POLARIZATION_PHYSICS_HPP
