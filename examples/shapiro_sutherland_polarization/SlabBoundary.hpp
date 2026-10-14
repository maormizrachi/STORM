#ifndef STORM_SHAPIRO_SUTHERLAND_SLAB_BOUNDARY_HPP
#define STORM_SHAPIRO_SUTHERLAND_SLAB_BOUNDARY_HPP

// Boundary of the plane-parallel slab: mirrors on the side faces (y, z), so the
// slab is laterally infinite, and escapes through the top and bottom (x).
//
// A mirror maps the packet onto the image of a plane-parallel field, which is
// the field itself: the velocity component normal to the wall flips, the
// polarization basis is mirrored with it, and because a reflection reverses
// orientation the position angle changes sign, so Stokes U flips while Q stays.

#include <cmath>

#include "boundary/BoundaryCondition.hpp"
#include "StormError.hpp"

namespace STORM {
namespace examples {

template<typename PointT, typename GridT>
class SlabBoundary final : public BoundaryCondition<PointT, GridT>
{
public:
    explicit SlabBoundary(const GridT &grid) : BoundaryCondition<PointT, GridT>(grid) {}

    ParticleStatus apply(Particle<PointT> &particle) override
    {
        const auto &[ll, ur] = this->grid.GetBoxCoordinates();
        bool reflected = false;
        reflected |= mirror(particle, &PointT::y, ll.y, ur.y);
        reflected |= mirror(particle, &PointT::z, ll.z, ur.z);
        if(reflected)
            return ParticleStatus::REFLECT;
        double const tolerance = 1e-9 * (ur.x - ll.x);
        if(!(std::abs(particle.location.x - ll.x) <= tolerance || std::abs(particle.location.x - ur.x) <= tolerance))
        {
            STORMError eo("SlabBoundary: particle is not on a box face");
            eo.addEntry("Particle", particle);
            throw eo;
        }
        return ParticleStatus::REMOVE;   // out through the top or the bottom
    }

    bool isEscape(ParticleStatus status) const override { return status == ParticleStatus::REMOVE; }

    std::vector<Particle<PointT>> generateNewBoundaryParticles(double) override { return {}; }

private:
    // Mirror on the wall at lo or hi along one lateral axis, if the packet is on
    // it; nudge it back inside as BoundaryCondition::reflectParticleOnBoxFace does.
    static bool mirror(Particle<PointT> &p, double PointT::*axis, double lo, double hi)
    {
        double const width = hi - lo;
        double const tolerance = 1e-9 * width;
        double const nudge = 1e-6 * width;
        double &x = p.location.*axis;
        double sign = 0.0;   // inward normal along the axis
        if(std::abs(x - lo) <= tolerance)
            sign = 1.0;
        else if(std::abs(x - hi) <= tolerance)
            sign = -1.0;
        else
            return false;
        x = (sign > 0.0 ? lo : hi) + sign * nudge;
        if(p.velocity.*axis * sign < 0.0)
        {
            p.velocity.*axis = -(p.velocity.*axis);
#ifdef MONTECARLO_POLARIZATION
            p.polarizationBasis.*axis = -(p.polarizationBasis.*axis);
            p.stokesU = -p.stokesU;
#endif
        }
        return true;
    }
};

} // namespace examples
} // namespace STORM

#endif // STORM_SHAPIRO_SUTHERLAND_SLAB_BOUNDARY_HPP
