#ifndef STORM_MOVING_SLAB_BOUNDARY_HPP
#define STORM_MOVING_SLAB_BOUNDARY_HPP

#include <cmath>
#include "boundary/BoundaryCondition.hpp"
#include "PhysicalConstants.hpp"
#include "elementary/PointOps.hpp"

namespace STORM {
namespace examples {

using namespace STORM::fallback;

/*
 * Boundary condition for the moving slab benchmark:
 *   x-min and x-max faces: transparent (photons escape the domain)
 *   y and z faces: specular reflection
 *
 * No boundary source — all photons originate from slab cells via
 * thermal emission.
 */
template<typename T, typename Grid>
class MovingSlabBoundary : public BoundaryCondition<T, Grid>
{
public:
    MovingSlabBoundary(const Grid &grid)
        : BoundaryCondition<T, Grid>(grid)
    {}

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double /*fullDt*/) override
    {
        return {};
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t /*faceIdx*/, size_t insideCellIndex, size_t outsidePointIndex) const override
    {
        T outward = this->grid.GetMeshPoint(outsidePointIndex) -
                    this->grid.GetMeshPoint(insideCellIndex);
        double ax = std::abs(outward.x);
        double ay = std::abs(outward.y);
        double az = std::abs(outward.z);
        if(ax > ay and ax > az)
        {
            return DDMCBoundaryFaceBehavior::Unsupported;
        }
        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }
};

template<typename T, typename Grid>
ParticleStatus MovingSlabBoundary<T, Grid>::apply(Particle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();

    double dxLeft  = std::abs(particle.location.x - ll.x);
    double dxRight = std::abs(particle.location.x - ur.x);
    double dyLo    = std::abs(particle.location.y - ll.y);
    double dyHi    = std::abs(particle.location.y - ur.y);
    double dzLo    = std::abs(particle.location.z - ll.z);
    double dzHi    = std::abs(particle.location.z - ur.z);
    double dMinYZ  = std::min({dyLo, dyHi, dzLo, dzHi});

    bool isLeftX  = (dxLeft  < dxRight) and (dxLeft  < dMinYZ);
    bool isRightX = (dxRight < dxLeft)  and (dxRight < dMinYZ);

    if(isRightX and particle.velocity.x > 0)
    {
        return ParticleStatus::REMOVE;
    }
    if(isLeftX and particle.velocity.x < 0)
    {
        return ParticleStatus::REMOVE;
    }

    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    ParticleStatus status = ParticleStatus::DONE;
    for(const typename Grid::Face_T &face : faces)
    {
        if(this->reflectParticleOnBoxFace(particle, face))
        {
            status = ParticleStatus::REFLECT;
        }
    }
    if(status == ParticleStatus::REFLECT)
    {
        return status;
    }

    std::cerr << "MovingSlabBoundary: particle not on any boundary" << std::endl;
    exit(1);
}

} // namespace examples
} // namespace STORM

#endif // STORM_MOVING_SLAB_BOUNDARY_HPP
