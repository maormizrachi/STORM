#ifndef STORM_RIGID_BOUNDARY_HPP
#define STORM_RIGID_BOUNDARY_HPP

#include "BoundaryCondition.hpp"
#include "../StormError.hpp"

namespace STORM {

template<typename T, typename Grid>
class RigidBoundary : public BoundaryCondition<T, Grid>
{
public:
    RigidBoundary(const Grid &grid);

    ~RigidBoundary() override;

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }
};

template<typename T, typename Grid>
RigidBoundary<T, Grid>::RigidBoundary(const Grid &grid)
    : BoundaryCondition<T, Grid>(grid)
{}

template<typename T, typename Grid>
RigidBoundary<T, Grid>::~RigidBoundary()
{}

template<typename T, typename Grid>
ParticleStatus RigidBoundary<T, Grid>::apply(Particle<T, Grid> &particle)
{
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    ParticleStatus status = ParticleStatus::DONE;
    for(const typename Grid::Face_T &face : faces)
    {
        if(this->reflectParticleOnBoxFace(particle, face))
            status = ParticleStatus::REFLECT;
    }
    if(status == ParticleStatus::REFLECT)
        return status;

    STORMError eo("Particle is not on any boundary");
    eo.addEntry("Particle", particle);
    throw eo;
}

template<typename T, typename Grid>
std::vector<Particle<T, Grid>> RigidBoundary<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    (void) fullDt;
    return {};
}

} // namespace STORM

// Back-compat alias
template<typename T, typename Grid>
using RigidBoundaryCondition = STORM::RigidBoundary<T, Grid>;

#endif // STORM_RIGID_BOUNDARY_HPP
