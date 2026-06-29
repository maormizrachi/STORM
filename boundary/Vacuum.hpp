#ifndef STORM_VACUUM_BOUNDARY_HPP
#define STORM_VACUUM_BOUNDARY_HPP

#include "BoundaryCondition.hpp"

namespace STORM {

template<typename T, typename Grid>
class VacuumBoundary : public BoundaryCondition<T, Grid>
{
public:
    using MCParticle = Particle<T, Grid>;

    explicit VacuumBoundary(const Grid &grid)
        : BoundaryCondition<T, Grid>(grid)
    {}

    ParticleStatus apply(MCParticle &particle) override
    {
        escapedEnergy_ += particle.weight;
        return ParticleStatus::REMOVE;
    }

    std::vector<MCParticle> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::Unsupported;
    }

    double getEscapedEnergy() const
    {
        return escapedEnergy_;
    }

    void resetEscapedEnergy()
    {
        escapedEnergy_ = 0.0;
    }

private:
    double escapedEnergy_ = 0.0;
};

} // namespace STORM

// Back-compat alias
template<typename T, typename Grid>
using VacuumBoundaryCondition = STORM::VacuumBoundary<T, Grid>;

#endif // STORM_VACUUM_BOUNDARY_HPP
