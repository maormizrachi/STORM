#ifndef STORM_RIGID_BOUNDARY_HPP
#define STORM_RIGID_BOUNDARY_HPP

#include "BoundaryCondition.hpp"
#include "monte/STORMError.hpp"

namespace STORM {

template<typename T, typename Grid>
class RigidBoundary : public BoundaryCondition<T, Grid>
{
public:
    RigidBoundary(const Grid &grid);

    ~RigidBoundary() override;

    ParticleStatus apply(Particle<T, Grid> &particle) override;

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;
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
            const double signedDistance = ScalarProd(particle.location - onFace, normal);
            particle.location -= 2 * signedDistance * normal;
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
            const T &center = this->grid.GetMeshPoint(particle.cellIndex);
            constexpr double nudge = 1e-6;
            particle.location = particle.location * (1 - nudge) + nudge * center;
            return ParticleStatus::REFLECT;
        }
    }

    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
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
