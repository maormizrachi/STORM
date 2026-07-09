#ifndef STORM_BOUNDARY_CONDITION_HPP
#define STORM_BOUNDARY_CONDITION_HPP

#include <algorithm>
#include <cmath>
#include "../particle/Particle.hpp"
#include "../particle/ParticleStatus.hpp"
#include "../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

// DDMCBoundaryFaceBehavior describes how DDMC should treat an outside-box
// face adjacent to a candidate DDMC cell.
//
// Unsupported:
//   DDMC does not know how to model this boundary face. The cell must be
//   excluded from DDMC acceleration.
//
// ReflectingRigid:
//   The face is a rigid reflecting wall. In DDMC this is a zero-normal-current
//   boundary. No leakage event should be added through this face, but the cell
//   should not be excluded solely because of this face.
enum class DDMCBoundaryFaceBehavior {
    Unsupported,
    ReflectingRigid
};

template<typename T, typename Grid>
class BoundaryCondition
{
public:
    BoundaryCondition(const Grid &grid);

    virtual ~BoundaryCondition() = default;

    virtual ParticleStatus apply(Particle<T, Grid> &particle) = 0;

    virtual std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) = 0;

    virtual DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::Unsupported;
    }

protected:
    const Grid &grid;

    bool getInwardBoxFaceNormalIfClose(
        const typename Grid::Face_T &face,
        const T &location,
        T &normal,
        double &faceScale) const
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        normal = CrossProduct(u, v);
        faceScale = std::min(abs(u), abs(v));

        double const normalNorm = abs(normal);
        if(!(normalNorm > 0.0) || !std::isfinite(normalNorm) ||
            !(faceScale > 0.0) || !std::isfinite(faceScale))
            return false;

        double const planeDistance = ScalarProd(normal, location - onFace);
        if(std::fabs(planeDistance) >= EPSILON * faceScale * faceScale * faceScale)
            return false;

        normal *= 1.0 / normalNorm;

        const auto &[boxLL, boxUR] = this->grid.GetBoxCoordinates();
        T const boxCenter = 0.5 * (boxLL + boxUR);
        if(ScalarProd(normal, boxCenter - onFace) < 0.0)
            normal *= -1.0;

        return true;
    }

    bool reflectParticleOnBoxFace(
        Particle<T, Grid> &particle,
        const typename Grid::Face_T &face) const
    {
        T normal;
        double faceScale = 0.0;
        if(!getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
            return false;

        const T &onFace = face.vertices[0];
        double const signedDistance = ScalarProd(particle.location - onFace, normal);
        particle.location -= signedDistance * normal;

        constexpr double nudge = 1e-6;
        particle.location += nudge * faceScale * normal;

        double const vn = ScalarProd(particle.velocity, normal);
        if(vn < 0.0)
            particle.velocity -= 2.0 * vn * normal;

        return true;
    }

    bool getDDMCOrientedOutwardNormal(
        size_t /*faceIdx*/,
        size_t insideCellIndex,
        size_t outsidePointIndex,
        T &nOut) const
    {
        nOut = this->grid.GetMeshPoint(outsidePointIndex) -
               this->grid.GetMeshPoint(insideCellIndex);

        double const nNorm = abs(nOut);
        if(!(nNorm > 0.0) || !std::isfinite(nNorm))
            return false;

        nOut *= 1.0 / nNorm;
        return true;
    }
};

template<typename T, typename Grid>
BoundaryCondition<T, Grid>::BoundaryCondition(const Grid &grid)
    : grid(grid)
{}

} // namespace STORM

#endif // STORM_BOUNDARY_CONDITION_HPP
