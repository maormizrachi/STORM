#ifndef STORM_SHAPIRO_SUTHERLAND_PLANE_OBSERVER_HPP
#define STORM_SHAPIRO_SUTHERLAND_PLANE_OBSERVER_HPP

// Plane observer for the plane-parallel atmosphere benchmark.  The atmosphere is
// stratified along x with its surface facing +x and mirror side walls
// (SlabBoundary.hpp), so every photon crossing the plane x = planeX upward
// counts; it is binned by mu = k.x.  Stokes Q and U are taken in the meridian
// frame (reference axis: the surface normal projected on the sky), so Q = I_l -
// I_r and a plane-parallel field has U = 0.  Emitted and timed-out energy are
// tallied too, so the run can check that no light is left in flight.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "radiation/Observer.hpp"
#include "radiation/Polarization.hpp"

namespace STORM {
namespace examples {

template<typename PointT>
class PlaneObserver final : public RadiationObserver<PointT>
{
public:
    using Crossing = typename RadiationObserver<PointT>::Crossing;
    using CrossingRecord = typename RadiationObserver<PointT>::CrossingRecord;

    PlaneObserver(double planeX, std::size_t muBins)
        : energy(muBins), stokesQ(muBins), stokesU(muBins), planeX_(planeX)
    {
        if(muBins == 0 || !std::isfinite(planeX))
            throw std::invalid_argument("PlaneObserver needs a finite plane and at least one mu bin");
    }

    Crossing nextOutwardCrossing(const PointT &position, const PointT &velocity,
                                 double maxTime) const override
    {
        Crossing result;
        if(!(velocity.x > 0.0) || position.x >= planeX_ || !std::isfinite(maxTime))
            return result;
        double const t = (planeX_ - position.x) / velocity.x;
        if(t > 0.0 && t <= maxTime)
        {
            result.hit = true;
            result.time = t;
            result.point = position + velocity * t;
        }
        return result;
    }

    void recordCrossing(const CrossingRecord &record) override
    {
        double const norm = fallback::abs(record.direction);
        if(!(norm > 0.0) || !std::isfinite(norm))
            throw std::invalid_argument("PlaneObserver crossing has no direction");
        PointT const k = record.direction / norm;
        if(!(k.x > 0.0))
            return;
        std::size_t const bin = std::min(energy.size() - 1, static_cast<std::size_t>(k.x * energy.size()));
        double const w = record.weight;
        energy[bin] += w;
        // A packet whose polarization was never initialized is unpolarized
        // (unscattered thermal light): Q = U = 0.  Straight down the normal the
        // meridian frame is undefined and Q = U = 0 by symmetry.
        PointT const normal(1.0, 0.0, 0.0);
        if(!record.polarizationInitialized || !(fallback::abs(normal - k * k.x) > 0.0))
            return;
        double q = 0.0, u = 0.0;
        if(!polarization::stokesInReferenceFrame(record.stokesQ, record.stokesU, record.polarizationBasis, k,
                                                 normal, q, u))
            throw std::runtime_error("PlaneObserver: polarized packet has no basis perpendicular to its direction");
        stokesQ[bin] += w * q;
        stokesU[bin] += w * u;
    }

    void resetTallies() override
    {
        std::fill(energy.begin(), energy.end(), 0.0);
        std::fill(stokesQ.begin(), stokesQ.end(), 0.0);
        std::fill(stokesU.begin(), stokesU.end(), 0.0);
        emitted = timedOut = 0.0;
    }

    void addEmittedEnergy(double e) override { emitted += e; }
    void addTimedOutEnergy(double e) override { timedOut += e; }
    void addAbsorbedEnergy(double) override {}
    void addBoxEscapeEnergy(double) override {}
    void addCutoffEnergy(double) override {}

    std::vector<double> energy, stokesQ, stokesU;   // per mu bin, this rank
    double emitted = 0.0, timedOut = 0.0;

private:
    double planeX_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_SHAPIRO_SUTHERLAND_PLANE_OBSERVER_HPP
