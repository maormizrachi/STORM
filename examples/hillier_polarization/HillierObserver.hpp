#ifndef STORM_HILLIER_POLARIZATION_OBSERVER_HPP
#define STORM_HILLIER_POLARIZATION_OBSERVER_HPP

// Ring observer for the Hillier (1994) benchmark: detectors placed only at the
// inclinations the paper tabulates, each accepting photons within a cone.
//
// Why not SphericalObserver:
//
//   1. SphericalObserver bins a crossing by WHERE it crosses the tally sphere
//      (the radius vector), not by WHERE IT IS GOING (record.direction).  Those
//      differ by up to arcsin(r_scatter / r_observer).  Here the density goes as
//      r^-4, so most scattering happens within a few R_min of the centre and the
//      discrepancy is only a few degrees -- but it is a real smearing of the
//      angular variable the benchmark is built on, and it is worst exactly where
//      p varies fastest with inclination.  This observer bins by direction.
//
//   2. Its per-crossing polarization rotation uses the same radius vector as the
//      sky normal.  The sky basis must be perpendicular to the PROPAGATION
//      direction, or Q and U are resolved in a plane the photon is not travelling
//      in.
//
//   3. Fibonacci directions land nowhere near the paper's inclinations, so the
//      comparison needs a +-7 deg band whose <sin^2 i> is biased away from
//      sin^2 of its centre -- 12% at i = 22.5 deg, where sin^2 i varies steeply.
//      Rings sit exactly on 22.5, 45, 67.5 and 90, so the bias is gone by
//      construction rather than corrected for.
//
// A crossing is credited to at most ONE detector -- the nearest direction, and
// only if within the acceptance half-angle.  Cones may overlap without double
// counting, so the pooled statistics stay clean, and photons heading nowhere near
// a detector are simply not seen, which is what a detector does.
//
// The accessors mirror SphericalObserver's (directions, observerEnergy, stokesQ,
// stokesU) so the analysis and plotting downstream do not care which is in use.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "radiation/Observer.hpp"

namespace STORM {
namespace examples {

using namespace STORM::fallback;

template<typename PointT>
class HillierRingObserver final : public STORM::RadiationObserver<PointT>
{
public:
    using Base = STORM::RadiationObserver<PointT>;
    using Crossing = typename Base::Crossing;
    using CrossingRecord = typename Base::CrossingRecord;

    //! \param inclinationsDeg Ring inclinations, e.g. {22.5, 45, 67.5, 90}
    //! \param azimuthsPerRing Detectors spanning phi on each ring
    //! \param acceptanceDeg   Cone half-angle about each detector direction
    //! \param mirror          Also place rings at 180-i; the envelope is
    //!                        symmetric about the equator, so this is free
    //!                        statistics rather than a new measurement
    HillierRingObserver(PointT center, double radius,
                        std::vector<double> const &inclinationsDeg,
                        std::size_t azimuthsPerRing,
                        double acceptanceDeg,
                        bool mirror = true)
        : center_(center), radius_(radius), radiusSquared_(radius * radius),
          cosAcceptance_(std::cos(acceptanceDeg * M_PI / 180.0))
    {
        if (!(std::isfinite(radius_) && radius_ > 0.0) || azimuthsPerRing == 0)
            throw std::invalid_argument(
                "HillierRingObserver requires positive radius and azimuths");
        if (!(acceptanceDeg > 0.0 && acceptanceDeg < 90.0))
            throw std::invalid_argument(
                "HillierRingObserver acceptance must be in (0, 90) degrees");

        for (double incDeg : inclinationsDeg)
        {
            addRing(incDeg, azimuthsPerRing);
            // 90 deg is its own mirror image; anything else gains a second ring.
            if (mirror && std::abs(incDeg - 90.0) > 1.0e-9)
                addRing(180.0 - incDeg, azimuthsPerRing);
        }

        std::size_t const n = directions_.size();
        observerEnergy_.assign(n, 0.0);
        observerStokesQ_.assign(n, 0.0);
        observerStokesU_.assign(n, 0.0);
        observerStokesQSquared_.assign(n, 0.0);
        crossingCount_.assign(n, 0);
    }

    // ---- geometry: identical sphere intersection to SphericalObserver -------
    Crossing nextOutwardCrossing(const PointT &position, const PointT &velocity,
                                 double maxTime) const override
    {
        Crossing result;
        if (!(maxTime > 0.0) || !std::isfinite(maxTime))
            return result;
        PointT const offset = position - center_;
        double const a = ScalarProd(velocity, velocity);
        double const b = 2.0 * ScalarProd(offset, velocity);
        double const c = ScalarProd(offset, offset) - radiusSquared_;
        if (!(a > 0.0) || !std::isfinite(a))
            return result;
        double const discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0 || !std::isfinite(discriminant))
            return result;
        double const root = std::sqrt(std::max(0.0, discriminant));
        double const epsilon = std::max(1.0e-14, maxTime * 1.0e-12);
        for (double t : {(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)})
        {
            if (!(t > epsilon) || t > maxTime + epsilon)
                continue;
            PointT const point = position + velocity * t;
            if (ScalarProd(point - center_, velocity) <= 0.0)
                continue;
            if (!result.hit || t < result.time)
            {
                result.hit = true;
                result.time = t;
                result.point = point;
            }
        }
        return result;
    }

    void recordCrossing(const CrossingRecord &record) override
    {
        double const directionNorm = abs(record.direction);
        if (!(directionNorm > 0.0) || !std::isfinite(directionNorm))
            throw std::invalid_argument("HillierRingObserver crossing has no direction");
        PointT const k = record.direction / directionNorm;

        // Nearest detector, and only if the photon is inside its cone.
        std::size_t best = 0;
        double bestDot = cosAcceptance_;
        bool found = false;
        for (std::size_t i = 0; i < directions_.size(); ++i)
        {
            double const dot = ScalarProd(directions_[i], k);
            if (dot > bestDot) { bestDot = dot; best = i; found = true; }
        }
        if (!found)
            return;                       // heading at no detector: not observed

        double const w = record.weight;
        observerEnergy_[best] += w;
        ++crossingCount_[best];
        totalCrossingEnergy_ += w;
#ifdef MONTECARLO_POLARIZATION
        if (polarizationEnabled_ && record.polarizationInitialized)
            accumulatePolarization(record, best, k, w);
#else
        (void) polarizationEnabled_;
#endif
    }

    void addEmittedEnergy(double energy) override { emittedEnergy_ += energy; }
    void addAbsorbedEnergy(double energy) override { absorbedEnergy_ += energy; }
    void addBoxEscapeEnergy(double energy) override { boxEscapeEnergy_ += energy; }
    void addTimedOutEnergy(double energy) override { timedOutEnergy_ += energy; }
    void addCutoffEnergy(double energy) override { cutoffEnergy_ += energy; }
    void setPolarizationEnabled(bool enabled) override
    {
#ifdef MONTECARLO_POLARIZATION
        polarizationEnabled_ = enabled;
#else
        (void) enabled;
#endif
    }

    void resetTallies() override
    {
        std::fill(observerEnergy_.begin(), observerEnergy_.end(), 0.0);
        std::fill(observerStokesQ_.begin(), observerStokesQ_.end(), 0.0);
        std::fill(observerStokesU_.begin(), observerStokesU_.end(), 0.0);
        std::fill(observerStokesQSquared_.begin(), observerStokesQSquared_.end(), 0.0);
        std::fill(crossingCount_.begin(), crossingCount_.end(), 0);
        emittedEnergy_ = absorbedEnergy_ = boxEscapeEnergy_ = timedOutEnergy_ =
            cutoffEnergy_ = totalCrossingEnergy_ = 0.0;
    }

    // ---- accessors, matching SphericalObserver ------------------------------
    const std::vector<PointT> &directions() const { return directions_; }
    std::vector<double> &observerEnergy() { return observerEnergy_; }
    std::vector<double> &stokesQ() { return observerStokesQ_; }
    std::vector<double> &stokesU() { return observerStokesU_; }
    const std::vector<double> &stokesQSquared() const { return observerStokesQSquared_; }
    const std::vector<std::size_t> &crossingCount() const { return crossingCount_; }
    //! Nominal inclination of the ring each detector belongs to [deg], folded
    //! about the equator so a mirrored ring reports its parent inclination.
    const std::vector<double> &ringInclinationDeg() const { return ringInclination_; }
    double acceptanceCosine() const { return cosAcceptance_; }
    double totalCrossingEnergy() const { return totalCrossingEnergy_; }

private:
    void addRing(double incDeg, std::size_t azimuths)
    {
        double const inc = incDeg * M_PI / 180.0;
        double const s = std::sin(inc), c = std::cos(inc);
        double const folded = incDeg > 90.0 ? 180.0 - incDeg : incDeg;
        for (std::size_t m = 0; m < azimuths; ++m)
        {
            double const phi = 2.0 * M_PI * static_cast<double>(m) /
                               static_cast<double>(azimuths);
            directions_.emplace_back(s * std::cos(phi), s * std::sin(phi), c);
            ringInclination_.push_back(folded);
        }
    }

#ifdef MONTECARLO_POLARIZATION
    //! Resolve Q, U against the projected symmetry axis, in the plane normal to
    //! the PROPAGATION direction.
    //!
    //! The reference axis is z projected perpendicular to k -- the rotation axis
    //! as it appears on the sky.  For an axisymmetric envelope the polarization
    //! is parallel or perpendicular to that axis, so u must average to zero; a
    //! non-zero u is then a sharp diagnostic of a basis-convention error rather
    //! than a quantity anyone has to interpret.  k is never within 12.5 deg of
    //! the pole for any ring used here, so the projection is well conditioned.
    void accumulatePolarization(const CrossingRecord &record, std::size_t observer,
                                const PointT &k, double w)
    {
        PointT basis = record.polarizationBasis;
        if (!(abs(basis) > 0.0))
            return;
        basis = normalize(basis - k * ScalarProd(basis, k));

        PointT const axis = PointT(0.0, 0.0, 1.0);
        PointT const projected = axis - k * ScalarProd(axis, k);
        double const projectedNorm = abs(projected);
        if (!(projectedNorm > 0.0))
            return;
        PointT const sky = projected / projectedNorm;

        double const cosine = std::max(-1.0, std::min(1.0, ScalarProd(basis, sky)));
        double const sine = ScalarProd(CrossProduct(basis, sky), k);
        double const c2 = cosine * cosine - sine * sine;
        double const s2 = 2.0 * sine * cosine;

        double const q = w * (record.stokesQ * c2 + record.stokesU * s2);
        observerStokesQ_[observer] += q;
        observerStokesU_[observer] += w * (-record.stokesQ * s2 + record.stokesU * c2);
        observerStokesQSquared_[observer] += q * q;
    }
#endif

    PointT center_;
    double radius_;
    double radiusSquared_;
    double cosAcceptance_;
    bool polarizationEnabled_ = true;
    std::vector<PointT> directions_;
    std::vector<double> ringInclination_;
    std::vector<double> observerEnergy_;
    std::vector<double> observerStokesQ_;
    std::vector<double> observerStokesU_;
    std::vector<double> observerStokesQSquared_;
    std::vector<std::size_t> crossingCount_;
    double emittedEnergy_ = 0.0;
    double absorbedEnergy_ = 0.0;
    double boxEscapeEnergy_ = 0.0;
    double timedOutEnergy_ = 0.0;
    double cutoffEnergy_ = 0.0;
    double totalCrossingEnergy_ = 0.0;
};

} // namespace examples
} // namespace STORM

#endif // STORM_HILLIER_POLARIZATION_OBSERVER_HPP
