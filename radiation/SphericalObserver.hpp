#ifndef STORM_RADIATION_SPHERICAL_OBSERVER_HPP
#define STORM_RADIATION_SPHERICAL_OBSERVER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Observer.hpp"

namespace STORM {

using namespace STORM::fallback;

/*
 * Lightweight, format-independent spherical observer.  It owns only the
 * crossing ledger and direction/group tallies; HDF5/VTK/TXT adapters can be
 * built on these accessors by an application without making the transport
 * library depend on an output stack.
 */
template<typename PointT, std::size_t NumGroups>
class SphericalObserver final : public RadiationObserver<PointT>
{
public:
    using Base = RadiationObserver<PointT>;
    using Crossing = typename Base::Crossing;
    using CrossingRecord = typename Base::CrossingRecord;
    using GroupArray = std::array<double, NumGroups>;

    struct Snapshot
    {
        std::vector<double> observerEnergy;
        std::vector<std::vector<double>> groupEnergy;
        std::vector<std::size_t> crossingCount;
        double emittedEnergy = 0.0;
        double absorbedEnergy = 0.0;
        double boxEscapeEnergy = 0.0;
        double timedOutEnergy = 0.0;
        double cutoffEnergy = 0.0;
        double totalCrossingEnergy = 0.0;
    };

    SphericalObserver(PointT center,
                      double radius,
                      std::size_t numObservers,
                      std::array<double, NumGroups + 1> groupBoundaries,
                      bool polarizationEnabled = false):
        center_(center),
        radius_(radius),
        radiusSquared_(radius * radius),
        groupBoundaries_(groupBoundaries),
        polarizationEnabled_(polarizationEnabled)
    {
        if(!(std::isfinite(radius_) && radius_ > 0.0) || numObservers == 0)
        {
            throw std::invalid_argument("SphericalObserver requires positive radius and observers");
        }
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            if(!std::isfinite(groupBoundaries_[g]) ||
               !std::isfinite(groupBoundaries_[g + 1]) ||
               groupBoundaries_[g + 1] <= groupBoundaries_[g])
            {
                throw std::invalid_argument("SphericalObserver group boundaries must increase");
            }
        }

        directions_.reserve(numObservers);
        constexpr double pi = 3.1415926535897932384626433832795;
        constexpr double goldenRatio = 1.6180339887498948482;
        for(std::size_t i = 0; i < numObservers; ++i)
        {
            double const z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5)
                / static_cast<double>(numObservers);
            double const phi = 2.0 * pi * static_cast<double>(i) / goldenRatio;
            double const radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            directions_.emplace_back(radial * std::cos(phi),
                                     radial * std::sin(phi), z);
        }
        observerEnergy_.assign(numObservers, 0.0);
        observerEnergySquared_.assign(numObservers, 0.0);
        crossingCount_.assign(numObservers, 0);
        groupEnergy_.assign(numObservers, GroupArray{});
        groupEnergySquared_.assign(numObservers, GroupArray{});
        groupCrossingCount_.assign(numObservers, std::array<std::size_t, NumGroups>{});
#ifdef MONTECARLO_POLARIZATION
        observerStokesQ_.assign(numObservers, 0.0);
        observerStokesU_.assign(numObservers, 0.0);
#endif
        buildSkyBases();
    }

    Crossing nextOutwardCrossing(const PointT &position,
                                 const PointT &velocity,
                                 double maxTime) const override
    {
        Crossing result;
        if(!(maxTime > 0.0) || !std::isfinite(maxTime))
        {
            return result;
        }
        PointT const offset = position - center_;
        double const a = ScalarProd(velocity, velocity);
        double const b = 2.0 * ScalarProd(offset, velocity);
        double const c = ScalarProd(offset, offset) - radiusSquared_;
        if(!(a > 0.0) || !std::isfinite(a))
        {
            return result;
        }
        double const discriminant = b * b - 4.0 * a * c;
        if(discriminant < 0.0 || !std::isfinite(discriminant))
        {
            return result;
        }
        double const root = std::sqrt(std::max(0.0, discriminant));
        double const t0 = (-b - root) / (2.0 * a);
        double const t1 = (-b + root) / (2.0 * a);
        double const epsilon = std::max(1.0e-14, maxTime * 1.0e-12);
        for(double t : {t0, t1})
        {
            if(!(t > epsilon) || t > maxTime + epsilon)
            {
                continue;
            }
            PointT const point = position + velocity * t;
            if(ScalarProd(point - center_, velocity) <= 0.0)
            {
                continue;
            }
            if(!result.hit || t < result.time)
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
        PointT const radiusVector = record.crossingPoint - center_;
        double const radiusNorm = abs(radiusVector);
        double const directionNorm = abs(record.direction);
        if(!(radiusNorm > 0.0) || !(directionNorm > 0.0) ||
           !std::isfinite(radiusNorm) || !std::isfinite(directionNorm))
        {
            throw std::invalid_argument("SphericalObserver crossing has invalid geometry");
        }
        std::size_t const observer = nearestDirection(radiusVector / radiusNorm);
        std::size_t const group = findGroup(record.frequency);
        double const signedWeight = record.weight;
        observerEnergy_[observer] += signedWeight;
        observerEnergySquared_[observer] += signedWeight * signedWeight;
        ++crossingCount_[observer];
        groupEnergy_[observer][group] += signedWeight;
        groupEnergySquared_[observer][group] += signedWeight * signedWeight;
        groupCrossingCount_[observer][group] += 1.0;
        totalCrossingEnergy_ += signedWeight;
        if(record.sourceCellID != std::numeric_limits<std::size_t>::max())
        {
            sourceCellEscape_[record.sourceCellID] += signedWeight;
            sourceCellGroupEscape_[record.sourceCellID][group] += signedWeight;
        }
#ifdef MONTECARLO_POLARIZATION
        if(polarizationEnabled_ && record.polarizationInitialized)
        {
            accumulatePolarization(record, observer, radiusVector / radiusNorm);
        }
#else
        (void) polarizationEnabled_;
#endif
    }

    void addEmittedEnergy(double energy) override { emittedEnergy_ += energy; }
    void addAbsorbedEnergy(double energy) override { absorbedEnergy_ += energy; }
    void addBoxEscapeEnergy(double energy) override { boxEscapeEnergy_ += energy; }
    void addTimedOutEnergy(double energy) override { timedOutEnergy_ += energy; }
    void addCutoffEnergy(double energy) override { cutoffEnergy_ += energy; }

    void resetTallies() override
    {
        std::fill(observerEnergy_.begin(), observerEnergy_.end(), 0.0);
        std::fill(observerEnergySquared_.begin(), observerEnergySquared_.end(), 0.0);
        std::fill(crossingCount_.begin(), crossingCount_.end(), 0);
        for(auto &row : groupEnergy_) row.fill(0.0);
        for(auto &row : groupEnergySquared_) row.fill(0.0);
        for(auto &row : groupCrossingCount_) row.fill(0);
        sourceCellEscape_.clear();
        sourceCellGroupEscape_.clear();
        emittedEnergy_ = absorbedEnergy_ = boxEscapeEnergy_ = timedOutEnergy_ = cutoffEnergy_ = 0.0;
        totalCrossingEnergy_ = 0.0;
#ifdef MONTECARLO_POLARIZATION
        std::fill(observerStokesQ_.begin(), observerStokesQ_.end(), 0.0);
        std::fill(observerStokesU_.begin(), observerStokesU_.end(), 0.0);
#endif
    }

    Snapshot snapshot() const
    {
        return Snapshot{observerEnergy_, groupEnergy(), crossingCount_, emittedEnergy_,
                        absorbedEnergy_, boxEscapeEnergy_, timedOutEnergy_, cutoffEnergy_, totalCrossingEnergy_};
    }

    const std::vector<PointT> &directions() const { return directions_; }
    const std::vector<double> &observerEnergy() const { return observerEnergy_; }
    const std::vector<std::vector<double>> &groupEnergy() const
    {
        groupEnergyView_.resize(groupEnergy_.size());
        for(std::size_t i = 0; i < groupEnergy_.size(); ++i)
        {
            groupEnergyView_[i].assign(groupEnergy_[i].begin(), groupEnergy_[i].end());
        }
        return groupEnergyView_;
    }
    const std::vector<std::size_t> &crossingCount() const { return crossingCount_; }
    const std::unordered_map<std::size_t, double> &sourceCellEscape() const { return sourceCellEscape_; }
    const std::unordered_map<std::size_t, GroupArray> &sourceCellGroupEscape() const
    {
        return sourceCellGroupEscape_;
    }
    double emittedEnergy() const { return emittedEnergy_; }
    double absorbedEnergy() const { return absorbedEnergy_; }
    double boxEscapeEnergy() const { return boxEscapeEnergy_; }
    double timedOutEnergy() const { return timedOutEnergy_; }
    double cutoffEnergy() const { return cutoffEnergy_; }
    double totalCrossingEnergy() const { return totalCrossingEnergy_; }
#ifdef MONTECARLO_POLARIZATION
    const std::vector<double> &stokesQ() const { return observerStokesQ_; }
    const std::vector<double> &stokesU() const { return observerStokesU_; }
#endif

    std::size_t findGroup(double frequency) const
    {
        if(frequency <= groupBoundaries_[0]) return 0;
        for(std::size_t g = 0; g + 1 < NumGroups + 1; ++g)
        {
            if(frequency < groupBoundaries_[g + 1]) return g;
        }
        return NumGroups - 1;
    }

private:
    std::size_t nearestDirection(const PointT &unitDirection) const
    {
        std::size_t best = 0;
        double bestDot = -std::numeric_limits<double>::infinity();
        for(std::size_t i = 0; i < directions_.size(); ++i)
        {
            double const dot = ScalarProd(directions_[i], unitDirection);
            if(dot > bestDot) { bestDot = dot; best = i; }
        }
        return best;
    }

    void buildSkyBases()
    {
#ifdef MONTECARLO_POLARIZATION
        skyE1_.resize(directions_.size());
        for(std::size_t i = 0; i < directions_.size(); ++i)
        {
            PointT helper = std::abs(directions_[i].z) < 0.9
                ? PointT(0.0, 0.0, 1.0) : PointT(0.0, 1.0, 0.0);
            skyE1_[i] = normalize(helper - directions_[i] * ScalarProd(helper, directions_[i]));
        }
#endif
    }

#ifdef MONTECARLO_POLARIZATION
    void accumulatePolarization(const CrossingRecord &record,
                                std::size_t observer,
                                const PointT &normal)
    {
        PointT basis = record.polarizationBasis;
        double const basisNorm = abs(basis);
        if(!(basisNorm > 0.0)) return;
        basis = normalize(basis - normal * ScalarProd(basis, normal));
        PointT const sky = skyE1_[observer];
        double const cosine = std::clamp(ScalarProd(basis, sky), -1.0, 1.0);
        double const sine = ScalarProd(CrossProduct(basis, sky), normal);
        double const c2 = cosine * cosine - sine * sine;
        double const s2 = 2.0 * sine * cosine;
        observerStokesQ_[observer] += record.stokesQ * c2 + record.stokesU * s2;
        observerStokesU_[observer] += -record.stokesQ * s2 + record.stokesU * c2;
    }
#endif

    PointT center_;
    double radius_;
    double radiusSquared_;
    std::array<double, NumGroups + 1> groupBoundaries_;
    bool polarizationEnabled_ = false;
    std::vector<PointT> directions_;
    std::vector<double> observerEnergy_;
    std::vector<double> observerEnergySquared_;
    std::vector<std::size_t> crossingCount_;
    std::vector<std::array<double, NumGroups>> groupEnergy_;
    std::vector<std::array<double, NumGroups>> groupEnergySquared_;
    std::vector<std::array<std::size_t, NumGroups>> groupCrossingCount_;
    mutable std::vector<std::vector<double>> groupEnergyView_;
    std::unordered_map<std::size_t, double> sourceCellEscape_;
    std::unordered_map<std::size_t, GroupArray> sourceCellGroupEscape_;
    double emittedEnergy_ = 0.0;
    double absorbedEnergy_ = 0.0;
    double boxEscapeEnergy_ = 0.0;
    double timedOutEnergy_ = 0.0;
    double cutoffEnergy_ = 0.0;
    double totalCrossingEnergy_ = 0.0;
#ifdef MONTECARLO_POLARIZATION
    std::vector<PointT> skyE1_;
    std::vector<double> observerStokesQ_;
    std::vector<double> observerStokesU_;
#endif
};

} // namespace STORM

#endif // STORM_RADIATION_SPHERICAL_OBSERVER_HPP
