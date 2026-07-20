#ifndef STORM_RADIATION_OBSERVER_HPP
#define STORM_RADIATION_OBSERVER_HPP

#include <cstddef>
#include <limits>

#include "elementary/PointOps.hpp"

namespace STORM {

template<typename PointT>
struct ObserverCrossing
{
    bool hit = false;
    double time = std::numeric_limits<double>::infinity();
    PointT point{};
};

template<typename PointT>
struct ObserverCrossingRecord
{
    PointT crossingPoint{};
    PointT direction{};
    double weight = 0.0;
    double frequency = 0.0;
    std::size_t sourceCellID = std::numeric_limits<std::size_t>::max();
#ifdef MONTECARLO_POLARIZATION
    double stokesQ = 0.0;
    double stokesU = 0.0;
    PointT polarizationBasis{};
    bool polarizationInitialized = false;
#endif
};

template<typename PointT>
class RadiationObserver
{
public:
    using Crossing = ObserverCrossing<PointT>;
    using CrossingRecord = ObserverCrossingRecord<PointT>;

    virtual ~RadiationObserver() = default;

    virtual Crossing nextOutwardCrossing(const PointT &position,
                                         const PointT &velocity,
                                         double maxTime) const = 0;
    virtual void recordCrossing(const CrossingRecord &record) = 0;
    virtual void addEmittedEnergy(double energy) = 0;
    virtual void addAbsorbedEnergy(double energy) = 0;
    virtual void addBoxEscapeEnergy(double energy) = 0;
    virtual void addTimedOutEnergy(double energy) = 0;
    virtual void addCutoffEnergy(double energy) = 0;
    virtual void resetTallies() = 0;
    // Transport owns the polarization switch.  Observers that expose
    // polarization tallies may override this; other observers can ignore it.
    virtual void setPolarizationEnabled(bool) {}
};

} // namespace STORM

#endif // STORM_RADIATION_OBSERVER_HPP
