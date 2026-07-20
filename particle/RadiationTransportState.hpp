#ifndef STORM_RADIATION_TRANSPORT_STATE_HPP
#define STORM_RADIATION_TRANSPORT_STATE_HPP

#include <cstdint>
#include <limits>

namespace STORM {

// State which belongs to the radiation representation of a packet rather
// than to the generic Monte Carlo particle machinery.  Keep this aggregate
// trivially copyable: the RDMA managers intentionally move particles as raw
// bytes and therefore copy this object together with Particle.
template<typename PointT>
struct RadiationTransportState
{
    enum Flag : std::uint8_t
    {
        DDMCMode = 1u << 0,
        DDMCCellResident = 1u << 1,
        DDMCComovingFrame = 1u << 2,
        PendingFlux = 1u << 3
    };

    std::uint8_t flags = 0;
    PointT pendingFlux{};
    std::size_t bypassCellID = std::numeric_limits<std::size_t>::max();
#ifdef MONTECARLO_POLARIZATION
    double pendingMeanScatterings = 0.0;
#endif

    bool has(Flag flag) const
    {
        return (this->flags & static_cast<std::uint8_t>(flag)) != 0;
    }

    void set(Flag flag, bool enabled = true)
    {
        if(enabled)
        {
            this->flags = static_cast<std::uint8_t>(
                this->flags | static_cast<std::uint8_t>(flag));
        }
        else
        {
            this->flags = static_cast<std::uint8_t>(
                this->flags & ~static_cast<std::uint8_t>(flag));
        }
    }

    bool isDDMC() const { return this->has(DDMCMode); }
    bool isResident() const { return this->has(DDMCCellResident); }
    bool isComoving() const { return this->has(DDMCComovingFrame); }
    bool hasPendingFlux() const { return this->has(PendingFlux); }
    bool invariantHolds() const
    {
        return !this->isDDMC() || (this->isResident() && this->isComoving());
    }

    void clearDDMC()
    {
        this->flags = static_cast<std::uint8_t>(
            this->flags & ~static_cast<std::uint8_t>(
                DDMCMode | DDMCCellResident | DDMCComovingFrame | PendingFlux));
        this->pendingFlux = PointT{};
        this->bypassCellID = std::numeric_limits<std::size_t>::max();
#ifdef MONTECARLO_POLARIZATION
        this->pendingMeanScatterings = 0.0;
#endif
    }

    void clearPendingFlux()
    {
        this->set(PendingFlux, false);
        this->pendingFlux = PointT{};
    }
};

} // namespace STORM

#endif // STORM_RADIATION_TRANSPORT_STATE_HPP
