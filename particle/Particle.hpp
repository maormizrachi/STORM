#ifndef STORM_PARTICLE_HPP
#define STORM_PARTICLE_HPP

#include <sstream>
#include <vector>
#include <limits>
#ifdef STORM_WITH_MPI
    #include <mpi.h>
    #include <functional>
    #include <mpi_utils/mpi_commands.hpp>
    #include <mpi_utils/serialize/Serializer.hpp>
#endif // STORM_WITH_MPI
#include "../StormError.hpp"
#include "ParticleStatus.hpp"
#include "../elementary/PointOps.hpp"
#include "RadiationTransportState.hpp"
#include "../types.hpp"

#define EPSILON 1e-12

namespace STORM {

using namespace STORM::fallback;

using dt_t = double;

#ifdef STORM_WITH_TRACING_HISTORY

template<typename T>
struct ParticleHistory
{
    cell_index_t cellIndex = 0;
    std::int32_t rank = -1;
    std::int32_t operation = 0;
    particle_step_t step = 0;
    std::uint8_t reflected = 0;
    T location = T();
    T velocity = T();
    T preReflectLocation = T();
    T preReflectVelocity = T();
};
#endif // STORM_WITH_TRACING_HISTORY

template<typename T>
struct ParticleRoutingState
{
#ifdef STORM_WITH_MPI
    rank_t rank = -1;
#ifdef STORM_DEBUG
    cell_index_t cellIndexInPrevRank = std::numeric_limits<cell_index_t>::max();
    T previousLocation = T(std::numeric_limits<double>::max());
    cell_index_t particleTHInLastRank = std::numeric_limits<cell_index_t>::max();
    cell_index_t particleIndexInLastRank = std::numeric_limits<cell_index_t>::max();
    std::uint8_t checkedHere = 1;
    cell_index_t ghostIndex = std::numeric_limits<cell_index_t>::max();
    T newCellValue = T(std::numeric_limits<double>::max());
    rank_t nextRank = std::numeric_limits<rank_t>::max();
    rank_t sentByRank = std::numeric_limits<rank_t>::max();
    std::uint8_t removedFromRank = 0;
    particle_step_t lastSeen = 0;
    rank_t lastSeenRank = std::numeric_limits<rank_t>::max();
    rank_t lastSeenRankBuf = std::numeric_limits<rank_t>::max();
    cell_index_t lastSeenIndex = std::numeric_limits<cell_index_t>::max();
#endif
#endif
    std::uint8_t sent = 0;
};

// Physics state only.  This is the payload to mirror in device memory; it has
// no MPI bookkeeping and no Serializable vtable.
template<typename T>
struct ParticleTransportData
{
    particle_id_t id = std::numeric_limits<particle_id_t>::max();
    cell_id_t cellID = std::numeric_limits<cell_id_t>::max();
    cell_id_t sourceCellID = std::numeric_limits<cell_id_t>::max();
    T location = T(std::numeric_limits<typename T::coord_type>::max());
    T velocity = T(std::numeric_limits<typename T::coord_type>::max());
    cell_index_t cellIndex = std::numeric_limits<cell_index_t>::max();
    dt_t timeLeft = std::numeric_limits<dt_t>::max();
    double frequency = std::numeric_limits<double>::max();
    double weight = 0.0;
    double initialWeight = 0.0;
    std::uint64_t rngKey = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t rngCounter = 0;
    RadiationTransportState<T> radiationState{};
#ifdef MONTECARLO_POLARIZATION
    double stokesQ = 0.0;
    double stokesU = 0.0;
    T polarizationBasis = T();
    std::uint8_t polarizationInitialized = 0;
#endif
    particle_step_t steps = 0;
    std::uint8_t on_track = 0;

    ParticleTransportData() = default;

    ParticleTransportData(particle_id_t id_, const T &location_,
                          const T &velocity_, dt_t timeLeft_)
        : id(id_), location(location_), velocity(velocity_),
          timeLeft(timeLeft_) {}
};

#ifdef STORM_WITH_MPI
template<typename T>
struct Particle : public Serializable, public ParticleTransportData<T>,
                  public ParticleRoutingState<T>
#else
template<typename T>
struct Particle : public ParticleTransportData<T>, public ParticleRoutingState<T>
#endif
{
    #ifdef STORM_WITH_TRACING_HISTORY
        ParticleHistory<T> tracingHistory[STORM_WITH_TRACING_HISTORY] = {};
        std::uint64_t tracingHistoryIndex = 0;
        std::uint64_t tracingHistoryCount = 0;

        inline void recordHistory(size_t cell, int rnk, int op)
        {
            ParticleHistory<T> &entry = this->tracingHistory[this->tracingHistoryIndex];
            entry.cellIndex = cell;
            entry.rank = rnk;
            entry.operation = op;
            entry.step = this->steps;
            entry.reflected = false;
            entry.location = this->location;
            entry.velocity = this->velocity;
            this->tracingHistoryIndex = (this->tracingHistoryIndex + 1) % STORM_WITH_TRACING_HISTORY;
            if(this->tracingHistoryCount < STORM_WITH_TRACING_HISTORY)
            {
                this->tracingHistoryCount++;
            }
        }

        inline void markLastHistoryReflected(const T &locBeforeReflect, const T &velBeforeReflect)
        {
            if(this->tracingHistoryCount == 0)
            {
                return;
            }
            size_t lastIdx = (this->tracingHistoryIndex + STORM_WITH_TRACING_HISTORY - 1) % STORM_WITH_TRACING_HISTORY;
            this->tracingHistory[lastIdx].reflected = true;
            this->tracingHistory[lastIdx].preReflectLocation = locBeforeReflect;
            this->tracingHistory[lastIdx].preReflectVelocity = velBeforeReflect;
        }

        inline void addTracingHistoryToError(StormError &eo) const
        {
            eo.addEntry("Tracing History Count", this->tracingHistoryCount);
            for(size_t h = 0; h < this->tracingHistoryCount; h++)
            {
                size_t idx = (this->tracingHistoryIndex - this->tracingHistoryCount + h + STORM_WITH_TRACING_HISTORY) % STORM_WITH_TRACING_HISTORY;
                const ParticleHistory<T> &hist = this->tracingHistory[idx];
                std::string prefix = "History[" + std::to_string(h) + "] ";
                eo.addEntry(prefix + "Cell", hist.cellIndex);
                eo.addEntry(prefix + "Rank", hist.rank);
                eo.addEntry(prefix + "Op", ParticleStatusToString(hist.operation));
                eo.addEntry(prefix + "Step", hist.step);
            }
        }
    #endif // STORM_WITH_TRACING_HISTORY

    explicit Particle(particle_id_t id_ = std::numeric_limits<particle_id_t>::max(), const T &location_ = T(std::numeric_limits<double>::max()), const T &velocity_ = T(std::numeric_limits<double>::max()), dt_t timeLeft_ = dt_t(std::numeric_limits<double>::max())):
        ParticleTransportData<T>(id_, location_, velocity_, timeLeft_)
    {
        #ifdef STORM_DEBUG
        this->checkedHere = true;
        this->ghostIndex = std::numeric_limits<cell_index_t>::max();
        this->newCellValue = T(std::numeric_limits<double>::max());
        this->nextRank = std::numeric_limits<rank_t>::max();
        this->removedFromRank = false;
        this->sentByRank = std::numeric_limits<rank_t>::max();
        #endif // STORM_DEBUG
#ifdef MONTECARLO_POLARIZATION
        this->stokesQ = 0.0;
        this->stokesU = 0.0;
        this->polarizationBasis = T();
        this->polarizationInitialized = false;
#endif
    };

    friend inline std::ostream &operator<<(std::ostream &stream, const Particle &particle)
    {
        #ifdef STORM_WITH_MPI
                stream << "Particle(ID " << particle.id << " of rank " << particle.rank << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps;
        #else // STORM_WITH_MPI
                stream << "Particle(ID " << particle.id << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps;
        #endif // STORM_WITH_MPI
#ifdef MONTECARLO_POLARIZATION
                stream << ", q " << particle.stokesQ
                       << ", u " << particle.stokesU
                       << ", polInit " << particle.polarizationInitialized;
#endif
                if(particle.radiationState.isDDMC())
                {
                    stream << ", ddmc resident=" << particle.radiationState.isResident()
                           << " comoving=" << particle.radiationState.isComoving();
                    if(particle.radiationState.hasPendingFlux())
                    {
                        stream << " pendingFlux=" << particle.radiationState.pendingFlux;
                    }
                }
                if(particle.radiationState.bypassCellID !=
                   std::numeric_limits<size_t>::max())
                {
                    stream << ", ddmcBypassCellID="
                           << particle.radiationState.bypassCellID;
                }
                return stream << ")";
    }

    inline bool operator==(const Particle &other) const
    {
        #ifdef STORM_WITH_MPI
            return this->id == other.id and this->rank == other.rank;
        #else
            return this->id == other.id;
        #endif
    }

    #ifdef STORM_WITH_MPI
        size_t dump(Serializer *serializer) const override;
        size_t load(const Serializer *serializer, size_t byteOffset) override;
    #endif // STORM_WITH_MPI
};

template<typename T>
Particle<T> cloneParticleWithNewIdentity(const Particle<T> &source)
{
    Particle<T> clone = source;
    clone.id = std::numeric_limits<particle_id_t>::max();
    clone.rngKey = std::numeric_limits<std::uint64_t>::max();
    clone.rngCounter = 0;
    return clone;
}

#ifdef STORM_WITH_MPI
template<typename T>
size_t Particle<T>::dump(Serializer *serializer) const
{
    size_t bytes = 0;
    bytes += serializer->insert(this->rank);
    bytes += serializer->insert(this->id);
    bytes += serializer->insert(this->cellID);
    bytes += serializer->insert(this->sourceCellID);
    bytes += serializer->insert(this->location);
    bytes += serializer->insert(this->velocity);
    bytes += serializer->insert(this->cellIndex);
    bytes += serializer->insert(this->timeLeft);
    bytes += serializer->insert(this->frequency);
    bytes += serializer->insert(this->weight);
    bytes += serializer->insert(this->initialWeight);
    bytes += serializer->insert(this->rngKey);
    bytes += serializer->insert(this->rngCounter);
    bytes += serializer->insert(this->radiationState.flags);
    bytes += serializer->insert(this->radiationState.pendingFlux);
    bytes += serializer->insert(this->radiationState.bypassCellID);
#ifdef MONTECARLO_POLARIZATION
    bytes += serializer->insert(this->stokesQ);
    bytes += serializer->insert(this->stokesU);
    bytes += serializer->insert(this->polarizationBasis);
    bytes += serializer->insert(this->polarizationInitialized);
    bytes += serializer->insert(this->radiationState.pendingMeanScatterings);
#endif
    bytes += serializer->insert(this->steps);
    bytes += serializer->insert(this->on_track);
    bytes += serializer->insert(this->sent);
    #ifdef STORM_DEBUG
    bytes += serializer->insert(this->checkedHere);
    bytes += serializer->insert(this->ghostIndex);
    bytes += serializer->insert(this->newCellValue);
    bytes += serializer->insert(this->nextRank);
    bytes += serializer->insert(this->removedFromRank);
    bytes += serializer->insert(this->sentByRank);
    bytes += serializer->insert(this->lastSeen);
    bytes += serializer->insert(this->lastSeenRank);
    bytes += serializer->insert(this->lastSeenRankBuf);
    bytes += serializer->insert(this->lastSeenIndex);
    #endif // STORM_DEBUG
    #ifdef STORM_WITH_TRACING_HISTORY
    for(size_t h = 0; h < STORM_WITH_TRACING_HISTORY; h++)
    {
        bytes += serializer->insert(this->tracingHistory[h].cellIndex);
        bytes += serializer->insert(this->tracingHistory[h].rank);
        bytes += serializer->insert(this->tracingHistory[h].operation);
        bytes += serializer->insert(this->tracingHistory[h].step);
        bytes += serializer->insert(this->tracingHistory[h].reflected);
        bytes += serializer->insert(this->tracingHistory[h].location);
        bytes += serializer->insert(this->tracingHistory[h].velocity);
        bytes += serializer->insert(this->tracingHistory[h].preReflectLocation);
        bytes += serializer->insert(this->tracingHistory[h].preReflectVelocity);
    }
    bytes += serializer->insert(this->tracingHistoryIndex);
    bytes += serializer->insert(this->tracingHistoryCount);
    #endif // STORM_WITH_TRACING_HISTORY
    return bytes;
}

template<typename T>
size_t Particle<T>::load(const Serializer *serializer, size_t byteOffset)
{
    size_t bytes = 0;
    bytes += serializer->extract(this->rank, byteOffset);
    bytes += serializer->extract(this->id, byteOffset + bytes);
    bytes += serializer->extract(this->cellID, byteOffset + bytes);
    bytes += serializer->extract(this->sourceCellID, byteOffset + bytes);
    bytes += serializer->extract(this->location, byteOffset + bytes);
    bytes += serializer->extract(this->velocity, byteOffset + bytes);
    bytes += serializer->extract(this->cellIndex, byteOffset + bytes);
    bytes += serializer->extract(this->timeLeft, byteOffset + bytes);
    bytes += serializer->extract(this->frequency, byteOffset + bytes);
    bytes += serializer->extract(this->weight, byteOffset + bytes);
    bytes += serializer->extract(this->initialWeight, byteOffset + bytes);
    bytes += serializer->extract(this->rngKey, byteOffset + bytes);
    bytes += serializer->extract(this->rngCounter, byteOffset + bytes);
    bytes += serializer->extract(this->radiationState.flags, byteOffset + bytes);
    bytes += serializer->extract(this->radiationState.pendingFlux, byteOffset + bytes);
    bytes += serializer->extract(this->radiationState.bypassCellID, byteOffset + bytes);
#ifdef MONTECARLO_POLARIZATION
    bytes += serializer->extract(this->stokesQ, byteOffset + bytes);
    bytes += serializer->extract(this->stokesU, byteOffset + bytes);
    bytes += serializer->extract(this->polarizationBasis, byteOffset + bytes);
    bytes += serializer->extract(this->polarizationInitialized, byteOffset + bytes);
    bytes += serializer->extract(this->radiationState.pendingMeanScatterings, byteOffset + bytes);
#endif
    bytes += serializer->extract(this->steps, byteOffset + bytes);
    bytes += serializer->extract(this->on_track, byteOffset + bytes);
    bytes += serializer->extract(this->sent, byteOffset + bytes);
    #ifdef STORM_DEBUG
    bytes += serializer->extract(this->checkedHere, byteOffset + bytes);
    bytes += serializer->extract(this->ghostIndex, byteOffset + bytes);
    bytes += serializer->extract(this->newCellValue, byteOffset + bytes);
    bytes += serializer->extract(this->nextRank, byteOffset + bytes);
    bytes += serializer->extract(this->removedFromRank, byteOffset + bytes);
    bytes += serializer->extract(this->sentByRank, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeen, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenRank, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenRankBuf, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenIndex, byteOffset + bytes);
    #endif // STORM_DEBUG
    #ifdef STORM_WITH_TRACING_HISTORY
    for(size_t h = 0; h < STORM_WITH_TRACING_HISTORY; h++)
    {
        bytes += serializer->extract(this->tracingHistory[h].cellIndex, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].rank, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].operation, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].step, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].reflected, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].location, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].velocity, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].preReflectLocation, byteOffset + bytes);
        bytes += serializer->extract(this->tracingHistory[h].preReflectVelocity, byteOffset + bytes);
    }
    bytes += serializer->extract(this->tracingHistoryIndex, byteOffset + bytes);
    bytes += serializer->extract(this->tracingHistoryCount, byteOffset + bytes);
    #endif // STORM_WITH_TRACING_HISTORY
    return bytes;
}
#endif // STORM_WITH_MPI

} // namespace STORM

// Back-compat alias
template<typename T>
using MonteCarloParticle = STORM::Particle<T>;

#endif // STORM_PARTICLE_HPP
