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

#define EPSILON 1e-12

namespace STORM {

using namespace STORM::fallback;

using dt_t = double;

#ifdef STORM_WITH_TRACING_HISTORY

template<typename T>
struct ParticleHistory
{
    size_t cellIndex = 0;
    int rank = -1;
    int operation = 0;
    size_t step = 0;
    bool reflected = false;
    T location = T();
    T velocity = T();
    T preReflectLocation = T();
    T preReflectVelocity = T();
};
#endif // STORM_WITH_TRACING_HISTORY

template<typename T, typename Grid>
struct Particle
                    #ifdef STORM_WITH_MPI
                        : public Serializable
                    #endif // STORM_WITH_MPI
{
    #ifdef STORM_WITH_MPI
        rank_t rank = -1;
        #ifdef STORM_DEBUG
            size_t cellIndexInPrevRank = std::numeric_limits<size_t>::max();
            T previousLocation = T(std::numeric_limits<double>::max());
            size_t particleTHInLastRank = std::numeric_limits<size_t>::max();
            size_t particleIndexInLastRank = std::numeric_limits<size_t>::max();
            bool checkedHere = true;
            size_t ghostIndex = std::numeric_limits<size_t>::max();
            T newCellValue = T(std::numeric_limits<double>::max());
            rank_t nextRank = std::numeric_limits<rank_t>::max();
            rank_t sentByRank = std::numeric_limits<rank_t>::max();
            bool removedFromRank = false;
            size_t lastSeen = 0;
            rank_t lastSeenRank = std::numeric_limits<rank_t>::max();
            rank_t lastSeenRankBuf = std::numeric_limits<rank_t>::max();
            size_t lastSeenIndex = std::numeric_limits<size_t>::max();
        #endif // STORM_DEBUG
    #endif // STORM_WITH_MPI
    size_t id = std::numeric_limits<size_t>::max();
    size_t cellID = std::numeric_limits<size_t>::max();
    size_t sourceCellID = std::numeric_limits<size_t>::max();
    T location = T(std::numeric_limits<typename T::coord_type>::max());
    T velocity = T(std::numeric_limits<typename T::coord_type>::max());
    size_t cellIndex = std::numeric_limits<size_t>::max();
    dt_t timeLeft = std::numeric_limits<dt_t>::max();
    double frequency = std::numeric_limits<double>::max();
    double weight = std::numeric_limits<double>::max();
    double initialWeight = std::numeric_limits<double>::max();
    RadiationTransportState<T> radiationState{};
#ifdef MONTECARLO_POLARIZATION
    double stokesQ = 0.0;
    double stokesU = 0.0;
    T polarizationBasis = T();
    bool polarizationInitialized = false;
#endif
    size_t steps = 0;
    bool on_track = false;
    bool sent = false;

    #ifdef STORM_WITH_TRACING_HISTORY
        ParticleHistory<T> tracingHistory[STORM_WITH_TRACING_HISTORY] = {};
        size_t tracingHistoryIndex = 0;
        size_t tracingHistoryCount = 0;

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

    explicit Particle(size_t id_ = std::numeric_limits<size_t>::max(), const T &location_ = T(std::numeric_limits<double>::max()), const T &velocity_ = T(std::numeric_limits<double>::max()), dt_t timeLeft_ = dt_t(std::numeric_limits<double>::max())):
        id(id_), location(location_), velocity(velocity_), cellIndex(std::numeric_limits<size_t>::max()), timeLeft(timeLeft_), frequency(std::numeric_limits<double>::max()), weight(0), initialWeight(0), steps(0), on_track(false)
    {
        #ifdef STORM_DEBUG
        this->checkedHere = true;
        this->ghostIndex = std::numeric_limits<size_t>::max();
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

    std::pair<size_t, typename T::coord_type> distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const;

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

template<typename T, typename Grid>
std::pair<size_t, typename T::coord_type> Particle<T, Grid>::distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const
{
    using coord_t = typename T::coord_type;
    std::pair<size_t, coord_t> best = {std::numeric_limits<size_t>::max(), std::numeric_limits<coord_t>::max()};
    size_t &min_face = best.first;
    coord_t &min_alpha = best.second;

    const double velocityAbs = EPSILON * fastabs(this->velocity);
    const auto &faces = grid.GetCellFaces(this->cellIndex);
    size_t Nfaces = faces.size();

    for(size_t i = 0; i < Nfaces; ++i)
    {
        const size_t &faceIdx = faces[i];
        const T &normal = normalsOfCell[i];

        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(__builtin_expect(normalVelocityScalarProd >= -velocityAbs, 0))
        {
            continue;
        }
        const T &pointOnFace = pointsOnFaces[i];

        coord_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;

        __builtin_prefetch(&Nfaces, 0, 0);

        if(i < Nfaces - 1)
        {
            const T *nextNormal = &normalsOfCell[i + 1];
            __builtin_prefetch(nextNormal, 0, 2);
        }

        __builtin_prefetch(&min_alpha, 1, 3);
        __builtin_prefetch(&min_face, 1, 3);
        if(__builtin_expect(alpha < min_alpha, 0))
        {
            if(alpha > 0)
            {
                min_alpha = alpha;
                min_face = faceIdx;
            }
        }
    }

    if(min_alpha != std::numeric_limits<coord_t>::max())
    {
        return best;
    }

    if(grid.IsPointOutsideBox(this->location))
    {
        StormError eo("Particle is outside the domain, but still considered");
        #ifdef STORM_WITH_MPI
            rank_t rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            eo.addEntry("Rank", rank);
        #endif // STORM_WITH_MPI
        eo.addEntry("Particle", *this);
        eo.addEntry("Cell Index", this->cellIndex);
        #ifdef STORM_WITH_TRACING_HISTORY
            this->addTracingHistoryToError(eo);
        #endif // STORM_WITH_TRACING_HISTORY
        throw eo;
    }
    size_t realContainingCell = grid.GetContainingCell(this->location);
    if(realContainingCell != this->cellIndex)
    {
        StormError eo("Particle::distanceToNearestFace: the containing cellIndex is incorrect");
        eo.addEntry("Real containing cell index", realContainingCell);
        eo.addEntry("Declared containing cell", this->cellIndex);
        eo.addEntry("Particle", (*this));
        #ifdef STORM_WITH_TRACING_HISTORY
            this->addTracingHistoryToError(eo);
        #endif // STORM_WITH_TRACING_HISTORY
        throw eo;
    }

    StormError eo("Particle::distanceToNearestFace: no face intersection found");
    #ifdef STORM_WITH_MPI
        rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        eo.addEntry("Rank", rank);
    #endif // STORM_WITH_MPI
    eo.addEntry("Particle", *this);
    eo.addEntry("Cell Index", this->cellIndex);
    #ifdef STORM_WITH_TRACING_HISTORY
        this->addTracingHistoryToError(eo);
    #endif // STORM_WITH_TRACING_HISTORY
    throw eo;
}

#ifdef STORM_WITH_MPI
template<typename T, typename Grid>
size_t Particle<T, Grid>::dump(Serializer *serializer) const
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

template<typename T, typename Grid>
size_t Particle<T, Grid>::load(const Serializer *serializer, size_t byteOffset)
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
template<typename T, typename Grid>
using MonteCarloParticle = STORM::Particle<T, Grid>;

#endif // STORM_PARTICLE_HPP
