#ifndef RDMONT_PARTICLE_HPP
#define RDMONT_PARTICLE_HPP

#include <sstream>
#include <vector>
#include <limits>
#ifdef RDMONT_WITH_MPI
    #include <mpi.h>
    #include <functional>
    #include "mpi_utils/mpi_commands.hpp"
    #include "mpi_utils/Serializer.hpp"
#endif // RDMONT_WITH_MPI
#include "monte/RDMontError.hpp"
#include "monte/particle/ParticleStatus.hpp"

#define EPSILON 1e-12

namespace RDMont {

using dt_t = double;
using distance_t = double;

#ifdef RDMONT_WITH_TRACING_HISTORY

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
#endif // RDMONT_WITH_TRACING_HISTORY

template<typename T, typename Grid>
struct Particle
                    #ifdef RDMONT_WITH_MPI
                        : public Serializable
                    #endif // RDMONT_WITH_MPI
{
    #ifdef RDMONT_WITH_MPI
        rank_t rank = -1;
        #ifdef RDMONT_DEBUG
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
        #endif // RDMONT_DEBUG
    #endif // RDMONT_WITH_MPI
    size_t id = std::numeric_limits<size_t>::max();
    size_t cellID = std::numeric_limits<size_t>::max();
    T location = T(std::numeric_limits<typename T::value_type>::max());
    T velocity = T(std::numeric_limits<typename T::value_type>::max());
    size_t cellIndex = std::numeric_limits<size_t>::max();
    dt_t timeLeft = std::numeric_limits<dt_t>::max();
    double frequency = std::numeric_limits<double>::max();
    double weight = std::numeric_limits<double>::max();
    double initialWeight = std::numeric_limits<double>::max();
    size_t steps = 0;
    bool on_track = false;
    bool sent = false;

    #ifdef RDMONT_WITH_TRACING_HISTORY
        ParticleHistory<T> tracingHistory[RDMONT_WITH_TRACING_HISTORY] = {};
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
            this->tracingHistoryIndex = (this->tracingHistoryIndex + 1) % RDMONT_WITH_TRACING_HISTORY;
            if(this->tracingHistoryCount < RDMONT_WITH_TRACING_HISTORY)
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
            size_t lastIdx = (this->tracingHistoryIndex + RDMONT_WITH_TRACING_HISTORY - 1) % RDMONT_WITH_TRACING_HISTORY;
            this->tracingHistory[lastIdx].reflected = true;
            this->tracingHistory[lastIdx].preReflectLocation = locBeforeReflect;
            this->tracingHistory[lastIdx].preReflectVelocity = velBeforeReflect;
        }

        inline void addTracingHistoryToError(RDMontError &eo) const
        {
            eo.addEntry("Tracing History Count", this->tracingHistoryCount);
            for(size_t h = 0; h < this->tracingHistoryCount; h++)
            {
                size_t idx = (this->tracingHistoryIndex - this->tracingHistoryCount + h + RDMONT_WITH_TRACING_HISTORY) % RDMONT_WITH_TRACING_HISTORY;
                const ParticleHistory<T> &hist = this->tracingHistory[idx];
                std::string prefix = "History[" + std::to_string(h) + "] ";
                eo.addEntry(prefix + "Cell", hist.cellIndex);
                eo.addEntry(prefix + "Rank", hist.rank);
                eo.addEntry(prefix + "Op", ParticleStatusToString(hist.operation));
                eo.addEntry(prefix + "Step", hist.step);
            }
        }
    #endif // RDMONT_WITH_TRACING_HISTORY

    explicit Particle(size_t id_ = std::numeric_limits<size_t>::max(), const T &location_ = T(std::numeric_limits<double>::max()), const T &velocity_ = T(std::numeric_limits<double>::max()), dt_t timeLeft_ = dt_t(std::numeric_limits<double>::max())):
        id(id_), location(location_), velocity(velocity_), cellIndex(std::numeric_limits<size_t>::max()), timeLeft(timeLeft_), frequency(std::numeric_limits<double>::max()), weight(0), initialWeight(0), steps(0), on_track(false)
    {
        #ifdef RDMONT_DEBUG
        this->checkedHere = true;
        this->ghostIndex = std::numeric_limits<size_t>::max();
        this->newCellValue = T(std::numeric_limits<double>::max());
        this->nextRank = std::numeric_limits<rank_t>::max();
        this->removedFromRank = false;
        this->sentByRank = std::numeric_limits<rank_t>::max();
        #endif // RDMONT_DEBUG
    };

    std::pair<size_t, distance_t> distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const;

    friend inline std::ostream &operator<<(std::ostream &stream, const Particle &particle)
    {
        #ifdef RDMONT_WITH_MPI
                return stream << "Particle(ID " << particle.id << " of rank " << particle.rank << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps << ")";
        #else // RDMONT_WITH_MPI
                return stream << "Particle(ID " << particle.id << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps << ")";
        #endif // RDMONT_WITH_MPI
    }

    inline bool operator==(const Particle &other) const
    {
        #ifdef RDMONT_WITH_MPI
            return this->id == other.id and this->rank == other.rank;
        #else
            return this->id == other.id;
        #endif
    }

    #ifdef RDMONT_WITH_MPI
        size_t dump(Serializer *serializer) const override;
        size_t load(const Serializer *serializer, size_t byteOffset) override;
    #endif // RDMONT_WITH_MPI
};

template<typename T, typename Grid>
std::pair<size_t, dt_t> Particle<T, Grid>::distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const
{
    std::pair<size_t, dt_t> best = {std::numeric_limits<size_t>::max(), std::numeric_limits<dt_t>::max()};
    size_t &min_face = best.first;
    dt_t &min_alpha = best.second;

    const double velocityAbs = EPSILON * fastabs(this->velocity);
    const auto &faces = grid.GetCellFaces(this->cellIndex);
    size_t Nfaces = faces.size();

    for(size_t i = 0; i < Nfaces; ++i)
    {
        const size_t &faceIdx = faces[i];
        const T &normal = normalsOfCell[i];

        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(BOOST_UNLIKELY(normalVelocityScalarProd >= -velocityAbs))
        {
            continue;
        }
        const T &pointOnFace = pointsOnFaces[i];

        dt_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;

        __builtin_prefetch(&Nfaces, 0, 0);

        if(i < Nfaces - 1)
        {
            const T *nextNormal = &normalsOfCell[i + 1];
            __builtin_prefetch(nextNormal, 0, 2);
        }

        __builtin_prefetch(&min_alpha, 1, 3);
        __builtin_prefetch(&min_face, 1, 3);
        if(BOOST_UNLIKELY(alpha < min_alpha))
        {
            if(alpha > 0)
            {
                min_alpha = alpha;
                min_face = faceIdx;
            }
        }
    }

    if(min_alpha != std::numeric_limits<distance_t>::max())
    {
        return best;
    }

    if(grid.IsPointOutsideBox(this->location))
    {
        RDMontError eo("Particle is outside the domain, but still considered");
        #ifdef RDMONT_WITH_MPI
            rank_t rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            eo.addEntry("Rank", rank);
        #endif // RDMONT_WITH_MPI
        eo.addEntry("Particle", *this);
        eo.addEntry("Cell Index", this->cellIndex);
        #ifdef RDMONT_WITH_TRACING_HISTORY
            this->addTracingHistoryToError(eo);
        #endif // RDMONT_WITH_TRACING_HISTORY
        throw eo;
    }
    size_t realContainingCell = grid.GetContainingCell(this->location);
    if(realContainingCell != this->cellIndex)
    {
        RDMontError eo("Particle::distanceToNearestFace: the containing cellIndex is incorrect");
        eo.addEntry("Real containing cell index", realContainingCell);
        eo.addEntry("Declared containing cell", this->cellIndex);
        eo.addEntry("Particle", (*this));
        #ifdef RDMONT_WITH_TRACING_HISTORY
            this->addTracingHistoryToError(eo);
        #endif // RDMONT_WITH_TRACING_HISTORY
        throw eo;
    }

    RDMontError eo("Particle::distanceToNearestFace: no face intersection found");
    #ifdef RDMONT_WITH_MPI
        rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        eo.addEntry("Rank", rank);
    #endif // RDMONT_WITH_MPI
    eo.addEntry("Particle", *this);
    eo.addEntry("Cell Index", this->cellIndex);
    #ifdef RDMONT_WITH_TRACING_HISTORY
        this->addTracingHistoryToError(eo);
    #endif // RDMONT_WITH_TRACING_HISTORY
    throw eo;
}

#ifdef RDMONT_WITH_MPI
template<typename T, typename Grid>
size_t Particle<T, Grid>::dump(Serializer *serializer) const
{
    size_t bytes = 0;
    bytes += serializer->insert(this->rank);
    bytes += serializer->insert(this->id);
    bytes += serializer->insert(this->cellID);
    bytes += serializer->insert(this->location);
    bytes += serializer->insert(this->velocity);
    bytes += serializer->insert(this->cellIndex);
    bytes += serializer->insert(this->timeLeft);
    bytes += serializer->insert(this->frequency);
    bytes += serializer->insert(this->weight);
    bytes += serializer->insert(this->initialWeight);
    bytes += serializer->insert(this->steps);
    bytes += serializer->insert(this->on_track);
    bytes += serializer->insert(this->sent);
    #ifdef RDMONT_DEBUG
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
    #endif // RDMONT_DEBUG
    #ifdef RDMONT_WITH_TRACING_HISTORY
    for(size_t h = 0; h < RDMONT_WITH_TRACING_HISTORY; h++)
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
    #endif // RDMONT_WITH_TRACING_HISTORY
    return bytes;
}

template<typename T, typename Grid>
size_t Particle<T, Grid>::load(const Serializer *serializer, size_t byteOffset)
{
    size_t bytes = 0;
    bytes += serializer->extract(this->rank, byteOffset);
    bytes += serializer->extract(this->id, byteOffset + bytes);
    bytes += serializer->extract(this->cellID, byteOffset + bytes);
    bytes += serializer->extract(this->location, byteOffset + bytes);
    bytes += serializer->extract(this->velocity, byteOffset + bytes);
    bytes += serializer->extract(this->cellIndex, byteOffset + bytes);
    bytes += serializer->extract(this->timeLeft, byteOffset + bytes);
    bytes += serializer->extract(this->frequency, byteOffset + bytes);
    bytes += serializer->extract(this->weight, byteOffset + bytes);
    bytes += serializer->extract(this->initialWeight, byteOffset + bytes);
    bytes += serializer->extract(this->steps, byteOffset + bytes);
    bytes += serializer->extract(this->on_track, byteOffset + bytes);
    bytes += serializer->extract(this->sent, byteOffset + bytes);
    #ifdef RDMONT_DEBUG
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
    #endif // RDMONT_DEBUG
    #ifdef RDMONT_WITH_TRACING_HISTORY
    for(size_t h = 0; h < RDMONT_WITH_TRACING_HISTORY; h++)
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
    #endif // RDMONT_WITH_TRACING_HISTORY
    return bytes;
}
#endif // RDMONT_WITH_MPI

} // namespace RDMont

// Back-compat alias
template<typename T, typename Grid>
using MonteCarloParticle = RDMont::Particle<T, Grid>;

#endif // RDMONT_PARTICLE_HPP
