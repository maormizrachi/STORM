#ifndef STORM_EXAMPLES_MPI_PARTICLE_DTYPE_HPP
#define STORM_EXAMPLES_MPI_PARTICLE_DTYPE_HPP

#ifdef STORM_WITH_MPI

#include <cassert>
#include <mpi.h>
#include <mpi_utils/MPI_complex_dtype.hpp>
#include "particle/Particle.hpp"
#include "Vector3D.hpp"

template<typename Grid>
struct MPI_has_complex_dtype<STORM::Particle<Vector3D, Grid>> : std::true_type
{
    using ParticleT = STORM::Particle<Vector3D, Grid>;

    static MPI_Datatype getDatatype()
    {
        static MPI_Datatype dtype = createDatatype();
        return dtype;
    }

private:
    static MPI_Datatype createDatatype()
    {
        ParticleT dummy{};

        constexpr int base_count = 14;
#ifdef STORM_DEBUG
        constexpr int debug_count = 10;
#else
        constexpr int debug_count = 0;
#endif
#ifdef STORM_WITH_TRACING_HISTORY
        constexpr int history_per_entry = 9;
        constexpr int history_count = STORM_WITH_TRACING_HISTORY * history_per_entry + 2;
#else
        constexpr int history_count = 0;
#endif
        constexpr int total_count = base_count + debug_count + history_count;

        MPI_Datatype types[total_count];
        int blocklengths[total_count];
        MPI_Aint displacements[total_count];

        MPI_Aint base;
        MPI_Get_address(&dummy, &base);

        int idx = 0;
        auto add_field = [&](auto &field, MPI_Datatype mpi_type, int count = 1)
        {
            MPI_Aint addr;
            MPI_Get_address(&field, &addr);
            types[idx] = mpi_type;
            blocklengths[idx] = count;
            displacements[idx] = addr - base;
            idx++;
        };

        add_field(dummy.rank, MPI_INT);
        add_field(dummy.id, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.cellID, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.sourceCellID, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.location.x, MPI_DOUBLE, 3);
        add_field(dummy.velocity.x, MPI_DOUBLE, 3);
        add_field(dummy.cellIndex, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.timeLeft, MPI_DOUBLE);
        add_field(dummy.frequency, MPI_DOUBLE);
        add_field(dummy.weight, MPI_DOUBLE);
        add_field(dummy.initialWeight, MPI_DOUBLE);
        add_field(dummy.steps, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.on_track, MPI_CXX_BOOL);
        add_field(dummy.sent, MPI_CXX_BOOL);

#ifdef STORM_DEBUG
        add_field(dummy.checkedHere, MPI_CXX_BOOL);
        add_field(dummy.ghostIndex, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.newCellValue.x, MPI_DOUBLE, 3);
        add_field(dummy.nextRank, MPI_INT);
        add_field(dummy.removedFromRank, MPI_CXX_BOOL);
        add_field(dummy.sentByRank, MPI_INT);
        add_field(dummy.lastSeen, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.lastSeenRank, MPI_INT);
        add_field(dummy.lastSeenRankBuf, MPI_INT);
        add_field(dummy.lastSeenIndex, MPI_UNSIGNED_LONG_LONG);
#endif

#ifdef STORM_WITH_TRACING_HISTORY
        for(int h = 0; h < STORM_WITH_TRACING_HISTORY; h++)
        {
            add_field(dummy.tracingHistory[h].cellIndex, MPI_UNSIGNED_LONG_LONG);
            add_field(dummy.tracingHistory[h].rank, MPI_INT);
            add_field(dummy.tracingHistory[h].operation, MPI_INT);
            add_field(dummy.tracingHistory[h].step, MPI_UNSIGNED_LONG_LONG);
            add_field(dummy.tracingHistory[h].reflected, MPI_CXX_BOOL);
            add_field(dummy.tracingHistory[h].location.x, MPI_DOUBLE, 3);
            add_field(dummy.tracingHistory[h].velocity.x, MPI_DOUBLE, 3);
            add_field(dummy.tracingHistory[h].preReflectLocation.x, MPI_DOUBLE, 3);
            add_field(dummy.tracingHistory[h].preReflectVelocity.x, MPI_DOUBLE, 3);
        }
        add_field(dummy.tracingHistoryIndex, MPI_UNSIGNED_LONG_LONG);
        add_field(dummy.tracingHistoryCount, MPI_UNSIGNED_LONG_LONG);
#endif

        assert(idx == total_count);

        MPI_Datatype raw_type;
        MPI_Type_create_struct(total_count, blocklengths, displacements, types, &raw_type);

        MPI_Datatype resized_type;
        MPI_Type_create_resized(raw_type, 0, sizeof(ParticleT), &resized_type);
        MPI_Type_commit(&resized_type);
        MPI_Type_free(&raw_type);

        return resized_type;
    }
};

#endif // STORM_WITH_MPI

#endif // STORM_EXAMPLES_MPI_PARTICLE_DTYPE_HPP
