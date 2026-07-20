#ifndef STORM_DDMC_GHOST_EXCHANGE_HPP
#define STORM_DDMC_GHOST_EXCHANGE_HPP

#include <climits>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi.h>
#include <mpi_utils/types.h>
#include "../../StormError.hpp"
#endif

namespace STORM::ddmc {

#ifdef STORM_WITH_MPI
namespace detail {

template<typename GridT, typename = void>
struct HasExchangeMaps : std::false_type {};

template<typename GridT>
struct HasExchangeMaps<GridT, std::void_t<
    decltype(std::declval<const GridT &>().GetSentProcs()),
    decltype(std::declval<const GridT &>().GetSentPoints()),
    decltype(std::declval<const GridT &>().GetDuplicatedProcs()),
    decltype(std::declval<const GridT &>().GetGhostIndeces())>>
    : std::true_type {};

} // namespace detail
#endif

/*
 * Exchange point-indexed DDMC metadata without assuming that the local
 * duplicated-point graph is symmetric.  The generic point exchange helper
 * uses point-to-point probes and is appropriate for its existing callers,
 * but a DDMC precompute must also be safe when one rank owns no cells or has
 * no local correspondent for a peer which sends it ghost data.  All ranks
 * therefore participate in the same Alltoallv exchange here.
 *
 * The grid contract is the same one used by the existing exchange helper:
 * GetSentPoints()/GetSentProcs() describe outgoing values, while
 * GetDuplicatedProcs()/GetGhostIndeces() describe the incoming ghost slots.
 */
template<typename T, typename GridT>
void ExchangePointMetadata(const GridT &grid, std::vector<T> &data)
{
#ifdef STORM_WITH_MPI
    static_assert(std::is_trivially_copyable_v<T>,
                  "DDMC point metadata must be trivially copyable");

    int initialized = 0;
    MPI_Initialized(&initialized);
    if(!initialized)
        return;

    if constexpr(!detail::HasExchangeMaps<GridT>::value)
    {
        int rankCount = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
        if(rankCount > 1)
        {
            StormError error("DDMC metadata exchange requires grid MPI maps");
            error.addEntry("Grid type", typeid(GridT).name());
            error.addEntry("MPI ranks", rankCount);
            throw error;
        }
        return;
    }
    else
    {

    int rankCount = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    const auto &sentRanks = grid.GetSentProcs();
    const auto &sentPoints = grid.GetSentPoints();
    if(sentRanks.size() != sentPoints.size())
    {
        StormError error("DDMC metadata exchange has mismatched outgoing maps");
        error.addEntry("Ranks", sentRanks.size());
        error.addEntry("Point lists", sentPoints.size());
        throw error;
    }

    std::vector<std::vector<T>> outgoing(static_cast<std::size_t>(rankCount));
    for(std::size_t correspondent = 0; correspondent < sentRanks.size(); ++correspondent)
    {
        int const destination = sentRanks[correspondent];
        if(destination < 0 || destination >= rankCount)
        {
            StormError error("DDMC metadata exchange has an invalid destination rank");
            error.addEntry("Destination", destination);
            error.addEntry("MPI ranks", rankCount);
            throw error;
        }
        std::vector<T> &values = outgoing[static_cast<std::size_t>(destination)];
        for(std::size_t point : sentPoints[correspondent])
        {
            if(point >= data.size())
            {
                StormError error("DDMC metadata exchange references an invalid local point");
                error.addEntry("Point", point);
                error.addEntry("Metadata size", data.size());
                throw error;
            }
            values.push_back(data[point]);
        }
    }

    std::vector<int> sendBytes(static_cast<std::size_t>(rankCount), 0);
    std::vector<int> receiveBytes(static_cast<std::size_t>(rankCount), 0);
    for(int rank = 0; rank < rankCount; ++rank)
    {
        std::size_t const bytes = outgoing[static_cast<std::size_t>(rank)].size() * sizeof(T);
        if(bytes > static_cast<std::size_t>(INT_MAX))
        {
            StormError error("DDMC metadata exchange message is too large");
            error.addEntry("Destination", rank);
            error.addEntry("Bytes", bytes);
            throw error;
        }
        sendBytes[static_cast<std::size_t>(rank)] = static_cast<int>(bytes);
    }
    MPI_Alltoall(sendBytes.data(), 1, MPI_INT,
                 receiveBytes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sendDisplacements(static_cast<std::size_t>(rankCount), 0);
    std::vector<int> receiveDisplacements(static_cast<std::size_t>(rankCount), 0);
    std::size_t totalSendBytes = 0;
    std::size_t totalReceiveBytes = 0;
    for(int rank = 0; rank < rankCount; ++rank)
    {
        if(totalSendBytes > static_cast<std::size_t>(INT_MAX) ||
           totalReceiveBytes > static_cast<std::size_t>(INT_MAX))
        {
            StormError error("DDMC metadata exchange displacement is too large");
            throw error;
        }
        sendDisplacements[static_cast<std::size_t>(rank)] =
            static_cast<int>(totalSendBytes);
        receiveDisplacements[static_cast<std::size_t>(rank)] =
            static_cast<int>(totalReceiveBytes);
        totalSendBytes += static_cast<std::size_t>(sendBytes[static_cast<std::size_t>(rank)]);
        totalReceiveBytes += static_cast<std::size_t>(receiveBytes[static_cast<std::size_t>(rank)]);
    }
    if(totalSendBytes > static_cast<std::size_t>(INT_MAX) ||
       totalReceiveBytes > static_cast<std::size_t>(INT_MAX))
    {
        StormError error("DDMC metadata exchange aggregate is too large");
        throw error;
    }

    std::vector<std::byte> sendBuffer(totalSendBytes);
    for(int rank = 0; rank < rankCount; ++rank)
    {
        auto const &values = outgoing[static_cast<std::size_t>(rank)];
        if(!values.empty())
        {
            std::memcpy(sendBuffer.data() + sendDisplacements[static_cast<std::size_t>(rank)],
                        values.data(), values.size() * sizeof(T));
        }
    }
    std::vector<std::byte> receiveBuffer(totalReceiveBytes);
    MPI_Alltoallv(sendBuffer.empty() ? nullptr : sendBuffer.data(),
                  sendBytes.data(), sendDisplacements.data(), MPI_BYTE,
                  receiveBuffer.empty() ? nullptr : receiveBuffer.data(),
                  receiveBytes.data(), receiveDisplacements.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    const auto &duplicatedRanks = grid.GetDuplicatedProcs();
    const auto &ghostIndices = grid.GetGhostIndeces();
    if(duplicatedRanks.size() != ghostIndices.size())
    {
        StormError error("DDMC metadata exchange has mismatched incoming maps");
        error.addEntry("Ranks", duplicatedRanks.size());
        error.addEntry("Ghost lists", ghostIndices.size());
        throw error;
    }

    data.resize(grid.GetTotalPointNumber(), T{});
    for(std::size_t incoming = 0; incoming < duplicatedRanks.size(); ++incoming)
    {
        int const source = duplicatedRanks[incoming];
        if(source < 0 || source >= rankCount)
        {
            StormError error("DDMC metadata exchange has an invalid source rank");
            error.addEntry("Source", source);
            error.addEntry("MPI ranks", rankCount);
            throw error;
        }
        std::size_t const sourceBytes =
            static_cast<std::size_t>(receiveBytes[static_cast<std::size_t>(source)]);
        if(sourceBytes % sizeof(T) != 0 ||
           sourceBytes / sizeof(T) != ghostIndices[incoming].size())
        {
            StormError error("DDMC metadata exchange incoming count does not match ghost map");
            error.addEntry("Source", source);
            error.addEntry("Received entries", sourceBytes / sizeof(T));
            error.addEntry("Ghost entries", ghostIndices[incoming].size());
            throw error;
        }
        auto const *values = reinterpret_cast<const T *>(
            receiveBuffer.data() + receiveDisplacements[static_cast<std::size_t>(source)]);
        for(std::size_t j = 0; j < ghostIndices[incoming].size(); ++j)
        {
            std::size_t const ghost = ghostIndices[incoming][j];
            if(ghost >= data.size())
            {
                StormError error("DDMC metadata exchange has an invalid ghost index");
                error.addEntry("Ghost", ghost);
                error.addEntry("Metadata size", data.size());
                throw error;
            }
            data[ghost] = values[j];
        }
    }
    }
#else
    (void) grid;
    (void) data;
#endif
}

/* Reduce contributions accumulated in ghost cells back to their owners. */
template<typename T, typename GridT>
void ReducePointContributions(const GridT &grid, std::vector<T> &data)
{
#ifdef STORM_WITH_MPI
    static_assert(std::is_trivially_copyable_v<T>,
                  "DDMC point contributions must be trivially copyable");

    int initialized = 0;
    MPI_Initialized(&initialized);
    if(!initialized)
        return;

    if constexpr(!detail::HasExchangeMaps<GridT>::value)
    {
        int rankCount = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
        if(rankCount > 1)
        {
            StormError error("DDMC contribution reduction requires grid MPI maps");
            error.addEntry("Grid type", typeid(GridT).name());
            error.addEntry("MPI ranks", rankCount);
            throw error;
        }
        return;
    }
    else
    {
        int rankCount = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
        const auto &peerRanks = grid.GetDuplicatedProcs();
        const auto &ghostIndices = grid.GetGhostIndeces();
        const auto &ownerIndices = grid.GetDuplicatedPoints();
        if(peerRanks.size() != ghostIndices.size() ||
           peerRanks.size() != ownerIndices.size())
        {
            StormError error("DDMC contribution reduction has inconsistent maps");
            throw error;
        }

        std::vector<std::vector<T>> outgoing(static_cast<std::size_t>(rankCount));
        for(std::size_t slot = 0; slot < peerRanks.size(); ++slot)
        {
            int const destination = peerRanks[slot];
            if(destination < 0 || destination >= rankCount ||
               ghostIndices[slot].size() != ownerIndices[slot].size())
            {
                StormError error("DDMC contribution reduction has an invalid peer map");
                throw error;
            }
            auto &values = outgoing[static_cast<std::size_t>(destination)];
            for(std::size_t ghost : ghostIndices[slot])
            {
                if(ghost >= data.size())
                {
                    StormError error("DDMC contribution reduction has an invalid ghost index");
                    throw error;
                }
                values.push_back(data[ghost]);
            }
        }

        std::vector<int> sendBytes(static_cast<std::size_t>(rankCount), 0);
        std::vector<int> receiveBytes(static_cast<std::size_t>(rankCount), 0);
        for(int rank = 0; rank < rankCount; ++rank)
        {
            std::size_t const bytes = outgoing[static_cast<std::size_t>(rank)].size() * sizeof(T);
            if(bytes > static_cast<std::size_t>(INT_MAX))
                throw StormError("DDMC contribution reduction message is too large");
            sendBytes[static_cast<std::size_t>(rank)] = static_cast<int>(bytes);
        }
        MPI_Alltoall(sendBytes.data(), 1, MPI_INT,
                     receiveBytes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> sendDisplacements(static_cast<std::size_t>(rankCount), 0);
        std::vector<int> receiveDisplacements(static_cast<std::size_t>(rankCount), 0);
        std::size_t sendTotal = 0;
        std::size_t receiveTotal = 0;
        for(int rank = 0; rank < rankCount; ++rank)
        {
            if(sendTotal > static_cast<std::size_t>(INT_MAX) ||
               receiveTotal > static_cast<std::size_t>(INT_MAX))
                throw StormError("DDMC contribution reduction displacement is too large");
            sendDisplacements[static_cast<std::size_t>(rank)] = static_cast<int>(sendTotal);
            receiveDisplacements[static_cast<std::size_t>(rank)] = static_cast<int>(receiveTotal);
            sendTotal += static_cast<std::size_t>(sendBytes[static_cast<std::size_t>(rank)]);
            receiveTotal += static_cast<std::size_t>(receiveBytes[static_cast<std::size_t>(rank)]);
        }
        if(sendTotal > static_cast<std::size_t>(INT_MAX) ||
           receiveTotal > static_cast<std::size_t>(INT_MAX))
            throw StormError("DDMC contribution reduction aggregate is too large");

        std::vector<std::byte> sendBuffer(sendTotal);
        for(int rank = 0; rank < rankCount; ++rank)
        {
            auto const &values = outgoing[static_cast<std::size_t>(rank)];
            if(!values.empty())
                std::memcpy(sendBuffer.data() + sendDisplacements[static_cast<std::size_t>(rank)],
                            values.data(), values.size() * sizeof(T));
        }
        std::vector<std::byte> receiveBuffer(receiveTotal);
        MPI_Alltoallv(sendBuffer.empty() ? nullptr : sendBuffer.data(),
                      sendBytes.data(), sendDisplacements.data(), MPI_BYTE,
                      receiveBuffer.empty() ? nullptr : receiveBuffer.data(),
                      receiveBytes.data(), receiveDisplacements.data(), MPI_BYTE,
                      MPI_COMM_WORLD);

        data.resize(grid.GetTotalPointNumber(), T{});
        for(std::size_t slot = 0; slot < peerRanks.size(); ++slot)
        {
            int const source = peerRanks[slot];
            std::size_t const bytes = static_cast<std::size_t>(
                receiveBytes[static_cast<std::size_t>(source)]);
            if(bytes % sizeof(T) != 0 ||
               bytes / sizeof(T) != ownerIndices[slot].size())
                throw StormError("DDMC contribution reduction count mismatch");
            auto const *values = reinterpret_cast<const T *>(
                receiveBuffer.data() + receiveDisplacements[static_cast<std::size_t>(source)]);
            for(std::size_t j = 0; j < ownerIndices[slot].size(); ++j)
            {
                std::size_t const owner = ownerIndices[slot][j];
                std::size_t const ghost = ghostIndices[slot][j];
                if(owner >= data.size() || ghost >= data.size())
                    throw StormError("DDMC contribution reduction index is out of range");
                data[owner] += values[j];
                data[ghost] = T{};
            }
        }
    }
#else
    (void) grid;
    (void) data;
#endif
}

} // namespace STORM::ddmc

#endif // STORM_DDMC_GHOST_EXCHANGE_HPP
