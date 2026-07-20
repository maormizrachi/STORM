#include "radiation/ddmc/DDMCGhostExchange.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi.h>

namespace {

struct TwoRankHalo
{
    std::vector<int> duplicatedRanks{1};
    std::vector<std::vector<std::size_t>> duplicatedPoints{{0}};
    std::vector<std::vector<std::size_t>> ghostIndices{{1}};

    explicit TwoRankHalo(int rank): duplicatedRanks{1 - rank} {}

    const std::vector<int> &GetDuplicatedProcs() const { return duplicatedRanks; }
    const std::vector<std::vector<std::size_t>> &GetDuplicatedPoints() const
    {
        return duplicatedPoints;
    }
    const std::vector<std::vector<std::size_t>> &GetGhostIndeces() const
    {
        return ghostIndices;
    }
    std::size_t GetTotalPointNumber() const { return 2; }
};

template<typename T>
bool checkExchange(const TwoRankHalo &grid, T owned, T expectedGhost)
{
    std::vector<T> values{owned};
    STORM::ddmc::ExchangePointMetadata(grid, values);
    return values.size() == 2 && values[0] == owned && values[1] == expectedGhost;
}

template<typename T>
bool checkReduction(const TwoRankHalo &grid, T owned, T ghost, T expectedOwned)
{
    std::vector<T> values{owned, ghost};
    STORM::ddmc::ReducePointContributions(grid, values);
    return values.size() == 2 && values[0] == expectedOwned && values[1] == T{};
}

} // namespace
#endif

int main(int argc, char **argv)
{
#ifdef STORM_WITH_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int localOk = ranks == 2 ? 1 : 0;
    if(ranks == 2)
    {
        TwoRankHalo grid(rank);
        auto check = [&](bool result, const char *name) {
            if(!result)
            {
                std::cerr << "rank " << rank << " failed " << name << "\n";
                localOk = 0;
            }
        };
        check(checkExchange<double>(grid, 1.0 + rank, 2.0 - rank), "double exchange");
        check(checkExchange<std::size_t>(
            grid, static_cast<std::size_t>(10 + rank), static_cast<std::size_t>(11 - rank)),
            "size_t exchange");
        check(checkExchange<std::uint8_t>(
            grid, static_cast<std::uint8_t>(rank == 0), static_cast<std::uint8_t>(rank != 0)),
            "flag exchange");
        check(checkExchange<std::array<double, 3>>(
            grid, std::array<double, 3>{1.0 + rank, 2.0, 3.0},
            std::array<double, 3>{2.0 - rank, 2.0, 3.0}), "point exchange");

        check(checkReduction<double>(grid, 10.0 + rank, 2.0 + rank, 13.0), "double reduction");
        check(checkReduction<std::size_t>(
            grid, static_cast<std::size_t>(20 + rank), static_cast<std::size_t>(3 + rank),
            static_cast<std::size_t>(24)), "size_t reduction");
        check(checkReduction<std::uint8_t>(
            grid, static_cast<std::uint8_t>(rank), static_cast<std::uint8_t>(1),
            static_cast<std::uint8_t>(rank + 1)), "flag reduction");
    }

    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(rank == 0 && globalOk == 0)
        std::cerr << "DDMC ghost exchange regression requires exactly two ranks and bit-safe maps\n";
    MPI_Finalize();
    return globalOk == 1 ? 0 : 1;
#else
    (void) argc;
    (void) argv;
    return 0;
#endif
}
