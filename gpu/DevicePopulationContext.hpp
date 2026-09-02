#ifndef STORM_GPU_DEVICE_POPULATION_CONTEXT_HPP
#define STORM_GPU_DEVICE_POPULATION_CONTEXT_HPP

#include <cstddef>
#include <cstdint>

#ifdef STORM_WITH_MPI
#include <mpi.h>
#else
using MPI_Comm = int;
#endif

#include "../population/CombCore.hpp"
#include "../types.hpp"
#include "KokkosLocalTransportExecutor.hpp"

namespace STORM
{
namespace gpu
{

struct DevicePopulationContext
{
    KokkosLocalTransportExecutor *executor = nullptr;
    std::size_t cellCount = 0;
    std::uint64_t activationEpoch = 0;
    rank_t rank = 0;
    MPI_Comm communicator = 0;
    comb::Parameters parameters{};
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_DEVICE_POPULATION_CONTEXT_HPP
