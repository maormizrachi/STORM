#ifndef STORM_GPU_DEVICE_SOURCE_CONTEXT_HPP
#define STORM_GPU_DEVICE_SOURCE_CONTEXT_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "../radiation/source/SourceCore.hpp"
#include "../types.hpp"
#include "KokkosLocalTransportExecutor.hpp"

namespace STORM
{
namespace gpu
{

class GreyIMCData;

struct DeviceSourceContext
{
    KokkosLocalTransportExecutor *executor = nullptr;
    std::unique_ptr<KokkosLocalTransportExecutor> *executorStorage = nullptr;
    std::size_t gpuMaxInnerSteps = 1;
    GreyIMCData *gpuData = nullptr;
    const source::Plan *plan = nullptr;
    std::uint64_t particleRngSeed = 0;
    std::uint64_t creationRank = 0;
    std::int32_t rank = 0;
    particle_id_t firstParticleId = 0;
    dt_t fullDt = 0.0;
    double speedOfLight = 0.0;
    double invClight2 = 0.0;
    std::uint8_t sampleFrequency = 0;
    std::uint8_t applyLabFrame = 0;
    std::size_t emittedCount = 0;
};

template<typename PhysicsT, typename = void>
struct HasDeviceSourceGeneration : std::false_type
{};

template<typename PhysicsT>
struct HasDeviceSourceGeneration<PhysicsT, std::void_t<
    decltype(std::declval<const PhysicsT &>().
                 SupportsDeviceSourceGeneration()),
    decltype(std::declval<PhysicsT &>().preStepOnDevice(
        std::declval<DeviceSourceContext &>()))>>
    : std::true_type
{};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_DEVICE_SOURCE_CONTEXT_HPP
