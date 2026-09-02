#ifndef STORM_GPU_PROFILE_REGION_HPP
#define STORM_GPU_PROFILE_REGION_HPP

#if defined(STORM_WITH_GPU) && defined(STORM_GPU_PROFILING)
#include <Kokkos_Core.hpp>

namespace STORM
{
namespace gpu
{

class ProfileRegion
{
public:
    explicit ProfileRegion(const char *name) : active_(Kokkos::is_initialized())
    {
        if(active_)
        {
            Kokkos::Profiling::pushRegion(name);
        }
    }

    ~ProfileRegion()
    {
        if(active_)
        {
            Kokkos::Profiling::popRegion();
        }
    }

    ProfileRegion(const ProfileRegion &) = delete;
    ProfileRegion &operator=(const ProfileRegion &) = delete;

private:
    bool active_;
};

} // namespace gpu
} // namespace STORM

#define STORM_PROFILE_CONCAT_IMPL(left, right) left##right
#define STORM_PROFILE_CONCAT(left, right) STORM_PROFILE_CONCAT_IMPL(left, right)
#define STORM_PROFILE_REGION(name) \
    ::STORM::gpu::ProfileRegion STORM_PROFILE_CONCAT(stormProfileRegion_, __LINE__)(name)
#else
#define STORM_PROFILE_REGION(name) ((void)0)
#endif

#endif // STORM_GPU_PROFILE_REGION_HPP
