#ifndef STORM_GPU_KOKKOS_RUNTIME_HPP
#define STORM_GPU_KOKKOS_RUNTIME_HPP

#include <cstddef>

#include "KokkosTypes.hpp"

namespace STORM
{
namespace gpu
{

class KokkosRuntime
{
public:
    KokkosRuntime()
    {
        if(ownerCount_++ == 0 && !Kokkos::is_initialized())
        {
            Kokkos::initialize();
            initializedHere_ = true;
        }
    }

    KokkosRuntime(const KokkosRuntime &) = delete;
    KokkosRuntime &operator=(const KokkosRuntime &) = delete;

    ~KokkosRuntime()
    {
        --ownerCount_;
    }

    // Call after MPI/OFI-owning simulation objects have been destroyed, but
    // before MPI_Finalize. Finalizing from a physics-object destructor can
    // otherwise tear HIP down before an OFI context releases its resources.
    static void Finalize()
    {
        if(ownerCount_ == 0 && initializedHere_ && !Kokkos::is_finalized())
        {
            Kokkos::finalize();
            initializedHere_ = false;
        }
    }

private:
    inline static std::size_t ownerCount_ = 0;
    inline static bool initializedHere_ = false;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_KOKKOS_RUNTIME_HPP
