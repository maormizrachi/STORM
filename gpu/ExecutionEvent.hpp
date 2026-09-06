#ifndef STORM_GPU_EXECUTION_EVENT_HPP
#define STORM_GPU_EXECUTION_EVENT_HPP

#include "KokkosTypes.hpp"
#include <stdexcept>

namespace STORM::gpu {

// Completion queries let the manager's owning thread progress MPI/OFI while
// the device runs. No background MPI thread or global Kokkos fence is needed.
class ExecutionEvent
{
public:
    ExecutionEvent()
    {
#if defined(KOKKOS_ENABLE_HIP)
        Check(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
#elif defined(KOKKOS_ENABLE_CUDA)
        Check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
#endif
    }
    ~ExecutionEvent()
    {
#if defined(KOKKOS_ENABLE_HIP)
        hipEventDestroy(event_);
#elif defined(KOKKOS_ENABLE_CUDA)
        cudaEventDestroy(event_);
#endif
    }
    ExecutionEvent(const ExecutionEvent &) = delete;
    ExecutionEvent &operator=(const ExecutionEvent &) = delete;

    void Record(const Kokkos::DefaultExecutionSpace &space)
    {
#if defined(KOKKOS_ENABLE_HIP)
        Check(hipEventRecord(event_, space.hip_stream()));
#elif defined(KOKKOS_ENABLE_CUDA)
        Check(cudaEventRecord(event_, space.cuda_stream()));
#else
        space.fence("STORM execution completion");
#endif
    }
    bool Ready() const
    {
#if defined(KOKKOS_ENABLE_HIP)
        const auto status = hipEventQuery(event_);
        if(status == hipErrorNotReady) return false;
        Check(status);
#elif defined(KOKKOS_ENABLE_CUDA)
        const auto status = cudaEventQuery(event_);
        if(status == cudaErrorNotReady) return false;
        Check(status);
#endif
        return true;
    }
private:
#if defined(KOKKOS_ENABLE_HIP)
    hipEvent_t event_{};
    static void Check(hipError_t status)
    {
        if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
    }
#elif defined(KOKKOS_ENABLE_CUDA)
    cudaEvent_t event_{};
    static void Check(cudaError_t status)
    {
        if(status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
#endif
};
} // namespace STORM::gpu
#endif
