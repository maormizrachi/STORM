#ifndef STORM_RADIATION_DDMC_CORE_CHECK_HPP
#define STORM_RADIATION_DDMC_CORE_CHECK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "DDMCSampling.hpp"

#ifdef STORM_WITH_GPU
#include <Kokkos_Core.hpp>
#endif

namespace STORM::ddmc
{

inline void RequireSharedSamplingCore()
{
    static bool checked = false;
    if(checked)
    {
        return;
    }
    checked = true;
    const double boundaries[] = {1.0, 2.0, 4.0};
    const double cdf[] = {0.0, 1.0, 3.0};
    const double sampled = SampleFrequencyInGroupFromCellCdf(
        boundaries, cdf, 2, 0, 1, 0.25);
    if(std::abs(sampled - 2.5) > 1.0e-14)
    {
        throw std::runtime_error(
            "DDMC in-group CDF sampling check failed");
    }
    const double planck = SamplePlanckFrequencyInGroup(
        boundaries, 2, 1, 1.0, 0.5);
    if(!(planck > boundaries[1] && planck < boundaries[2]))
    {
        throw std::runtime_error(
            "DDMC in-group Planck sampling check failed");
    }
}

#ifdef STORM_WITH_GPU
inline void RequireHostDeviceSamplingKernelMatch()
{
    static bool checked = false;
    if(checked)
    {
        return;
    }
    if(!Kokkos::is_initialized())
    {
        throw std::runtime_error(
            "DDMC GPU sampling check requires Kokkos initialization");
    }
    checked = true;
    Kokkos::View<double *> boundaries("ddmc_check_boundaries", 3);
    Kokkos::View<double *> cdf("ddmc_check_cdf", 3);
    Kokkos::View<double *> output("ddmc_check_output", 2);
    const double hostBoundaries[] = {1.0, 2.0, 4.0};
    const double hostCdf[] = {0.0, 1.0, 3.0};
    Kokkos::View<double *>::HostMirror hostBoundaryView =
        Kokkos::create_mirror_view(boundaries);
    Kokkos::View<double *>::HostMirror hostCdfView =
        Kokkos::create_mirror_view(cdf);
    for(std::size_t i = 0; i < 3; ++i)
    {
        hostBoundaryView(i) = hostBoundaries[i];
        hostCdfView(i) = hostCdf[i];
    }
    Kokkos::deep_copy(boundaries, hostBoundaryView);
    Kokkos::deep_copy(cdf, hostCdfView);
    Kokkos::parallel_for(
        "storm_ddmc_sampling_core_check",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int)
        {
            output(0) = SampleFrequencyInGroupFromCellCdf(
                boundaries.data(), cdf.data(), 2, 0, 1, 0.25);
            output(1) = SamplePlanckFrequencyInGroup(
                boundaries.data(), 2, 1, 1.0, 0.5);
        });
    Kokkos::View<double *>::HostMirror hostOutput =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), output);
    const double expectedCdf = SampleFrequencyInGroupFromCellCdf(
        hostBoundaries, hostCdf, 2, 0, 1, 0.25);
    const double expectedPlanck = SamplePlanckFrequencyInGroup(
        hostBoundaries, 2, 1, 1.0, 0.5);
    const double scale = std::max(1.0, std::abs(expectedPlanck));
    if(std::abs(hostOutput(0) - expectedCdf) > 1.0e-14 ||
       std::abs(hostOutput(1) - expectedPlanck) > 1.0e-12 * scale)
    {
        throw std::runtime_error(
            std::string("DDMC host/device sampling mismatch: host=") +
            std::to_string(expectedPlanck) +
            " device=" + std::to_string(hostOutput(1)));
    }
}
#else
inline void RequireHostDeviceSamplingKernelMatch()
{
}
#endif

} // namespace STORM::ddmc

#endif // STORM_RADIATION_DDMC_CORE_CHECK_HPP
