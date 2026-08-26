#ifndef STORM_RADIATION_TRANSPORT_PORTABILITY_HPP
#define STORM_RADIATION_TRANSPORT_PORTABILITY_HPP

#include <cmath>

#ifdef STORM_WITH_GPU
#include <Kokkos_Core.hpp>
#define STORM_TRANSPORT_INLINE KOKKOS_INLINE_FUNCTION
#define STORM_TRANSPORT_ACCUMULATE(target, value) \
    Kokkos::atomic_add(&(target), (value))
#else
#define STORM_TRANSPORT_INLINE inline
#define STORM_TRANSPORT_ACCUMULATE(target, value) \
    ((target) += (value))
#endif

namespace STORM
{
namespace transport
{

STORM_TRANSPORT_INLINE
double Abs(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::abs(value);
#else
    return std::abs(value);
#endif
}

STORM_TRANSPORT_INLINE
double Cos(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::cos(value);
#else
    return std::cos(value);
#endif
}

STORM_TRANSPORT_INLINE
double Exp(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::exp(value);
#else
    return std::exp(value);
#endif
}

STORM_TRANSPORT_INLINE
double Expm1(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::expm1(value);
#else
    return std::expm1(value);
#endif
}

STORM_TRANSPORT_INLINE
bool IsFinite(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

STORM_TRANSPORT_INLINE
double Log(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::log(value);
#else
    return std::log(value);
#endif
}

STORM_TRANSPORT_INLINE
double Log1p(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::log1p(value);
#else
    return std::log1p(value);
#endif
}

STORM_TRANSPORT_INLINE
double Sin(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::sin(value);
#else
    return std::sin(value);
#endif
}

STORM_TRANSPORT_INLINE
double Sqrt(const double value)
{
#ifdef STORM_WITH_GPU
    return Kokkos::sqrt(value);
#else
    return std::sqrt(value);
#endif
}

} // namespace transport
} // namespace STORM

#endif // STORM_RADIATION_TRANSPORT_PORTABILITY_HPP
