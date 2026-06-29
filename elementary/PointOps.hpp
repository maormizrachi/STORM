#ifndef STORM_POINT_OPS_HPP
#define STORM_POINT_OPS_HPP

#include <cmath>
#include <type_traits>

namespace STORM
{
namespace fallback
{

namespace detail
{
template <typename T, typename = void>
struct is_point3d : std::false_type {};

template <typename T>
struct is_point3d<T, std::void_t<
    typename T::coord_type,
    decltype(std::declval<T>().x),
    decltype(std::declval<T>().y),
    decltype(std::declval<T>().z)
>> : std::true_type {};
} // namespace detail

// Arithmetic operators
template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator+(const PointT &a, const PointT &b) { return PointT(a.x + b.x, a.y + b.y, a.z + b.z); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator-(const PointT &a, const PointT &b) { return PointT(a.x - b.x, a.y - b.y, a.z - b.z); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator*(const PointT &v, typename PointT::coord_type s) { return PointT(v.x * s, v.y * s, v.z * s); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator*(typename PointT::coord_type s, const PointT &v) { return PointT(v.x * s, v.y * s, v.z * s); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator/(const PointT &v, typename PointT::coord_type s) { return PointT(v.x / s, v.y / s, v.z / s); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT operator-(const PointT &v) { return PointT(-v.x, -v.y, -v.z); }

// Compound assignment
template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT &operator+=(PointT &a, const PointT &b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT &operator-=(PointT &a, const PointT &b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT &operator*=(PointT &v, typename PointT::coord_type s) { v.x *= s; v.y *= s; v.z *= s; return v; }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT &operator/=(PointT &v, typename PointT::coord_type s) { v.x /= s; v.y /= s; v.z /= s; return v; }

// Geometric operations
template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline auto ScalarProd(const PointT &a, const PointT &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT CrossProduct(const PointT &a, const PointT &b)
{
    return PointT(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline auto abs(const PointT &v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline auto fastabs(const PointT &v) { return abs(v); }

template <typename T>
inline T fastsqrt(T x) { return std::sqrt(x); }

template <typename PointT, std::enable_if_t<detail::is_point3d<PointT>::value, int> = 0>
inline PointT normalize(const PointT &v)
{
    auto len = abs(v);
    return (len > 0) ? PointT(v.x / len, v.y / len, v.z / len) : v;
}

} // namespace fallback
} // namespace STORM

#endif // STORM_POINT_OPS_HPP
