#ifndef STORM_LINEAR_INTERPOLATION_HPP
#define STORM_LINEAR_INTERPOLATION_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../StormError.hpp"

namespace STORM {

inline bool Close2Zero(double x)
{
    return std::abs(x) < 1e-12;
}

// |a - b| within 1e-12 of `scale`, the magnitude of the data. An absolute
// tolerance would snap every point of data far below 1 (cumulative opacities
// in cm^-1 can be ~1e-40) onto the next node.
inline bool CloseRelative(double a, double b, double scale)
{
    return std::abs(a - b) <= 1e-12 * scale;
}

template<typename T, typename ContainerX, typename ContainerY>
T LinearInterpolation(const ContainerX &x, const ContainerY &y, T xi)
{
    double const scale = std::max(std::abs(static_cast<double>(x.front())),
                                  std::abs(static_cast<double>(x.back())));
    auto it = std::upper_bound(x.begin(), x.end(), xi);
    if(it == x.end())
    {
        if(CloseRelative(x.back(), xi, scale))
        {
            return y.back();
        }
        throw STORMError("X too large in LinearInterpolation: x_i = " + std::to_string(xi));
    }
    // upper_bound at the first node means xi < x.front(): only a value within
    // the tolerance of the first node is accepted (returned below), anything
    // further below would interpolate with the node before the array.
    if(it == x.begin() && !CloseRelative(*it, xi, scale))
    {
        throw STORMError("X too small in LinearInterpolation: x_i = " + std::to_string(xi));
    }
    if(CloseRelative(*it, xi, scale))
    {
        return y[static_cast<std::size_t>(it - x.begin())];
    }

    return y[static_cast<std::size_t>(it - x.begin())] + (xi - *it) * (y[static_cast<std::size_t>(it - 1 - x.begin())] - y[static_cast<std::size_t>(it - x.begin())]) / (*(it - 1) - *it);
}

} // namespace STORM

#endif // STORM_LINEAR_INTERPOLATION_HPP
