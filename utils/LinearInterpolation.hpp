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

template<typename T, typename ContainerX, typename ContainerY>
T LinearInterpolation(const ContainerX &x, const ContainerY &y, T xi)
{
    auto it = std::upper_bound(x.begin(), x.end(), xi);
    if(it == x.end())
    {
        if(Close2Zero(x.back() - xi))
        {
            return y.back();
        }
        throw STORMError("X too large in LinearInterpolation: x_i = " + std::to_string(xi));
    }
    if(it == x.begin())
    {
        if(*it < x.at(0))
        {
            throw STORMError("X too small in LinearInterpolation: x_i = " + std::to_string(xi));
        }
    }
    if(Close2Zero(*it - xi))
    {
        return y[static_cast<std::size_t>(it - x.begin())];
    }

    return y[static_cast<std::size_t>(it - x.begin())] + (xi - *it) * (y[static_cast<std::size_t>(it - 1 - x.begin())] - y[static_cast<std::size_t>(it - x.begin())]) / (*(it - 1) - *it);
}

} // namespace STORM

#endif // STORM_LINEAR_INTERPOLATION_HPP
