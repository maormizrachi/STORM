#ifndef RDMONT_LINEAR_INTERPOLATION_HPP
#define RDMONT_LINEAR_INTERPOLATION_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "monte/RDMontError.hpp"

namespace RDMont {

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
        throw RDMontError("X too large in LinearInterpolation: x_i = " + std::to_string(xi));
    }
    if(it == x.begin())
    {
        if(*it < x.at(0))
        {
            throw RDMontError("X too small in LinearInterpolation: x_i = " + std::to_string(xi));
        }
    }
    if(Close2Zero(*it - xi))
    {
        return y[static_cast<std::size_t>(it - x.begin())];
    }

    return y[static_cast<std::size_t>(it - x.begin())] + (xi - *it) * (y[static_cast<std::size_t>(it - 1 - x.begin())] - y[static_cast<std::size_t>(it - x.begin())]) / (*(it - 1) - *it);
}

} // namespace RDMont

#endif // RDMONT_LINEAR_INTERPOLATION_HPP
