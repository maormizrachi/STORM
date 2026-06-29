#ifndef STORM_EXAMPLE_VECTOR3D_HPP
#define STORM_EXAMPLE_VECTOR3D_HPP

#include <ostream>

struct Vector3D
{
    using coord_type = double;

    double x, y, z;

    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double v) : x(v), y(v), z(v) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    friend inline std::ostream &operator<<(std::ostream &os, const Vector3D &v)
    {
        return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }
};

#endif // STORM_EXAMPLE_VECTOR3D_HPP
