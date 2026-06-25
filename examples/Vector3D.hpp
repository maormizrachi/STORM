#ifndef STORM_EXAMPLE_VECTOR3D_HPP
#define STORM_EXAMPLE_VECTOR3D_HPP

#include <cmath>
#include <iostream>

struct Vector3D
{
    using value_type = double;

    double x, y, z;

    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double v) : x(v), y(v), z(v) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3D &operator+=(const Vector3D &o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vector3D &operator-=(const Vector3D &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vector3D &operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    Vector3D &operator/=(double s) { x /= s; y /= s; z /= s; return *this; }
};

inline Vector3D operator+(const Vector3D &a, const Vector3D &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vector3D operator-(const Vector3D &a, const Vector3D &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vector3D operator*(const Vector3D &v, double s) { return {v.x * s, v.y * s, v.z * s}; }
inline Vector3D operator*(double s, const Vector3D &v) { return v * s; }
inline Vector3D operator/(const Vector3D &v, double s) { return {v.x / s, v.y / s, v.z / s}; }

inline double ScalarProd(const Vector3D &a, const Vector3D &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vector3D CrossProduct(const Vector3D &a, const Vector3D &b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double abs(const Vector3D &v) { return std::sqrt(ScalarProd(v, v)); }
inline double fastabs(const Vector3D &v) { return abs(v); }
inline Vector3D normalize(const Vector3D &v) { double m = abs(v); return (m > 0) ? v / m : Vector3D(0); }

inline std::ostream &operator<<(std::ostream &os, const Vector3D &v) { return os << "(" << v.x << ", " << v.y << ", " << v.z << ")"; }

#endif // STORM_EXAMPLE_VECTOR3D_HPP
