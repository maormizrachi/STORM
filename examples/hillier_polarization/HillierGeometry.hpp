#ifndef STORM_HILLIER_POLARIZATION_GEOMETRY_HPP
#define STORM_HILLIER_POLARIZATION_GEOMETRY_HPP

// Mesh and initial conditions for the Hillier (1994) polarized-scattering
// benchmark: Fibonacci directions placed on geometrically spaced radial shells,
// with guard shells either side of the detachment radius and a fine core.
//
// The shells are geometric (uniform in log r) with the radial spacing tied to the
// angular spacing, so cells come out close to isotropic and every cell stays far
// below the DDMC conversion threshold -- the accelerators must not fire in an
// optically thin problem, and the test asserts that they do not.

#include <cmath>
#include <cstddef>
#include <vector>

namespace STORM {
namespace examples {

struct HillierSetup
{
    double chi0 = 0.02;          //!< density normalisation [1/length]
    double rMin = 2.0;           //!< envelope inner edge
    double rMaxRatio = 30.0;     //!< outer edge in units of rMin
    double radialExponent = 4.0; //!< n in (R_min/r)^n -- Hillier uses 4
    bool prolate = true;         //!< false selects the spherical null control
    std::size_t directionsPerShell = 2048; //!< Fibonacci directions on each shell
    std::size_t corePoints = 400;//!< generator points inside the source core

    double rMax() const { return rMaxRatio * rMin; }

    //! \brief Radial spacing fraction; ties shell spacing to the angular spacing.
    double shellSpacing() const
    {
        return std::sqrt(4.0 * M_PI / static_cast<double>(directionsPerShell));
    }

    //! \brief Outermost generator shell, including the two outer guards.
    double outerGuardRadius() const
    {
        return rMax() * std::pow(1.0 + shellSpacing(), 2.0);
    }

    //! \brief Domain half width.
    //!
    //! Must clear the outer guard shell by a healthy margin: generator points
    //! sitting essentially on the domain face send MadVoro's Hilbert ordering into
    //! unbounded recursion, which presents as a stack overflow during the build.
    double boxHalfWidth() const { return 1.2 * outerGuardRadius(); }
    double rSource() const { return 0.1 * rMin; }      //!< "point" source radius
    double rObserver() const { return 1.0167 * rMax(); }

    //! \brief Solid-angle-averaged radial optical depth of the envelope.
    //!
    //! tau_bar = chi0 <1+10cos^2 b> (R_min/(n-1)) (1 - (R_min/R_max)^(n-1))
    //! = 2.8887 chi0 for n = 4 and R_max = 30 R_min, matching the tau_ave quoted
    //! by Bulla, Sim & Kromer (2015).
    double tauBar() const
    {
        double const n = radialExponent;
        return chi0 * (13.0 / 3.0) * (rMin / (n - 1.0))
               * (1.0 - std::pow(rMin / rMax(), n - 1.0));
    }
};

//! \brief Near-uniform directions on the unit sphere (golden-angle lattice).
template<typename PointT>
std::vector<PointT> FibonacciDirections(std::size_t n)
{
    std::vector<PointT> out;
    out.reserve(n);
    double const golden = M_PI * (3.0 - std::sqrt(5.0));
    for (std::size_t i = 0; i < n; ++i)
    {
        double const z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) /
                                   static_cast<double>(n);
        double const r = std::sqrt(std::max(0.0, 1.0 - z * z));
        double const phi = golden * static_cast<double>(i);
        out.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
    }
    return out;
}

//! \brief Generator points: geometric shells plus guards plus a resolved core.
template<typename PointT>
std::vector<PointT> HillierMeshPoints(HillierSetup const &setup)
{
    std::vector<PointT> const dirs =
        FibonacciDirections<PointT>(setup.directionsPerShell);
    // Radial spacing matched to the angular spacing gives near-isotropic cells.
    double const f = setup.shellSpacing();

    std::vector<double> radii;
    for (double r = setup.rMax(); r > setup.rMin; r *= (1.0 - f))
        radii.push_back(r);
    // Guard shells straddling both edges keep the detachment sharp: a cell that
    // spans the inner boundary would smear the very discontinuity that defines
    // the geometry.
    for (std::size_t k = 1; k <= 2; ++k)
    {
        radii.push_back(setup.rMin * std::pow(1.0 - f, static_cast<double>(k)));
        radii.push_back(setup.rMax() * std::pow(1.0 + f, static_cast<double>(k)));
    }

    std::vector<PointT> points;
    points.reserve(radii.size() * dirs.size() + setup.corePoints);
    for (double r : radii)
        for (PointT const &d : dirs)
            points.emplace_back(d.x * r, d.y * r, d.z * r);

    // Seed the core explicitly rather than relying on the shell spacing, so the
    // source size does not drift when the angular resolution changes.
    double const rCore = 1.5 * setup.rSource();
    std::size_t const nSide = static_cast<std::size_t>(
        std::ceil(std::cbrt(static_cast<double>(setup.corePoints) * 6.0 / M_PI)));
    double const h = 2.0 * rCore / static_cast<double>(nSide);
    for (std::size_t i = 0; i < nSide; ++i)
        for (std::size_t j = 0; j < nSide; ++j)
            for (std::size_t k = 0; k < nSide; ++k)
            {
                double const x = -rCore + (static_cast<double>(i) + 0.5) * h;
                double const y = -rCore + (static_cast<double>(j) + 0.5) * h;
                double const z = -rCore + (static_cast<double>(k) + 0.5) * h;
                if (x * x + y * y + z * z < rCore * rCore)
                    points.emplace_back(x, y, z);
            }
    return points;
}

//! \brief sigma_e N_e at a point; zero inside the source core and outside the shell.
inline double HillierScatteringCoefficient(double x, double y, double z,
                                           HillierSetup const &setup)
{
    double const r = std::sqrt(x * x + y * y + z * z);
    if (r < setup.rMin || r > setup.rMax())
        return 0.0;
    double const mu = z / r;
    // The spherical control carries the same solid-angle-averaged depth as the
    // prolate case, so a difference between them is shape, not normalisation.
    double const angular = setup.prolate ? (1.0 + 10.0 * mu * mu) : (13.0 / 3.0);
    return setup.chi0 * std::pow(setup.rMin / r, setup.radialExponent) * angular;
}

} // namespace examples
} // namespace STORM

#endif // STORM_HILLIER_POLARIZATION_GEOMETRY_HPP
