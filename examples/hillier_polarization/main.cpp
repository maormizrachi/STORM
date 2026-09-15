// Hillier (1994) polarized-scattering benchmark.
//
// A point source inside a detached prolate electron-scattering envelope produces a
// continuum polarization that, in the optically thin single-scattering limit, is
//
//     p(i) = (3/8) (1 - 3 gamma) tau_bar sin^2 i,   gamma = <cos^2 beta>_rho,
//
// with tau_bar the SOLID-ANGLE-AVERAGED radial optical depth.  For the density law
// used here gamma = 7/13, so
//
//     p(i) = -(3/13) tau_bar sin^2 i.
//
// (The coefficient is 3/8 rather than the 3/16 usually quoted because Brown &
// McLean integrate tau over mu in [-1,1], twice the solid-angle average.  It was
// checked against a direct numerical integration of the Thomson phase matrix.)
//
// The test asserts four things, all against that closed form -- no external
// reference data, so it stands alone as a regression gate:
//
//   1. a spherical envelope produces no net polarization (the null control);
//   2. the prolate envelope reproduces the thin-limit amplitude;
//   3. p follows sin^2 i across inclination -- normalisation-free, so this one
//      tests the phase matrix and the basis rotations on their own;
//   4. Stokes u is consistent with zero, which is the specific signature of a
//      sky-basis convention error and is invisible to checks 1-3.
//
// Usage:  hillier_polarization [--chi0 X] [--spherical] [--generations N]
//                              [--photons-per-cell N] [--directions N] [--observers N]
//                              [--tag NAME] [--no-assert] [--sweep]
//                              [--rings | --fibonacci] [--azimuths N]
//                              [--acceptance DEG]
//
// --rings (the default) places detectors only on the inclinations the paper
// tabulates, several per ring spanning phi, each accepting photons whose
// PROPAGATION DIRECTION lies within --acceptance of it.  --fibonacci selects
// STORM's SphericalObserver instead, which spreads directions over the whole
// sphere and bins by crossing position; the two are kept selectable so the
// difference can be measured.
//
// --tag suffixes the per-observer dump so a chi0 sweep can keep its points apart.
//
// --sweep walks two decades in chi0 after the gating point, writing one dump per
// value for plot_hillier_fig5.py to lay over Hillier's published curves.  Most of
// that span is optically thick, where the thin limit above does not apply, so the
// sweep points report their numbers and never gate.  --no-assert does the same
// for a single point.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>
#include <units/units.hpp>

#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"

#include "boundary/Vacuum.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/SphericalObserver.hpp"

#include "HillierGeometry.hpp"
#include "HillierObserver.hpp"
#include "HillierPhysics.hpp"

namespace {

using Grid = MadVoro::Voronoi3D<Vector3D>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell,
                                STORM::SimpleExtensives, STORM::examples::HillierEOS, 1,
                                STORM::examples::HillierOpacity<Vector3D, Grid>>;

constexpr double GAMMA = 7.0 / 13.0;
constexpr double THIN_COEFF = 3.0 / 8.0 * (3.0 * GAMMA - 1.0);   // = 3/13

// Inclination bands.  22.5 deg is deliberately absent: a fixed angular band has
// <sin^2 i> 13% above sin^2 of its centre at that inclination, because sin^2 i
// varies steeply there, which biases the comparison by more than the tolerance.
// Fibonacci directions land nowhere near 22.5 deg, and a +-7 deg band there has
// <sin^2 i> 12% above sin^2 of its centre, so that inclination is only measurable
// with ring detectors placed on it.
constexpr double PAPER_INCLINATIONS[] = {22.5, 45.0, 67.5, 90.0};
constexpr double FIBONACCI_BAND_CENTRES[] = {45.0, 67.5, 90.0};
constexpr double BAND_HALF_WIDTH_DEG = 7.0;

constexpr double NULL_SIGMA_TOLERANCE = 4.0;   // null case, |p| < n sigma
constexpr double AMPLITUDE_TOLERANCE = 0.25;   // prolate, fractional
constexpr double SIN2_TOLERANCE = 0.15;        // sin^2 i ratio, fractional

struct BandResult
{
    double inclination = 0.0;   //!< energy-weighted mean inclination [rad]
    double q = 0.0;             //!< Stokes q = Q/I
    double u = 0.0;
    double sigma = 0.0;         //!< Monte Carlo error on q
    double sin2 = 0.0;          //!< energy-weighted <sin^2 i> over the band
    std::size_t count = 0;
};

//! \brief Average the observer directions falling inside one inclination band.
//!
//! The configuration is axisymmetric, so every observer at a given inclination
//! measures the same quantity; averaging over azimuth is a free noise reduction.
//! Weighting by energy makes this the pooled Stokes ratio sum(Q)/sum(I) rather
//! than a mean of per-observer ratios, which would be biased.
BandResult BandAverage(const std::vector<Vector3D> &directions,
                       const std::vector<double> &energy,
                       const std::vector<double> &stokesQ,
                       const std::vector<double> &stokesU,
                       double centreDeg)
{
    BandResult out;
    double const lo = (centreDeg - BAND_HALF_WIDTH_DEG) * M_PI / 180.0;
    double const hi = (centreDeg + BAND_HALF_WIDTH_DEG) * M_PI / 180.0;
    double wSum = 0.0, qSum = 0.0, uSum = 0.0, incSum = 0.0, sin2Sum = 0.0;
    double varSum = 0.0;

    for (std::size_t i = 0; i < directions.size(); ++i)
    {
        Vector3D const &d = directions[i];
        double const norm = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (norm <= 0.0 || energy[i] <= 0.0)
            continue;
        double theta = std::acos(std::max(-1.0, std::min(1.0, d.z / norm)));
        // Fold about the equator: the prolate envelope is symmetric in beta.
        double const inc = std::min(theta, M_PI - theta);
        if (inc < lo || inc > hi)
            continue;

        double const w = energy[i];
        wSum += w;
        qSum += stokesQ[i];
        uSum += stokesU[i];
        incSum += w * inc;
        sin2Sum += w * std::sin(inc) * std::sin(inc);
        // Per-observer shot noise on q, propagated into the band mean.
        varSum += stokesQ[i] * stokesQ[i];
        ++out.count;
    }
    if (wSum <= 0.0)
        return out;

    out.q = qSum / wSum;
    out.u = uSum / wSum;
    out.inclination = incSum / wSum;
    out.sin2 = sin2Sum / wSum;
    // sqrt(sum Q^2) / sum(I) is the standard error of the pooled ratio when the
    // per-observer contributions are independent, which they are here.
    out.sigma = std::sqrt(std::max(0.0, varSum - qSum * qSum / out.count)) / wSum;
    return out;
}

//! \brief Everything about a run that is not the envelope geometry.
struct RunOptions
{
    std::size_t generations = 8;
    std::size_t photonsPerCell = 40;
    std::size_t numObservers = 256;
    //! Suffix for the per-observer dump.  A chi0 sweep writes one file per
    //! point, so the caller has to be able to keep them apart.  Kept free of
    //! dots: downstream tooling splits on the last '.' to find the extension.
    std::string tag;
    //! Sweep points exist to put STORM on Hillier's figure, not to gate
    //! anything.  The analytic thin limit this test asserts against is only
    //! valid while tau_bar << 1, so a run at chi0 ~ 1 reports its numbers and
    //! stays silent about pass/fail rather than failing a check that does not
    //! apply to it.
    bool asserting = true;
    //! Ring detectors on the paper's inclinations, rather than Fibonacci
    //! directions covering the whole sphere.
    bool useRings = true;
    std::size_t azimuths = 16;      //!< detectors spanning phi on each ring
    // A 2 deg cone accepts 1/25 the solid angle of a 10 deg one, so the packet
    //! budget has to rise with it or the measurement goes to noise; see the
    //! generation count in REGRESSION_INFO.
    double acceptanceDeg = 2.0;     //!< cone half-angle about each detector
};

//! \brief Run one chi0 point end to end: mesh, transport, analysis.
//!
//! Factored out of main so a single invocation can walk a chi0 sweep.  The
//! mesh does not depend on chi0 -- only the tabulated scattering coefficient
//! does -- but each point is built from scratch anyway: sharing a grid across
//! points would couple them through the population control and the census, and
//! a few seconds of mesh build is not worth that.
//!
//! \return true when the point's checks pass, or when it is not gating.
bool RunPoint(STORM::examples::HillierSetup const &setup,
              RunOptions const &opt, int rank, int nprocs)
{
    bool ok = true;
    std::size_t const generations = opt.generations;
    std::size_t const photonsPerCell = opt.photonsPerCell;
    std::size_t const numObservers = opt.numObservers;
    std::string const &tag = opt.tag;
    bool const asserting = opt.asserting;

        double const tauBar = setup.tauBar();
        if (rank == 0)
        {
            std::cout << "Hillier polarization benchmark\n"
                      << "  chi0            = " << setup.chi0 << "\n"
                      << "  geometry        = " << (setup.prolate ? "prolate" : "spherical (null control)") << "\n"
                      << "  R_min, R_max    = " << setup.rMin << ", " << setup.rMax() << "\n"
                      << "  radial exponent = " << setup.radialExponent << "\n"
                      << "  tau_bar         = " << tauBar << "  (tau_ave = " << 2.0 * tauBar << ")\n"
                      << "  ranks           = " << nprocs << std::endl;
        }

        // ---- mesh -------------------------------------------------------------
        double const box = setup.boxHalfWidth();
        Vector3D const ll(-box, -box, -box), ur(box, box, box);
        std::vector<Vector3D> points =
            STORM::examples::HillierMeshPoints<Vector3D>(setup);
        // Fill out to the box so the region between the envelope and the tally sphere
        // is meshed; packets have to traverse it to be counted.
        {
            std::size_t const nFill = 12;
            double const h = 2.0 * box / static_cast<double>(nFill);
            for (std::size_t i = 0; i < nFill; ++i)
                for (std::size_t j = 0; j < nFill; ++j)
                    for (std::size_t k = 0; k < nFill; ++k)
                    {
                        Vector3D p(-box + (i + 0.5) * h, -box + (j + 0.5) * h,
                                   -box + (k + 0.5) * h);
                        if (std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) >
                            1.05 * setup.rMax())
                            points.push_back(p);
                    }
        }

        Grid grid(ll, ur);
    #ifdef STORM_WITH_MPI
        // Every rank generates the same deterministic point set, then keeps a
        // contiguous slice of it.  BuildParallel redistributes spatially afterwards,
        // but it must not be handed the same point more than once globally: exact
        // duplicates send MadVoro's Hilbert ordering into unbounded recursion, which
        // shows up as a stack overflow inside the mesh build rather than an error.
        {
            std::size_t const total = points.size();
            std::size_t const lo = (total * static_cast<std::size_t>(rank)) /
                                   static_cast<std::size_t>(nprocs);
            std::size_t const hi = (total * static_cast<std::size_t>(rank + 1)) /
                                   static_cast<std::size_t>(nprocs);
            std::vector<Vector3D> local(points.begin() + static_cast<std::ptrdiff_t>(lo),
                                        points.begin() + static_cast<std::ptrdiff_t>(hi));
            points.swap(local);
        }
        grid.BuildParallel(points);
    #else
        grid.Build(points);
    #endif
        std::size_t const Ncells = grid.GetPointNo();

        // ---- initial conditions ----------------------------------------------
        double const cvPerVolume = 1.0e6;         // large: the material stays inert
        double const T_source = 1.0e4;            // only sets the source luminosity
        double const T_floor = 1.0;               // must be > 0; radiates nothing
        double const coreAbsorption = 1.0e-2 / setup.rSource();   // core optically thin

        std::vector<STORM::RadiationCell> cells(Ncells);
        std::vector<STORM::SimpleExtensives> extensives(Ncells);
        std::vector<double> scattering(Ncells, 0.0);
        std::vector<char> isCore(Ncells, 0);

        for (std::size_t i = 0; i < Ncells; ++i)
        {
            Vector3D const c = grid.GetMeshPoint(i);
            double const r = std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
            double const volume = grid.GetVolume(i);

            isCore[i] = (r < setup.rSource()) ? 1 : 0;
            scattering[i] = isCore[i] ? 0.0
                : STORM::examples::HillierScatteringCoefficient(c.x, c.y, c.z, setup);

            cells[i].temperature = isCore[i] ? T_source : T_floor;
            cells[i].internalEnergy = cvPerVolume * cells[i].temperature * volume;
            cells[i].Erad = 0.0;
            extensives[i].mass = volume;          // density 1; the opacity is tabulated
            extensives[i].internal_energy = cells[i].internalEnergy;
            extensives[i].Erad = 0.0;
        }

        // ---- physics ----------------------------------------------------------
        STORM::RadiationIMCParameters<1> params;
        params.newPhotonsPerCell = photonsPerCell;
        params.withHydro = false;
        params.staticScatterers = true;    // Hillier's envelope does not move
        params.withPolarization = true;
        params.withDDMC = true;            // must stay inert here -- asserted below
        params.withRandomWalk = false;
        params.energyBoundaries = {0.0, 1.0e30};
        params.energyBoundariesProvided = true;

        auto eos = std::make_shared<STORM::examples::HillierEOS>(cvPerVolume);
        auto opacity = std::make_shared<STORM::examples::HillierOpacity<Vector3D, Grid>>(
            scattering, isCore, cells, coreAbsorption);
        auto boundary = std::make_shared<STORM::VacuumBoundary<Vector3D, Grid>>(grid);
        auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos,
                                             opacity, params);
        auto popControl =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 1, 6.0);

        // Two observers, one of which is used.  The ring observer bins by propagation
        // direction and resolves Q, U in the plane normal to it; SphericalObserver bins
        // by crossing position and rotates about that instead, which smears the angular
        // variable this benchmark is built on.  Keeping both selectable makes that
        // difference measurable rather than asserted.
        std::shared_ptr<STORM::SphericalObserver<Vector3D, 1>> fibonacciObserver;
        std::shared_ptr<STORM::examples::HillierRingObserver<Vector3D>> ringObserver;
        if (opt.useRings)
        {
            ringObserver = std::make_shared<STORM::examples::HillierRingObserver<Vector3D>>(
                Vector3D(0.0, 0.0, 0.0), setup.rObserver(),
                std::vector<double>(std::begin(PAPER_INCLINATIONS),
                                    std::end(PAPER_INCLINATIONS)),
                opt.azimuths, opt.acceptanceDeg);
            physics->setObserver(ringObserver);
        }
        else
        {
            fibonacciObserver = std::make_shared<STORM::SphericalObserver<Vector3D, 1>>(
                Vector3D(0.0, 0.0, 0.0), setup.rObserver(), numObservers,
                std::array<double, 2>{0.0, 1.0e30}, /*polarizationEnabled=*/true);
            physics->setObserver(fibonacciObserver);
        }

        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, popControl,
                                                           boundary);

        // A step long compared with the light-crossing time lets every packet escape
        // within the step, so each step is an independent realisation of the source.
        double const dt = 1.0e3 * setup.rMax() / units::clight;
        for (std::size_t g = 0; g < generations; ++g)
        {
            manager.getParticles().clear();
            manager.step(dt);
            if (rank == 0)
                std::cout << "  generation " << (g + 1) << "/" << generations << std::endl;
        }

        // ---- analysis ---------------------------------------------------------
        // Each rank tallies only the crossings of the packets it owned, so the
        // per-observer sums have to be reduced before anything is read off them.
        // Without this the analysis sees 1/nprocs of the statistics and the bands
        // come out nearly empty.
        std::vector<Vector3D> dirs;
        std::vector<double> energy, sq, su;
        if (ringObserver)
        {
            dirs = ringObserver->directions();
            energy = ringObserver->observerEnergy();
            sq = ringObserver->stokesQ();
            su = ringObserver->stokesU();
        }
        else
        {
            dirs = fibonacciObserver->directions();
            energy = fibonacciObserver->observerEnergy();
            sq = fibonacciObserver->stokesQ();
            su = fibonacciObserver->stokesU();
        }
        MPI_Allreduce(MPI_IN_PLACE, energy.data(), static_cast<int>(energy.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, sq.data(), static_cast<int>(sq.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, su.data(), static_cast<int>(su.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        std::size_t ddmcSteps = physics->getDDMCStepCount();
        {
            unsigned long long local = static_cast<unsigned long long>(ddmcSteps);
            MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
            ddmcSteps = static_cast<std::size_t>(local);
        }

        if (rank == 0)
        {

        // Per-observer dump for plot_hillier.py.  All 256 directions, not just the
        // three assertion bands: the sin^2 i trend is far easier to judge by eye from
        // the full set, and the scatter about it is the Monte Carlo noise.
        {
            std::string const name =
                std::string("hillier_observers_") +
                (setup.prolate ? "prolate" : "spherical") +
                (tag.empty() ? std::string() : "_" + tag) + ".txt";
            std::FILE *fh = std::fopen(name.c_str(), "w");
            if (fh != nullptr)
            {
                std::fprintf(fh, "# Hillier polarization benchmark, per-observer tallies\n");
                std::fprintf(fh, "# chi0 = %.6g   tau_bar = %.6g   geometry = %s\n",
                             setup.chi0, tauBar, setup.prolate ? "prolate" : "spherical");
                std::fprintf(fh, "# thin limit: q(i) = -(3/13) tau_bar sin^2 i\n");
            if (ringObserver)
                std::fprintf(fh, "# observer = rings   azimuths = %zu   "
                             "acceptance_deg = %.3f   detectors = %zu\n",
                             opt.azimuths, opt.acceptanceDeg, dirs.size());
            else
                std::fprintf(fh, "# observer = fibonacci   directions = %zu\n",
                             dirs.size());
                std::fprintf(fh, "# inclination_deg   q            u            energy\n");
                for (std::size_t i = 0; i < dirs.size(); ++i)
                {
                    double const n = std::sqrt(dirs[i].x * dirs[i].x +
                                               dirs[i].y * dirs[i].y +
                                               dirs[i].z * dirs[i].z);
                    if (n <= 0.0 || energy[i] <= 0.0)
                        continue;
                    double const th = std::acos(std::max(-1.0, std::min(1.0, dirs[i].z / n)));
                    double const inc = std::min(th, M_PI - th) * 180.0 / M_PI;
                    std::fprintf(fh, "%14.6f  %+.6e  %+.6e  %.6e\n",
                                 inc, sq[i] / energy[i], su[i] / energy[i], energy[i]);
                }
                std::fclose(fh);
                std::cout << "  wrote " << name << std::endl;
            }
        }

        std::cout << "\n  incl_deg      q          sigma_q       u        expected_q   n\n";
        std::vector<BandResult> bands;
        std::vector<double> bandCentres;
        if (opt.useRings)
            bandCentres.assign(std::begin(PAPER_INCLINATIONS), std::end(PAPER_INCLINATIONS));
        else
            bandCentres.assign(std::begin(FIBONACCI_BAND_CENTRES),
                               std::end(FIBONACCI_BAND_CENTRES));
        for (double centre : bandCentres)
        {
            BandResult const b = BandAverage(dirs, energy, sq, su, centre);
            if (b.count == 0)
            {
                std::cout << "  " << centre << ": no observers in band\n";
                ok = false;
                continue;
            }
            bands.push_back(b);
            double const expected = setup.prolate ? -THIN_COEFF * tauBar * b.sin2 : 0.0;
            std::printf("  %8.3f  %+.5e  %.3e  %+.3e  %+.5e  %zu\n",
                        b.inclination * 180.0 / M_PI, b.q, b.sigma, b.u, expected,
                        b.count);

            if (!setup.prolate)
            {
                // 1. null control
                if (std::abs(b.q) > NULL_SIGMA_TOLERANCE * std::max(b.sigma, 1e-30))
                {
                    std::cout << "  FAIL: spherical envelope is polarized at "
                              << centre << " deg\n";
                    ok = false;
                }
            }
            else
            {
                // 2. thin-limit amplitude
                double const ratio = b.q / expected;
                if (std::abs(ratio - 1.0) > AMPLITUDE_TOLERANCE)
                {
                    std::cout << "  FAIL: amplitude off by " << ratio << "x at "
                              << centre << " deg\n";
                    ok = false;
                }
            }
            // 4. axisymmetry: u carries no signal, so a non-zero u is a basis bug
            if (std::abs(b.u) > NULL_SIGMA_TOLERANCE * std::max(b.sigma, 1e-30))
            {
                std::cout << "  FAIL: u is non-zero at " << centre
                          << " deg (sky-basis rotation?)\n";
                ok = false;
            }
        }

        // 3. sin^2 i law -- needs no reference data and no normalisation
        if (setup.prolate && bands.size() >= 2)
        {
            BandResult const &ref = bands.back();
            std::cout << "  sin^2 i law:  incl_deg   observed    expected      slack\n";
            for (BandResult const &b : bands)
            {
                double const predicted = b.sin2 / ref.sin2;
                double const observed = b.q / ref.q;
                // Propagated shot noise on the ratio of two band means.
                double const relative =
                    std::sqrt(std::pow(b.sigma / b.q, 2.0) +
                              std::pow(ref.sigma / ref.q, 2.0));
                double const sigmaObserved = std::abs(observed) * relative;
                // Two allowances, because two different things move the ratio off
                // sin^2 i.  The fractional term covers the physical departure: the
                // law is exact only as tau_bar -> 0, and at the tau_bar used here
                // Hillier's own curves already sit ~13% off it at the smallest
                // inclination.  The 3-sigma term covers shot noise, which dominates
                // at small inclination where the signal is weakest -- a flat
                // fractional tolerance there rejects good runs.
                double const slack = SIN2_TOLERANCE * predicted + 3.0 * sigmaObserved;
                std::printf("              %8.3f   %8.4f   %8.4f   %8.4f\n",
                            b.inclination * 180.0 / M_PI, observed, predicted, slack);
                if (std::abs(observed - predicted) > slack)
                {
                    std::cout << "  FAIL: sin^2 i law violated at "
                              << b.inclination * 180.0 / M_PI << " deg\n";
                    ok = false;
                }
            }
        }

        // The envelope is optically thin everywhere, so no cell may convert to DDMC.
        // If one does the comparison above is meaningless, and the accelerated
        // polarization closure -- which this test does not validate -- has run.
        if (ddmcSteps != 0)
        {
            std::cout << "  FAIL: DDMC fired (" << ddmcSteps
                      << " steps); this problem must stay in explicit transport\n";
            ok = false;
        }

        // Deliberately not the PASS/FAIL token: the harness greps for it, and a sweep
        // point must not be mistaken for a gate that ran.
        if (!asserting)
        {
            std::cout << "\n  SWEEP POINT (tau_bar = " << tauBar
                      << "): figure data only, checks not gating.\n";
            ok = true;
        }
        else
        {
            std::cout << (ok ? "\nPASS hillier_polarization\n"
                             : "\nFAIL hillier_polarization\n");
        }
        }   // rank 0 analysis
    return ok;
}

} // namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    bool ok = true;
    // Inner scope: the grid, manager and observer must be destroyed before
    // MPI_Finalize, because their destructors still call into MPI.  Letting them
    // die at the end of main instead gives "MPI_Comm_size() was called after
    // MPI_FINALIZE" and a non-zero exit even on a passing run.
    {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    STORM::examples::HillierSetup setup;
    RunOptions opt;
    bool sweep = false;
    // The null control is one of the four checks, so it has to run whenever the
    // test runs: the harness invokes the binary once, and a control that only
    // ever ran by hand is not a gate.  --spherical still selects it on its own
    // for manual work.
    bool controlOnly = false;
    bool prolateOnly = false;

    for (int a = 1; a < argc; ++a)
    {
        std::string const arg(argv[a]);
        auto next = [&]() { return (a + 1 < argc) ? std::atof(argv[++a]) : 0.0; };
        if (arg == "--chi0")                 setup.chi0 = next();
        else if (arg == "--spherical")       controlOnly = true;
        else if (arg == "--prolate-only")    prolateOnly = true;
        else if (arg == "--directions")      setup.directionsPerShell = static_cast<std::size_t>(next());
        else if (arg == "--generations")     opt.generations = static_cast<std::size_t>(next());
        else if (arg == "--photons-per-cell")opt.photonsPerCell = static_cast<std::size_t>(next());
        else if (arg == "--observers")       opt.numObservers = static_cast<std::size_t>(next());
        else if (arg == "--tag" && a + 1 < argc) opt.tag = argv[++a];
        else if (arg == "--no-assert")       opt.asserting = false;
        else if (arg == "--sweep")           sweep = true;
        else if (arg == "--fibonacci")       opt.useRings = false;
        else if (arg == "--rings")           opt.useRings = true;
        else if (arg == "--azimuths")        opt.azimuths = static_cast<std::size_t>(next());
        else if (arg == "--acceptance")      opt.acceptanceDeg = next();
    }

    if (!controlOnly)
    {
        setup.prolate = true;
        ok = RunPoint(setup, opt, rank, nprocs);
    }
    if (!prolateOnly)
    {
        // Same chi0, same mesh, same statistics: a difference between the two is
        // envelope shape and nothing else.
        STORM::examples::HillierSetup control = setup;
        control.prolate = false;
        ok = RunPoint(control, opt, rank, nprocs) && ok;
    }

    // --sweep adds the rest of Hillier's Fig. 5 after the gating point above.
    // Most of that figure lies at tau_bar of order unity, far outside the thin
    // limit this test asserts against, so the extra points produce the reference
    // comparison and nothing else: they are explicitly not gating.
    if (sweep)
    {
        // Two decades in chi0, matching the span of Hillier's published figure.
        static constexpr struct { double chi0; const char *tag; } kSweep[] = {
            {0.01,   "0010"}, {0.0215, "0022"}, {0.0464, "0046"},
            {0.1,    "0100"}, {0.215,  "0215"}, {0.464,  "0464"},
            {1.0,    "1000"},
        };
        for (auto const &point : kSweep)
        {
            STORM::examples::HillierSetup sweepSetup = setup;
            sweepSetup.prolate = true;
            sweepSetup.chi0 = point.chi0;
            RunOptions sweepOpt = opt;
            sweepOpt.tag = point.tag;
            sweepOpt.asserting = false;
            // Return value deliberately dropped: a sweep point cannot fail the
            // test, and RunPoint already reported its numbers.
            (void)RunPoint(sweepSetup, sweepOpt, rank, nprocs);
        }
    }
    }   // inner scope: STORM objects destroyed before MPI_Finalize

    MPI_Finalize();
    return ok ? 0 : 1;
}
