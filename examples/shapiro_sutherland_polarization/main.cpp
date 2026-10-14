// Plane-parallel polarized atmosphere benchmark: Shapiro & Sutherland (1982,
// ApJ 263, 902), Table 1.
//
// Emergent I(0,mu)/F and polarization Q/I of a semi-infinite Thomson-scattering
// atmosphere: pure scattering (Chandrasekhar 1960), and a fraction lambda = 1/6
// or lambda = tau/(1+tau) of true absorption with LTE emission lambda B_nu(tau).
// B_nu follows Harrington (1969) as reconstructed in planeparallel_reference.py:
// u0 = h nu / k Teff = 20 and the grey Hopf T(tau).  The material is inert (huge
// heat capacity: Fleck factor one, frozen T) and each cell's a T^4 follows B_nu,
// so the grey transport solves the monochromatic problem.
//
// Geometry: a slab along x (Cartesian mesh, fine layers near the surface), the
// surface facing +x, vacuum above with a plane observer that counts every
// escaping photon, and mirror side walls that make the slab laterally infinite
// (SlabBoundary.hpp).  Pure scattering is lit from below by a thin emitting
// layer under tau = 10 of scatterers; the absorption cases thermalize long
// before their tau = 40 bottom.
//
// The bin-averaged I/F and Q/I are checked against the deterministic solution
// (ReferenceTables.hpp); per-generation tallies go to PREFIX_CASE[_sSEED].txt
// for plot_shapiro.py.
//
// Usage: shapiro_sutherland_polarization [--case scattering|lambda16|lambdatau|all]
//            [--generations N] [--photons-per-cell N] [--lateral-cells N]
//            [--prefix NAME] [--seed N] [--no-assert]
// The default photons per cell differ by case (about 1.5M packets per
// generation for the scattering and lambda = 1/6 cases, 5M for the cheaper
// lambda = tau/(1+tau)); --photons-per-cell overrides all three.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <units/units.hpp>

#include "examples/Vector3D.hpp"
#include "examples/hillier_polarization/HillierPhysics.hpp"
#include "MadCart/CartesianMesh3D.hpp"

#include "manager/MonteCarloManagerFactory.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationOpacityModel.hpp"

#ifndef MONTECARLO_POLARIZATION
#error "shapiro_sutherland_polarization measures Stokes Q and U. Configure STORM with -DSTORM_MONTECARLO_POLARIZATION=ON."
#endif

#include "PlaneObserver.hpp"
#include "ReferenceTables.hpp"
#include "SlabBoundary.hpp"

namespace {

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using STORM::RadiationCell;

constexpr std::size_t MU_BINS = 20;
// Per generation: energy, Q and U per mu bin, then emitted and timed-out energy.
using Tally = std::array<double, 3 * MU_BINS + 2>;
constexpr std::size_t EMITTED = 3 * MU_BINS, TIMED_OUT = 3 * MU_BINS + 1;
// Light still in flight when a generation ends is dropped with the census.
constexpr double MAX_TIMED_OUT_FRACTION = 1e-3;

enum Case { Scattering, Lambda16, LambdaTau };
char const *const CASE_NAMES[] = {"scattering", "lambda16", "lambdatau"};
constexpr std::size_t DEFAULT_PHOTONS_PER_CELL[] = {75000, 3000, 10000};
// Tolerance in generation-scatter sigmas.  The gate makes 180 comparisons (20
// bins, I/F, Q/I, U/I, three cases) with sigmas estimated from ~32 generations
// (Student t tails): 5 sigma keeps a false FAIL near 0.5% per run, 4 sigma
// would fail ~10% of 16-generation runs by chance.
constexpr double N_SIGMA = 5.0;

// Per-cell absorption and scattering coefficients from the initial conditions.
class SlabOpacity final : public STORM::RadiationOpacityModel<Vector3D, Grid, RadiationCell, 1>
{
public:
    using STORM::RadiationOpacityModel<Vector3D, Grid, RadiationCell, 1>::CalcScatteringOpacity;

    SlabOpacity(std::vector<double> absorption, std::vector<double> scattering,
                std::vector<RadiationCell> const &cells)
        : absorption_(std::move(absorption)), scattering_(std::move(scattering)), cells_(cells) {}

    double CalcPlanckOpacity(RadiationCell const &c) override { return absorption_[index(c)]; }
    double CalcPlanckOpacityAtTemperature(RadiationCell const &c, double) override { return absorption_[index(c)]; }
    double CalcAbsorptionOpacity(RadiationCell const &c, double) override { return absorption_[index(c)]; }
    double CalcAbsorptionOpacityAtTemperature(RadiationCell const &c, double, double) override
    {
        return absorption_[index(c)];
    }
    double CalcScatteringOpacity(RadiationCell const &c) override { return scattering_[index(c)]; }
    double CalcScatteringOpacity(RadiationCell const &c, double) override { return scattering_[index(c)]; }
    double CalcScatteringOpacityAtTemperature(RadiationCell const &c, double) override
    {
        return scattering_[index(c)];
    }
    double CalcScatteringOpacityAtTemperature(RadiationCell const &c, double, double) override
    {
        return scattering_[index(c)];
    }

private:
    std::size_t index(RadiationCell const &c) const { return static_cast<std::size_t>(&c - cells_.data()); }

    std::vector<double> absorption_, scattering_;
    std::vector<RadiationCell> const &cells_;
};

using IMC = STORM::RadiationIMC<Vector3D, Grid, RadiationCell, STORM::SimpleExtensives,
                                STORM::examples::HillierEOS, 1, SlabOpacity>;

struct Options
{
    std::size_t generations = 32;
    std::size_t photonsPerCell = 0;   // 0: the per-case default
    std::size_t lateralCells = 1;     // the mirror walls make one cell per layer enough
    std::string prefix = "ss_table1";
    long seed = -1;   // >= 0: independent single-rank realisation, files tagged _sSEED
    bool asserting = true;
};

// Harrington's B_nu(tau) up to a constant: u0 = h nu / k Teff = 20, grey
// T^4 = (3/4) Teff^4 (tau + q(tau)) with q a fit to the Hopf function.
constexpr double U0_TEFF = 20.0;
double hopf(double tau) { return 0.710446 - 0.133311 * std::exp(-3.4488 * tau); }
double const U_SURFACE = U0_TEFF * std::pow(4.0 / (3.0 * hopf(0.0)), 0.25);   // h nu / k T(0)

double planck(double tau)
{
    return 1.0 / std::expm1(U_SURFACE * std::pow(hopf(0.0) / (tau + hopf(tau)), 0.25));
}

double layerAverage(double tauLo, double tauHi)
{
    constexpr int n = 32;
    double sum = 0.0;
    for(int i = 0; i < n; ++i)
        sum += planck(tauLo + (tauHi - tauLo) * (i + 0.5) / n);
    return sum / n;
}

// Opacity is one per cm, so x and optical depth share units.
struct Slab
{
    std::vector<double> xEdges;
    double tauMax = 0.0;
    double sourceTop = 0.0;   // scattering case: top of the emitting layer
    double surface = 0.0;
    double observerX = 0.0;
};

Slab makeSlab(Case c)
{
    Slab s;
    s.tauMax = (c == Scattering) ? 10.0 : 40.0;
    // Layers in optical depth: geometric from dtau0 up to 1, then unit layers.
    double d = (c == Scattering) ? 0.05 : 0.01;
    double const growth = (c == Scattering) ? 1.2 : 1.15;
    std::vector<double> tau{0.0};
    while(tau.back() < s.tauMax - 1e-12)
    {
        tau.push_back(std::min(s.tauMax, tau.back() + d));
        d = std::min(1.0, d * growth);
    }
    s.sourceTop = (c == Scattering) ? 1.0 : 0.0;
    s.surface = s.sourceTop + s.tauMax;
    if(c == Scattering)
        s.xEdges.push_back(0.0);
    for(auto it = tau.rbegin(); it != tau.rend(); ++it)
        s.xEdges.push_back(s.surface - *it);
    // Vacuum above; the observer sits mid-cell in a thin first cell (a plane on
    // a face is reached together with the face, and the face wins).
    s.xEdges.push_back(s.surface + 0.02);
    s.xEdges.push_back(s.surface + 2.0);
    s.observerX = s.surface + 0.01;
    return s;
}

void writeTallies(std::string const &name, Case c, Slab const &slab, Options const &opt, std::size_t photonsPerCell,
                  std::vector<Tally> const &tallies)
{
    std::FILE *fh = std::fopen(name.c_str(), "w");
    if(!fh)
        throw std::runtime_error("cannot write " + name);
    std::fprintf(fh, "# Shapiro & Sutherland (1982) Table 1, STORM plane observer, case %s\n", CASE_NAMES[c]);
    std::fprintf(fh, "# tau_max %.6g  u0_teff %.6g  generations %zu  photons_per_cell %zu\n",
                 slab.tauMax, U0_TEFF, opt.generations, photonsPerCell);
    std::fprintf(fh, "# generation bin mu_lo mu_hi energy stokesQ_meridian stokesU_meridian\n");
    std::fprintf(fh, "# Q = I_l - I_r in the meridian frame; Shapiro & Sutherland's Q/I is -Q/I here\n");
    for(std::size_t g = 0; g < tallies.size(); ++g)
        for(std::size_t b = 0; b < MU_BINS; ++b)
            std::fprintf(fh, "%zu %zu %.4f %.4f %.10e %.10e %.10e\n", g, b,
                         static_cast<double>(b) / MU_BINS, static_cast<double>(b + 1) / MU_BINS,
                         tallies[g][b], tallies[g][MU_BINS + b], tallies[g][2 * MU_BINS + b]);
    std::fclose(fh);
    std::cout << "  wrote " << name << std::endl;
}

// Bin I/F, Q/I and U/I with errors from the scatter between generations (delta
// method on the ratios), checked against the deterministic bins; and the light
// left in flight.
bool checkBins(Case c, std::vector<Tally> const &tallies)
{
    std::size_t const n = tallies.size();
    Tally sum{};
    std::vector<double> genTotal(n, 0.0);
    for(std::size_t g = 0; g < n; ++g)
        for(std::size_t j = 0; j < sum.size(); ++j)
        {
            sum[j] += tallies[g][j];
            if(j < MU_BINS)
                genTotal[g] += tallies[g][j];
        }
    double const total = std::accumulate(genTotal.begin(), genTotal.end(), 0.0);
    double const corr = n > 1 ? static_cast<double>(n) / static_cast<double>(n - 1) : 0.0;
    auto const &ref = shapiro_reference::table(c);

    bool ok = true;
    double const timedOut = sum[EMITTED] > 0.0 ? sum[TIMED_OUT] / sum[EMITTED] : 0.0;
    std::printf("  timed-out fraction %.3g\n", timedOut);
    if(!(timedOut <= MAX_TIMED_OUT_FRACTION))
    {
        std::printf("  FAIL: %s timed-out fraction %.3g above %.0e: lengthen dt\n", CASE_NAMES[c], timedOut,
                    MAX_TIMED_OUT_FRACTION);
        ok = false;
    }
    std::cout << "   mu_bin       I/F    +-      ref      Q/I     +-      ref      U/I\n";
    for(std::size_t b = 0; b < MU_BINS; ++b)
    {
        double const lo = static_cast<double>(b) / MU_BINS, hi = static_cast<double>(b + 1) / MU_BINS;
        double const E = sum[b];
        double const iOverF = E / (0.5 * (hi * hi - lo * lo)) / (2.0 * total);
        double const q = sum[MU_BINS + b] / E, u = sum[2 * MU_BINS + b] / E;
        double vi = 0.0, vq = 0.0;
        for(std::size_t g = 0; g < n; ++g)
        {
            double const di = tallies[g][b] / E - genTotal[g] / total;
            double const dq = (tallies[g][MU_BINS + b] - q * tallies[g][b]) / E;
            vi += di * di;
            vq += dq * dq;
        }
        double const sigmaI = iOverF * std::sqrt(corr * vi), sigmaQ = std::sqrt(corr * vq);
        double const p = -q;   // Shapiro & Sutherland's sign: positive for E parallel to the surface
        std::printf("  %4.2f-%4.2f  %7.4f %6.4f  %7.4f  %7.4f %6.4f  %7.4f  %+7.4f\n", lo, hi, iOverF, sigmaI,
                    ref[b][0], p, sigmaQ, ref[b][1], u);
        std::string bad;
        if(!(E > 0.0 && std::isfinite(iOverF) && std::isfinite(p)))
            bad = "no light recorded";
        else
        {
            if(!(std::abs(iOverF - ref[b][0]) <= std::max(N_SIGMA * sigmaI, 0.01 * ref[b][0])))
                bad += " I/F " + std::to_string(iOverF) + " vs " + std::to_string(ref[b][0]);
            if(!(std::abs(p - ref[b][1]) <= std::max(N_SIGMA * sigmaQ, 0.003)))
                bad += " Q/I " + std::to_string(p) + " vs " + std::to_string(ref[b][1]);
            if(!(std::abs(u) <= std::max(N_SIGMA * sigmaQ, 0.003)))
                bad += " U/I " + std::to_string(u) + " not zero";
        }
        if(!bad.empty())
        {
            std::printf("  FAIL: %s bin %zu:%s%s\n", CASE_NAMES[c], b, bad[0] == ' ' ? "" : " ", bad.c_str());
            ok = false;
        }
    }
    return ok;
}

bool runCase(Case c, Options const &opt, int rank, int nprocs)
{
    Slab const slab = makeSlab(c);
    std::size_t const photonsPerCell = opt.photonsPerCell > 0 ? opt.photonsPerCell : DEFAULT_PHOTONS_PER_CELL[c];
    double const halfSide = 100.0;   // any width: the mirror walls make the slab infinite
    std::size_t const nl = opt.lateralCells;
    Grid grid(slab.xEdges, -halfSide, halfSide, nl, -halfSide, halfSide, nl);
    std::vector<double> weights((slab.xEdges.size() - 1) * nl * nl, 1.0);
    grid.BuildParallel(weights);
    std::size_t const Ncells = grid.GetPointNo();

    double const cvPerVolume = 1.0e20, Tref = 1.0e4, Tfloor = 1.0;
    double const Bref = planck(slab.tauMax);
    std::vector<RadiationCell> cells(Ncells);
    std::vector<STORM::SimpleExtensives> extensives(Ncells);
    std::vector<double> absorption(Ncells, 0.0), scattering(Ncells, 0.0);
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        double const x = grid.GetCellCM(i).x;
        double T = Tfloor;
        if(c == Scattering && x < slab.sourceTop)
        {
            // Thin emitting layer (tau_a = 0.2), so it does not reabsorb its own
            // light; under a thick conservative slab the emergent law does not
            // depend on how the slab is lit.
            absorption[i] = 0.2;
            T = Tref;
        }
        else if(c == Scattering && x < slab.surface)
            scattering[i] = 1.0;
        else if(x < slab.surface)
        {
            auto const top = std::upper_bound(slab.xEdges.begin(), slab.xEdges.end(), x);
            double const tau = slab.surface - x;
            double const lambda = (c == Lambda16) ? 1.0 / 6.0 : tau / (1.0 + tau);
            absorption[i] = lambda;
            scattering[i] = 1.0 - lambda;
            T = Tref * std::pow(layerAverage(slab.surface - *top, slab.surface - *(top - 1)) / Bref, 0.25);
        }
        double const volume = grid.GetVolume(i);
        cells[i].temperature = T;
        cells[i].internalEnergy = cvPerVolume * T * volume;
        cells[i].Erad = 0.0;
        extensives[i].mass = volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
        extensives[i].Erad = 0.0;
    }

    STORM::RadiationIMCParameters<1> params;
    params.newPhotonsPerCell = photonsPerCell;
    params.withHydro = false;
    params.staticScatterers = true;
    params.withPolarization = true;
    params.withDDMC = false;
    params.withRandomWalk = false;
    params.energyBoundaries = {0.0, 1.0e30};
    params.energyBoundariesProvided = true;
    // The escaping light of the lambda = 1/6 atmosphere comes from tau ~ 5-10;
    // STORM's default cutoff (1e-3 of the initial weight) would cut part of it.
    params.weightCutoffFraction = 1e-8;

    auto eos = std::make_shared<STORM::examples::HillierEOS>(cvPerVolume);
    auto opacity = std::make_shared<SlabOpacity>(std::move(absorption), std::move(scattering), cells);
    auto boundary = std::make_shared<STORM::examples::SlabBoundary<Vector3D, Grid>>(grid);
    auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacity, params);
    auto popControl = std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 1, 6.0);
    auto observer = std::make_shared<STORM::examples::PlaneObserver<Vector3D>>(slab.observerX, MU_BINS);
    physics->setObserver(observer);
    STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
        STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, popControl, boundary);

    if(rank == 0)
        std::cout << "\nShapiro & Sutherland Table 1, case " << CASE_NAMES[c] << ": tau_max = " << slab.tauMax
                  << ", layers = " << slab.xEdges.size() - 1 << ", lateral cells = " << nl << "x" << nl
                  << ", photons per cell = " << photonsPerCell << ", ranks = " << nprocs << std::endl;

    // Long enough for every photon to escape or be absorbed; the census is
    // dropped between generations.
    double const dt = 1.0e5 / units::clight;
    std::vector<Tally> tallies;
    for(std::size_t g = 0; g < opt.generations; ++g)
    {
        manager.getParticles().clear();
        observer->resetTallies();
        if(opt.seed >= 0)
            physics->reseedRNG(static_cast<std::uint64_t>(opt.seed) * 1000003ULL +
                               static_cast<std::uint64_t>(c) * 10007ULL + g);
        manager.step(dt);
        Tally t{};
        std::copy(observer->energy.begin(), observer->energy.end(), t.begin());
        std::copy(observer->stokesQ.begin(), observer->stokesQ.end(), t.begin() + MU_BINS);
        std::copy(observer->stokesU.begin(), observer->stokesU.end(), t.begin() + 2 * MU_BINS);
        t[EMITTED] = observer->emitted;
        t[TIMED_OUT] = observer->timedOut;
        MPI_Allreduce(MPI_IN_PLACE, t.data(), static_cast<int>(t.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tallies.push_back(t);
        if(rank == 0 && ((g + 1) % 8 == 0 || g + 1 == opt.generations))
            std::cout << "  generation " << (g + 1) << "/" << opt.generations << std::endl;
    }

    int ok = 1;
    if(rank == 0)
    {
        writeTallies(opt.prefix + "_" + CASE_NAMES[c] + (opt.seed >= 0 ? "_s" + std::to_string(opt.seed) : "") + ".txt",
                     c, slab, opt, photonsPerCell, tallies);
        ok = checkBins(c, tallies) || !opt.asserting;
        std::cout << (ok ? "PASS" : "FAIL") << " shapiro_sutherland_polarization " << CASE_NAMES[c] << std::endl;
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

// Integer option value of at least min; std::stoul would accept "-1" as a huge count.
std::size_t parseCount(std::string const &option, std::string const &value, long long min)
{
    std::size_t used = 0;
    long long v = min - 1;
    try
    {
        v = std::stoll(value, &used);
    }
    catch(std::exception const &)
    {
        used = 0;
    }
    if(used == 0 || used != value.size() || v < min)
        throw std::invalid_argument(option + " needs an integer >= " + std::to_string(min) + ", got '" + value + "'");
    return static_cast<std::size_t>(v);
}

} // namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    bool ok = true;
    try
    {
        int rank = 0, nprocs = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        Options opt;
        std::string which = "all";
        for(int a = 1; a < argc; ++a)
        {
            std::string const arg(argv[a]);
            auto next = [&]() {
                if(a + 1 >= argc)
                    throw std::invalid_argument(arg + " needs a value");
                return std::string(argv[++a]);
            };
            if(arg == "--case") which = next();
            else if(arg == "--generations") opt.generations = parseCount(arg, next(), 2);   // errors come from the scatter
            else if(arg == "--photons-per-cell") opt.photonsPerCell = parseCount(arg, next(), 1);
            else if(arg == "--lateral-cells") opt.lateralCells = parseCount(arg, next(), 1);
            else if(arg == "--prefix") opt.prefix = next();
            else if(arg == "--seed") opt.seed = static_cast<long>(parseCount(arg, next(), 0));
            else if(arg == "--no-assert") opt.asserting = false;
            else throw std::invalid_argument("unknown option " + arg);
        }
        if(which != "all" && which != CASE_NAMES[Scattering] && which != CASE_NAMES[Lambda16] &&
           which != CASE_NAMES[LambdaTau])
            throw std::invalid_argument("unknown case " + which);
        for(Case c : {Scattering, Lambda16, LambdaTau})
            if(which == "all" || which == CASE_NAMES[c])
                ok = runCase(c, opt, rank, nprocs) && ok;
    }
    catch(std::exception const &e)
    {
        // Errors can be raised on rank 0 alone (output, analysis) while the other
        // ranks wait in a collective: abort the whole job rather than hang.
        std::cerr << "shapiro_sutherland_polarization: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return ok ? 0 : 1;
}
