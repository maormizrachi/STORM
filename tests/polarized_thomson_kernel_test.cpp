// Checks polarization::scatterPolarizedThomson, the polarized Thomson
// scattering kernel, against the implementation it replaced (a separate
// samplePolarizedThomsonDirection followed by applyThomsonScatter, copied
// below as the reference).  Both consume the same random numbers in the same
// order, so on a shared counter stream they must give the same direction,
// Stokes state and basis to rounding and leave the stream at the same counter.
//
// Covered: single scatterings from random states (polarized, unpolarized,
// fully polarized, basis not yet perpendicular to the direction, packets not
// yet initialized); forward/backward scattering, the 10000-rejection
// fallback and a non-finite Stokes state, driven by scripted random numbers;
// and the whole DDMC-exit rebuild
// finalizeAcceleratedPolarizationHistory for pending means from below one to
// far above the manual cap.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "radiation/Polarization.hpp"
#include "utils/CounterRNG.hpp"

#ifndef MONTECARLO_POLARIZATION
#error "polarized_thomson_kernel_test needs MONTECARLO_POLARIZATION"
#endif

namespace {

struct TestPoint
{
    using coord_type = double;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    TestPoint() = default;
    TestPoint(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
};

TestPoint operator+(TestPoint const &a, TestPoint const &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
TestPoint operator-(TestPoint const &a, TestPoint const &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
TestPoint operator*(TestPoint const &a, double s) { return {a.x * s, a.y * s, a.z * s}; }
TestPoint operator/(TestPoint const &a, double s) { return {a.x / s, a.y / s, a.z / s}; }
TestPoint &operator/=(TestPoint &a, double s) { a.x /= s; a.y /= s; a.z /= s; return a; }

struct TestRadiationState
{
    double pendingMeanScatterings = 0.0;
};

struct TestParticle
{
    TestPoint velocity;
    TestPoint polarizationBasis;
    double stokesQ = 0.0;
    double stokesU = 0.0;
    bool polarizationInitialized = false;
    TestRadiationState radiationState;
};

namespace pol = STORM::polarization;
namespace fb = STORM::fallback;

// ---- Reference: the implementation before scatterPolarizedThomson ----------
namespace reference {

template<typename PointT>
void buildBasisAroundDirection(const PointT &k, PointT &e1, PointT &e2)
{
    PointT const kk = pol::safeNormalize(k, PointT(0.0, 0.0, 1.0));
    e1 = pol::choosePerpendicularBasis(kk);
    e2 = pol::safeNormalize(fb::CrossProduct(kk, e1), PointT(0.0, 1.0, 0.0));
}

template<typename PointT, typename Uniform01>
PointT sampleSyntheticThomsonDirection(const PointT &oldVelocity, Uniform01 &&u01)
{
    PointT const k = pol::safeNormalize(oldVelocity, PointT(0.0, 0.0, 1.0));
    double const speed = std::max(fb::abs(oldVelocity), pol::kEpsilon);
    PointT e1, e2;
    buildBasisAroundDirection(k, e1, e2);
    double const mu = pol::sampleThomsonMu(u01);
    double const phi = 2.0 * pol::kPi * u01();
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    PointT const dir = k * mu + e1 * (sinTheta * std::cos(phi)) +
                       e2 * (sinTheta * std::sin(phi));
    return pol::safeNormalize(dir, k) * speed;
}

template<typename PointT, typename ParticleT>
std::pair<double, double> projectToBasis(const ParticleT &p, const PointT &direction,
                                         const PointT &basis)
{
    if(!p.polarizationInitialized)
        return {0.0, 0.0};
    PointT const k = pol::safeNormalize(direction, PointT(0.0, 0.0, 1.0));
    PointT const eOld = pol::projectBasisToDirection(p.polarizationBasis, k);
    PointT const eNew = pol::projectBasisToDirection(basis, k);
    double const c = std::clamp(fb::ScalarProd(eOld, eNew), -1.0, 1.0);
    double const s = fb::ScalarProd(k, fb::CrossProduct(eOld, eNew));
    double const cos2 = c * c - s * s;
    double const sin2 = 2.0 * c * s;
    double q = p.stokesQ * cos2 + p.stokesU * sin2;
    double u = -p.stokesQ * sin2 + p.stokesU * cos2;
    pol::clampLinearPolarization(q, u);
    return {q, u};
}

template<typename PointT, typename ParticleT, typename Uniform01>
PointT samplePolarizedThomsonDirection(const ParticleT &p, const PointT &oldVelocity,
                                       Uniform01 &&u01)
{
    PointT const kIn = pol::safeNormalize(oldVelocity, PointT(0.0, 0.0, 1.0));
    for(int tries = 0; tries < 10000; ++tries)
    {
        PointT const candidate = sampleSyntheticThomsonDirection(oldVelocity, u01);
        PointT const kOut = pol::safeNormalize(candidate, kIn);
        PointT planeNormal = fb::CrossProduct(kIn, kOut);
        if(fb::abs(planeNormal) <= 1.0e-10)
            return candidate;
        planeNormal = pol::safeNormalize(planeNormal, PointT(0.0, 0.0, 1.0));
        auto const qu = projectToBasis(p, oldVelocity, planeNormal);
        double const mu = std::clamp(fb::ScalarProd(kIn, kOut), -1.0, 1.0);
        double const mu2 = mu * mu;
        double const ratio = 1.0 + (1.0 - mu2) / (1.0 + mu2) * qu.first;
        if(u01() <= std::clamp(0.5 * ratio, 0.0, 1.0))
            return candidate;
    }
    return sampleSyntheticThomsonDirection(oldVelocity, u01);
}

template<typename PointT, typename ParticleT, typename Uniform01>
void scatter(ParticleT &p, Uniform01 &&u01)
{
    PointT const oldVelocity = p.velocity;
    PointT const next = samplePolarizedThomsonDirection(p, oldVelocity, u01);
    pol::applyThomsonScatter<PointT>(p, oldVelocity, next);
}

template<typename PointT, typename ParticleT, typename Uniform01>
void applyManualSyntheticScatterings(ParticleT &p, int nManual, const PointT &finalVelocity,
                                      Uniform01 &&u01)
{
    if(nManual <= 0)
    {
        p.velocity = finalVelocity;
        p.polarizationBasis = pol::projectBasisToDirection(p.polarizationBasis, finalVelocity);
        return;
    }
    PointT current = p.velocity;
    for(int i = 0; i < nManual; ++i)
    {
        PointT const next = i == nManual - 1
            ? finalVelocity : samplePolarizedThomsonDirection(p, current, u01);
        pol::applyThomsonScatter<PointT>(p, current, next);
        current = next;
    }
}

template<typename PointT, typename ParticleT, typename RandomEngine, typename UniformDist>
void finalizeAcceleratedPolarizationHistory(ParticleT &p, const PointT &finalVelocityCo,
                                             int manualScatteringsAfterAcceleration,
                                             double depolarizationScatterings,
                                             RandomEngine &engine, UniformDist &uniformDist)
{
    pol::initializeIfNeeded<PointT>(p);
    auto u01 = [&]() {
        return std::clamp(uniformDist(engine), std::numeric_limits<double>::min(),
                          1.0 - std::numeric_limits<double>::epsilon());
    };
    int const K = std::max(0, manualScatteringsAfterAcceleration);
    double mean = p.radiationState.pendingMeanScatterings;
    if(std::isnan(mean)) mean = std::numeric_limits<double>::max();
    else if(!(mean > 0.0)) mean = 0.0;
    else if(!std::isfinite(mean)) mean = std::numeric_limits<double>::max();
    pol::clearAcceleratedHistory<PointT>(p);

    std::uint64_t n = 0;
    if(mean > 0.0 && std::isfinite(depolarizationScatterings) && depolarizationScatterings > 0.0)
    {
        double const cutoff = static_cast<double>(K) + 80.0 * depolarizationScatterings;
        if(mean <= cutoff + 10.0 * std::sqrt(std::max(1.0, mean)))
        {
            std::poisson_distribution<unsigned long long> poisson(mean);
            n = static_cast<std::uint64_t>(poisson(engine));
        }
        else
        {
            n = static_cast<std::uint64_t>(std::ceil(cutoff + 1.0));
        }
    }
    std::uint64_t const nDamped = n > static_cast<std::uint64_t>(K)
        ? n - static_cast<std::uint64_t>(K) : 0;
    double damping = 1.0;
    if(nDamped > 0)
    {
        double const exponent = -static_cast<double>(nDamped) / depolarizationScatterings;
        damping = exponent < -745.0 ? 0.0 : std::exp(exponent);
        p.velocity = pol::sampleIsotropicVelocity<PointT>(p.velocity, u01);
        p.polarizationBasis = pol::projectBasisToDirection(p.polarizationBasis, p.velocity);
    }
    p.stokesQ *= damping;
    p.stokesU *= damping;
    pol::clampLinearPolarization(p.stokesQ, p.stokesU);
    applyManualSyntheticScatterings<PointT>(p, static_cast<int>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(K), n)), finalVelocityCo, u01);
    p.velocity = finalVelocityCo;
    p.polarizationBasis = pol::projectBasisToDirection(p.polarizationBasis, p.velocity);
    pol::clampLinearPolarization(p.stokesQ, p.stokesU);
}

} // namespace reference

// ---- Test helpers ------------------------------------------------------------

constexpr double kSpeed = 2.99792458e10;

struct Difference
{
    double velocity = 0.0;
    double stokes = 0.0;
    double basis = 0.0;
    std::uint64_t counterMismatches = 0;
    std::uint64_t badInvariants = 0;
    std::uint64_t cases = 0;

    void add(TestParticle const &a, std::uint64_t counterA,
             TestParticle const &b, std::uint64_t counterB)
    {
        ++cases;
        if(counterA != counterB)
        {
            ++counterMismatches;
            return;
        }
        velocity = std::max(velocity, fb::abs(a.velocity - b.velocity) / kSpeed);
        stokes = std::max(stokes, std::max(std::abs(a.stokesQ - b.stokesQ),
                                           std::abs(a.stokesU - b.stokesU)));
        basis = std::max(basis, fb::abs(a.polarizationBasis - b.polarizationBasis));
        double const k = fb::abs(b.velocity);
        double const e = fb::abs(b.polarizationBasis);
        bool const ok = std::abs(k / kSpeed - 1.0) < 1e-12 && std::abs(e - 1.0) < 1e-12 &&
                        std::abs(fb::ScalarProd(b.velocity / k, b.polarizationBasis)) < 1e-12 &&
                        b.stokesQ * b.stokesQ + b.stokesU * b.stokesU <= 1.0 + 1e-15 &&
                        b.polarizationInitialized;
        if(!ok)
            ++badInvariants;
    }

    bool pass(double tolerance) const
    {
        return counterMismatches == 0 && badInvariants == 0 && velocity <= tolerance &&
               stokes <= tolerance && basis <= tolerance;
    }

    void print(char const *what) const
    {
        std::printf("%-44s cases %8llu  max |dv|/c %.2e  max |dqu| %.2e  max |de| %.2e"
                    "  counter mismatches %llu  bad invariants %llu\n",
                    what, static_cast<unsigned long long>(cases), velocity, stokes, basis,
                    static_cast<unsigned long long>(counterMismatches),
                    static_cast<unsigned long long>(badInvariants));
    }
};

TestPoint randomUnit(std::mt19937_64 &rng)
{
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    double const mu = unit(rng);
    double const phi = std::acos(-1.0) * unit(rng);
    double const s = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return {s * std::cos(phi), s * std::sin(phi), mu};
}

TestParticle randomState(std::mt19937_64 &rng, int variant)
{
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    TestParticle p;
    p.velocity = randomUnit(rng) * kSpeed;
    // Not exactly perpendicular, so the projection onto the plane of the sky
    // is exercised as well.
    TestPoint const k = p.velocity / kSpeed;
    TestPoint basis = randomUnit(rng);
    basis = basis - k * (0.9 * fb::ScalarProd(basis, k));
    p.polarizationBasis = basis;
    double degree = 0.0;
    switch(variant % 4)
    {
    case 0: degree = unit01(rng); break;           // partly polarized
    case 1: degree = 0.0; break;                   // unpolarized
    case 2: degree = 1.0; break;                   // fully polarized
    default: degree = 0.05 * unit01(rng); break;   // weakly polarized
    }
    double const angle = 2.0 * std::acos(-1.0) * unit01(rng);
    p.stokesQ = degree * std::cos(angle);
    p.stokesU = degree * std::sin(angle);
    p.polarizationInitialized = variant % 17 != 0;
    if(!p.polarizationInitialized)
        p.stokesQ = p.stokesU = 0.0;
    return p;
}

template<typename Engine>
auto makeUnit01(Engine &engine, std::uniform_real_distribution<double> &dist)
{
    return [&engine, &dist]() {
        return std::clamp(dist(engine), std::numeric_limits<double>::min(),
                          1.0 - std::numeric_limits<double>::epsilon());
    };
}

bool testSingleScatterings(std::uint64_t cases)
{
    std::mt19937_64 rng(20261118);
    Difference diff;
    for(std::uint64_t i = 0; i < cases; ++i)
    {
        TestParticle const start = randomState(rng, static_cast<int>(i));
        std::uint64_t const key = STORM::CounterRNG::makeKey(7, 3, i);
        std::uint64_t counterRef = 1000 + i;
        std::uint64_t counterNew = counterRef;
        STORM::ParticleCounterEngine engineRef(key, counterRef);
        STORM::ParticleCounterEngine engineNew(key, counterNew);
        std::uniform_real_distribution<double> distRef(0.0, 1.0);
        std::uniform_real_distribution<double> distNew(0.0, 1.0);
        auto u01Ref = makeUnit01(engineRef, distRef);
        auto u01New = makeUnit01(engineNew, distNew);

        TestParticle ref = start;
        TestParticle fast = start;
        // Successive scatterings of one packet, as in IMC transport.
        for(int s = 0; s < 4; ++s)
        {
            reference::scatter<TestPoint>(ref, u01Ref);
            pol::scatterPolarizedThomson<TestPoint>(fast, u01New);
        }
        diff.add(ref, counterRef, fast, counterNew);
    }
    diff.print("4 successive scatterings");
    return diff.pass(1e-10);
}

bool testFinalize(double pendingMean, int manual, std::uint64_t cases)
{
    std::mt19937_64 rng(static_cast<std::uint64_t>(pendingMean * 1000.0) + 17);
    Difference diff;
    for(std::uint64_t i = 0; i < cases; ++i)
    {
        TestParticle start = randomState(rng, static_cast<int>(i));
        start.radiationState.pendingMeanScatterings = pendingMean;
        TestPoint const finalVelocity = randomUnit(rng) * kSpeed;
        std::uint64_t const key = STORM::CounterRNG::makeKey(11, 5, i);
        std::uint64_t counterRef = 77 + 3 * i;
        std::uint64_t counterNew = counterRef;
        STORM::ParticleCounterEngine engineRef(key, counterRef);
        STORM::ParticleCounterEngine engineNew(key, counterNew);
        std::uniform_real_distribution<double> distRef(0.0, 1.0);
        std::uniform_real_distribution<double> distNew(0.0, 1.0);

        TestParticle ref = start;
        TestParticle fast = start;
        reference::finalizeAcceleratedPolarizationHistory<TestPoint>(
            ref, finalVelocity, manual, 0.5, engineRef, distRef);
        pol::finalizeAcceleratedPolarizationHistory<TestPoint>(
            fast, finalVelocity, manual, 0.5, engineNew, distNew);
        diff.add(ref, counterRef, fast, counterNew);
    }
    char label[96];
    std::snprintf(label, sizeof(label), "DDMC exit, pending mean %g, K %d", pendingMean, manual);
    diff.print(label);
    return diff.pass(1e-9);
}

// Replays a fixed cycle of "random" numbers and counts the draws, to drive
// the branches random states practically never reach.
struct ScriptedUnit
{
    std::vector<double> cycle;
    std::uint64_t draws = 0;
    double operator()() { return cycle[draws++ % cycle.size()]; }
};

bool testScripted(char const *what, TestParticle const &start, std::vector<double> const &cycle)
{
    ScriptedUnit unitRef{cycle};
    ScriptedUnit unitNew{cycle};
    TestParticle ref = start;
    TestParticle fast = start;
    reference::scatter<TestPoint>(ref, unitRef);
    pol::scatterPolarizedThomson<TestPoint>(fast, unitNew);
    Difference diff;
    diff.add(ref, unitRef.draws, fast, unitNew.draws);
    diff.print(what);
    return diff.pass(1e-12);
}

bool testRareBranches()
{
    std::mt19937_64 rng(31);
    TestParticle polarized = randomState(rng, 0);
    polarized.polarizationInitialized = true;
    TestParticle unpolarized = randomState(rng, 1);
    unpolarized.polarizationInitialized = true;
    bool pass = true;
    // mu = -1 and mu = +1 exactly: no scattering plane.
    pass = testScripted("backward scattering (degenerate)", polarized, {0.0, 0.0, 0.3}) && pass;
    pass = testScripted("forward scattering (degenerate)", polarized, {1.0, 0.0, 0.3}) && pass;
    // mu = 0 accepted, then the polarization acceptance always rejects
    // (q_n = 0, threshold 1/2): 10000 rejections and the fallback draw.
    pass = testScripted("10000 rejections, fallback candidate", unpolarized, {0.5, 0.25, 0.3, 0.999}) && pass;
    // A non-finite Stokes component.  The old code rejected every candidate
    // (its acceptance threshold was NaN) and took the unpolarized fallback
    // after 10000 draws; the new one treats the state as unpolarized at once.
    // Either way the packet scatters as an unpolarized one, which is what is
    // checked here (the draw counts of old and new differ).
    TestParticle broken = polarized;
    broken.stokesQ = std::numeric_limits<double>::quiet_NaN();
    broken.stokesU = 0.3;
    TestParticle clean = polarized;
    clean.stokesQ = 0.0;
    clean.stokesU = 0.0;
    ScriptedUnit unitBroken{{0.4, 0.1, 0.7, 0.2}};
    ScriptedUnit unitClean{{0.4, 0.1, 0.7, 0.2}};
    pol::scatterPolarizedThomson<TestPoint>(broken, unitBroken);
    pol::scatterPolarizedThomson<TestPoint>(clean, unitClean);
    Difference diff;
    diff.add(clean, unitClean.draws, broken, unitBroken.draws);
    diff.print("non-finite Stokes Q scatters as unpolarized");
    return diff.pass(0.0) && pass;
}

} // namespace

int main()
{
    bool pass = testSingleScatterings(400000);
    pass = testRareBranches() && pass;
    for(double mean : {0.0, 0.3, 2.0, 30.0, 127.0, 160.0, 400.0, 1.0e6})
        pass = testFinalize(mean, 128, mean > 100.0 ? 4000 : 20000) && pass;
    pass = testFinalize(50.0, 1, 20000) && pass;
    pass = testFinalize(50.0, 0, 20000) && pass;
    std::printf("%s\n", pass ? "polarized_thomson_kernel_test PASSED"
                             : "polarized_thomson_kernel_test FAILED");
    return pass ? 0 : 1;
}
