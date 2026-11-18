#ifndef STORM_RADIATION_POLARIZATION_HPP
#define STORM_RADIATION_POLARIZATION_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <units/units.hpp>
#include "../elementary/PointOps.hpp"

namespace STORM::polarization {

#ifdef MONTECARLO_POLARIZATION

inline constexpr double kEpsilon = 1.0e-14;
inline constexpr double kPi = 3.141592653589793238462643383279502884;

template<typename PointT>
PointT safeNormalize(const PointT &v, const PointT &fallback)
{
    double const n = fallback::abs(v);
    if(n > kEpsilon && std::isfinite(n))
    {
        return v / n;
    }
    return fallback;
}

template<typename PointT>
PointT choosePerpendicularBasis(const PointT &direction)
{
    PointT const k = safeNormalize(direction, PointT(0.0, 0.0, 1.0));
    PointT const helper = std::abs(k.z) < 0.9
        ? PointT(0.0, 0.0, 1.0) : PointT(0.0, 1.0, 0.0);
    return safeNormalize(helper - k * fallback::ScalarProd(helper, k),
                         PointT(1.0, 0.0, 0.0));
}

template<typename PointT>
PointT projectBasisToDirection(const PointT &basis, const PointT &direction)
{
    PointT const k = safeNormalize(direction, PointT(0.0, 0.0, 1.0));
    PointT const projected = basis - k * fallback::ScalarProd(basis, k);
    // safeNormalize with the fallback built only when it is needed: this runs
    // several times per scattering and the fallback almost never is.
    double const n = fallback::abs(projected);
    if(n > kEpsilon && std::isfinite(n))
    {
        return projected / n;
    }
    return choosePerpendicularBasis(k);
}

inline void clampLinearPolarization(double &q, double &u)
{
    if(!std::isfinite(q)) q = 0.0;
    if(!std::isfinite(u)) u = 0.0;
    double const p2 = q * q + u * u;
    if(p2 > 1.0)
    {
        double const scale = 1.0 / std::sqrt(p2);
        q *= scale;
        u *= scale;
    }
}

template<typename PointT, typename ParticleT>
void initializeIfNeeded(ParticleT &p)
{
    if(p.polarizationInitialized)
    {
        return;
    }
    p.stokesQ = 0.0;
    p.stokesU = 0.0;
    p.polarizationBasis = choosePerpendicularBasis<PointT>(p.velocity);
    p.polarizationInitialized = true;
    p.radiationState.pendingMeanScatterings = 0.0;
}

template<typename PointT, typename ParticleT>
void resetUnpolarized(ParticleT &p)
{
    p.stokesQ = 0.0;
    p.stokesU = 0.0;
    p.polarizationBasis = choosePerpendicularBasis<PointT>(p.velocity);
    p.polarizationInitialized = true;
    p.radiationState.pendingMeanScatterings = 0.0;
}

// Stokes (q, u) of a packet travelling along the unit vector k with polarization
// basis `basis`, re-expressed against `reference` projected on the plane of the
// sky (perpendicular to k).  Observers use it to resolve tallies in a fixed
// frame.  Returns false, leaving qOut and uOut untouched, when the basis or the
// reference has no component perpendicular to k.
template<typename PointT>
bool stokesInReferenceFrame(double q, double u, const PointT &basis, const PointT &k,
                            const PointT &reference, double &qOut, double &uOut)
{
    PointT const e = basis - k * fallback::ScalarProd(basis, k);
    PointT const projected = reference - k * fallback::ScalarProd(reference, k);
    double const projectedNorm = fallback::abs(projected);
    if(!(fallback::abs(e) > 0.0) || !(projectedNorm > 0.0))
        return false;
    PointT const eUnit = fallback::normalize(e);
    PointT const sky = projected / projectedNorm;
    double const c = std::max(-1.0, std::min(1.0, fallback::ScalarProd(eUnit, sky)));
    double const s = fallback::ScalarProd(fallback::CrossProduct(eUnit, sky), k);
    double const c2 = c * c - s * s;
    double const s2 = 2.0 * s * c;
    qOut = q * c2 + u * s2;
    uOut = -q * s2 + u * c2;
    return true;
}

template<typename PointT, typename ParticleT>
void rotateStokesToBasis(ParticleT &p, const PointT &newBasis)
{
    initializeIfNeeded<PointT>(p);
    PointT const k = safeNormalize(p.velocity, PointT(0.0, 0.0, 1.0));
    PointT const eOld = projectBasisToDirection(p.polarizationBasis, k);
    PointT const eNew = projectBasisToDirection(newBasis, k);
    double const c = std::clamp(fallback::ScalarProd(eOld, eNew), -1.0, 1.0);
    double const s = fallback::ScalarProd(k, fallback::CrossProduct(eOld, eNew));
    double const cos2 = c * c - s * s;
    double const sin2 = 2.0 * c * s;
    double const q = p.stokesQ;
    double const u = p.stokesU;
    p.stokesQ = q * cos2 + u * sin2;
    p.stokesU = -q * sin2 + u * cos2;
    p.polarizationBasis = eNew;
    clampLinearPolarization(p.stokesQ, p.stokesU);
}

// Apply the Thomson Mueller matrix in the scattering-plane bases.  The
// particle direction and basis are updated together, so later observer
// projections see a self-consistent Stokes state.
template<typename PointT, typename ParticleT>
void applyThomsonScatter(ParticleT &p,
                         const PointT &oldVelocity,
                         const PointT &newVelocity)
{
    initializeIfNeeded<PointT>(p);
    PointT const kIn = safeNormalize(oldVelocity, PointT(0.0, 0.0, 1.0));
    PointT const kOut = safeNormalize(newVelocity, kIn);
    PointT planeNormal = fallback::CrossProduct(kIn, kOut);
    double const planeNorm = fallback::abs(planeNormal);

    if(planeNorm <= 1.0e-10)
    {
        p.velocity = newVelocity;
        p.polarizationBasis = projectBasisToDirection(p.polarizationBasis, kOut);
        clampLinearPolarization(p.stokesQ, p.stokesU);
        return;
    }

    planeNormal /= planeNorm;
    double const mu = std::clamp(fallback::ScalarProd(kIn, kOut), -1.0, 1.0);
    double const mu2 = mu * mu;
    rotateStokesToBasis(p, planeNormal);
    double const q = p.stokesQ;
    double const u = p.stokesU;
    double const Iprime = (1.0 + mu2) + (1.0 - mu2) * q;
    double const Qprime = (1.0 - mu2) + (1.0 + mu2) * q;
    double const Uprime = 2.0 * mu * u;

    p.velocity = newVelocity;
    if(std::abs(Iprime) <= kEpsilon || !std::isfinite(Iprime))
    {
        p.stokesQ = 0.0;
        p.stokesU = 0.0;
    }
    else
    {
        p.stokesQ = Qprime / Iprime;
        p.stokesU = Uprime / Iprime;
    }
    p.polarizationBasis = projectBasisToDirection(planeNormal, kOut);
    p.polarizationInitialized = true;
    clampLinearPolarization(p.stokesQ, p.stokesU);
}

template<typename Uniform01>
double sampleThomsonMu(Uniform01 &&u01)
{
    for(int tries = 0; tries < 10000; ++tries)
    {
        double const mu = 2.0 * u01() - 1.0;
        if(u01() <= 0.5 * (1.0 + mu * mu))
        {
            return mu;
        }
    }
    return 2.0 * u01() - 1.0;
}

// One polarized Thomson scattering of p in place: samples the new direction
// from the phase function for the packet's current Stokes state, applies the
// Thomson Mueller matrix, and leaves the basis on the new scattering-plane
// normal.
//
// Sampling is by rejection: the direction is drawn from the unpolarized
// (1 + mu^2) phase function with the azimuth uniform in the frame (e1, e2)
// around the incoming direction k, and accepted with probability
// 0.5 [1 + (1 - mu^2) / (1 + mu^2) q_n], q_n being the packet's Q in the
// frame of the scattering-plane normal n.  That gives the polarized phase
// function (1 + mu^2) + (1 - mu^2) q_n; the acceptance is 1/2 on average.
//
// Everything that depends only on k is built once per scattering, not per
// candidate, and the plane normal and the Stokes rotation of a candidate
// follow algebraically from its azimuth: with kOut = mu k + s (cos phi e1 +
// sin phi e2), k x kOut = s (cos phi e2 - sin phi e1), so the unit normal is
// n = cos phi e2 - sin phi e1 and the rotation from the packet basis e to n is
// read off e.n and (k x e).n.  The random numbers are drawn in the same order
// and number as by the separate sample-then-scatter it replaces (mu, phi, then
// the acceptance draw), so the result matches that to rounding.
template<typename PointT, typename ParticleT, typename Uniform01>
void scatterPolarizedThomson(ParticleT &p, Uniform01 &&u01)
{
    initializeIfNeeded<PointT>(p);
    // Unit vectors are normalized with one reciprocal each; vectors that are
    // unit by construction (e2, f, the plane normal) are not renormalized.
    auto unitOr = [](const PointT &v, const PointT &fallbackVector)
    {
        double const n = fallback::abs(v);
        return n > kEpsilon && std::isfinite(n) ? v * (1.0 / n) : fallbackVector;
    };
    double const speed = std::max(fallback::abs(p.velocity), kEpsilon);
    PointT const k = unitOr(p.velocity, PointT(0.0, 0.0, 1.0));
    // The azimuth frame of the sampling (as choosePerpendicularBasis).
    PointT const helper = std::abs(k.z) < 0.9 ? PointT(0.0, 0.0, 1.0) : PointT(0.0, 1.0, 0.0);
    PointT const e1 = unitOr(helper - k * fallback::ScalarProd(helper, k), PointT(1.0, 0.0, 0.0));
    PointT const e2 = fallback::CrossProduct(k, e1);
    // The packet's polarization basis on the plane of the sky.
    PointT const basis = p.polarizationBasis;
    PointT e = basis - k * fallback::ScalarProd(basis, k);
    {
        double const n = fallback::abs(e);
        e = n > kEpsilon && std::isfinite(n) ? e * (1.0 / n) : choosePerpendicularBasis(k);
    }
    PointT const f = fallback::CrossProduct(k, e);
    double q = p.stokesQ;
    double u = p.stokesU;
    // A non-finite component makes the whole state unpolarized (the rotation
    // spreads it to both components before they are clamped).
    if(!std::isfinite(q) || !std::isfinite(u))
    {
        q = 0.0;
        u = 0.0;
    }
    clampLinearPolarization(q, u);

    double mu = 0.0;
    double sinTheta = 0.0;
    double cosPhi = 1.0;
    double sinPhi = 0.0;
    double qn = 0.0;
    double un = 0.0;
    bool degenerate = false;
    auto drawCandidate = [&]()
    {
        mu = sampleThomsonMu(u01);
        double const phi = 2.0 * kPi * u01();
        sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        cosPhi = std::cos(phi);
        sinPhi = std::sin(phi);
        degenerate = sinTheta <= 1.0e-10;
        if(degenerate)
        {
            return;
        }
        PointT const n = e2 * cosPhi - e1 * sinPhi;
        double const c = std::clamp(fallback::ScalarProd(e, n), -1.0, 1.0);
        double const s = fallback::ScalarProd(f, n);
        double const cos2 = c * c - s * s;
        double const sin2 = 2.0 * c * s;
        qn = q * cos2 + u * sin2;
        un = -q * sin2 + u * cos2;
        clampLinearPolarization(qn, un);
    };

    // Accept with probability 0.5 [1 + (1 - mu^2) / (1 + mu^2) q_n], written
    // without the division; |q_n| <= 1 keeps it in [0, 1].
    bool accepted = false;
    for(int tries = 0; tries < 10000 && !accepted; ++tries)
    {
        drawCandidate();
        if(degenerate)
        {
            accepted = true;
            break;
        }
        double const mu2 = mu * mu;
        accepted = 2.0 * u01() * (1.0 + mu2) <= (1.0 + mu2) + (1.0 - mu2) * qn;
    }
    if(!accepted)
    {
        drawCandidate();
    }

    PointT const kOut = unitOr(k * mu + e1 * (sinTheta * cosPhi) + e2 * (sinTheta * sinPhi), k);
    p.velocity = kOut * speed;
    p.polarizationInitialized = true;
    if(degenerate)
    {
        // Forward or backward: no scattering plane, the Stokes state carries over.
        p.polarizationBasis = e;
        p.stokesQ = q;
        p.stokesU = u;
        return;
    }
    double const mu2 = mu * mu;
    double const Iprime = (1.0 + mu2) + (1.0 - mu2) * qn;
    double const Qprime = (1.0 - mu2) + (1.0 + mu2) * qn;
    double const Uprime = 2.0 * mu * un;
    if(std::abs(Iprime) <= kEpsilon || !std::isfinite(Iprime))
    {
        p.stokesQ = 0.0;
        p.stokesU = 0.0;
    }
    else
    {
        double const inverseI = 1.0 / Iprime;
        p.stokesQ = Qprime * inverseI;
        p.stokesU = Uprime * inverseI;
    }
    // The scattering-plane normal: unit and perpendicular to kOut by
    // construction.
    p.polarizationBasis = e2 * cosPhi - e1 * sinPhi;
    clampLinearPolarization(p.stokesQ, p.stokesU);
}

template<typename PointT, typename Uniform01>
PointT sampleIsotropicVelocity(const PointT &referenceVelocity, Uniform01 &&u01)
{
    double speed = fallback::abs(referenceVelocity);
    if(!(speed > kEpsilon) || !std::isfinite(speed))
    {
        speed = units::clight;
    }
    double const mu = 2.0 * u01() - 1.0;
    double const phi = 2.0 * kPi * u01();
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return PointT(sinTheta * std::cos(phi), sinTheta * std::sin(phi), mu) * speed;
}

inline double saturatingProduct(double a, double b)
{
    double const maxValue = std::numeric_limits<double>::max();
    if(std::isnan(a) || std::isnan(b)) return maxValue;
    if(a <= 0.0 || b <= 0.0) return 0.0;
    if(!std::isfinite(a) || !std::isfinite(b) || a > maxValue / b) return maxValue;
    return a * b;
}

inline double saturatingAdd(double a, double b)
{
    double const maxValue = std::numeric_limits<double>::max();
    if(std::isnan(a) || std::isnan(b)) return maxValue;
    a = a < 0.0 ? 0.0 : (!std::isfinite(a) ? maxValue : a);
    b = b < 0.0 ? 0.0 : (!std::isfinite(b) ? maxValue : b);
    return a > maxValue - b ? maxValue : a + b;
}

template<typename Uniform01>
double sampleAgeSinceLastReset(double resetRate, double dt, Uniform01 &&u01,
                               bool &resetOccurred)
{
    resetOccurred = false;
    if(!(resetRate > 0.0) || !(dt > 0.0) ||
       !std::isfinite(resetRate) || !std::isfinite(dt)) return dt;
    double const pReset = -std::expm1(-resetRate * dt);
    if(u01() >= pReset) return dt;
    resetOccurred = true;
    double const xi = std::clamp(u01(), std::numeric_limits<double>::min(),
                                 1.0 - std::numeric_limits<double>::epsilon());
    return std::min(-std::log1p(-xi * pReset) / resetRate, dt);
}

template<typename PointT, typename ParticleT>
void clearAcceleratedHistory(ParticleT &p)
{
    p.radiationState.pendingMeanScatterings = 0.0;
}

template<typename PointT, typename ParticleT, typename Uniform01>
void applyManualSyntheticScatterings(ParticleT &p, int nManual,
                                      const PointT &finalVelocity, Uniform01 &&u01)
{
    if(nManual <= 0)
    {
        p.velocity = finalVelocity;
        p.polarizationBasis = projectBasisToDirection(p.polarizationBasis, finalVelocity);
        return;
    }
    for(int i = 0; i < nManual - 1; ++i)
    {
        scatterPolarizedThomson<PointT>(p, u01);
    }
    // The last scattering is forced onto the exit direction.
    PointT const current = p.velocity;
    applyThomsonScatter<PointT>(p, current, finalVelocity);
}

template<typename PointT, typename ParticleT, typename RandomEngine, typename UniformDist>
void accumulateAcceleratedPolarizationHistory(ParticleT &p, double dtCo,
                                               double sigmaScattering,
                                               double sigmaUnresolvedReset,
                                               RandomEngine &engine,
                                               UniformDist &uniformDist)
{
    initializeIfNeeded<PointT>(p);
    if(!(dtCo > 0.0) || !std::isfinite(dtCo)) return;
    auto u01 = [&]() {
        return std::clamp(uniformDist(engine), std::numeric_limits<double>::min(),
                          1.0 - std::numeric_limits<double>::epsilon());
    };
    double lightSpeed = fallback::abs(p.velocity);
    if(!(lightSpeed > kEpsilon) || !std::isfinite(lightSpeed))
    {
        lightSpeed = units::clight;
    }
    double const resetRate = saturatingProduct(lightSpeed, sigmaUnresolvedReset);
    double const scatterRate = saturatingProduct(lightSpeed, sigmaScattering);
    bool resetOccurred = false;
    double const age = sampleAgeSinceLastReset(resetRate, dtCo, u01, resetOccurred);
    double const intervalMean = saturatingProduct(scatterRate, resetOccurred ? age : dtCo);
    if(resetOccurred)
    {
        p.velocity = sampleIsotropicVelocity<PointT>(p.velocity, u01);
        resetUnpolarized<PointT>(p);
        p.radiationState.pendingMeanScatterings = intervalMean;
    }
    else
    {
        p.radiationState.pendingMeanScatterings = saturatingAdd(
            p.radiationState.pendingMeanScatterings, intervalMean);
    }
}

template<typename PointT, typename ParticleT, typename RandomEngine, typename UniformDist>
void finalizeAcceleratedPolarizationHistory(ParticleT &p, const PointT &finalVelocityCo,
                                             int manualScatteringsAfterAcceleration,
                                             double depolarizationScatterings,
                                             RandomEngine &engine, UniformDist &uniformDist)
{
    initializeIfNeeded<PointT>(p);
    auto u01 = [&]() {
        return std::clamp(uniformDist(engine), std::numeric_limits<double>::min(),
                          1.0 - std::numeric_limits<double>::epsilon());
    };
    int const K = std::max(0, manualScatteringsAfterAcceleration);
    double mean = p.radiationState.pendingMeanScatterings;
    if(std::isnan(mean)) mean = std::numeric_limits<double>::max();
    else if(!(mean > 0.0)) mean = 0.0;
    else if(!std::isfinite(mean)) mean = std::numeric_limits<double>::max();
    clearAcceleratedHistory<PointT>(p);

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
        p.velocity = sampleIsotropicVelocity<PointT>(p.velocity, u01);
        p.polarizationBasis = projectBasisToDirection(p.polarizationBasis, p.velocity);
    }
    p.stokesQ *= damping;
    p.stokesU *= damping;
    clampLinearPolarization(p.stokesQ, p.stokesU);
    applyManualSyntheticScatterings<PointT>(p, static_cast<int>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(K), n)), finalVelocityCo, u01);
    p.velocity = finalVelocityCo;
    p.polarizationBasis = projectBasisToDirection(p.polarizationBasis, p.velocity);
    clampLinearPolarization(p.stokesQ, p.stokesU);
}

template<typename PointT, typename ParticleT, typename RandomEngine, typename UniformDist>
void applyAcceleratedPolarizationHistory(ParticleT &p, double dtCo,
                                          double sigmaScattering,
                                          double sigmaEffectiveReset,
                                          const PointT &finalVelocityCo,
                                          int manualScatteringsAfterAcceleration,
                                          double depolarizationScatterings,
                                          RandomEngine &engine, UniformDist &uniformDist)
{
    accumulateAcceleratedPolarizationHistory<PointT>(p, dtCo, sigmaScattering,
        sigmaEffectiveReset, engine, uniformDist);
    finalizeAcceleratedPolarizationHistory<PointT>(p, finalVelocityCo,
        manualScatteringsAfterAcceleration, depolarizationScatterings,
        engine, uniformDist);
}

#endif // MONTECARLO_POLARIZATION

} // namespace STORM::polarization

#endif // STORM_RADIATION_POLARIZATION_HPP
