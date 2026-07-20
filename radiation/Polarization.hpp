#ifndef STORM_RADIATION_POLARIZATION_HPP
#define STORM_RADIATION_POLARIZATION_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

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
    return safeNormalize(projected, choosePerpendicularBasis(k));
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

template<typename PointT>
void buildBasisAroundDirection(const PointT &k, PointT &e1, PointT &e2)
{
    PointT const kk = safeNormalize(k, PointT(0.0, 0.0, 1.0));
    e1 = choosePerpendicularBasis(kk);
    e2 = safeNormalize(fallback::CrossProduct(kk, e1),
                       PointT(0.0, 1.0, 0.0));
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

template<typename PointT, typename Uniform01>
PointT sampleSyntheticThomsonDirection(const PointT &oldVelocity, Uniform01 &&u01)
{
    PointT const k = safeNormalize(oldVelocity, PointT(0.0, 0.0, 1.0));
    double const speed = std::max(fallback::abs(oldVelocity), kEpsilon);
    PointT e1, e2;
    buildBasisAroundDirection(k, e1, e2);
    double const mu = sampleThomsonMu(u01);
    double const phi = 2.0 * kPi * u01();
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    PointT const dir = k * mu + e1 * (sinTheta * std::cos(phi)) +
                       e2 * (sinTheta * std::sin(phi));
    return safeNormalize(dir, k) * speed;
}

template<typename PointT, typename ParticleT>
std::pair<double, double> projectToBasis(const ParticleT &p,
                                         const PointT &direction,
                                         const PointT &basis)
{
    if(!p.polarizationInitialized)
    {
        return {0.0, 0.0};
    }
    PointT const k = safeNormalize(direction, PointT(0.0, 0.0, 1.0));
    PointT const eOld = projectBasisToDirection(p.polarizationBasis, k);
    PointT const eNew = projectBasisToDirection(basis, k);
    double const c = std::clamp(fallback::ScalarProd(eOld, eNew), -1.0, 1.0);
    double const s = fallback::ScalarProd(k, fallback::CrossProduct(eOld, eNew));
    double const cos2 = c * c - s * s;
    double const sin2 = 2.0 * c * s;
    double q = p.stokesQ * cos2 + p.stokesU * sin2;
    double u = -p.stokesQ * sin2 + p.stokesU * cos2;
    clampLinearPolarization(q, u);
    return {q, u};
}

template<typename PointT, typename ParticleT, typename Uniform01>
PointT samplePolarizedThomsonDirection(const ParticleT &p,
                                       const PointT &oldVelocity,
                                       Uniform01 &&u01)
{
    PointT const kIn = safeNormalize(oldVelocity, PointT(0.0, 0.0, 1.0));
    for(int tries = 0; tries < 10000; ++tries)
    {
        PointT const candidate = sampleSyntheticThomsonDirection(oldVelocity, u01);
        PointT const kOut = safeNormalize(candidate, kIn);
        PointT planeNormal = fallback::CrossProduct(kIn, kOut);
        if(fallback::abs(planeNormal) <= 1.0e-10)
        {
            return candidate;
        }
        planeNormal = safeNormalize(planeNormal, PointT(0.0, 0.0, 1.0));
        auto const qu = projectToBasis(p, oldVelocity, planeNormal);
        double const mu = std::clamp(fallback::ScalarProd(kIn, kOut), -1.0, 1.0);
        double const mu2 = mu * mu;
        double const ratio = 1.0 + (1.0 - mu2) / (1.0 + mu2) * qu.first;
        if(u01() <= std::clamp(0.5 * ratio, 0.0, 1.0))
        {
            return candidate;
        }
    }
    return sampleSyntheticThomsonDirection(oldVelocity, u01);
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
    PointT current = p.velocity;
    for(int i = 0; i < nManual; ++i)
    {
        PointT const next = i == nManual - 1
            ? finalVelocity : samplePolarizedThomsonDirection(p, current, u01);
        applyThomsonScatter<PointT>(p, current, next);
        current = next;
    }
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
    double const resetRate = saturatingProduct(units::clight, sigmaUnresolvedReset);
    double const scatterRate = saturatingProduct(units::clight, sigmaScattering);
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
