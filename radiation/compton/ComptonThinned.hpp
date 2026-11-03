#ifndef STORM_RADIATION_COMPTON_THINNED_HPP
#define STORM_RADIATION_COMPTON_THINNED_HPP

// Pure arithmetic of the Thinned Compton event mode (plan v10,
// runs/imc_compton_paper/compton_fix_plan_2026-10-30.md): grid checks, re-entry
// of out-of-grid packets, the post-jump interval check, the gas-noise predictor,
// the photon-neutral settlement of the energy ledger and the projection onto
// fixed energy and photon number.  Nothing here touches packets or cells, so
// every piece can be tested on its own.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace STORM
{
namespace compton_thinned
{

// Per-step counters of the Thinned mode (rank-local; reset in preStep).
struct Diagnostics
{
    std::size_t events = 0;
    std::size_t jumps = 0;
    // Events per source group (cost diagnostics): out-of-group and in-group.
    std::vector<double> eventsByGroup;
    std::vector<double> inGroupEventsByGroup;
    std::size_t reentryLower = 0;
    std::size_t reentryUpper = 0;
    // Lab energy the re-entries handed to the gas (frequency-boundary work).
    double boundaryWorkLower = 0.0;
    double boundaryWorkUpper = 0.0;
    std::size_t nudges = 0;
    double nudgeEnergy = 0.0;
    std::size_t gasFirstChoices = 0;
    std::size_t radiationFirstChoices = 0;
    std::size_t unqualifiedStencils = 0;
    // Index = settlement tier 1..4 (0 unused).
    std::array<std::size_t, 5> tiers{};
    double ledgerAbsSum = 0.0;
    double predictedSigmaSum = 0.0;
    double maxStencilAmplification = 0.0;
    // Photon change of the planned stencils with end-of-step coefficients.
    double settlementPhotonChange = 0.0;
    // Changes the (E, N) projection made to the correction endpoint.
    double projectionEnergyRepair = 0.0;
    double projectionPhotonRepair = 0.0;
    // Photon targets outside [E min h, E max h] (not representable at the
    // accepted energy on the grid) are clamped to the nearest representable
    // count; number of cells and summed |change|.
    std::size_t photonClamps = 0;
    double photonClampDefect = 0.0;
    // Photon projection fallbacks: Euclidean on the groups that hold energy, and Euclidean on all groups.
    std::size_t projectionSupportFallbacks = 0;
    std::size_t projectionFullFallbacks = 0;
    std::size_t limiterActive = 0;
    std::size_t limiterInfeasible = 0;
    std::size_t correctionBacktracks = 0;
    double correctionScaleMin = 1.0;
    std::size_t reconcileUpscales = 0;
    std::size_t reconcileSpawns = 0;
    // Actual census photons: before settlement (post-comb) and after reconciliation.
    double censusPhotonsRaw = 0.0;
    double censusPhotonsFinal = 0.0;
    double closureEnergy = 0.0;
};

// Log-uniform group grid with arithmetic group centres.  A ratio-preserving
// jump nu -> nu eps_t / eps_s maps group s onto group t because nu / eps_g has
// the same range for every group.  xiLower and xiUpper bound that common
// normalized interval, shrunk by the relative margin.
struct Grid
{
    bool valid = false;
    double rho = 0.0;
    double xiLower = 0.0;
    double xiUpper = 0.0;
    double margin = 0.0;
    // Which check failed when valid is false.
    const char *reason = "";
};

template<std::size_t G>
Grid BuildGrid(const std::array<double, G + 1> &boundaries, const std::array<double, G> &centres, double tolerance,
               double margin)
{
    Grid grid;
    grid.margin = margin;
    if(G == 0)
    {
        grid.reason = "no groups";
        return grid;
    }
    for(std::size_t g = 0; g <= G; ++g)
    {
        if(!(boundaries[g] > 0.0) || !std::isfinite(boundaries[g]))
        {
            grid.reason = "group boundaries must be positive and finite";
            return grid;
        }
        if(g > 0 && !(boundaries[g] > boundaries[g - 1]))
        {
            grid.reason = "group boundaries must increase";
            return grid;
        }
    }
    grid.rho = boundaries[1] / boundaries[0];
    for(std::size_t g = 0; g < G; ++g)
    {
        double const ratio = boundaries[g + 1] / boundaries[g];
        if(std::abs(ratio - grid.rho) > tolerance * grid.rho)
        {
            grid.reason = "group boundaries are not log-uniform";
            return grid;
        }
        if(!(centres[g] > boundaries[g]) || !(centres[g] < boundaries[g + 1]))
        {
            grid.reason = "group centre outside its group";
            return grid;
        }
        if(g > 0 && std::abs(centres[g] / centres[g - 1] - grid.rho) > tolerance * grid.rho)
        {
            grid.reason = "group centres are not log-uniform";
            return grid;
        }
    }
    double lower = 0.0;
    double upper = std::numeric_limits<double>::infinity();
    for(std::size_t g = 0; g < G; ++g)
    {
        lower = std::max(lower, boundaries[g] / centres[g]);
        upper = std::min(upper, boundaries[g + 1] / centres[g]);
    }
    grid.xiLower = lower * (1.0 + margin);
    grid.xiUpper = upper * (1.0 - margin);
    if(!(grid.xiLower < grid.xiUpper))
    {
        grid.reason = "the common normalized interval is empty after the margin";
        return grid;
    }
    for(std::size_t g = 0; g < G; ++g)
    {
        if(!(boundaries[g] * (1.0 + margin) < boundaries[g + 1] * (1.0 - margin)))
        {
            grid.reason = "a group is narrower than the nudge margin";
            return grid;
        }
    }
    grid.valid = true;
    return grid;
}

// Factor F = nu_e / nu that moves a comoving frequency outside [B_0, B_G) to an
// interior point next to the near edge; 1 inside the grid.  Applying F to the
// frequency and to the weight keeps the photon number w / nu.
inline double ReentryFactor(double frequency, double lowestBoundary, double highestBoundary, double lowestCentre,
                            double highestCentre, const Grid &grid)
{
    if(frequency >= lowestBoundary && frequency < highestBoundary)
    {
        return 1.0;
    }
    if(!(frequency > 0.0) || !std::isfinite(frequency))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double const target = frequency < lowestBoundary ? lowestCentre * grid.xiLower : highestCentre * grid.xiUpper;
    return target / frequency;
}

// Post-jump check: a frequency already in [lower, upper) is returned unchanged;
// a rounding-level miss is moved to the nearest point of the group shrunk by
// the margin.  The caller scales the weight by the returned/old ratio.
inline double NudgeIntoGroup(double frequency, double lower, double upper, double margin)
{
    if(frequency >= lower && frequency < upper)
    {
        return frequency;
    }
    return std::clamp(frequency, lower * (1.0 + margin), upper * (1.0 - margin));
}

// Variance factor of one out-of-group event of a unit-weight packet in group s:
// Var(x | w) = w^2 [f V2_s - f^2 (mu_s - 1)^2].
inline double EventVarianceFactor(double fleck, double mu, double secondMoment)
{
    return std::max(0.0, fleck * secondMoment - fleck * fleck * (mu - 1.0) * (mu - 1.0));
}

// Predicted variance of the per-cell energy ledger over one step:
// sum_s c dt q_s M2_s [p_s V2_s - p_s^2 (mu_s - 1)^2], M2_s in erg^2, with q_s
// the sampled event rate and p_s the jump probability per event (f without
// the throttle).
template<std::size_t G>
double PredictLedgerVariance(double cdt, const std::array<double, G> &jumpProbability,
                             const std::array<double, G> &outRate, const std::array<double, G> &mu,
                             const std::array<double, G> &secondMoment, const std::array<double, G> &weightSquared)
{
    double variance = 0.0;
    for(std::size_t s = 0; s < G; ++s)
    {
        variance += cdt * outRate[s] * weightSquared[s] *
            EventVarianceFactor(jumpProbability[s], mu[s], secondMoment[s]);
    }
    return variance;
}

// Optional non-jump throttle (plan 1a, F6-2): events are sampled at
// q_MC = min(q, max(f q, Q / (c dt))) and jump with probability f q / q_MC, so
// the jump rate f q, hence H = f S, is unchanged while a packet does at most
// about max(Q, f q c dt) events per step.  Q <= 0 disables it.
struct ThrottledRate
{
    double rate = 0.0;
    double jumpProbability = 0.0;
};

inline ThrottledRate ThrottleEvents(double outRate, double fleck, double cdt, double eventsPerStep)
{
    ThrottledRate result{outRate, fleck};
    if(!(eventsPerStep > 0.0) || !(outRate > 0.0) || !(cdt > 0.0))
    {
        return result;
    }
    double const sampled = std::min(outRate, std::max(fleck * outRate, eventsPerStep / cdt));
    if(sampled > 0.0 && sampled < outRate)
    {
        result.rate = sampled;
        result.jumpProbability = std::min(1.0, fleck * outRate / sampled);
    }
    return result;
}

// Two-group photon-neutral map that removes an energy X from the radiation:
// E_b -= X cb, E_a -= X ca, cb = h_a / (h_a - h_b), ca = -h_b / (h_a - h_b),
// so the removed energy is X and the photon change -(ca h_a + cb h_b) X is zero.
struct Stencil
{
    bool found = false;
    // True when the pair meets the ratio and margin requirements of tier (i).
    bool qualified = false;
    std::size_t a = 0;
    std::size_t b = 0;
    double ca = 0.0;
    double cb = 0.0;
};

inline Stencil MakeStencil(std::size_t a, std::size_t b, double ha, double hb)
{
    Stencil stencil;
    if(a == b || !(ha > hb) || !(hb > 0.0) || !std::isfinite(ha))
    {
        return stencil;
    }
    stencil.found = true;
    stencil.a = a;
    stencil.b = b;
    stencil.cb = ha / (ha - hb);
    stencil.ca = -hb / (ha - hb);
    return stencil;
}

// Amplification |A| = |ca| + |cb| = (h_a + h_b) / (h_a - h_b) of a stencil.
inline double StencilAmplification(const Stencil &stencil)
{
    return std::abs(stencil.ca) + std::abs(stencil.cb);
}

// Tier (i) stencil of the reservoir plan (section 2b): b = the largest group,
// a = the lowest group with h_a / h_b >= minimumRatio holding energy and the
// margin E_a >= K sigma h_b / (h_a - h_b).  Without such a group the pair
// maximizing min(E_a (h_a - h_b) / h_b, E_b (h_a - h_b) / h_a) is returned with
// qualified = false.
template<std::size_t G>
Stencil SelectPlannedStencil(const std::array<double, G> &energy, const std::array<double, G> &h, double sigma,
                             double marginFactor, double minimumRatio)
{
    Stencil none;
    std::size_t b = G;
    double largest = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        if(energy[g] > largest)
        {
            largest = energy[g];
            b = g;
        }
    }
    if(b == G || !(h[b] > 0.0))
    {
        return none;
    }
    for(std::size_t a = 0; a < G; ++a)
    {
        if(a == b || !(h[a] >= minimumRatio * h[b]) || !(energy[a] > 0.0))
        {
            continue;
        }
        if(energy[a] >= marginFactor * sigma * h[b] / (h[a] - h[b]))
        {
            Stencil stencil = MakeStencil(a, b, h[a], h[b]);
            stencil.qualified = stencil.found;
            return stencil;
        }
    }
    Stencil best;
    double bestMargin = -1.0;
    for(std::size_t a = 0; a < G; ++a)
    {
        if(a == b || !(h[a] > h[b]) || !(energy[a] > 0.0))
        {
            continue;
        }
        double const margin = std::min(energy[a] * (h[a] - h[b]) / h[b], energy[b] * (h[a] - h[b]) / h[a]);
        if(margin > bestMargin)
        {
            bestMargin = margin;
            best = MakeStencil(a, b, h[a], h[b]);
        }
    }
    return best;
}

// Energy removed from group a and b by a stencil for a ledger X keeps both >= 0.
template<std::size_t G>
bool StencilFeasible(const Stencil &stencil, double ledger, const std::array<double, G> &energy)
{
    return stencil.found && energy[stencil.a] - ledger * stencil.ca >= 0.0 &&
           energy[stencil.b] - ledger * stencil.cb >= 0.0;
}

enum class Reservoir
{
    Gas,
    Radiation
};

// Outcome of the settlement cascade of section 2c.
struct Settlement
{
    // 1: planned first choice, 2: the other reservoir, 3: any photon-neutral
    // pair with end-of-step coefficients, 4: nothing could pay (failure).
    int tier = 4;
    Reservoir payer = Reservoir::Gas;
    Stencil stencil;
    double gasChange = 0.0;
};

// Chooses who pays the ledger X (energy the packets carry beyond the expected
// exchange).  The gas pays if U - X stays at or above its floor.  The planned
// stencil uses the pre-step coefficients (fixed before the step, zero-mean
// spectral change); fallback stencils use the end-of-step coefficients, which
// makes them photon-exact.  Energy is exact in every tier.
template<std::size_t G>
Settlement Settle(double ledger, Reservoir planned, const Stencil &plannedStencil, double gasEnergy,
                  double gasFloor, bool gasAvailable, const std::array<double, G> &energy,
                  const std::array<double, G> &hEnd)
{
    Settlement settlement;
    auto gasCanPay = [&]()
    {
        return gasAvailable && gasEnergy - ledger >= gasFloor;
    };
    auto endStencil = [&](const Stencil &pattern)
    {
        return pattern.found ? MakeStencil(pattern.a, pattern.b, hEnd[pattern.a], hEnd[pattern.b]) : Stencil{};
    };
    if(ledger == 0.0)
    {
        settlement.tier = 1;
        settlement.payer = planned;
        return settlement;
    }
    if(planned == Reservoir::Gas)
    {
        if(gasCanPay())
        {
            settlement.tier = 1;
            settlement.payer = Reservoir::Gas;
            settlement.gasChange = -ledger;
            return settlement;
        }
        Stencil const fallback = endStencil(plannedStencil);
        if(StencilFeasible(fallback, ledger, energy))
        {
            settlement.tier = 2;
            settlement.payer = Reservoir::Radiation;
            settlement.stencil = fallback;
            return settlement;
        }
    }
    else
    {
        if(StencilFeasible(plannedStencil, ledger, energy))
        {
            settlement.tier = 1;
            settlement.payer = Reservoir::Radiation;
            settlement.stencil = plannedStencil;
            return settlement;
        }
        if(gasCanPay())
        {
            settlement.tier = 2;
            settlement.payer = Reservoir::Gas;
            settlement.gasChange = -ledger;
            return settlement;
        }
    }
    // Tier (iii): any pair, the highest-energy groups first.
    std::array<std::size_t, G> order{};
    for(std::size_t g = 0; g < G; ++g)
    {
        order[g] = g;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
    {
        return energy[left] > energy[right];
    });
    for(std::size_t i = 0; i < G; ++i)
    {
        for(std::size_t j = i + 1; j < G; ++j)
        {
            std::size_t const first = order[i];
            std::size_t const second = order[j];
            if(!(energy[first] > 0.0) && !(energy[second] > 0.0))
            {
                continue;
            }
            std::size_t const a = hEnd[first] > hEnd[second] ? first : second;
            std::size_t const b = a == first ? second : first;
            Stencil const pair = MakeStencil(a, b, hEnd[a], hEnd[b]);
            if(StencilFeasible(pair, ledger, energy))
            {
                settlement.tier = 3;
                settlement.payer = Reservoir::Radiation;
                settlement.stencil = pair;
                return settlement;
            }
        }
    }
    settlement.tier = 4;
    return settlement;
}

// Applies a radiation settlement to the group energies.
template<std::size_t G>
void ApplyStencil(const Stencil &stencil, double ledger, std::array<double, G> &energy)
{
    energy[stencil.a] -= ledger * stencil.ca;
    energy[stencil.b] -= ledger * stencil.cb;
}

// Euclidean projection of y onto {x >= 0, sum x = E, sum h x = N}:
// x_g = max(0, y_g - lambda - kappa h_g).
template<std::size_t G>
struct Projection
{
    bool success = false;
    std::array<double, G> endpoint{};
    double lambda = 0.0;
    double kappa = 0.0;
    double energyError = 0.0;
    double photonError = 0.0;
};

namespace detail
{

// lambda with sum max(0, z_g - lambda) = target (target > 0).
template<std::size_t G>
double SimplexShift(const std::array<double, G> &z, double target)
{
    std::array<double, G> sorted = z;
    std::sort(sorted.begin(), sorted.end(), [](double left, double right)
    {
        return left > right;
    });
    double cumulative = 0.0;
    double shift = sorted[0] - target;
    for(std::size_t k = 0; k < G; ++k)
    {
        cumulative += sorted[k];
        double const candidate = (cumulative - target) / static_cast<double>(k + 1);
        if(k + 1 == G || sorted[k + 1] <= candidate)
        {
            shift = candidate;
            break;
        }
    }
    return shift;
}

template<std::size_t G>
void Evaluate(const std::array<double, G> &y, const std::array<double, G> &h, double energyTarget, double kappa,
              double &lambda, double &photons, std::array<double, G> &x, const std::array<bool, G> *support = nullptr)
{
    std::array<double, G> z{};
    for(std::size_t g = 0; g < G; ++g)
    {
        // Outside the support z = -inf: such a group never becomes positive.
        z[g] = (!support || (*support)[g]) ? y[g] - kappa * h[g] : -std::numeric_limits<double>::infinity();
    }
    lambda = SimplexShift(z, energyTarget);
    photons = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        x[g] = std::max(0.0, z[g] - lambda);
        photons += h[g] * x[g];
    }
}

} // namespace detail

// Requires E > 0 and h > 0.  The photon number of the energy-only projection
// decreases monotonically with kappa, so kappa is found by bracketing and
// bisection; the active set of the bisected solution then gives lambda and
// kappa from a 2x2 system, which meets both targets to roundoff.  Fails when N
// lies outside [E min h, E max h].
template<std::size_t G>
Projection<G> ProjectEnergyPhoton(const std::array<double, G> &y, const std::array<double, G> &h, double energyTarget,
                                  double photonTarget, const std::array<bool, G> *support = nullptr)
{
    Projection<G> result;
    if(!(energyTarget > 0.0) || !(photonTarget > 0.0) || !std::isfinite(energyTarget) ||
       !std::isfinite(photonTarget))
    {
        return result;
    }
    double hMin = std::numeric_limits<double>::infinity();
    double hMax = 0.0;
    double yScale = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        if(!(h[g] > 0.0) || !std::isfinite(h[g]) || !std::isfinite(y[g]))
        {
            return result;
        }
        if(!support || (*support)[g])
        {
            hMin = std::min(hMin, h[g]);
            hMax = std::max(hMax, h[g]);
            yScale = std::max(yScale, std::abs(y[g]));
        }
    }
    if(!(hMax > 0.0) || photonTarget < energyTarget * hMin * (1.0 - 1e-14) ||
       photonTarget > energyTarget * hMax * (1.0 + 1e-14))
    {
        return result;
    }
    std::array<double, G> x{};
    double lambda = 0.0;
    double photons = 0.0;
    auto photonsAt = [&](double kappa)
    {
        detail::Evaluate(y, h, energyTarget, kappa, lambda, photons, x, support);
        return photons;
    };
    // Bracket: large |kappa| pushes all energy to the group with the smallest
    // (kappa > 0) or largest (kappa < 0) h.  Start where kappa h_max is of the
    // size of y: h spans the grid (1/eps, up to ~1e13 per erg), and a larger
    // start would cancel y against kappa h in the active groups.
    double const kappaScale = (yScale + energyTarget) / hMax;
    double low = -kappaScale;
    double high = kappaScale;
    for(int expand = 0; expand < 200 && photonsAt(low) < photonTarget; ++expand)
    {
        low *= 2.0;
    }
    for(int expand = 0; expand < 200 && photonsAt(high) > photonTarget; ++expand)
    {
        high *= 2.0;
    }
    double kappa = 0.0;
    for(int iteration = 0; iteration < 200; ++iteration)
    {
        kappa = 0.5 * (low + high);
        if(photonsAt(kappa) > photonTarget)
        {
            low = kappa;
        }
        else
        {
            high = kappa;
        }
        if(high - low <= 1e-15 * std::max(std::abs(low), std::abs(high)))
        {
            break;
        }
    }
    photonsAt(kappa);
    // Polish on the active set: sum (y - lambda - kappa h) = E and
    // sum h (y - lambda - kappa h) = N over the groups with x > 0.
    double count = 0.0;
    double sumH = 0.0;
    double sumH2 = 0.0;
    double sumY = 0.0;
    double sumHY = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        if(x[g] > 0.0)
        {
            count += 1.0;
            sumH += h[g];
            sumH2 += h[g] * h[g];
            sumY += y[g];
            sumHY += h[g] * y[g];
        }
    }
    double const determinant = count * sumH2 - sumH * sumH;
    if(count >= 2.0 && std::abs(determinant) > 1e-12 * sumH * sumH)
    {
        double const rhsEnergy = sumY - energyTarget;
        double const rhsPhoton = sumHY - photonTarget;
        double const polishedLambda = (rhsEnergy * sumH2 - sumH * rhsPhoton) / determinant;
        double const polishedKappa = (count * rhsPhoton - sumH * rhsEnergy) / determinant;
        bool consistent = true;
        std::array<double, G> polished{};
        for(std::size_t g = 0; g < G; ++g)
        {
            double const value = y[g] - polishedLambda - polishedKappa * h[g];
            if(x[g] > 0.0)
            {
                polished[g] = std::max(0.0, value);
                consistent = consistent && value >= -1e-12 * energyTarget;
            }
            else
            {
                polished[g] = 0.0;
                // Groups outside the support stay at zero whatever their value.
                consistent = consistent && (value <= 1e-12 * energyTarget || (support && !(*support)[g]));
            }
        }
        if(consistent)
        {
            x = polished;
            lambda = polishedLambda;
            kappa = polishedKappa;
        }
    }
    double energy = 0.0;
    photons = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        energy += x[g];
        photons += h[g] * x[g];
    }
    result.endpoint = x;
    result.lambda = lambda;
    result.kappa = kappa;
    result.energyError = (energy - energyTarget) / energyTarget;
    result.photonError = (photons - photonTarget) / photonTarget;
    result.success = std::abs(result.energyError) <= 1e-10 && std::abs(result.photonError) <= 1e-10;
    return result;
}

// Relative projection onto energy E and photons N: x_g = y_g (1 - lambda -
// kappa h_g) on the groups that hold energy (x = 0 where the factor would be
// negative), from the 2x2 system on that active set.  Unlike
// ProjectEnergyPhoton it never puts energy into an empty group and changes
// each group in proportion to its content.  Fails (success = false) when the
// active set cannot meet both targets.
template<std::size_t G>
Projection<G> ProjectEnergyPhotonRelative(const std::array<double, G> &y, const std::array<double, G> &h,
                                          double energyTarget, double photonTarget)
{
    Projection<G> result;
    if(!(energyTarget > 0.0) || !(photonTarget > 0.0) || !std::isfinite(energyTarget) ||
       !std::isfinite(photonTarget))
    {
        return result;
    }
    std::array<bool, G> active{};
    for(std::size_t g = 0; g < G; ++g)
    {
        if(!(h[g] > 0.0) || !std::isfinite(h[g]) || !std::isfinite(y[g]))
        {
            return result;
        }
        active[g] = y[g] > 0.0;
    }
    std::array<double, G> x{};
    for(int pass = 0; pass < static_cast<int>(G); ++pass)
    {
        // Normalised h (its y-weighted mean) keeps the 2x2 system conditioned.
        double sumY = 0.0;
        double sumYH = 0.0;
        for(std::size_t g = 0; g < G; ++g)
        {
            if(active[g])
            {
                sumY += y[g];
                sumYH += y[g] * h[g];
            }
        }
        if(!(sumY > 0.0) || !(sumYH > 0.0))
        {
            return result;
        }
        double const hMean = sumYH / sumY;
        double a11 = 0.0;
        double a12 = 0.0;
        double a22 = 0.0;
        for(std::size_t g = 0; g < G; ++g)
        {
            if(active[g])
            {
                double const hn = h[g] / hMean;
                a11 += y[g];
                a12 += y[g] * hn;
                a22 += y[g] * hn * hn;
            }
        }
        double const determinant = a11 * a22 - a12 * a12;
        if(!(determinant > 1e-14 * a11 * a22))
        {
            return result;
        }
        // sum y (1 - lambda - kappa' hn) = E and sum y hn (1 - lambda - kappa' hn) = N / hMean.
        double const rhsEnergy = a11 - energyTarget;
        double const rhsPhoton = a12 - photonTarget / hMean;
        double const lambda = (rhsEnergy * a22 - a12 * rhsPhoton) / determinant;
        double const kappaNormalised = (a11 * rhsPhoton - a12 * rhsEnergy) / determinant;
        bool negative = false;
        for(std::size_t g = 0; g < G; ++g)
        {
            double const factor = 1.0 - lambda - kappaNormalised * h[g] / hMean;
            x[g] = active[g] ? y[g] * factor : 0.0;
            if(active[g] && factor < 0.0)
            {
                active[g] = false;
                negative = true;
            }
        }
        result.lambda = lambda;
        result.kappa = kappaNormalised / hMean;
        if(!negative)
        {
            break;
        }
    }
    double energy = 0.0;
    double photons = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        x[g] = std::max(0.0, x[g]);
        energy += x[g];
        photons += h[g] * x[g];
    }
    result.endpoint = x;
    result.energyError = (energy - energyTarget) / energyTarget;
    result.photonError = (photons - photonTarget) / photonTarget;
    result.success = std::abs(result.energyError) <= 1e-10 && std::abs(result.photonError) <= 1e-10;
    return result;
}

// Limiter of plan 3c: v'_g = clip(v_g - s_g (lambda + kappa w_g), lower_g, upper_g)
// keeping sum v and sum w v, with group scales s (nullptr: s = 1, the
// Euclidean projection).  Scales proportional to the group's size keep the
// adjustment relative, so a nearly empty group is not pushed to its bound by a
// shift sized for the dominant groups; a group with s_g = 0 is only clipped.
// Returns false (values unchanged) when no such clip exists; values already
// inside the bounds are returned unchanged.
template<std::size_t G>
bool ClipKeepingMoments(std::array<double, G> &values, const std::array<double, G> &w,
                        const std::array<double, G> &lower, const std::array<double, G> &upper,
                        const std::array<double, G> *scales = nullptr)
{
    // Scales below 1e-14 of the largest are frozen (only clipped): dividing by them would overflow.
    double scaleMax = 0.0;
    if(scales)
    {
        for(double const value : *scales)
        {
            if(!(value >= 0.0) || !std::isfinite(value))
            {
                return false;
            }
            scaleMax = std::max(scaleMax, value);
        }
    }
    auto weight = [&](std::size_t g)
    {
        if(!scales)
        {
            return 1.0;
        }
        return (*scales)[g] > 1e-14 * scaleMax ? (*scales)[g] : 0.0;
    };
    bool inside = true;
    double sum = 0.0;
    double moment = 0.0;
    double lowerSum = 0.0;
    double upperSum = 0.0;
    double scale = 0.0;
    double wMax = 0.0;
    bool movable = false;
    for(std::size_t g = 0; g < G; ++g)
    {
        if(!(lower[g] <= upper[g]) || !std::isfinite(values[g]) || !std::isfinite(lower[g]) ||
           !std::isfinite(upper[g]) || !(w[g] > 0.0) || !std::isfinite(w[g]))
        {
            return false;
        }
        inside = inside && values[g] >= lower[g] && values[g] <= upper[g];
        sum += values[g];
        moment += w[g] * values[g];
        lowerSum += lower[g];
        upperSum += upper[g];
        scale = std::max({scale, std::abs(values[g]), std::abs(lower[g]), std::abs(upper[g])});
        wMax = std::max(wMax, w[g]);
        movable = movable || weight(g) > 0.0;
    }
    if(inside)
    {
        return true;
    }
    if(sum < lowerSum || sum > upperSum || !(scale > 0.0) || !movable)
    {
        return false;
    }
    std::array<double, G> clipped{};
    bool finite = true;
    auto evaluate = [&](double kappa)
    {
        // lambda with sum clip(v - s (lambda + kappa w)) = sum, by bisection.
        double low = std::numeric_limits<double>::infinity();
        double high = -std::numeric_limits<double>::infinity();
        for(std::size_t g = 0; g < G; ++g)
        {
            if(weight(g) > 0.0)
            {
                low = std::min(low, (values[g] - upper[g]) / weight(g) - kappa * w[g]);
                high = std::max(high, (values[g] - lower[g]) / weight(g) - kappa * w[g]);
            }
        }
        if(!std::isfinite(low) || !std::isfinite(high))
        {
            finite = false;
            return 0.0;
        }
        for(int iteration = 0; iteration < 200 && high - low > 1e-16 * (std::abs(low) + std::abs(high)); ++iteration)
        {
            double const lambda = 0.5 * (low + high);
            double total = 0.0;
            for(std::size_t g = 0; g < G; ++g)
            {
                total += std::clamp(values[g] - weight(g) * (lambda + kappa * w[g]), lower[g], upper[g]);
            }
            (total > sum ? low : high) = lambda;
        }
        double const lambda = 0.5 * (low + high);
        double result = 0.0;
        for(std::size_t g = 0; g < G; ++g)
        {
            clipped[g] = std::clamp(values[g] - weight(g) * (lambda + kappa * w[g]), lower[g], upper[g]);
            result += w[g] * clipped[g];
        }
        return result;
    };
    // kappa w_max of the size of the values (w = 1/eps spans many decades);
    // with scales, kappa w is relative.
    double const kappaScale = (scales ? 1.0 : scale) / wMax;
    double low = -kappaScale;
    double high = kappaScale;
    for(int expand = 0; expand < 100 && evaluate(low) < moment; ++expand)
    {
        low *= 2.0;
    }
    for(int expand = 0; expand < 100 && evaluate(high) > moment; ++expand)
    {
        high *= 2.0;
    }
    if(!finite || !(evaluate(low) >= moment) || !(evaluate(high) <= moment) || !finite)
    {
        return false;
    }
    for(int iteration = 0; iteration < 300 && high - low > 1e-16 * (std::abs(low) + std::abs(high)); ++iteration)
    {
        double const kappa = 0.5 * (low + high);
        (evaluate(kappa) > moment ? low : high) = kappa;
    }
    double const finalMoment = evaluate(0.5 * (low + high));
    double finalSum = 0.0;
    for(std::size_t g = 0; g < G; ++g)
    {
        finalSum += clipped[g];
    }
    bool clippedFinite = finite;
    for(double const value : clipped)
    {
        clippedFinite = clippedFinite && std::isfinite(value);
    }
    // Written so that a NaN fails: values are replaced only by a finite clip meeting both moments.
    if(!clippedFinite || !(std::abs(finalSum - sum) <= 1e-10 * scale * static_cast<double>(G)) ||
       !(std::abs(finalMoment - moment) <= 1e-8 * scale * wMax * static_cast<double>(G)))
    {
        return false;
    }
    values = clipped;
    return true;
}

// Photon-preserving reconciliation of one group: the census packets of the
// group hold energy E and photons N (h = N / E); the accepted target is
// (Etarget, Ntarget).  Packets are scaled by min(scale, upscaleCap); the rest
// of the target is spawned at the photon energy 1 / h of the group (inside
// the group for in-grid packets), or at the centre for an empty group.
struct GroupReconciliation
{
    double packetScale = 1.0;
    double spawnEnergy = 0.0;
    double spawnPhotonEnergy = 0.0;
};

inline GroupReconciliation ReconcileGroup(double censusEnergy, double censusPhotons, double targetEnergy,
                                          double centre, double upscaleCap)
{
    GroupReconciliation result;
    if(!(censusEnergy > 0.0) || !(censusPhotons > 0.0))
    {
        result.packetScale = 0.0;
        result.spawnEnergy = std::max(0.0, targetEnergy);
        result.spawnPhotonEnergy = centre;
        return result;
    }
    // A nearly empty census (packets of ~1e-300 erg) with a finite target would overflow target / censusEnergy, which
    // floating-point traps turn into a crash. Only min(target, cap E) is divided by E, so the quotient stays at most the
    // cap even when the compiler evaluates it for both outcomes (branchless code does); the result is the cap either way.
    double const target = std::max(0.0, targetEnergy);
    double const capEnergy = upscaleCap * censusEnergy;
    double const quotient = std::min(target, capEnergy) / censusEnergy;
    result.packetScale = target <= capEnergy ? std::min(quotient, upscaleCap) : upscaleCap;
    result.spawnEnergy = std::max(0.0, targetEnergy - result.packetScale * censusEnergy);
    result.spawnPhotonEnergy = censusEnergy / censusPhotons;
    return result;
}

} // namespace compton_thinned
} // namespace STORM

#endif // STORM_RADIATION_COMPTON_THINNED_HPP
