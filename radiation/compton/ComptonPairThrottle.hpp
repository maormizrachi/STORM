#ifndef STORM_RADIATION_COMPTON_PAIR_THROTTLE_HPP
#define STORM_RADIATION_COMPTON_PAIR_THROTTLE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace STORM {

// Symmetric pair throttle (a cost option, on by default with nMax = 100 in both codes): multiplies both directions of every pair i != j, the rates and their
// derivatives, by lambda_ij = min(1, (1 + nMax) / (1 + max(n_i, n_j))).  The net photon flow of each pair becomes
// lambda_ij F_ij, so every fixed point of the operator (discrete Planck, the discrete Bose-Einstein family) stays one,
// while the stimulated rate tau_st (1 + n_t) of each direction is bounded by (1 + nMax) tau_st.  Unlike the occupation
// cap, which scales the two directions differently, it adds no spurious flow.  n must be the occupations used in the
// (1 + n) factors; the diagonal (in-group rate, top-group term) is untouched; nMax <= 0 is off.  The low-frequency
// kinetics of throttled pairs are slowed by lambda, and so is their stimulated transport opacity.  A pair with a
// non-finite occupation is left untouched.
template<class Occupations, class Matrix>
void ApplyComptonPairThrottle(const Occupations &n, double nMax, Matrix &tau, Matrix &dtau)
{
    if(!(nMax > 0.0))
    {
        return;
    }
    std::size_t const groups = n.size();
    for(std::size_t i = 0; i < groups; ++i)
    {
        for(std::size_t j = i + 1; j < groups; ++j)
        {
            if(!std::isfinite(n[i]) || !std::isfinite(n[j]))
            {
                continue;
            }
            double const occupation = std::max(n[i], n[j]);
            if(!(occupation > nMax))
            {
                continue;
            }
            double const lambda = (1.0 + nMax) / (1.0 + occupation);
            tau[i][j] *= lambda;
            tau[j][i] *= lambda;
            dtau[i][j] *= lambda;
            dtau[j][i] *= lambda;
        }
    }
}

} // namespace STORM

#endif // STORM_RADIATION_COMPTON_PAIR_THROTTLE_HPP
