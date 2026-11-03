#ifndef STORM_RADIATION_COMPTON_DETAILED_BALANCE_HPP
#define STORM_RADIATION_COMPTON_DETAILED_BALANCE_HPP

#include <vector>

#include "radiation/compton/ComptonTableBackend.hpp"

namespace STORM {

// Detailed balance on Compton rates at the electron temperature T, as CMMC enforces it: for each pair i < j the
// smaller rate follows the larger through f = (1 + n_j) B_i e_j / ((1 + n_i) B_j e_i), with B_g and n_g the
// centre-form Planck group energy and occupation at T.  The temperature derivative of the derived rate follows the
// same relation, d tau_derived/dT = d tau_kept/dT f + tau_kept df/dT; otherwise the induced-scattering factors
// (1 + n) ~ 1e3-1e5 of Rayleigh-Jeans groups turn an O(1) raw derivative imbalance into a large spurious source in
// the temperature-linearized operator.  derivativesValid = false (a clamped temperature) leaves the derivatives
// unchanged.  adjustRates = false keeps the rates as given (already balanced, e.g. by CMMC) and only makes the
// derivative of each pair's smaller rate follow the relation.
void EnforceComptonDetailedBalance(const std::vector<double> &centers,
                                   const std::vector<double> &boundaries,
                                   double temperature,
                                   bool derivativesValid,
                                   bool adjustRates,
                                   ComptonMatrixData &data);

} // namespace STORM

#endif // STORM_RADIATION_COMPTON_DETAILED_BALANCE_HPP
