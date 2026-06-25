#ifndef STORM_PHYSICAL_CONSTANTS_HPP
#define STORM_PHYSICAL_CONSTANTS_HPP

namespace STORM {
namespace constants {

static constexpr double clight = 2.99792458e10;           // cm/s
static constexpr double inv_clight = 1.0 / clight;        // s/cm
static constexpr double sigma_sb = 5.670374419e-5;        // erg cm^-2 s^-1 K^-4
static constexpr double arad = 4.0 * sigma_sb / clight;   // erg cm^-3 K^-4
static constexpr double k_boltz = 1.380649e-16;           // erg/K

} // namespace constants
} // namespace STORM

#endif // STORM_PHYSICAL_CONSTANTS_HPP
