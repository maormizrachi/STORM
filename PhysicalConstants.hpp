#ifndef STORM_PHYSICAL_CONSTANTS_HPP
#define STORM_PHYSICAL_CONSTANTS_HPP

namespace STORM {
namespace constants {

static constexpr double clight = 2.99792458e10;           // cm/s
static constexpr double inv_clight = 1.0 / clight;        // s/cm
static constexpr double clight2 = clight * clight;        // cm^2/s^2
static constexpr double inv_clight2 = 1.0 / clight2;      // s^2/cm^2
static constexpr double sigma_sb = 5.670374419e-5;        // erg cm^-2 s^-1 K^-4
static constexpr double arad = 4.0 * sigma_sb / clight;   // erg cm^-3 K^-4
static constexpr double k_boltz = 1.380649e-16;           // erg/K
static constexpr double me = 9.109383713928e-28;          // g
static constexpr double me_c2 = me * clight2;             // erg
static constexpr double sigma_thomson = 6.652458732160e-25; // cm^2
static constexpr double ev = 1.602176634e-12;             // erg
static constexpr double kev = 1.0e3 * ev;                 // erg
static constexpr double ev_kelvin = ev / k_boltz;         // K
static constexpr double kev_kelvin = 1.0e3 * ev_kelvin;   // K
static constexpr double Navogadro = 6.02214076e23;        // mol^-1
static constexpr double planck_constant = 6.62607015e-27; // erg s

} // namespace constants
} // namespace STORM

#endif // STORM_PHYSICAL_CONSTANTS_HPP
