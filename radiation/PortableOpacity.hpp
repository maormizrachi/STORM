#ifndef STORM_PORTABLE_OPACITY_HPP
#define STORM_PORTABLE_OPACITY_HPP

#include <cstdint>

namespace STORM {
// Explicit contracts for the device snapshots. Unsupported models use their
// host implementation; sampling probes cannot establish an opacity law.
enum class PortableAbsorptionLaw : std::uint8_t
{
    Unsupported, InverseCube, PiecewiseConstant
};
enum class ThermalFrequencyLaw : std::uint8_t
{
    Unsupported, LinearInGroup, BoseEinstein0
};
}
#endif
