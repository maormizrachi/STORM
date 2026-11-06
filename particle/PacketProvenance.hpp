#ifndef STORM_PARTICLE_PACKET_PROVENANCE_HPP
#define STORM_PARTICLE_PACKET_PROVENANCE_HPP

#include <cstdint>

namespace STORM
{
// Post-process packet provenance (diagnostic only; never changes a transport
// decision or consumes a random number). Three bytes of the particle:
//
//   birthGroup   lab-frame group at the completed birth, kUnknownGroup when
//                the packet was not born through a post-process source path.
//   provenance0  birth classification and representation history:
//                bits 0-1 birth kind (volume/face x physical/learned group
//                pdf, as GroupSamplingDiagnostics::birthKind), bit 2 born in
//                an exploration-fallback cell, bit 3 was ever a DDMC
//                resident, bit 4 ever carried a pending DDMC flux, bit 5
//                descends from a split the copy budget reduced or refused,
//                bits 6-7 for candidate window 0/1: in a valid owned IMC
//                state of a cell with zero importance, the packet was
//                heavier than that candidate's importance-one window.
//   provenance1  split opportunities and frequency history:
//                bits 0/2 candidate 0/1 had an overweight split opportunity
//                (valid owned IMC state, importance > 0, |w| > W_g / I),
//                bits 1/3 a committed stochastic event (scattering,
//                re-emission, random-walk step, DDMC step) followed that
//                opportunity, bits 4-7 the lab-frame group changed through
//                moving-frame elastic scattering, effective thermal
//                scattering, or source-face re-emission (IMC channels), or
//                the packet went through DDMC frequency redistribution or
//                compression (history only; a resident has no lab group).
// All bits are history: they are copied to split copies and kept through
// migration.
namespace provenance
{
constexpr std::uint8_t kUnknownGroup = 255;
constexpr unsigned kCandidates = 2;

constexpr std::uint8_t kBirthKindMask = 0x03;
constexpr std::uint8_t kExploration = 0x04;
constexpr std::uint8_t kEverDDMC = 0x08;
constexpr std::uint8_t kEverPendingFlux = 0x10;
constexpr std::uint8_t kRefusedAncestor = 0x20;
inline constexpr std::uint8_t kZeroImportanceOverweight[kCandidates] = {0x40, 0x80};

inline constexpr std::uint8_t kOpportunity[kCandidates] = {0x01, 0x04};
inline constexpr std::uint8_t kFollowed[kCandidates] = {0x02, 0x08};
constexpr std::uint8_t kChannelElastic = 0x10;
constexpr std::uint8_t kChannelThermal = 0x20;
constexpr std::uint8_t kChannelDDMC = 0x40;
constexpr std::uint8_t kChannelFace = 0x80;
constexpr unsigned kChannelShift = 4;

inline std::uint8_t birthBits(unsigned kind, bool exploration)
{
    return static_cast<std::uint8_t>((kind & kBirthKindMask) | (exploration ? kExploration : 0u));
}

// A committed stochastic event: every candidate whose opportunity was
// already recorded is now followed by independent transport.
inline void markStochastic(std::uint8_t &provenance1)
{
    for(unsigned c = 0; c < kCandidates; ++c)
    {
        if(provenance1 & kOpportunity[c])
        {
            provenance1 = static_cast<std::uint8_t>(provenance1 | kFollowed[c]);
        }
    }
}

// Mutually exclusive tail classes of an escaping packet for one candidate,
// in priority order.
enum class TailClass : unsigned
{
    Useful = 0,          // opportunity followed by stochastic transport
    Ballistic = 1,       // opportunity, nothing stochastic after it
    ZeroImportance = 2,  // no opportunity; overweight in a zero-importance cell
    DDMC = 3,            // none of the above; DDMC resident or pending flux
    None = 4             // none of the above
};
constexpr unsigned kTailClasses = 5;

inline TailClass classify(std::uint8_t provenance0, std::uint8_t provenance1, unsigned candidate)
{
    if(provenance1 & kOpportunity[candidate])
    {
        return (provenance1 & kFollowed[candidate]) ? TailClass::Useful : TailClass::Ballistic;
    }
    if(provenance0 & kZeroImportanceOverweight[candidate])
    {
        return TailClass::ZeroImportance;
    }
    if(provenance0 & (kEverDDMC | kEverPendingFlux))
    {
        return TailClass::DDMC;
    }
    return TailClass::None;
}

inline unsigned channelMask(std::uint8_t provenance1)
{
    return static_cast<unsigned>(provenance1 >> kChannelShift) & 0x0Fu;
}
} // namespace provenance
} // namespace STORM

#endif // STORM_PARTICLE_PACKET_PROVENANCE_HPP
