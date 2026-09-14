#ifndef STORM_COUNTER_RNG_HPP
#define STORM_COUNTER_RNG_HPP

#include <cstdint>

#ifdef STORM_WITH_GPU
#include "../gpu/KokkosTypes.hpp"
#define STORM_RNG_INLINE STORM_GPU_INLINE_FUNCTION
#else
#define STORM_RNG_INLINE inline
#endif

namespace STORM {

class CounterRNG
{
public:
    STORM_RNG_INLINE static std::uint64_t mix(std::uint64_t value)
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    STORM_RNG_INLINE static std::uint64_t makeKey(std::uint64_t seed,
                                                  std::uint64_t creationRank,
                                                  std::uint64_t particleID)
    {
        // Fold fields in order: XORing independently mixed rank and ID
        // aliases (rank, ID) with (ID, rank), and cancels when rank == ID.
        const std::uint64_t rankKey = mix(mix(seed) ^ creationRank);
        return mix(rankKey ^ particleID);
    }

    STORM_RNG_INLINE static std::uint64_t next(std::uint64_t key, std::uint64_t counter)
    {
        // A counter-based stream: the result is a pure function of key and
        // counter, so scheduling and rank migration cannot change the stream.
        return mix(key + counter * 0x9e3779b97f4a7c15ULL);
    }

    STORM_RNG_INLINE static double unitOpen(std::uint64_t key, std::uint64_t counter)
    {
        // Use 52 bits so the half-bin offset remains exactly representable.
        // With 53 bits, the highest bin rounds to 1.0 in double precision.
        const std::uint64_t mantissa = next(key, counter) >> 12U;
        return (static_cast<double>(mantissa) + 0.5) * 0x1.0p-52;
    }
};

// Standard UniformRandomBitGenerator facade over a particle-owned key/counter.
// It lets existing templated sampling code consume the same migration-stable
// stream without sharing mutable engine state between particles.
class ParticleCounterEngine
{
public:
    using result_type = std::uint64_t;

    ParticleCounterEngine(std::uint64_t key, std::uint64_t &counter)
        : key_(key), counter_(counter) {}

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    result_type operator()()
    {
        return CounterRNG::next(this->key_, this->counter_++);
    }

private:
    std::uint64_t key_;
    std::uint64_t &counter_;
};

} // namespace STORM

#undef STORM_RNG_INLINE

#endif // STORM_COUNTER_RNG_HPP
