#ifndef STORM_COUNTER_RNG_HPP
#define STORM_COUNTER_RNG_HPP

#include <cstdint>

namespace STORM {

namespace counter_rng_detail {

constexpr double counterBitsToUnitOpen(std::uint64_t bits)
{
    // A 52-bit mantissa keeps the half-bin offset exactly representable at
    // both ends of the binary64 interval.
    const std::uint64_t mantissa = bits >> 12U;
    return (static_cast<double>(mantissa) + 0.5) * 0x1.0p-52;
}

} // namespace counter_rng_detail

class CounterRNG
{
public:
    static std::uint64_t mix(std::uint64_t value)
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    static std::uint64_t makeKey(std::uint64_t seed,
                                 std::uint64_t creationRank,
                                 std::uint64_t particleID)
    {
        // Mix fields in order with distinct domain tags so creation rank and
        // particle ID are not interchangeable inputs.
        std::uint64_t key = mix(seed ^ 0x243f6a8885a308d3ULL);
        key = mix(key ^ mix(creationRank ^ 0x13198a2e03707344ULL));
        return mix(key ^ mix(particleID ^ 0xa4093822299f31d0ULL));
    }

    static std::uint64_t next(std::uint64_t key, std::uint64_t counter)
    {
        // A counter-based stream: the result is a pure function of key and
        // counter, so scheduling and rank migration cannot change the stream.
        return mix(key + counter * 0x9e3779b97f4a7c15ULL);
    }

    static double unitOpen(std::uint64_t key, std::uint64_t counter)
    {
        return counter_rng_detail::counterBitsToUnitOpen(next(key, counter));
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

#endif // STORM_COUNTER_RNG_HPP
