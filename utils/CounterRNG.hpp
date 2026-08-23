#ifndef STORM_COUNTER_RNG_HPP
#define STORM_COUNTER_RNG_HPP

#include <cstdint>

namespace STORM {

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
        return mix(seed ^ mix(creationRank) ^ mix(particleID));
    }

    static std::uint64_t next(std::uint64_t key, std::uint64_t counter)
    {
        // A counter-based stream: the result is a pure function of key and
        // counter, so scheduling and rank migration cannot change the stream.
        return mix(key + counter * 0x9e3779b97f4a7c15ULL);
    }

    static double unitOpen(std::uint64_t key, std::uint64_t counter)
    {
        const std::uint64_t mantissa = next(key, counter) >> 11U;
        return (static_cast<double>(mantissa) + 0.5) * 0x1.0p-53;
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
