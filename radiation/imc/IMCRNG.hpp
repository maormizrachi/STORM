#ifndef STORM_RADIATION_IMC_RNG_HPP
#define STORM_RADIATION_IMC_RNG_HPP

#include <cstdint>
#include <random>

namespace STORM::radiation_imc_detail {

/// Random streams and reusable sampler scratch storage for IMC phases.
template<typename PositionDecompositionT>
struct IMCRNG
{
    explicit IMCRNG(std::uint64_t seed) : rng_(seed), dist_(0.0, 1.0)
    {}

    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_;
    std::uint64_t particleRngSeed_ = 42;
    std::uint64_t sourceRngStreamCounter_ = 0;
    std::uint64_t creationRank_ = 0;
    bool creationRankCached_ = false;
    PositionDecompositionT scratchDecomposition_;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_RNG_HPP
