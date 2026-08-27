#ifndef STORM_RADIATION_IMC_CONCEPTS_HPP
#define STORM_RADIATION_IMC_CONCEPTS_HPP

#include <type_traits>

namespace STORM::radiation_imc_detail {

/// Detect position samplers exposing the reusable-decomposition API.
template<typename SamplerT, typename = void>
struct sampler_decomposition
{
    struct type {};
    static constexpr bool supported = false;
};

template<typename SamplerT>
struct sampler_decomposition<SamplerT, std::void_t<typename SamplerT::Decomposition>>
{
    using type = typename SamplerT::Decomposition;
    static constexpr bool supported = true;
};

} // namespace STORM::radiation_imc_detail

#endif
