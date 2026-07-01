#ifndef STORM_RADIATION_IMC_TRAITS_HPP
#define STORM_RADIATION_IMC_TRAITS_HPP

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace STORM {

namespace radiation_imc_detail {

template<typename T, typename = void>
struct has_static_energy_boundaries : std::false_type {};

template<typename T>
struct has_static_energy_boundaries<T, std::void_t<decltype(T::energyBoundaries)>> : std::true_type {};

template<typename T, typename = void>
struct has_static_tracer_names : std::false_type {};

template<typename T>
struct has_static_tracer_names<T, std::void_t<decltype(T::tracerNames)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_tracer_names : std::false_type {};

template<typename T>
struct has_member_tracer_names<T, std::void_t<decltype(std::declval<const T &>().tracerNames)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_tracers : std::false_type {};

template<typename T>
struct has_member_tracers<T, std::void_t<decltype(std::declval<const T &>().tracers)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_group_energy : std::false_type {};

template<typename T>
struct has_member_group_energy<T, std::void_t<decltype(std::declval<const T &>().Eg)>> : std::true_type {};

inline const std::vector<double> &emptyDoubleVector()
{
    static const std::vector<double> empty;
    return empty;
}

inline const std::vector<std::string> &emptyStringVector()
{
    static const std::vector<std::string> empty;
    return empty;
}

} // namespace radiation_imc_detail

template<typename PointT, typename CellT, typename ExtensivesT, std::size_t NumGroups>
struct DirectRadiationIMCTraits
{
    using Point = PointT;
    using Cell = CellT;
    using Extensives = ExtensivesT;
    using GroupArray = std::array<double, NumGroups>;
    using GroupBoundaries = std::array<double, NumGroups + 1>;

    GroupBoundaries energyBoundaries(const CellT &cell) const
    {
        (void) cell;
        GroupBoundaries boundaries{};
        if constexpr(radiation_imc_detail::has_static_energy_boundaries<CellT>::value)
        {
            for(std::size_t g = 0; g < NumGroups + 1; ++g)
            {
                boundaries[g] = CellT::energyBoundaries[g];
            }
        }
        else
        {
            for(std::size_t g = 0; g < NumGroups + 1; ++g)
            {
                boundaries[g] = static_cast<double>(g);
            }
        }
        return boundaries;
    }

    decltype(auto) tracers(const CellT &cell) const
    {
        if constexpr(radiation_imc_detail::has_member_tracers<CellT>::value)
        {
            return (cell.tracers);
        }
        else
        {
            (void) cell;
            return (radiation_imc_detail::emptyDoubleVector());
        }
    }

    decltype(auto) tracerNames(const CellT &cell) const
    {
        if constexpr(radiation_imc_detail::has_static_tracer_names<CellT>::value)
        {
            (void) cell;
            return (CellT::tracerNames);
        }
        else if constexpr(radiation_imc_detail::has_member_tracer_names<CellT>::value)
        {
            return (cell.tracerNames);
        }
        else
        {
            (void) cell;
            return (radiation_imc_detail::emptyStringVector());
        }
    }

    double groupEnergyPerMass(const CellT &cell, std::size_t group) const
    {
        if constexpr(radiation_imc_detail::has_member_group_energy<CellT>::value)
        {
            return cell.Eg[group];
        }
        else
        {
            (void) cell;
            (void) group;
            return 0.0;
        }
    }

    double extensiveGroupEnergy(const ExtensivesT &extensives, std::size_t group) const
    {
        if constexpr(radiation_imc_detail::has_member_group_energy<ExtensivesT>::value)
        {
            return extensives.Eg[group];
        }
        else
        {
            (void) extensives;
            (void) group;
            return 0.0;
        }
    }
};

} // namespace STORM

#endif // STORM_RADIATION_IMC_TRAITS_HPP
