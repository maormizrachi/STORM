#ifndef STORM_RADIATION_IMC_MATERIAL_ACCESS_HPP
#define STORM_RADIATION_IMC_MATERIAL_ACCESS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <units/units.hpp>
#include "../../StormError.hpp"
#include "../../elementary/PointOps.hpp"

namespace STORM::radiation_imc_detail {

template<typename T, typename = void>
struct has_member_ID : std::false_type {};

template<typename T>
struct has_member_ID<T, std::void_t<decltype(std::declval<const T &>().ID)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_density : std::false_type {};

template<typename T>
struct has_member_density<T, std::void_t<decltype(std::declval<const T &>().density)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_pressure : std::false_type {};

template<typename T>
struct has_member_pressure<T, std::void_t<decltype(std::declval<T &>().pressure)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_internal_energy_specific : std::false_type {};

template<typename T>
struct has_member_internal_energy_specific<T, std::void_t<decltype(std::declval<T &>().internal_energy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_internal_energy_density : std::false_type {};

template<typename T>
struct has_member_internal_energy_density<T, std::void_t<decltype(std::declval<T &>().internalEnergy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_total_energy : std::false_type {};

template<typename T>
struct has_member_total_energy<T, std::void_t<decltype(std::declval<T &>().energy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_mass : std::false_type {};

template<typename T>
struct has_member_mass<T, std::void_t<decltype(std::declval<const T &>().mass)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_radiation_energy : std::false_type {};

template<typename T>
struct has_member_radiation_energy<T, std::void_t<decltype(std::declval<T &>().Erad)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_group_energy_mutable : std::false_type {};

template<typename T>
struct has_member_group_energy_mutable<T, std::void_t<decltype(std::declval<T &>().Eg)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_velocity : std::false_type {};

template<typename T>
struct has_member_velocity<T, std::void_t<decltype(std::declval<const T &>().velocity)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_momentum : std::false_type {};

template<typename T>
struct has_member_momentum<T, std::void_t<decltype(std::declval<T &>().momentum)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_temperature : std::false_type {};

template<typename T>
struct has_member_temperature<T, std::void_t<decltype(std::declval<const T &>().temperature)>> : std::true_type {};

template<typename EOST, typename TracersT, typename TracerNamesT, typename = void>
struct has_dT2e : std::false_type {};

template<typename EOST, typename TracersT, typename TracerNamesT>
struct has_dT2e<EOST, TracersT, TracerNamesT, std::void_t<decltype(
    std::declval<const EOST &>().dT2e(
        std::declval<double>(), std::declval<double>(),
        std::declval<const TracersT &>(),
        std::declval<const TracerNamesT &>()))>> : std::true_type {};

template<typename OpacityT, typename CellT, typename = void>
struct has_opacity_calc_planck : std::false_type {};

template<typename OpacityT, typename CellT>
struct has_opacity_calc_planck<OpacityT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().CalcPlanckOpacity(std::declval<const CellT &>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().CalcPlanckOpacity(std::declval<const CellT &>())),
        double> {};

template<typename OpacityT, typename CellT, typename = void>
struct has_opacity_calc_absorption : std::false_type {};

template<typename OpacityT, typename CellT>
struct has_opacity_calc_absorption<OpacityT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().CalcAbsorptionOpacity(
        std::declval<const CellT &>(), std::declval<double>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().CalcAbsorptionOpacity(
            std::declval<const CellT &>(), std::declval<double>())),
        double> {};

template<typename OpacityT, typename CellT, typename = void>
struct has_opacity_calc_scattering : std::false_type {};

template<typename OpacityT, typename CellT>
struct has_opacity_calc_scattering<OpacityT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().CalcScatteringOpacity(std::declval<const CellT &>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().CalcScatteringOpacity(std::declval<const CellT &>())),
        double> {};

template<typename OpacityT, typename CellT, typename = void>
struct has_opacity_calc_scattering_frequency : std::false_type {};

template<typename OpacityT, typename CellT>
struct has_opacity_calc_scattering_frequency<OpacityT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().CalcScatteringOpacity(
        std::declval<const CellT &>(), std::declval<double>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().CalcScatteringOpacity(
            std::declval<const CellT &>(), std::declval<double>())),
        double> {};

template<typename OpacityT, typename PointT, typename CellT, typename = void>
struct has_opacity_random_velocity : std::false_type {};

template<typename OpacityT, typename PointT, typename CellT>
struct has_opacity_random_velocity<OpacityT, PointT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().getRandomVelocity(
        std::declval<const CellT &>(), std::declval<double>(), std::declval<double>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().getRandomVelocity(
            std::declval<const CellT &>(), std::declval<double>(), std::declval<double>())),
        PointT> {};

template<typename OpacityT, typename PointT, typename CellT, typename = void>
struct has_opacity_scatter_velocity : std::false_type {};

template<typename OpacityT, typename PointT, typename CellT>
struct has_opacity_scatter_velocity<OpacityT, PointT, CellT, std::void_t<
    decltype(std::declval<OpacityT &>().getNewScatterVelocity(
        std::declval<const CellT &>(), std::declval<const PointT &>(),
        std::declval<double>(), std::declval<double>(), std::declval<double>()))
    >> : std::is_convertible<
        decltype(std::declval<OpacityT &>().getNewScatterVelocity(
            std::declval<const CellT &>(), std::declval<const PointT &>(),
            std::declval<double>(), std::declval<double>(), std::declval<double>())),
        PointT> {};

template<typename OpacityT, std::size_t NumGroups, typename = void>
struct has_opacity_find_group : std::false_type {};

template<typename OpacityT, std::size_t NumGroups>
struct has_opacity_find_group<OpacityT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().findGroup(
        std::declval<double>(),
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_convertible<
        decltype(std::declval<const OpacityT &>().findGroup(
            std::declval<double>(),
            std::declval<const std::array<double, NumGroups + 1> &>())),
        std::size_t> {};

template<typename OpacityT, typename CellT, std::size_t NumGroups, typename = void>
struct has_opacity_thermal_energy : std::false_type {};

template<typename OpacityT, typename CellT, std::size_t NumGroups>
struct has_opacity_thermal_energy<OpacityT, CellT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().GetThermalEnergy(
        std::declval<const CellT &>(), std::declval<double>(),
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_convertible<
        decltype(std::declval<const OpacityT &>().GetThermalEnergy(
            std::declval<const CellT &>(), std::declval<double>(),
            std::declval<const std::array<double, NumGroups + 1> &>())),
        double> {};

template<typename OpacityT, typename CellT, std::size_t NumGroups, typename = void>
struct has_opacity_sample_thermal_in_group : std::false_type {};

template<typename OpacityT, typename CellT, std::size_t NumGroups>
struct has_opacity_sample_thermal_in_group<OpacityT, CellT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().SampleThermalEnergyInGroup(
        std::declval<const CellT &>(), std::declval<std::size_t>(),
        std::declval<double>(),
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_convertible<
        decltype(std::declval<const OpacityT &>().SampleThermalEnergyInGroup(
            std::declval<const CellT &>(), std::declval<std::size_t>(),
            std::declval<double>(),
            std::declval<const std::array<double, NumGroups + 1> &>())),
        double> {};

template<typename OpacityT, typename CellT, std::size_t NumGroups, typename = void>
struct has_opacity_thermal_group_pdf : std::false_type {};

template<typename OpacityT, typename CellT, std::size_t NumGroups>
struct has_opacity_thermal_group_pdf<OpacityT, CellT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().GetThermalGroupPdf(
        std::declval<const CellT &>(),
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_same<
        std::remove_cv_t<std::remove_reference_t<
            decltype(std::declval<const OpacityT &>().GetThermalGroupPdf(
                std::declval<const CellT &>(),
                std::declval<const std::array<double, NumGroups + 1> &>()))>>,
        std::array<double, NumGroups>> {};

template<typename OpacityT, typename CellT, std::size_t NumGroups, typename = void>
struct has_opacity_cumulative_opacity : std::false_type {};

template<typename OpacityT, typename CellT, std::size_t NumGroups>
struct has_opacity_cumulative_opacity<OpacityT, CellT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().GetCumulativeOpacity(
        std::declval<const CellT &>(),
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_same<
        std::remove_cv_t<std::remove_reference_t<
            decltype(std::declval<const OpacityT &>().GetCumulativeOpacity(
                std::declval<const CellT &>(),
                std::declval<const std::array<double, NumGroups + 1> &>()))>>,
        std::array<double, NumGroups>> {};

template<typename OpacityT, std::size_t NumGroups, typename = void>
struct has_opacity_energy_centers : std::false_type {};

template<typename OpacityT, std::size_t NumGroups>
struct has_opacity_energy_centers<OpacityT, NumGroups, std::void_t<
    decltype(std::declval<const OpacityT &>().getEnergyCenters(
        std::declval<const std::array<double, NumGroups + 1> &>()))
    >> : std::is_same<
        std::remove_cv_t<std::remove_reference_t<
            decltype(std::declval<const OpacityT &>().getEnergyCenters(
                std::declval<const std::array<double, NumGroups + 1> &>()))>>,
        std::array<double, NumGroups>> {};

template<typename OpacityT, typename = void>
struct has_opacity_reseed : std::false_type {};

template<typename OpacityT>
struct has_opacity_reseed<OpacityT, std::void_t<
    decltype(std::declval<OpacityT &>().reseed(std::declval<std::uint64_t>()))
    >> : std::true_type {};

template<typename CellT>
std::size_t cellID(const CellT &cell)
{
    if constexpr(has_member_ID<CellT>::value)
    {
        return cell.ID;
    }
    else
    {
        (void) cell;
        return std::numeric_limits<std::size_t>::max();
    }
}

template<typename T, typename = void>
struct has_get_self_index : std::false_type {};

template<typename T>
struct has_get_self_index<T, std::void_t<
    decltype(std::declval<const T &>().GetSelfIndex())>> : std::true_type {};

template<typename GridT, typename CellT>
std::size_t ddmcStableCellID(const GridT &grid,
                             std::size_t cellIndex,
                             const CellT &cell)
{
    if constexpr(has_member_ID<CellT>::value)
    {
        return static_cast<std::size_t>(cell.ID);
    }
    else if constexpr(has_get_self_index<GridT>::value)
    {
        const std::vector<std::size_t> &selfIndex = grid.GetSelfIndex();
        if(cellIndex < selfIndex.size())
        {
            return selfIndex[cellIndex];
        }
    }
    return cellIndex;
}

template<typename ExtensivesT>
void addTotalEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_total_energy<ExtensivesT>::value)
    {
        extensives.energy += energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
void clearRadiationEnergyIfPresent(ExtensivesT &extensives)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad = 0.0;
    }
    else
    {
        (void) extensives;
    }
}

template<typename ExtensivesT>
void addRadiationEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad += energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
void setRadiationEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad = energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
double radiationEnergyIfPresent(const ExtensivesT &extensives)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        return extensives.Erad;
    }
    else
    {
        (void) extensives;
        return 0.0;
    }
}

template<typename ExtensivesT>
void clearGroupEnergyIfPresent(ExtensivesT &extensives)
{
    if constexpr(has_member_group_energy_mutable<ExtensivesT>::value)
    {
        std::fill(extensives.Eg.begin(), extensives.Eg.end(), 0.0);
    }
    else
    {
        (void) extensives;
    }
}

template<typename CellT>
void setCellRadiationEnergyIfPresent(CellT &cell, double value)
{
    if constexpr(has_member_radiation_energy<CellT>::value)
    {
        cell.Erad = value;
    }
    else
    {
        (void) cell;
        (void) value;
    }
}

template<typename CellT>
void setCellGroupEnergyIfPresent(CellT &cell, std::size_t group, double value)
{
    if constexpr(has_member_group_energy_mutable<CellT>::value)
    {
        cell.Eg[group] = value;
    }
    else
    {
        (void) cell;
        (void) group;
        (void) value;
    }
}

template<typename PointT, typename ParticleT, typename CellT>
double computeDopplerShift(const ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return 1.0;
        }
        double gamma = 1.0 / std::sqrt(1.0 - v2 * units::inv_clight2);
        return gamma * (1.0 - ScalarProd(cell.velocity, particle.velocity) * units::inv_clight2);
    }
    else
    {
        (void) particle;
        (void) cell;
    }
    return 1.0;
}

template<typename PointT, typename ParticleT, typename CellT>
void lorentzTransformToComoving(ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double const v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return;
        }
        double const gamma = 1.0 / std::sqrt(
            1.0 - v2 * units::inv_clight2);
        double const dopplerShift = gamma *
            (1.0 - ScalarProd(cell.velocity, particle.velocity) *
             units::inv_clight2);
        if(!(dopplerShift > 0.0) || !std::isfinite(dopplerShift))
        {
            throw StormError(
                "RadiationIMC received an invalid lab-to-comoving Doppler factor");
        }
        particle.frequency *= dopplerShift;
        particle.weight *= dopplerShift;
        double const vDotP = ScalarProd(particle.velocity, cell.velocity);
        particle.velocity = particle.velocity + cell.velocity *
            ((gamma - 1.0) * vDotP / v2 - gamma);
        double const newSpeed = fastabs(particle.velocity);
        if(newSpeed > 0.0)
        {
            particle.velocity *= units::clight / newSpeed;
        }
    }
    else
    {
        (void) particle;
        (void) cell;
    }
}

template<typename PointT, typename ParticleT, typename CellT>
void lorentzTransformToLab(ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return;
        }
        double gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
        PointT negV = cell.velocity * (-1.0);
        double dopplerShift = gamma * (1.0 - ScalarProd(negV, particle.velocity) * units::inv_clight2);
        particle.frequency *= dopplerShift;
        particle.weight *= dopplerShift;
        double vDotP = ScalarProd(particle.velocity, negV);
        particle.velocity = particle.velocity + negV * ((gamma - 1.0) * vDotP / v2 - gamma);
        double newSpeed = fastabs(particle.velocity);
        if(newSpeed > 0.0)
        {
            particle.velocity = particle.velocity * (units::clight / newSpeed);
        }
    }
    else
    {
        (void) particle;
        (void) cell;
    }
}

} // namespace radiation_imc_detail

#endif // STORM_RADIATION_IMC_MATERIAL_ACCESS_HPP
