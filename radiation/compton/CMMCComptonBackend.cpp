#include "radiation/compton/CMMCComptonBackend.hpp"

#include <stdexcept>
#include <utility>

#ifdef STORM_WITH_COMPTON
#include "compton_matrix_mc.hpp"
#endif

namespace STORM {

struct CMMCComptonBackendCore::Impl
{
#ifdef STORM_WITH_COMPTON
    std::size_t groupCount = 0;
    mutable ComptonMatrixMC matrix;
#endif

#ifdef STORM_WITH_COMPTON
    Impl(std::vector<double> centers,
         std::vector<double> boundaries,
         std::size_t samples,
         bool detailedBalance,
         int seed):
        groupCount(centers.size()),
        matrix(std::move(centers),
               std::move(boundaries),
               samples,
               detailedBalance,
               seed)
#else
    Impl(std::vector<double> centers,
         std::vector<double> boundaries,
         std::size_t samples,
         bool detailedBalance,
         int seed)
#endif
    {
#ifndef STORM_WITH_COMPTON
        (void) centers;
        (void) boundaries;
        (void) samples;
        (void) detailedBalance;
        (void) seed;
#endif
    }
};

CMMCComptonBackendCore::CMMCComptonBackendCore(
    std::vector<double> centers,
    std::vector<double> boundaries,
    std::size_t samples,
    bool detailedBalance,
    int seed):
    impl_(std::make_unique<Impl>(
        std::move(centers),
        std::move(boundaries),
        samples,
        detailedBalance,
        seed))
{}

CMMCComptonBackendCore::~CMMCComptonBackendCore() = default;

CMMCComptonBackendCore::CMMCComptonBackendCore(
    CMMCComptonBackendCore &&other) noexcept = default;

CMMCComptonBackendCore &CMMCComptonBackendCore::operator=(
    CMMCComptonBackendCore &&other) noexcept = default;

void CMMCComptonBackendCore::SetTables(
    const std::vector<double> &temperatures)
{
#ifdef STORM_WITH_COMPTON
    this->impl_->matrix.set_tables(temperatures);
#else
    (void) temperatures;
    throw std::runtime_error(
        "STORM was built without internal Compton support");
#endif
}

CMMCComptonMatrixData CMMCComptonBackendCore::GetTauMatrix(
    double temperature,
    double density,
    double atomicWeight,
    double ionization) const
{
#ifdef STORM_WITH_COMPTON
    CMMCComptonMatrixData result;
    result.tau.assign(
        this->impl_->groupCount,
        std::vector<double>(this->impl_->groupCount, 0.0));
    result.dtau_dUm.assign(
        this->impl_->groupCount,
        std::vector<double>(this->impl_->groupCount, 0.0));
    this->impl_->matrix.get_tau_matrix(
        temperature,
        density,
        atomicWeight,
        ionization,
        result.tau,
        result.dtau_dUm);
    std::pair<double, double> const lastGroup =
        this->impl_->matrix.get_last_group_upscattering_and_downscattering(
            temperature,
            density,
            atomicWeight,
            ionization);
    result.lastGroupUpScatter = lastGroup.first;
    result.lastGroupDownScatter = lastGroup.second;
    return result;
#else
    (void) temperature;
    (void) density;
    (void) atomicWeight;
    (void) ionization;
    throw std::runtime_error(
        "STORM was built without internal Compton support");
#endif
}

double CMMCComptonBackendCore::GetMaximumTemperature() const
{
#ifdef STORM_WITH_COMPTON
    return this->impl_->matrix.get_maximum_temperature_grid();
#else
    throw std::runtime_error(
        "STORM was built without internal Compton support");
#endif
}

void CMMCComptonBackendCore::GetAngleCdf(
    double temperature,
    std::size_t sourceGroup,
    std::size_t targetGroup,
    std::vector<double> &cdf) const
{
#ifdef STORM_WITH_COMPTON
    this->impl_->matrix.get_angle_cdf(
        temperature, sourceGroup, targetGroup, cdf);
#else
    (void) temperature;
    (void) sourceGroup;
    (void) targetGroup;
    (void) cdf;
    throw std::runtime_error(
        "STORM was built without internal Compton support");
#endif
}

std::size_t CMMCComptonBackendCore::GetAngleBinCount() const
{
#ifdef STORM_WITH_COMPTON
    return ComptonMatrixMC::NUM_ANGLE_BINS;
#else
    return 0;
#endif
}

} // namespace STORM
