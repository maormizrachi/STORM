#ifndef STORM_RADIATION_COMPTON_PROCESS_HPP
#define STORM_RADIATION_COMPTON_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"
#include "../source/SourceCore.hpp"
#include "ComptonPairThrottle.hpp"

namespace STORM::radiation_imc_detail {

template<typename Owner>
class ComptonProcess final : public IMCComponentBase<Owner>
{
    using Base = IMCComponentBase<Owner>;
    using Base::owner_;
    using typename Base::PointT;
    using typename Base::GridT;
    using typename Base::CellT;
    using typename Base::ExtensivesT;
    using typename Base::EOST;
    using typename Base::OpacityT;
    using typename Base::TraitsT;
    using typename Base::PositionSamplerT;
    using typename Base::Parameters;
    using typename Base::MCParticle;
    using typename Base::Functionality;
    using typename Base::BoundaryCond;
    using typename Base::PositionDecomposition;
    using typename Base::GroupArray;
    using typename Base::GroupBoundaries;
    using typename Base::GroupCdf;
    using typename Base::GroupMatrix;
    using typename Base::GroupCdfMatrix;
    using typename Base::ComptonCellData;
    using typename Base::Observer;
    using typename Base::DDMCCellData;
    using typename Base::DDMCFaceLeak;
    using typename Base::SourceAllocationSummary;
    using typename Base::GroupSamplingDiagnostics;
    using typename Base::PostProcessExternalSource;
    using typename Base::ComptonProjectionResult;
    using typename Base::ComptonCorrectionResult;
    using typename Base::ComptonCorrectionFailure;
    using Base::NumGroups;
public:
    explicit ComptonProcess(Owner &owner) : Base(owner)
    {}

    // Number of low groups printed by the LGDIAG diagnostic for cell 0 (STORM_COMPTON_LOWGROUP_DIAG; 0 = off).
    static std::size_t lowGroupDiagnosticGroups()
    {
        static std::size_t const groups = []
        {
            char const *value = std::getenv("STORM_COMPTON_LOWGROUP_DIAG");
            return value ? static_cast<std::size_t>(std::max(0, std::atoi(value))) : std::size_t{0};
        }();
        return groups;
    }

    bool thinnedEvents() const
    {
        return owner_.parameters_.comptonThinnedEvents();
    }

    bool exactEvents() const
    {
        return !this->thinnedEvents() && (owner_.parameters_.comptonExactEventEnergy ||
                                          owner_.parameters_.comptonEventMode == ComptonEventMode::Exact);
    }

    // Thinned mode needs a log-uniform grid with a nonempty common normalized
    // interval (ComptonThinned.hpp); checked once, the run stops otherwise.
    void checkThinnedGrid()
    {
        if(!this->thinnedEvents() || owner_.comptonThinnedGridChecked_)
        {
            return;
        }
        constexpr double gridTolerance = 1e-12;
        constexpr double placementMargin = 1e-9;
        owner_.comptonThinnedGrid_ = compton_thinned::BuildGrid<NumGroups>(
            owner_.energyBoundaries_, owner_.comptonGroupCenters_, gridTolerance, placementMargin);
        if(!owner_.comptonThinnedGrid_.valid)
        {
            StormError eo("Thinned Compton events need a log-uniform group grid with arithmetic centres");
            eo.addEntry("Reason", std::string(owner_.comptonThinnedGrid_.reason));
            eo.addEntry("Lowest boundary", owner_.energyBoundaries_[0]);
            eo.addEntry("Highest boundary", owner_.energyBoundaries_[NumGroups]);
            throw eo;
        }
        if(!(owner_.parameters_.comptonLimiterFraction > 0.0) || owner_.parameters_.comptonLimiterFraction > 1.0)
        {
            StormError eo("comptonLimiterFraction must lie in (0, 1]");
            eo.addEntry("comptonLimiterFraction", owner_.parameters_.comptonLimiterFraction);
            throw eo;
        }
        owner_.comptonThinnedGridChecked_ = true;
    }

    void precomputeComptonData(double sourceDt)
    {

            owner_.comptonRiskPrecomputeDt_ = sourceDt;
            owner_.comptonGroupCenters_ = owner_.opacity_->getEnergyCenters(owner_.energyBoundaries_);
            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                owner_.comptonGroupWidths_[g] = owner_.energyBoundaries_[g + 1] - owner_.energyBoundaries_[g];
                if(!std::isfinite(owner_.comptonGroupCenters_[g]) ||
                   !std::isfinite(owner_.comptonGroupWidths_[g]) ||
                   owner_.comptonGroupWidths_[g] <= 0.0)
                {
                    StormError eo("RadiationIMC Compton groups have invalid centers or widths");
                    eo.addEntry("Group", g);
                    eo.addEntry("Center", owner_.comptonGroupCenters_[g]);
                    eo.addEntry("Width", owner_.comptonGroupWidths_[g]);
                    throw eo;
                }
            }

            if(!owner_.parameters_.withCompton)
            {
                owner_.comptonData_.clear();
                return;
            }
            this->checkThinnedGrid();
            owner_.initializeComptonMatrixGenerator();
            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            owner_.comptonData_.assign(Ncells, ComptonCellData{});
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                CellT const &cell = owner_.cells_[i];
                ComptonCellData &data = owner_.comptonData_[i];
                data.volume = owner_.componentGrid().GetVolume(i);
                data.temperature = cell.temperature;
                data.Um = units::arad * boost::math::pow<4>(cell.temperature);
                const auto &tracers = owner_.traits_.tracers(cell);
                const auto &tracerNames = owner_.traits_.tracerNames(cell);
                data.cv = owner_.eos_->dT2cv(
                    owner_.density(i), cell.temperature, tracers, tracerNames);
                if(!std::isfinite(data.cv) || data.cv <= 0.0)
                {
                    StormError eo("RadiationIMC Compton precompute requires positive heat capacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Heat capacity", data.cv);
                    throw eo;
                }
                data.beta = 4.0 * units::arad *
                    boost::math::pow<3>(cell.temperature) / data.cv;

                double planckIntegralTotal = 0.0;
                double const kT = units::k_boltz * cell.temperature;
                if(kT > 0.0 && std::isfinite(kT))
                {
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        double const lower = owner_.energyBoundaries_[group] / kT;
                        double const upper = owner_.energyBoundaries_[group + 1] / kT;
                        data.planckFraction[group] =
                            planck_integral::planck_integral(lower, upper);
                        planckIntegralTotal += data.planckFraction[group];
                    }
                }

                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(planckIntegralTotal > 0.0)
                    {
                        data.planckFraction[group] /= planckIntegralTotal;
                    }
                    else
                    {
                        data.planckFraction[group] = 0.0;
                    }
                    double absorptionOpacity = owner_.opacity_->CalcAbsorptionOpacity(
                        cell, owner_.comptonGroupCenters_[group]);
                    if(!std::isfinite(absorptionOpacity) || absorptionOpacity < 0.0)
                    {
                        StormError eo("RadiationIMC Compton precompute received invalid absorption opacity");
                        eo.addEntry("Cell index", i);
                        eo.addEntry("Group", group);
                        eo.addEntry("Absorption opacity", absorptionOpacity);
                        throw eo;
                    }
                    data.absorptionOpacity[group] = absorptionOpacity;
                    data.planckOpacity += absorptionOpacity * data.planckFraction[group];
                    data.oldRadiationEnergy[group] = std::max(
                        0.0,
                        owner_.traits_.groupEnergyPerMass(cell, group) * owner_.density(i));
                }
                owner_.planckOpacities_[i] = data.planckOpacity;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    data.baseSourceFraction[group] = data.planckOpacity > 0.0
                        ? data.absorptionOpacity[group] *
                            data.planckFraction[group] / data.planckOpacity
                        : 0.0;
                }
                data.planckCdf = owner_.buildSafeComptonCdf(data.planckFraction);
                data.baseSourceCdf =
                    owner_.buildSafeComptonCdf(data.baseSourceFraction);
                data.groupCenters = owner_.comptonGroupCenters_;
                data.groupWidths = owner_.comptonGroupWidths_;

                ComptonOccupationMode occupationMode =
                    owner_.parameters_.comptonUseInduced
                        ? ComptonOccupationMode::RadiationField
                        : ComptonOccupationMode::Zero;
                owner_.buildComptonMatricesForCell(
                    cell, i, occupationMode, data);
                owner_.recomputeComptonContractions(data);
                double gamma = 1.0;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                       !owner_.parameters_.staticScatterers)
                    {
                        gamma = 1.0 / std::sqrt(
                            1.0 - ScalarProd(cell.velocity, cell.velocity) *
                            owner_.inverseLightSpeedSquared());
                    }
                }
                // The gas temperature evolves in its proper time, dtau = dt / gamma,
                // as in the non-Compton Fleck factor.
                double const cdtEff = owner_.lightSpeed() * sourceDt / gamma;
                double denominator = 1.0 + data.beta * cdtEff * data.Gamma;
                if((denominator <= 0.0 || data.Upsilon < 0.0) &&
                   owner_.parameters_.comptonAllowNZeroFallback)
                {
                    ComptonOccupationMode fallbackMode = ComptonOccupationMode::Zero;
                    if(data.Upsilon < 0.0 &&
                       owner_.parameters_.comptonUseInduced &&
                       owner_.parameters_.comptonInducedMode ==
                           ComptonInducedMode::AdaptivePlanckFallback &&
                       data.planckOpacity * owner_.lightSpeed() * sourceDt >= 1.0)
                    {
                        fallbackMode = ComptonOccupationMode::PlanckFunction;
                    }
                    owner_.buildComptonMatricesForCell(
                        cell, i, fallbackMode, data);
                    data.useNZero = fallbackMode == ComptonOccupationMode::Zero;
                    data.usePlanckInduced =
                        fallbackMode == ComptonOccupationMode::PlanckFunction;
                    owner_.recomputeComptonContractions(data);
                    denominator = 1.0 + data.beta * cdtEff * data.Gamma;
                }
                if(!std::isfinite(denominator) || denominator <= 0.0)
                {
                    StormError eo("Compton Fleck denominator is nonpositive");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Denominator", denominator);
                    eo.addEntry("Gamma", data.Gamma);
                    eo.addEntry("Upsilon", data.Upsilon);
                    throw eo;
                }
                data.fleck = 1.0 / denominator;
                if(!std::isfinite(data.fleck) || data.fleck < 0.0 || data.fleck > 1.0)
                {
                    StormError eo("Invalid Compton-modified Fleck factor");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Fleck", data.fleck);
                    eo.addEntry("Gamma", data.Gamma);
                    eo.addEntry("Upsilon", data.Upsilon);
                    eo.addEntry("Planck opacity", data.planckOpacity);
                    throw eo;
                }
                data.betaCdtF = data.beta * cdtEff * data.fleck;
                if(std::abs(data.Gamma) > 1e-200)
                {
                    data.betaCdtF = (1.0 - data.fleck) / data.Gamma;
                }
                owner_.factorFleck_[i] = data.fleck;
                data.sourceCdt = owner_.lightSpeed() * sourceDt;
                if(this->thinnedEvents())
                {
                    this->limitThinnedContraction(data, sourceDt);
                }
                owner_.buildComptonEventData(data);
                owner_.buildComptonSources(sourceDt, data);
                owner_.computeComptonRiskForCell(sourceDt, data);
                data.active = true;
            }
    }

    void initializeComptonMatrixGenerator()
    {

            if(owner_.comptonMatrixGen_)
            {
                return;
            }
        #ifndef STORM_WITH_COMPTON
            throw StormError(
                "RadiationIMC requested Compton support, but STORM was built without CMMC");
        #else
            std::vector<double> centers(owner_.comptonGroupCenters_.begin(), owner_.comptonGroupCenters_.end());
            std::vector<double> boundaries(owner_.energyBoundaries_.begin(), owner_.energyBoundaries_.end());
            if(owner_.parameters_.comptonTableSource == ComptonTableSource::MACE)
            {
                // MACE tables carry no top-group in-group energy term (plan v10 keeps it off).
                if(owner_.parameters_.comptonTopGroupTerm)
                {
                    throw StormError("comptonTopGroupTerm needs CMMC tables; MACE tables have no top-group term");
                }
                MACEComptonTableOptions options;
                options.cacheDir = owner_.parameters_.comptonTableCacheDir;
                options.angleBins = owner_.parameters_.comptonTableAngleBins;
                options.exactSolver = owner_.parameters_.comptonTableSolver == ComptonTableSolver::Exact;
                options.threads = owner_.parameters_.comptonTableThreads;
                owner_.comptonMatrixGen_ = std::make_unique<ComptonTableBackend<NumGroups>>(
                    std::make_unique<MACEComptonBackendCore>(std::move(centers), std::move(boundaries), options));
            }
            else
            {
                auto core = std::make_unique<CMMCComptonBackendCore>(
                    std::move(centers),
                    std::move(boundaries),
                    owner_.parameters_.comptonMatrixSamples,
                    true,
                    1);
                core->SetTables(owner_.buildComptonTemperatures());
                owner_.comptonMatrixGen_ = std::make_unique<ComptonTableBackend<NumGroups>>(std::move(core));
            }
            owner_.comptonGroupsInitialized_ = true;
        #endif
    }

    std::vector<double> buildComptonTemperatures() const
    {

            std::vector<double> temperatures;
            temperatures.reserve(109);
            temperatures.push_back(0.0001 * units::kev_kelvin);
            temperatures.push_back(0.001 * units::kev_kelvin);
            temperatures.push_back(0.005 * units::kev_kelvin);
            for(std::size_t index = 0; index < 128; ++index)
            {
                double const exponent = -2.0 + 6.0 *
                    static_cast<double>(index) / 127.0;
                double const temperature =
                    std::pow(10.0, exponent) * units::kev_kelvin;
                if(temperature > 1000.0 * units::kev_kelvin)
                {
                    break;
                }
                temperatures.push_back(temperature);
            }
            return temperatures;
    }

    typename Owner::GroupCdf
    buildSafeComptonCdf(
        const GroupArray &weights) const
    {

            GroupCdf cdf{};
            double total = 0.0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                total += std::max(0.0, weights[group]);
                cdf[group + 1] = total;
            }
            if(!(total > 0.0) || !std::isfinite(total))
            {
                for(std::size_t group = 0; group <= NumGroups; ++group)
                {
                    cdf[group] = static_cast<double>(group) /
                        static_cast<double>(NumGroups);
                }
                return cdf;
            }
            for(double &value : cdf)
            {
                value /= total;
            }
            cdf[NumGroups] = 1.0;
            return cdf;
    }

    void buildComptonMatricesForCell(
        const CellT &cell,
        std::size_t cellIndex,
        ComptonOccupationMode occupationMode,
        ComptonCellData &data)
    {

            bool const usePlanckLTE =
                occupationMode == ComptonOccupationMode::PlanckFunction;
            double const lteTemperature = usePlanckLTE
                ? owner_.computeLteTemperature(cell, cellIndex) : cell.temperature;
            GroupArray ltePlanckFractions{};
            if(usePlanckLTE)
            {
                double const kT = units::k_boltz * lteTemperature;
                double total = 0.0;
                if(kT > 0.0)
                {
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        double const mass = planck_integral::planck_integral(
                            owner_.energyBoundaries_[group] / kT,
                            owner_.energyBoundaries_[group + 1] / kT);
                        ltePlanckFractions[group] =
                            (mass > 0.0 && std::isfinite(mass)) ? mass : 0.0;
                        total += ltePlanckFractions[group];
                    }
                }
                if(total > 0.0)
                {
                    for(double &fraction : ltePlanckFractions)
                    {
                        fraction /= total;
                    }
                }
            }
            double const lteRadiationEnergyDensity = usePlanckLTE
                ? units::arad * boost::math::pow<4>(lteTemperature) : 0.0;
            double const pi = 3.141592653589793238462643383279502884;
            double const occupationFactor = boost::math::pow<3>(owner_.lightSpeed()) /
                (8.0 * pi * units::planck_constant);
            // Occupation cap per group: max(100, factor nP_g(T_cap)), nP_g the
            // group-integrated Planck occupation at T_cap = max(T_e, T_rad), T_rad
            // the radiation-density temperature; optionally also a fixed nMax.
            GroupArray occupationCap{};
            occupationCap.fill(100.0);
            if(owner_.parameters_.comptonOccupationCapFactor > 0.0)
            {
                double radiationDensity = 0.0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    radiationDensity += std::max(0.0, data.oldRadiationEnergy[group]);
                }
                double const capTemperature = std::max(
                    cell.temperature, std::pow(radiationDensity / units::arad, 0.25));
                double const kTCap = units::k_boltz * capTemperature;
                double const capRadiationDensity = units::arad * boost::math::pow<4>(capTemperature);
                if(kTCap > 0.0 && std::isfinite(kTCap))
                {
                    GroupArray capFractions{};
                    double total = 0.0;
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        double const mass = planck_integral::planck_integral(
                            owner_.energyBoundaries_[group] / kTCap, owner_.energyBoundaries_[group + 1] / kTCap);
                        capFractions[group] = (mass > 0.0 && std::isfinite(mass)) ? mass : 0.0;
                        total += capFractions[group];
                    }
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        double const nu = owner_.comptonGroupCenters_[group] / units::planck_constant;
                        double const dnu = owner_.comptonGroupWidths_[group] / units::planck_constant;
                        double const planckOccupation = total > 0.0
                            ? occupationFactor * capFractions[group] / total * capRadiationDensity /
                                (boost::math::pow<3>(nu) * dnu)
                            : 0.0;
                        if(std::isfinite(planckOccupation))
                        {
                            occupationCap[group] = std::max(
                                100.0, owner_.parameters_.comptonOccupationCapFactor * planckOccupation);
                        }
                    }
                }
            }
            if(owner_.parameters_.comptonOccupationMax > 0.0)
            {
                for(double &cap : occupationCap)
                {
                    cap = std::min(cap, owner_.parameters_.comptonOccupationMax);
                }
            }
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const dnu = owner_.comptonGroupWidths_[group] /
                    units::planck_constant;
                double const nu = owner_.comptonGroupCenters_[group] /
                    units::planck_constant;
                double occupation = 0.0;
                if(occupationMode == ComptonOccupationMode::RadiationField)
                {
                    occupation = occupationFactor * data.oldRadiationEnergy[group] /
                        (boost::math::pow<3>(nu) * dnu);
                }
                else if(usePlanckLTE)
                {
                    occupation = occupationFactor *
                        ltePlanckFractions[group] * lteRadiationEnergyDensity /
                        (boost::math::pow<3>(nu) * dnu);
                }
                data.occupation[group] = std::clamp(
                    std::isfinite(occupation) ? occupation : 0.0, 0.0, occupationCap[group]);
            }

            // A backend that clamps itself (MACE) gets the physical temperature, so its clamp and zero derivative
            // outside the table apply; CMMC needs a temperature inside its table.
            double const temperature = owner_.comptonMatrixGen_->ClampsTemperature()
                ? std::max(lteTemperature, std::numeric_limits<double>::min())
                : std::clamp(lteTemperature,
                             std::max(0.0001 * units::kev_kelvin, owner_.comptonMatrixGen_->GetMinimumTemperature()),
                             owner_.comptonMatrixGen_->GetMaximumTemperature() * 0.9999);
            double lastGroupUpScatter = 0.0;
            double lastGroupDownScatter = 0.0;
            owner_.comptonMatrixGen_->GetTauMatrix(
                temperature,
                std::max(0.0, owner_.density(cellIndex)),
                1.0,
                1.0,
                data.tau,
                data.dtau_dUm,
                lastGroupUpScatter,
                lastGroupDownScatter);
            ApplyComptonPairThrottle(data.occupation, owner_.parameters_.comptonPairThrottleOccupation,
                                     data.tau, data.dtau_dUm);
            data.rates = data.tau;
            data.derivative = data.dtau_dUm;
            data.S = GroupMatrix{};
            data.dSdUm = GroupMatrix{};
            for(std::size_t source = 0; source < NumGroups; ++source)
            {
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    if(source + 1 == NumGroups && target + 1 == NumGroups)
                    {
                        if(!owner_.parameters_.comptonTopGroupTerm)
                        {
                            continue;
                        }
                        data.S[source][source] +=
                            (lastGroupUpScatter - lastGroupDownScatter) *
                            (1.0 + data.occupation[source]);
                        data.dSdUm[source][source] +=
                            data.dtau_dUm[source][source] *
                            (1.0 + data.occupation[source]);
                        continue;
                    }
                    // In-group scattering moves no energy between groups; adding and then subtracting
                    // tau_gg (1 + n_g) would only cost precision against the (possibly throttled) out rates.
                    if(source == target)
                    {
                        continue;
                    }
                    double const inFactor =
                        owner_.comptonGroupCenters_[source] /
                        owner_.comptonGroupCenters_[target] *
                        (1.0 + data.occupation[source]);
                    data.S[target][source] +=
                        data.tau[target][source] * inFactor;
                    data.dSdUm[target][source] +=
                        data.dtau_dUm[target][source] * inFactor;
                    double const outFactor = 1.0 + data.occupation[target];
                    data.S[source][source] -=
                        data.tau[source][target] * outFactor;
                    data.dSdUm[source][source] -=
                        data.dtau_dUm[source][target] * outFactor;
                }
            }
            double const derivativeScale =
                owner_.parameters_.comptonLinearizeTemperature
                ? 1.0 / (4.0 * units::arad * boost::math::pow<3>(lteTemperature))
                : 0.0;
            for(std::size_t source = 0; source < NumGroups; ++source)
            {
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    data.dSdUm[source][target] *= derivativeScale;
                }
            }
    }

    double computeLteTemperature(
        const CellT &cell, std::size_t cellIndex) const
    {

            double const rho = owner_.density(cellIndex);
            if(!(rho > 0.0) || !std::isfinite(rho))
            {
                return cell.temperature;
            }
            double radiationSpecificEnergy = 0.0;
            if constexpr(
                radiation_imc_detail::has_member_radiation_energy<CellT>::value)
            {
                radiationSpecificEnergy = std::max(0.0, cell.Erad);
            }
            double const totalSpecificEnergy =
                owner_.specificInternalEnergy(cellIndex) + radiationSpecificEnergy;
            if(!(totalSpecificEnergy > 0.0) || !std::isfinite(totalSpecificEnergy))
            {
                return cell.temperature;
            }

            double const maximumTemperature = owner_.comptonMatrixGen_
                ? owner_.comptonMatrixGen_->GetMaximumTemperature() * 0.9999
                : std::max(cell.temperature, 1.0);
            double const radiationTemperature = std::pow(
                std::max(radiationSpecificEnergy, 0.0) * rho / units::arad,
                0.25);
            double temperature = std::clamp(
                std::max(cell.temperature, radiationTemperature),
                1.0e-30, maximumTemperature);
            auto const &tracers = owner_.traits_.tracers(cell);
            auto const &tracerNames = owner_.traits_.tracerNames(cell);
            auto matterSpecificEnergy = [&](double candidateTemperature) -> double
            {
                using TracersType = std::decay_t<decltype(tracers)>;
                using TracerNamesType = std::decay_t<decltype(tracerNames)>;
                if constexpr(radiation_imc_detail::has_dT2e<
                                 EOST, TracersType, TracerNamesType>::value)
                {
                    return owner_.eos_->dT2e(
                        rho, candidateTemperature, tracers, tracerNames);
                }
                else
                {
                    throw StormError(
                        "Planck-function Compton occupation requires an EOS dT2e method");
                }
            };
            for(int iteration = 0; iteration < 50; ++iteration)
            {
                double const matterEnergy = matterSpecificEnergy(temperature);
                double const radiationEnergy = units::arad *
                    boost::math::pow<4>(temperature) / rho;
                double const residual =
                    matterEnergy + radiationEnergy - totalSpecificEnergy;
                double const scale = std::max(
                    std::abs(totalSpecificEnergy),
                    std::numeric_limits<double>::min());
                if(std::abs(residual) <= 1.0e-10 * scale)
                {
                    break;
                }
                double const cv = owner_.eos_->dT2cv(
                    rho, temperature, tracers, tracerNames);
                double const derivative = cv + 4.0 * units::arad *
                    boost::math::pow<3>(temperature) / rho;
                if(!(derivative > 0.0) || !std::isfinite(derivative))
                {
                    break;
                }
                double const candidate = temperature - residual / derivative;
                if(!std::isfinite(candidate))
                {
                    break;
                }
                temperature = std::clamp(
                    candidate, 1.0e-30, maximumTemperature);
            }
            return temperature;
    }

    void recomputeComptonContractions(
        ComptonCellData &data) const
    {

            data.Upsilon = 0.0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                data.D[group] = 0.0;
                for(std::size_t source = 0; source < NumGroups; ++source)
                {
                    data.D[group] += data.dSdUm[source][group] *
                        data.oldRadiationEnergy[source];
                }
                data.Upsilon += data.D[group];
                data.M[group] = data.absorptionOpacity[group] *
                    data.planckFraction[group] + data.D[group];
            }
            for(std::size_t source = 0; source < NumGroups; ++source)
            {
                data.rowS[source] = 0.0;
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    data.rowS[source] += data.S[source][target];
                }
                data.Lambda[source] = data.absorptionOpacity[source] -
                    data.rowS[source];
            }
            data.Gamma = data.planckOpacity + data.Upsilon;
    }

    void buildComptonSources(
        double sourceDt,
        ComptonCellData &data) const
    {

            double const cdt = owner_.lightSpeed() * sourceDt;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const kgbg = data.absorptionOpacity[group] *
                    data.planckFraction[group];
                data.Bbase[group] = data.volume * cdt * data.fleck *
                    kgbg * data.Um;
                if(data.planckOpacity > 0.0)
                {
                    // betaCdtF = (1 - f) / Gamma is beta c dt f with the step used in
                    // the Fleck factor, so the corrections sum to zero over groups.
                    data.Bcorr[group] = data.volume * cdt * data.planckOpacity *
                        data.Um *
                        ((kgbg / data.planckOpacity) *
                         (1.0 - data.fleck -
                          data.planckOpacity * data.betaCdtF) -
                         data.betaCdtF * data.D[group]);
                }
                else
                {
                    data.Bcorr[group] = 0.0;
                }
                data.Btotal[group] = data.Bbase[group] + data.Bcorr[group];
                data.Bpos[group] = std::max(0.0, data.Btotal[group]);
                data.Bres[group] = data.Btotal[group] - data.Bpos[group];
                data.residualSource[group] = data.Bcorr[group];
            }
    }

    void buildComptonEventData(
        ComptonCellData &data) const
    {

            data.segmentKernel = GroupMatrix{};
            data.residualKernel = GroupMatrix{};
            data.Ktotal = GroupMatrix{};
            data.implicitKernel = GroupMatrix{};
            data.implicitEventRateMatrix = GroupMatrix{};
            for(std::size_t source = 0; source < NumGroups; ++source)
            {
                GroupArray weights{};
                double outRate = 0.0;
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    if(target == source)
                    {
                        continue;
                    }
                    weights[target] = std::max(
                        0.0,
                        data.tau[source][target] *
                        (1.0 + data.occupation[target]));
                    outRate += weights[target];
                }
                data.outRate[source] = outRate;
                data.comptonOutRate[source] = outRate;
                // An in-group scattering only changes direction. For such a scattering, induced scattering into
                // and out of a direction cancel exactly, so the rate has no (1 + n) factor.
                data.comptonInGroupRate[source] = std::max(0.0, data.tau[source][source]);
                data.targetCdf[source] = owner_.buildSafeComptonCdf(weights);
                data.implicitEventCdf[source] = data.targetCdf[source];
                double baseOpacity = 0.0;
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    baseOpacity += data.betaCdtF * data.absorptionOpacity[source] *
                        data.absorptionOpacity[target] *
                        data.planckFraction[target];
                }
                data.baseEffectiveOpacity[source] = baseOpacity;

                double mu = 1.0;
                if(outRate > 0.0)
                {
                    mu = 0.0;
                    for(std::size_t target = 0; target < NumGroups; ++target)
                    {
                        if(target == source)
                        {
                            continue;
                        }
                        mu += weights[target] / outRate *
                            owner_.comptonGroupCenters_[target] /
                            std::max(owner_.comptonGroupCenters_[source],
                                     std::numeric_limits<double>::min());
                    }
                }
                data.comptonMu[source] = mu;
                data.comptonMh[source] =
                    1.0 + data.fleck * (mu - 1.0);
                data.thinnedJumpProbability[source] = data.fleck;
                if(this->thinnedEvents())
                {
                    // Optional non-jump throttle; comptonMh becomes the expected
                    // multiplier of one sampled event, 1 + p (mu - 1).
                    compton_thinned::ThrottledRate const throttled = compton_thinned::ThrottleEvents(
                        outRate, data.fleck, data.sourceCdt, owner_.parameters_.comptonEventThrottle);
                    data.comptonOutRate[source] = throttled.rate;
                    data.thinnedJumpProbability[source] = throttled.jumpProbability;
                    data.comptonMh[source] = 1.0 + throttled.jumpProbability * (mu - 1.0);
                }
                // Second moment of the energy ratio, for the ledger variance
                // w^2 [f V2 - f^2 (mu - 1)^2] of a Thinned event.
                data.thinnedV2[source] = 0.0;
                if(outRate > 0.0)
                {
                    for(std::size_t target = 0; target < NumGroups; ++target)
                    {
                        if(target != source)
                        {
                            double const ratio = owner_.comptonGroupCenters_[target] /
                                owner_.comptonGroupCenters_[source];
                            data.thinnedV2[source] += weights[target] / outRate * (ratio - 1.0) * (ratio - 1.0);
                        }
                    }
                }
                data.meanEnergyRatio[source] = mu;
                data.modifiedFleck[source] = data.fleck;
                data.implicitEventRate[source] = outRate;
                data.implicitDiagonalCorrection[source] =
                    outRate * (data.comptonMh[source] - 1.0);

                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    double const kgbg = data.absorptionOpacity[target] *
                        data.planckFraction[target];
                    double const kTotal = data.S[source][target] +
                        data.betaCdtF * data.M[target] * data.Lambda[source];
                    double const hBase = data.betaCdtF *
                        data.absorptionOpacity[source] * kgbg;
                    data.segmentKernel[source][target] = kTotal - hBase;
                    data.Ktotal[source][target] =
                        (source == target ? -data.absorptionOpacity[source] : 0.0) +
                        kTotal;
                    double const kImc = source == target
                        ? -data.absorptionOpacity[source] +
                            (1.0 - data.fleck) *
                            data.absorptionOpacity[source] *
                            data.baseSourceFraction[source]
                        : (1.0 - data.fleck) *
                            data.absorptionOpacity[source] *
                            data.baseSourceFraction[target];
                    double const eventMultiplier = this->exactEvents()
                        ? owner_.comptonGroupCenters_[target] /
                            owner_.comptonGroupCenters_[source]
                        : data.comptonMh[source];
                    double eventKernel = source == target
                        ? -outRate
                        : (outRate > 0.0
                            ? outRate * weights[target] / outRate *
                                eventMultiplier
                            : 0.0);
                    if(this->thinnedEvents())
                    {
                        // Thinned jumps realize H = f S exactly: a fraction f of
                        // the events moves the packet with the centre ratio.
                        eventKernel = source == target
                            ? -data.fleck * outRate
                            : data.fleck * weights[target] *
                                owner_.comptonGroupCenters_[target] / owner_.comptonGroupCenters_[source];
                    }
                    data.implicitKernel[source][target] = eventKernel;
                    data.implicitEventRateMatrix[source][target] = eventKernel;
                    data.residualKernel[source][target] =
                        data.Ktotal[source][target] - kImc - eventKernel;
                }
            }
    }

    void computeComptonRiskForCell(
        double fullDt,
        ComptonCellData &data) const
    {

            GroupArray rhs{};
            GroupArray predicted{};
            GroupMatrix residualMatrix{};
            double totalOldExtensive = 0.0;

            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const oldExtensive =
                    std::max(0.0, data.oldRadiationEnergy[group]) * data.volume;
                rhs[group] = oldExtensive + data.Bcorr[group];
                totalOldExtensive += oldExtensive;
                data.riskScore[group] = 0.0;
                data.riskTargetPackets[group] = 0;
            }
            if(!(totalOldExtensive > 0.0))
            {
                return;
            }

            for(std::size_t row = 0; row < NumGroups; ++row)
            {
                for(std::size_t column = 0; column < NumGroups; ++column)
                {
                    residualMatrix[row][column] =
                        (row == column ? 1.0 : 0.0) -
                        fullDt * owner_.lightSpeed() *
                            data.residualKernel[column][row];
                }
            }

            bool const solved = this->solveComptonGroupSystem(
                residualMatrix, rhs, predicted);
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const oldExtensive =
                    std::max(0.0, data.oldRadiationEnergy[group]) * data.volume;
                double const groupFloor =
                    std::max(1e-10 * totalOldExtensive, 1.0);
                double const scale = std::max(oldExtensive, groupFloor);
                double score = std::abs(data.Bcorr[group]) / scale;
                if(solved)
                {
                    double const depletion = oldExtensive - predicted[group];
                    if(depletion > 0.0)
                    {
                        score = std::max(score, depletion / scale);
                    }
                    if(predicted[group] < 0.0)
                    {
                        score = std::max(
                            score, 1.0 + std::abs(predicted[group]) / scale);
                    }
                }
                else
                {
                    score = std::max(score, 2.0);
                }

                if(oldExtensive <= 1e-8 * totalOldExtensive && score < 10.0)
                {
                    continue;
                }
                if(score < 0.5)
                {
                    continue;
                }

                data.riskScore[group] = score;
                data.riskTargetPackets[group] = score >= 10.0 ? 96
                    : score >= 3.0 ? 64
                    : score >= 1.0 ? 32 : 16;
            }
    }

    std::size_t sampleComptonTarget(
        const ComptonCellData &data, std::size_t sourceGroup,
        MCParticle &particle)
    {

            if(sourceGroup >= NumGroups || !(data.outRate[sourceGroup] > 0.0))
            {
                return sourceGroup;
            }
            return owner_.sampleComptonCdf(
                data.targetCdf[sourceGroup], owner_.randomUnitOpen(particle));
    }

    void addComptonMaterialExchange(
        std::size_t cellIndex, double energy)
    {

            if(owner_.parameters_.noHydroFeedback || owner_.parameters_.postProcess.enabled)
            {
                return;
            }
            owner_.tallyMaterialEnergy(cellIndex, energy);
    }

    // Packets per group of a cell's Compton emission (base source), with the
    // risk top-up; shared by the source plan and the Thinned noise predictor.
    std::array<std::size_t, NumGroups> allocateComptonSourcePackets(const ComptonCellData &data) const
    {
            GroupArray sourceEnergy{};
            GroupArray fractional{};
            std::array<std::size_t, NumGroups> groupCounts{};
            double totalSourceEnergy = 0.0;
            std::size_t activeGroups = 0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                sourceEnergy[group] = std::max(0.0, data.Bbase[group]);
                totalSourceEnergy += sourceEnergy[group];
                if(sourceEnergy[group] > 0.0)
                {
                    ++activeGroups;
                }
            }
            std::size_t const packetCount = std::max(
                owner_.parameters_.newPhotonsPerCell, activeGroups);
            if(packetCount == 0 || !(totalSourceEnergy > 0.0))
            {
                return groupCounts;
            }

            std::size_t allocated = 0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                if(sourceEnergy[group] > 0.0)
                {
                    groupCounts[group] = 1;
                    ++allocated;
                }
            }
            std::size_t const remainingBudget = packetCount - allocated;
            std::size_t extraAllocated = 0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                if(!(sourceEnergy[group] > 0.0))
                {
                    continue;
                }
                double const exactExtra = static_cast<double>(remainingBudget) *
                    sourceEnergy[group] / totalSourceEnergy;
                std::size_t const extra =
                    static_cast<std::size_t>(std::floor(exactExtra));
                groupCounts[group] += extra;
                fractional[group] = exactExtra - static_cast<double>(extra);
                extraAllocated += extra;
            }
            while(extraAllocated < remainingBudget)
            {
                std::size_t bestGroup = NumGroups;
                double bestFraction = -1.0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(sourceEnergy[group] > 0.0 &&
                       fractional[group] > bestFraction)
                    {
                        bestGroup = group;
                        bestFraction = fractional[group];
                    }
                }
                if(bestGroup == NumGroups)
                {
                    break;
                }
                ++groupCounts[bestGroup];
                fractional[bestGroup] = 0.0;
                ++extraAllocated;
            }

            std::size_t riskBudget = std::max<std::size_t>(8, packetCount / 4);
            std::array<std::size_t, NumGroups> riskOrder{};
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                riskOrder[group] = group;
            }
            std::sort(riskOrder.begin(), riskOrder.end(),
                [&](std::size_t left, std::size_t right)
                {
                    return data.riskScore[left] > data.riskScore[right];
                });
            for(std::size_t order = 0;
                order < NumGroups && riskBudget > 0; ++order)
            {
                std::size_t const group = riskOrder[order];
                std::size_t const target = data.riskTargetPackets[group];
                if(target == 0 || !(sourceEnergy[group] > 0.0) ||
                   groupCounts[group] >= target)
                {
                    continue;
                }
                std::size_t const add =
                    std::min(target - groupCounts[group], riskBudget);
                groupCounts[group] += add;
                riskBudget -= add;
            }
            return groupCounts;
    }

    // Compton decides what each cell emits: the source energy per group,
    // the packets per group and their group-centre frequencies.  It adds one
    // plan entry per (cell, group); IMCSourceProcess emits the plan and
    // debits every cell by what its packets carry, as for thermal emission.
    void buildComptonSourcePlan(source::Plan &plan) const
    {

            plan.fixedFrequencies = true;
            // Thinned mode: the clamp would change w/nu of edge-group packets
            // born in moving cells (plan 1b, R1).
            plan.clampLabFrequency = !this->thinnedEvents();
            for(std::size_t cellIndex = 0; cellIndex < owner_.comptonData_.size(); ++cellIndex)
            {
                ComptonCellData const &data = owner_.comptonData_[cellIndex];
                std::array<std::size_t, NumGroups> const groupCounts = this->allocateComptonSourcePackets(data);
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    std::size_t const groupPackets = groupCounts[group];
                    double const sourceEnergy = std::max(0.0, data.Bbase[group]);
                    if(groupPackets == 0 || !(sourceEnergy > 0.0))
                    {
                        continue;
                    }
                    plan.addEntry(cellIndex, groupPackets,
                        sourceEnergy / static_cast<double>(groupPackets),
                        this->frequencyForComptonGroup(group));
                }
            }
    }

    std::size_t sampleComptonCdf(
        const GroupCdf &cdf, double random) const
    {

            double const value = std::clamp(
                random, 0.0, std::nextafter(1.0, 0.0));
            auto iterator = std::upper_bound(cdf.begin(), cdf.end(), value);
            if(iterator == cdf.begin())
            {
                return 0;
            }
            std::size_t group = static_cast<std::size_t>(
                std::distance(cdf.begin(), iterator) - 1);
            return std::min(group, NumGroups - 1);
    }

    double frequencyForComptonGroup(
        std::size_t group) const
    {

            if(group >= NumGroups)
            {
                throw StormError("RadiationIMC received an invalid Compton target group");
            }
            double frequency = owner_.comptonGroupCenters_[group];
            owner_.clampFrequencyToBounds(frequency);
            return frequency;
    }

    double sumComptonGroups(
        const GroupArray &values) const
    {

            return this->compensatedSumComptonGroups(values);
    }

    double compensatedSumComptonGroups(
        const GroupArray &values) const
    {

            double sum = 0.0;
            double compensation = 0.0;
            for(double const value : values)
            {
                double const corrected = value - compensation;
                double const next = sum + corrected;
                compensation = (next - sum) - corrected;
                sum = next;
            }
            return sum;
    }

    const char *comptonCorrectionFailureName(
        ComptonCorrectionFailure failure) const
    {

            switch(failure)
            {
                case ComptonCorrectionFailure::None:
                    return "None";
                case ComptonCorrectionFailure::DirectLinearSolveFailed:
                    return "DirectLinearSolveFailed";
                case ComptonCorrectionFailure::DirectNegativeMass:
                    return "DirectNegativeMass";
                case ComptonCorrectionFailure::DirectMaterialCap:
                    return "DirectMaterialCap";
                case ComptonCorrectionFailure::DirectProjectedResidual:
                    return "DirectProjectedResidual";
                case ComptonCorrectionFailure::AdaptiveLinearSolveFailed:
                    return "AdaptiveLinearSolveFailed";
                case ComptonCorrectionFailure::AdaptiveMaximumSubsteps:
                    return "AdaptiveMaximumSubsteps";
                case ComptonCorrectionFailure::AdaptiveMaximumRejectedTrials:
                    return "AdaptiveMaximumRejectedTrials";
                case ComptonCorrectionFailure::AdaptiveFractionBelowMinimum:
                    return "AdaptiveFractionBelowMinimum";
                case ComptonCorrectionFailure::AdaptiveNoProgress:
                    return "AdaptiveNoProgress";
                case ComptonCorrectionFailure::NonFiniteState:
                    return "NonFiniteState";
                case ComptonCorrectionFailure::InvalidEnergyBudget:
                    return "InvalidEnergyBudget";
                case ComptonCorrectionFailure::EnergyClosureFailure:
                    return "EnergyClosureFailure";
            }
            return "Unknown";
    }

    double minComptonGroup(
        const GroupArray &values) const
    {

            return *std::min_element(values.begin(), values.end());
    }

    double maxAbsComptonGroup(
        const GroupArray &values) const
    {

            double maximum = 0.0;
            for(double const value : values)
            {
                maximum = std::max(maximum, std::abs(value));
            }
            return maximum;
    }

    double normComptonGroups(
        const GroupArray &values) const
    {

            double sumSquares = 0.0;
            for(double const value : values)
            {
                sumSquares += value * value;
            }
            return std::sqrt(sumSquares);
    }

    typename Owner::GroupArray
    multiplyComptonMatrix(
        const GroupMatrix &matrix,
        const GroupArray &values) const
    {

            GroupArray result{};
            for(std::size_t row = 0; row < NumGroups; ++row)
            {
                for(std::size_t column = 0; column < NumGroups; ++column)
                {
                    result[row] += matrix[row][column] * values[column];
                }
            }
            return result;
    }

    double relativeComptonResidual(
        const GroupMatrix &matrix,
        const GroupArray &solution,
        const GroupArray &rhs,
        double scale) const
    {

            GroupArray residual = this->multiplyComptonMatrix(matrix, solution);
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                residual[group] -= rhs[group];
            }
            return this->normComptonGroups(residual) /
                std::max(1.0, std::max(this->normComptonGroups(rhs), scale));
    }

    typename Owner::ComptonProjectionResult
    projectNonnegativeConservative(
        const GroupArray &candidate,
        double targetTotal,
        double energyScale,
        double perGroupNegativeTolerance,
        double totalNegativeTolerance) const
    {

            ComptonProjectionResult result;
            result.inputTotal = this->compensatedSumComptonGroups(candidate);
            result.targetTotal = targetTotal;
            if(!std::isfinite(result.inputTotal) ||
               !std::isfinite(targetTotal) ||
               targetTotal < -totalNegativeTolerance)
            {
                return result;
            }
            if(targetTotal < 0.0)
            {
                result.targetTotal = 0.0;
            }

            GroupArray positive{};
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const value = candidate[group];
                if(!std::isfinite(value))
                {
                    return result;
                }
                if(value < 0.0)
                {
                    double const negative = -value;
                    result.negativeMass += negative;
                    if(negative > result.worstNegative)
                    {
                        result.worstNegative = negative;
                        result.worstNegativeGroup = group;
                    }
                }
                positive[group] = std::max(0.0, value);
            }
            double const positiveTotal =
                this->compensatedSumComptonGroups(positive);

            if(result.worstNegative > perGroupNegativeTolerance ||
               result.negativeMass > totalNegativeTolerance)
            {
                return result;
            }
            if(result.targetTotal > 0.0 && !(positiveTotal > 0.0))
            {
                return result;
            }

            result.usedProjection = result.negativeMass > 0.0 ||
                std::abs(result.inputTotal - result.targetTotal) >
                    1e-14 * std::max(1.0, energyScale);
            if(result.targetTotal > 0.0)
            {
                // Project onto the nonnegative simplex in the Euclidean norm:
                //
                //     minimize ||x - candidate||_2
                //     subject to x_g >= 0 and sum_g x_g = targetTotal.
                //
                // The previous clip-and-rescale operation changed every positive bin
                // multiplicatively.  When many small negative bins were present, that
                // unnecessarily enlarged the correction residual and forced dozens of
                // path-dependent substeps.  The simplex projection is the smallest
                // conservative nonnegative change to the direct solution.
                GroupArray sorted = candidate;
                std::sort(sorted.begin(), sorted.end(),
                    [](double left, double right)
                    {
                        return left > right;
                    });
                static constexpr GroupArray inverseActiveCounts = []()
                {
                    GroupArray values{};
                    for(std::size_t index = 0; index < NumGroups; ++index)
                    {
                        values[index] =
                            1.0 / static_cast<double>(index + 1);
                    }
                    return values;
                }();
                double cumulative = 0.0;
                double theta = 0.0;
                std::size_t active = 0;
                for(std::size_t index = 0; index < NumGroups; ++index)
                {
                    cumulative += sorted[index];
                    // Keep this as a multiply by a compile-time reciprocal.  Intel LLVM
                    // otherwise vectorizes the prefix scan with a speculative zero
                    // divisor in a lane that is blended out later; trapping floating
                    // point environments still raise SIGFPE on that discarded lane.
                    double const trialTheta =
                        (cumulative - result.targetTotal) *
                        inverseActiveCounts[index];
                    if(sorted[index] > trialTheta)
                    {
                        active = index + 1;
                        theta = trialTheta;
                    }
                }
                if(active == 0 || !std::isfinite(theta))
                {
                    return result;
                }
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    result.endpoint[group] = std::max(0.0, candidate[group] - theta);
                }

                // Close the conserved radiation total at roundoff level.  Adjusting
                // the largest active component is the least fragile option because it
                // cannot turn a marginally active group negative.
                double simplexTotal =
                    this->compensatedSumComptonGroups(result.endpoint);
                if(!std::isfinite(simplexTotal))
                {
                    return result;
                }
                double const closure = result.targetTotal - simplexTotal;
                auto const largest = std::max_element(
                    result.endpoint.begin(), result.endpoint.end());
                if(largest == result.endpoint.end() ||
                   !std::isfinite(*largest + closure) ||
                   *largest + closure < 0.0)
                {
                    return result;
                }
                *largest += closure;

                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const scale = std::max(
                        {1.0, std::abs(candidate[group]), std::abs(result.endpoint[group])});
                    result.maximumRelativeChange = std::max(
                        result.maximumRelativeChange,
                        std::abs(result.endpoint[group] - candidate[group]) / scale);
                }
            }
            else
            {
                result.endpoint.fill(0.0);
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    result.maximumRelativeChange = std::max(
                        result.maximumRelativeChange,
                        std::abs(candidate[group]));
                }
            }

            double const projectedTotal =
                this->compensatedSumComptonGroups(result.endpoint);
            if(!std::isfinite(projectedTotal) ||
               std::abs(projectedTotal - result.targetTotal) >
                   1e-12 * std::max(1.0, energyScale))
            {
                return result;
            }
            result.success = true;
            return result;
    }

    bool solveComptonGroupSystem(
        GroupMatrix matrix,
        GroupArray rhs,
        GroupArray &solution) const
    {

            for(std::size_t row = 0; row < NumGroups; ++row)
            {
                for(std::size_t column = 0; column < NumGroups; ++column)
                {
                    if(!std::isfinite(matrix[row][column]))
                    {
                        return false;
                    }
                }
                if(!std::isfinite(rhs[row]))
                {
                    return false;
                }
            }
            for(std::size_t column = 0; column < NumGroups; ++column)
            {
                std::size_t pivot = column;
                double pivotMagnitude = std::abs(matrix[column][column]);
                for(std::size_t row = column + 1; row < NumGroups; ++row)
                {
                    double const candidate = std::abs(matrix[row][column]);
                    if(candidate > pivotMagnitude)
                    {
                        pivot = row;
                        pivotMagnitude = candidate;
                    }
                }
                if(!(pivotMagnitude > 1e-200) || !std::isfinite(pivotMagnitude))
                {
                    return false;
                }
                if(pivot != column)
                {
                    std::swap(matrix[pivot], matrix[column]);
                    std::swap(rhs[pivot], rhs[column]);
                }
                double const pivotValue = matrix[column][column];
                for(std::size_t row = column + 1; row < NumGroups; ++row)
                {
                    double const factor = matrix[row][column] / pivotValue;
                    if(factor == 0.0)
                    {
                        continue;
                    }
                    matrix[row][column] = 0.0;
                    for(std::size_t j = column + 1; j < NumGroups; ++j)
                    {
                        matrix[row][j] -= factor * matrix[column][j];
                        if(!std::isfinite(matrix[row][j]))
                        {
                            return false;
                        }
                    }
                    rhs[row] -= factor * rhs[column];
                    if(!std::isfinite(rhs[row]))
                    {
                        return false;
                    }
                }
            }

            solution.fill(0.0);
            for(std::size_t reverse = 0; reverse < NumGroups; ++reverse)
            {
                std::size_t const row = NumGroups - 1 - reverse;
                double value = rhs[row];
                for(std::size_t column = row + 1; column < NumGroups; ++column)
                {
                    value -= matrix[row][column] * solution[column];
                }
                solution[row] = value / matrix[row][row];
                if(!std::isfinite(solution[row]))
                {
                    return false;
                }
            }
            return true;
    }

    // Lab-to-comoving Doppler factor of a packet in its cell, with the same frame
    // choice as setPacketFromComovingState.  The Compton correction works with
    // comoving group energies, so the census, the path-length tallies and the
    // reconciliation all convert packets with this factor.
    double comptonComovingDoppler(
        const MCParticle &particle) const
    {

            double dopplerShift = 1.0;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(particle.cellIndex < owner_.cells_.size() &&
                   ((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                     !owner_.parameters_.staticScatterers) ||
                    (owner_.parameters_.postProcess.enabled &&
                     owner_.parameters_.postProcess.useCellVelocities)))
                {
                    dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(
                        particle, owner_.cells_[particle.cellIndex], owner_.lightSpeed());
                }
            }
            return dopplerShift;
    }

    // Comoving group of a packet, clamped to the group grid as in the Compton events.
    std::size_t comptonComovingGroup(
        const MCParticle &particle, double dopplerShift) const
    {

            double frequency = particle.frequency * dopplerShift;
            owner_.clampFrequencyToBounds(frequency);
            return owner_.opacity_->findGroup(frequency, owner_.energyBoundaries_);
    }

    void setPacketFromComovingState(
        MCParticle &particle,
        const CellT &cell,
        double comovingFrequency,
        double comovingWeight) const
    {

            double dopplerShift = 1.0;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                    !owner_.parameters_.staticScatterers) ||
                   (owner_.parameters_.postProcess.enabled &&
                    owner_.parameters_.postProcess.useCellVelocities))
                {
                    dopplerShift =
                        radiation_imc_detail::computeDopplerShift<PointT>(
                            particle, cell, owner_.lightSpeed());
                }
            }
            if(!(dopplerShift > 0.0) || !std::isfinite(dopplerShift))
            {
                throw StormError(
                    "RadiationIMC received an invalid source-packet Doppler factor");
            }
            particle.frequency = comovingFrequency / dopplerShift;
            particle.weight = comovingWeight / dopplerShift;
            // Thinned mode never clamps a stored frequency (plan 1b): a lab
            // frequency outside the grid re-enters, photons kept, in transport.
            if(!this->thinnedEvents())
            {
                owner_.clampFrequencyToBounds(particle.frequency);
            }
    }

    double applyComptonScatterEvent(
        std::size_t cellIndex,
        CellT &cell,
        std::size_t sourceGroup,
        MCParticle &particle,
        const PointT &oldVelocity,
        double oldWeight,
        bool inGroup)
    {

            if(sourceGroup >= NumGroups)
            {
                return 0.0;
            }
            ComptonCellData &data = owner_.comptonData_[cellIndex];
            double const eventRate = inGroup
                ? data.comptonInGroupRate[sourceGroup]
                : data.comptonOutRate[sourceGroup];
            if(!(eventRate > 0.0))
            {
                return 0.0;
            }
            if(this->thinnedEvents())
            {
                compton_thinned::Diagnostics &costDiagnostics = owner_.comptonThinnedDiagnostics_;
                std::vector<double> &counts = inGroup ? costDiagnostics.inGroupEventsByGroup
                                                      : costDiagnostics.eventsByGroup;
                if(sourceGroup < counts.size())
                {
                    counts[sourceGroup] += 1.0;
                }
            }
            std::size_t const targetGroup = inGroup ? sourceGroup
                : owner_.sampleComptonTarget(data, sourceGroup, particle);
            if((!inGroup && targetGroup == sourceGroup) || targetGroup >= NumGroups)
            {
                throw StormError("RadiationIMC Compton CDF returned an invalid target group");
            }

            if(owner_.parameters_.comptonAngleDependent)
            {
                std::vector<double> angleCdf;
                owner_.comptonMatrixGen_->GetAngleCdf(
                    data.temperature, sourceGroup, targetGroup, angleCdf);
                std::size_t const angleBins =
                    owner_.comptonMatrixGen_->GetAngleBinCount();
                if(angleBins == 0 || angleCdf.size() != angleBins + 1)
                {
                    throw StormError("RadiationIMC Compton backend returned an invalid angle CDF");
                }
                double const random = owner_.randomUnitOpen(particle);
                std::size_t iteratorBin = 0;
                auto const iterator = std::upper_bound(
                    angleCdf.begin(), angleCdf.end(), random);
                if(iterator != angleCdf.begin())
                {
                    iteratorBin = static_cast<std::size_t>(
                        std::distance(angleCdf.begin(), iterator) - 1);
                }
                iteratorBin = std::min(iteratorBin, angleBins - 1);
                double const lowerCdf = angleCdf[iteratorBin];
                double const upperCdf = angleCdf[iteratorBin + 1];
                double const fraction = upperCdf > lowerCdf
                    ? (random - lowerCdf) / (upperCdf - lowerCdf) : 0.5;
                double const binWidth = 2.0 /
                    static_cast<double>(angleBins);
                double const cosine = -1.0 +
                    (static_cast<double>(iteratorBin) +
                     std::clamp(fraction, 0.0, 1.0)) * binWidth;
                double const sine = std::sqrt(
                    std::max(0.0, 1.0 - cosine * cosine));
                double const pi = 3.141592653589793238462643383279502884;
                double const phi = 2.0 * pi * owner_.randomUnitOpen(particle);
                double const speed = fastabs(oldVelocity);
                PointT const oldDirection = speed > 0.0
                    ? oldVelocity / speed : PointT(0.0, 0.0, 1.0);
                PointT helper = std::abs(oldDirection.z) < 0.9
                    ? PointT(0.0, 0.0, 1.0)
                    : PointT(1.0, 0.0, 0.0);
                PointT perpendicular1 = helper -
                    ScalarProd(helper, oldDirection) * oldDirection;
                if(fastabs(perpendicular1) > 0.0)
                {
                    perpendicular1 = normalize(perpendicular1);
                }
                else
                {
                    perpendicular1 = PointT(1.0, 0.0, 0.0);
                }
                PointT const perpendicular2 =
                    normalize(CrossProduct(oldDirection, perpendicular1));
                PointT const newDirection = normalize(
                    oldDirection * cosine +
                    sine * (std::cos(phi) * perpendicular1 +
                            std::sin(phi) * perpendicular2));
                particle.velocity = newDirection * owner_.lightSpeed();
            }
            else
            {
                particle.velocity = owner_.sampleScatterVelocity(cell, particle);
            }

            // An in-group scattering changes only the direction.
            if(inGroup)
            {
                return 0.0;
            }

            if(this->thinnedEvents())
            {
                // Plan 1a: with probability f the packet jumps, frequency and
                // weight scaled by the centre ratio (photons w/nu kept); otherwise
                // only its direction changed.  The gas receives the expected
                // exchange w (1 - eta); the return value is the ledger mismatch
                // x = w' - w eta (zero mean), from the weight before any nudge.
                compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
                ++diagnostics.events;
                double const expectedWeight = oldWeight * data.comptonMh[sourceGroup];
                if(owner_.randomUnitOpen(particle) >= data.thinnedJumpProbability[sourceGroup])
                {
                    return oldWeight - expectedWeight;
                }
                ++diagnostics.jumps;
                double const ratio = owner_.comptonGroupCenters_[targetGroup] /
                    owner_.comptonGroupCenters_[sourceGroup];
                double const jumpWeight = oldWeight * ratio;
                double const jumpFrequency = particle.frequency * ratio;
                // Rounding can leave the product a hair outside the target group:
                // move it inside, photons kept; the transport tally books the
                // (roundoff-size) energy change.
                double const placedFrequency = compton_thinned::NudgeIntoGroup(
                    jumpFrequency, owner_.energyBoundaries_[targetGroup], owner_.energyBoundaries_[targetGroup + 1],
                    owner_.comptonThinnedGrid_.margin);
                particle.frequency = placedFrequency;
                particle.weight = jumpWeight * placedFrequency / jumpFrequency;
                if(placedFrequency != jumpFrequency)
                {
                    ++diagnostics.nudges;
                    diagnostics.nudgeEnergy += particle.weight - jumpWeight;
                }
                return jumpWeight - expectedWeight;
            }

            // By default use the source-group mean energy multiplier rather than
            // the sampled target ratio. This preserves the mean Compton exchange
            // while avoiding target-to-target packet-weight noise; the
            // deterministic residual correction carries the spectral difference.
            // comptonExactEventEnergy uses the target ratio instead, which keeps a
            // Planck spectrum invariant during transport.
            double const eventMultiplier = this->exactEvents()
                ? owner_.comptonGroupCenters_[targetGroup] /
                    owner_.comptonGroupCenters_[sourceGroup]
                : data.comptonMh[sourceGroup];
            if(!(eventMultiplier > 0.0) || !std::isfinite(eventMultiplier))
            {
                throw StormError("RadiationIMC produced an invalid averaged Compton event multiplier");
            }
            double const newWeight = oldWeight * eventMultiplier;
            particle.weight = newWeight;
            particle.frequency = owner_.frequencyForComptonGroup(targetGroup);
            return oldWeight - newWeight;
    }

    typename Owner::ComptonCorrectionResult
    solveComptonCorrection(
        std::size_t cellIndex,
        double fullDt,
        const ComptonCellData &data,
        const GroupArray &rawGroupEnergy,
        const GroupArray &timeAvgGroupEnergy,
        double budgetBefore,
        double materialFloor,
        double preStepRadiation,
        const GroupArray *bcorrOverride = nullptr,
        double correctionScale = 1.0) const
    {

            // Thinned mode passes Bcorr built from the time averages (plan 3b);
            // it needs no rescaling with the post-transport radiation.
            GroupArray const &bcorr = bcorrOverride ? *bcorrOverride : data.Bcorr;
            ComptonCorrectionResult result;
            result.materialEnergyBefore =
                owner_.extensives_[cellIndex].internal_energy;
            result.budgetBefore = budgetBefore;

            auto fail = [&](ComptonCorrectionFailure failure)
            {
                result.failure = failure;
                result.success = false;
                return result;
            };

            double const rawTotal =
                this->compensatedSumComptonGroups(rawGroupEnergy);
            double const timeAverageTotal =
                this->compensatedSumComptonGroups(timeAvgGroupEnergy);
            if(!std::isfinite(rawTotal) ||
               !std::isfinite(timeAverageTotal) ||
               !std::isfinite(budgetBefore) ||
               !std::isfinite(materialFloor) ||
               materialFloor < 0.0)
            {
                return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
            }

            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                if(!std::isfinite(rawGroupEnergy[group]) ||
                   rawGroupEnergy[group] < 0.0 ||
                   !std::isfinite(timeAvgGroupEnergy[group]) ||
                   timeAvgGroupEnergy[group] < 0.0 ||
                   !std::isfinite(bcorr[group]))
                {
                    return fail(ComptonCorrectionFailure::NonFiniteState);
                }
            }
            if(rawTotal < 0.0)
            {
                return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
            }

            double const totalPostTransportRadiation = rawTotal;
            if(!std::isfinite(preStepRadiation) ||
               preStepRadiation < 0.0)
            {
                return fail(ComptonCorrectionFailure::NonFiniteState);
            }
            double const bcorrScale = !bcorrOverride && preStepRadiation > 0.0
                ? std::clamp(
                    totalPostTransportRadiation / preStepRadiation, 0.0, 1.0)
                : 1.0;
            double const materialCap = owner_.parameters_.noHydroFeedback
                ? std::numeric_limits<double>::infinity()
                : budgetBefore - materialFloor;
            if(!owner_.parameters_.noHydroFeedback &&
               owner_.extensives_[cellIndex].internal_energy < materialFloor)
            {
                return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
            }

            GroupArray oldGroupEnergy{};
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                oldGroupEnergy[group] = std::max(
                    0.0, data.oldRadiationEnergy[group] * data.volume);
                if(!std::isfinite(oldGroupEnergy[group]))
                {
                    return fail(ComptonCorrectionFailure::NonFiniteState);
                }
            }

            GroupArray drive{};
            GroupArray conservativeTimeAverageDrive{};
            GroupArray materialCouplingDrive{};
            GroupArray residualColumnSum{};
            GroupMatrix residualOperator{};
            GroupMatrix matrix{};
            for(std::size_t row = 0; row < NumGroups; ++row)
            {
                for(std::size_t column = 0; column < NumGroups; ++column)
                {
                    double const Lrc = correctionScale * fullDt * owner_.lightSpeed() *
                        data.residualKernel[column][row];
                    if(!std::isfinite(Lrc))
                    {
                        return fail(ComptonCorrectionFailure::NonFiniteState);
                    }
                    residualOperator[row][column] = Lrc;
                    residualColumnSum[column] += Lrc;
                    matrix[row][column] =
                        (row == column ? 1.0 : 0.0) - Lrc;
                }
            }

            // The path-length estimator has far better group support than the
            // endpoint census, but applying it to the full residual operator also
            // changes the operator's column-sum component and therefore the net
            // radiation/material exchange.  That destroys the well-balanced state
            // used to build Bcorr and the modified Fleck factor.  Separate
            //
            //     L = C + diag(1^T L),       1^T C = 0,
            //
            // use E_avg only in C E_avg, and evaluate the net coupling with the
            // pre-step state used to construct the Compton coefficients.  Scale that
            // net source together with Bcorr so transport depletion cannot break the
            // balance between the two terms.
            for(std::size_t row = 0; row < NumGroups; ++row)
            {
                for(std::size_t column = 0; column < NumGroups; ++column)
                {
                    double const Lrc = residualOperator[row][column];
                    double const conservativeLrc = Lrc -
                        (row == column ? residualColumnSum[column] : 0.0);
                    conservativeTimeAverageDrive[row] +=
                        conservativeLrc * timeAvgGroupEnergy[column];
                }
                materialCouplingDrive[row] = bcorrScale *
                    (correctionScale * bcorr[row] +
                     residualColumnSum[row] * oldGroupEnergy[row]);
                drive[row] = conservativeTimeAverageDrive[row] +
                    materialCouplingDrive[row];
                if(!std::isfinite(drive[row]) ||
                   !std::isfinite(conservativeTimeAverageDrive[row]) ||
                   !std::isfinite(materialCouplingDrive[row]) ||
                   !std::isfinite(residualColumnSum[row]))
                {
                    return fail(ComptonCorrectionFailure::NonFiniteState);
                }
            }
            double energyScale = std::max({
                1.0,
                std::abs(budgetBefore),
                std::abs(rawTotal),
                std::abs(timeAverageTotal),
                this->maxAbsComptonGroup(rawGroupEnergy),
                this->maxAbsComptonGroup(timeAvgGroupEnergy),
                this->maxAbsComptonGroup(oldGroupEnergy),
                this->normComptonGroups(conservativeTimeAverageDrive),
                this->normComptonGroups(materialCouplingDrive),
                this->normComptonGroups(drive),
                this->maxAbsComptonGroup(drive)});
            if(!std::isfinite(energyScale))
            {
                return fail(ComptonCorrectionFailure::NonFiniteState);
            }
            GroupArray directDelta{};
            bool const directOk = this->solveComptonGroupSystem(
                matrix, drive, directDelta);
            // Low-group diagnostic, off unless STORM_COMPTON_LOWGROUP_DIAG gives a group count: the drive of
            // the lowest groups split into the (1 - f) S part and the Lambda M^T (temperature linearization)
            // part, the material coupling and the direct correction.
            if(correctionScale == 1.0 && cellIndex == 0 && this->lowGroupDiagnosticGroups() > 0)
            {
                double exchange = 0.0;
                double sumM = 0.0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    exchange += data.Lambda[group] * timeAvgGroupEnergy[group];
                    sumM += data.M[group];
                }
                double const lambdaScale = fullDt * owner_.lightSpeed() * data.betaCdtF;
                std::cout << "LGDIAG correction cell " << cellIndex << " dt " << fullDt << " fleck " << data.fleck
                          << " betaCdtF " << data.betaCdtF << " sumM " << sumM << " sumLambdaAvg " << exchange << '\n';
                for(std::size_t group = 0; group < std::min(this->lowGroupDiagnosticGroups(), NumGroups); ++group)
                {
                    // M_t = D_t split into the self term and the inflow from other groups (largest source named).
                    double const self = data.dSdUm[group][group] * data.oldRadiationEnergy[group];
                    double inflow = 0.0;
                    double largest = 0.0;
                    std::size_t largestSource = group;
                    for(std::size_t source = 0; source < NumGroups; ++source)
                    {
                        double const term = data.dSdUm[source][group] * data.oldRadiationEnergy[source];
                        if(source != group)
                        {
                            inflow += term;
                            if(std::abs(term) > std::abs(largest))
                            {
                                largest = term;
                                largestSource = source;
                            }
                        }
                    }
                    std::cout << "LGDIAG m " << group << " self " << self << " inflow " << inflow << " largest "
                              << largest << " from " << largestSource << " dSdUm_self " << data.dSdUm[group][group]
                              << " dtau_self_row " << data.dtau_dUm[group][std::min(group + 1, NumGroups - 1)] << '\n';
                    double const lambdaPart = lambdaScale *
                        (data.M[group] * exchange - sumM * data.Lambda[group] * timeAvgGroupEnergy[group]);
                    std::cout << "LGDIAG g " << group << " old " << oldGroupEnergy[group] << " raw "
                              << rawGroupEnergy[group] << " avg " << timeAvgGroupEnergy[group] << " sPart "
                              << conservativeTimeAverageDrive[group] - lambdaPart << " lambdaPart " << lambdaPart
                              << " coupling " << materialCouplingDrive[group] << " bcorr " << bcorr[group]
                              << " delta " << directDelta[group] << " M " << data.M[group] << " Lambda "
                              << data.Lambda[group] << " n " << data.occupation[group] << '\n';
                }
            }

            GroupArray directEndpoint{};
            double directTotal = 0.0;
            ComptonProjectionResult directProjection;
            bool directCandidateFinite = directOk;
            double directResidual = std::numeric_limits<double>::infinity();
            if(directOk)
            {
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    directEndpoint[group] =
                        rawGroupEnergy[group] + directDelta[group];
                    if(!std::isfinite(directEndpoint[group]))
                    {
                        directCandidateFinite = false;
                    }
                }
                directResidual = this->relativeComptonResidual(
                    matrix, directDelta, drive, energyScale);
                directTotal =
                    this->compensatedSumComptonGroups(directEndpoint);
                if(!std::isfinite(directTotal))
                {
                    directCandidateFinite = false;
                }
                if(directCandidateFinite)
                {
                    energyScale = std::max(
                        energyScale,
                        std::max(
                            std::abs(directTotal),
                            this->maxAbsComptonGroup(directEndpoint)));
                }
            }

            // These are deliberately loose engineering tolerances for a noisy,
            // low-packet spectrum.  They are not machine-roundoff epsilons.  The
            // aggregate negative-mass limit must allow several individually small
            // negative bins to be removed in one conservative projection.  The
            // cycle-309 failure had 0.661% aggregate negative mass while every group
            // was below 0.01% of the cell energy, so the former 0.2% aggregate limit
            // forced many path-dependent projected substeps.  Permit up to a 1%
            // one-shot projection and require the projected equations to remain
            // accurate to the same 1% engineering scale.
            constexpr double perGroupNegativeToleranceFraction = 1e-3;
            constexpr double totalNegativeToleranceFraction = 1e-2;
            constexpr double capToleranceFraction = 1e-6;
            constexpr double relativeResidualTolerance = 1e-2;
            constexpr double grossResidualTolerance = 1e-2;
            double const perGroupNegativeTolerance =
                perGroupNegativeToleranceFraction * energyScale;
            double const totalNegativeTolerance =
                totalNegativeToleranceFraction * energyScale;
            double const capTolerance = capToleranceFraction * energyScale;
            bool const directCapWithinTolerance =
                directCandidateFinite &&
                (!std::isfinite(materialCap) ||
                 directTotal <= materialCap + capTolerance);
            bool const directCapRepair =
                directCandidateFinite &&
                std::isfinite(materialCap) &&
                directTotal > materialCap &&
                directTotal <= materialCap + capTolerance;
            double directTargetTotal = directTotal;
            if(directCapRepair)
            {
                directTargetTotal = materialCap;
            }
            if(directCandidateFinite)
            {
                directProjection = this->projectNonnegativeConservative(
                    directEndpoint,
                    directTargetTotal,
                    energyScale,
                    perGroupNegativeTolerance,
                    totalNegativeTolerance);
                directProjection.usedCapRepair = directCapRepair;
            }
            result.correctionScale = correctionScale;
            result.directResidual = directResidual;
            result.directTotal = directTotal;
            if(directCandidateFinite)
            {
                result.directWorstNegative = directProjection.worstNegative;
                result.directNegativeMass = directProjection.negativeMass;
            }
            bool const directClampAcceptable = directCandidateFinite &&
                directProjection.worstNegative <= perGroupNegativeTolerance &&
                directProjection.negativeMass <= totalNegativeTolerance;

            if(directCandidateFinite && directProjection.success)
            {
                GroupArray projectedDelta{};
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    projectedDelta[group] =
                        directProjection.endpoint[group] - rawGroupEnergy[group];
                }
                double const projectedResidual =
                    this->relativeComptonResidual(
                        matrix, projectedDelta, drive, energyScale);
                bool const residualAcceptable =
                    directResidual <= grossResidualTolerance &&
                    projectedResidual <= relativeResidualTolerance;
                bool const endpointAcceptable =
                    directClampAcceptable &&
                    directCapWithinTolerance &&
                    residualAcceptable;
                if(endpointAcceptable)
                {
                    result.endpoint = directProjection.endpoint;
                    result.delta = projectedDelta;
                    result.radiationTotal =
                        this->compensatedSumComptonGroups(result.endpoint);
                    result.materialEnergyBefore =
                        owner_.extensives_[cellIndex].internal_energy;
                    result.materialEnergyAfter = owner_.parameters_.noHydroFeedback
                        ? result.materialEnergyBefore
                        : budgetBefore - result.radiationTotal;
                    result.budgetBefore = budgetBefore;
                    result.usedProjection = directProjection.usedProjection;
                    result.usedCapRepair = directProjection.usedCapRepair;
                    result.energyClosureResidual =
                        owner_.parameters_.noHydroFeedback
                        ? 0.0
                        : budgetBefore - (result.materialEnergyAfter +
                            result.radiationTotal);
                    double const closureTolerance =
                        128.0 * std::numeric_limits<double>::epsilon() *
                        std::max(1.0, std::abs(budgetBefore));
                    if(!owner_.parameters_.noHydroFeedback &&
                       std::isfinite(result.energyClosureResidual) &&
                       std::abs(result.energyClosureResidual) <= closureTolerance)
                    {
                        result.materialEnergyAfter +=
                            result.energyClosureResidual;
                        result.energyClosureResidual =
                            budgetBefore - (result.materialEnergyAfter +
                                result.radiationTotal);
                    }
                    if(!std::isfinite(result.materialEnergyAfter) ||
                       result.materialEnergyAfter < materialFloor - capTolerance)
                    {
                        result.failure =
                            ComptonCorrectionFailure::InvalidEnergyBudget;
                    }
                    else if(!std::isfinite(result.energyClosureResidual) ||
                            std::abs(result.energyClosureResidual) >
                                closureTolerance)
                    {
                        result.failure =
                            ComptonCorrectionFailure::EnergyClosureFailure;
                    }
                    else
                    {
                        result.success = true;
                        result.failure = ComptonCorrectionFailure::None;
                        return result;
                    }
                }
            }

            GroupArray currentEndpoint = rawGroupEnergy;
            GroupArray currentDelta{};
            double tau = 0.0;
            double fraction = 1.0;
            std::size_t acceptedSubsteps = 0;
            std::size_t rejectedTrials = 0;
            double minimumAcceptedFraction =
                std::numeric_limits<double>::infinity();
            double maximumAcceptedFraction = 0.0;
            bool usedProjection = false;
            bool usedCapRepair = false;
            ComptonCorrectionFailure lastFailure;
            if(!directOk)
            {
                lastFailure = ComptonCorrectionFailure::DirectLinearSolveFailed;
            }
            else if(!directCandidateFinite)
            {
                lastFailure = ComptonCorrectionFailure::NonFiniteState;
            }
            else if(!directCapWithinTolerance)
            {
                lastFailure = ComptonCorrectionFailure::DirectMaterialCap;
            }
            else if(!directProjection.success || !directClampAcceptable)
            {
                lastFailure = ComptonCorrectionFailure::DirectNegativeMass;
            }
            else
            {
                lastFailure = ComptonCorrectionFailure::DirectProjectedResidual;
            }

            constexpr std::size_t maxAcceptedSubsteps = 32;
            // Rejected trial solves are line-search work, not physical substeps.  Keep
            // the requested 32 accepted-substep limit, but do not abort merely because
            // two trial reductions were needed per accepted step.
            constexpr std::size_t maxRejectedTrials = 128;
            constexpr double minimumFraction = 1e-12;
            constexpr double completionTolerance = 1e-12;

            while(tau < 1.0 - completionTolerance &&
                  acceptedSubsteps < maxAcceptedSubsteps &&
                  rejectedTrials < maxRejectedTrials)
            {
                fraction = std::min(fraction, 1.0 - tau);
                if(fraction < minimumFraction)
                {
                    lastFailure = ComptonCorrectionFailure::AdaptiveFractionBelowMinimum;
                    break;
                }

                GroupMatrix fractionalMatrix{};
                GroupArray fractionalRhs{};
                for(std::size_t row = 0; row < NumGroups; ++row)
                {
                    fractionalRhs[row] =
                        currentDelta[row] + fraction * drive[row];
                    for(std::size_t column = 0; column < NumGroups; ++column)
                    {
                        double const Lrc = correctionScale * fullDt * owner_.lightSpeed() *
                            data.residualKernel[column][row];
                        fractionalMatrix[row][column] =
                            (row == column ? 1.0 : 0.0) - fraction * Lrc;
                    }
                }

                GroupArray trialDelta{};
                bool const trialSolveOk = this->solveComptonGroupSystem(
                    fractionalMatrix,
                    fractionalRhs,
                    trialDelta);
                if(!trialSolveOk)
                {
                    ++rejectedTrials;
                    lastFailure =
                        ComptonCorrectionFailure::AdaptiveLinearSolveFailed;
                    fraction *= 0.5;
                    continue;
                }

                GroupArray trialEndpoint{};
                bool finiteTrial = true;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    trialEndpoint[group] =
                        rawGroupEnergy[group] + trialDelta[group];
                    if(!std::isfinite(trialEndpoint[group]))
                    {
                        finiteTrial = false;
                    }
                }
                if(!finiteTrial)
                {
                    ++rejectedTrials;
                    lastFailure = ComptonCorrectionFailure::NonFiniteState;
                    fraction *= 0.5;
                    continue;
                }

                double const trialTotal =
                    this->compensatedSumComptonGroups(trialEndpoint);
                double const capOvershoot = std::isfinite(materialCap)
                    ? trialTotal - materialCap : 0.0;
                bool const capTooLarge = std::isfinite(materialCap) &&
                    capOvershoot > capTolerance;
                bool const targetIsNegative =
                    trialTotal < -totalNegativeTolerance;
                double trialTargetTotal = trialTotal;
                bool const trialCapRepair = std::isfinite(materialCap) &&
                    capOvershoot > 0.0 && !capTooLarge;
                if(trialCapRepair)
                {
                    trialTargetTotal = materialCap;
                }

                ComptonProjectionResult trialProjection =
                    this->projectNonnegativeConservative(
                        trialEndpoint,
                        trialTargetTotal,
                        energyScale,
                        perGroupNegativeTolerance,
                        totalNegativeTolerance);
                trialProjection.usedCapRepair = trialCapRepair;
                GroupArray projectedDelta{};
                if(trialProjection.success)
                {
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        projectedDelta[group] =
                            trialProjection.endpoint[group] -
                            rawGroupEnergy[group];
                    }
                }
                double fractionalResidual = std::numeric_limits<double>::infinity();
                if(trialProjection.success)
                {
                    fractionalResidual =
                        this->relativeComptonResidual(
                            fractionalMatrix,
                            projectedDelta,
                            fractionalRhs,
                            energyScale);
                }
                bool const trialAcceptable =
                    !capTooLarge &&
                    !targetIsNegative &&
                    trialProjection.success &&
                    fractionalResidual <= relativeResidualTolerance &&
                    trialProjection.worstNegative <=
                        perGroupNegativeTolerance &&
                    trialProjection.negativeMass <= totalNegativeTolerance;
                if(trialAcceptable)
                {
                    currentEndpoint = trialProjection.endpoint;
                    currentDelta = projectedDelta;
                    tau += fraction;
                    ++acceptedSubsteps;
                    minimumAcceptedFraction = std::min(
                        minimumAcceptedFraction, fraction);
                    maximumAcceptedFraction = std::max(
                        maximumAcceptedFraction, fraction);
                    usedProjection = usedProjection ||
                        trialProjection.usedProjection;
                    usedCapRepair = usedCapRepair ||
                        trialProjection.usedCapRepair;
                    // Grow from the last successful fraction, but also request at
                    // least the average fraction needed to finish in the remaining
                    // accepted-step budget.  Retrying the complete remaining interval
                    // after every success caused repeated reject/shrink cycles and
                    // exhausted the rejected-trial guard without improving tau.
                    double const remaining = std::max(0.0, 1.0 - tau);
                    std::size_t const remainingSlots =
                        maxAcceptedSubsteps - acceptedSubsteps;
                    if(remainingSlots > 0)
                    {
                        double const requiredAverage = remaining /
                            static_cast<double>(remainingSlots);
                        fraction = std::min(
                            remaining,
                            std::max(1.5 * fraction, requiredAverage));
                    }
                    else
                    {
                        fraction = remaining;
                    }
                    continue;
                }

                ++rejectedTrials;
                lastFailure = capTooLarge
                    ? ComptonCorrectionFailure::DirectMaterialCap
                    : ComptonCorrectionFailure::AdaptiveNoProgress;
                double theta = 1.0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(trialEndpoint[group] < 0.0)
                    {
                        double const denominator =
                            currentEndpoint[group] - trialEndpoint[group];
                        theta = denominator > 0.0
                            ? std::min(theta,
                                currentEndpoint[group] / denominator)
                            : 0.0;
                    }
                }
                if(std::isfinite(materialCap) && trialTotal > materialCap)
                {
                    double const currentTotal =
                        this->compensatedSumComptonGroups(currentEndpoint);
                    double const totalChange = trialTotal - currentTotal;
                    if(totalChange > 0.0)
                    {
                        theta = std::min(theta,
                            (materialCap - currentTotal) / totalChange);
                    }
                    else
                    {
                        theta = 0.0;
                    }
                }
                double const newFraction = std::clamp(
                    0.8 * theta * fraction,
                    0.1 * fraction,
                    0.7 * fraction);
                if(newFraction < minimumFraction ||
                   newFraction >= 0.99 * fraction)
                {
                    lastFailure = ComptonCorrectionFailure::AdaptiveNoProgress;
                    break;
                }
                fraction = newFraction;
            }

            if(tau < 1.0 - completionTolerance)
            {
                if(acceptedSubsteps >= maxAcceptedSubsteps)
                {
                    lastFailure =
                        ComptonCorrectionFailure::AdaptiveMaximumSubsteps;
                }
                else if(rejectedTrials >= maxRejectedTrials)
                {
                    lastFailure =
                        ComptonCorrectionFailure::AdaptiveMaximumRejectedTrials;
                }
                else if(fraction < minimumFraction)
                {
                    lastFailure =
                        ComptonCorrectionFailure::AdaptiveFractionBelowMinimum;
                }
                return fail(lastFailure);
            }

            result.endpoint = currentEndpoint;
            result.delta = currentDelta;
            result.radiationTotal =
                this->compensatedSumComptonGroups(result.endpoint);
            result.materialEnergyBefore =
                owner_.extensives_[cellIndex].internal_energy;
            result.materialEnergyAfter = owner_.parameters_.noHydroFeedback
                ? result.materialEnergyBefore
                : budgetBefore - result.radiationTotal;
            result.budgetBefore = budgetBefore;
            result.usedProjection = usedProjection;
            result.usedCapRepair = usedCapRepair;
            result.energyClosureResidual =
                owner_.parameters_.noHydroFeedback
                ? 0.0
                : budgetBefore - (result.materialEnergyAfter +
                    result.radiationTotal);
            double const closureTolerance =
                128.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(budgetBefore));
            if(!owner_.parameters_.noHydroFeedback &&
               std::isfinite(result.energyClosureResidual) &&
               std::abs(result.energyClosureResidual) <= closureTolerance)
            {
                result.materialEnergyAfter +=
                    result.energyClosureResidual;
                result.energyClosureResidual =
                    budgetBefore - (result.materialEnergyAfter +
                        result.radiationTotal);
            }
            if(!std::isfinite(result.materialEnergyAfter) ||
               result.materialEnergyAfter < materialFloor - capTolerance)
            {
                return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
            }
            if(!std::isfinite(result.energyClosureResidual) ||
               std::abs(result.energyClosureResidual) > closureTolerance)
            {
                return fail(ComptonCorrectionFailure::EnergyClosureFailure);
            }
            result.success = true;
            result.failure = ComptonCorrectionFailure::None;
            return result;
    }

    // ----- Thinned event mode (plan v10, ComptonThinned.hpp) -----

    // Same frame choice as comptonComovingDoppler and setPacketFromComovingState.
    bool comptonMovingFrame() const
    {
        return (owner_.parameters_.withHydro && !owner_.parameters_.MMC && !owner_.parameters_.staticScatterers) ||
               (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities);
    }

    // Velocity and Lorentz factor of a cell's transport frame (zero and 1 when
    // transport uses no moving frame).
    std::pair<PointT, double> comptonTransportFrame(std::size_t cellIndex) const
    {
        PointT velocity{};
        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->comptonMovingFrame())
            {
                velocity = cellIndex < owner_.transportCellVelocities_.size()
                    ? owner_.transportCellVelocities_[cellIndex] : owner_.cells_[cellIndex].velocity;
                gamma = 1.0 / std::sqrt(1.0 - ScalarProd(velocity, velocity) * owner_.inverseLightSpeedSquared());
            }
        }
        return {velocity, gamma};
    }

    // Plan 1b: a packet whose comoving frequency lies outside [B_0, B_G)
    // re-enters next to the near edge; frequency and weight are scaled by the
    // same factor, so photons are kept, and the lab energy and momentum
    // difference goes to the gas as frequency-boundary work.
    void reenterComptonGrid(std::size_t cellIndex, MCParticle &particle, double dopplerShift)
    {
        double const comovingFrequency = particle.frequency * dopplerShift;
        double const factor = compton_thinned::ReentryFactor(
            comovingFrequency, owner_.energyBoundaries_[0], owner_.energyBoundaries_[NumGroups],
            owner_.comptonGroupCenters_[0], owner_.comptonGroupCenters_[NumGroups - 1], owner_.comptonThinnedGrid_);
        if(factor == 1.0)
        {
            return;
        }
        if(!(factor > 0.0) || !std::isfinite(factor))
        {
            StormError eo("Thinned Compton re-entry received an invalid comoving frequency");
            eo.addEntry("Cell index", cellIndex);
            eo.addEntry("Comoving frequency", comovingFrequency);
            throw eo;
        }
        double const work = (1.0 - factor) * particle.weight;
        particle.frequency *= factor;
        particle.weight *= factor;
        particle.initialWeight *= factor;
        compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
        if(comovingFrequency < owner_.energyBoundaries_[0])
        {
            ++diagnostics.reentryLower;
            diagnostics.boundaryWorkLower += work;
        }
        else
        {
            ++diagnostics.reentryUpper;
            diagnostics.boundaryWorkUpper += work;
        }
        if(!owner_.parameters_.noHydroFeedback && !owner_.parameters_.postProcess.enabled)
        {
            owner_.tallyMaterialEnergy(cellIndex, work);
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                if(owner_.parameters_.withHydro && !owner_.parameters_.diffusionPressureGradient)
                {
                    owner_.tallyMomentum(cellIndex, particle.velocity * (work * owner_.inverseLightSpeedSquared()));
                }
            }
        }
    }

    void tallyThinnedHazard(std::size_t cellIndex, std::size_t group, double energyIntegral, double squaredIntegral)
    {
        owner_.comptonHazardEnergy_[cellIndex][group] += energyIntegral;
        owner_.comptonHazardWeightSquared_[cellIndex][group] += squaredIntegral;
    }

    // preStep: the census preparation (adjustExistingParticles) must have
    // planned this step with the same dt; then the step's ledgers and hazard
    // tallies start from zero.
    void beginThinnedStep(std::size_t cellCount, bool reusedPreparation)
    {
        if(!this->thinnedEvents())
        {
            return;
        }
        if(!owner_.parameters_.postProcess.enabled && (!owner_.comptonCensusPrepared_ || !reusedPreparation))
        {
            StormError eo("Thinned Compton events need adjustExistingParticles(particles, dt) before every step, "
                          "with the step's dt");
            eo.addEntry("Census prepared", owner_.comptonCensusPrepared_);
            eo.addEntry("Preparation reused", reusedPreparation);
            throw eo;
        }
        owner_.comptonCensusPrepared_ = false;
        owner_.comptonLedger_.assign(cellCount, 0.0);
        owner_.comptonHazardEnergy_.assign(cellCount, GroupArray{});
        owner_.comptonHazardWeightSquared_.assign(cellCount, GroupArray{});
        owner_.comptonRawPhotons_.assign(cellCount, 0.0);
        owner_.comptonEndH_.assign(cellCount, GroupArray{});
        owner_.comptonGasSettlement_.assign(cellCount, 0.0);
        owner_.comptonCorrectionGasChange_.assign(cellCount, 0.0);
    }

    // Census moments of the cell's packets per comoving group: energy, actual
    // photons sum w/nu and sum w^2, all comoving.
    void accumulateComptonCensus(const std::vector<MCParticle> &particles, std::vector<GroupArray> &energy,
                                 std::vector<GroupArray> *photons, std::vector<GroupArray> *weightSquared) const
    {
        for(const MCParticle &particle : particles)
        {
            if(particle.cellIndex >= energy.size())
            {
                continue;
            }
            double const dopplerShift = this->comptonComovingDoppler(particle);
            std::size_t const group = this->comptonComovingGroup(particle, dopplerShift);
            if(group >= NumGroups)
            {
                continue;
            }
            double const weight = particle.weight * dopplerShift;
            energy[particle.cellIndex][group] += weight;
            if(photons)
            {
                (*photons)[particle.cellIndex][group] += particle.weight / particle.frequency;
            }
            if(weightSquared)
            {
                (*weightSquared)[particle.cellIndex][group] += weight * weight;
            }
        }
    }

    // h_g = N_g / E_g of the census, 1/eps_g for an empty group.
    GroupArray comptonPhotonsPerEnergy(const GroupArray &energy, const GroupArray &photons) const
    {
        GroupArray h{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            h[group] = energy[group] > 0.0 && photons[group] > 0.0
                ? photons[group] / energy[group] : 1.0 / owner_.comptonGroupCenters_[group];
        }
        return h;
    }

    // Plan 2b: once the census is final (after risk splitting), record its
    // moments and fix each cell's reservoir plan for the step from the
    // predicted ledger noise sigma^2 = sum_s c dt q_s M2_s [f V2_s - f^2 (mu_s - 1)^2],
    // M2_s = max(Dmax (C_s + P_s), T_s^prev) (extensive, erg^2).
    void prepareThinnedCensus(const std::vector<MCParticle> &particles, double fullDt)
    {
        if(!this->thinnedEvents() || owner_.parameters_.postProcess.enabled)
        {
            return;
        }
        if(owner_.comptonCensusPrepared_)
        {
            throw StormError("Thinned Compton census was prepared twice for one step");
        }
        std::size_t const Ncells = owner_.comptonData_.size();
        owner_.comptonThinnedDiagnostics_ = compton_thinned::Diagnostics{};
        owner_.comptonThinnedDiagnostics_.eventsByGroup.assign(NumGroups, 0.0);
        owner_.comptonThinnedDiagnostics_.inGroupEventsByGroup.assign(NumGroups, 0.0);
        std::vector<GroupArray> energy(Ncells, GroupArray{});
        std::vector<GroupArray> photons(Ncells, GroupArray{});
        std::vector<GroupArray> weightSquared(Ncells, GroupArray{});
        this->accumulateComptonCensus(particles, energy, &photons, &weightSquared);
        if(owner_.comptonPreviousWeightSquared_.size() != Ncells)
        {
            owner_.comptonPreviousWeightSquared_.assign(Ncells, GroupArray{});
        }
        double const cdt = owner_.lightSpeed() * fullDt;
        compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
        for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
        {
            ComptonCellData &data = owner_.comptonData_[cellIndex];
            data.censusEnergy = energy[cellIndex];
            data.censusPhotons = photons[cellIndex];
            data.censusWeightSquared = weightSquared[cellIndex];
            data.censusPrepared = true;
            auto const [velocity, gamma] = this->comptonTransportFrame(cellIndex);
            double const maximumDoppler = gamma *
                (1.0 + std::sqrt(ScalarProd(velocity, velocity) * owner_.inverseLightSpeedSquared()));
            std::array<std::size_t, NumGroups> const emission = this->allocateComptonSourcePackets(data);
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const emitted = emission[group] > 0
                    ? std::max(0.0, data.Bbase[group]) * std::max(0.0, data.Bbase[group]) /
                        static_cast<double>(emission[group])
                    : 0.0;
                data.plannedWeightSquared[group] = std::max(
                    maximumDoppler * (weightSquared[cellIndex][group] + emitted),
                    owner_.comptonPreviousWeightSquared_[cellIndex][group]);
            }
            double const variance = compton_thinned::PredictLedgerVariance<NumGroups>(
                cdt, data.thinnedJumpProbability, data.comptonOutRate, data.comptonMu, data.thinnedV2,
                data.plannedWeightSquared);
            data.predictedSigma = std::sqrt(std::max(0.0, variance));
            double const gasEnergy = owner_.extensives_[cellIndex].internal_energy;
            data.plannedGas = !owner_.parameters_.noHydroFeedback &&
                data.predictedSigma <= owner_.parameters_.comptonSettlementGasTolerance * gasEnergy;
            GroupArray const hPre = this->comptonPhotonsPerEnergy(energy[cellIndex], photons[cellIndex]);
            constexpr double minimumStencilRatio = 8.0;
            data.plannedStencil = compton_thinned::SelectPlannedStencil<NumGroups>(
                energy[cellIndex], hPre, data.predictedSigma, owner_.parameters_.comptonSettlementSigmaMargin,
                minimumStencilRatio);
            diagnostics.predictedSigmaSum += data.predictedSigma;
            diagnostics.limiterActive += data.limiterActive ? 1 : 0;
            diagnostics.limiterInfeasible += data.limiterInfeasible ? 1 : 0;
            if(data.plannedGas)
            {
                ++diagnostics.gasFirstChoices;
            }
            else
            {
                ++diagnostics.radiationFirstChoices;
                if(!data.plannedStencil.qualified)
                {
                    ++diagnostics.unqualifiedStencils;
                }
            }
            if(data.plannedStencil.found)
            {
                diagnostics.maxStencilAmplification = std::max(
                    diagnostics.maxStencilAmplification,
                    compton_thinned::StencilAmplification(data.plannedStencil));
            }
        }
        owner_.comptonCensusPrepared_ = true;
    }

    // Plan 2c, between the census tally and the correction: every cell's ledger
    // X is paid photon-neutrally by the planned reservoir, else the other one,
    // else any photon-neutral pair; nothing paying is a failure.  Records the
    // actual photons before settlement and h^end for the correction.
    void settleThinnedLedgers(const std::vector<MCParticle> &particles)
    {
        if(!this->thinnedEvents() || owner_.parameters_.postProcess.enabled)
        {
            return;
        }
        if constexpr(radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
        {
            std::size_t const Ncells = owner_.comptonData_.size();
            std::vector<GroupArray> energy(Ncells, GroupArray{});
            std::vector<GroupArray> photons(Ncells, GroupArray{});
            this->accumulateComptonCensus(particles, energy, &photons, nullptr);
            compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                ComptonCellData const &data = owner_.comptonData_[cellIndex];
                GroupArray const hEnd = this->comptonPhotonsPerEnergy(energy[cellIndex], photons[cellIndex]);
                owner_.comptonEndH_[cellIndex] = hEnd;
                owner_.comptonRawPhotons_[cellIndex] = this->compensatedSumComptonGroups(photons[cellIndex]);
                diagnostics.censusPhotonsRaw += owner_.comptonRawPhotons_[cellIndex];
                // The ledger stays recorded: the lab closure books it (plan 4).
                double const ledger = owner_.comptonLedger_[cellIndex];
                if(owner_.parameters_.noHydroFeedback || ledger == 0.0)
                {
                    continue;
                }
                diagnostics.ledgerAbsSum += std::abs(ledger);
                GroupArray groupEnergy{};
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    groupEnergy[group] = owner_.extensives_[cellIndex].Eg[group];
                }
                compton_thinned::Settlement const settlement = compton_thinned::Settle<NumGroups>(
                    ledger, data.plannedGas ? compton_thinned::Reservoir::Gas : compton_thinned::Reservoir::Radiation,
                    data.plannedStencil, owner_.extensives_[cellIndex].internal_energy, 0.0, true, groupEnergy, hEnd);
                ++diagnostics.tiers[static_cast<std::size_t>(settlement.tier)];
                if(settlement.tier == 4)
                {
                    StormError eo("Thinned Compton settlement found no reservoir for the energy ledger");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Ledger", ledger);
                    eo.addEntry("Predicted sigma", data.predictedSigma);
                    eo.addEntry("Internal energy", owner_.extensives_[cellIndex].internal_energy);
                    eo.addEntry("Radiation energy", this->compensatedSumComptonGroups(groupEnergy));
                    throw eo;
                }
                if(settlement.payer == compton_thinned::Reservoir::Gas)
                {
                    // The gas pays the comoving energy X: lab (-gamma X, -gamma X v/c^2).
                    auto const [velocity, gamma] = this->comptonTransportFrame(cellIndex);
                    owner_.applyMaterialExchange(
                        cellIndex, -gamma * ledger, velocity * (-gamma * ledger * owner_.inverseLightSpeedSquared()));
                    owner_.comptonGasSettlement_[cellIndex] = -ledger;
                    continue;
                }
                compton_thinned::ApplyStencil<NumGroups>(settlement.stencil, ledger, groupEnergy);
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    owner_.extensives_[cellIndex].Eg[group] = groupEnergy[group];
                }
                diagnostics.settlementPhotonChange -= ledger *
                    (settlement.stencil.ca * hEnd[settlement.stencil.a] +
                     settlement.stencil.cb * hEnd[settlement.stencil.b]);
            }
        }
        else
        {
            (void) particles;
        }
    }

    // Plan 3c, after f: the pre-step contraction D is clipped so that the
    // linearized drive withdraws at most phi of a group's old energy,
    // c dt (1 - f) |H_imb| |D'_t| / Gamma <= phi E_t^old, keeping sum D (so
    // Upsilon, Gamma and f) and sum D/eps; infeasible leaves D unlimited.
    // M (hence the kernels, sources and risk scores) is rebuilt from D'.
    void limitThinnedContraction(ComptonCellData &data, double sourceDt) const
    {
        data.thinnedLimiterDelta = GroupArray{};
        data.limiterActive = false;
        data.limiterInfeasible = false;
        double const cdt = owner_.lightSpeed() * sourceDt;
        double imbalance = -data.planckOpacity * data.Um * data.volume;
        GroupArray oldEnergy{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            oldEnergy[group] = std::max(0.0, data.oldRadiationEnergy[group]) * data.volume;
            imbalance += (data.absorptionOpacity[group] - data.rowS[group]) * oldEnergy[group];
        }
        double const withdrawal = cdt * (1.0 - data.fleck) * std::abs(imbalance);
        if(!(data.Gamma > 0.0) || !(withdrawal > 0.0) || !std::isfinite(withdrawal))
        {
            return;
        }
        GroupArray lower{};
        GroupArray upper{};
        GroupArray inverseCentre{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            upper[group] = owner_.parameters_.comptonLimiterFraction * oldEnergy[group] * data.Gamma / withdrawal;
            lower[group] = -upper[group];
            inverseCentre[group] = 1.0 / owner_.comptonGroupCenters_[group];
        }
        GroupArray limited = data.D;
        // Scales = the bounds (proportional to the group energy): the adjustment that restores the moments is
        // relative, so nearly empty groups are not pinned at their bounds.
        if(!compton_thinned::ClipKeepingMoments<NumGroups>(limited, inverseCentre, lower, upper, &upper))
        {
            data.limiterInfeasible = true;
            return;
        }
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            data.thinnedLimiterDelta[group] = limited[group] - data.D[group];
            data.limiterActive = data.limiterActive || data.thinnedLimiterDelta[group] != 0.0;
            data.D[group] = limited[group];
            data.M[group] = data.absorptionOpacity[group] * data.planckFraction[group] + data.D[group];
        }
    }

    // Plan 3b: Bcorr from the time averages, affine in the hazard-weighted
    // average densities u_avg (erg/cm^3): D and Upsilon contract dSdUm
    // (d/dU_m, U_m = a T^4) with u_avg; extensive through V, current dt.
    GroupArray thinnedAverageBcorr(const ComptonCellData &data, const GroupArray &averageDensity, double fullDt) const
    {
        GroupArray bcorr{};
        if(!(data.planckOpacity > 0.0))
        {
            return bcorr;
        }
        GroupArray averageD{};
        double averageUpsilon = 0.0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            for(std::size_t source = 0; source < NumGroups; ++source)
            {
                averageD[group] += data.dSdUm[source][group] * averageDensity[source];
            }
            averageUpsilon += averageD[group];
        }
        double const cdt = owner_.lightSpeed() * fullDt;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const kgbg = data.absorptionOpacity[group] * data.planckFraction[group];
            bcorr[group] = data.volume * cdt * data.Um * data.betaCdtF *
                (kgbg * averageUpsilon - data.planckOpacity * (averageD[group] + data.thinnedLimiterDelta[group]));
        }
        return bcorr;
    }

    // postStep end: this step's hazard-weighted w^2 per unit time becomes the
    // next step's history term T^prev (erg^2, divided by time only).
    void endThinnedStep(double fullDt)
    {
        if(!this->thinnedEvents() || owner_.parameters_.postProcess.enabled)
        {
            return;
        }
        owner_.comptonPreviousWeightSquared_ = owner_.comptonHazardWeightSquared_;
        for(GroupArray &values : owner_.comptonPreviousWeightSquared_)
        {
            for(double &value : values)
            {
                value /= fullDt;
            }
        }
    }

    // Plan 3e: the accepted correction endpoint is projected onto fixed energy
    // and photon number, x_g = max(0, y_g - lambda - kappa h_g) with h = h^end.
    // Compton-only cells keep the photons counted before settlement; with
    // absorption the target adds the correction's own photon change.
    void projectThinnedPhotons(std::size_t cellIndex, const ComptonCellData &data, const GroupArray &rawGroupEnergy,
                               ComptonCorrectionResult &result) const
    {
        GroupArray const &h = owner_.comptonEndH_[cellIndex];
        double photonTarget = owner_.comptonRawPhotons_[cellIndex];
        if(data.planckOpacity > 0.0)
        {
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                photonTarget += (result.endpoint[group] - rawGroupEnergy[group]) * h[group];
            }
        }
        double const energyTarget = result.radiationTotal;
        if(!(energyTarget > 0.0))
        {
            return;
        }
        double photonsBefore = 0.0;
        double hMin = std::numeric_limits<double>::infinity();
        double hMax = 0.0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            photonsBefore += h[group] * result.endpoint[group];
            hMin = std::min(hMin, h[group]);
            hMax = std::max(hMax, h[group]);
        }
        // At the accepted energy the grid holds between E min h and E max h
        // photons (e.g. a cell whose photons all sit in the lowest group cannot
        // lose energy at fixed photon number).  Clamp to that range, just
        // inside it, and count the defect.
        compton_thinned::Diagnostics &clampDiagnostics = owner_.comptonThinnedDiagnostics_;
        constexpr double clampMargin = 1e-12;
        double const lowestPhotons = energyTarget * hMin * (1.0 + clampMargin);
        double const highestPhotons = energyTarget * hMax * (1.0 - clampMargin);
        if(lowestPhotons < highestPhotons && (photonTarget < lowestPhotons || photonTarget > highestPhotons))
        {
            double const clamped = std::clamp(photonTarget, lowestPhotons, highestPhotons);
            if(std::abs(clamped - photonTarget) > 1e-10 * photonTarget)
            {
                ++clampDiagnostics.photonClamps;
                clampDiagnostics.photonClampDefect += std::abs(clamped - photonTarget);
            }
            photonTarget = clamped;
        }
        // Relative projection first (never fills an empty group); then the Euclidean one restricted to the groups
        // that hold energy; the unrestricted Euclidean one (which can fill empty groups) only as a last resort.
        compton_thinned::Projection<NumGroups> projection =
            compton_thinned::ProjectEnergyPhotonRelative<NumGroups>(result.endpoint, h, energyTarget, photonTarget);
        if(!projection.success)
        {
            std::array<bool, NumGroups> support{};
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                support[group] = result.endpoint[group] > 0.0;
            }
            projection = compton_thinned::ProjectEnergyPhoton<NumGroups>(result.endpoint, h, energyTarget,
                                                                          photonTarget, &support);
            if(projection.success)
            {
                ++owner_.comptonThinnedDiagnostics_.projectionSupportFallbacks;
            }
            else
            {
                projection = compton_thinned::ProjectEnergyPhoton<NumGroups>(result.endpoint, h, energyTarget,
                                                                              photonTarget);
                ++owner_.comptonThinnedDiagnostics_.projectionFullFallbacks;
            }
        }
        if(!projection.success)
        {
            StormError eo("Thinned Compton correction could not meet the photon target");
            eo.addEntry("Cell index", cellIndex);
            eo.addEntry("Energy target", energyTarget);
            eo.addEntry("Photon target", photonTarget);
            eo.addEntry("Photons before projection", photonsBefore);
            eo.addEntry("Relative energy error", projection.energyError);
            eo.addEntry("Relative photon error", projection.photonError);
            throw eo;
        }
        compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
        if(cellIndex == 0 && this->lowGroupDiagnosticGroups() > 0)
        {
            std::cout << "LGDIAG projection cell " << cellIndex << " photonTarget " << photonTarget
                      << " photonsBefore " << photonsBefore << " energyTarget " << energyTarget << '\n';
            for(std::size_t group = 0; group < std::min(this->lowGroupDiagnosticGroups(), NumGroups); ++group)
            {
                std::cout << "LGDIAG pg " << group << " before " << result.endpoint[group] << " after "
                          << projection.endpoint[group] << " h " << h[group] << '\n';
            }
        }
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            diagnostics.projectionEnergyRepair += std::abs(projection.endpoint[group] - result.endpoint[group]);
            result.endpoint[group] = projection.endpoint[group];
            result.delta[group] = result.endpoint[group] - rawGroupEnergy[group];
        }
        diagnostics.projectionPhotonRepair += photonTarget - photonsBefore;
        double const projectedTotal = this->compensatedSumComptonGroups(result.endpoint);
        if(!owner_.parameters_.noHydroFeedback)
        {
            result.materialEnergyAfter += result.radiationTotal - projectedTotal;
        }
        result.radiationTotal = projectedTotal;
    }

    // Plan 5 and 4: packets of each comoving group are scaled to the accepted
    // group energy (upward at most comptonReconcileUpscaleCap); a remainder is
    // spawned at the group's photon energy 1/h = E/N (centre for an empty
    // group), so energy and actual photons per group meet the targets.  Then
    // the lab closure books the packets' lab energy and momentum change against
    // the ledger, the gas settlement and the correction's gas change.
    void reconcileThinnedParticles(std::vector<MCParticle> &particles)
    {
        if constexpr(radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
        {
            std::size_t const Ncells = owner_.componentGrid().GetPointNo();
            std::vector<GroupArray> energy(Ncells, GroupArray{});
            std::vector<GroupArray> photons(Ncells, GroupArray{});
            this->accumulateComptonCensus(particles, energy, &photons, nullptr);
            std::vector<double> labEnergyBefore(Ncells, 0.0);
            std::vector<PointT> labMomentumBefore(Ncells, PointT{});
            for(const MCParticle &particle : particles)
            {
                if(particle.cellIndex < Ncells)
                {
                    labEnergyBefore[particle.cellIndex] += particle.weight;
                    labMomentumBefore[particle.cellIndex] +=
                        particle.velocity * (particle.weight * owner_.inverseLightSpeedSquared());
                }
            }

            compton_thinned::Diagnostics &diagnostics = owner_.comptonThinnedDiagnostics_;
            std::vector<std::array<compton_thinned::GroupReconciliation, NumGroups>> plan(Ncells);
            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(0.0, owner_.extensives_[cellIndex].Eg[group]);
                    plan[cellIndex][group] = compton_thinned::ReconcileGroup(
                        energy[cellIndex][group], photons[cellIndex][group], target,
                        owner_.comptonGroupCenters_[group], owner_.parameters_.comptonReconcileUpscaleCap);
                    if(plan[cellIndex][group].packetScale > 1.0)
                    {
                        ++diagnostics.reconcileUpscales;
                    }
                }
            }
            if(this->lowGroupDiagnosticGroups() > 0 && Ncells > 0)
            {
                std::vector<std::size_t> packets(NumGroups, 0);
                for(const MCParticle &particle : particles)
                {
                    std::size_t const group = this->comptonComovingGroup(particle, this->comptonComovingDoppler(particle));
                    if(particle.cellIndex == 0 && group < NumGroups)
                    {
                        ++packets[group];
                    }
                }
                for(std::size_t group = 0; group < std::min(this->lowGroupDiagnosticGroups(), NumGroups); ++group)
                {
                    std::cout << "LGDIAG rg " << group << " packets " << packets[group] << " census "
                              << energy[0][group] << " target " << owner_.extensives_[0].Eg[group] << " scale "
                              << plan[0][group].packetScale << " spawn " << plan[0][group].spawnEnergy << '\n';
                }
            }

            auto iterator = particles.begin();
            while(iterator != particles.end())
            {
                MCParticle &particle = *iterator;
                if(particle.cellIndex < Ncells)
                {
                    std::size_t const group = this->comptonComovingGroup(
                        particle, this->comptonComovingDoppler(particle));
                    if(group < NumGroups)
                    {
                        particle.weight *= plan[particle.cellIndex][group].packetScale;
                        owner_.setInitialWeightFromWeight(particle);
                    }
                }
                iterator = particle.weight > 0.0 ? iterator + 1 : particles.erase(iterator);
            }

            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                GroupArray deficits{};
                double totalDeficit = 0.0;
                std::size_t activeGroups = 0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(0.0, owner_.extensives_[cellIndex].Eg[group]);
                    double const deficit = plan[cellIndex][group].spawnEnergy;
                    if(deficit <= 1e-12 * std::max(1.0, target))
                    {
                        continue;
                    }
                    deficits[group] = deficit;
                    totalDeficit += deficit;
                    ++activeGroups;
                }
                if(!(totalDeficit > 0.0))
                {
                    continue;
                }
                std::size_t const budget = std::max(owner_.parameters_.newPhotonsPerCell, activeGroups);
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(!(deficits[group] > 0.0))
                    {
                        continue;
                    }
                    std::size_t const packetCount = std::max<std::size_t>(
                        1, static_cast<std::size_t>(std::floor(static_cast<double>(budget) *
                                                               deficits[group] / totalDeficit)));
                    double const packetWeight = deficits[group] / static_cast<double>(packetCount);
                    for(std::size_t packetIndex = 0; packetIndex < packetCount; ++packetIndex)
                    {
                        MCParticle particle = owner_.generateSingleParticle(cellIndex, owner_.cells_[cellIndex]);
                        owner_.setPacketFromComovingState(
                            particle, owner_.cells_[cellIndex], plan[cellIndex][group].spawnPhotonEnergy,
                            packetWeight);
                        owner_.setInitialWeightFromWeight(particle);
                        particles.push_back(particle);
                        ++diagnostics.reconcileSpawns;
                    }
                }
            }

            // Check energy and actual photons per group, then close lab energy
            // and momentum: dE_gas = -dE_p - B_E - G_E - dU_corr (plan 4).
            std::vector<GroupArray> energyAfter(Ncells, GroupArray{});
            std::vector<GroupArray> photonsAfter(Ncells, GroupArray{});
            this->accumulateComptonCensus(particles, energyAfter, &photonsAfter, nullptr);
            std::vector<double> labEnergyAfter(Ncells, 0.0);
            std::vector<PointT> labMomentumAfter(Ncells, PointT{});
            for(const MCParticle &particle : particles)
            {
                if(particle.cellIndex < Ncells)
                {
                    labEnergyAfter[particle.cellIndex] += particle.weight;
                    labMomentumAfter[particle.cellIndex] +=
                        particle.velocity * (particle.weight * owner_.inverseLightSpeedSquared());
                    diagnostics.censusPhotonsFinal += particle.weight / particle.frequency;
                }
            }
            constexpr double reconciliationTolerance = 1e-8;
            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                double const cellEnergy = std::max(1e-300, this->compensatedSumComptonGroups(energyAfter[cellIndex]));
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(0.0, owner_.extensives_[cellIndex].Eg[group]);
                    double const photonTarget = target *
                        (energy[cellIndex][group] > 0.0 && photons[cellIndex][group] > 0.0
                            ? photons[cellIndex][group] / energy[cellIndex][group]
                            : 1.0 / owner_.comptonGroupCenters_[group]);
                    double const energyError = std::abs(energyAfter[cellIndex][group] - target) /
                        std::max(target, cellEnergy);
                    double const photonError = std::abs(photonsAfter[cellIndex][group] - photonTarget) /
                        std::max(photonTarget, cellEnergy / owner_.comptonGroupCenters_[group]);
                    if(energyError > reconciliationTolerance || photonError > reconciliationTolerance)
                    {
                        StormError eo("Thinned Compton reconciliation missed the group energy or photon target");
                        eo.addEntry("Cell index", cellIndex);
                        eo.addEntry("Group", group);
                        eo.addEntry("Energy target", target);
                        eo.addEntry("Energy after", energyAfter[cellIndex][group]);
                        eo.addEntry("Photon target", photonTarget);
                        eo.addEntry("Photons after", photonsAfter[cellIndex][group]);
                        throw eo;
                    }
                }
                if(!owner_.parameters_.comptonLabClosure || owner_.parameters_.noHydroFeedback)
                {
                    continue;
                }
                auto const [velocity, gamma] = this->comptonTransportFrame(cellIndex);
                double const ledger = owner_.comptonLedger_[cellIndex];
                double const gasSettlement = owner_.comptonGasSettlement_[cellIndex];
                double const closureEnergy = -(labEnergyAfter[cellIndex] - labEnergyBefore[cellIndex]) -
                    gamma * (ledger + gasSettlement) - owner_.comptonCorrectionGasChange_[cellIndex];
                PointT closureMomentum = (labMomentumBefore[cellIndex] - labMomentumAfter[cellIndex]) -
                    velocity * (gamma * (ledger + gasSettlement) * owner_.inverseLightSpeedSquared());
                if(!owner_.parameters_.withHydro || owner_.parameters_.diffusionPressureGradient)
                {
                    closureMomentum = PointT{};
                }
                diagnostics.closureEnergy += std::abs(closureEnergy);
                owner_.applyMaterialExchange(cellIndex, closureEnergy, closureMomentum);
            }
        }
        else
        {
            (void) particles;
        }
    }

    void applyComptonEndOfStepCorrection(
        double fullDt)
    {

            if(!owner_.parameters_.withCompton)
            {
                return;
            }
            if constexpr(!radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
            {
                return;
            }
            else
            {
                const std::size_t Ncells = owner_.componentGrid().GetPointNo();
                std::vector<ComptonCorrectionResult> results(Ncells);
                for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
                {
                    ComptonCellData const &data = owner_.comptonData_[cellIndex];
                    std::size_t cellID =
                        radiation_imc_detail::cellID(owner_.cells_[cellIndex]);
                    if(cellID == std::numeric_limits<std::size_t>::max())
                    {
                        cellID = cellIndex;
                    }

                    GroupArray rawGroupEnergy{};
                    GroupArray timeAvgGroupEnergy{};
                    double totalPreStepRadiation = 0.0;
                    bool const thinned = this->thinnedEvents();
                    GroupArray averageDensity{};
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        rawGroupEnergy[group] =
                            owner_.extensives_[cellIndex].Eg[group];
                        if(thinned)
                        {
                            // Hazard-weighted average (plan 3d): integral D w_c dt / dt.
                            timeAvgGroupEnergy[group] = std::max(
                                0.0, owner_.comptonHazardEnergy_[cellIndex][group] / fullDt);
                            averageDensity[group] = timeAvgGroupEnergy[group] / data.volume;
                        }
                        else if(cellIndex < owner_.Eg_time_avg_.size())
                        {
                            timeAvgGroupEnergy[group] = std::max(
                                0.0,
                                owner_.Eg_time_avg_[cellIndex][group] *
                                    data.volume);
                        }
                        totalPreStepRadiation += data.oldRadiationEnergy[group];
                    }
                    GroupArray const averageBcorr = thinned
                        ? this->thinnedAverageBcorr(data, averageDensity, fullDt) : GroupArray{};
                    double const rawTotal =
                        this->compensatedSumComptonGroups(rawGroupEnergy);
                    double const preStepExtensive =
                        totalPreStepRadiation * data.volume;
                    double const budgetBefore = owner_.parameters_.noHydroFeedback
                        ? rawTotal
                        : owner_.extensives_[cellIndex].internal_energy + rawTotal;
                    ComptonCorrectionResult result =
                        this->solveComptonCorrection(
                            cellIndex,
                            fullDt,
                            data,
                            rawGroupEnergy,
                            timeAvgGroupEnergy,
                            budgetBefore,
                            0.0,
                            preStepExtensive,
                            thinned ? &averageBcorr : nullptr);
                    if(!result.success && thinned && owner_.parameters_.comptonCorrectionBacktracking)
                    {
                        // Explicit limiter: damp the complete residual correction
                        // (matrix and drive together, so its zero row sums and the
                        // equilibrium cancellation are kept) until it is admissible.
                        compton_thinned::Diagnostics &backtrackDiagnostics = owner_.comptonThinnedDiagnostics_;
                        for(double scale = 0.5; scale >= 1.0 / 1024.0 && !result.success; scale *= 0.5)
                        {
                            ComptonCorrectionResult const firstFailure = result;
                            result = this->solveComptonCorrection(
                                cellIndex, fullDt, data, rawGroupEnergy, timeAvgGroupEnergy, budgetBefore, 0.0,
                                preStepExtensive, &averageBcorr, scale);
                            ++backtrackDiagnostics.correctionBacktracks;
                            if(result.success)
                            {
                                backtrackDiagnostics.correctionScaleMin =
                                    std::min(backtrackDiagnostics.correctionScaleMin, scale);
                            }
                            else
                            {
                                result.directWorstNegative = firstFailure.directWorstNegative;
                                result.directNegativeMass = firstFailure.directNegativeMass;
                            }
                        }
                    }
                    if(result.success && thinned)
                    {
                        this->projectThinnedPhotons(cellIndex, data, rawGroupEnergy, result);
                    }

                    if(!result.success)
                    {
                        StormError eo(
                            "Compton correction failed to integrate the complete timestep; "
                            "no partial correction was committed");
                        eo.addEntry("Cell index", cellIndex);
                        eo.addEntry("Cell ID", cellID);
                        eo.addEntry("Full dt", fullDt);
                        eo.addEntry(
                            "Failure",
                            this->comptonCorrectionFailureName(
                                result.failure));
                        eo.addEntry("Direct worst negative group", result.directWorstNegative);
                        eo.addEntry("Direct negative mass", result.directNegativeMass);
                        eo.addEntry("Direct relative residual", result.directResidual);
                        eo.addEntry("Direct endpoint total", result.directTotal);
                        eo.addEntry("Last correction scale", result.correctionScale);
                        eo.addEntry("Fleck", data.fleck);
                        eo.addEntry("Gamma", data.Gamma);
                        eo.addEntry("Upsilon", data.Upsilon);
                        eo.addEntry("Thinned limiter active", data.limiterActive);
                        eo.addEntry("Thinned limiter infeasible", data.limiterInfeasible);
                        double minimumDrive = 0.0;
                        double maximumDrive = 0.0;
                        std::size_t worstGroup = 0;
                        for(std::size_t group = 0; group < NumGroups; ++group)
                        {
                            // Relative to the group, but not below 1e-12 of the cell's radiation (empty groups).
                            double const relative = (thinned ? averageBcorr[group] : data.Bcorr[group]) /
                                std::max(rawGroupEnergy[group], std::max(1e-12 * rawTotal, 1e-300));
                            if(relative < minimumDrive)
                            {
                                minimumDrive = relative;
                                worstGroup = group;
                            }
                            maximumDrive = std::max(maximumDrive, relative);
                        }
                        eo.addEntry("Most negative Bcorr/E_raw", minimumDrive);
                        eo.addEntry("At group", worstGroup);
                        eo.addEntry("Largest Bcorr/E_raw", maximumDrive);
                        eo.addEntry("Raw radiation", rawTotal);
                        eo.addEntry("Internal energy", owner_.extensives_[cellIndex].internal_energy);
                        throw eo;
                    }
                    results[cellIndex] = result;
                }

                for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
                {
                    ComptonCorrectionResult const &result = results[cellIndex];
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        owner_.extensives_[cellIndex].Eg[group] =
                            result.endpoint[group];
                    }
                    radiation_imc_detail::setRadiationEnergyIfPresent(
                        owner_.extensives_[cellIndex], result.radiationTotal);
                    if(cellIndex < owner_.comptonCorrectionGasChange_.size())
                    {
                        owner_.comptonCorrectionGasChange_[cellIndex] =
                            result.materialEnergyAfter - result.materialEnergyBefore;
                    }
                    if(!owner_.parameters_.noHydroFeedback)
                    {
                        owner_.extensives_[cellIndex].internal_energy =
                            result.materialEnergyAfter;
                        radiation_imc_detail::addTotalEnergyIfPresent(
                            owner_.extensives_[cellIndex],
                            result.materialEnergyAfter -
                                result.materialEnergyBefore);
                    }
                }
            } // else (has_member_group_energy_mutable)
    }

    void reconcileComptonParticles(
        std::vector<MCParticle> &particles)
    {

            if(!owner_.parameters_.withCompton)
            {
                return;
            }
            if(this->thinnedEvents())
            {
                this->reconcileThinnedParticles(particles);
                return;
            }
            if constexpr(!radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
            {
                return;
            }
            else
            {
            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            // Group energies are comoving: sum comoving packet weights by comoving group.
            std::vector<GroupArray> raw(Ncells, GroupArray{});
            for(const MCParticle &particle : particles)
            {
                if(particle.cellIndex >= Ncells)
                {
                    continue;
                }
                if(particle.weight < 0.0)
                {
                    throw StormError(
                        "Negative particle weight in positive-only Compton reconciliation");
                }
                double const dopplerShift = this->comptonComovingDoppler(particle);
                std::size_t const group = this->comptonComovingGroup(particle, dopplerShift);
                if(group < NumGroups)
                {
                    raw[particle.cellIndex][group] += particle.weight * dopplerShift;
                }
            }

            std::vector<GroupArray> scale(Ncells, GroupArray{});
            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(
                        0.0, owner_.extensives_[cellIndex].Eg[group]);
                    scale[cellIndex][group] =
                        raw[cellIndex][group] > 0.0 && target < raw[cellIndex][group]
                        ? target / raw[cellIndex][group] : 1.0;
                }
            }

            auto iterator = particles.begin();
            while(iterator != particles.end())
            {
                MCParticle &particle = *iterator;
                if(particle.cellIndex < Ncells)
                {
                    std::size_t const group = this->comptonComovingGroup(
                        particle, this->comptonComovingDoppler(particle));
                    if(group < NumGroups)
                    {
                        particle.weight *= scale[particle.cellIndex][group];
                        owner_.setInitialWeightFromWeight(particle);
                    }
                }
                if(!(particle.weight > 0.0))
                {
                    iterator = particles.erase(iterator);
                }
                else
                {
                    ++iterator;
                }
            }

            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                GroupArray deficits{};
                GroupArray fractional{};
                std::array<std::size_t, NumGroups> groupCounts{};
                double totalDeficit = 0.0;
                std::size_t activeDeficitGroups = 0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(
                        0.0, owner_.extensives_[cellIndex].Eg[group]);
                    double const represented = raw[cellIndex][group] > target
                        ? target : raw[cellIndex][group];
                    double const deficit = target - represented;
                    if(deficit <= 1e-12 * std::max(1.0, target))
                    {
                        continue;
                    }
                    deficits[group] = deficit;
                    totalDeficit += deficit;
                    ++activeDeficitGroups;
                }
                if(!(totalDeficit > 0.0))
                {
                    continue;
                }

                std::size_t const packetBudget = std::max(
                    owner_.parameters_.newPhotonsPerCell,
                    activeDeficitGroups);
                std::size_t allocated = 0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(deficits[group] > 0.0)
                    {
                        groupCounts[group] = 1;
                        ++allocated;
                    }
                }
                std::size_t const remainingBudget = packetBudget - allocated;
                std::size_t extraAllocated = 0;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    if(!(deficits[group] > 0.0))
                    {
                        continue;
                    }
                    double const exactExtra = static_cast<double>(remainingBudget) *
                        deficits[group] / totalDeficit;
                    std::size_t const extra =
                        static_cast<std::size_t>(std::floor(exactExtra));
                    groupCounts[group] += extra;
                    fractional[group] = exactExtra - static_cast<double>(extra);
                    extraAllocated += extra;
                }
                while(extraAllocated < remainingBudget)
                {
                    std::size_t bestGroup = NumGroups;
                    double bestFraction = -1.0;
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        if(deficits[group] > 0.0 &&
                           fractional[group] > bestFraction)
                        {
                            bestGroup = group;
                            bestFraction = fractional[group];
                        }
                    }
                    if(bestGroup == NumGroups)
                    {
                        break;
                    }
                    ++groupCounts[bestGroup];
                    fractional[bestGroup] = 0.0;
                    ++extraAllocated;
                }

                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    std::size_t const packetCount = groupCounts[group];
                    if(packetCount == 0)
                    {
                        continue;
                    }
                    double const packetWeight = deficits[group] /
                        static_cast<double>(packetCount);
                    for(std::size_t packetIndex = 0;
                        packetIndex < packetCount; ++packetIndex)
                    {
                        MCParticle particle = owner_.generateSingleParticle(
                            cellIndex, owner_.cells_[cellIndex]);
                        owner_.setPacketFromComovingState(
                            particle,
                            owner_.cells_[cellIndex],
                            owner_.frequencyForComptonGroup(group),
                            packetWeight);
                        owner_.setInitialWeightFromWeight(particle);
                        particles.push_back(particle);
                    }
                }
            }

            std::vector<GroupArray> representedAfter(Ncells, GroupArray{});
            for(const MCParticle &particle : particles)
            {
                if(particle.cellIndex >= Ncells)
                {
                    continue;
                }
                double const dopplerShift = this->comptonComovingDoppler(particle);
                std::size_t const group = this->comptonComovingGroup(particle, dopplerShift);
                if(group < NumGroups)
                {
                    representedAfter[particle.cellIndex][group] += particle.weight * dopplerShift;
                }
            }
            constexpr double reconciliationTolerance = 1e-8;
            for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
            {
                double maxRelativeError = 0.0;
                double targetTotal = 0.0;
                double actualTotal = 0.0;
                std::size_t worstGroup = NumGroups;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    double const target = std::max(
                        0.0, owner_.extensives_[cellIndex].Eg[group]);
                    double const actual = representedAfter[cellIndex][group];
                    targetTotal += target;
                    actualTotal += actual;
                    double const scale = std::max(
                        {1.0, std::abs(target), std::abs(actual)});
                    double const relativeError =
                        std::abs(actual - target) / scale;
                    if(relativeError > maxRelativeError)
                    {
                        maxRelativeError = relativeError;
                        worstGroup = group;
                    }
                }
                double const totalScale = std::max(
                    {1.0, std::abs(targetTotal), std::abs(actualTotal)});
                double const totalRelativeError =
                    std::abs(actualTotal - targetTotal) / totalScale;
                maxRelativeError = std::max(maxRelativeError, totalRelativeError);
                if(maxRelativeError > reconciliationTolerance)
                {
                    StormError eo(
                        "Compton particle reconciliation did not reproduce the "
                        "accepted deterministic endpoint");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Cell ID",
                                radiation_imc_detail::cellID(owner_.cells_[cellIndex]));
                    eo.addEntry("Worst group", worstGroup);
                    if(worstGroup < NumGroups)
                    {
                        eo.addEntry("Worst group target", std::max(
                            0.0, owner_.extensives_[cellIndex].Eg[worstGroup]));
                        eo.addEntry("Worst group actual",
                            representedAfter[cellIndex][worstGroup]);
                        eo.addEntry("Worst group raw census", raw[cellIndex][worstGroup]);
                    }
                    eo.addEntry("Maximum relative error", maxRelativeError);
                    eo.addEntry("Target total", targetTotal);
                    eo.addEntry("Actual total", actualTotal);
                    eo.addEntry("Tolerance", reconciliationTolerance);
                    throw eo;
                }
            }
            } // else (has_member_group_energy_mutable)
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_COMPTON_PROCESS_HPP
