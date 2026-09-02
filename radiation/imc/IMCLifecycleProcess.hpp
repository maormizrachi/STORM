#ifndef STORM_RADIATION_IMCLIFECYCLE_PROCESS_HPP
#define STORM_RADIATION_IMCLIFECYCLE_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"
#include "../../gpu/ProfileRegion.hpp"

namespace STORM::radiation_imc_detail {

template<typename Owner>
class IMCLifecycleProcess final : public IMCComponentBase<Owner>
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
    explicit IMCLifecycleProcess(Owner &owner) : Base(owner)
    {}

    double randomUnitOpen()
    {
        double value = owner_.dist_(owner_.rng_);
        if(value <= 0.0)
        {
            return std::numeric_limits<double>::min();
        }
        if(value >= 1.0)
        {
            return std::nextafter(1.0, 0.0);
        }
        return value;
    }

    void validateGridSizedState() const
    {

            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            if(owner_.cells_.size() < Ncells)
            {
                StormError eo("RadiationIMC cells vector is smaller than the grid cell count");
                eo.addEntry("Grid cells", Ncells);
                eo.addEntry("Cells size", owner_.cells_.size());
                throw eo;
            }
            if(owner_.extensives_.size() < Ncells)
            {
                StormError eo("RadiationIMC extensives vector is smaller than the grid cell count");
                eo.addEntry("Grid cells", Ncells);
                eo.addEntry("Extensives size", owner_.extensives_.size());
                throw eo;
            }
    }

    void validateEnergyBoundaries() const
    {

            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                if(!std::isfinite(owner_.energyBoundaries_[g]) ||
                   !std::isfinite(owner_.energyBoundaries_[g + 1]) ||
                   owner_.energyBoundaries_[g + 1] <= owner_.energyBoundaries_[g])
                {
                    StormError eo("RadiationIMC energy boundaries must be finite and strictly increasing");
                    eo.addEntry("Group", g);
                    eo.addEntry("Lower", owner_.energyBoundaries_[g]);
                    eo.addEntry("Upper", owner_.energyBoundaries_[g + 1]);
                    throw eo;
                }
            }
    }

    void rejectUnsupportedParameter(const std::string &name) const
    {

            StormError eo("RadiationIMC option is planned but not implemented in the initial STORM port");
            eo.addEntry("Unsupported option", name);
            throw eo;
    }

    void rejectUnsupportedParameters() const
    {

            if(owner_.parameters_.withCompton && owner_.parameters_.withDDMC)
            {
                StormError eo("RadiationIMC configuration is invalid: Compton and DDMC are incompatible");
                eo.addEntry("withCompton", true);
                eo.addEntry("withDDMC", true);
                eo.addEntry("Reason", "Compton group-changing transport has no DDMC derivation");
                throw eo;
            }
            if(owner_.parameters_.withDDMC && !owner_.componentBoundary())
            {
                StormError eo("RadiationIMC DDMC requires a boundary-condition object");
                eo.addEntry("Reason", "DDMC precompute must classify every external face");
                throw eo;
            }
            if(owner_.parameters_.withRandomWalk &&
               (!std::isfinite(owner_.parameters_.rwMinCellOpticalDepth) ||
                owner_.parameters_.rwMinCellOpticalDepth <= 0.0))
            {
                StormError eo("RadiationIMC random-walk cell optical-depth threshold must be finite and positive");
                eo.addEntry("rwMinCellOpticalDepth", owner_.parameters_.rwMinCellOpticalDepth);
                throw eo;
            }
            if(owner_.parameters_.withRandomWalk &&
               (!std::isfinite(owner_.parameters_.rwMinParticleOpticalDepth) ||
                owner_.parameters_.rwMinParticleOpticalDepth <= 0.0))
            {
                StormError eo("RadiationIMC random-walk particle optical-depth threshold must be finite and positive");
                eo.addEntry("rwMinParticleOpticalDepth", owner_.parameters_.rwMinParticleOpticalDepth);
                throw eo;
            }
            if(owner_.parameters_.withDDMC &&
               (!std::isfinite(owner_.parameters_.ddmcMinCellOpticalDepth) ||
                owner_.parameters_.ddmcMinCellOpticalDepth <= 0.0))
            {
                StormError eo("RadiationIMC DDMC cell optical-depth threshold must be finite and positive");
                eo.addEntry("ddmcMinCellOpticalDepth", owner_.parameters_.ddmcMinCellOpticalDepth);
                throw eo;
            }
            if(owner_.parameters_.withDDMC &&
               (!std::isfinite(owner_.parameters_.ddmcMinParticleOpticalDepth) ||
                owner_.parameters_.ddmcMinParticleOpticalDepth <= 0.0))
            {
                StormError eo("RadiationIMC DDMC particle optical-depth threshold must be finite and positive");
                eo.addEntry("ddmcMinParticleOpticalDepth", owner_.parameters_.ddmcMinParticleOpticalDepth);
                throw eo;
            }
            if(owner_.parameters_.withDDMC &&
               (!std::isfinite(
                    owner_.parameters_.ddmcExternalSourceMinFaceOpticalDepth) ||
                owner_.parameters_.ddmcExternalSourceMinFaceOpticalDepth <= 0.0))
            {
                StormError eo(
                    "RadiationIMC DDMC external-source face optical-depth threshold must be finite and positive");
                eo.addEntry("ddmcExternalSourceMinFaceOpticalDepth",
                            owner_.parameters_.ddmcExternalSourceMinFaceOpticalDepth);
                throw eo;
            }
            if(owner_.parameters_.withDDMC &&
               (!(owner_.parameters_.ddmcMaxInterfaceVelocityOverC > 0.0) ||
                !std::isfinite(owner_.parameters_.ddmcMaxInterfaceVelocityOverC) ||
                !(owner_.parameters_.ddmcInterfaceTargetWeightRatio > 0.0) ||
                !std::isfinite(owner_.parameters_.ddmcInterfaceTargetWeightRatio) ||
                owner_.parameters_.ddmcMaxInterfaceSplits == 0 ||
                owner_.parameters_.ddmcMaxGroupCutoff == 0 ||
                owner_.parameters_.ddmcMaxGroupCutoff > NumGroups))
            {
                throw StormError(
                    "RadiationIMC DDMC interface controls are outside their valid ranges");
            }
            if(owner_.parameters_.withDDMC &&
               (!std::isfinite(owner_.parameters_.ddmcMaxMovingInterfaceWeightCorrection) ||
                owner_.parameters_.ddmcMaxMovingInterfaceWeightCorrection <= 0.0))
            {
                StormError eo("RadiationIMC DDMC moving-interface weight correction cap must be finite and positive");
                eo.addEntry("ddmcMaxMovingInterfaceWeightCorrection",
                            owner_.parameters_.ddmcMaxMovingInterfaceWeightCorrection);
                throw eo;
            }
            if(owner_.parameters_.withDDMC && owner_.parameters_.ddmcUseMultigroupPGRW &&
               !owner_.parameters_.withMultigroupOpacity)
            {
                rejectUnsupportedParameter("ddmcUseMultigroupPGRW requires withMultigroupOpacity");
            }
            if(owner_.parameters_.withCompton && !owner_.parameters_.withMultigroupOpacity)
            {
                StormError eo("RadiationIMC Compton transport requires multigroup opacity");
                eo.addEntry("withCompton", true);
                eo.addEntry("withMultigroupOpacity", false);
                throw eo;
            }
            if(owner_.parameters_.withCompton && owner_.parameters_.withRandomWalk)
            {
                StormError eo("RadiationIMC configuration is invalid: Compton and random walk are incompatible");
                eo.addEntry("withCompton", true);
                eo.addEntry("withRandomWalk", true);
                eo.addEntry("Reason", "Compton event kernels are not represented by the random-walk closure");
                throw eo;
            }
            if(owner_.parameters_.postProcess.polarization.enabled &&
               !owner_.parameters_.postProcess.enabled &&
               !owner_.parameters_.withPolarization)
            {
                throw StormError("RadiationIMC post-process polarization requires postProcess.enabled");
            }
            if(owner_.polarizationEnabled())
            {
                if(owner_.parameters_.withCompton)
                {
                    throw StormError("RadiationIMC polarization does not support Compton transport yet");
                }
        #ifndef MONTECARLO_POLARIZATION
                throw StormError("RadiationIMC polarization requires a build with MONTECARLO_POLARIZATION");
        #else
                const typename Parameters::PostProcessParameters::PolarizationParameters &polarization =
                    owner_.parameters_.postProcess.polarization;
                if(polarization.manualScatteringsAfterAcceleration < 0 ||
                   polarization.manualScatteringsAfterAcceleration > 128)
                {
                    StormError eo("RadiationIMC polarization manual scatter count must be in [0, 128]");
                    eo.addEntry("manualScatteringsAfterAcceleration",
                                polarization.manualScatteringsAfterAcceleration);
                    throw eo;
                }
                if(!std::isfinite(polarization.depolarizationScatterings) ||
                   polarization.depolarizationScatterings <= 0.0)
                {
                    StormError eo("RadiationIMC polarization depolarizationScatterings must be finite and positive");
                    eo.addEntry("depolarizationScatterings", polarization.depolarizationScatterings);
                    throw eo;
                }
                if(polarization.acceleratedClosure != "damped_last_scatterings")
                {
                    StormError eo("RadiationIMC polarization acceleratedClosure is unsupported");
                    eo.addEntry("acceleratedClosure", polarization.acceleratedClosure);
                    throw eo;
                }
        #endif
            }
            if(owner_.parameters_.postProcess.enabled)
            {
                if(!std::isfinite(owner_.parameters_.postProcess.sourceDt) ||
                   owner_.parameters_.postProcess.sourceDt <= 0.0)
                {
                    StormError eo("RadiationIMC post-process sourceDt must be finite and positive");
                    eo.addEntry("sourceDt", owner_.parameters_.postProcess.sourceDt);
                    throw eo;
                }
                if(!std::isfinite(owner_.parameters_.postProcess.transportTime) ||
                   owner_.parameters_.postProcess.transportTime <= 0.0)
                {
                    StormError eo("RadiationIMC post-process transportTime must be finite and positive");
                    eo.addEntry("transportTime", owner_.parameters_.postProcess.transportTime);
                    throw eo;
                }
            }
    }

    double randomUnitOpen(MCParticle &particle)
    {

            if(particle.rngKey == std::numeric_limits<std::uint64_t>::max())
            {
                std::uint64_t creationRank = 0;
        #ifdef STORM_WITH_MPI
                creationRank = static_cast<std::uint64_t>(
                    std::max<rank_t>(particle.rank, 0));
        #endif
                particle.rngKey = CounterRNG::makeKey(
                    owner_.particleRngSeed_, creationRank,
                    static_cast<std::uint64_t>(particle.id));
                particle.rngCounter = 0;
            }
            return CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    }

    void initializeParticleRNG(MCParticle &particle)
    {

            // Queried once: this runs per emitted particle, and the rank cannot change.
            if(!owner_.creationRankCached_)
            {
        #ifdef STORM_WITH_MPI
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                owner_.creationRank_ = static_cast<std::uint64_t>(rank);
        #else
                owner_.creationRank_ = 0;
        #endif
                owner_.creationRankCached_ = true;
            }
            particle.rngKey = CounterRNG::makeKey(
                owner_.particleRngSeed_, owner_.creationRank_, owner_.sourceRngStreamCounter_++);
            particle.rngCounter = 0;
    }

    PointT sampleRandomVelocity(
        const CellT &cell, MCParticle &particle)
    {

            const double random1 = owner_.randomUnitOpen(particle);
            const double random2 = owner_.randomUnitOpen(particle);
            return owner_.opacity_->getRandomVelocity(cell, random1, random2);
    }

    PointT sampleScatterVelocity(
        const CellT &cell, MCParticle &particle)
    {

            const double random1 = owner_.randomUnitOpen(particle);
            const double random2 = owner_.randomUnitOpen(particle);
            return owner_.opacity_->getNewScatterVelocity(
                cell, particle.velocity, particle.frequency, random1, random2);
    }

    void resetTransportTallies(std::size_t cellCount)
    {

            owner_.pendingMaterialEnergy_.assign(cellCount, 0.0);
            owner_.pendingTotalEnergy_.assign(cellCount, 0.0);
            owner_.pendingMomentum_.assign(cellCount, PointT{});
            owner_.transportCellVelocities_.assign(cellCount, PointT{});
            if constexpr(
                radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                for(std::size_t i = 0; i < cellCount; ++i)
                {
                    owner_.transportCellVelocities_[i] =
                        owner_.cells_[i].velocity;
                }
            }
            owner_.pendingRadiationEnergy_.assign(cellCount, 0.0);
            owner_.pendingGroupRadiationEnergy_.assign(
                cellCount * NumGroups, 0.0);
            owner_.spectralAbsorptionScale_.assign(cellCount, 0.0);
            owner_.thermalEmissionCdf_.assign(
                cellCount * (NumGroups + 1), 0.0);
    }

    void tallyMaterialEnergy(
        std::size_t cellIndex, double energy, bool addToTotalEnergy)
    {

            owner_.pendingMaterialEnergy_[cellIndex] += energy;
            if(addToTotalEnergy)
            {
                owner_.pendingTotalEnergy_[cellIndex] += energy;
            }
    }

    void tallyMomentum(
        std::size_t cellIndex, const PointT &momentum)
    {

            owner_.pendingMomentum_[cellIndex] += momentum;
    }

    void tallyRadiationEnergy(
        std::size_t cellIndex, double integratedEnergy)
    {

            owner_.pendingRadiationEnergy_[cellIndex] += integratedEnergy;
    }

    void tallyGroupRadiationEnergy(
        std::size_t cellIndex, std::size_t group, double integratedEnergy)
    {

            owner_.pendingGroupRadiationEnergy_[
                cellIndex * NumGroups + group] += integratedEnergy;
    }

    void applyTransportTallies()
    {

            for(std::size_t i = 0; i < owner_.pendingMaterialEnergy_.size(); ++i)
            {
                owner_.extensives_[i].internal_energy += owner_.pendingMaterialEnergy_[i];
                radiation_imc_detail::addTotalEnergyIfPresent(
                    owner_.extensives_[i], owner_.pendingTotalEnergy_[i]);
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    owner_.extensives_[i].momentum += owner_.pendingMomentum_[i];
                }
                owner_.Erad_time_avg_[i] += owner_.pendingRadiationEnergy_[i];
                if((owner_.parameters_.withEgTimeAvg ||
                    owner_.parameters_.withCompton) &&
                   owner_.parameters_.withMultigroupOpacity)
                {
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        owner_.Eg_time_avg_[i][group] +=
                            owner_.pendingGroupRadiationEnergy_[
                                i * NumGroups + group];
                    }
                }
            }
            owner_.pendingMaterialEnergy_.clear();
            owner_.pendingTotalEnergy_.clear();
            owner_.pendingMomentum_.clear();
            owner_.pendingRadiationEnergy_.clear();
            owner_.pendingGroupRadiationEnergy_.clear();
    }

    void setInitialWeightFromWeight(MCParticle &particle) const
    {

            particle.initialWeight = std::abs(particle.weight);
    }

    double density(std::size_t cellIndex) const
    {

            if constexpr(radiation_imc_detail::has_member_density<CellT>::value)
            {
                return owner_.cells_[cellIndex].density;
            }
            else
            {
                static_assert(radiation_imc_detail::has_member_mass<ExtensivesT>::value,
                              "RadiationIMC requires CellT::density or ExtensivesT::mass");
                return owner_.extensives_[cellIndex].mass / owner_.componentGrid().GetVolume(cellIndex);
            }
    }

    double specificInternalEnergy(std::size_t cellIndex) const
    {

            static_assert(radiation_imc_detail::has_member_mass<ExtensivesT>::value,
                          "RadiationIMC requires ExtensivesT::mass for specific internal energy");
            return owner_.extensives_[cellIndex].internal_energy / owner_.extensives_[cellIndex].mass;
    }

    double totalRadiationEnergy(std::size_t cellIndex) const
    {

            if constexpr(radiation_imc_detail::has_member_radiation_energy<CellT>::value &&
                         radiation_imc_detail::has_member_density<CellT>::value)
            {
                return owner_.cells_[cellIndex].Erad * owner_.cells_[cellIndex].density * owner_.componentGrid().GetVolume(cellIndex);
            }
            else
            {
                const double extensiveRadiation = radiation_imc_detail::radiationEnergyIfPresent(owner_.extensives_[cellIndex]);
                if(extensiveRadiation > 0.0)
                {
                    return extensiveRadiation;
                }
                if constexpr(radiation_imc_detail::has_member_radiation_energy<CellT>::value &&
                             radiation_imc_detail::has_member_mass<ExtensivesT>::value)
                {
                    return owner_.cells_[cellIndex].Erad * owner_.extensives_[cellIndex].mass;
                }
                else
                {
                    return 0.0;
                }
            }
    }

    void throwIfNegativeInternalEnergy(std::size_t cellIndex, const std::string &where)
    {

            double &E = owner_.extensives_[cellIndex].internal_energy;
            if(E >= 0.0)
            {
                return;
            }
            double volume = owner_.componentGrid().GetVolume(cellIndex);
            double thermalScale = units::arad * std::pow(owner_.cells_[cellIndex].temperature, 4) * volume;
            if(thermalScale < 1e-30)
            {
                thermalScale = 1e-30;
            }
            double ratio = std::abs(E) / thermalScale;
            if(ratio < 0.1)
            {
                // A small negative excursion is Monte-Carlo roundoff/noise.  Do not
                // turn it into a zero-temperature cell: that changes the opacity by
                // many orders of magnitude (the Marshak opacity is proportional to
                // T^-4.5) and creates isolated hot/cold points in profiles.
                // RadiationCell keeps the material energy from before this step, so
                // use it as the positivity floor when available.
                if constexpr(radiation_imc_detail::has_member_internal_energy_density<CellT>::value)
                {
                    E = std::max(0.0, owner_.cells_[cellIndex].internalEnergy);
                }
                else
                {
                    E = 0.0;
                }
                return;
            }

            StormError eo("Negative material internal energy in RadiationIMC");
            eo.addEntry("Where", where);
            eo.addEntry("Cell index", cellIndex);
            eo.addEntry("Cell ID", radiation_imc_detail::cellID(owner_.cells_[cellIndex]));
            eo.addEntry("Internal energy", E);
            if constexpr(radiation_imc_detail::has_member_mass<ExtensivesT>::value)
            {
                eo.addEntry("Mass", owner_.extensives_[cellIndex].mass);
            }
            eo.addEntry("Density", owner_.density(cellIndex));
            eo.addEntry("Temperature", owner_.cells_[cellIndex].temperature);
            throw eo;
    }

    void depositMaterialEnergy(std::size_t cellIndex, double energy)
    {

            if(owner_.parameters_.noHydroFeedback)
            {
                return;
            }
            owner_.tallyMaterialEnergy(cellIndex, energy);
    }

    void synchronizeMaterialCell(std::size_t cellIndex)
    {

            CellT &cell = owner_.cells_[cellIndex];
            const double volume = owner_.componentGrid().GetVolume(cellIndex);
            const double specificEnergy = owner_.specificInternalEnergy(cellIndex);

            if constexpr(radiation_imc_detail::has_member_internal_energy_specific<CellT>::value)
            {
                cell.internal_energy = specificEnergy;
            }
            else if constexpr(radiation_imc_detail::has_member_internal_energy_density<CellT>::value)
            {
                cell.internalEnergy = owner_.extensives_[cellIndex].internal_energy;
            }

            const auto &tracers = owner_.traits_.tracers(cell);
            const auto &tracerNames = owner_.traits_.tracerNames(cell);
            cell.temperature = owner_.eos_->de2T(owner_.density(cellIndex), specificEnergy, tracers, tracerNames);
            if constexpr(radiation_imc_detail::has_member_pressure<CellT>::value)
            {
                cell.pressure = owner_.eos_->de2p(owner_.density(cellIndex), specificEnergy, tracers, tracerNames);
            }
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value && radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                if(owner_.parameters_.withHydro)
                {
                    cell.velocity = owner_.extensives_[cellIndex].momentum / owner_.extensives_[cellIndex].mass;
                    if constexpr(radiation_imc_detail::has_member_total_energy<ExtensivesT>::value)
                    {
                        owner_.extensives_[cellIndex].energy = owner_.extensives_[cellIndex].internal_energy
                            + 0.5 * ScalarProd(owner_.extensives_[cellIndex].momentum, owner_.extensives_[cellIndex].momentum)
                            / owner_.extensives_[cellIndex].mass;
                    }
                }
            }
    }

    void clampFrequencyToBounds(double &frequency) const
    {

            frequency = std::clamp(frequency, owner_.energyBoundaries_[0], owner_.energyBoundaries_[NumGroups]);
    }

    void setNewPhotonsPerCell(std::size_t n)
    {

            owner_.parameters_.newPhotonsPerCell = n;
    }

    void setAdaptiveSourceCellScores(
        std::unordered_map<std::size_t, double> scores, double strength, double maxFactor,
        double learnedReserveFrac, double learnedMinFactor,
        double observerBudgetMultiplier, std::size_t learnedMinPhotons,
        std::size_t learnedMaxPhotons, double scorePower)
    {

            owner_.adaptiveSourceScores_ = std::move(scores);
            owner_.adaptiveSourceStrength_ = std::clamp(strength, 0.0, 1.0);
            owner_.adaptiveSourceMaxFactor_ = std::max(1.0, maxFactor);
            owner_.adaptiveSourceLearnedReserveFrac_ = std::clamp(learnedReserveFrac, 0.0, 1.0);
            owner_.adaptiveSourceLearnedMinFactor_ = std::max(1.0, learnedMinFactor);
            owner_.adaptiveSourceObserverBudgetMultiplier_ = std::max(1.0, observerBudgetMultiplier);
            owner_.adaptiveSourceLearnedMinPhotons_ = learnedMinPhotons;
            owner_.adaptiveSourceLearnedMaxPhotons_ = learnedMaxPhotons;
            owner_.adaptiveSourceScorePower_ =
                (scorePower > 0.0 && std::isfinite(scorePower)) ? scorePower : 1.0;
            owner_.adaptiveSourceScoresEnabled_ = !owner_.adaptiveSourceScores_.empty();
    }

    void clearAdaptiveSourceCellScores()
    {

            owner_.adaptiveSourceScores_.clear();
            owner_.adaptiveSourceScoresEnabled_ = false;
            owner_.adaptiveSourceStrength_ = 0.0;
            owner_.adaptiveSourceMaxFactor_ = 1.0;
            owner_.adaptiveSourceLearnedReserveFrac_ = 0.0;
            owner_.adaptiveSourceLearnedMinFactor_ = 1.0;
            owner_.adaptiveSourceObserverBudgetMultiplier_ = 1.0;
            owner_.adaptiveSourceLearnedMinPhotons_ = 0;
            owner_.adaptiveSourceLearnedMaxPhotons_ = 0;
            owner_.adaptiveSourceScorePower_ = 1.0;
    }

    void setAdaptiveSourceCellGroupScores(
        std::unordered_map<std::size_t, GroupArray> scores, double strength, double pdfFloor, double maxBias, double maxWeightCorrection)
    {

            owner_.adaptiveSourceCellGroupScores_ = std::move(scores);
            owner_.adaptiveGroupStrength_ = std::clamp(strength, 0.0, 1.0);
            owner_.adaptiveGroupPdfFloor_ = std::clamp(pdfFloor, 0.0, 1.0);
            owner_.adaptiveGroupMaxBias_ = std::max(1.0, maxBias);
            owner_.adaptiveGroupMaxWeightCorrection_ = std::max(1.0, maxWeightCorrection);
            owner_.adaptiveSourceCellGroupScoresEnabled_ = !owner_.adaptiveSourceCellGroupScores_.empty() && owner_.adaptiveGroupStrength_ > 0.0;
    }

    void clearAdaptiveSourceCellGroupScores()
    {

            owner_.adaptiveSourceCellGroupScores_.clear();
            owner_.adaptiveSourceCellGroupScoresEnabled_ = false;
            owner_.adaptiveGroupStrength_ = 0.0;
            owner_.adaptiveGroupPdfFloor_ = 0.0;
            owner_.adaptiveGroupMaxBias_ = 1.0;
            owner_.adaptiveGroupMaxWeightCorrection_ = 1.0;
    }

    void setSourceEmissionControl(
        bool useLearnedScores, bool includeUniformBase, std::size_t baseMultiplier, std::size_t learnedBoostFactor, std::size_t learnedExtraBudget)
    {

            owner_.sourceEmissionControlEnabled_ = true;
            owner_.sourceEmissionUseLearnedScores_ = useLearnedScores;
            owner_.sourceEmissionIncludeUniformBase_ = includeUniformBase;
            owner_.sourceEmissionBaseMultiplier_ = std::max<std::size_t>(1, baseMultiplier);
            owner_.sourceEmissionLearnedBoostFactor_ = std::max<std::size_t>(1, learnedBoostFactor);
            owner_.sourceEmissionLearnedExtraBudget_ = learnedExtraBudget;
    }

    void clearSourceEmissionControl()
    {

            owner_.sourceEmissionControlEnabled_ = false;
            owner_.sourceEmissionUseLearnedScores_ = false;
            owner_.sourceEmissionIncludeUniformBase_ = true;
            owner_.sourceEmissionBaseMultiplier_ = 1;
            owner_.sourceEmissionLearnedBoostFactor_ = 20;
            owner_.sourceEmissionLearnedExtraBudget_ = 0;
    }

    double preparePreStepTables(double fullDt)
    {

            if(!std::isfinite(fullDt) || fullDt <= 0.0)
            {
                StormError eo("RadiationIMC::preStep requires a finite, positive timestep");
                eo.addEntry("Full dt", fullDt);
                throw eo;
            }
            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            owner_.resetTransportTallies(Ncells);
            double const fleckDt = owner_.parameters_.postProcess.enabled
                ? owner_.parameters_.postProcess.sourceDt : fullDt;
            double const sourceDt = owner_.parameters_.postProcess.enabled
                ? owner_.parameters_.postProcess.sourceDt : fullDt;
            bool const reuseComptonPrecompute =
                !owner_.parameters_.postProcess.enabled &&
                owner_.parameters_.withCompton &&
                owner_.parameters_.withMultigroupOpacity &&
                owner_.comptonDataReusableInPreStep_ &&
                owner_.comptonData_.size() == Ncells &&
                owner_.factorFleck_.size() == Ncells &&
                owner_.planckOpacities_.size() == Ncells &&
                owner_.comptonRiskPrecomputeDt_ == sourceDt;
            if(!reuseComptonPrecompute)
            {
                owner_.planckOpacities_.assign(Ncells, 0.0);
                owner_.factorFleck_.assign(Ncells, 1.0);
            }
            owner_.scatteringOpacities_.assign(Ncells, 0.0);
            owner_.Erad_time_avg_.assign(Ncells, 0.0);
            if((owner_.parameters_.withEgTimeAvg ||
                owner_.parameters_.withCompton) &&
               owner_.parameters_.withMultigroupOpacity)
            {
                GroupArray zeros{};
                owner_.Eg_time_avg_.assign(Ncells, zeros);
            }

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                CellT &cell = owner_.cells_[i];

                double const volume = owner_.componentGrid().GetVolume(i);
                if(!std::isfinite(volume) || volume <= 0.0)
                {
                    StormError eo("RadiationIMC::preStep requires finite, positive cell volumes");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Volume", volume);
                    throw eo;
                }

                double gamma = 1.0;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                       (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                    {
                        gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
                    }
                }

                if(owner_.parameters_.withCompton)
                {
                    if(!reuseComptonPrecompute)
                    {
                        owner_.planckOpacities_[i] = 0.0;
                        owner_.factorFleck_[i] = 1.0;
                    }
                    continue;
                }

                owner_.planckOpacities_[i] = owner_.opacity_->CalcPlanckOpacity(cell);
                if(!std::isfinite(owner_.planckOpacities_[i]) ||
                   owner_.planckOpacities_[i] < 0.0)
                {
                    StormError eo("RadiationIMC::preStep received an invalid Planck opacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Planck opacity", owner_.planckOpacities_[i]);
                    throw eo;
                }
                owner_.scatteringOpacities_[i] =
                    owner_.opacity_->CalcScatteringOpacity(cell);
                if(!std::isfinite(owner_.scatteringOpacities_[i]) ||
                   owner_.scatteringOpacities_[i] < 0.0)
                {
                    StormError eo("RadiationIMC::preStep received an invalid scattering opacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Scattering opacity", owner_.scatteringOpacities_[i]);
                    throw eo;
                }
                const auto &tracers = owner_.traits_.tracers(cell);
                const auto &tracerNames = owner_.traits_.tracerNames(cell);
                double cv = owner_.eos_->dT2cv(owner_.density(i), cell.temperature, tracers, tracerNames);
                if(!std::isfinite(cv) || cv <= 0.0)
                {
                    StormError eo("RadiationIMC::preStep requires a finite, positive heat capacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("cv", cv);
                    throw eo;
                }
                owner_.factorFleck_[i] = 1.0 / (1.0 + (4.0 * units::arad * boost::math::pow<3>(cell.temperature) * owner_.planckOpacities_[i] * units::clight * fleckDt * gamma) / cv);
                if(!std::isfinite(owner_.factorFleck_[i]) ||
                   owner_.factorFleck_[i] < 0.0 || owner_.factorFleck_[i] > 1.0)
                {
                    StormError eo("Invalid factor fleck in RadiationIMC::preStep");
                    eo.addEntry("Factor fleck", owner_.factorFleck_[i]);
                    eo.addEntry("Planck opacity", owner_.planckOpacities_[i]);
                    eo.addEntry("Temperature", cell.temperature);
                    eo.addEntry("Density", owner_.density(i));
                    eo.addEntry("Gamma", gamma);
                    eo.addEntry("cv", cv);
                    eo.addEntry("Full dt", fleckDt);
                    throw eo;
                }
            }

            if(owner_.SharedFullIMCKernelEligible() ||
               (owner_.parameters_.withDDMC &&
                owner_.parameters_.withMultigroupOpacity) ||
               (owner_.SharedRandomWalkKernelEligible() &&
                owner_.parameters_.withMultigroupOpacity))
            {
                const double referenceEnergy =
                    owner_.energyBoundaries_[0];
                const double referenceEnergyCubed =
                    referenceEnergy * referenceEnergy * referenceEnergy;
                for(std::size_t i = 0; i < Ncells; ++i)
                {
                    const double absorption =
                        owner_.opacity_->CalcAbsorptionOpacity(
                            owner_.cells_[i], referenceEnergy);
                    owner_.spectralAbsorptionScale_[i] =
                        absorption * referenceEnergyCubed;
                    const GroupArray upper =
                        owner_.opacity_->GetCumulativeOpacity(
                            owner_.cells_[i], owner_.energyBoundaries_);
                    owner_.thermalEmissionCdf_[i * (NumGroups + 1)] = 0.0;
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        owner_.thermalEmissionCdf_[
                            i * (NumGroups + 1) + group + 1] =
                            upper[group];
                    }
                }
            }

            if(owner_.GreyKernelEligible() ||
               owner_.SharedFullIMCKernelEligible() ||
               owner_.parameters_.withRandomWalk ||
               owner_.parameters_.withDDMC)
            {
                owner_.updateGridData();
            }
            if(owner_.parameters_.withRandomWalk)
            {
                owner_.precomputeRandomWalkData();
                owner_.rwStepCount_ = 0;
            }
            if(owner_.parameters_.withDDMC)
            {
                owner_.ddmcFluxRhsIntegrated_.assign(Ncells, PointT{});
                owner_.ddmcMomentumFeedbackCount_ = 0;
                owner_.ddmcMomentumMatrixFallbackCount_ = 0;
                owner_.precomputeDDMCData();
                owner_.ddmcStepCount_ = 0;
                owner_.ddmcLeakCount_ = 0;
                owner_.ddmcCensusCount_ = 0;
                owner_.ddmcUpscatterCount_ = 0;
                owner_.ddmcFallbackCount_ = 0;
                owner_.ddmcMovingInterfaceBypassCount_ = 0;
                owner_.ddmcMovingInterfaceMaxFactor_ = 0.0;
            }
            else
            {
                owner_.ddmcFluxRhsIntegrated_.clear();
            }

            if(owner_.parameters_.withCompton && !reuseComptonPrecompute)
            {
                owner_.precomputeComptonData(sourceDt);
            }
            owner_.comptonDataReusableInPreStep_ = false;
            owner_.deviceExecutor_->prepareStep();
            return sourceDt;
    }

    void appendBoundaryParticles(
        std::vector<MCParticle> &newParticles,
        double fullDt,
        double transportDt)
    {
            if(!owner_.componentBoundary())
            {
                return;
            }
            std::vector<MCParticle> boundaryParticles =
                owner_.componentBoundary()->generateNewBoundaryParticles(fullDt);
            for(MCParticle &particle : boundaryParticles)
            {
                if(particle.rngKey == std::numeric_limits<std::uint64_t>::max())
                {
                    owner_.initializeParticleRNG(particle);
                }
                owner_.setInitialWeightFromWeight(particle);
                if(owner_.parameters_.postProcess.enabled)
                {
                    particle.timeLeft = transportDt * owner_.randomUnitOpen(particle);
                }
            }
            newParticles.insert(
                newParticles.end(),
                boundaryParticles.begin(),
                boundaryParticles.end());
    }

    void recordEmittedEnergy(
        const std::vector<MCParticle> &newParticles,
        double extraEnergy)
    {
            if(!owner_.observer_)
            {
                return;
            }
            double emittedEnergy = extraEnergy;
            double emittedPositiveEnergy = extraEnergy > 0.0 ? extraEnergy : 0.0;
            double emittedNegativeEnergy = extraEnergy < 0.0 ? -extraEnergy : 0.0;
            for(const MCParticle &particle : newParticles)
            {
                emittedEnergy += particle.weight;
                if(particle.weight >= 0.0)
                {
                    emittedPositiveEnergy += particle.weight;
                }
                else
                {
                    emittedNegativeEnergy -= particle.weight;
                }
            }
            owner_.observer_->addEmittedEnergy(emittedEnergy);
            owner_.observer_->addEmittedEnergyComponents(
                emittedPositiveEnergy, emittedNegativeEnergy);
    }

    std::vector<typename Owner::MCParticle>
    preStep(double fullDt)
    {
            const double sourceDt = this->preparePreStepTables(fullDt);
            const double transportDt = owner_.parameters_.postProcess.enabled
                ? owner_.parameters_.postProcess.transportTime : fullDt;
            std::vector<MCParticle> newParticles = owner_.generateParticles(sourceDt);
            if(owner_.parameters_.postProcess.enabled)
            {
                for(MCParticle &particle : newParticles)
                {
                    particle.timeLeft = transportDt * owner_.randomUnitOpen(particle);
                }
            }
            this->appendBoundaryParticles(newParticles, fullDt, transportDt);
            this->recordEmittedEnergy(newParticles, 0.0);
            owner_.preStepInitialized_ = true;
            return newParticles;
    }

#ifdef STORM_WITH_GPU
    std::vector<typename Owner::MCParticle>
    preStepOnDevice(gpu::DeviceSourceContext &context)
    {
            double sourceDt = 0.0;
            {
                STORM_PROFILE_REGION("storm/generation/tables");
                sourceDt = this->preparePreStepTables(context.fullDt);
            }
            ddmc::RequireHostDeviceSamplingKernelMatch();
            if(context.executor == nullptr && context.executorStorage != nullptr)
            {
                if(*context.executorStorage == nullptr)
                {
                    *context.executorStorage =
                        std::make_unique<gpu::KokkosLocalTransportExecutor>(
                            context.gpuMaxInnerSteps);
                }
                context.executor = context.executorStorage->get();
            }
            const double transportDt = owner_.parameters_.postProcess.enabled
                ? owner_.parameters_.postProcess.transportTime : context.fullDt;
            {
                STORM_PROFILE_REGION("storm/generation/emit");
                owner_.sourceProcess_->generateParticles(sourceDt, false);
                context.gpuData = owner_.deviceExecutor_->DeviceData();
                owner_.sourceProcess_->emitPlanToDevice(
                    context, owner_.lastSourcePlan_);
            }
            std::vector<MCParticle> boundaryParticles;
            {
                STORM_PROFILE_REGION("storm/generation/boundary");
                this->appendBoundaryParticles(
                    boundaryParticles, context.fullDt, transportDt);
            }
            this->recordEmittedEnergy(
                boundaryParticles, owner_.lastSourcePlan_.emittedEnergy);
            owner_.preStepInitialized_ = true;
            return boundaryParticles;
    }
#endif

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMCLIFECYCLE_PROCESS_HPP
