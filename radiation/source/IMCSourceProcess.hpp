#ifndef STORM_RADIATION_IMCSOURCE_PROCESS_HPP
#define STORM_RADIATION_IMCSOURCE_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"
#include "SourceCore.hpp"

#ifdef STORM_WITH_GPU
#include "../../gpu/DeviceSourceContext.hpp"
#include "../../gpu/SourceDeviceEmit.hpp"
#endif

namespace STORM::radiation_imc_detail {

template<typename Owner>
class IMCSourceProcess final : public IMCComponentBase<Owner>
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
    using Base::kSamplerHasDecomposition;
public:
    explicit IMCSourceProcess(Owner &owner) : Base(owner)
    {}

    source::SampleViews<PointT> hostSampleViews() const
    {
        source::SampleViews<PointT> views;
        const auto &gridData = owner_.componentGridData();
        views.tetOffsets = gridData.tetOffsets.empty()
            ? nullptr : gridData.tetOffsets.data();
        views.tetCumVolumes = gridData.tetCumVolumes.empty()
            ? nullptr : gridData.tetCumVolumes.data();
        views.tetTris = gridData.tetTris.empty()
            ? nullptr : gridData.tetTris.data();
        views.vertices = gridData.vertices.empty()
            ? nullptr : gridData.vertices.data();
        views.cellCenters = gridData.cellCenters.empty()
            ? nullptr : gridData.cellCenters.data();
        views.thermalEmissionCdf = owner_.thermalEmissionCdf_.empty()
            ? nullptr : owner_.thermalEmissionCdf_.data();
        views.energyBoundaries = owner_.energyBoundaries_.data();
        views.cellVelocities = owner_.transportCellVelocities_.empty()
            ? nullptr : owner_.transportCellVelocities_.data();
        views.cellCount = owner_.componentGrid().GetPointNo();
        views.groupCount = owner_.parameters_.withMultigroupOpacity
            ? NumGroups : 0;
        views.thermalKT = owner_.thermalKT_.data();
        views.thermalFrequencyLaw = owner_.opacity_->GetPortableThermalFrequencyLaw();
        views.speedOfLight = owner_.lightSpeed();
        views.invClight2 = owner_.inverseLightSpeedSquared();
        views.sampleFrequency =
            (owner_.parameters_.withMultigroupOpacity &&
             owner_.thermalEmissionCdf_.size() ==
                 views.cellCount * (NumGroups + 1))
                ? 1 : 0;
        views.applyLabFrame = 0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                !owner_.parameters_.staticScatterers) ||
               (owner_.parameters_.postProcess.enabled &&
                owner_.parameters_.postProcess.useCellVelocities))
            {
                views.applyLabFrame = 1;
            }
        }
        return views;
    }

    void emitPlanToHost(
        const source::Plan &plan,
        std::vector<MCParticle> &newParticles, double fullDt)
    {
        if(plan.totalPhotons == 0)
        {
            return;
        }
        source::SampleViews<PointT> views = this->hostSampleViews();
        views.fullDt = fullDt;
        newParticles.resize(plan.totalPhotons);
        for(std::size_t slot = 0; slot < plan.totalPhotons; ++slot)
        {
            const std::size_t cellIndex = source::CellFromPhotonSlot(
                plan.photonOffsets.data(),
                plan.nPhotons.size(),
                slot);
            const std::uint64_t rngKey = source::MakeSourceRngKey(
                owner_.particleRngSeed_,
                owner_.creationRank_,
                plan.rngStreamBase + slot);
            source::EmittedScalars scalars;
            PointT location{};
            PointT velocity{};
            source::EmitThermalPacket(
                views,
                cellIndex,
                rngKey,
                plan.energyPerPhoton[cellIndex],
                location,
                velocity,
                scalars);
            MCParticle &particle = newParticles[slot];
            particle.location = location;
            particle.velocity = velocity;
            particle.frequency = scalars.frequency;
            particle.weight = scalars.weight;
            particle.initialWeight = scalars.initialWeight;
            particle.rngKey = scalars.rngKey;
            particle.rngCounter = scalars.rngCounter;
            particle.cellIndex = scalars.cellIndex;
            particle.cellID = radiation_imc_detail::cellID(
                owner_.cells_[cellIndex]);
            particle.sourceCellID = particle.cellID;
            particle.id = std::numeric_limits<std::size_t>::max();
            particle.timeLeft = scalars.timeLeft;
            particle.steps = 0;
#ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                polarization::resetUnpolarized<PointT>(particle);
            }
#endif
        }
        owner_.sourceRngStreamCounter_ =
            plan.rngStreamBase + plan.totalPhotons;
    }

#ifdef STORM_WITH_GPU
    void emitPlanToDevice(
        gpu::DeviceSourceContext &context,
        const source::Plan &plan)
    {
        context.plan = &plan;
        context.particleRngSeed = owner_.particleRngSeed_;
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
        context.creationRank = owner_.creationRank_;
        const source::SampleViews<PointT> views = this->hostSampleViews();
        context.sampleFrequency = views.sampleFrequency;
        context.applyLabFrame = views.applyLabFrame;
        context.speedOfLight = views.speedOfLight;
        context.invClight2 = views.invClight2;
        gpu::EmitSourcesOnDevice(context);
        owner_.sourceRngStreamCounter_ =
            plan.rngStreamBase + plan.totalPhotons;
    }
#endif

    typename Owner::MCParticle
    generateSingleParticle(std::size_t cellIndex, const CellT &cell)
    {
        return generateSingleParticle(cellIndex, cell, nullptr);
    }

    std::vector<typename Owner::MCParticle>
    generateParticles(double fullDt)
    {
        return this->generateParticles(fullDt, true);
    }

    std::vector<typename Owner::MCParticle>
    generateParticles(double fullDt, bool materializeHost)
    {

        if(owner_.parameters_.withCompton)
        {
            if(owner_.postProcessExternalSourceMode_)
            {
                throw StormError("External fixed-flux post-process sources do not support Compton yet");
            }
            (void) materializeHost;
            return owner_.generateComptonParticles(fullDt);
        }
        std::vector<MCParticle> newParticles;
        const std::size_t Ncells = owner_.componentGrid().GetPointNo();
        owner_.lastGroupSamplingDiagnostics_ = GroupSamplingDiagnostics{};

        std::vector<std::size_t> externalSourceOffsets(Ncells + 1, 0);
        std::vector<std::size_t> externalSourceIndices;
        if(owner_.postProcessExternalSourceMode_)
        {
            if(owner_.postProcessExternalSourceLocalCellIndices_.size() !=
                owner_.postProcessExternalSources_.size())
            {
                throw StormError("External source-to-cell map has inconsistent size");
            }
            for(std::size_t sourceIndex = 0;
                sourceIndex < owner_.postProcessExternalSources_.size();
                ++sourceIndex)
            {
                PostProcessExternalSource const &source =
                    owner_.postProcessExternalSources_[sourceIndex];
                std::size_t const cellIndex =
                    owner_.postProcessExternalSourceLocalCellIndices_[sourceIndex];
                if(source.luminosity > 0.0 && cellIndex < Ncells)
                {
                    ++externalSourceOffsets[cellIndex + 1];
                }
            }
            for(std::size_t i = 1; i < externalSourceOffsets.size(); ++i)
            {
                externalSourceOffsets[i] += externalSourceOffsets[i - 1];
            }
            externalSourceIndices.resize(externalSourceOffsets.back());
            std::vector<std::size_t> cursor = externalSourceOffsets;
            for(std::size_t sourceIndex = 0;
                sourceIndex < owner_.postProcessExternalSources_.size();
                ++sourceIndex)
            {
                PostProcessExternalSource const &source =
                    owner_.postProcessExternalSources_[sourceIndex];
                std::size_t const cellIndex =
                    owner_.postProcessExternalSourceLocalCellIndices_[sourceIndex];
                if(source.luminosity > 0.0 && cellIndex < Ncells)
                {
                    externalSourceIndices[cursor[cellIndex]++] = sourceIndex;
                }
            }
        }

        std::vector<double> energyToCreateVec(Ncells);
        std::vector<double> gammaVec(Ncells);
        // Post-process volume emission per cell (already divided by the
        // subsample fraction for included cells; zero otherwise).
        std::vector<double> volumeEnergyVec;
        bool const volumeEmission = owner_.postProcessExternalSourceMode_ &&
                                    owner_.postProcessVolumeEmission_;
        if(volumeEmission)
        {
            if(owner_.postProcessVolumeEmissionMask_.size() != Ncells)
            {
                throw StormError("Post-process volume emission mask does not match the cell count");
            }
            volumeEnergyVec.assign(Ncells, 0.0);
        }
        owner_.lastVolumeEmissionEnergy_ = 0.0;
        owner_.lastVolumeEmissionCells_ = 0;
        owner_.lastVolumeEmissionSelectedCells_ = 0;
        auto subsampleIncludes = [&](std::size_t cellId) -> bool
        {
            double const fraction = owner_.postProcessVolumeEmissionSubsample_;
            if(fraction >= 1.0)
            {
                return true;
            }
            // splitmix64 of (seed, cellID): the same cell set on every rank
            // that holds the cell, and a fresh set per generation.
            std::uint64_t z = owner_.postProcessVolumeEmissionSeed_ + 0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(cellId) + 1ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            double const u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
            return u < fraction;
        };
        double localTotalEnergy = 0.0;
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            CellT &cell = owner_.cells_[i];
            double gamma = 1.0;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                    !owner_.parameters_.staticScatterers) ||
                    (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                {
                    gamma = 1.0 / std::sqrt(
                        1.0 - ScalarProd(cell.velocity, cell.velocity) *
                        owner_.inverseLightSpeedSquared());
                }
            }
            gammaVec[i] = gamma;
            if(owner_.postProcessExternalSourceMode_)
            {
                energyToCreateVec[i] = 0.0;
                for(std::size_t offset = externalSourceOffsets[i];
                    offset < externalSourceOffsets[i + 1]; ++offset)
                {
                    energyToCreateVec[i] +=
                        owner_.postProcessExternalSources_[
                            externalSourceIndices[offset]].luminosity * fullDt;
                }
                if(volumeEmission && owner_.postProcessVolumeEmissionMask_[i] != 0)
                {
                    // Thermal emission at the snapshot temperature with Fleck
                    // factor 1: the instantaneous emissivity of a frozen state.
                    // Only the groups outside their own thermalization surface
                    // are emitted from this cell (the others cannot escape). The
                    // emission opacity is the Planck-weighted sum of the *transport*
                    // group opacities over those groups, so emission and absorption
                    // use the same (scaled) opacities and a cell in radiative
                    // equilibrium is energy neutral. The table's own Planck mean
                    // (planckOpacities_) is only used in the grey case.
                    double emissionOpacity = owner_.planckOpacities_[i];
                    if(owner_.parameters_.withMultigroupOpacity)
                    {
                        auto const cumulative = owner_.opacity_->GetCumulativeOpacity(cell, owner_.energyBoundaries_);
                        std::uint16_t bits = static_cast<std::uint16_t>((1u << NumGroups) - 1u);
                        if(owner_.postProcessGroupFleck_ && i < owner_.postProcessGroupOutsideBits_.size())
                        {
                            bits = owner_.postProcessGroupOutsideBits_[i];
                        }
                        emissionOpacity = 0.0;
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            if((bits >> g) & 1u)
                            {
                                double const lower = g == 0 ? 0.0 : cumulative[g - 1];
                                double const weight = cumulative[g] - lower;
                                if(weight > 0.0 && std::isfinite(weight))
                                {
                                    emissionOpacity += weight;
                                }
                            }
                        }
                    }
                    double const volumeEnergy = source::CellEmissionEnergy(
                        1.0,
                        owner_.componentGrid().GetVolume(i),
                        cell.temperature,
                        emissionOpacity,
                        fullDt,
                        units::arad,
                        owner_.lightSpeed());
                    if(volumeEnergy > 0.0 && std::isfinite(volumeEnergy))
                    {
                        ++owner_.lastVolumeEmissionCells_;
                        owner_.lastVolumeEmissionEnergy_ += volumeEnergy;
                        if(subsampleIncludes(radiation_imc_detail::cellID(cell)))
                        {
                            ++owner_.lastVolumeEmissionSelectedCells_;
                            volumeEnergyVec[i] = volumeEnergy / owner_.postProcessVolumeEmissionSubsample_;
                            energyToCreateVec[i] += volumeEnergyVec[i];
                        }
                    }
                }
            }
            else
            {
                energyToCreateVec[i] = source::CellEmissionEnergy(
                    owner_.factorFleck_[i],
                    owner_.componentGrid().GetVolume(i),
                    cell.temperature,
                    owner_.planckOpacities_[i],
                    fullDt,
                    units::arad,
                    owner_.lightSpeed());
            }
            localTotalEnergy += energyToCreateVec[i];
        }

        double globalTotalEnergy = localTotalEnergy;
        std::size_t globalTotalCells = Ncells;
        std::size_t globalSourceCells = static_cast<std::size_t>(std::count_if(
            energyToCreateVec.begin(), energyToCreateVec.end(),
                [](double energy)
                {
                    return energy > 0.0;
                }));
    #ifdef STORM_WITH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &globalTotalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalTotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalSourceCells, 1,
                        MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif

        std::size_t const budgetCells = owner_.postProcessExternalSourceMode_
            ? globalSourceCells : globalTotalCells;
        if(owner_.parameters_.newPhotonsPerCell >
                std::numeric_limits<std::size_t>::max() / 10 ||
            (owner_.parameters_.newPhotonsPerCell > 0 &&
            budgetCells > std::numeric_limits<std::size_t>::max() /
                (10 * owner_.parameters_.newPhotonsPerCell)))
        {
            throw StormError("External source particle budget overflow");
        }
        std::size_t totalParticles =
            budgetCells * owner_.parameters_.newPhotonsPerCell * 10;
        std::vector<std::size_t> nPhotonsVec(Ncells);
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            std::size_t const baseMin = owner_.parameters_.emissionFloorPhotonsPerCell > 0
                    ? owner_.parameters_.emissionFloorPhotonsPerCell
                    : owner_.parameters_.newPhotonsPerCell;
            // With explicit volume emission the thick cells hold nearly all the
            // energy but none of the escaping light, so the burn-in explores
            // every cell with the same packet count and leaves importance to the
            // learned allocation.
            std::size_t const baseMax = (owner_.postProcessVolumeEmission_ && owner_.postProcessVolumeEmissionExactBase_)
                    ? std::max<std::size_t>(baseMin, owner_.parameters_.newPhotonsPerCell)
                    : owner_.parameters_.newPhotonsPerCell * 20;
            nPhotonsVec[i] = source::PhotonCount(
                energyToCreateVec[i],
                globalTotalEnergy,
                totalParticles,
                baseMin,
                baseMax);
        }

        if(owner_.sourceEmissionControlEnabled_)
        {
            double scoreSum = 0.0;
            for(std::pair<std::size_t const, double> const &kv : owner_.adaptiveSourceScores_)
            {
                if(std::isfinite(kv.second) && kv.second > 0.0)
                {
                    scoreSum += std::pow(
                        kv.second, owner_.adaptiveSourceScorePower_);
                }
            }

            std::size_t const basePhotons = owner_.parameters_.newPhotonsPerCell * owner_.sourceEmissionBaseMultiplier_;
            std::size_t const maxPhotons = static_cast<std::size_t>(std::ceil(
                static_cast<double>(std::max<std::size_t>(1, owner_.parameters_.newPhotonsPerCell))
                * owner_.adaptiveSourceMaxFactor_ * owner_.adaptiveSourceObserverBudgetMultiplier_));
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                // A cell without emission energy never gets packets, whatever the
                // uniform base says. In fixed-flux post-processing this is every
                // cell without a source face, and emitting from it would throw.
                if(!(energyToCreateVec[i] > 0.0))
                {
                    nPhotonsVec[i] = 0;
                    continue;
                }
                std::size_t cellId = radiation_imc_detail::cellID(owner_.cells_[i]);
                auto const it = owner_.adaptiveSourceScores_.find(cellId);
                bool const learned = owner_.adaptiveSourceScoresEnabled_ && it != owner_.adaptiveSourceScores_.end()
                    && std::isfinite(it->second) && it->second > 0.0;

                std::size_t photons = owner_.sourceEmissionIncludeUniformBase_ ? basePhotons : 0;
                if(owner_.sourceEmissionUseLearnedScores_ && learned)
                {
                    std::size_t learnedPhotons = owner_.parameters_.newPhotonsPerCell * owner_.sourceEmissionLearnedBoostFactor_;
                    if(scoreSum > 0.0 && owner_.sourceEmissionLearnedExtraBudget_ > 0)
                    {
                        learnedPhotons += static_cast<std::size_t>(std::ceil(
                            owner_.adaptiveSourceStrength_ * static_cast<double>(owner_.sourceEmissionLearnedExtraBudget_)
                            * std::pow(it->second,
                                        owner_.adaptiveSourceScorePower_) /
                                scoreSum));
                    }
                    std::size_t const minLearned = static_cast<std::size_t>(std::ceil(
                        static_cast<double>(std::max<std::size_t>(1, owner_.parameters_.newPhotonsPerCell))
                        * owner_.adaptiveSourceLearnedMinFactor_));
                    learnedPhotons = std::max(learnedPhotons, minLearned);
                    if(owner_.adaptiveSourceLearnedMinPhotons_ > 0)
                    {
                        learnedPhotons = std::max(
                            learnedPhotons,
                            owner_.adaptiveSourceLearnedMinPhotons_);
                    }
                    if(owner_.adaptiveSourceLearnedMaxPhotons_ > 0)
                    {
                        learnedPhotons = std::min(
                            learnedPhotons,
                            owner_.adaptiveSourceLearnedMaxPhotons_);
                    }
                    photons = std::max(photons, learnedPhotons);
                }
                nPhotonsVec[i] = std::min(photons, std::max<std::size_t>(1, maxPhotons));
            }
        }

        if(owner_.postProcessExternalSourceMode_)
        {
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                if(energyToCreateVec[i] > 0.0 && nPhotonsVec[i] == 0)
                {
                    // Every emitting cell keeps at least one packet: face cells so
                    // the injected flux is conserved, volume cells as an
                    // exploration floor so a cell the burn-in never saw escape can
                    // still enter the learned set. Deep packets are absorbed within
                    // a mean free path, so the floor is cheap. A thick cell holds
                    // far more energy than the whole escaping luminosity, so its
                    // exploration is split into packets of bounded weight; one
                    // rare escape then moves the tally by a bounded amount instead
                    // of swamping the generation.
                    std::size_t exploration = 1;
                    if(owner_.postProcessExplorationMaxWeight_ > 0.0)
                    {
                        double const wanted = std::ceil(energyToCreateVec[i] / owner_.postProcessExplorationMaxWeight_);
                        if(std::isfinite(wanted) && wanted > 1.0)
                            exploration = static_cast<std::size_t>(std::min(wanted, 1.0e7));
                    }
                    nPhotonsVec[i] = exploration;
                }
            }
        }
        owner_.lastSourcePhotonsPerCell_ = nPhotonsVec;
        owner_.lastSourceAllocationSummary_ = SourceAllocationSummary{};
        owner_.lastSourceAllocationSummary_.adaptiveEnabled =
            owner_.sourceEmissionControlEnabled_ && owner_.sourceEmissionUseLearnedScores_ && owner_.adaptiveSourceScoresEnabled_;
        std::vector<double> adaptiveScores;
        adaptiveScores.reserve(owner_.adaptiveSourceScores_.size());
        for(std::pair<std::size_t const, double> const &entry : owner_.adaptiveSourceScores_)
        {
            if(entry.second > 0.0 && std::isfinite(entry.second))
            {
                adaptiveScores.push_back(entry.second);
            }
        }
        if(!adaptiveScores.empty())
        {
            std::sort(adaptiveScores.begin(), adaptiveScores.end());
            auto percentile = [&adaptiveScores](double quantile)
            {
                double const position = quantile *
                    static_cast<double>(adaptiveScores.size() - 1);
                std::size_t const lower = static_cast<std::size_t>(position);
                std::size_t const upper = std::min(
                    lower + 1, adaptiveScores.size() - 1);
                double const fraction = position - static_cast<double>(lower);
                return (1.0 - fraction) * adaptiveScores[lower] +
                    fraction * adaptiveScores[upper];
            };
            owner_.lastSourceAllocationSummary_.adaptiveScoreP05 =
                percentile(0.05);
            owner_.lastSourceAllocationSummary_.adaptiveScoreP50 =
                percentile(0.50);
            owner_.lastSourceAllocationSummary_.adaptiveScoreP95 =
                percentile(0.95);
            owner_.lastSourceAllocationSummary_.adaptiveScoreMax =
                adaptiveScores.back();
            owner_.lastSourceAllocationSummary_.adaptiveScoreSpanLow =
                adaptiveScores.front();
            owner_.lastSourceAllocationSummary_.adaptiveScoreSpanHigh =
                adaptiveScores.back();
        }
        owner_.lastSourceAllocationSummary_.minPhotons = std::numeric_limits<std::size_t>::max();
        owner_.lastSourceAllocationSummary_.learnedMinPhotons = std::numeric_limits<std::size_t>::max();
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            std::size_t const photons = nPhotonsVec[i];
            if(photons == 0)
            {
                continue;
            }
            ++owner_.lastSourceAllocationSummary_.sourceCells;
            owner_.lastSourceAllocationSummary_.totalPhotons += photons;
            owner_.lastSourceAllocationSummary_.minPhotons = std::min(owner_.lastSourceAllocationSummary_.minPhotons, photons);
            owner_.lastSourceAllocationSummary_.maxPhotons = std::max(owner_.lastSourceAllocationSummary_.maxPhotons, photons);
            if(photons > owner_.parameters_.newPhotonsPerCell)
            {
                ++owner_.lastSourceAllocationSummary_.boostedCells;
            }

            std::size_t cellId = radiation_imc_detail::cellID(owner_.cells_[i]);
            auto const it = owner_.adaptiveSourceScores_.find(cellId);
            bool const learned = owner_.adaptiveSourceScoresEnabled_ && it != owner_.adaptiveSourceScores_.end()
                && std::isfinite(it->second) && it->second > 0.0;
            if(learned)
            {
                ++owner_.lastSourceAllocationSummary_.learnedCells;
                owner_.lastSourceAllocationSummary_.learnedPhotons += photons;
                owner_.lastSourceAllocationSummary_.adaptiveScoreSum += it->second;
                owner_.lastSourceAllocationSummary_.learnedMinPhotons =
                    std::min(owner_.lastSourceAllocationSummary_.learnedMinPhotons, photons);
                owner_.lastSourceAllocationSummary_.learnedMaxPhotons =
                    std::max(owner_.lastSourceAllocationSummary_.learnedMaxPhotons, photons);
                if(photons >= 1000)
                {
                    ++owner_.lastSourceAllocationSummary_.learnedPhotonsAtLeast1000;
                }
                if(photons >= 2000)
                {
                    ++owner_.lastSourceAllocationSummary_.learnedPhotonsAtLeast2000;
                }
                if(photons > owner_.parameters_.newPhotonsPerCell)
                {
                    ++owner_.lastSourceAllocationSummary_.learnedBoostedCells;
                    owner_.lastSourceAllocationSummary_.learnedExtraPhotons += photons - owner_.parameters_.newPhotonsPerCell;
                }
            }
        }
        if(owner_.lastSourceAllocationSummary_.minPhotons == std::numeric_limits<std::size_t>::max())
        {
            owner_.lastSourceAllocationSummary_.minPhotons = 0;
        }
        if(owner_.lastSourceAllocationSummary_.learnedMinPhotons == std::numeric_limits<std::size_t>::max())
        {
            owner_.lastSourceAllocationSummary_.learnedMinPhotons = 0;
        }

        newParticles.reserve(owner_.lastSourceAllocationSummary_.totalPhotons);

        source::Plan &plan = owner_.lastSourcePlan_;
        plan.nPhotons = nPhotonsVec;
        plan.energyToCreate = energyToCreateVec;
        plan.gamma = gammaVec;
        plan.energyPerPhoton.assign(Ncells, 0.0);
        plan.photonOffsets.assign(Ncells + 1, 0);
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
        plan.rngStreamBase = owner_.sourceRngStreamCounter_;
        plan.emittedEnergy = 0.0;
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            const std::size_t nPhotonsCell = nPhotonsVec[i];
            plan.photonOffsets[i + 1] =
                plan.photonOffsets[i] + nPhotonsCell;
            if(nPhotonsCell == 0)
            {
                continue;
            }
            const double energyToCreate = energyToCreateVec[i];
            const double gamma = gammaVec[i];
            if(!owner_.parameters_.noHydroFeedback)
            {
                // Isotropic comoving emission of energyToCreate carries the
                // lab four-momentum (gamma E, gamma E v/c^2); remove exactly
                // that from the material.  The internal-energy debit becomes
                // E/gamma rather than E (O(beta^2)), the price of keeping the
                // Newtonian material total energy conserved.
                PointT emittedMomentum{};
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro &&
                        !owner_.parameters_.staticScatterers &&
                        !owner_.parameters_.diffusionPressureGradient)
                    {
                        emittedMomentum = energyToCreate * gamma *
                            owner_.cells_[i].velocity * owner_.inverseLightSpeedSquared();
                    }
                }
                owner_.applyMaterialExchange(i, -energyToCreate * gamma, -1.0 * emittedMomentum);
            }
            plan.energyPerPhoton[i] =
                energyToCreate * gamma / static_cast<double>(nPhotonsCell);
            plan.emittedEnergy += energyToCreate * gamma;
        }
        plan.totalPhotons = plan.photonOffsets.back();

        const bool useSharedThermalEmit =
            !owner_.postProcessExternalSourceMode_ &&
            !owner_.adaptiveSourceCellGroupScoresEnabled_ &&
            (!owner_.parameters_.withMultigroupOpacity ||
             owner_.opacity_->GetPortableThermalFrequencyLaw() != ThermalFrequencyLaw::Unsupported);
        if(useSharedThermalEmit)
        {
            if(materializeHost)
            {
                this->emitPlanToHost(plan, newParticles, fullDt);
            }
            return newParticles;
        }

        for(std::size_t i = 0; i < Ncells; ++i)
        {
            CellT &cell = owner_.cells_[i];
            double energyToCreate = energyToCreateVec[i];
            std::size_t nPhotonsCell = nPhotonsVec[i];
            if(nPhotonsCell == 0)
            {
                continue;
            }

            // The decomposition is identical for every photon emitted from this cell.
            const PositionDecomposition *cellDecomposition = nullptr;
            if constexpr(kSamplerHasDecomposition)
            {
                owner_.positionSampler_.BuildDecomposition(
                    owner_.componentGrid(), i, owner_.scratchDecomposition_);
                cellDecomposition = &owner_.scratchDecomposition_;
            }

            double energyPerPhoton = plan.energyPerPhoton[i];

            bool useGroupFreqSampling = owner_.adaptiveSourceCellGroupScoresEnabled_
                && owner_.parameters_.withMultigroupOpacity
                && !owner_.parameters_.withCompton;
            GroupArray physicalPdf{};
            GroupArray samplingPdf{};
            bool groupPdfValid = false;
            bool groupScoreAvailable = false;
            if(useGroupFreqSampling)
            {
                std::size_t cellId = radiation_imc_detail::cellID(cell);
                auto it = owner_.adaptiveSourceCellGroupScores_.find(cellId);
                if(it != owner_.adaptiveSourceCellGroupScores_.end())
                {
                    groupScoreAvailable = true;
                    physicalPdf = owner_.postProcessExternalSourceMode_
                        ? owner_.buildPostProcessExternalSourcePlanckPdf(cell)
                        : owner_.opacity_->GetThermalGroupPdf(
                            cell, owner_.energyBoundaries_);
                    double totalPhys = 0.0;
                    std::size_t nPhysGroups = 0;
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        if(physicalPdf[g] > 0.0)
                        {
                            ++nPhysGroups;
                            totalPhys += physicalPdf[g];
                        }
                    }
                    if(totalPhys > 0.0 && nPhysGroups > 0)
                    {
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            physicalPdf[g] = (physicalPdf[g] > 0.0) ? physicalPdf[g] / totalPhys : 0.0;
                        }
                        GroupArray const &learnedScoreRaw = it->second;
                        double const scoreFloor = 1e-12;
                        GroupArray learnedPdf{};
                        double learnedTotal = 0.0;
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            if(physicalPdf[g] > 0.0)
                            {
                                learnedPdf[g] = std::max(learnedScoreRaw[g], scoreFloor);
                                learnedTotal += learnedPdf[g];
                            }
                        }
                        if(learnedTotal > 0.0)
                        {
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                learnedPdf[g] /= learnedTotal;
                            }
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                samplingPdf[g] = (1.0 - owner_.adaptiveGroupStrength_) * physicalPdf[g]
                                    + owner_.adaptiveGroupStrength_ * learnedPdf[g];
                            }
                            double floorPerGroup = (nPhysGroups > 0) ? owner_.adaptiveGroupPdfFloor_ / static_cast<double>(nPhysGroups) : 0.0;
                            GroupArray lowerBound{};
                            GroupArray upperBound{};
                            double lowerTotal = 0.0;
                            double upperTotal = 0.0;
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                if(physicalPdf[g] > 0.0)
                                {
                                    lowerBound[g] = std::max(floorPerGroup, physicalPdf[g] / owner_.adaptiveGroupMaxWeightCorrection_);
                                    upperBound[g] = std::min(1.0, owner_.adaptiveGroupMaxBias_ * physicalPdf[g]);
                                    lowerBound[g] = std::min(lowerBound[g], upperBound[g]);
                                    lowerTotal += lowerBound[g];
                                    upperTotal += upperBound[g];
                                }
                                else
                                {
                                    samplingPdf[g] = 0.0;
                                }
                            }

                            if(lowerTotal <= 1.0 + 1e-12 && upperTotal >= 1.0 - 1e-12)
                            {
                                std::array<bool, NumGroups> fixed{};
                                double remaining = 1.0;
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    if(!(physicalPdf[g] > 0.0))
                                    {
                                        fixed[g] = true;
                                        samplingPdf[g] = 0.0;
                                    }
                                }

                                for(std::size_t iter = 0; iter < NumGroups + 2; ++iter)
                                {
                                    double freeTotal = 0.0;
                                    for(std::size_t g = 0; g < NumGroups; ++g)
                                    {
                                        if(!fixed[g])
                                        {
                                            freeTotal += std::max(samplingPdf[g], 0.0);
                                        }
                                    }
                                    if(!(freeTotal > 0.0))
                                    {
                                        groupPdfValid = false;
                                        break;
                                    }

                                    bool clamped = false;
                                    double const scale = remaining / freeTotal;
                                    for(std::size_t g = 0; g < NumGroups; ++g)
                                    {
                                        if(fixed[g])
                                        {
                                            continue;
                                        }
                                        double const candidate = std::max(samplingPdf[g], 0.0) * scale;
                                        if(candidate < lowerBound[g])
                                        {
                                            samplingPdf[g] = lowerBound[g];
                                            fixed[g] = true;
                                            remaining -= lowerBound[g];
                                            clamped = true;
                                        }
                                        else if(candidate > upperBound[g])
                                        {
                                            samplingPdf[g] = upperBound[g];
                                            fixed[g] = true;
                                            remaining -= upperBound[g];
                                            clamped = true;
                                        }
                                    }

                                    if(!clamped)
                                    {
                                        for(std::size_t g = 0; g < NumGroups; ++g)
                                        {
                                            if(!fixed[g])
                                            {
                                                samplingPdf[g] = std::max(samplingPdf[g], 0.0) * scale;
                                            }
                                        }
                                        remaining = 0.0;
                                        break;
                                    }
                                    if(remaining < 0.0)
                                    {
                                        break;
                                    }
                                }
                            }
                            else
                            {
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    samplingPdf[g] = 0.0;
                                }
                            }

                            double sampTotal = 0.0;
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                sampTotal += samplingPdf[g];
                            }
                            if(sampTotal > 0.0)
                            {
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    samplingPdf[g] /= sampTotal;
                                }
                                groupPdfValid = true;
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    if(physicalPdf[g] > 0.0)
                                    {
                                        double const correction = physicalPdf[g] / samplingPdf[g];
                                        if(!(samplingPdf[g] > 0.0)
                                            || correction > owner_.adaptiveGroupMaxWeightCorrection_ * (1.0 + 1e-10)
                                            || samplingPdf[g] > owner_.adaptiveGroupMaxBias_ * physicalPdf[g] * (1.0 + 1e-10))
                                        {
                                            groupPdfValid = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if(useGroupFreqSampling && groupScoreAvailable && !groupPdfValid)
            {
                ++owner_.lastGroupSamplingDiagnostics_.invalidPdfFallback;
                owner_.lastGroupSamplingDiagnostics_.invalidPdfFallbackPackets += nPhotonsCell;
            }

            for(std::size_t j = 0; j < nPhotonsCell; ++j)
            {
                MCParticle particle;
                owner_.initializeParticleRNG(particle);
                bool volumePacket = false;
                if(owner_.postProcessExternalSourceMode_)
                {
                    std::size_t const begin = externalSourceOffsets[i];
                    std::size_t const end = externalSourceOffsets[i + 1];
                    double const volumeEnergy = volumeEmission ? volumeEnergyVec[i] : 0.0;
                    if(begin == end && !(volumeEnergy > 0.0))
                    {
                        throw StormError(
                            "External source cell " + std::to_string(i) +
                            " was allocated " + std::to_string(nPhotonsCell) +
                            " packets with energy " + std::to_string(energyToCreate) +
                            " but has neither source faces nor volume emission");
                    }
                    // Split packets between the face sources and the volume
                    // term in proportion to their energy.
                    if(volumeEnergy > 0.0 &&
                       (begin == end || owner_.randomUnitOpen(particle) * energyToCreate < volumeEnergy))
                    {
                        volumePacket = true;
                        particle = owner_.generateSingleParticle(i, cell, cellDecomposition);
                    }
                    else
                    {
                    double const faceEnergy = energyToCreate - volumeEnergy;
                    double const totalLuminosity = faceEnergy / fullDt;
                    double const target = owner_.randomUnitOpen(particle) * totalLuminosity;
                    double cumulative = 0.0;
                    std::size_t selectedSource = externalSourceIndices[end - 1];
                    for(std::size_t offset = begin; offset < end; ++offset)
                    {
                        std::size_t const sourceIndex =
                            externalSourceIndices[offset];
                        cumulative += owner_.postProcessExternalSources_[
                            sourceIndex].luminosity;
                        if(target <= cumulative)
                        {
                            selectedSource = sourceIndex;
                            break;
                        }
                    }
                    particle = owner_.generatePostProcessExternalSourceParticle(
                        i, cell,
                        owner_.postProcessExternalSources_[selectedSource]);
                    }
                }
                else
                {
                    particle = owner_.generateSingleParticle(i, cell, cellDecomposition);
                }
                // Volume packets sample the thermal emission spectrum of the
                // cell (kappa_a B_nu), not the face Planck spectrum, and skip
                // the learned group sampling built for the face spectrum.
                bool const faceSpectrum = owner_.postProcessExternalSourceMode_ && !volumePacket;
                particle.cellID = radiation_imc_detail::cellID(cell);
                particle.sourceCellID = particle.cellID;
                particle.timeLeft = fullDt * owner_.randomUnitOpen(particle);

                double weightCorrection = 1.0;
                bool usedGroupFrequencySampling = false;

                if(volumePacket && owner_.postProcessGroupFleck_ &&
                   owner_.parameters_.withMultigroupOpacity &&
                   i < owner_.postProcessGroupOutsideBits_.size())
                {
                    // Volume packets carry only the groups whose Fleck factor
                    // is 1 in this cell, weighted by the thermal spectrum.
                    GroupArray const thermalPdf =
                        owner_.opacity_->GetThermalGroupPdf(cell, owner_.energyBoundaries_);
                    std::uint16_t const bits = owner_.postProcessGroupOutsideBits_[i];
                    double allowedTotal = 0.0;
                    std::size_t lastAllowed = NumGroups;
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        if((bits >> g) & 1u)
                        {
                            allowedTotal += thermalPdf[g];
                            lastAllowed = g;
                        }
                    }
                    if(allowedTotal > 0.0 && lastAllowed < NumGroups)
                    {
                        double const target = owner_.randomUnitOpen(particle) * allowedTotal;
                        double cumulative = 0.0;
                        std::size_t selectedGroup = lastAllowed;
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            if(!((bits >> g) & 1u))
                            {
                                continue;
                            }
                            cumulative += thermalPdf[g];
                            if(target <= cumulative)
                            {
                                selectedGroup = g;
                                break;
                            }
                        }
                        double const freqCo = owner_.opacity_->SampleThermalEnergyInGroup(
                            cell, selectedGroup, owner_.randomUnitOpen(particle),
                            owner_.energyBoundaries_);
                        if((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                            !owner_.parameters_.staticScatterers) ||
                           (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                        {
                            double D = radiation_imc_detail::computeDopplerShift<PointT>(
                                particle, cell, owner_.lightSpeed());
                            particle.frequency = freqCo / D;
                            particle.weight = energyToCreate / (nPhotonsCell * D);
                        }
                        else
                        {
                            particle.frequency = freqCo;
                            particle.weight = energyPerPhoton;
                        }
                        usedGroupFrequencySampling = true;
                    }
                }

                if(groupPdfValid && !volumePacket)
                {
                    double rndGroup = owner_.randomUnitOpen(particle);
                    double cumul = 0.0;
                    std::size_t selectedGroup = NumGroups - 1;
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        cumul += samplingPdf[g];
                        if(rndGroup <= cumul)
                        {
                            selectedGroup = g;
                            break;
                        }
                    }
                    double freqCo = 0.0;

                    if(samplingPdf[selectedGroup] > 0.0)
                    {
                        weightCorrection = physicalPdf[selectedGroup] / samplingPdf[selectedGroup];
                        if(weightCorrection > owner_.adaptiveGroupMaxWeightCorrection_)
                        {
                            ++owner_.lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                        }
                        else if(weightCorrection > 0.0 && std::isfinite(weightCorrection))
                        {
                            if(owner_.lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                            {
                                owner_.lastGroupSamplingDiagnostics_.weightCorrectionMin = weightCorrection;
                            }
                            else
                            {
                                owner_.lastGroupSamplingDiagnostics_.weightCorrectionMin = std::min(owner_.lastGroupSamplingDiagnostics_.weightCorrectionMin, weightCorrection);
                            }
                            if(owner_.lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                            {
                                owner_.lastGroupSamplingDiagnostics_.weightCorrectionMax = weightCorrection;
                            }
                            else
                            {
                                owner_.lastGroupSamplingDiagnostics_.weightCorrectionMax = std::max(owner_.lastGroupSamplingDiagnostics_.weightCorrectionMax, weightCorrection);
                            }
                            owner_.lastGroupSamplingDiagnostics_.weightCorrectionSum += weightCorrection;
                            ++owner_.lastGroupSamplingDiagnostics_.weightCorrectionCount;
                            ++owner_.lastGroupSamplingDiagnostics_.totalSampled;
                            owner_.lastGroupSamplingDiagnostics_.sampledEnergy += energyPerPhoton;
                            double rndFreq = owner_.randomUnitOpen(particle);
                            freqCo = faceSpectrum
                                ? owner_.samplePostProcessExternalSourcePlanckFrequencyInGroup(
                                    cell, selectedGroup)
                                : owner_.opacity_->SampleThermalEnergyInGroup(
                                    cell, selectedGroup, rndFreq,
                                    owner_.energyBoundaries_);
                            usedGroupFrequencySampling = true;
                        }
                        else
                        {
                            ++owner_.lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                        }
                    }
                    else
                    {
                        ++owner_.lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                    }

                    if(usedGroupFrequencySampling &&
                        ((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                          !owner_.parameters_.staticScatterers) ||
                        (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities)))
                    {
                        double D = radiation_imc_detail::computeDopplerShift<PointT>(
                            particle, cell, owner_.lightSpeed());
                        particle.frequency = freqCo / D;
                        particle.weight = energyToCreate / (nPhotonsCell * D) * weightCorrection;
                    }
                    else if(usedGroupFrequencySampling)
                    {
                        particle.frequency = freqCo;
                        particle.weight = energyPerPhoton * weightCorrection;
                    }
                }

                if(!usedGroupFrequencySampling &&
                    ((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                      !owner_.parameters_.staticScatterers) ||
                    (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities)))
                {
                    double D = radiation_imc_detail::computeDopplerShift<PointT>(
                        particle, cell, owner_.lightSpeed());
                    if(owner_.parameters_.withMultigroupOpacity)
                    {
                        double rnd = owner_.randomUnitOpen(particle);
                        double freqCo = faceSpectrum
                            ? owner_.samplePostProcessExternalSourcePlanckFrequency(cell)
                            : owner_.opacity_->GetThermalEnergy(
                                cell, rnd, owner_.energyBoundaries_);
                        particle.frequency = freqCo / D;
                    }
                    particle.weight = energyToCreate / (nPhotonsCell * D);
                }
                else if(!usedGroupFrequencySampling)
                {
                    if(owner_.parameters_.withMultigroupOpacity)
                    {
                        particle.frequency = faceSpectrum
                            ? owner_.samplePostProcessExternalSourcePlanckFrequency(cell)
                            : owner_.opacity_->GetThermalEnergy(
                                cell, owner_.randomUnitOpen(particle),
                                owner_.energyBoundaries_);
                    }
                    particle.weight = energyPerPhoton;
                }
                owner_.setInitialWeightFromWeight(particle);
                newParticles.push_back(particle);
            }
        }

        return newParticles;
    }

    typename Owner::MCParticle generateSingleParticle(std::size_t cellIndex, const CellT &cell, const PositionDecomposition *decomposition)
    {
        MCParticle particle;
        owner_.initializeParticleRNG(particle);
        particle.id = std::numeric_limits<std::size_t>::max();
        particle.cellIndex = cellIndex;
        particle.cellID = radiation_imc_detail::cellID(cell);
        particle.sourceCellID = particle.cellID;
        particle.frequency = 0.0;
        const auto &gridData = owner_.componentGridData();
        if(not gridData.tetOffsets.empty() and gridData.tetOffsets.size() == owner_.componentGrid().GetPointNo() + 1 and not gridData.cellCenters.empty())
        {
            const source::SampleViews<PointT> views = this->hostSampleViews();
            particle.location = source::SamplePositionFromTetTables(views, cellIndex, particle.rngKey, particle.rngCounter);
        }
        else if constexpr(kSamplerHasDecomposition)
        {
            if(decomposition != nullptr)
            {
                particle.location = owner_.positionSampler_.Sample(owner_.componentGrid(), cellIndex, *decomposition, particle.rngKey, particle.rngCounter);
            }
            else
            {
                owner_.positionSampler_.BuildDecomposition(owner_.componentGrid(), cellIndex, owner_.scratchDecomposition_);
                particle.location = owner_.positionSampler_.Sample(owner_.componentGrid(), cellIndex, owner_.scratchDecomposition_, particle.rngKey, particle.rngCounter);
            }
        }
        else
        {
            (void) decomposition;
            particle.location = owner_.positionSampler_(owner_.componentGrid(), cellIndex, owner_.rng_, owner_.dist_);
        }
        if(owner_.componentGrid().IsPointOutsideBox(particle.location))
        {
            PointT meshPoint = owner_.componentGrid().GetMeshPoint(cellIndex);
            PointT original = particle.location;
            PointT direction = meshPoint - original;
            double t = 1e-8;
            while(owner_.componentGrid().IsPointOutsideBox(particle.location) && t < 1.0)
            {
                particle.location = original + t * direction;
                t *= 2;
            }
            particle.location = particle.location + 1e-8 * (meshPoint - particle.location);
        }

        particle.velocity = owner_.sampleRandomVelocity(cell, particle);

    #ifdef MONTECARLO_POLARIZATION
        if(owner_.polarizationEnabled())
        {
            polarization::resetUnpolarized<PointT>(particle);
        }
    #endif

        if((owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
            !owner_.parameters_.staticScatterers) ||
            (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
        {
            radiation_imc_detail::lorentzTransformToLab<PointT>(
                particle, cell, owner_.lightSpeed());
    #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                particle.polarizationBasis = polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity);
            }
    #endif
        }

        particle.timeLeft = 0.0;
        particle.steps = 0;
        return particle;
    }

    std::vector<typename Owner::MCParticle> generateInitialParticles(std::size_t particlesPerCell)
    {
        if(particlesPerCell == 0)
        {
            return {};
        }

        std::vector<MCParticle> result;
        const std::size_t Ncells = owner_.componentGrid().GetPointNo();
        result.reserve(Ncells * particlesPerCell);

        const std::size_t Ngroups = owner_.energyBoundaries_.empty() ? 0 : owner_.energyBoundaries_.size() - 1;

        for(std::size_t i = 0; i < Ncells; ++i)
        {
            const double totalErad = owner_.totalRadiationEnergy(i);
            if(totalErad <= 0.0)
            {
                continue;
            }

            std::vector<double> cumulativePlanck;
            if(owner_.parameters_.withMultigroupOpacity && Ngroups > 0)
            {
                cumulativePlanck.resize(Ngroups + 1);
                cumulativePlanck[0] = 0.0;
                if(owner_.parameters_.withCompton)
                {
                    for(std::size_t g = 1; g <= Ngroups; ++g)
                    {
                        double const groupEnergy = std::max(0.0, owner_.traits_.groupEnergyPerMass(owner_.cells_[i], g - 1) * owner_.density(i) * owner_.componentGrid().GetVolume(i));
                        cumulativePlanck[g] = cumulativePlanck[g - 1] + groupEnergy;
                    }
                }
                else
                {
                    double const kT =
                        units::k_boltz * owner_.cells_[i].temperature;
                    for(std::size_t g = 1; g <= Ngroups; ++g)
                    {
                        double const a = owner_.energyBoundaries_[g - 1] / kT;
                        double const b = owner_.energyBoundaries_[g] / kT;
                        cumulativePlanck[g] =
                            planck_integral::planck_integral(a, b) +
                            cumulativePlanck[g - 1];
                    }
                }
            }

            const double weightPerPhoton = totalErad / static_cast<double>(particlesPerCell);
            for(std::size_t j = 0; j < particlesPerCell; ++j)
            {
                MCParticle particle = owner_.generateSingleParticle(i, owner_.cells_[i]);
                double comovingFrequency = 0.0;
                if(owner_.parameters_.withMultigroupOpacity and not cumulativePlanck.empty())
                {
                    double rnd = owner_.randomUnitOpen(particle);
                    double const total = cumulativePlanck.back();
                    if(owner_.parameters_.withCompton)
                    {
                        rnd *= total;
                    }
                    comovingFrequency = STORM::LinearInterpolation(cumulativePlanck, owner_.energyBoundaries_, rnd);
                }
                if(owner_.parameters_.withCompton)
                {
                    owner_.setPacketFromComovingState(particle, owner_.cells_[i], comovingFrequency, weightPerPhoton);
                }
                else
                {
                    particle.weight = weightPerPhoton;
                    if(owner_.parameters_.withMultigroupOpacity and not cumulativePlanck.empty())
                    {
                        particle.frequency = comovingFrequency;
                        owner_.clampFrequencyToBounds(particle.frequency);
                    }
                }
                owner_.setInitialWeightFromWeight(particle);
                result.push_back(particle);
            }
        }
        return result;
    }

    void splitComptonRiskyParticles(std::vector<MCParticle> &particles, double fullDt)
    {
        owner_.comptonDataReusableInPreStep_ = false;
        if(owner_.parameters_.postProcess.enabled or not owner_.parameters_.withCompton or not owner_.parameters_.withMultigroupOpacity)
        {
            return;
        }

        std::size_t const Ncells = owner_.componentGrid().GetPointNo();
        owner_.factorFleck_.assign(Ncells, 1.0);
        owner_.planckOpacities_.assign(Ncells, 0.0);
        owner_.precomputeComptonData(fullDt);
        owner_.comptonDataReusableInPreStep_ = true;

        std::vector<std::vector<std::size_t>> bins(Ncells * NumGroups);
        for(std::size_t particleIndex = 0; particleIndex < particles.size(); ++particleIndex)
        {
            MCParticle &particle = particles[particleIndex];
            if(particle.cellIndex >= Ncells or not (particle.weight > 0.0))
            {
                continue;
            }
            double frequency = particle.frequency;
            owner_.clampFrequencyToBounds(frequency);
            std::size_t const group = owner_.opacity_->findGroup(frequency, owner_.energyBoundaries_);
            if(group >= NumGroups or owner_.comptonData_[particle.cellIndex].riskTargetPackets[group] == 0)
            {
                continue;
            }
            bins[particle.cellIndex * NumGroups + group].push_back(particleIndex);
        }

        std::size_t const maxExtra =
            std::max<std::size_t>(1, particles.size() / 10);
        constexpr std::size_t maxExtraPerCell = 200;
        std::vector<std::size_t> extraPerCell(Ncells, 0);
        std::size_t extraCount = 0;
        particles.reserve(particles.size() + maxExtra);

        for(std::size_t cellIndex = 0;
            cellIndex < Ncells && extraCount < maxExtra; ++cellIndex)
        {
            std::array<std::size_t, NumGroups> riskOrder{};
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                riskOrder[group] = group;
            }
            std::sort(riskOrder.begin(), riskOrder.end(),
                [&](std::size_t left, std::size_t right)
                {
                    return owner_.comptonData_[cellIndex].riskScore[left] >
                        owner_.comptonData_[cellIndex].riskScore[right];
                });

            for(std::size_t orderIndex = 0;
                orderIndex < NumGroups && extraCount < maxExtra;
                ++orderIndex)
            {
                std::size_t const group = riskOrder[orderIndex];
                std::size_t const target =
                    owner_.comptonData_[cellIndex].riskTargetPackets[group];
                std::vector<std::size_t> const &bin =
                    bins[cellIndex * NumGroups + group];
                std::size_t const count = bin.size();
                if(target == 0 || count == 0 || count >= target)
                {
                    continue;
                }

                std::size_t allowed = target - count;
                allowed = std::min(allowed, maxExtra - extraCount);
                allowed = std::min(
                    allowed, maxExtraPerCell -
                        std::min(extraPerCell[cellIndex], maxExtraPerCell));
                if(allowed == 0)
                {
                    continue;
                }

                std::vector<std::size_t> copiesPerOriginal(count, 0);
                for(std::size_t copy = 0; copy < allowed; ++copy)
                {
                    ++copiesPerOriginal[copy % count];
                }
                for(std::size_t index = 0; index < count; ++index)
                {
                    std::size_t const copies = copiesPerOriginal[index];
                    if(copies == 0)
                    {
                        continue;
                    }
                    std::size_t const particleIndex = bin[index];
                    std::size_t const pieces = copies + 1;
                    double const splitWeight =
                        particles[particleIndex].weight /
                        static_cast<double>(pieces);
                    particles[particleIndex].weight = splitWeight;
                    owner_.setInitialWeightFromWeight(particles[particleIndex]);
                    for(std::size_t copy = 0; copy < copies; ++copy)
                    {
                        MCParticle duplicate = particles[particleIndex];
                        duplicate.weight = splitWeight;
                        duplicate.id = std::numeric_limits<std::size_t>::max();
                        duplicate.steps = 0;
                        owner_.setInitialWeightFromWeight(duplicate);
                        particles.push_back(duplicate);
                    }
                }
                extraCount += allowed;
                extraPerCell[cellIndex] += allowed;
            }
        }
    }

    void adjustExistingParticles(std::vector<MCParticle> &particles, double fullDt)
    {
        if(owner_.lightSpeed() != units::clight)
        {
            for(MCParticle &particle : particles)
            {
                if(!particle.radiationState.isResident())
                {
                    owner_.normalizeParticleSpeed(particle);
                }
            }
        }
        owner_.splitComptonRiskyParticles(particles, fullDt);
        if(not owner_.parameters_.MMC)
        {
            return;
        }

        const std::size_t Ncells = owner_.componentGrid().GetPointNo();
        std::vector<double> divV(Ncells, 0.0);

        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                PointT r_i = owner_.componentGrid().GetMeshPoint(i);
                for(std::size_t faceIdx : owner_.componentGrid().GetCellFaces(i))
                {
                    const std::pair<std::size_t, std::size_t> &neighbors = owner_.componentGrid().GetFaceNeighbors(faceIdx);
                    std::size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
                    PointT neighborPoint;
                    PointT neighborVelocity;
                    if(neighborIdx < Ncells and not owner_.componentGrid().IsPointOutsideBox(neighborIdx))
                    {
                        neighborPoint = owner_.componentGrid().GetMeshPoint(neighborIdx);
                        neighborVelocity = owner_.cells_[neighborIdx].velocity;
                    }
                    else
                    {
                        neighborPoint = owner_.componentGrid().FaceCM(faceIdx);
                        neighborVelocity = owner_.cells_[i].velocity;
                    }
                    PointT diff = r_i - neighborPoint;
                    double distMag = fastabs(diff);
                    if(distMag <= 0.0)
                    {
                        continue;
                    }
                    PointT r_ij = diff / distMag;
                    double A_ij = owner_.componentGrid().GetArea(faceIdx);
                    divV[i] -= 0.5 * ScalarProd(owner_.cells_[i].velocity + neighborVelocity, r_ij) * A_ij;
                }
                divV[i] /= owner_.componentGrid().GetVolume(i);
            }

            const auto [ll, ur] = owner_.componentGrid().GetBoxCoordinates();

            auto it = particles.begin();
            while(it != particles.end())
            {
                MCParticle &p = *it;
                std::size_t ci = p.cellIndex;
                if(ci < Ncells)
                {
                    p.location += owner_.cells_[ci].velocity * fullDt;
                    p.weight += -p.weight * fullDt * divV[ci] / 3.0;
                }

                if(owner_.componentGrid().IsPointOutsideBox(p.location))
                {
                    p.location.x = std::max(ll.x, std::min(ur.x, p.location.x));
                    p.location.y = std::max(ll.y, std::min(ur.y, p.location.y));
                    p.location.z = std::max(ll.z, std::min(ur.z, p.location.z));
                    if(owner_.componentBoundary())
                    {
                        ParticleStatus status = owner_.componentBoundary()->apply(p);
                        if(status == ParticleStatus::REMOVE)
                        {
                            it = particles.erase(it);
                            continue;
                        }
                    }
                }
                ++it;
            }
        }
        else
        {
            (void) particles;
            (void) fullDt;
        }

        UpdateNewCells<PointT>(owner_.componentGrid(), particles);
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMCSOURCE_PROCESS_HPP
