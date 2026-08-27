#ifndef STORM_RADIATION_IMCSOURCE_PROCESS_HPP
#define STORM_RADIATION_IMCSOURCE_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"

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

    typename Owner::MCParticle
    generateSingleParticle(std::size_t cellIndex, const CellT &cell)
    {
        return generateSingleParticle(cellIndex, cell, nullptr);
    }

    std::vector<typename Owner::MCParticle>
    generateParticles(double fullDt)
    {

            if(owner_.parameters_.withCompton)
            {
                if(owner_.postProcessExternalSourceMode_)
                {
                    throw StormError(
                        "External fixed-flux post-process sources do not support Compton yet");
                }
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
            double localTotalEnergy = 0.0;
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                CellT &cell = owner_.cells_[i];
                double gamma = 1.0;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                       (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                    {
                        gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
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
                }
                else
                {
                    energyToCreateVec[i] = owner_.factorFleck_[i] *
                        owner_.componentGrid().GetVolume(i) * units::arad *
                        boost::math::pow<4>(cell.temperature) *
                        owner_.planckOpacities_[i] * fullDt * units::clight;
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
                std::size_t proportionalShare = (globalTotalEnergy > 0)
                    ? static_cast<std::size_t>(energyToCreateVec[i] / globalTotalEnergy * totalParticles)
                    : owner_.parameters_.newPhotonsPerCell;
                nPhotonsVec[i] = energyToCreateVec[i] > 0.0
                    ? std::max(owner_.parameters_.newPhotonsPerCell,
                               std::min(proportionalShare, owner_.parameters_.newPhotonsPerCell * 20))
                    : 0;
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
                        nPhotonsVec[i] = 1;
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

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                CellT &cell = owner_.cells_[i];
                double energyToCreate = energyToCreateVec[i];
                double gamma = gammaVec[i];
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

                if(!owner_.parameters_.noHydroFeedback)
                {
                    owner_.extensives_[i].internal_energy -= energyToCreate;
                    if constexpr(radiation_imc_detail::has_member_total_energy<ExtensivesT>::value)
                    {
                        owner_.extensives_[i].energy -= energyToCreate * gamma;
                    }
                    if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                    {
                        if(owner_.parameters_.withHydro && !owner_.parameters_.diffusionPressureGradient)
                        {
                            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                            {
                                owner_.extensives_[i].momentum -= energyToCreate * cell.velocity * units::inv_clight2 * gamma;
                            }
                        }
                    }
                }
                double energyPerPhoton = energyToCreate * gamma / nPhotonsCell;

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
                    if(owner_.postProcessExternalSourceMode_)
                    {
                        std::size_t const begin = externalSourceOffsets[i];
                        std::size_t const end = externalSourceOffsets[i + 1];
                        if(begin == end)
                        {
                            throw StormError(
                                "External source cell has energy but no source faces");
                        }
                        double const totalLuminosity = energyToCreate / fullDt;
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
                    else
                    {
                        particle = owner_.generateSingleParticle(i, cell, cellDecomposition);
                    }
                    particle.cellID = radiation_imc_detail::cellID(cell);
                    particle.sourceCellID = particle.cellID;
                    particle.timeLeft = fullDt * owner_.randomUnitOpen(particle);

                    double weightCorrection = 1.0;
                    bool usedGroupFrequencySampling = false;

                    if(groupPdfValid)
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
                                freqCo = owner_.postProcessExternalSourceMode_
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
                           ((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                            (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities)))
                        {
                            double D = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
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
                       ((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                        (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities)))
                    {
                        double D = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                        if(owner_.parameters_.withMultigroupOpacity)
                        {
                            double rnd = owner_.randomUnitOpen(particle);
                            double freqCo = owner_.postProcessExternalSourceMode_
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
                            particle.frequency = owner_.postProcessExternalSourceMode_
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

    typename Owner::MCParticle
    generateSingleParticle(
        std::size_t cellIndex,
        const CellT &cell,
        const PositionDecomposition *decomposition)
    {

            MCParticle particle;
            owner_.initializeParticleRNG(particle);
            particle.id = std::numeric_limits<std::size_t>::max();
            particle.cellIndex = cellIndex;
            particle.cellID = radiation_imc_detail::cellID(cell);
            particle.sourceCellID = particle.cellID;
            particle.frequency = 0.0;
            if constexpr(kSamplerHasDecomposition)
            {
                particle.location = (decomposition != nullptr)
                    ? owner_.positionSampler_.Sample(owner_.componentGrid(), cellIndex, *decomposition,
                                                    owner_.rng_, owner_.dist_)
                    : owner_.positionSampler_(owner_.componentGrid(), cellIndex, owner_.rng_, owner_.dist_);
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

            if((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
               (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
            {
                radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
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

    std::vector<typename Owner::MCParticle>
    generateInitialParticles(std::size_t particlesPerCell)
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
                            double const groupEnergy = std::max(
                                0.0,
                                owner_.traits_.groupEnergyPerMass(
                                    owner_.cells_[i], g - 1) *
                                owner_.density(i) * owner_.componentGrid().GetVolume(i));
                            cumulativePlanck[g] =
                                cumulativePlanck[g - 1] + groupEnergy;
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
                    if(owner_.parameters_.withMultigroupOpacity && !cumulativePlanck.empty())
                    {
                        double rnd = owner_.randomUnitOpen(particle);
                        double const total = cumulativePlanck.back();
                        if(owner_.parameters_.withCompton)
                        {
                            rnd *= total;
                        }
                        comovingFrequency = STORM::LinearInterpolation(
                            cumulativePlanck, owner_.energyBoundaries_, rnd);
                    }
                    if(owner_.parameters_.withCompton)
                    {
                        owner_.setPacketFromComovingState(
                            particle,
                            owner_.cells_[i],
                            comovingFrequency,
                            weightPerPhoton);
                    }
                    else
                    {
                        particle.weight = weightPerPhoton;
                        if(owner_.parameters_.withMultigroupOpacity &&
                           !cumulativePlanck.empty())
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

    void splitComptonRiskyParticles(
        std::vector<MCParticle> &particles,
        double fullDt)
    {

            owner_.comptonDataReusableInPreStep_ = false;
            if(owner_.parameters_.postProcess.enabled ||
               !owner_.parameters_.withCompton ||
               !owner_.parameters_.withMultigroupOpacity)
            {
                return;
            }

            std::size_t const Ncells = owner_.componentGrid().GetPointNo();
            owner_.factorFleck_.assign(Ncells, 1.0);
            owner_.planckOpacities_.assign(Ncells, 0.0);
            owner_.precomputeComptonData(fullDt);
            owner_.comptonDataReusableInPreStep_ = true;

            std::vector<std::vector<std::size_t>> bins(Ncells * NumGroups);
            for(std::size_t particleIndex = 0;
                particleIndex < particles.size(); ++particleIndex)
            {
                MCParticle &particle = particles[particleIndex];
                if(particle.cellIndex >= Ncells || !(particle.weight > 0.0))
                {
                    continue;
                }
                double frequency = particle.frequency;
                owner_.clampFrequencyToBounds(frequency);
                std::size_t const group = owner_.opacity_->findGroup(
                    frequency, owner_.energyBoundaries_);
                if(group >= NumGroups ||
                   owner_.comptonData_[particle.cellIndex].riskTargetPackets[group] == 0)
                {
                    continue;
                }
                bins[particle.cellIndex * NumGroups + group].push_back(
                    particleIndex);
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

    void adjustExistingParticles(
        std::vector<MCParticle> &particles,
        double fullDt)
    {

            owner_.splitComptonRiskyParticles(particles, fullDt);
            if(!owner_.parameters_.MMC)
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
                        const std::pair<std::size_t, std::size_t> &neighbors =
                            owner_.componentGrid().GetFaceNeighbors(faceIdx);
                        std::size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
                        PointT neighborPoint;
                        PointT neighborVelocity;
                        if(neighborIdx < Ncells && !owner_.componentGrid().IsPointOutsideBox(neighborIdx))
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
