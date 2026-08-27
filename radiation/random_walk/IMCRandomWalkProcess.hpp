#ifndef STORM_RADIATION_IMCRANDOM_WALK_PROCESS_HPP
#define STORM_RADIATION_IMCRANDOM_WALK_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"

namespace STORM::radiation_imc_detail {

template<typename Owner>
class IMCRandomWalkProcess final : public IMCComponentBase<Owner>
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
    explicit IMCRandomWalkProcess(Owner &owner) : Base(owner)
    {}

    double computeMinDistanceToFaces(
        std::size_t cellIndex, const PointT &location) const
    {

            double Ro = std::numeric_limits<double>::max();
            const std::size_t begin = owner_.componentGridData().cellFaceOffsets[cellIndex];
            const std::size_t end = owner_.componentGridData().cellFaceOffsets[cellIndex + 1];
            for(std::size_t f = begin; f < end; ++f)
            {
                double d = ScalarProd(location - owner_.componentGridData().pointsOnFaces[f],
                                      owner_.componentGridData().normals[f]);
                Ro = std::min(Ro, d);
            }
            return (Ro > 0.0) ? Ro : 0.0;
    }

    double computeCellSurfaceArea(std::size_t cellIndex) const
    {

            double surfaceArea = 0.0;
            for(std::size_t faceIdx : owner_.componentGrid().GetCellFaces(cellIndex))
            {
                surfaceArea += owner_.componentGrid().GetArea(faceIdx);
            }
            return surfaceArea;
    }

    void precomputeRandomWalkData()
    {

            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            owner_.rwCellEligible_.assign(Ncells, false);
            owner_.rwCellTotalOpacity_.assign(Ncells, 0.0);
            if(owner_.parameters_.withMultigroupOpacity)
            {
                owner_.rwCellData_.resize(Ncells);
            }

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                const CellT &cell = owner_.cells_[i];
                double scatOp = owner_.scatteringOpacities_[i];
                if(!std::isfinite(scatOp) || scatOp < 0.0)
                {
                    StormError eo("RadiationIMC random-walk precompute received an invalid scattering opacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Scattering opacity", scatOp);
                    throw eo;
                }
                double sigmaT_gray = owner_.planckOpacities_[i] + scatOp;
                owner_.rwCellTotalOpacity_[i] = sigmaT_gray;

                double surfaceArea = owner_.computeCellSurfaceArea(i);
                double volume = owner_.componentGrid().GetVolume(i);
                double meanChordLength = (surfaceArea > 0.0) ? 4.0 * volume / surfaceArea : 0.0;

                if(!owner_.parameters_.withMultigroupOpacity)
                {
                    owner_.rwCellEligible_[i] = (sigmaT_gray * meanChordLength >= owner_.parameters_.rwMinCellOpticalDepth);
                }
                else
                {
                    GroupArray energyCenters = owner_.opacity_->getEnergyCenters(owner_.energyBoundaries_);
                    double kT = units::k_boltz * cell.temperature;

                    double totalSigABgAll = 0.0;
                    double totalBgDiff = 0.0, sumBgSigADiff = 0.0, sumBgSigTDiff = 0.0, sumBgOverSigTDiff = 0.0;
                    std::size_t cutoff = 0;
                    bool foundNonDiffusive = false;

                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        double Bg = ddmc::PlanckBandMass(
                            owner_.energyBoundaries_, kT, g, g + 1);
                        double sigA_g = owner_.opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                        double scatOp_g = owner_.opacity_->CalcScatteringOpacity(
                            cell, energyCenters[g]);
                        if(!std::isfinite(sigA_g) || sigA_g < 0.0 ||
                           !std::isfinite(scatOp_g) || scatOp_g < 0.0)
                        {
                            StormError eo("RadiationIMC random-walk precompute received an invalid multigroup opacity");
                            eo.addEntry("Cell index", i);
                            eo.addEntry("Group", g);
                            eo.addEntry("Absorption opacity", sigA_g);
                            eo.addEntry("Scattering opacity", scatOp_g);
                            throw eo;
                        }
                        double sigT_g = sigA_g + scatOp_g;

                        totalSigABgAll += sigA_g * Bg;

                        if(!foundNonDiffusive && sigT_g * meanChordLength >= owner_.parameters_.rwMinCellOpticalDepth)
                        {
                            cutoff = g + 1;
                            totalBgDiff += Bg;
                            sumBgSigADiff += Bg * sigA_g;
                            sumBgSigTDiff += Bg * sigT_g;
                            if(sigT_g > 0.0)
                            {
                                sumBgOverSigTDiff += Bg / sigT_g;
                            }
                        }
                        else
                        {
                            foundNonDiffusive = true;
                        }
                    }

                    PGRWCellData &data = owner_.rwCellData_[i];
                    if(cutoff > 0 && totalBgDiff > 0.0)
                    {
                        data.groupCutoff = cutoff;
                        data.sigmaA_bar = sumBgSigADiff / totalBgDiff;
                        data.sigmaT_bar = sumBgSigTDiff / totalBgDiff;
                        data.D = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                        data.gamma = (totalSigABgAll > 0.0) ? sumBgSigADiff / totalSigABgAll : 1.0;
                        owner_.rwCellTotalOpacity_[i] = data.sigmaT_bar;
                        owner_.rwCellEligible_[i] = true;
                    }
                    else
                    {
                        data = PGRWCellData{};
                        owner_.rwCellEligible_[i] = false;
                    }
                }
            }
    }

    bool tryRandomWalkStep(
        MCParticle &particle, Functionality &functionality)
    {

            std::size_t cellIndex = particle.cellIndex;
            CellT &cell = owner_.cells_[cellIndex];

            if(owner_.SharedRandomWalkKernelEligible())
            {
                gpu::GreyIMCViews<PointT> views =
                    owner_.GetHostTransportViews();
                transport::RandomWalkResult result =
                    transport::TryAdvanceRandomWalk(particle, views);
                if(result.invalid)
                {
                    StormError eo(
                        "RadiationIMC GPU-compatible random walk received invalid data");
                    eo.addEntry("Cell index", particle.cellIndex);
                    throw eo;
                }
                if(!result.taken)
                {
                    return false;
                }
                functionality = result.step;
                return true;
            }

            double Ro = owner_.computeMinDistanceToFaces(cellIndex, particle.location);

            double sigmaT, sigma_a_eff, D_phys, gamma_rw;
            bool isPGRW = owner_.parameters_.withMultigroupOpacity;
            std::size_t groupCutoff = 0;

            if(isPGRW)
            {
                const PGRWCellData &rwd = owner_.rwCellData_[cellIndex];
                sigmaT = rwd.sigmaT_bar;
                sigma_a_eff = rwd.sigmaA_bar;
                D_phys = rwd.D;
                gamma_rw = rwd.gamma;
                groupCutoff = rwd.groupCutoff;
            }
            else
            {
                sigmaT = owner_.rwCellTotalOpacity_[cellIndex];
                sigma_a_eff = owner_.planckOpacities_[cellIndex];
                D_phys = (sigmaT > 0.0) ? units::clight / (3.0 * sigmaT) : 0.0;
                gamma_rw = 1.0;
            }

            bool doRW = (Ro > 0.0 && sigmaT > 0.0 && D_phys > 0.0
                         && Ro * sigmaT >= owner_.parameters_.rwMinParticleOpticalDepth);

            if(doRW && isPGRW)
            {
                double cutoffEnergy = owner_.energyBoundaries_[groupCutoff];
                double coFreq = particle.frequency;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro && !owner_.parameters_.MMC)
                    {
                        double dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                        coFreq *= dopplerShift;
                    }
                }
                owner_.clampFrequencyToBounds(coFreq);
                if(coFreq >= cutoffEnergy)
                {
                    doRW = false;
                }
            }

            if(!doRW)
            {
                return false;
            }

            PointT oldVelocity = particle.velocity;
            double oldWeight = particle.weight;
            double f = owner_.factorFleck_[cellIndex];
            double tauLeak = owner_.randomWalk_->sampleLeakTime(owner_.randomUnitOpen(particle));
            double tLeak = tauLeak * Ro * Ro / D_phys;

            double tCensus = particle.timeLeft;

            double tUpscatter = std::numeric_limits<double>::max();
            if(isPGRW && gamma_rw < 1.0 && sigma_a_eff > 0.0 && f > 0.0)
            {
                double xiUp = owner_.randomUnitOpen(particle);
                tUpscatter = -std::log(xiUp) / (units::clight * (1.0 - f) * sigma_a_eff * (1.0 - gamma_rw));
            }

            enum { RW_LEAK, RW_CENSUS, RW_UPSCATTER };
            int rwEvent;
            double dt;
            if(tLeak <= tCensus && tLeak <= tUpscatter)
            {
                rwEvent = RW_LEAK;
                dt = tLeak;
            }
            else if(tCensus <= tUpscatter)
            {
                rwEvent = RW_CENSUS;
                dt = tCensus;
            }
            else
            {
                rwEvent = RW_UPSCATTER;
                dt = tUpscatter;
            }

            double rwAbsRate = sigma_a_eff * f * units::clight;
            double rwExp = std::expm1(-dt * rwAbsRate);
            if(!owner_.parameters_.noHydroFeedback)
            {
                owner_.tallyMaterialEnergy(cellIndex, -rwExp * particle.weight);
            }
            if(rwAbsRate > 0.0)
            {
                owner_.tallyRadiationEnergy(
                    cellIndex, particle.weight * rwExp * (-1.0 / rwAbsRate));
                if(owner_.parameters_.withEgTimeAvg && owner_.parameters_.withMultigroupOpacity)
                {
                    std::size_t g = owner_.opacity_->findGroup(particle.frequency, owner_.energyBoundaries_);
                    if(g < NumGroups)
                    {
                        owner_.tallyGroupRadiationEnergy(
                            cellIndex, g,
                            particle.weight * rwExp * (-1.0 / rwAbsRate));
                    }
                }
            }
            particle.weight *= 1.0 + rwExp;

            particle.timeLeft -= dt;

            if(std::abs(particle.weight) < particle.initialWeight * 1e-4)
            {
                functionality.change = ParticleStatus::REMOVE;
                if(!owner_.parameters_.noHydroFeedback)
                {
                    owner_.tallyMaterialEnergy(cellIndex, particle.weight);
                }
                return true;
            }

            constexpr double RW_PI = 3.14159265358979323846;
            double cosTheta = 2.0 * owner_.randomUnitOpen(particle) - 1.0;
            double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
            double phi = 2.0 * RW_PI * owner_.randomUnitOpen(particle);
            PointT posDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);

            double displacement;
            if(rwEvent == RW_LEAK)
            {
                displacement = Ro;
            }
            else
            {
                double tauPos = D_phys * dt / (Ro * Ro);
                displacement = Ro * owner_.randomWalk_->sampleRadius(tauPos, owner_.randomUnitOpen(particle));
            }

            if(displacement > Ro * (1.0 + 1e-12))
            {
                displacement = Ro;
            }

            PointT rwCenter = particle.location;
            particle.location = rwCenter + displacement * posDir;

            static constexpr double nudge = 1e-6;
            particle.location = particle.location * (1.0 - nudge) + nudge * owner_.componentGrid().GetMeshPoint(cellIndex);

            const std::size_t faceBegin = owner_.componentGridData().cellFaceOffsets[cellIndex];
            const std::size_t faceEnd = owner_.componentGridData().cellFaceOffsets[cellIndex + 1];
            for(std::size_t fi = faceBegin; fi < faceEnd; ++fi)
            {
                double d = ScalarProd(
                    particle.location - owner_.componentGridData().pointsOnFaces[fi],
                    owner_.componentGridData().normals[fi]);
                if(d < 0.0)
                {
                    displacement *= 0.99;
                    particle.location = rwCenter + displacement * posDir;
                    fi = faceBegin - 1;
                }
            }

            particle.velocity = owner_.sampleRandomVelocity(cell, particle);

        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                MCParticle polarizationParticle = particle;
                polarizationParticle.velocity = oldVelocity;
                double dtCo = dt;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro && !owner_.parameters_.MMC)
                    {
                        dtCo *= radiation_imc_detail::computeDopplerShift<PointT>(
                            polarizationParticle, cell);
                        radiation_imc_detail::lorentzTransformToComoving<PointT>(
                            polarizationParticle, cell);
                    }
                }
                ParticleCounterEngine polarizationEngine(
                    particle.rngKey, particle.rngCounter);
                std::uniform_real_distribution<double> polarizationUnit(0.0, 1.0);
                polarization::applyAcceleratedPolarizationHistory<PointT>(
                    polarizationParticle, dtCo,
                    std::max(0.0, sigmaT - sigma_a_eff),
                    std::max(0.0, (1.0 - f) * sigma_a_eff),
                    particle.velocity,
                    owner_.parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
                    owner_.parameters_.postProcess.polarization.depolarizationScatterings,
                    polarizationEngine, polarizationUnit);
                particle.stokesQ = polarizationParticle.stokesQ;
                particle.stokesU = polarizationParticle.stokesU;
                particle.polarizationBasis = polarizationParticle.polarizationBasis;
                particle.polarizationInitialized = polarizationParticle.polarizationInitialized;
                particle.radiationState.pendingMeanScatterings =
                    polarizationParticle.radiationState.pendingMeanScatterings;
                particle.polarizationBasis = polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity);
            }
        #endif

            if(rwEvent == RW_UPSCATTER && isPGRW)
            {
                GroupArray cumOp = owner_.opacity_->GetCumulativeOpacity(cell, owner_.energyBoundaries_);
                double cdfAtCutoff = cumOp[groupCutoff - 1];
                double cdfTotal = cumOp[NumGroups - 1];
                if(cdfTotal > cdfAtCutoff)
                {
                    double lo = cdfAtCutoff / cdfTotal;
                    double xi = owner_.randomUnitOpen(particle);
                    particle.frequency = owner_.opacity_->GetThermalEnergy(cell, lo + xi * (1.0 - lo), owner_.energyBoundaries_);
                }
                else
                {
                    particle.frequency = std::nextafter(
                        owner_.energyBoundaries_[groupCutoff],
                        std::numeric_limits<double>::max());
                }
                owner_.clampFrequencyToBounds(particle.frequency);
            }

            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(owner_.parameters_.withHydro && !owner_.parameters_.MMC)
                {
                    radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
                    if(owner_.parameters_.withMultigroupOpacity)
                    {
                        owner_.clampFrequencyToBounds(particle.frequency);
                    }
        #ifdef MONTECARLO_POLARIZATION
                    if(owner_.polarizationEnabled())
                    {
                        particle.polarizationBasis = polarization::projectBasisToDirection(
                            particle.polarizationBasis, particle.velocity);
                    }
        #endif
                    if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                    {
                        if(!owner_.parameters_.diffusionPressureGradient && !owner_.parameters_.noHydroFeedback)
                        {
                            owner_.tallyMomentum(
                                cellIndex,
                                (oldWeight * oldVelocity -
                                 particle.weight * particle.velocity) *
                                    units::inv_clight2);
                        }
                    }
                }
            }

            if(rwEvent == RW_CENSUS)
            {
                functionality.change = ParticleStatus::DONE;
            }
            return true;
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMCRANDOM_WALK_PROCESS_HPP
