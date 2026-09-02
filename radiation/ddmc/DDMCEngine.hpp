#ifndef STORM_RADIATION_DDMCENGINE_HPP
#define STORM_RADIATION_DDMCENGINE_HPP

#include "../imc/IMCComponentBase.hpp"

namespace STORM::radiation_imc_detail {

template<typename Owner>
class DDMCEngine final : public IMCComponentBase<Owner>
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
    explicit DDMCEngine(Owner &owner) : Base(owner)
    {}

    void precomputeDDMCData()
    {

            const std::size_t Ncells = owner_.componentGrid().GetPointNo();
            owner_.ddmcCellData_.assign(Ncells, DDMCCellData{});
            owner_.ddmcLeakReciprocityResidualMax_ = 0.0;
            owner_.ddmcLeakReciprocityCheckCount_ = 0;
            owner_.ddmcResidentLeakCount_ = 0;
            owner_.ddmcTransportLeakCount_ = 0;
            owner_.ddmcRemoteResidentLeakCount_ = 0;
            owner_.ddmcMPIFaceFluxReductionCount_ = 0;
            owner_.ddmcLeakInvalidGeometryCount_ = 0;
            owner_.ddmcUnsupportedBoundaryFaceCount_ = 0;
            owner_.ddmcInterfaceIncidentCount_ = 0;
            owner_.ddmcInterfaceAdmittedCount_ = 0;
            owner_.ddmcInterfaceReflectedCount_ = 0;
            owner_.ddmcInterfaceGuAppliedCount_ = 0;
            owner_.ddmcInterfaceGuFallbackCount_ = 0;
            owner_.ddmcInterfaceBypassCount_ = 0;
            owner_.ddmcInterfaceSplitPacketCount_ = 0;
            owner_.ddmcInterfaceFluxTallyCount_ = 0;
            owner_.ddmcInterfaceMinimumMu_ = std::numeric_limits<double>::infinity();
            owner_.ddmcDiagnosticEvents_.clear();
            owner_.ddmcExternalSourceCandidateFaceCount_ = 0;
            owner_.ddmcExternalSourceAcceleratedFaceCount_ = 0;
            owner_.ddmcExternalSourceExplicitFallbackFaceCount_ = 0;
            owner_.ddmcExternalSourceInteriorExcludedCellCount_ = 0;
            owner_.ddmcExternalSourceThermalizationCount_ = 0;
            owner_.ddmcExternalSourceStayDDMCCount_ = 0;
            owner_.ddmcExternalSourceToIMCCount_ = 0;
            owner_.ddmcExternalSourceThermalizedEnergy_ = 0.0;
            owner_.ddmcExternalSourceToIMCEnergy_ = 0.0;
            owner_.ddmcExternalSourceMinimumFaceOpticalDepth_ = std::numeric_limits<double>::infinity();

            // Eligibility is exchanged separately from the local cell data.  This
            // is important for Voronoi/MPI grids: a ghost index is not a local cell
            // index and must never index ddmcCellData_.
            const std::size_t pointCount = std::max(owner_.componentGrid().GetTotalPointNumber(), owner_.componentGrid().getMeshPoints().size());
            owner_.ddmcPointEligible_.assign(pointCount, 0);
            owner_.ddmcPointDiffusionCoefficient_.assign(pointCount, 0.0);
            owner_.ddmcPointSigmaDiffusion_.assign(pointCount, 0.0);
            owner_.ddmcPointSigmaParticleGate_.assign(pointCount, 0.0);
            owner_.ddmcPointGroupCutoff_.assign(pointCount, 0);
            owner_.ddmcPointVelocity_.assign(pointCount, PointT{});
            owner_.ddmcPointCellID_.assign(pointCount, std::numeric_limits<std::size_t>::max());

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                DDMCCellData &data = owner_.ddmcCellData_[i];
                const CellT &cell = owner_.cells_[i];
                data.eligibilityReason = ddmc::EligibilityReason::InvalidGeometry;
                double scatOp = owner_.scatteringOpacities_[i];
                if(not std::isfinite(scatOp) or scatOp < 0.0)
                {
                    StormError eo("RadiationIMC DDMC precompute received an invalid scattering opacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Scattering opacity", scatOp);
                    throw eo;
                }
                double volume = owner_.componentGrid().GetVolume(i);
                double surfaceArea = owner_.computeCellSurfaceArea(i);
                if(volume <= 0.0 or surfaceArea <= 0.0)
                {
                    continue;
                }
                double meanChordLength = 4.0 * volume / surfaceArea;

                if(owner_.parameters_.ddmcUseMultigroupPGRW and owner_.parameters_.withMultigroupOpacity)
                {
                    GroupArray energyCenters = owner_.opacity_->getEnergyCenters(owner_.energyBoundaries_);
                    double kT = units::k_boltz * cell.temperature;
                    if(kT <= 0.0)
                    {
                        data.eligibilityReason = ddmc::EligibilityReason::InvalidThermalState;
                        continue;
                    }
                    double totalBgDiff = 0.0;
                    double totalSigABg = 0.0;
                    double sumBgSigADiff = 0.0;
                    double sumBgSigTDiff = 0.0;
                    double sumBgOverSigTDiff = 0.0;
                    std::size_t cutoff = 0;
                    bool foundNonDiffusive = false;
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        double sigA_g = owner_.opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                        double scatOp_g = owner_.opacity_->CalcScatteringOpacity(cell, energyCenters[g]);
                        if(not std::isfinite(sigA_g) or sigA_g < 0.0 or not std::isfinite(scatOp_g) or scatOp_g < 0.0)
                        {
                            StormError eo("RadiationIMC DDMC precompute received an invalid multigroup opacity");
                            eo.addEntry("Cell index", i);
                            eo.addEntry("Group", g);
                            eo.addEntry("Absorption opacity", sigA_g);
                            eo.addEntry("Scattering opacity", scatOp_g);
                            throw eo;
                        }
                        double sigT_g = sigA_g + scatOp_g;
                        double Bg = ddmc::PlanckBandMass(owner_.energyBoundaries_, kT, g, g + 1);
                        totalSigABg += sigA_g * Bg;
                        if(not foundNonDiffusive and sigT_g * meanChordLength >= owner_.parameters_.ddmcMinCellOpticalDepth)
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
                    if(cutoff > 0 and totalBgDiff > 0.0)
                    {
                        data.groupCutoff = std::min(cutoff, owner_.parameters_.ddmcMaxGroupCutoff);
                        data.sigmaA = sumBgSigADiff / totalBgDiff;
                        data.sigmaT = sumBgSigTDiff / totalBgDiff;
                        data.sigmaEnergyAbs = data.sigmaA;
                        data.sigmaMomentum = data.sigmaT;
                        data.sigmaDiffusion = data.sigmaT;
                        data.sigmaParticleGate = data.sigmaT;
                        data.sigmaGroupExit = data.sigmaT;
                        data.diffusionCoefficient = (sumBgOverSigTDiff > 0.0)? (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff : 0.0;
                        data.gamma = (totalSigABg > 0.0)? sumBgSigADiff / totalSigABg : 1.0;
                        data.eligible = data.sigmaParticleGate > 0.0 and
                                        data.sigmaParticleGate * meanChordLength >= owner_.parameters_.ddmcMinCellOpticalDepth and
                                        data.diffusionCoefficient > 0.0;
                    }
                }
                else
                {
                    data.sigmaA = owner_.planckOpacities_[i];
                    data.sigmaT = data.sigmaA + scatOp;
                    data.sigmaEnergyAbs = data.sigmaA;
                    data.sigmaMomentum = data.sigmaT;
                    data.sigmaDiffusion = data.sigmaT;
                    data.sigmaParticleGate = data.sigmaT;
                    data.sigmaGroupExit = data.sigmaT;
                    data.diffusionCoefficient = (data.sigmaDiffusion > 0.0)? units::clight / (3.0 * data.sigmaDiffusion) : 0.0;
                    data.gamma = 1.0;
                    data.eligible = (data.sigmaParticleGate * meanChordLength >= owner_.parameters_.ddmcMinCellOpticalDepth and data.diffusionCoefficient > 0.0);
                }

                if(!data.eligible)
                {
                    data.eligibilityReason = (data.diffusionCoefficient > 0.0)? ddmc::EligibilityReason::OpticallyThin : ddmc::EligibilityReason::NoDiffusionCoefficient;
                }

                // External-face exclusions are local properties and must be applied
                // before eligibility is sent to a neighboring rank.  Otherwise a
                // ghost can incorrectly advertise DDMC eligibility even though the
                // owner later rejects the cell because of an unsupported boundary.
                if(data.eligible)
                {
                    for(std::size_t faceIdx : owner_.componentGrid().GetCellFaces(i))
                    {
                        const std::pair<std::size_t, std::size_t> &neighbors = owner_.componentGrid().GetFaceNeighbors(faceIdx);
                        std::size_t const next = (neighbors.first == i)? neighbors.second : neighbors.first;
                        if(owner_.componentGrid().IsPointOutsideBox(next))
                        {
                            DDMCBoundaryFaceBehavior const behavior = owner_.componentBoundary()->getDDMCBoundaryFaceBehavior(faceIdx, i, next);
                            if(behavior == DDMCBoundaryFaceBehavior::ReflectingRigid)
                            {
                                ++data.rigidBoundaryFaceCount;
                            }
                            else
                            {
                                ++data.unsupportedBoundaryFaceCount;
                                ++owner_.ddmcUnsupportedBoundaryFaceCount_;
                                if(data.firstUnsupportedBoundaryFace == std::numeric_limits<std::size_t>::max())
                                {
                                    data.firstUnsupportedBoundaryFace = faceIdx;
                                }
                                data.boundaryExcluded = true;
                                data.eligible = false;
                                data.eligibilityReason = ddmc::EligibilityReason::BoundaryExcluded;
                            }
                        }
                    }
                }

                owner_.ddmcPointEligible_[i] = data.eligible ? 1 : 0;
                owner_.ddmcPointDiffusionCoefficient_[i] = data.diffusionCoefficient;
                owner_.ddmcPointSigmaDiffusion_[i] = data.sigmaDiffusion;
                owner_.ddmcPointSigmaParticleGate_[i] = data.sigmaParticleGate;
                owner_.ddmcPointGroupCutoff_[i] = data.groupCutoff;
                owner_.ddmcPointCellID_[i] = radiation_imc_detail::ddmcStableCellID(owner_.componentGrid(), i, cell);
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    owner_.ddmcPointVelocity_[i] = cell.velocity;
                }
            }

            if(owner_.postProcessExternalSourceMode_)
            {
                if(owner_.postProcessExternalSourceLocalCellIndices_.size() != owner_.postProcessExternalSources_.size())
                {
                    throw StormError("DDMC external source-to-cell map has inconsistent size");
                }
                for(std::size_t i = 0; i < Ncells; ++i)
                {
                    DDMCCellData &data = owner_.ddmcCellData_[i];
                    std::size_t const cellID = radiation_imc_detail::ddmcStableCellID(owner_.componentGrid(), i, owner_.cells_[i]);
                    if(owner_.postProcessExternalSourceInteriorCellIDs_.count(cellID))
                    {
                        data.externalSourceInteriorExcluded = true;
                        data.eligible = false;
                        owner_.ddmcPointEligible_[i] = 0;
                        ++owner_.ddmcExternalSourceInteriorExcludedCellCount_;
                    }
                }
                for(std::size_t sourceIndex = 0; sourceIndex < owner_.postProcessExternalSources_.size(); sourceIndex++)
                {
                    PostProcessExternalSource const &source = owner_.postProcessExternalSources_[sourceIndex];
                    std::size_t const i = owner_.postProcessExternalSourceLocalCellIndices_[sourceIndex];
                    if(i >= Ncells)
                    {
                        throw StormError("DDMC external source-to-cell map is stale");
                    }
                    DDMCCellData &data = owner_.ddmcCellData_[i];
                    ++data.externalSourceBoundaryFaceCount;
                    ++owner_.ddmcExternalSourceCandidateFaceCount_;
                    PointT const normal = source.outwardNormal / std::max(fastabs(source.outwardNormal), std::numeric_limits<double>::min());
                    double const faceDistance = std::abs(ScalarProd(owner_.componentGrid().FaceCM(source.faceIndex) - owner_.componentGrid().GetMeshPoint(i), normal));
                    double const faceTau = data.sigmaDiffusion * faceDistance;
                    double const diagnosticFaceTau = (faceTau >= 0.0 && std::isfinite(faceTau)) ? faceTau : 0.0;
                    data.minExternalSourceFaceOpticalDepth = std::min(data.minExternalSourceFaceOpticalDepth, diagnosticFaceTau);
                    owner_.ddmcExternalSourceMinimumFaceOpticalDepth_ = std::min(owner_.ddmcExternalSourceMinimumFaceOpticalDepth_, diagnosticFaceTau);
                    if(not (faceTau >= owner_.parameters_.ddmcExternalSourceMinFaceOpticalDepth) or not std::isfinite(faceTau))
                    {
                        data.externalSourceFaceOpticalDepthExcluded = true;
                        data.eligible = false;
                        owner_.ddmcPointEligible_[i] = 0;
                    }
                }
                for(DDMCCellData const &data : owner_.ddmcCellData_)
                {
                    if(data.externalSourceBoundaryFaceCount > 0 and not data.eligible)
                    {
                        owner_.ddmcExternalSourceExplicitFallbackFaceCount_ += data.externalSourceBoundaryFaceCount;
                    }
                }
            }

        #ifdef STORM_WITH_MPI
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointEligible_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointDiffusionCoefficient_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointSigmaDiffusion_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointSigmaParticleGate_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointGroupCutoff_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointVelocity_, true);
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointCellID_, true);
        #endif

            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                // Compute the velocity-divergence operator after ghost velocities are
                // available.  Static cells simply compile this block away.
                for(std::size_t i = 0; i < Ncells; ++i)
                {
                    DDMCCellData &data = owner_.ddmcCellData_[i];
                    double const volume = owner_.componentGrid().GetVolume(i);
                    if(!(volume > 0.0))
                    {
                        continue;
                    }
                    PointT const center = owner_.componentGrid().GetMeshPoint(i);
                    double divergence = 0.0;
                    double maxJump = 0.0;
                    for(std::size_t faceIndex : owner_.componentGrid().GetCellFaces(i))
                    {
                        std::pair<std::size_t, std::size_t> const &neighbors =
                            owner_.componentGrid().GetFaceNeighbors(faceIndex);
                        std::size_t const next = neighbors.first == i
                            ? neighbors.second : neighbors.first;
                        if(owner_.componentGrid().IsPointOutsideBox(next))
                        {
                            continue;
                        }
                        PointT normal = owner_.componentGrid().Normal(faceIndex);
                        double const normalMagnitude = fastabs(normal);
                        if(!(normalMagnitude > 0.0))
                        {
                            continue;
                        }
                        normal = normal / normalMagnitude;
                        PointT const targetCenter = owner_.componentGrid().GetMeshPoint(next);
                        if(ScalarProd(normal, targetCenter - center) < 0.0)
                        {
                            normal = -normal;
                        }
                        PointT targetVelocity = owner_.ddmcPointVelocity_[next];
                        divergence += 0.5 * ScalarProd(
                            owner_.cells_[i].velocity + targetVelocity, normal) *
                            owner_.componentGrid().GetArea(faceIndex);
                        maxJump = std::max(maxJump,
                            fastabs(targetVelocity - owner_.cells_[i].velocity) *
                            units::inv_clight);
                    }
                    data.velocityDivergence = divergence / volume;
                    data.maxFaceVelocityJumpOverC = maxJump;
                }
            }

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                DDMCCellData &data = owner_.ddmcCellData_[i];
                if(!data.eligible)
                {
                    continue;
                }
                double volume = owner_.componentGrid().GetVolume(i);
                PointT const cellCenter = owner_.componentGrid().GetMeshPoint(i);
                double const sourceBandMass =
                    (owner_.parameters_.ddmcUseMultigroupPGRW &&
                     owner_.parameters_.withMultigroupOpacity)
                    ? ddmc::PlanckBandMass(
                        owner_.energyBoundaries_, units::k_boltz * owner_.cells_[i].temperature,
                        0, data.groupCutoff) : 1.0;
                for(std::size_t faceIdx : owner_.componentGrid().GetCellFaces(i))
                {
                    auto const sourceFace =
                        owner_.postProcessExternalSourceFaceIndex_.find(faceIdx);
                    if(owner_.postProcessExternalSourceMode_ &&
                       sourceFace != owner_.postProcessExternalSourceFaceIndex_.end())
                    {
                        if(sourceFace->second >=
                           owner_.postProcessExternalSources_.size())
                        {
                            throw StormError(
                                "DDMC external-source face map contains an invalid source index");
                        }
                        PostProcessExternalSource const &source =
                            owner_.postProcessExternalSources_[sourceFace->second];
                        if(radiation_imc_detail::ddmcStableCellID(
                               owner_.componentGrid(), i, owner_.cells_[i]) != source.cellID)
                        {
                            throw StormError(
                                "DDMC attempted to build an external-source leak from the interior side");
                        }
                        PointT sourceNormal = source.outwardNormal /
                            std::max(fastabs(source.outwardNormal),
                                     std::numeric_limits<double>::min());
                        double const sourceDistanceToFace = std::abs(ScalarProd(
                            owner_.componentGrid().FaceCM(faceIdx) - cellCenter, sourceNormal));
                        double const area = owner_.componentGrid().GetArea(faceIdx);
                        double const boundaryRate = ddmc::BoundaryLeakRate(
                            area, volume, data.sigmaDiffusion,
                            sourceDistanceToFace, units::clight);
                        if(!(boundaryRate > 0.0) || !std::isfinite(boundaryRate))
                        {
                            throw StormError(
                                "DDMC external-source boundary has an invalid leak rate");
                        }
                        DDMCFaceLeak faceLeak;
                        faceLeak.faceIndex = faceIdx;
                        faceLeak.nextCellIndex = i;
                        faceLeak.kind = ddmc::FaceKind::ThermalizingBoundary;
                        faceLeak.rate = boundaryRate;
                        faceLeak.boundaryRate = boundaryRate;
                        faceLeak.transportRate = boundaryRate;
                        faceLeak.sourceBandMass = sourceBandMass;
                        faceLeak.commonBandMass = sourceBandMass;
                        faceLeak.ddmcFraction = 1.0;
                        faceLeak.area = area;
                        faceLeak.sourceDistanceToFace = sourceDistanceToFace;
                        faceLeak.targetDDMCEligible = true;
                        faceLeak.targetGroupCutoff = data.groupCutoff;
                        faceLeak.outwardNormal = sourceNormal;
                        faceLeak.thermalizingLocation = source.location;
                        data.faceLeaks.push_back(faceLeak);
                        data.totalLeakRate += boundaryRate;
                        data.faceAreaSum += area;
                        double const nx = sourceNormal[0];
                        double const ny = sourceNormal[1];
                        double const nz = sourceNormal[2];
                        data.fluxMatrix[0] += area * nx * nx;
                        data.fluxMatrix[1] += area * nx * ny;
                        data.fluxMatrix[2] += area * nx * nz;
                        data.fluxMatrix[3] += area * ny * ny;
                        data.fluxMatrix[4] += area * ny * nz;
                        data.fluxMatrix[5] += area * nz * nz;
                        ++owner_.ddmcExternalSourceAcceleratedFaceCount_;
                        continue;
                    }
                    const std::pair<std::size_t, std::size_t> &neighbors =
                        owner_.componentGrid().GetFaceNeighbors(faceIdx);
                    std::size_t nextCellIndex = (neighbors.first == i) ? neighbors.second : neighbors.first;
                    if(owner_.componentGrid().IsPointOutsideBox(nextCellIndex))
                    {
                        if(owner_.componentBoundary()->getDDMCBoundaryFaceBehavior(
                               faceIdx, i, nextCellIndex) !=
                           DDMCBoundaryFaceBehavior::ReflectingRigid)
                        {
                            data.boundaryExcluded = true;
                        }
                        continue;
                    }
                    PointT normal = owner_.componentGrid().Normal(faceIdx);
                    double const normalMag = fastabs(normal);
                    double const area = owner_.componentGrid().GetArea(faceIdx);
                    if(!(normalMag > 0.0) || !std::isfinite(normalMag) ||
                       !(area > 0.0) || !std::isfinite(area))
                    {
                        ++owner_.ddmcLeakInvalidGeometryCount_;
                        continue;
                    }
                    normal = normal / normalMag;
                    PointT faceCenter = owner_.componentGrid().FaceCM(faceIdx);
                    PointT outwardReference = owner_.componentGrid().IsPointOutsideBox(nextCellIndex)
                        ? faceCenter - cellCenter
                        : owner_.componentGrid().GetMeshPoint(nextCellIndex) - cellCenter;
                    if(ScalarProd(normal, outwardReference) < 0.0)
                    {
                        normal = -normal;
                    }
                    double sourceDistance = std::abs(ScalarProd(
                        faceCenter - cellCenter, normal));
                    if(sourceDistance <= 0.0)
                    {
                        sourceDistance = 0.5 * std::abs(ScalarProd(
                            owner_.componentGrid().GetMeshPoint(nextCellIndex) - cellCenter,
                            normal));
                    }
                    if(!(sourceDistance > 0.0) || !std::isfinite(sourceDistance))
                    {
                        ++owner_.ddmcLeakInvalidGeometryCount_;
                        continue;
                    }

                    double targetDistance = 0.0;
                    if(nextCellIndex < owner_.componentGrid().getMeshPoints().size())
                    {
                        targetDistance = std::abs(ScalarProd(
                            owner_.componentGrid().GetMeshPoint(nextCellIndex) - faceCenter,
                            normal));
                    }

                    bool const targetEligible =
                        nextCellIndex < owner_.ddmcPointEligible_.size() &&
                        owner_.ddmcPointEligible_[nextCellIndex] != 0;
                    double internalRate = 0.0;
                    double conductance = 0.0;
                    if(targetEligible && targetDistance > 0.0)
                    {
                        conductance = ddmc::TwoSidedConductance(
                            area, sourceDistance,
                            data.diffusionCoefficient, targetDistance,
                            owner_.ddmcPointDiffusionCoefficient_[nextCellIndex]);
                        internalRate = conductance / volume;
                    }

                    double boundaryRate = ddmc::BoundaryLeakRate(
                        area, volume, data.sigmaDiffusion,
                        sourceDistance, units::clight);
                    std::size_t const targetCutoff =
                        nextCellIndex < owner_.ddmcPointGroupCutoff_.size()
                        ? owner_.ddmcPointGroupCutoff_[nextCellIndex] : 0;
                    double ddmcFraction = 0.0;
                    if(targetEligible && internalRate > 0.0)
                    {
                        if(!(owner_.parameters_.ddmcUseMultigroupPGRW &&
                             owner_.parameters_.withMultigroupOpacity) ||
                           targetCutoff >= data.groupCutoff)
                        {
                            ddmcFraction = 1.0;
                        }
                        else if(targetCutoff > 0 && sourceBandMass > 0.0)
                        {
                            ddmcFraction = std::clamp(
                                ddmc::PlanckBandMass(
                                    owner_.energyBoundaries_,
                                    units::k_boltz * owner_.cells_[i].temperature,
                                    0, targetCutoff) / sourceBandMass,
                                0.0, 1.0);
                        }
                    }

                    double const ddmcRate = ddmcFraction * internalRate;
                    double const transportRate =
                        (1.0 - ddmcFraction) * boundaryRate;
                    double const rate = ddmcRate + transportRate;
                    if(rate > 0.0 && std::isfinite(rate))
                    {
                        DDMCFaceLeak faceLeak;
                        faceLeak.faceIndex = faceIdx;
                        faceLeak.nextCellIndex = nextCellIndex;
                        faceLeak.kind = ddmcRate > 0.0
                            ? ddmc::FaceKind::Internal
                            : ddmc::FaceKind::InterfaceToIMC;
                        faceLeak.rate = rate;
                        faceLeak.internalRate = internalRate;
                        faceLeak.boundaryRate = boundaryRate;
                        faceLeak.ddmcRate = ddmcRate;
                        faceLeak.transportRate = transportRate;
                        faceLeak.ddmcFraction = ddmcFraction;
                        faceLeak.area = area;
                        faceLeak.sourceDistanceToFace = sourceDistance;
                        faceLeak.targetDistanceToFace = targetDistance;
                        faceLeak.conductance = conductance;
                        faceLeak.sourceBandMass = sourceBandMass;
                        faceLeak.commonBandMass = ddmcFraction * sourceBandMass;
                        faceLeak.targetGroupCutoff = targetCutoff;
                        faceLeak.targetDDMCEligible = ddmcRate > 0.0;
                        faceLeak.outwardNormal = normal;
                        data.faceLeaks.push_back(faceLeak);
                        double const nx = normal[0];
                        double const ny = normal[1];
                        double const nz = normal[2];
                        data.fluxMatrix[0] += faceLeak.area * nx * nx;
                        data.fluxMatrix[1] += faceLeak.area * nx * ny;
                        data.fluxMatrix[2] += faceLeak.area * nx * nz;
                        data.fluxMatrix[3] += faceLeak.area * ny * ny;
                        data.fluxMatrix[4] += faceLeak.area * ny * nz;
                        data.fluxMatrix[5] += faceLeak.area * nz * nz;
                        data.totalLeakRate += rate;
                        data.faceAreaSum += faceLeak.area;
                    }
                }
                if(data.boundaryExcluded || data.totalLeakRate <= 0.0)
                {
                    data.eligible = false;
                    data.faceLeaks.clear();
                    data.totalLeakRate = 0.0;
                    data.faceAreaSum = 0.0;
                    data.eligibilityReason = data.boundaryExcluded
                        ? ddmc::EligibilityReason::BoundaryExcluded
                        : ddmc::EligibilityReason::NoLeakage;
                }
                else
                {
                    data.eligibilityReason = ddmc::EligibilityReason::Eligible;
                }
            }

            // A first face pass uses the provisional local/ghost eligibility.  A
            // cell with no usable outgoing leak can only be rejected after that
            // pass, so refresh the advertised eligibility and downgrade any face
            // whose target was rejected.  The second exchange is important for
            // local/remote parity: a remote target must not remain an internal DDMC
            // channel merely because its owner had no usable face set.
            auto refreshMixedFaceChannels = [this](DDMCCellData &data)
            {
                data.totalLeakRate = 0.0;
                data.faceAreaSum = 0.0;
                data.fluxMatrix.fill(0.0);
                for(DDMCFaceLeak &face : data.faceLeaks)
                {
                    bool const targetEligible = face.nextCellIndex <
                        owner_.ddmcPointEligible_.size() &&
                        owner_.ddmcPointEligible_[face.nextCellIndex] != 0;
                    if(face.ddmcRate > 0.0 && !targetEligible)
                    {
                        face.kind = ddmc::FaceKind::InterfaceToIMC;
                        face.ddmcRate = 0.0;
                        face.ddmcFraction = 0.0;
                        face.transportRate = face.boundaryRate;
                        face.rate = face.transportRate;
                        face.targetDDMCEligible = false;
                    }
                    if(!(face.rate > 0.0) || !std::isfinite(face.rate))
                    {
                        continue;
                    }
                    data.totalLeakRate += face.rate;
                    data.faceAreaSum += face.area;
                    double const nx = face.outwardNormal[0];
                    double const ny = face.outwardNormal[1];
                    double const nz = face.outwardNormal[2];
                    data.fluxMatrix[0] += face.area * nx * nx;
                    data.fluxMatrix[1] += face.area * nx * ny;
                    data.fluxMatrix[2] += face.area * nx * nz;
                    data.fluxMatrix[3] += face.area * ny * ny;
                    data.fluxMatrix[4] += face.area * ny * nz;
                    data.fluxMatrix[5] += face.area * nz * nz;
                }
                if(data.totalLeakRate <= 0.0)
                {
                    data.eligible = false;
                    data.faceLeaks.clear();
                    data.faceAreaSum = 0.0;
                    data.eligibilityReason = ddmc::EligibilityReason::NoLeakage;
                }
            };

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                owner_.ddmcPointEligible_[i] =
                    owner_.ddmcCellData_[i].eligible ? 1 : 0;
            }
#ifdef STORM_WITH_MPI
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointEligible_, true);
#endif
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                refreshMixedFaceChannels(owner_.ddmcCellData_[i]);
            }
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                owner_.ddmcPointEligible_[i] =
                    owner_.ddmcCellData_[i].eligible ? 1 : 0;
            }
#ifdef STORM_WITH_MPI
            STORM::MPI_exchange_data(owner_.componentGrid(), owner_.ddmcPointEligible_, true);
#endif
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                refreshMixedFaceChannels(owner_.ddmcCellData_[i]);
            }

            // Deterministic local reciprocity check.  The same identity is used by
            // the distributed validation, where target coefficients come from the
            // exchanged ghost arrays above.
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                for(const DDMCFaceLeak &forward : owner_.ddmcCellData_[i].faceLeaks)
                {
                    std::size_t const j = forward.nextCellIndex;
                    if(forward.kind != ddmc::FaceKind::Internal || j >= Ncells || j <= i)
                    {
                        continue;
                    }
                    for(const DDMCFaceLeak &reverse : owner_.ddmcCellData_[j].faceLeaks)
                    {
                        if(reverse.faceIndex != forward.faceIndex ||
                           reverse.nextCellIndex != i ||
                           reverse.kind != ddmc::FaceKind::Internal)
                        {
                            continue;
                        }
                        double const residual = ddmc::ReciprocityResidual(
                            owner_.componentGrid().GetVolume(i), forward.internalRate,
                            owner_.componentGrid().GetVolume(j), reverse.internalRate);
                        owner_.ddmcLeakReciprocityResidualMax_ = std::max(
                            owner_.ddmcLeakReciprocityResidualMax_, residual);
                        ++owner_.ddmcLeakReciprocityCheckCount_;
                        break;
                    }
                }
            }
    }

    void addDDMCFluxContribution(
        std::size_t cellIndex, const PointT &contribution)
    {

            if(cellIndex < owner_.ddmcFluxRhsIntegrated_.size())
            {
                owner_.ddmcFluxRhsIntegrated_[cellIndex] += contribution;
            }
    }

    void applyDDMCMomentumFeedback(double fullDt)
    {

            (void)fullDt;
            if constexpr(!radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                return;
            }
            else
            {
                if(owner_.parameters_.noHydroFeedback || !owner_.parameters_.withHydro ||
                   owner_.parameters_.diffusionPressureGradient)
                {
                    return;
                }

        #ifdef STORM_WITH_MPI
                STORM::MPI_reduce_ghost_data(owner_.componentGrid(), owner_.ddmcFluxRhsIntegrated_);
                ++owner_.ddmcMPIFaceFluxReductionCount_;
        #endif

                for(std::size_t i = 0; i < owner_.componentGrid().GetPointNo(); ++i)
                {
                    if(i >= owner_.ddmcCellData_.size() ||
                       i >= owner_.ddmcFluxRhsIntegrated_.size())
                    {
                        continue;
                    }
                    DDMCCellData const &data = owner_.ddmcCellData_[i];
                    PointT const rhs = owner_.ddmcFluxRhsIntegrated_[i];
                    if(!data.eligible || !(data.sigmaMomentum > 0.0) ||
                       !(data.faceAreaSum > 0.0) || !(fastabs(rhs) > 0.0))
                    {
                        continue;
                    }

                    // The full RICH path solves this face-normal moment system.  The
                    // same system is retained here, with an explicit diagonal solve
                    // and a stable area-weighted fallback for degenerate geometry.
                    double const xx = data.fluxMatrix[0];
                    double const xy = data.fluxMatrix[1];
                    double const xz = data.fluxMatrix[2];
                    double const yy = data.fluxMatrix[3];
                    double const yz = data.fluxMatrix[4];
                    double const zz = data.fluxMatrix[5];
                    double const determinant = xx * (yy * zz - yz * yz)
                        - xy * (xy * zz - yz * xz)
                        + xz * (xy * yz - yy * xz);

                    PointT fluxDt{};
                    double const rhsX = rhs[0];
                    double const rhsY = rhs[1];
                    double const rhsZ = rhs[2];
                    double const scale = std::max({std::abs(xx), std::abs(yy),
                                                   std::abs(zz), 1.0});
                    if(std::isfinite(determinant) &&
                       std::abs(determinant) > 1.0e-12 * scale * scale * scale)
                    {
                        fluxDt = PointT(
                            ((yy * zz - yz * yz) * rhsX +
                             (xz * yz - xy * zz) * rhsY +
                             (xy * yz - xz * yy) * rhsZ) / determinant,
                            ((xz * yz - xy * zz) * rhsX +
                             (xx * zz - xz * xz) * rhsY +
                             (xy * xz - xx * yz) * rhsZ) / determinant,
                            ((xy * yz - xz * yy) * rhsX +
                             (xy * xz - xx * yz) * rhsY +
                             (xx * yy - xy * xy) * rhsZ) / determinant);
                    }
                    else
                    {
                        fluxDt = rhs / data.faceAreaSum;
                        ++owner_.ddmcMomentumMatrixFallbackCount_;
                    }

                    PointT const deltaP = data.sigmaMomentum *
                        owner_.componentGrid().GetVolume(i) * units::inv_clight * fluxDt;
                    if(!(std::isfinite(deltaP[0]) && std::isfinite(deltaP[1]) &&
                         std::isfinite(deltaP[2])))
                    {
                        continue;
                    }
                    owner_.extensives_[i].momentum += deltaP;
                    ++owner_.ddmcMomentumFeedbackCount_;
                }
            }
    }

    bool tryDDMCStep(
        MCParticle &particle, Functionality &functionality)
    {

            std::size_t cellIndex = particle.cellIndex;
            bool const packetInDDMC = particle.radiationState.isDDMC();
            bool convertedIncomingToComoving = false;
            bool useComovingFrame = false;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                useComovingFrame =
                    (owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                    (owner_.parameters_.postProcess.enabled &&
                     owner_.parameters_.postProcess.useCellVelocities);
            }

            auto finalizePolarization = [&](const PointT &finalVelocity)
            {
        #ifdef MONTECARLO_POLARIZATION
                if(owner_.polarizationEnabled())
                {
                    ParticleCounterEngine polarizationEngine(
                        particle.rngKey, particle.rngCounter);
                    std::uniform_real_distribution<double> polarizationUnit(0.0, 1.0);
                    polarization::finalizeAcceleratedPolarizationHistory<PointT>(
                        particle, finalVelocity,
                        owner_.parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
                        owner_.parameters_.postProcess.polarization.depolarizationScatterings,
                        polarizationEngine, polarizationUnit);
                }
        #else
                (void) finalVelocity;
        #endif
            };

            // A DDMC packet is represented in the cell-comoving frame and has no
            // usable microscopic ray while resident.  Every fallback must restore a
            // complete lab-frame IMC packet before returning false to the caller.
            auto exitDDMCToTransport = [&](bool sampleDirection)
            {
                if(!packetInDDMC && !convertedIncomingToComoving)
                {
                    return;
                }

                bool const wasResident = particle.radiationState.isResident();
                bool const wasComoving = particle.radiationState.isComoving();
                if(wasResident || packetInDDMC)
                {
                    particle.location = owner_.componentGrid().GetMeshPoint(cellIndex);
                    if(sampleDirection)
                    {
                        particle.velocity = owner_.sampleRandomVelocity(
                            owner_.cells_[cellIndex], particle);
                    }
        #ifdef MONTECARLO_POLARIZATION
                    finalizePolarization(particle.velocity);
        #endif
                }

                if((wasComoving || convertedIncomingToComoving) && useComovingFrame)
                {
                    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                    {
                        radiation_imc_detail::lorentzTransformToLab<PointT>(
                            particle, owner_.cells_[cellIndex]);
                    }
                }
                particle.radiationState.clearDDMC();
                particle.initialWeight = std::abs(particle.weight);
        #ifdef MONTECARLO_POLARIZATION
                if(owner_.polarizationEnabled())
                {
                    polarization::initializeIfNeeded<PointT>(particle);
                    particle.polarizationBasis = polarization::projectBasisToDirection(
                        particle.polarizationBasis, particle.velocity);
                }
        #endif
            };

            if(!std::isfinite(particle.weight) || particle.weight == 0.0)
            {
                particle.radiationState.clearDDMC();
                functionality.change = ParticleStatus::REMOVE;
                return true;
            }
            if(packetInDDMC && !owner_.parameters_.withDDMC)
            {
                exitDDMCToTransport(true);
                functionality.change = ParticleStatus::NO_CELL_MOVE;
                return true;
            }
            if(cellIndex >= owner_.ddmcCellData_.size())
            {
                exitDDMCToTransport(true);
                return false;
            }
            if(!particle.radiationState.invariantHolds())
            {
                exitDDMCToTransport(true);
                ++owner_.ddmcFallbackCount_;
                return false;
            }
            DDMCCellData const &data = owner_.ddmcCellData_[cellIndex];
            if(!data.eligible || data.totalLeakRate <= 0.0 || data.faceLeaks.empty())
            {
                exitDDMCToTransport(true);
                return false;
            }

            if(!particle.radiationState.isResident() &&
               particle.radiationState.bypassCellID !=
                   std::numeric_limits<std::size_t>::max())
            {
                std::size_t const exchangedCellID = cellIndex <
                    owner_.ddmcPointCellID_.size()
                    ? owner_.ddmcPointCellID_[cellIndex]
                    : std::numeric_limits<std::size_t>::max();
                std::size_t const currentCellID = exchangedCellID ==
                    std::numeric_limits<std::size_t>::max()
                    ? cellIndex : exchangedCellID;
                if(currentCellID == particle.radiationState.bypassCellID)
                {
                    ++owner_.ddmcFallbackCount_;
                    return false;
                }
            }

            double Ro = owner_.computeMinDistanceToFaces(cellIndex, particle.location);
            if(!particle.radiationState.isResident() &&
               Ro * data.sigmaParticleGate <
                   owner_.parameters_.ddmcMinParticleOpticalDepth)
            {
                ++owner_.ddmcFallbackCount_;
                return false;
            }

            if(owner_.parameters_.ddmcUseMultigroupPGRW && owner_.parameters_.withMultigroupOpacity)
            {
                if(data.groupCutoff == 0 || data.groupCutoff > NumGroups)
                {
                    exitDDMCToTransport(true);
                    ++owner_.ddmcFallbackCount_;
                    return false;
                }
                MCParticle frequencyProbe = particle;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(!particle.radiationState.isResident() && useComovingFrame)
                    {
                        radiation_imc_detail::lorentzTransformToComoving<PointT>(
                            frequencyProbe, owner_.cells_[cellIndex]);
                    }
                }
                double coFreq = frequencyProbe.frequency;
                owner_.clampFrequencyToBounds(coFreq);
                if(coFreq >= owner_.energyBoundaries_[data.groupCutoff])
                {
                    exitDDMCToTransport(true);
                    ++owner_.ddmcFallbackCount_;
                    return false;
                }
            }

            if(!particle.radiationState.isResident())
            {
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(useComovingFrame)
                    {
                        radiation_imc_detail::lorentzTransformToComoving<PointT>(
                            particle, owner_.cells_[cellIndex]);
                        owner_.clampFrequencyToBounds(particle.frequency);
                        convertedIncomingToComoving = true;
                    }
                }
            }

        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                polarization::initializeIfNeeded<PointT>(particle);
                particle.polarizationBasis = polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity);
            }
        #endif

            auto convertResidentToLab = [&]()
            {
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(useComovingFrame)
                    {
                        radiation_imc_detail::lorentzTransformToLab<PointT>(
                            particle, owner_.cells_[cellIndex]);
                        owner_.clampFrequencyToBounds(particle.frequency);
                        particle.initialWeight = std::abs(particle.weight);
                    }
                }
        #ifdef MONTECARLO_POLARIZATION
                if(owner_.polarizationEnabled())
                {
                    polarization::initializeIfNeeded<PointT>(particle);
                    particle.polarizationBasis = polarization::projectBasisToDirection(
                        particle.polarizationBasis, particle.velocity);
                }
        #endif
            };

            double f = owner_.factorFleck_[cellIndex];
            double upscatterRate = 0.0;
            if(owner_.parameters_.ddmcUseMultigroupPGRW && data.gamma < 1.0 &&
               data.sigmaEnergyAbs > 0.0 &&
               (f > 0.0 || owner_.postProcessExternalSourceMode_))
            {
                upscatterRate = units::clight * (1.0 - f) * data.sigmaEnergyAbs *
                    (1.0 - data.gamma);
            }
            double eventRate = data.totalLeakRate + upscatterRate;
            if(eventRate <= 0.0)
            {
                exitDDMCToTransport(true);
                ++owner_.ddmcFallbackCount_;
                return false;
            }

            double tEvent = -std::log(owner_.randomUnitOpen(particle)) / eventRate;
            double tCensus = particle.timeLeft;
            double tCutoff = std::numeric_limits<double>::max();
            if(owner_.parameters_.ddmcUseMultigroupPGRW &&
               data.groupCutoff > 0 && data.groupCutoff <= NumGroups &&
               data.velocityDivergence < 0.0)
            {
                double frequency = particle.frequency;
                owner_.clampFrequencyToBounds(frequency);
                double const cutoffFrequency =
                    owner_.energyBoundaries_[data.groupCutoff];
                double const growthRate = -data.velocityDivergence / 3.0;
                if(frequency > 0.0 && frequency < cutoffFrequency &&
                   growthRate > 0.0)
                {
                    tCutoff = std::log(cutoffFrequency / frequency) / growthRate;
                }
            }
            double dt = std::min({tEvent, tCensus, tCutoff});
            bool const censusEvent = tCensus <= tEvent && tCensus <= tCutoff;
            bool const cutoffEvent = tCutoff < tEvent && tCutoff < tCensus;

        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                double const fHistory = owner_.factorFleck_[cellIndex];
                double const scatteringOpacity =
                    owner_.scatteringOpacities_[cellIndex];
                double const explicitResetOpacity = upscatterRate / units::clight;
                ParticleCounterEngine polarizationEngine(
                    particle.rngKey, particle.rngCounter);
                std::uniform_real_distribution<double> polarizationUnit(0.0, 1.0);
                polarization::accumulateAcceleratedPolarizationHistory<PointT>(
                    particle, dt,
                    scatteringOpacity,
                    std::max(0.0, (1.0 - fHistory) * data.sigmaEnergyAbs -
                                      explicitResetOpacity),
                    polarizationEngine, polarizationUnit);
            }
        #endif

            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                   data.velocityDivergence != 0.0)
                {
                    double const logShift = -data.velocityDivergence * dt / 3.0;
                    if(std::isfinite(logShift) && logShift != 0.0)
                    {
                        double const boundedLogShift = std::clamp(logShift, -50.0, 50.0);
                        double const shift = std::exp(boundedLogShift);
                        particle.frequency *= shift;
                        particle.weight *= shift;
                        owner_.clampFrequencyToBounds(particle.frequency);
                    }
                }
            }

            double absRate = data.sigmaEnergyAbs * f * units::clight;
            double oldWeight = particle.weight;
            double expFactor = std::expm1(-dt * absRate);

            if(!owner_.parameters_.noHydroFeedback)
            {
                double const absorbedEnergy = -expFactor * oldWeight;
                owner_.tallyMaterialEnergy(cellIndex, absorbedEnergy);
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                             radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro &&
                       !owner_.parameters_.diffusionPressureGradient)
                    {
                        owner_.tallyMomentum(
                            cellIndex, absorbedEnergy * owner_.cells_[cellIndex].velocity *
                            units::inv_clight2);
                    }
                }
            }

            double integratedForTally = (absRate > 0.0)
                ? oldWeight * expFactor * (-1.0 / absRate)
                : oldWeight * dt;
            owner_.tallyRadiationEnergy(cellIndex, integratedForTally);

            if(owner_.parameters_.withEgTimeAvg && owner_.parameters_.withMultigroupOpacity)
            {
                if(owner_.parameters_.ddmcUseMultigroupPGRW && data.groupCutoff > 0 &&
                   data.groupCutoff <= NumGroups)
                {
                    double const kT = units::k_boltz *
                        owner_.cells_[cellIndex].temperature;
                    double const bandMass = ddmc::PlanckBandMass(
                        owner_.energyBoundaries_, kT, 0, data.groupCutoff);
                    if(bandMass > 0.0)
                    {
                        for(std::size_t g = 0; g < data.groupCutoff; ++g)
                        {
                            double const groupMass = ddmc::PlanckBandMass(
                                owner_.energyBoundaries_, kT, g, g + 1);
                            owner_.tallyGroupRadiationEnergy(
                                cellIndex, g,
                                integratedForTally * groupMass / bandMass);
                        }
                    }
                }
                else
                {
                    std::size_t g = owner_.opacity_->findGroup(
                        particle.frequency, owner_.energyBoundaries_);
                    if(g < NumGroups)
                    {
                        owner_.tallyGroupRadiationEnergy(
                            cellIndex, g, integratedForTally);
                    }
                }
            }

            particle.weight *= 1.0 + expFactor;
            particle.timeLeft -= dt;

            if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
            {
                particle.radiationState.clearDDMC();
                functionality.change = ParticleStatus::REMOVE;
                if(!owner_.parameters_.noHydroFeedback)
                {
                    owner_.tallyMaterialEnergy(cellIndex, particle.weight);
                    if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                                 radiation_imc_detail::has_member_velocity<CellT>::value)
                    {
                        if(owner_.parameters_.withHydro &&
                           !owner_.parameters_.diffusionPressureGradient)
                        {
                            owner_.tallyMomentum(
                                cellIndex, particle.weight * owner_.cells_[cellIndex].velocity *
                                units::inv_clight2);
                        }
                    }
                }
                ++owner_.ddmcStepCount_;
                return true;
            }

            ++owner_.ddmcStepCount_;

            if(cutoffEvent)
            {
                particle.frequency = std::nextafter(
                    owner_.energyBoundaries_[data.groupCutoff],
                    std::numeric_limits<double>::max());
                particle.velocity = owner_.sampleRandomVelocity(
                    owner_.cells_[cellIndex], particle);
                finalizePolarization(particle.velocity);
                convertResidentToLab();
                particle.radiationState.clearDDMC();
                functionality.change = ParticleStatus::NO_CELL_MOVE;
                return true;
            }

            if(!particle.radiationState.isResident())
            {
                double const entrySpeed = fastabs(particle.velocity);
                if(entrySpeed > 0.0 && std::isfinite(entrySpeed))
                {
                    owner_.addDDMCFluxContribution(
                        cellIndex, particle.weight * (particle.velocity / entrySpeed));
                }
                particle.radiationState.set(RadiationTransportState<PointT>::DDMCMode);
                particle.radiationState.set(
                    RadiationTransportState<PointT>::DDMCCellResident);
                particle.radiationState.set(
                    RadiationTransportState<PointT>::DDMCComovingFrame);
                // A DDMC resident has no microscopic location/direction.  Keeping a
                // controlled representative position avoids asking the transport
                // geometry to interpret a stale IMC ray on the next event.
                particle.location = owner_.componentGrid().GetMeshPoint(cellIndex);
                particle.velocity = owner_.sampleRandomVelocity(
                    owner_.cells_[cellIndex], particle);
            }

            if(censusEvent)
            {
                // Census is a representation boundary.  Reconstruct a valid IMC
                // packet before returning it to the manager; the next time step must
                // not carry a stale DDMC direction or frame.
                particle.location = owner_.componentGrid().GetMeshPoint(cellIndex);
                if(owner_.parameters_.withMultigroupOpacity)
                {
                    bool sampledResidentBand = false;
                    if(owner_.parameters_.ddmcUseMultigroupPGRW &&
                       data.groupCutoff > 0 && data.groupCutoff <= NumGroups)
                    {
                        double const kT = units::k_boltz *
                            owner_.cells_[cellIndex].temperature;
                        double const bandMass = ddmc::PlanckBandMass(
                            owner_.energyBoundaries_, kT, 0, data.groupCutoff);
                        if(!(bandMass > 0.0))
                        {
                            StormError eo("RadiationIMC DDMC census has no resident-band Planck mass");
                            eo.addEntry("Cell index", cellIndex);
                            eo.addEntry("Group cutoff", data.groupCutoff);
                            throw eo;
                        }
                        double remaining = owner_.randomUnitOpen(particle) * bandMass;
                        for(std::size_t group = 0; group < data.groupCutoff; ++group)
                        {
                            double const groupMass = ddmc::PlanckBandMass(
                                owner_.energyBoundaries_, kT, group, group + 1);
                            if(remaining <= groupMass || group + 1 == data.groupCutoff)
                            {
                                double const localRandom = groupMass > 0.0
                                    ? std::clamp(remaining / groupMass, 0.0, 1.0)
                                    : owner_.randomUnitOpen(particle);
                                particle.frequency = owner_.opacity_->SampleThermalEnergyInGroup(
                                    owner_.cells_[cellIndex], group, localRandom,
                                    owner_.energyBoundaries_);
                                double const upperBand =
                                    owner_.energyBoundaries_[data.groupCutoff];
                                particle.frequency = std::min(
                                    particle.frequency,
                                    std::nextafter(upperBand,
                                        owner_.energyBoundaries_[0]));
                                sampledResidentBand = true;
                                break;
                            }
                            remaining -= groupMass;
                        }
                    }
                    if(!sampledResidentBand)
                    {
                        particle.frequency = owner_.opacity_->GetThermalEnergy(
                            owner_.cells_[cellIndex], owner_.randomUnitOpen(particle),
                            owner_.energyBoundaries_);
                    }
                    owner_.clampFrequencyToBounds(particle.frequency);
                }
                particle.velocity = owner_.sampleRandomVelocity(
                    owner_.cells_[cellIndex], particle);
                finalizePolarization(particle.velocity);
                convertResidentToLab();
                particle.radiationState.clearDDMC();
                functionality.change = ParticleStatus::DONE;
                ++owner_.ddmcCensusCount_;
                return true;
            }

            double eventPick = owner_.randomUnitOpen(particle) * eventRate;
            if(eventPick <= data.totalLeakRate)
            {
                double facePick = owner_.randomUnitOpen(particle) * data.totalLeakRate;
                const DDMCFaceLeak *chosen = nullptr;
                for(const DDMCFaceLeak &fl : data.faceLeaks)
                {
                    facePick -= fl.rate;
                    if(facePick <= 0.0)
                    {
                        chosen = &fl;
                        break;
                    }
                }
                if(!chosen && !data.faceLeaks.empty())
                {
                    chosen = &data.faceLeaks.back();
                }
                if(!chosen)
                {
                    StormError eo("RadiationIMC DDMC could not resolve its selected leakage face");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Total leak rate", data.totalLeakRate);
                    eo.addEntry("Face count", data.faceLeaks.size());
                    eo.addEntry("Event rate", eventRate);
                    throw eo;
                }

                PointT leakFaceCenter = owner_.componentGrid().FaceCM(chosen->faceIndex);
                PointT nOut = chosen->outwardNormal;
                double nOutMag = fastabs(nOut);
                if(!(nOutMag > 0.0) || !std::isfinite(nOutMag))
                {
                    StormError eo("RadiationIMC DDMC selected a face with an invalid outward normal");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Face index", chosen->faceIndex);
                    eo.addEntry("Normal magnitude", nOutMag);
                    eo.addEntry("Total leak rate", data.totalLeakRate);
                    throw eo;
                }
                nOut = nOut / nOutMag;

                if(chosen->kind == ddmc::FaceKind::ThermalizingBoundary)
                {
                    auto const sourceFace =
                        owner_.postProcessExternalSourceFaceIndex_.find(
                            chosen->faceIndex);
                    if(sourceFace ==
                           owner_.postProcessExternalSourceFaceIndex_.end() ||
                       sourceFace->second >=
                           owner_.postProcessExternalSources_.size())
                    {
                        throw StormError(
                            "DDMC selected an external-source face without an installed source");
                    }
                    PostProcessExternalSource const &source =
                        owner_.postProcessExternalSources_[sourceFace->second];
                    if(radiation_imc_detail::ddmcStableCellID(
                           owner_.componentGrid(), cellIndex, owner_.cells_[cellIndex]) !=
                       source.cellID)
                    {
                        throw StormError(
                            "DDMC selected an external-source face from the wrong transport cell");
                    }

                    nOut = source.outwardNormal /
                        std::max(fastabs(source.outwardNormal),
                                 std::numeric_limits<double>::min());
                    double const eventEnergy = std::abs(particle.weight);
                    ++owner_.ddmcExternalSourceThermalizationCount_;
                    owner_.ddmcExternalSourceThermalizedEnergy_ += eventEnergy;
                    if(owner_.parameters_.withMultigroupOpacity)
                    {
                        particle.frequency =
                            owner_.samplePostProcessExternalSourcePlanckFrequency(
                                owner_.cells_[cellIndex]);
                        owner_.clampFrequencyToBounds(particle.frequency);
                    }

                    bool const leaveDDMCBand =
                        owner_.parameters_.ddmcUseMultigroupPGRW &&
                        data.groupCutoff < NumGroups &&
                        particle.frequency >=
                            owner_.energyBoundaries_[data.groupCutoff];
                    if(leaveDDMCBand)
                    {
                        particle.velocity = units::clight *
                            owner_.samplePostProcessExternalSourceDirection(nOut, particle);
        #ifdef MONTECARLO_POLARIZATION
                        if(owner_.polarizationEnabled())
                        {
                            particle.stokesQ = 0.0;
                            particle.stokesU = 0.0;
                            particle.polarizationInitialized = false;
                        }
        #endif
                        if constexpr(
                            radiation_imc_detail::has_member_velocity<CellT>::value)
                        {
                            if(useComovingFrame)
                            {
                                radiation_imc_detail::lorentzTransformToLab<PointT>(
                                    particle, owner_.cells_[cellIndex]);
                                owner_.clampFrequencyToBounds(particle.frequency);
                            }
                        }
                        static constexpr double nudge = 1.0e-8;
                        particle.location = (1.0 - nudge) * source.location +
                            nudge * owner_.componentGrid().GetMeshPoint(cellIndex);
                        particle.radiationState.clearDDMC();
                        particle.initialWeight = std::abs(particle.weight);
                        functionality.change = ParticleStatus::NO_CELL_MOVE;
                        ++owner_.ddmcExternalSourceToIMCCount_;
                        owner_.ddmcExternalSourceToIMCEnergy_ += eventEnergy;
                        ++owner_.ddmcLeakCount_;
                        ++owner_.ddmcTransportLeakCount_;
                        return true;
                    }

                    particle.location = owner_.componentGrid().GetMeshPoint(cellIndex);
                    particle.velocity = owner_.sampleRandomVelocity(
                        owner_.cells_[cellIndex], particle);
        #ifdef MONTECARLO_POLARIZATION
                    if(owner_.polarizationEnabled())
                    {
                        particle.stokesQ = 0.0;
                        particle.stokesU = 0.0;
                        particle.polarizationInitialized = false;
                    }
        #endif
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCMode);
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCCellResident);
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCComovingFrame);
                    functionality.change = ParticleStatus::NO_CELL_MOVE;
                    ++owner_.ddmcExternalSourceStayDDMCCount_;
                    ++owner_.ddmcLeakCount_;
                    ++owner_.ddmcResidentLeakCount_;
                    return true;
                }

                constexpr double DDMC_PI = 3.14159265358979323846;
                bool const useDDMCChannel =
                    chosen->ddmcRate > 0.0 &&
                    owner_.randomUnitOpen(particle) < chosen->ddmcRate / chosen->rate;
                double mu = useDDMCChannel
                    ? ddmc::SampleAsymptoticMu(owner_.randomUnitOpen(particle))
                    : std::sqrt(owner_.randomUnitOpen(particle));
                double phiLeak = 2.0 * DDMC_PI * owner_.randomUnitOpen(particle);
                double sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

                PointT helper = (std::abs(ScalarProd(
                    nOut, PointT(1.0, 0.0, 0.0))) < 0.9)
                    ? PointT(1.0, 0.0, 0.0)
                    : PointT(0.0, 1.0, 0.0);
                PointT e1 = helper - ScalarProd(helper, nOut) * nOut;
                double e1Mag = fastabs(e1);
                if(e1Mag > 0.0)
                {
                    e1 = e1 / e1Mag;
                }
                PointT e2 = CrossProduct(nOut, e1);
                double e2Mag = fastabs(e2);
                if(e2Mag > 0.0)
                {
                    e2 = e2 / e2Mag;
                }

                PointT dir = mu * nOut + sinTheta * std::cos(phiLeak) * e1 + sinTheta * std::sin(phiLeak) * e2;
                double dirMag = fastabs(dir);
                if(dirMag > 0.0)
                {
                    dir = dir / dirMag;
                }

                particle.location = leakFaceCenter;
                particle.velocity = dir * units::clight;

                bool const targetDDMC = useDDMCChannel && chosen->targetDDMCEligible;
                if(!targetDDMC && owner_.parameters_.ddmcUseMultigroupPGRW &&
                   owner_.parameters_.withMultigroupOpacity)
                {
                    std::size_t beginGroup = 0;
                    if(chosen->targetDDMCEligible &&
                       chosen->targetGroupCutoff > 0 &&
                       chosen->targetGroupCutoff < data.groupCutoff)
                    {
                        beginGroup = chosen->targetGroupCutoff;
                    }
                    double const kT = units::k_boltz *
                        owner_.cells_[cellIndex].temperature;
                    double const bandMass = ddmc::PlanckBandMass(
                        owner_.energyBoundaries_, kT, beginGroup, data.groupCutoff);
                    if(bandMass > 0.0)
                    {
                        double remaining = owner_.randomUnitOpen(particle) * bandMass;
                        for(std::size_t group = beginGroup;
                            group < data.groupCutoff; ++group)
                        {
                            double const groupMass = ddmc::PlanckBandMass(
                                owner_.energyBoundaries_, kT, group, group + 1);
                            if(remaining <= groupMass || group + 1 == data.groupCutoff)
                            {
                                double const localRandom = groupMass > 0.0
                                    ? std::clamp(remaining / groupMass, 0.0, 1.0)
                                    : owner_.randomUnitOpen(particle);
                                particle.frequency = owner_.opacity_->SampleThermalEnergyInGroup(
                                    owner_.cells_[cellIndex], group, localRandom,
                                    owner_.energyBoundaries_);
                                break;
                            }
                            remaining -= groupMass;
                        }
                    }
                }

                if(!targetDDMC)
                {
                    finalizePolarization(particle.velocity);
                }

                double const fluxWeightComoving = particle.weight;
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                       !targetDDMC)
                    {
                        CellT sourceCell = owner_.cells_[cellIndex];
                        sourceCell.velocity = owner_.cells_[cellIndex].velocity;
                        MCParticle transportParticle = particle;
                        radiation_imc_detail::lorentzTransformToLab<PointT>(
                            transportParticle, sourceCell);
                        particle.velocity = transportParticle.velocity;
                        particle.frequency = transportParticle.frequency;
                        particle.weight = transportParticle.weight;
                        particle.initialWeight = std::abs(particle.weight);
                    }
                }

                std::size_t const diagnosticGroup = targetDDMC ||
                    !owner_.parameters_.withMultigroupOpacity
                    ? DDMC_DIAGNOSTIC_GREY_GROUP
                    : owner_.opacity_->findGroup(
                        particle.frequency, owner_.energyBoundaries_);
                owner_.recordDDMCDiagnosticEvent(
                    targetDDMC ? DDMCDiagnosticEventKind::DDMCToDDMC
                               : DDMCDiagnosticEventKind::DDMCToIMC,
                    cellIndex, chosen->nextCellIndex, chosen->faceIndex,
                    diagnosticGroup, fluxWeightComoving, data.groupCutoff,
                    chosen->targetGroupCutoff,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN());

                PointT const fluxContribution = fluxWeightComoving * dir;
                owner_.addDDMCFluxContribution(cellIndex, fluxContribution);
                if(targetDDMC)
                {
                    if(chosen->nextCellIndex < owner_.componentGrid().GetPointNo())
                    {
                        owner_.addDDMCFluxContribution(
                            chosen->nextCellIndex, fluxContribution);
                    }
                    else
                    {
                        particle.radiationState.pendingFlux = fluxContribution;
                        particle.radiationState.set(
                            RadiationTransportState<PointT>::PendingFlux);
                    }
                }

                if(targetDDMC)
                {
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCMode);
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCCellResident);
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::DDMCComovingFrame);
                }
                else
                {
                    particle.radiationState.clearDDMC();
                }

                functionality.change = ParticleStatus::CELL_MOVE;
                functionality.nextCellIndex = chosen->nextCellIndex;
                if(targetDDMC)
                {
                    ++owner_.ddmcResidentLeakCount_;
                    if(chosen->nextCellIndex >= owner_.componentGrid().GetPointNo())
                    {
                        ++owner_.ddmcRemoteResidentLeakCount_;
                    }
                }
                else
                {
                    ++owner_.ddmcTransportLeakCount_;
                }
                ++owner_.ddmcLeakCount_;
            }
            else
            {
                if(!owner_.parameters_.ddmcUseMultigroupPGRW)
                {
                    finalizePolarization(particle.velocity);
                    particle.radiationState.clearDDMC();
                    functionality.change = ParticleStatus::DONE;
                    ++owner_.ddmcCensusCount_;
                    return true;
                }
                CellT &cell = owner_.cells_[cellIndex];
                double const kT = units::k_boltz * cell.temperature;
                double const upperBandMass = ddmc::PlanckBandMass(
                    owner_.energyBoundaries_, kT, data.groupCutoff, NumGroups);
                if(!(upperBandMass > 0.0))
                {
                    StormError eo("RadiationIMC DDMC upscatter has no representable upper frequency band");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Group cutoff", data.groupCutoff);
                    eo.addEntry("Upper-band Planck mass", upperBandMass);
                    throw eo;
                }
                double remaining = owner_.randomUnitOpen(particle) * upperBandMass;
                std::size_t selectedGroup = data.groupCutoff;
                for(std::size_t group = data.groupCutoff; group < NumGroups; ++group)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        owner_.energyBoundaries_, kT, group, group + 1);
                    if(remaining <= groupMass || group + 1 == NumGroups)
                    {
                        selectedGroup = group;
                        double const localRandom = groupMass > 0.0
                            ? std::clamp(remaining / groupMass, 0.0, 1.0)
                            : owner_.randomUnitOpen(particle);
                        particle.frequency = owner_.opacity_->SampleThermalEnergyInGroup(
                            cell, selectedGroup, localRandom, owner_.energyBoundaries_);
                        break;
                    }
                    remaining -= groupMass;
                }
                owner_.clampFrequencyToBounds(particle.frequency);
                particle.velocity = owner_.sampleRandomVelocity(cell, particle);
                exitDDMCToTransport(false);
                functionality.change = ParticleStatus::NO_CELL_MOVE;
                ++owner_.ddmcUpscatterCount_;
                return true;
            }
            return true;
    }

    void recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind kind,
        std::size_t sourceCellIndex,
        std::size_t targetCellIndex,
        std::size_t faceIndex,
        std::size_t group,
        double energy,
        std::size_t sourceGroupCutoff,
        std::size_t targetGroupCutoff,
        double mu,
        double admissionProbability)
    {

            if(!owner_.parameters_.withDDMC ||
               !owner_.parameters_.ddmcInterfaceDiagnostics)
            {
                return;
            }

            auto pointID = [this](std::size_t index)
            {
                if(index < owner_.ddmcPointCellID_.size() &&
                   owner_.ddmcPointCellID_[index] !=
                       std::numeric_limits<std::size_t>::max())
                {
                    return owner_.ddmcPointCellID_[index];
                }
                if(index < owner_.cells_.size())
                {
                    return radiation_imc_detail::ddmcStableCellID(
                        owner_.componentGrid(), index, owner_.cells_[index]);
                }
                return std::numeric_limits<std::size_t>::max();
            };
            auto pointX = [this](std::size_t index)
            {
                if(index < owner_.componentGrid().getMeshPoints().size())
                {
                    return static_cast<double>(owner_.componentGrid().GetMeshPoint(index)[0]);
                }
                return std::numeric_limits<double>::quiet_NaN();
            };

            std::size_t const sourceCellID = pointID(sourceCellIndex);
            std::size_t const targetCellID = pointID(targetCellIndex);
            DDMCDiagnosticEventKey const key{
                kind, faceIndex, sourceCellID, targetCellID, group};
            auto inserted = owner_.ddmcDiagnosticEvents_.emplace(
                key, DDMCDiagnosticEventAccumulator{});
            DDMCDiagnosticEventAccumulator &entry = inserted.first->second;
            if(inserted.second)
            {
                entry.faceIndex = faceIndex;
                entry.sourceCellID = sourceCellID;
                entry.targetCellID = targetCellID;
                entry.group = group;
                entry.sourceGroupCutoff = sourceGroupCutoff;
                entry.targetGroupCutoff = targetGroupCutoff;
                entry.faceX = owner_.componentGrid().FaceCM(faceIndex)[0];
                entry.sourceGeneratorX = pointX(sourceCellIndex);
                entry.targetGeneratorX = pointX(targetCellIndex);
            }

            ++entry.count;
            if(std::isfinite(energy))
            {
                entry.signedEnergy += energy;
                entry.absoluteEnergy += std::abs(energy);
            }
            if(std::isfinite(mu))
            {
                entry.muSum += mu;
                ++entry.muCount;
            }
            if(std::isfinite(admissionProbability))
            {
                entry.admissionProbabilitySum += admissionProbability;
                ++entry.admissionProbabilityCount;
            }
    }

    bool tryIMCToDDMCInterface(
        MCParticle &particle,
        Functionality &functionality,
        std::vector<MCParticle> &particlesToAdd,
        std::size_t sourceCellIndex,
        std::size_t targetCellIndex,
        std::size_t faceIndex)
    {

            if(sourceCellIndex >= owner_.cells_.size() ||
               targetCellIndex >= owner_.ddmcPointEligible_.size() ||
               owner_.componentGrid().IsPointOutsideBox(targetCellIndex) ||
               owner_.ddmcPointEligible_[targetCellIndex] == 0)
            {
                return false;
            }

            std::size_t const exchangedTargetID = owner_.ddmcPointCellID_[targetCellIndex];
            // A cell type without an explicit stable ID still needs a serial-safe
            // bypass key.  Distributed grids should provide the exchanged owner ID;
            // the point index is only a fallback for the local-only case.
            std::size_t const targetID = exchangedTargetID ==
                std::numeric_limits<std::size_t>::max()
                ? targetCellIndex : exchangedTargetID;
            if(particle.radiationState.bypassCellID == targetID)
            {
                return false;
            }

            PointT normal = owner_.componentGrid().Normal(faceIndex);
            double const normalMagnitude = fastabs(normal);
            if(!(normalMagnitude > 0.0) || !std::isfinite(normalMagnitude))
            {
                return false;
            }
            normal = normal / normalMagnitude;

            PointT const sourceCenter = owner_.componentGrid().GetMeshPoint(sourceCellIndex);
            PointT const targetCenter = owner_.componentGrid().GetMeshPoint(targetCellIndex);
            if(ScalarProd(normal, targetCenter - sourceCenter) < 0.0)
            {
                normal = -normal;
            }

            PointT faceVelocity{};
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                PointT targetVelocity = owner_.ddmcPointVelocity_[targetCellIndex];
                double const sourceDistance = std::abs(ScalarProd(
                    owner_.componentGrid().FaceCM(faceIndex) - sourceCenter, normal));
                double const targetDistance = std::abs(ScalarProd(
                    targetCenter - owner_.componentGrid().FaceCM(faceIndex), normal));
                double const distanceSum = sourceDistance + targetDistance;
                faceVelocity = distanceSum > 0.0
                    ? (targetDistance * owner_.cells_[sourceCellIndex].velocity +
                       sourceDistance * targetVelocity) / distanceSum
                    : 0.5 * (owner_.cells_[sourceCellIndex].velocity + targetVelocity);
            }

            MCParticle faceComoving = particle;
            MCParticle targetComoving = particle;
        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                polarization::initializeIfNeeded<PointT>(faceComoving);
            }
        #endif
            bool useVelocityFrames = false;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                   (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                {
                    CellT faceCell = owner_.cells_[sourceCellIndex];
                    CellT targetCell = faceCell;
                    faceCell.velocity = faceVelocity;
                    targetCell.velocity = owner_.ddmcPointVelocity_[targetCellIndex];
                    radiation_imc_detail::lorentzTransformToComoving<PointT>(
                        faceComoving, faceCell);
        #ifdef MONTECARLO_POLARIZATION
                    if(owner_.polarizationEnabled())
                    {
                        faceComoving.polarizationBasis = polarization::projectBasisToDirection(
                            faceComoving.polarizationBasis, faceComoving.velocity);
                    }
        #endif
                    targetComoving = faceComoving;
                    radiation_imc_detail::lorentzTransformToLab<PointT>(
                        targetComoving, faceCell);
                    radiation_imc_detail::lorentzTransformToComoving<PointT>(
                        targetComoving, targetCell);
                    useVelocityFrames = true;
                }
            }

            std::size_t const sourceGroupCutoff = sourceCellIndex <
                owner_.ddmcPointGroupCutoff_.size()
                ? owner_.ddmcPointGroupCutoff_[sourceCellIndex] : 0;
            std::size_t const targetGroupCutoff = targetCellIndex <
                owner_.ddmcPointGroupCutoff_.size()
                ? owner_.ddmcPointGroupCutoff_[targetCellIndex] : 0;
            std::size_t const diagnosticGroup = owner_.parameters_.withMultigroupOpacity
                ? owner_.opacity_->findGroup(
                    targetComoving.frequency, owner_.energyBoundaries_)
                : DDMC_DIAGNOSTIC_GREY_GROUP;
            owner_.recordDDMCDiagnosticEvent(
                DDMCDiagnosticEventKind::IMCCandidate, sourceCellIndex,
                targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
                sourceGroupCutoff, targetGroupCutoff,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());

            if(owner_.parameters_.ddmcUseMultigroupPGRW &&
               owner_.parameters_.withMultigroupOpacity)
            {
                std::size_t const cutoff = owner_.ddmcPointGroupCutoff_[targetCellIndex];
                double frequency = targetComoving.frequency;
                owner_.clampFrequencyToBounds(frequency);
                if(cutoff == 0 || cutoff > NumGroups ||
                   frequency >= owner_.energyBoundaries_[cutoff])
                {
                    owner_.recordDDMCDiagnosticEvent(
                        DDMCDiagnosticEventKind::IMCFrequencyReject,
                        sourceCellIndex, targetCellIndex, faceIndex,
                        diagnosticGroup, faceComoving.weight, sourceGroupCutoff,
                        targetGroupCutoff,
                        std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
                    return false;
                }
            }

            double const speed = fastabs(faceComoving.velocity);
            if(!(speed > 0.0) || !std::isfinite(speed))
            {
                return false;
            }
            double const mu = ScalarProd(faceComoving.velocity / speed, normal);
            if(!(mu > 0.0) || !std::isfinite(mu))
            {
                return false;
            }
            ++owner_.ddmcInterfaceIncidentCount_;
            owner_.ddmcInterfaceMinimumMu_ = std::min(
                owner_.ddmcInterfaceMinimumMu_, mu);

            double movingFactor = 1.0;
            auto bypassMovingInterface = [&]()
            {
                owner_.recordDDMCDiagnosticEvent(
                    DDMCDiagnosticEventKind::IMCBypass, sourceCellIndex,
                    targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
                    sourceGroupCutoff, targetGroupCutoff, mu,
                    std::numeric_limits<double>::quiet_NaN());
                ++owner_.ddmcMovingInterfaceBypassCount_;
                ++owner_.ddmcInterfaceBypassCount_;
                functionality.change = ParticleStatus::CELL_MOVE;
                functionality.nextCellIndex = targetCellIndex;
                return true;
            };
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(useVelocityFrames &&
                   owner_.parameters_.ddmcUseMovingInterfaceCorrection)
                {
                    double const betaNormal = -ScalarProd(faceVelocity, normal) *
                        units::inv_clight;
                    if(!std::isfinite(betaNormal) ||
                       std::abs(betaNormal) >
                           owner_.parameters_.ddmcMaxInterfaceVelocityOverC)
                    {
                        ++owner_.ddmcInterfaceGuFallbackCount_;
                        particle.radiationState.bypassCellID = targetID;
                        return bypassMovingInterface();
                    }
                    movingFactor = ddmc::MovingFactor(mu, betaNormal);
                    if(std::isfinite(movingFactor))
                    {
                        owner_.ddmcMovingInterfaceMaxFactor_ = std::max(owner_.ddmcMovingInterfaceMaxFactor_, movingFactor);
                    }
                    if(!(movingFactor > 0.0) ||
                       !std::isfinite(movingFactor) ||
                       movingFactor > owner_.parameters_.ddmcMaxMovingInterfaceWeightCorrection)
                    {
                        ++owner_.ddmcInterfaceGuFallbackCount_;
                        particle.radiationState.bypassCellID = targetID;
                        return bypassMovingInterface();
                    }
                    ++owner_.ddmcInterfaceGuAppliedCount_;
                }
            }

            double const targetWeight = owner_.parameters_.ddmcInterfaceTargetWeightRatio * std::max(std::abs(particle.weight), std::numeric_limits<double>::min());
            std::size_t requiredSplitCount = 1;
            if(targetWeight > 0.0)
            {
                requiredSplitCount = static_cast<std::size_t>(std::ceil(std::abs(faceComoving.weight * movingFactor) / targetWeight));
                requiredSplitCount = std::max<std::size_t>(1, requiredSplitCount);
            }
            if(requiredSplitCount > std::max<std::size_t>(1, owner_.parameters_.ddmcMaxInterfaceSplits))
            {
                particle.radiationState.bypassCellID = targetID;
                return bypassMovingInterface();
            }

            double const targetOpacity = owner_.ddmcPointSigmaDiffusion_[targetCellIndex];
            double const targetDistance = std::abs(ScalarProd(targetCenter - owner_.componentGrid().FaceCM(faceIndex), normal));
            double const admission = ddmc::StaticAdmissionProbability(mu, targetOpacity, targetDistance);
            owner_.recordDDMCDiagnosticEvent(
                DDMCDiagnosticEventKind::IMCIncident, sourceCellIndex,
                targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
                sourceGroupCutoff, targetGroupCutoff, mu, admission);

            if(owner_.randomUnitOpen(particle) > admission)
            {
                ++owner_.ddmcInterfaceReflectedCount_;
                // Diffuse-albedo rejection stays in the source IMC cell.  The
                // incoming direction is not reflected specularly at a transport-
                // diffusion interface.
                constexpr double pi = 3.14159265358979323846;
                double const reflectedMu = std::sqrt(owner_.randomUnitOpen(particle));
                double const sinTheta = std::sqrt(
                    std::max(0.0, 1.0 - reflectedMu * reflectedMu));
                double const phi = 2.0 * pi * owner_.randomUnitOpen(particle);
                PointT helper = std::abs(ScalarProd(normal, PointT(1.0, 0.0, 0.0))) < 0.9
                    ? PointT(1.0, 0.0, 0.0) : PointT(0.0, 1.0, 0.0);
                PointT e1 = helper - ScalarProd(helper, normal) * normal;
                e1 = e1 / std::max(fastabs(e1), std::numeric_limits<double>::min());
                PointT e2 = CrossProduct(normal, e1);
                e2 = e2 / std::max(fastabs(e2), std::numeric_limits<double>::min());
                faceComoving.velocity = units::clight *
                    (-reflectedMu * normal +
                     sinTheta * std::cos(phi) * e1 +
                     sinTheta * std::sin(phi) * e2);
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(useVelocityFrames)
                    {
                        CellT faceCell = owner_.cells_[sourceCellIndex];
                        faceCell.velocity = faceVelocity;
                        radiation_imc_detail::lorentzTransformToLab<PointT>(faceComoving, faceCell);
                    }
                }
                particle.velocity = faceComoving.velocity;
                particle.frequency = faceComoving.frequency;
                particle.weight = faceComoving.weight;
        #ifdef MONTECARLO_POLARIZATION
                if(owner_.polarizationEnabled())
                {
                    particle.stokesQ = faceComoving.stokesQ;
                    particle.stokesU = faceComoving.stokesU;
                    particle.polarizationBasis = polarization::projectBasisToDirection(
                        faceComoving.polarizationBasis, particle.velocity);
                    particle.polarizationInitialized = faceComoving.polarizationInitialized;
                }
        #endif
                particle.location = (1.0 - 1.0e-10) * owner_.componentGrid().FaceCM(faceIndex) +
                    1.0e-10 * sourceCenter;
                functionality.change = ParticleStatus::NO_CELL_MOVE;
                owner_.recordDDMCDiagnosticEvent(
                    DDMCDiagnosticEventKind::IMCReflected, sourceCellIndex,
                    targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
                    sourceGroupCutoff, targetGroupCutoff, mu, admission);
                return true;
            }

            faceComoving.weight *= movingFactor;
            targetComoving = faceComoving;
            ++owner_.ddmcInterfaceAdmittedCount_;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(useVelocityFrames)
                {
                    CellT faceCell = owner_.cells_[sourceCellIndex];
                    CellT targetCell = faceCell;
                    faceCell.velocity = faceVelocity;
                    targetCell.velocity = owner_.ddmcPointVelocity_[targetCellIndex];
                    targetComoving = faceComoving;
                    radiation_imc_detail::lorentzTransformToLab<PointT>(
                        targetComoving, faceCell);
                    radiation_imc_detail::lorentzTransformToComoving<PointT>(
                        targetComoving, targetCell);
        #ifdef MONTECARLO_POLARIZATION
                    if(owner_.polarizationEnabled())
                    {
                        targetComoving.polarizationBasis = polarization::projectBasisToDirection(
                            targetComoving.polarizationBasis, targetComoving.velocity);
                    }
        #endif
                }
            }
            double const admittedTargetWeight = targetComoving.weight;
            std::size_t splitCount = requiredSplitCount;
            // Additional packets are inserted directly into a local target.  A
            // remote target keeps one unbiased corrected-weight packet.
            if(targetCellIndex >= owner_.componentGrid().GetPointNo())
            {
                splitCount = 1;
            }
            targetComoving.weight /= static_cast<double>(splitCount);
            particle.weight = targetComoving.weight;
            particle.frequency = targetComoving.frequency;
            particle.initialWeight = std::abs(particle.weight);
        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                particle.stokesQ = targetComoving.stokesQ;
                particle.stokesU = targetComoving.stokesU;
                particle.polarizationBasis = targetComoving.polarizationBasis;
                particle.polarizationInitialized = targetComoving.polarizationInitialized;
            }
        #endif
            particle.radiationState.set(RadiationTransportState<PointT>::DDMCMode);
            particle.radiationState.set(
                RadiationTransportState<PointT>::DDMCCellResident);
            particle.radiationState.set(
                RadiationTransportState<PointT>::DDMCComovingFrame);
            particle.radiationState.clearPendingFlux();
            double const admittedSpeed = fastabs(targetComoving.velocity);
            if(admittedSpeed > 0.0 && std::isfinite(admittedSpeed))
            {
                PointT const contribution = admittedTargetWeight *
                    (targetComoving.velocity / admittedSpeed);
                if(targetCellIndex < owner_.componentGrid().GetPointNo())
                {
                    owner_.addDDMCFluxContribution(targetCellIndex, contribution);
                    ++owner_.ddmcInterfaceFluxTallyCount_;
                }
                else
                {
                    particle.radiationState.pendingFlux = contribution;
                    particle.radiationState.set(
                        RadiationTransportState<PointT>::PendingFlux);
                }
            }
            particle.location = owner_.componentGrid().FaceCM(faceIndex);
            particle.velocity = owner_.sampleRandomVelocity(
                owner_.cells_[sourceCellIndex], particle);
        #ifdef MONTECARLO_POLARIZATION
            if(owner_.polarizationEnabled())
            {
                particle.polarizationBasis = polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity);
            }
        #endif
            functionality.change = ParticleStatus::CELL_MOVE;
            functionality.nextCellIndex = targetCellIndex;
            for(std::size_t copy = 1; copy < splitCount; ++copy)
            {
                MCParticle extra = particle;
                extra.id = std::numeric_limits<std::size_t>::max();
                extra.cellID = targetID;
                extra.cellIndex = targetCellIndex;
                extra.location = targetCenter;
                particlesToAdd.push_back(std::move(extra));
                ++owner_.ddmcInterfaceSplitPacketCount_;
            }
            owner_.recordDDMCDiagnosticEvent(
                DDMCDiagnosticEventKind::IMCAdmitted, sourceCellIndex,
                targetCellIndex, faceIndex, diagnosticGroup, admittedTargetWeight,
                sourceGroupCutoff, targetGroupCutoff, mu, admission);
            return true;
    }

    std::string getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const
    {

            std::ostringstream out;
            out << std::setprecision(17);
            if(!owner_.parameters_.withDDMC ||
               !owner_.parameters_.ddmcInterfaceDiagnostics)
            {
                return out.str();
            }
            for(std::size_t cellIndex = 0;
                 cellIndex < owner_.ddmcCellData_.size(); ++cellIndex)
            {
                DDMCCellData const &data = owner_.ddmcCellData_[cellIndex];
                std::size_t const sourceID = cellIndex < owner_.ddmcPointCellID_.size()
                    ? owner_.ddmcPointCellID_[cellIndex]
                    : radiation_imc_detail::ddmcStableCellID(
                        owner_.componentGrid(), cellIndex, owner_.cells_[cellIndex]);
                double const sourceGeneratorX =
                    owner_.componentGrid().GetMeshPoint(cellIndex)[0];
                double const sourceCellCMX = owner_.componentGrid().GetCellCM(cellIndex)[0];
                double const volume = owner_.componentGrid().GetVolume(cellIndex);
                for(DDMCFaceLeak const &face :
                    data.faceLeaks)
                {
                    PointT const center = owner_.componentGrid().FaceCM(face.faceIndex);
                    if(center[0] < xMin || center[0] > xMax)
                    {
                        continue;
                    }

                    std::size_t const target = face.nextCellIndex;
                    std::size_t const targetID = target < owner_.ddmcPointCellID_.size()
                        ? owner_.ddmcPointCellID_[target]
                        : std::numeric_limits<std::size_t>::max();
                    double const targetGeneratorX =
                        target < owner_.componentGrid().getMeshPoints().size()
                        ? static_cast<double>(owner_.componentGrid().GetMeshPoint(target)[0])
                        : std::numeric_limits<double>::quiet_NaN();
                    int const targetEligible = target < owner_.ddmcPointEligible_.size()
                        ? owner_.ddmcPointEligible_[target] : 0;
                    double const targetSigma =
                        target < owner_.ddmcPointSigmaDiffusion_.size()
                        ? owner_.ddmcPointSigmaDiffusion_[target] : 0.0;
                    double const targetD =
                        target < owner_.ddmcPointDiffusionCoefficient_.size()
                        ? owner_.ddmcPointDiffusionCoefficient_[target] : 0.0;

                    out << sourceID << '\t' << targetID
                        << '\t' << face.faceIndex
                        << '\t' << sourceGeneratorX
                        << '\t' << sourceCellCMX
                        << '\t' << targetGeneratorX
                        << '\t' << center[0]
                        << '\t' << volume
                        << '\t' << data.groupCutoff
                        << '\t' << face.targetGroupCutoff
                        << '\t' << (data.eligible ? 1 : 0)
                        << '\t' << targetEligible
                        << '\t' << data.sigmaDiffusion
                        << '\t' << targetSigma
                        << '\t' << data.diffusionCoefficient
                        << '\t' << targetD
                        << '\t' << face.sourceDistanceToFace
                        << '\t' << face.targetDistanceToFace
                        << '\t' << face.area
                        << '\t' << face.conductance
                        << '\t' << face.internalRate
                        << '\t' << face.boundaryRate
                        << '\t' << face.sourceBandMass
                        << '\t' << face.commonBandMass
                        << '\t' << face.ddmcFraction
                        << '\t' << face.ddmcRate
                        << '\t' << face.transportRate
                        << '\t' << face.rate
                        << '\n';
                }
            }
            return out.str();
    }

    std::string getDDMCInterfaceEventDiagnosticsTSV(double xMin, double xMax) const
    {

            std::ostringstream out;
            out << std::setprecision(17);
            if(!owner_.parameters_.withDDMC ||
               !owner_.parameters_.ddmcInterfaceDiagnostics)
            {
                return out.str();
            }

            auto eventName = [](DDMCDiagnosticEventKind kind)
            {
                switch(kind)
                {
                    case DDMCDiagnosticEventKind::IMCCandidate:
                        return "imc_candidate";
                    case DDMCDiagnosticEventKind::IMCFrequencyReject:
                        return "imc_frequency_reject";
                    case DDMCDiagnosticEventKind::IMCIncident:
                        return "imc_incident";
                    case DDMCDiagnosticEventKind::IMCAdmitted:
                        return "imc_admitted";
                    case DDMCDiagnosticEventKind::IMCReflected:
                        return "imc_reflected";
                    case DDMCDiagnosticEventKind::IMCBypass:
                        return "imc_bypass";
                    case DDMCDiagnosticEventKind::DDMCToDDMC:
                        return "ddmc_to_ddmc";
                    case DDMCDiagnosticEventKind::DDMCToIMC:
                        return "ddmc_to_imc";
                }
                return "unknown";
            };

            for(auto const &item : owner_.ddmcDiagnosticEvents_)
            {
                DDMCDiagnosticEventKey const &key = item.first;
                DDMCDiagnosticEventAccumulator const &entry = item.second;
                if(entry.faceX < xMin || entry.faceX > xMax)
                {
                    continue;
                }
                long long const outputGroup = key.group == DDMC_DIAGNOSTIC_GREY_GROUP
                    ? -1LL : static_cast<long long>(key.group);
                out << eventName(key.kind)
                    << '\t' << entry.sourceCellID
                    << '\t' << entry.targetCellID
                    << '\t' << entry.faceIndex
                    << '\t' << entry.sourceGeneratorX
                    << '\t' << entry.targetGeneratorX
                    << '\t' << entry.faceX
                    << '\t' << outputGroup
                    << '\t' << entry.sourceGroupCutoff
                    << '\t' << entry.targetGroupCutoff
                    << '\t' << entry.count
                    << '\t' << entry.signedEnergy
                    << '\t' << entry.absoluteEnergy
                    << '\t' << entry.muSum
                    << '\t' << entry.muCount
                    << '\t' << entry.admissionProbabilitySum
                    << '\t' << entry.admissionProbabilityCount
                    << '\n';
            }
            return out.str();
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_DDMCENGINE_HPP
