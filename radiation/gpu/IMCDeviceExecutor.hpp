#ifndef STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP
#define STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../imc/IMCComponentBase.hpp"
#include "../../gpu/ProfileRegion.hpp"

namespace STORM::radiation_imc_detail {

// Owns the transport bridge between the portable IMC event kernel and the
// Kokkos/GreyIMC data.  The public RadiationIMC methods below are compatibility
// forwarding functions; all eligibility and view construction lives here.
template<typename Owner>
class IMCDeviceExecutor final : public IMCComponentBase<Owner>
{
    using Base = IMCComponentBase<Owner>;
    using Base::owner_;
    using typename Base::PointT;
    using typename Base::CellT;
    using typename Base::ExtensivesT;
    using Base::NumGroups;

public:
    explicit IMCDeviceExecutor(Owner &owner) : Base(owner)
    {}

    void InvalidateHostTransportViews()
    {
        hostTransportViewsValid_ = false;
    }

    // Build/synchronize the device-side transport data for a completed
    // pre-step. Host views are rebuilt lazily after all pre-step allocations;
    // device synchronization additionally runs when GPU support is enabled.
    void prepareStep()
    {
        this->InvalidateHostTransportViews();
        this->PrepareSlabTransport();
        ddmcThermalSamplingEligible_ = this->ThermalSamplingSnapshotEligible();
        const bool ddmcEventKernelEligible = this->SharedDDMCEventKernelEligible();
        if(ddmcEventKernelEligible)
        {
            std::vector<double> temperatures(owner_.cells_.size(), 0.0);
            for(std::size_t i = 0; i < owner_.cells_.size(); ++i)
            {
                temperatures[i] = owner_.cells_[i].temperature;
            }
            ddmcSnapshot_.Build(
                owner_.ddmcCellData_, owner_.componentGrid(),
                temperatures,
                owner_.ddmcPointCellID_,
                owner_.parameters_.withMultigroupDDMC &&
                    owner_.parameters_.withMultigroupOpacity);
            const auto &gridData = owner_.componentGridData();
            ddmcSnapshot_.BuildInterface(
                owner_.componentGrid(),
                gridData.cellFaceOffsets,
                gridData.normals,
                gridData.pointsOnFaces,
                gridData.nextCellIndices,
                owner_.transportCellVelocities_,
                owner_.ddmcPointEligible_,
                owner_.ddmcPointSigmaDiffusion_,
                owner_.ddmcPointSingleScatterAlbedo_,
                owner_.ddmcPointGroupCutoff_,
                owner_.ddmcPointVelocity_,
                owner_.ddmcPointCellID_,
                owner_.parameters_.withHydro &&
                    !owner_.parameters_.MMC &&
                    !owner_.parameters_.staticScatterers &&
                    owner_.parameters_.ddmcUseMovingInterfaceCorrection,
                owner_.parameters_.ddmcMaxInterfaceVelocityOverC,
                owner_.parameters_.ddmcInterfaceTargetWeightRatio,
                owner_.parameters_.ddmcMaxInterfaceSplits,
                owner_.parameters_.ddmcMaxMovingInterfaceWeightCorrection);
        }
        else
        {
            ddmcSnapshot_.enabled = false;
            ddmcSnapshot_.fluxRhs.clear();
        }
#ifdef STORM_WITH_GPU
        const bool ddmcDeviceEligible =
            ddmcEventKernelEligible &&
            owner_.parameters_.ddmcGpuEnable;
        gpuTransportEnabled_ = owner_.GreyKernelEligible() or
                               (owner_.SharedFullIMCKernelEligible() && this->PortableSpectralEligible()) or
                               ddmcDeviceEligible;
        if(!gpuTransportEnabled_)
        {
            return;
        }
        if(!gpuRuntime_)
        {
            gpuRuntime_ = std::make_unique<gpu::KokkosRuntime>();
        }
        if(!gpuData_)
        {
            gpuData_ = std::make_unique<gpu::GreyIMCData>();
        }
        STORM_PROFILE_REGION("storm/upload");

        const std::size_t buildGeneration = owner_.componentGrid().GetBuildGeneration();
        if(gpuGridBuildGeneration_ != buildGeneration)
        {
            STORM_PROFILE_REGION("storm/upload/grid");
            const auto &gridData = owner_.componentGridData();
            // RICH can retain ghost material cells beyond the owned geometry.
            // Device transport indexes only the owned cells in this snapshot.
            std::vector<cell_id_t> cellIDs(gridData.cellCenters.size());
            for(std::size_t i = 0; i < cellIDs.size(); ++i)
            {
                cellIDs[i] = static_cast<cell_id_t>(
                    radiation_imc_detail::ddmcStableCellID(
                        owner_.componentGrid(), i,
                        owner_.cells_[i]));
            }
            gpuData_->UploadGrid(
                gridData.cellFaceOffsets,
                gridData.cellCenters,
                cellIDs,
                gridData.normals,
                gridData.facePlaneOffsets,
                gridData.nextCellIndices,
                gridData.boundaryCrossings,
                gridData.deviceBoundaryBehaviors);
            gpuData_->UploadSourceTets(
                gridData.tetOffsets,
                gridData.tetCumVolumes,
                gridData.tetTris,
                gridData.vertices);
            gpuGridBuildGeneration_ = buildGeneration;
        }
        {
            STORM_PROFILE_REGION("storm/upload/tables");
            gpuData_->UploadTables(owner_.planckOpacities_, owner_.scatteringOpacities_, owner_.factorFleck_);
        }
        if(ddmcDeviceEligible)
        {
            STORM_PROFILE_REGION("storm/upload/ddmc");
            gpuData_->UploadDDMC(ddmcSnapshot_);
        }
        else
        {
            gpuData_->DisableDDMC();
        }
        if(owner_.parameters_.withHydro)
        {
            STORM_PROFILE_REGION("storm/upload/hydro");
            gpuData_->UploadHydro(owner_.transportCellVelocities_);
        }
        else
        {
            gpuData_->DisableHydro();
        }

        if(owner_.parameters_.withMultigroupOpacity &&
           (owner_.SharedFullIMCKernelEligible() || ddmcDeviceEligible))
        {
            STORM_PROFILE_REGION("storm/upload/spectral");
            std::vector<double> energyBoundaries(
                owner_.energyBoundaries_.begin(),
                owner_.energyBoundaries_.end());
            gpuData_->UploadSpectral(
                energyBoundaries,
                owner_.spectralAbsorptionScale_,
                owner_.thermalEmissionCdf_,
                owner_.groupAbsorptionOpacities_, owner_.thermalKT_,
                owner_.opacity_->GetPortableThermalFrequencyLaw());
        }
        else
            gpuData_->DisableSpectral();
        gpuData_->ResetCensusTallies();

        if(owner_.parameters_.withRandomWalk && owner_.randomWalk_ &&
           (owner_.GreyKernelEligible() ||
            owner_.SharedFullIMCKernelEligible() ||
            ddmcDeviceEligible))
        {
            STORM_PROFILE_REGION("storm/upload/random_walk");
            gpuData_->UploadRandomWalk(
                owner_.rwCellEligible_,
                owner_.rwCellTotalOpacity_,
                owner_.rwCellData_,
                *owner_.randomWalk_,
                owner_.parameters_.rwMinParticleOpticalDepth);
        }
        else
            gpuData_->DisableRandomWalk();
#endif
    }

    bool SupportsDeviceCensusTallies() const
    {
#ifdef STORM_WITH_GPU
        return gpuTransportEnabled_ && gpuData_ &&
               !owner_.parameters_.withCompton &&
               !owner_.parameters_.postProcess.enabled;
#else
        return false;
#endif
    }

    bool copyCensusTallies(std::vector<double> &radiationEnergy, std::vector<double> &groupRadiationEnergy)
    {
#ifdef STORM_WITH_GPU
        if(!this->SupportsDeviceCensusTallies())
        {
            return false;
        }
        gpuData_->CopyCensusTallies(radiationEnergy, groupRadiationEnergy);
        return true;
#else
        (void) radiationEnergy;
        (void) groupRadiationEnergy;
        return false;
#endif
    }

    void addDDMCDiagnostics()
    {
#ifdef STORM_WITH_GPU
        if(gpuTransportEnabled_ && gpuData_)
        {
            gpuData_->AddDDMCDiagnostics(
                owner_.ddmcInterfaceIncidentCount_,
                owner_.ddmcInterfaceAdmittedCount_,
                owner_.ddmcInterfaceReflectedCount_,
                owner_.ddmcInterfaceGuAppliedCount_,
                owner_.ddmcInterfaceGuFallbackCount_,
                owner_.ddmcInterfaceBypassCount_,
                owner_.ddmcInterfaceSplitPacketCount_,
                owner_.ddmcFallbackCount_,
                owner_.ddmcExternalSourceThermalizationCount_,
                owner_.ddmcExternalSourceStayDDMCCount_,
                owner_.ddmcExternalSourceToIMCCount_,
                owner_.ddmcExternalSourceThermalizedEnergy_,
                owner_.ddmcExternalSourceToIMCEnergy_);
        }
#else
        (void) 0;
#endif
    }

    void addTallies(std::vector<double> &material,
                    std::vector<double> &radiation,
                    std::vector<double> &groupRadiation,
                    std::vector<PointT> &momentum,
                    std::vector<PointT> &ddmcFluxRhs,
                    std::size_t &randomWalkSteps,
                    std::size_t &ddmcSteps,
                    std::size_t &ddmcLeaks,
                    std::size_t &ddmcResidentLeaks,
                    std::size_t &ddmcTransportLeaks,
                    std::size_t &ddmcRemoteResidentLeaks,
                    std::size_t &ddmcCensus)
    {
#ifdef STORM_WITH_GPU
        if(gpuTransportEnabled_)
        {
            gpuData_->AddTallies(
                material, radiation, groupRadiation, momentum,
                ddmcFluxRhs,
                randomWalkSteps, ddmcSteps, ddmcLeaks,
                ddmcResidentLeaks, ddmcTransportLeaks,
                ddmcRemoteResidentLeaks, ddmcCensus);
        }
#else
        (void) material;
        (void) radiation;
        (void) groupRadiation;
        (void) momentum;
        (void) ddmcFluxRhs;
        (void) randomWalkSteps;
        (void) ddmcSteps;
        (void) ddmcLeaks;
        (void) ddmcResidentLeaks;
        (void) ddmcTransportLeaks;
        (void) ddmcRemoteResidentLeaks;
        (void) ddmcCensus;
#endif
    }

#ifdef STORM_WITH_GPU
    bool UsesDeviceTransport() const
    {
        return gpuTransportEnabled_;
    }

    gpu::GreyIMCData *DeviceData()
    {
        return gpuData_.get();
    }

    bool SupportsDeviceSourceGeneration() const
    {
        return !owner_.parameters_.withCompton &&
               !owner_.parameters_.postProcess.enabled &&
               !owner_.adaptiveSourceCellGroupScoresEnabled_ &&
               !owner_.postProcessExternalSourceMode_ &&
               (owner_.GreyKernelEligible() ||
                (owner_.SharedFullIMCKernelEligible() && this->PortableSpectralEligible()) ||
                owner_.SharedDDMCKernelEligible());
    }

    gpu::GreyIMCViews<gpu::DeviceVec3> GetDeviceTransportViews() const
    {
        bool comovingTransport = false;
        bool depositMomentum = false;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            comovingTransport = owner_.parameters_.withHydro and
                not owner_.parameters_.MMC and
                not owner_.parameters_.staticScatterers;
        }
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            depositMomentum = owner_.parameters_.withHydro &&
                              !owner_.parameters_.diffusionPressureGradient &&
                              !owner_.parameters_.noHydroFeedback;
        }
        gpu::GreyIMCViews<gpu::DeviceVec3> result = gpuData_->Views(owner_.lightSpeed(), !owner_.parameters_.noHydroFeedback, comovingTransport, depositMomentum, owner_.parameters_.staticScatterers);
        // Only a DDMC-restricted launch may reject IMC packets on the device.
        // Grey and full-IMC eligibility also enable device transport, so
        // keying this off DDMC alone forces every IMC packet to host fallback.
        result.ddmcOnlyTransport = (owner_.GreyKernelEligible() or owner_.SharedFullIMCKernelEligible())? 0u : 1u;
        this->SetSlabView(result.grid);
        return result;
    }
#endif

    bool GreyKernelEligible() const
    {
#if defined(STORM_DEBUG) || defined(STORM_WITH_TRACING_HISTORY)
        return false;
#else
        return !owner_.parameters_.withMultigroupOpacity &&
               !owner_.parameters_.withDDMC &&
               !owner_.parameters_.withCompton &&
               !(owner_.parameters_.withHydro &&
                 owner_.parameters_.withRandomWalk) &&
               !owner_.parameters_.postProcess.enabled &&
               !owner_.observer_ &&
               !owner_.polarizationEnabled();
#endif
    }

    bool SharedRandomWalkKernelEligible() const
    {
        return owner_.parameters_.withRandomWalk &&
               this->ThermalSamplingSnapshotEligible() &&
               !owner_.parameters_.withCompton &&
               !owner_.parameters_.withHydro &&
               !(owner_.parameters_.postProcess.enabled &&
                 owner_.parameters_.postProcess.useCellVelocities) &&
               !owner_.polarizationEnabled();
    }

    const gpu::GreyIMCViews<PointT> &GetHostTransportViews()
    {
        // Geometry, material tables, DDMC snapshots and tally storage are
        // fixed during transport. prepareStep invalidates these pointers on
        // every step, including after mesh changes or vector reallocations.
        // The differential harness can toggle kernel eligibility mid-step.
        if(!hostTransportViewsValid_ || hostViewsForceLegacy_ != owner_.imcDiffForceLegacy_)
        {
            hostTransportViews_ = this->BuildHostTransportViews();
            hostViewsForceLegacy_ = owner_.imcDiffForceLegacy_;
            hostTransportViewsValid_ = true;
        }
        return hostTransportViews_;
    }

private:
    gpu::GreyIMCViews<PointT> BuildHostTransportViews()
    {
        gpu::GreyIMCViews<PointT> result;
        const auto &gridData = owner_.componentGridData();
        result.grid.cellFaceOffsets = gridData.cellFaceOffsets.data();
        result.grid.cellCenters = gridData.cellCenters.data();
        result.grid.normals = gridData.normals.data();
        result.grid.facePlaneOffsets = gridData.facePlaneOffsets.data();
        result.grid.nextCellIndices = gridData.nextCellIndices.data();
        result.grid.boundaryCrossings = gridData.boundaryCrossings.data();
        result.grid.deviceBoundaryBehaviors = gridData.deviceBoundaryBehaviors.data();
        result.grid.cellCount = owner_.componentGrid().GetPointNo();
        this->SetSlabView(result.grid);
        result.absorptionOpacities = owner_.planckOpacities_.data();
        result.scatteringOpacities = owner_.scatteringOpacities_.data();
        result.fleckFactors = owner_.factorFleck_.data();
        result.cellVelocities = owner_.transportCellVelocities_.data();
        result.pendingMaterialEnergy = owner_.pendingMaterialEnergy_.data();
        result.pendingRadiationEnergy = owner_.pendingRadiationEnergy_.data();
        result.pendingMomentum = owner_.pendingMomentum_.data();
        result.ddmc = ddmcSnapshot_.View();
        result.ddmc.cellTetOffsets = gridData.tetOffsets.data();
        result.ddmc.cellTetCumVolumes = gridData.tetCumVolumes.data();
        result.ddmc.cellTetTris = gridData.tetTris.data();
        result.ddmc.cellVertices = gridData.vertices.data();
        result.ddmc.fluxRhs = owner_.ddmcFluxRhsIntegrated_.data();
        result.ddmc.interfaceIncidentCount = &owner_.ddmcInterfaceIncidentCount_;
        result.ddmc.interfaceAdmittedCount = &owner_.ddmcInterfaceAdmittedCount_;
        result.ddmc.interfaceReflectedCount = &owner_.ddmcInterfaceReflectedCount_;
        result.ddmc.interfaceGuAppliedCount = &owner_.ddmcInterfaceGuAppliedCount_;
        result.ddmc.interfaceGuFallbackCount = &owner_.ddmcInterfaceGuFallbackCount_;
        result.ddmc.interfaceBypassCount = &owner_.ddmcInterfaceBypassCount_;
        result.ddmc.interfaceSplitPacketCount = &owner_.ddmcInterfaceSplitPacketCount_;
        result.ddmc.hostFallbackCount = &owner_.ddmcFallbackCount_;
        result.ddmc.externalSourceThermalizationCount = &owner_.ddmcExternalSourceThermalizationCount_;
        result.ddmc.externalSourceStayDDMCCount = &owner_.ddmcExternalSourceStayDDMCCount_;
        result.ddmc.externalSourceToIMCCount = &owner_.ddmcExternalSourceToIMCCount_;
        result.ddmc.externalSourceThermalizedEnergy = &owner_.ddmcExternalSourceThermalizedEnergy_;
        result.ddmc.externalSourceToIMCEnergy = &owner_.ddmcExternalSourceToIMCEnergy_;
        result.energyBoundaries = owner_.energyBoundaries_.data();
        result.spectralAbsorptionScale = owner_.spectralAbsorptionScale_.data();
        result.thermalEmissionCdf = owner_.thermalEmissionCdf_.data();
        result.groupAbsorptionOpacities = owner_.groupAbsorptionOpacities_.empty() ? nullptr : owner_.groupAbsorptionOpacities_.data();
        result.thermalKT = owner_.thermalKT_.data();
        result.thermalFrequencyLaw = owner_.opacity_->GetPortableThermalFrequencyLaw();
        result.pendingGroupRadiationEnergy = (owner_.parameters_.withEgTimeAvg and not owner_.pendingGroupRadiationEnergy_.empty())?
                                                owner_.pendingGroupRadiationEnergy_.data() : nullptr;
        result.groupCount = NumGroups;
        if(owner_.parameters_.withRandomWalk and owner_.randomWalk_)
        {
            result.randomWalk.cellEligible = owner_.rwCellEligible_.data();
            result.randomWalk.cellTotalOpacity = owner_.rwCellTotalOpacity_.data();
            result.randomWalk.pgrwCells = owner_.rwCellData_.data();
            result.randomWalk.tables.tau = owner_.randomWalk_->GetTauTable().data();
            result.randomWalk.tables.survival = owner_.randomWalk_->GetSurvivalTable().data();
            result.randomWalk.tables.radius = owner_.randomWalk_->GetRadiusTable().data();
            result.randomWalk.tables.tableSize = owner_.randomWalk_->GetTauTable().size();
            result.randomWalk.tables.radiusTableSize = RandomWalk::GetRadiusTableSize();
            result.randomWalk.tables.tauMin = RandomWalk::GetMinimumTau();
            result.randomWalk.tables.tauMax = RandomWalk::GetMaximumTau();
            result.randomWalk.minimumParticleOpticalDepth = owner_.parameters_.rwMinParticleOpticalDepth;
            result.randomWalk.enabled = 1;
            result.randomWalk.spectralEnabled = owner_.parameters_.withMultigroupOpacity and owner_.rwCellData_.size() == result.grid.cellCount;
        }
        result.speedOfLight = owner_.lightSpeed();
        result.weightCutoffFraction = owner_.parameters_.weightCutoffFraction;
        result.depositMaterialEnergy = not owner_.parameters_.noHydroFeedback and not owner_.parameters_.postProcess.enabled;
        result.staticScatterers = owner_.parameters_.staticScatterers;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            result.comovingTransport = owner_.parameters_.withHydro and
                not owner_.parameters_.MMC and
                not owner_.parameters_.staticScatterers;
        }
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            result.depositMomentum = owner_.parameters_.withHydro and not owner_.parameters_.diffusionPressureGradient and not owner_.parameters_.noHydroFeedback;
        }
        result.spectralEnabled = (owner_.SharedFullIMCKernelEligible() or(this->SharedDDMCEventKernelEligible() and owner_.parameters_.withMultigroupOpacity))? 1 : 0;
        return result;
    }

public:
    bool SharedFullIMCKernelEligible() const
    {
#if STORM_DEBUG
        return false;
#endif
#ifdef STORM_WITH_TRACING_HISTORY
        return false;
#endif
        if(owner_.imcDiffForceLegacy_)
        {
            return false;
        }
        return owner_.parameters_.withMultigroupOpacity &&
               !owner_.parameters_.withDDMC &&
               !owner_.parameters_.withCompton &&
               !(owner_.parameters_.withHydro &&
                 owner_.parameters_.withRandomWalk) &&
               !owner_.parameters_.postProcess.enabled &&
               !owner_.observer_ &&
               !owner_.polarizationEnabled();
    }

    bool SharedDDMCKernelEligible() const
    {
#if defined(STORM_DEBUG) || defined(STORM_WITH_TRACING_HISTORY)
        return false;
#else
        return owner_.parameters_.withDDMC &&
               !owner_.parameters_.withCompton &&
               !owner_.parameters_.postProcess.enabled &&
               !owner_.observer_ &&
               !owner_.polarizationEnabled() &&
               ddmcThermalSamplingEligible_ && this->PortableSpectralEligible();
#endif
    }

    bool SharedDDMCEventKernelEligible() const
    {
#if defined(STORM_DEBUG) || defined(STORM_WITH_TRACING_HISTORY)
        return false;
#else
        return owner_.parameters_.withDDMC &&
               !owner_.parameters_.withCompton &&
               (!owner_.parameters_.postProcess.enabled ||
                owner_.postProcessExternalSourceMode_) &&
               !owner_.polarizationEnabled() &&
               ddmcThermalSamplingEligible_ && this->PortableSpectralEligible();
#endif
    }

private:
    void PrepareSlabTransport()
    {
        slabTransport_ = owner_.parameters_.withSlabTransport;
        if(!slabTransport_) return;
        if(owner_.parameters_.withHydro || owner_.parameters_.withDDMC ||
           owner_.parameters_.withCompton || owner_.parameters_.postProcess.enabled ||
           owner_.observer_ || owner_.polarizationEnabled() ||
           !(owner_.GreyKernelEligible() || owner_.SharedFullIMCKernelEligible()))
        {
            throw StormError("Slab transport requires static shared IMC without DDMC, observers, or polarization");
        }
        const auto &data = owner_.componentGridData();
        // Validate the opt-in against the actual mesh, including every local
        // transverse face. Interior y/z interfaces or non-box walls cannot
        // be skipped. A zero-cell rank never launches transport.
        for(std::size_t cell = 0; cell < owner_.componentGrid().GetPointNo(); ++cell)
        {
            unsigned seen = 0;
            for(std::size_t face = data.cellFaceOffsets[cell];
                face < data.cellFaceOffsets[cell + 1]; ++face)
            {
                const auto &normal = data.normals[face];
                const double components[] = {normal.x, normal.y, normal.z};
                int axis = -1;
                for(int d = 0; d < 3; ++d)
                {
                    if(components[d] != 0.0)
                    {
                        if(axis != -1 || std::abs(components[d]) != 1.0)
                            throw StormError("Slab transport requires axis-aligned box cells");
                        axis = d;
                    }
                }
                if(axis < 0) throw StormError("Slab transport encountered an invalid face normal");
                const bool upper = components[axis] < 0.0;
                const unsigned bit = 1u << (2 * axis + unsigned(upper));
                if(seen & bit) throw StormError("Slab transport requires six distinct box faces");
                seen |= bit;
                if(axis == 0) continue;
                if(!data.boundaryCrossings[face] ||
                   data.deviceBoundaryBehaviors[face] !=
                       static_cast<std::uint8_t>(DeviceBoundaryFaceBehavior::ReflectingRigid))
                    throw StormError("Slab transport requires reflecting external y/z faces");
                const std::size_t index = 2 * (axis - 1) + unsigned(upper);
                const double bound = data.facePlaneOffsets[face] / components[axis];
                if(cell == 0) slabBounds_[index] = bound;
                else if(std::abs(bound - slabBounds_[index]) >
                        1e-12 * std::max(1.0, std::abs(bound)))
                    throw StormError("Slab transport requires common transverse bounds");
            }
            if(seen != 63u || !(slabBounds_[1] > slabBounds_[0]) ||
               !(slabBounds_[3] > slabBounds_[2]))
                throw StormError("Slab transport requires nondegenerate box cells");
        }
    }

    template<typename P>
    void SetSlabView(gpu::FlatGridView<P> &grid) const
    {
        grid.slabTransport = slabTransport_;
        grid.slabLowerY = slabBounds_[0];
        grid.slabUpperY = slabBounds_[1];
        grid.slabLowerZ = slabBounds_[2];
        grid.slabUpperZ = slabBounds_[3];
    }

    bool PortableSpectralEligible() const
    {
        return !owner_.parameters_.withMultigroupOpacity ||
            (owner_.opacity_->GetPortableAbsorptionLaw() != PortableAbsorptionLaw::Unsupported &&
             owner_.opacity_->GetPortableThermalFrequencyLaw() != ThermalFrequencyLaw::Unsupported);
    }

    bool ThermalSamplingSnapshotEligible() const
    {
        return not owner_.parameters_.withMultigroupOpacity or (owner_.opacity_->GetPortableThermalFrequencyLaw() != ThermalFrequencyLaw::Unsupported);
    }

    ddmc::HostSnapshot<PointT> ddmcSnapshot_;
    gpu::GreyIMCViews<PointT> hostTransportViews_;
    bool hostTransportViewsValid_ = false;
    bool hostViewsForceLegacy_ = false;
    bool ddmcThermalSamplingEligible_ = true;
    bool slabTransport_ = false;
    double slabBounds_[4] = {};
#ifdef STORM_WITH_GPU
    std::unique_ptr<gpu::KokkosRuntime> gpuRuntime_;
    std::unique_ptr<gpu::GreyIMCData> gpuData_;
    bool gpuTransportEnabled_ = false;
    std::size_t gpuGridBuildGeneration_ = std::numeric_limits<std::size_t>::max();
#endif
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP
