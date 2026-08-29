#ifndef STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP
#define STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP

#include <limits>
#include <vector>

#include "../imc/IMCComponentBase.hpp"

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

    // Build/synchronize the device-side transport data for a completed
    // pre-step.  The CPU configuration intentionally compiles this as a
    // no-op, keeping the lifecycle process independent of Kokkos headers.
    void prepareStep()
    {
        const bool ddmcKernelEligible = this->SharedDDMCKernelEligible();
        if(ddmcKernelEligible)
        {
            std::vector<double> temperatures(owner_.cells_.size(), 0.0);
            for(std::size_t i = 0; i < owner_.cells_.size(); ++i)
            {
                temperatures[i] = owner_.cells_[i].temperature;
            }
            ddmcSnapshot_.Build(
                owner_.ddmcCellData_, owner_.componentGrid(),
                owner_.parameters_.ddmcMinParticleOpticalDepth,
                temperatures,
                owner_.ddmcPointCellID_,
                owner_.parameters_.ddmcUseMultigroupPGRW &&
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
                owner_.ddmcPointGroupCutoff_,
                owner_.ddmcPointVelocity_,
                owner_.ddmcPointCellID_,
                owner_.parameters_.withHydro &&
                    !owner_.parameters_.MMC &&
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
            ddmcKernelEligible && owner_.parameters_.ddmcGpuEnable;
        gpuTransportEnabled_ = owner_.GreyKernelEligible() or
                               owner_.SharedFullIMCKernelEligible() or
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

        const std::size_t buildGeneration = owner_.componentGrid().GetBuildGeneration();
        if(gpuGridBuildGeneration_ != buildGeneration)
        {
            const auto &gridData = owner_.componentGridData();
            gpuData_->UploadGrid(
                gridData.cellFaceOffsets,
                gridData.cellCenters,
                gridData.normals,
                gridData.facePlaneOffsets,
                gridData.nextCellIndices,
                gridData.boundaryCrossings,
                gridData.deviceBoundaryBehaviors);
            gpuGridBuildGeneration_ = buildGeneration;
        }
        gpuData_->UploadTables(
            owner_.planckOpacities_,
            owner_.scatteringOpacities_,
            owner_.factorFleck_);
        if(ddmcDeviceEligible)
            gpuData_->UploadDDMC(ddmcSnapshot_);
        else
            gpuData_->DisableDDMC();
        if(owner_.parameters_.withHydro)
        {
            gpuData_->UploadHydro(owner_.transportCellVelocities_);
        }
        else
        {
            gpuData_->DisableHydro();
        }

        if(owner_.parameters_.withMultigroupOpacity &&
           (owner_.SharedFullIMCKernelEligible() || ddmcDeviceEligible))
        {
            std::vector<double> energyBoundaries(
                owner_.energyBoundaries_.begin(),
                owner_.energyBoundaries_.end());
            gpuData_->UploadSpectral(
                energyBoundaries,
                owner_.spectralAbsorptionScale_,
                owner_.thermalEmissionCdf_);
        }
        else
            gpuData_->DisableSpectral();

        if(owner_.parameters_.withRandomWalk && owner_.randomWalk_ &&
           (owner_.GreyKernelEligible() ||
            owner_.SharedFullIMCKernelEligible() ||
            ddmcDeviceEligible))
        {
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
                owner_.ddmcFallbackCount_);
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

    gpu::GreyIMCViews<gpu::DeviceVec3> GetDeviceTransportViews() const
    {
        bool comovingTransport = false;
        bool depositMomentum = false;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            comovingTransport = owner_.parameters_.withHydro and not owner_.parameters_.MMC;
        }
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            depositMomentum = owner_.parameters_.withHydro &&
                              !owner_.parameters_.diffusionPressureGradient &&
                              !owner_.parameters_.noHydroFeedback;
        }
        return gpuData_->Views(
            units::clight,
            !owner_.parameters_.noHydroFeedback,
            comovingTransport,
            depositMomentum);
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
               !owner_.parameters_.withCompton &&
               !owner_.parameters_.withHydro &&
               !(owner_.parameters_.postProcess.enabled &&
                 owner_.parameters_.postProcess.useCellVelocities) &&
               !owner_.polarizationEnabled();
    }

    gpu::GreyIMCViews<PointT> GetHostTransportViews()
    {
        gpu::GreyIMCViews<PointT> result;
        const auto &gridData = owner_.componentGridData();
        result.grid.cellFaceOffsets = gridData.cellFaceOffsets.data();
        result.grid.cellCenters = gridData.cellCenters.data();
        result.grid.normals = gridData.normals.data();
        result.grid.facePlaneOffsets = gridData.facePlaneOffsets.data();
        result.grid.nextCellIndices = gridData.nextCellIndices.data();
        result.grid.boundaryCrossings = gridData.boundaryCrossings.data();
        result.grid.deviceBoundaryBehaviors =
            gridData.deviceBoundaryBehaviors.data();
        result.grid.cellCount = owner_.componentGrid().GetPointNo();
        result.absorptionOpacities = owner_.planckOpacities_.data();
        result.scatteringOpacities = owner_.scatteringOpacities_.data();
        result.fleckFactors = owner_.factorFleck_.data();
        result.cellVelocities = owner_.transportCellVelocities_.data();
        result.pendingMaterialEnergy = owner_.pendingMaterialEnergy_.data();
        result.pendingRadiationEnergy = owner_.pendingRadiationEnergy_.data();
        result.pendingMomentum = owner_.pendingMomentum_.data();
        result.ddmc = ddmcSnapshot_.View();
        result.ddmc.fluxRhs =
            owner_.ddmcFluxRhsIntegrated_.data();
        result.ddmc.interfaceIncidentCount =
            &owner_.ddmcInterfaceIncidentCount_;
        result.ddmc.interfaceAdmittedCount =
            &owner_.ddmcInterfaceAdmittedCount_;
        result.ddmc.interfaceReflectedCount =
            &owner_.ddmcInterfaceReflectedCount_;
        result.ddmc.interfaceGuAppliedCount =
            &owner_.ddmcInterfaceGuAppliedCount_;
        result.ddmc.interfaceGuFallbackCount =
            &owner_.ddmcInterfaceGuFallbackCount_;
        result.ddmc.interfaceBypassCount =
            &owner_.ddmcInterfaceBypassCount_;
        result.ddmc.interfaceSplitPacketCount =
            &owner_.ddmcInterfaceSplitPacketCount_;
        result.ddmc.hostFallbackCount = &owner_.ddmcFallbackCount_;
        result.energyBoundaries = owner_.energyBoundaries_.data();
        result.spectralAbsorptionScale =
            owner_.spectralAbsorptionScale_.data();
        result.thermalEmissionCdf = owner_.thermalEmissionCdf_.data();
        result.pendingGroupRadiationEnergy =
            owner_.parameters_.withEgTimeAvg &&
            !owner_.pendingGroupRadiationEnergy_.empty()
                ? owner_.pendingGroupRadiationEnergy_.data()
                : nullptr;
        result.groupCount = NumGroups;
        if(owner_.parameters_.withRandomWalk && owner_.randomWalk_)
        {
            result.randomWalk.cellEligible = owner_.rwCellEligible_.data();
            result.randomWalk.cellTotalOpacity =
                owner_.rwCellTotalOpacity_.data();
            result.randomWalk.pgrwCells =
                owner_.rwCellData_.data();
            result.randomWalk.tables.tau =
                owner_.randomWalk_->GetTauTable().data();
            result.randomWalk.tables.survival =
                owner_.randomWalk_->GetSurvivalTable().data();
            result.randomWalk.tables.radius =
                owner_.randomWalk_->GetRadiusTable().data();
            result.randomWalk.tables.tableSize =
                owner_.randomWalk_->GetTauTable().size();
            result.randomWalk.tables.radiusTableSize =
                RandomWalk::GetRadiusTableSize();
            result.randomWalk.tables.tauMin = RandomWalk::GetMinimumTau();
            result.randomWalk.tables.tauMax = RandomWalk::GetMaximumTau();
            result.randomWalk.minimumParticleOpticalDepth =
                owner_.parameters_.rwMinParticleOpticalDepth;
            result.randomWalk.enabled = 1;
            result.randomWalk.spectralEnabled =
                owner_.parameters_.withMultigroupOpacity &&
                owner_.rwCellData_.size() == result.grid.cellCount;
        }
        result.speedOfLight = units::clight;
        result.depositMaterialEnergy =
            !owner_.parameters_.noHydroFeedback &&
            !owner_.parameters_.postProcess.enabled;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            result.comovingTransport = owner_.parameters_.withHydro &&
                                        !owner_.parameters_.MMC;
        }
        if constexpr(radiation_imc_detail::has_member_momentum<
                         ExtensivesT>::value)
        {
            result.depositMomentum = owner_.parameters_.withHydro &&
                                      !owner_.parameters_.diffusionPressureGradient &&
                                      !owner_.parameters_.noHydroFeedback;
        }
        result.spectralEnabled =
            (owner_.SharedFullIMCKernelEligible() ||
             (this->SharedDDMCKernelEligible() &&
              owner_.parameters_.withMultigroupOpacity))
                ? 1
                : 0;
        return result;
    }

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
               !owner_.polarizationEnabled();
#endif
    }

private:
    ddmc::HostSnapshot<PointT> ddmcSnapshot_;
#ifdef STORM_WITH_GPU
    std::unique_ptr<gpu::KokkosRuntime> gpuRuntime_;
    std::unique_ptr<gpu::GreyIMCData> gpuData_;
    bool gpuTransportEnabled_ = false;
    std::size_t gpuGridBuildGeneration_ =
        std::numeric_limits<std::size_t>::max();
#endif
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_DEVICE_EXECUTOR_HPP
