#ifndef STORM_GPU_GREY_IMC_DATA_HPP
#define STORM_GPU_GREY_IMC_DATA_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "DeviceParticle.hpp"
#include "GreyIMCKernel.hpp"
#include "../radiation/RandomWalk.hpp"
#include "../radiation/ddmc/DDMCTypes.hpp"

namespace STORM
{
namespace gpu
{

class GreyIMCData
{
public:
    GreyIMCData()
        : cellFaceOffsets_("storm_cell_face_offsets", 0),
          cellCenters_("storm_cell_centers", 0),
          normals_("storm_face_normals", 0),
          facePlaneOffsets_("storm_face_plane_offsets", 0),
          nextCellIndices_("storm_next_cells", 0),
          boundaryCrossings_("storm_boundary_crossings", 0),
          deviceBoundaryBehaviors_("storm_device_boundary_behaviors", 0),
          cellTables_("storm_cell_tables", 0),
          cellVelocities_("storm_cell_velocities", 0),
          pendingMaterialEnergy_("storm_material_tally", 0),
          pendingRadiationEnergy_("storm_radiation_tally", 0),
          pendingMomentum_("storm_momentum_tally", 0),
          pendingGroupRadiationEnergy_("storm_group_radiation_tally", 0),
          energyBoundaries_("storm_energy_boundaries", 0),
          spectralAbsorptionScale_("storm_spectral_absorption_scale", 0),
          thermalEmissionCdf_("storm_thermal_emission_cdf", 0),
          ddmcCellEligible_("storm_ddmc_cell_eligible", 0),
          ddmcSigmaEnergyAbs_("storm_ddmc_sigma_energy_abs", 0),
          ddmcSigmaParticleGate_("storm_ddmc_sigma_particle_gate", 0),
          ddmcTotalLeakRate_("storm_ddmc_total_leak_rate", 0),
          ddmcLeakOffsets_("storm_ddmc_leak_offsets", 0),
          ddmcLeakRates_("storm_ddmc_leak_rates", 0),
          ddmcLeakDDMCRates_("storm_ddmc_leak_ddmc_rates", 0),
          ddmcNextCellIndices_("storm_ddmc_next_cells", 0),
          ddmcFaceKinds_("storm_ddmc_face_kinds", 0),
          ddmcTargetEligible_("storm_ddmc_target_eligible", 0),
          ddmcOutwardNormals_("storm_ddmc_outward_normals", 0),
          ddmcFaceCenters_("storm_ddmc_face_centers", 0),
          ddmcFluxRhs_("storm_ddmc_flux_rhs", 0),
          ddmcStepCount_("storm_ddmc_step_count"),
          ddmcLeakCount_("storm_ddmc_leak_count"),
          ddmcResidentLeakCount_("storm_ddmc_resident_leak_count"),
          ddmcTransportLeakCount_("storm_ddmc_transport_leak_count"),
          ddmcRemoteResidentLeakCount_("storm_ddmc_remote_leak_count"),
          ddmcCensusCount_("storm_ddmc_census_count"),
          randomWalkEligible_("storm_rw_eligible", 0),
          randomWalkTotalOpacity_("storm_rw_total_opacity", 0),
          randomWalkPGRWCells_("storm_rw_pgrw_cells", 0),
          randomWalkTau_("storm_rw_tau", 0),
          randomWalkSurvival_("storm_rw_survival", 0),
          randomWalkRadius_("storm_rw_radius", 0),
          randomWalkStepCounter_("storm_rw_step_counter")
    {}

    template<typename PointT>
    void UploadGrid(const std::vector<std::size_t> &cellFaceOffsets,
                    const std::vector<PointT> &cellCenters,
                    const std::vector<PointT> &normals,
                    const std::vector<double> &facePlaneOffsets,
                    const std::vector<cell_index_t> &nextCellIndices,
                    const std::vector<std::uint8_t> &boundaryCrossings,
                    const std::vector<std::uint8_t> &deviceBoundaryBehaviors)
    {
        const std::size_t directedFaceCount = normals.size();
        if(facePlaneOffsets.size() != directedFaceCount or nextCellIndices.size() != directedFaceCount or
           boundaryCrossings.size() != directedFaceCount or deviceBoundaryBehaviors.size() != directedFaceCount)
        {
            throw std::runtime_error("GreyIMCData::UploadGrid: directed-face table size mismatch");
        }
        Resize(this->cellFaceOffsets_, cellFaceOffsets.size());
        Resize(this->cellCenters_, cellCenters.size());
        Resize(this->normals_, normals.size());
        Resize(this->facePlaneOffsets_, facePlaneOffsets.size());
        Resize(this->nextCellIndices_, nextCellIndices.size());
        Resize(this->boundaryCrossings_, boundaryCrossings.size());
        Resize(this->deviceBoundaryBehaviors_, deviceBoundaryBehaviors.size());

        for(std::size_t i = 0; i < cellFaceOffsets.size(); ++i)
        {
            this->cellFaceOffsets_.h_view(i) = cellFaceOffsets[i];
        }
        for(std::size_t i = 0; i < cellCenters.size(); ++i)
        {
            this->cellCenters_.h_view(i) = DeviceVec3(cellCenters[i].x, cellCenters[i].y, cellCenters[i].z);
        }
        for(std::size_t i = 0; i < normals.size(); ++i)
        {
            this->normals_.h_view(i) = DeviceVec3(normals[i].x, normals[i].y, normals[i].z);
            this->facePlaneOffsets_.h_view(i) = facePlaneOffsets[i];
            this->nextCellIndices_.h_view(i) = nextCellIndices[i];
            this->boundaryCrossings_.h_view(i) = boundaryCrossings[i];
            this->deviceBoundaryBehaviors_.h_view(i) = deviceBoundaryBehaviors[i];
        }

        SyncToDevice(this->cellFaceOffsets_);
        SyncToDevice(this->cellCenters_);
        SyncToDevice(this->normals_);
        SyncToDevice(this->facePlaneOffsets_);
        SyncToDevice(this->nextCellIndices_);
        SyncToDevice(this->boundaryCrossings_);
        SyncToDevice(this->deviceBoundaryBehaviors_);
        this->cellCount_ = cellFaceOffsets.empty() ? 0 : cellFaceOffsets.size() - 1;
    }

    template<typename PointT>
    void UploadHydro(const std::vector<PointT> &cellVelocities)
    {
        if(cellVelocities.size() != this->cellCount_)
        {
            throw std::runtime_error("GreyIMCData::UploadHydro: cell count mismatch");
        }
        Resize(this->cellVelocities_, cellVelocities.size());
        Resize(this->pendingMomentum_, cellVelocities.size());
        for(std::size_t i = 0; i < cellVelocities.size(); ++i)
        {
            this->cellVelocities_.h_view(i) = DeviceVec3(cellVelocities[i].x, cellVelocities[i].y, cellVelocities[i].z);
        }
        SyncToDevice(this->cellVelocities_);
        Kokkos::deep_copy(this->pendingMomentum_.d_view, DeviceVec3{});
        this->pendingMomentum_.modify_device();
    }

    void DisableHydro()
    {
        Resize(this->cellVelocities_, 0);
        Resize(this->pendingMomentum_, 0);
    }

    void UploadSpectral(
        const std::vector<double> &energyBoundaries,
        const std::vector<double> &absorptionScale,
        const std::vector<double> &thermalEmissionCdf)
    {
        if(absorptionScale.size() != this->cellCount_)
        {
            throw std::runtime_error("GreyIMCData::UploadSpectral: cell count mismatch");
        }
        const std::size_t groupCount = energyBoundaries.empty()? 0: energyBoundaries.size() - 1;
        if(thermalEmissionCdf.size() != this->cellCount_ * (groupCount + 1))
        {
            throw std::runtime_error("GreyIMCData::UploadSpectral: CDF size mismatch");
        }
        CopyToDevice(energyBoundaries, this->energyBoundaries_);
        CopyToDevice(absorptionScale, this->spectralAbsorptionScale_);
        CopyToDevice(thermalEmissionCdf, this->thermalEmissionCdf_);
        this->groupCount_ = groupCount;
        this->spectralEnabled_ = true;
        Resize(this->pendingGroupRadiationEnergy_, this->cellCount_ * this->groupCount_);
        Kokkos::deep_copy(this->pendingGroupRadiationEnergy_.d_view, 0.0);
        this->pendingGroupRadiationEnergy_.modify_device();
    }

    void DisableSpectral()
    {
        this->spectralEnabled_ = false;
        this->groupCount_ = 0;
        Resize(this->energyBoundaries_, 0);
        Resize(this->spectralAbsorptionScale_, 0);
        Resize(this->thermalEmissionCdf_, 0);
        Resize(this->pendingGroupRadiationEnergy_, 0);
    }

    template<typename PointT>
    void UploadDDMC(const ddmc::HostSnapshot<PointT> &snapshot)
    {
        if(snapshot.cellEligible.size() != this->cellCount_)
        {
            throw std::runtime_error(
                "GreyIMCData::UploadDDMC: cell count mismatch");
        }

        const std::size_t leakCount = snapshot.leakRates.size();

        Resize(this->ddmcCellEligible_, snapshot.cellEligible.size());
        Resize(this->ddmcSigmaEnergyAbs_, snapshot.sigmaEnergyAbs.size());
        Resize(this->ddmcSigmaParticleGate_, snapshot.sigmaParticleGate.size());
        Resize(this->ddmcTotalLeakRate_, snapshot.totalLeakRate.size());
        Resize(this->ddmcLeakOffsets_, snapshot.leakOffsets.size());
        Resize(this->ddmcLeakRates_, leakCount);
        Resize(this->ddmcLeakDDMCRates_, leakCount);
        Resize(this->ddmcNextCellIndices_, leakCount);
        Resize(this->ddmcFaceKinds_, leakCount);
        Resize(this->ddmcTargetEligible_, leakCount);
        Resize(this->ddmcOutwardNormals_, leakCount);
        Resize(this->ddmcFaceCenters_, leakCount);
        Resize(this->ddmcFluxRhs_, snapshot.fluxRhs.size());

        for(std::size_t cellIndex = 0;
            cellIndex < snapshot.cellEligible.size(); ++cellIndex)
        {
            this->ddmcCellEligible_.h_view(cellIndex) =
                snapshot.cellEligible[cellIndex];
            this->ddmcSigmaEnergyAbs_.h_view(cellIndex) =
                snapshot.sigmaEnergyAbs[cellIndex];
            this->ddmcSigmaParticleGate_.h_view(cellIndex) =
                snapshot.sigmaParticleGate[cellIndex];
            this->ddmcTotalLeakRate_.h_view(cellIndex) =
                snapshot.totalLeakRate[cellIndex];
        }
        for(std::size_t offset = 0;
            offset < snapshot.leakOffsets.size(); ++offset)
            this->ddmcLeakOffsets_.h_view(offset) =
                snapshot.leakOffsets[offset];
        for(std::size_t leak = 0; leak < leakCount; ++leak)
        {
            this->ddmcLeakRates_.h_view(leak) =
                snapshot.leakRates[leak];
            this->ddmcLeakDDMCRates_.h_view(leak) =
                snapshot.ddmcLeakRates[leak];
            this->ddmcNextCellIndices_.h_view(leak) =
                snapshot.nextCellIndices[leak];
            this->ddmcFaceKinds_.h_view(leak) =
                snapshot.faceKinds[leak];
            this->ddmcTargetEligible_.h_view(leak) =
                snapshot.targetDDMCEligible[leak];
            this->ddmcOutwardNormals_.h_view(leak) =
                DeviceVec3(snapshot.outwardNormals[leak].x,
                           snapshot.outwardNormals[leak].y,
                           snapshot.outwardNormals[leak].z);
            this->ddmcFaceCenters_.h_view(leak) =
                DeviceVec3(snapshot.faceCenters[leak].x,
                           snapshot.faceCenters[leak].y,
                           snapshot.faceCenters[leak].z);
        }

        SyncToDevice(this->ddmcCellEligible_);
        SyncToDevice(this->ddmcSigmaEnergyAbs_);
        SyncToDevice(this->ddmcSigmaParticleGate_);
        SyncToDevice(this->ddmcTotalLeakRate_);
        SyncToDevice(this->ddmcLeakOffsets_);
        SyncToDevice(this->ddmcLeakRates_);
        SyncToDevice(this->ddmcLeakDDMCRates_);
        SyncToDevice(this->ddmcNextCellIndices_);
        SyncToDevice(this->ddmcFaceKinds_);
        SyncToDevice(this->ddmcTargetEligible_);
        SyncToDevice(this->ddmcOutwardNormals_);
        SyncToDevice(this->ddmcFaceCenters_);
        Kokkos::deep_copy(this->ddmcFluxRhs_.d_view, DeviceVec3{});
        this->ddmcFluxRhs_.modify_device();
        Kokkos::deep_copy(this->ddmcStepCount_, std::size_t(0));
        Kokkos::deep_copy(this->ddmcLeakCount_, std::size_t(0));
        Kokkos::deep_copy(this->ddmcResidentLeakCount_, std::size_t(0));
        Kokkos::deep_copy(this->ddmcTransportLeakCount_, std::size_t(0));
        Kokkos::deep_copy(
            this->ddmcRemoteResidentLeakCount_, std::size_t(0));
        Kokkos::deep_copy(this->ddmcCensusCount_, std::size_t(0));
        this->ddmcMinimumParticleOpticalDepth_ =
            snapshot.minimumParticleOpticalDepth;
        this->ddmcEnabled_ = true;
    }

    void DisableDDMC()
    {
        this->ddmcEnabled_ = false;
        Resize(this->ddmcFluxRhs_, 0);
    }

    void UploadTables(const std::vector<double> &absorptionOpacities,
                      const std::vector<double> &scatteringOpacities,
                      const std::vector<double> &fleckFactors)
    {
        const std::size_t n = absorptionOpacities.size();
        if(scatteringOpacities.size() != n || fleckFactors.size() != n)
        {
            throw std::runtime_error("GreyIMCData::UploadTables: table size mismatch");
        }
        if(this->cellCount_ != 0 && this->cellCount_ != n)
        {
            throw std::runtime_error("GreyIMCData::UploadTables: cell count mismatch");
        }
        this->cellCount_ = n;

        Resize(this->cellTables_, 3 * n);
        if(n > 0)
        {
            double *host = this->cellTables_.h_view.data();
            std::memcpy(host, absorptionOpacities.data(), n * sizeof(double));
            std::memcpy(host + n, scatteringOpacities.data(), n * sizeof(double));
            std::memcpy(host + 2 * n, fleckFactors.data(), n * sizeof(double));
        }
        this->cellTables_.modify_host();
        this->cellTables_.sync_device();

        Resize(this->pendingMaterialEnergy_, n);
        Resize(this->pendingRadiationEnergy_, n);
        auto material = this->pendingMaterialEnergy_.d_view;
        auto radiation = this->pendingRadiationEnergy_.d_view;
        Kokkos::parallel_for(
            "storm_zero_gpu_tallies",
            n,
            KOKKOS_LAMBDA(const std::size_t i)
            {
                material(i) = 0.0;
                radiation(i) = 0.0;
            });
        this->pendingMaterialEnergy_.modify_device();
        this->pendingRadiationEnergy_.modify_device();
    }

    void UploadRandomWalk(
        const std::vector<std::uint8_t> &cellEligible,
        const std::vector<double> &cellTotalOpacity,
        const std::vector<PGRWCellData> &pgrwCellData,
        const RandomWalk &randomWalk,
        const double minimumParticleOpticalDepth)
    {
        if(cellEligible.size() != this->cellCount_ ||
           cellTotalOpacity.size() != this->cellCount_ ||
           (!pgrwCellData.empty() &&
            pgrwCellData.size() != this->cellCount_))
        {
            throw std::runtime_error(
                "GreyIMCData::UploadRandomWalk: cell table size mismatch");
        }

        Resize(this->randomWalkEligible_, cellEligible.size());
        Resize(this->randomWalkTotalOpacity_, cellTotalOpacity.size());
        for(std::size_t i = 0; i < cellEligible.size(); ++i)
        {
            this->randomWalkEligible_.h_view(i) = cellEligible[i];
            this->randomWalkTotalOpacity_.h_view(i) =
                cellTotalOpacity[i];
        }
        SyncToDevice(this->randomWalkEligible_);
        SyncToDevice(this->randomWalkTotalOpacity_);
        Resize(this->randomWalkPGRWCells_, pgrwCellData.size());
        for(std::size_t i = 0; i < pgrwCellData.size(); ++i)
        {
            const PGRWCellData &source = pgrwCellData[i];
            PGRWCellData &destination =
                this->randomWalkPGRWCells_.h_view(i);
            destination = source;
        }
        SyncToDevice(this->randomWalkPGRWCells_);

        CopyToDevice(randomWalk.GetTauTable(), this->randomWalkTau_);
        CopyToDevice(
            randomWalk.GetSurvivalTable(), this->randomWalkSurvival_);
        CopyToDevice(
            randomWalk.GetRadiusTable(), this->randomWalkRadius_);

        this->randomWalkMinimumTau_ = RandomWalk::GetMinimumTau();
        this->randomWalkMaximumTau_ = RandomWalk::GetMaximumTau();
        this->randomWalkRadiusTableSize_ =
            RandomWalk::GetRadiusTableSize();
        this->randomWalkMinimumParticleOpticalDepth_ =
            minimumParticleOpticalDepth;
        this->randomWalkEnabled_ = true;
        Kokkos::deep_copy(
            this->randomWalkStepCounter_, std::size_t(0));
    }

    void DisableRandomWalk()
    {
        this->randomWalkEnabled_ = false;
        Resize(this->randomWalkPGRWCells_, 0);
        Kokkos::deep_copy(
            this->randomWalkStepCounter_, std::size_t(0));
    }

    GreyIMCViews<DeviceVec3> Views(
        double speedOfLight,
        bool depositMaterialEnergy,
        bool comovingTransport,
        bool depositMomentum) const
    {
        GreyIMCViews<DeviceVec3> result;
        result.grid.cellFaceOffsets = this->cellFaceOffsets_.d_view.data();
        result.grid.cellCenters = this->cellCenters_.d_view.data();
        result.grid.normals = this->normals_.d_view.data();
        result.grid.facePlaneOffsets = this->facePlaneOffsets_.d_view.data();
        result.grid.nextCellIndices = this->nextCellIndices_.d_view.data();
        result.grid.boundaryCrossings = this->boundaryCrossings_.d_view.data();
        result.grid.deviceBoundaryBehaviors =
            this->deviceBoundaryBehaviors_.d_view.data();
        result.grid.cellCount = this->cellCount_;
        result.absorptionOpacities = this->cellTables_.d_view.data();
        result.scatteringOpacities =
            this->cellCount_ > 0 ? this->cellTables_.d_view.data() + this->cellCount_ : nullptr;
        result.fleckFactors =
            this->cellCount_ > 0 ? this->cellTables_.d_view.data() + 2 * this->cellCount_ : nullptr;
        result.cellVelocities =
            this->cellVelocities_.d_view.data();
        result.pendingMaterialEnergy = this->pendingMaterialEnergy_.d_view.data();
        result.pendingRadiationEnergy = this->pendingRadiationEnergy_.d_view.data();
        result.pendingMomentum =
            this->pendingMomentum_.d_view.data();
        result.ddmc.cellEligible =
            this->ddmcCellEligible_.d_view.data();
        result.ddmc.sigmaEnergyAbs =
            this->ddmcSigmaEnergyAbs_.d_view.data();
        result.ddmc.sigmaParticleGate =
            this->ddmcSigmaParticleGate_.d_view.data();
        result.ddmc.totalLeakRate =
            this->ddmcTotalLeakRate_.d_view.data();
        result.ddmc.leakOffsets =
            this->ddmcLeakOffsets_.d_view.data();
        result.ddmc.leakRates =
            this->ddmcLeakRates_.d_view.data();
        result.ddmc.ddmcLeakRates =
            this->ddmcLeakDDMCRates_.d_view.data();
        result.ddmc.nextCellIndices =
            this->ddmcNextCellIndices_.d_view.data();
        result.ddmc.faceKinds =
            this->ddmcFaceKinds_.d_view.data();
        result.ddmc.targetDDMCEligible =
            this->ddmcTargetEligible_.d_view.data();
        result.ddmc.outwardNormals =
            this->ddmcOutwardNormals_.d_view.data();
        result.ddmc.faceCenters =
            this->ddmcFaceCenters_.d_view.data();
        result.ddmc.fluxRhs = this->ddmcFluxRhs_.d_view.data();
        result.ddmc.stepCount = this->ddmcStepCount_.data();
        result.ddmc.leakCount = this->ddmcLeakCount_.data();
        result.ddmc.residentLeakCount =
            this->ddmcResidentLeakCount_.data();
        result.ddmc.transportLeakCount =
            this->ddmcTransportLeakCount_.data();
        result.ddmc.remoteResidentLeakCount =
            this->ddmcRemoteResidentLeakCount_.data();
        result.ddmc.censusCount = this->ddmcCensusCount_.data();
        result.ddmc.cellCount = this->cellCount_;
        result.ddmc.minimumParticleOpticalDepth =
            this->ddmcMinimumParticleOpticalDepth_;
        result.ddmc.enabled = this->ddmcEnabled_;
        result.energyBoundaries =
            this->energyBoundaries_.d_view.data();
        result.spectralAbsorptionScale =
            this->spectralAbsorptionScale_.d_view.data();
        result.thermalEmissionCdf =
            this->thermalEmissionCdf_.d_view.data();
        result.pendingGroupRadiationEnergy =
            this->pendingGroupRadiationEnergy_.d_view.data();
        result.groupCount = this->groupCount_;
        result.randomWalk.cellEligible =
            this->randomWalkEligible_.d_view.data();
        result.randomWalk.cellTotalOpacity =
            this->randomWalkTotalOpacity_.d_view.data();
        result.randomWalk.pgrwCells =
            this->randomWalkPGRWCells_.d_view.data();
        result.randomWalk.tables.tau =
            this->randomWalkTau_.d_view.data();
        result.randomWalk.tables.survival =
            this->randomWalkSurvival_.d_view.data();
        result.randomWalk.tables.radius =
            this->randomWalkRadius_.d_view.data();
        result.randomWalk.tables.tableSize =
            this->randomWalkTau_.extent(0);
        result.randomWalk.tables.radiusTableSize =
            this->randomWalkRadiusTableSize_;
        result.randomWalk.tables.tauMin =
            this->randomWalkMinimumTau_;
        result.randomWalk.tables.tauMax =
            this->randomWalkMaximumTau_;
        result.randomWalk.stepCounter =
            this->randomWalkStepCounter_.data();
        result.randomWalk.minimumParticleOpticalDepth =
            this->randomWalkMinimumParticleOpticalDepth_;
        result.randomWalk.enabled = this->randomWalkEnabled_;
        result.randomWalk.spectralEnabled =
            this->spectralEnabled_ &&
            this->randomWalkPGRWCells_.extent(0) ==
                this->cellCount_;
        result.speedOfLight = speedOfLight;
        result.depositMaterialEnergy = depositMaterialEnergy;
        result.comovingTransport = comovingTransport;
        result.depositMomentum = depositMomentum;
        result.spectralEnabled = this->spectralEnabled_;
        return result;
    }

    template<typename PointT>
    void AddTallies(std::vector<double> &materialEnergy,
                    std::vector<double> &radiationEnergy,
                    std::vector<double> &groupRadiationEnergy,
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
        this->pendingMaterialEnergy_.sync_host();
        this->pendingRadiationEnergy_.sync_host();
        if(this->pendingGroupRadiationEnergy_.extent(0) > 0)
        {
            this->pendingGroupRadiationEnergy_.sync_host();
        }
        if(this->pendingMomentum_.extent(0) > 0)
        {
            this->pendingMomentum_.sync_host();
        }
        if(this->ddmcFluxRhs_.extent(0) > 0)
        {
            this->ddmcFluxRhs_.sync_host();
        }
        for(std::size_t i = 0; i < this->cellCount_; i++)
        {
            materialEnergy[i] += this->pendingMaterialEnergy_.h_view(i);
            radiationEnergy[i] += this->pendingRadiationEnergy_.h_view(i);
            if((i + 1) * this->groupCount_ <=
               groupRadiationEnergy.size())
            {
                for(std::size_t group = 0; group < this->groupCount_; group++)
                {
                    groupRadiationEnergy[
                        i * this->groupCount_ + group] +=
                        this->pendingGroupRadiationEnergy_.h_view(
                            i * this->groupCount_ + group);
                }
            }
            if(i < momentum.size() and i < this->pendingMomentum_.extent(0))
            {
                momentum[i].x += this->pendingMomentum_.h_view(i).x;
                momentum[i].y += this->pendingMomentum_.h_view(i).y;
                momentum[i].z += this->pendingMomentum_.h_view(i).z;
            }
            if(i < ddmcFluxRhs.size() &&
               i < this->ddmcFluxRhs_.extent(0))
            {
                ddmcFluxRhs[i].x += this->ddmcFluxRhs_.h_view(i).x;
                ddmcFluxRhs[i].y += this->ddmcFluxRhs_.h_view(i).y;
                ddmcFluxRhs[i].z += this->ddmcFluxRhs_.h_view(i).z;
            }
        }
        std::size_t deviceRandomWalkSteps = 0;
        Kokkos::deep_copy(deviceRandomWalkSteps, this->randomWalkStepCounter_);
        randomWalkSteps += deviceRandomWalkSteps;
        std::size_t count = 0;
        Kokkos::deep_copy(count, this->ddmcStepCount_);
        ddmcSteps += count;
        Kokkos::deep_copy(count, this->ddmcLeakCount_);
        ddmcLeaks += count;
        Kokkos::deep_copy(count, this->ddmcResidentLeakCount_);
        ddmcResidentLeaks += count;
        Kokkos::deep_copy(count, this->ddmcTransportLeakCount_);
        ddmcTransportLeaks += count;
        Kokkos::deep_copy(count, this->ddmcRemoteResidentLeakCount_);
        ddmcRemoteResidentLeaks += count;
        Kokkos::deep_copy(count, this->ddmcCensusCount_);
        ddmcCensus += count;
    }

private:
    template<typename T>
    static void Resize(Kokkos::DualView<T*> &view, std::size_t size)
    {
        if(view.extent(0) != size)
        {
            Kokkos::resize(view, size);
        }
    }

    template<typename T>
    static void SyncToDevice(Kokkos::DualView<T*> &view)
    {
        view.modify_host();
        view.sync_device();
    }

    template<typename T>
    static void CopyToDevice(const std::vector<T> &source, Kokkos::DualView<T*> &destination)
    {
        Resize(destination, source.size());
        for(std::size_t i = 0; i < source.size(); ++i)
        {
            destination.h_view(i) = source[i];
        }
        SyncToDevice(destination);
    }

    Kokkos::DualView<std::size_t*> cellFaceOffsets_;
    Kokkos::DualView<DeviceVec3*> cellCenters_;
    Kokkos::DualView<DeviceVec3*> normals_;
    Kokkos::DualView<double*> facePlaneOffsets_;
    Kokkos::DualView<cell_index_t*> nextCellIndices_;
    Kokkos::DualView<std::uint8_t*> boundaryCrossings_;
    Kokkos::DualView<std::uint8_t*> deviceBoundaryBehaviors_;
    /// Packed [absorption | scattering | fleck], each of length cellCount_.
    Kokkos::DualView<double*> cellTables_;
    Kokkos::DualView<DeviceVec3*> cellVelocities_;
    Kokkos::DualView<double*> pendingMaterialEnergy_;
    Kokkos::DualView<double*> pendingRadiationEnergy_;
    Kokkos::DualView<DeviceVec3*> pendingMomentum_;
    Kokkos::DualView<double*> pendingGroupRadiationEnergy_;
    Kokkos::DualView<double*> energyBoundaries_;
    Kokkos::DualView<double*> spectralAbsorptionScale_;
    Kokkos::DualView<double*> thermalEmissionCdf_;
    Kokkos::DualView<std::uint8_t*> ddmcCellEligible_;
    Kokkos::DualView<double*> ddmcSigmaEnergyAbs_;
    Kokkos::DualView<double*> ddmcSigmaParticleGate_;
    Kokkos::DualView<double*> ddmcTotalLeakRate_;
    Kokkos::DualView<std::size_t*> ddmcLeakOffsets_;
    Kokkos::DualView<double*> ddmcLeakRates_;
    Kokkos::DualView<double*> ddmcLeakDDMCRates_;
    Kokkos::DualView<cell_index_t*> ddmcNextCellIndices_;
    Kokkos::DualView<std::uint8_t*> ddmcFaceKinds_;
    Kokkos::DualView<std::uint8_t*> ddmcTargetEligible_;
    Kokkos::DualView<DeviceVec3*> ddmcOutwardNormals_;
    Kokkos::DualView<DeviceVec3*> ddmcFaceCenters_;
    Kokkos::DualView<DeviceVec3*> ddmcFluxRhs_;
    Kokkos::View<std::size_t> ddmcStepCount_;
    Kokkos::View<std::size_t> ddmcLeakCount_;
    Kokkos::View<std::size_t> ddmcResidentLeakCount_;
    Kokkos::View<std::size_t> ddmcTransportLeakCount_;
    Kokkos::View<std::size_t> ddmcRemoteResidentLeakCount_;
    Kokkos::View<std::size_t> ddmcCensusCount_;
    Kokkos::DualView<std::uint8_t*> randomWalkEligible_;
    Kokkos::DualView<double*> randomWalkTotalOpacity_;
    Kokkos::DualView<PGRWCellData*> randomWalkPGRWCells_;
    Kokkos::DualView<double*> randomWalkTau_;
    Kokkos::DualView<double*> randomWalkSurvival_;
    Kokkos::DualView<double*> randomWalkRadius_;
    Kokkos::View<std::size_t> randomWalkStepCounter_;
    double randomWalkMinimumTau_ = 0.0;
    double randomWalkMaximumTau_ = 0.0;
    double randomWalkMinimumParticleOpticalDepth_ = 0.0;
    std::size_t randomWalkRadiusTableSize_ = 0;
    bool randomWalkEnabled_ = false;
    bool spectralEnabled_ = false;
    bool ddmcEnabled_ = false;
    double ddmcMinimumParticleOpticalDepth_ = 0.0;
    std::size_t groupCount_ = 0;
    std::size_t cellCount_ = 0;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_IMC_DATA_HPP
