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
          pendingMaterialEnergy_("storm_material_tally", 0),
          pendingRadiationEnergy_("storm_radiation_tally", 0),
          randomWalkEligible_("storm_rw_eligible", 0),
          randomWalkTotalOpacity_("storm_rw_total_opacity", 0),
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
        if(facePlaneOffsets.size() != directedFaceCount ||
           nextCellIndices.size() != directedFaceCount ||
           boundaryCrossings.size() != directedFaceCount ||
           deviceBoundaryBehaviors.size() != directedFaceCount)
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
        const RandomWalk &randomWalk,
        const double minimumParticleOpticalDepth)
    {
        if(cellEligible.size() != this->cellCount_ ||
           cellTotalOpacity.size() != this->cellCount_)
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
        Kokkos::deep_copy(
            this->randomWalkStepCounter_, std::size_t(0));
    }

    GreyIMCViews<DeviceVec3> Views(double speedOfLight, bool depositMaterialEnergy) const
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
        result.pendingMaterialEnergy = this->pendingMaterialEnergy_.d_view.data();
        result.pendingRadiationEnergy = this->pendingRadiationEnergy_.d_view.data();
        result.randomWalk.cellEligible =
            this->randomWalkEligible_.d_view.data();
        result.randomWalk.cellTotalOpacity =
            this->randomWalkTotalOpacity_.d_view.data();
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
        result.speedOfLight = speedOfLight;
        result.depositMaterialEnergy = depositMaterialEnergy;
        return result;
    }

    void AddTallies(std::vector<double> &materialEnergy,
                    std::vector<double> &radiationEnergy,
                    std::size_t &randomWalkSteps)
    {
        this->pendingMaterialEnergy_.sync_host();
        this->pendingRadiationEnergy_.sync_host();
        for(std::size_t i = 0; i < this->cellCount_; ++i)
        {
            materialEnergy[i] += this->pendingMaterialEnergy_.h_view(i);
            radiationEnergy[i] += this->pendingRadiationEnergy_.h_view(i);
        }
        std::size_t deviceRandomWalkSteps = 0;
        Kokkos::deep_copy(
            deviceRandomWalkSteps, this->randomWalkStepCounter_);
        randomWalkSteps += deviceRandomWalkSteps;
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
    static void CopyToDevice(const std::vector<T> &source,
                             Kokkos::DualView<T*> &destination)
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
    Kokkos::DualView<double*> pendingMaterialEnergy_;
    Kokkos::DualView<double*> pendingRadiationEnergy_;
    Kokkos::DualView<std::uint8_t*> randomWalkEligible_;
    Kokkos::DualView<double*> randomWalkTotalOpacity_;
    Kokkos::DualView<double*> randomWalkTau_;
    Kokkos::DualView<double*> randomWalkSurvival_;
    Kokkos::DualView<double*> randomWalkRadius_;
    Kokkos::View<std::size_t> randomWalkStepCounter_;
    double randomWalkMinimumTau_ = 0.0;
    double randomWalkMaximumTau_ = 0.0;
    double randomWalkMinimumParticleOpticalDepth_ = 0.0;
    std::size_t randomWalkRadiusTableSize_ = 0;
    bool randomWalkEnabled_ = false;
    std::size_t cellCount_ = 0;
};

} // namespace gpu
} // namespace STORM

#endif // STORM_GPU_GREY_IMC_DATA_HPP
