#ifndef STORM_RADIATION_IMCOBSERVER_PROCESS_HPP
#define STORM_RADIATION_IMCOBSERVER_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"
#ifdef STORM_WITH_MPI
#include <mpi_utils/mpi_collectives.hpp>
#endif

namespace STORM::radiation_imc_detail {

template<typename Owner>
class IMCObserverProcess final : public IMCComponentBase<Owner>
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
    explicit IMCObserverProcess(Owner &owner) : Base(owner)
    {}

    void setPostProcessExternalSources(std::vector<PostProcessExternalSource> sources)
    {
        if(!owner_.parameters_.postProcess.enabled)
        {
            throw StormError("External sources require RadiationIMC post-process mode");
        }
        if(owner_.parameters_.withRandomWalk)
        {
            throw StormError("External source surfaces require random-walk acceleration to be disabled");
        }

        std::size_t const localCellCount = owner_.componentGrid().GetPointNo();
        if(owner_.cells_.size() < localCellCount)
        {
            throw StormError("External source installation has fewer cells than local tessellation points");
        }

        std::size_t const invalidCellID = std::numeric_limits<std::size_t>::max();
        std::size_t const pointCount = std::max(owner_.componentGrid().GetTotalPointNumber(), owner_.componentGrid().getMeshPoints().size());
        std::vector<std::size_t> pointCellIDs(pointCount, invalidCellID);
        for(std::size_t cellIndex = 0; cellIndex < localCellCount; ++cellIndex)
        {
            pointCellIDs[cellIndex] = radiation_imc_detail::ddmcStableCellID(owner_.componentGrid(), cellIndex, owner_.cells_[cellIndex]);
        }
    #ifdef STORM_WITH_MPI
        STORM::MPI_exchange_data(owner_.componentGrid(), pointCellIDs, true);
    #endif

        std::unordered_map<std::size_t, std::size_t> localCellIndexByID;
        localCellIndexByID.reserve(localCellCount);
        for(std::size_t cellIndex = 0; cellIndex < localCellCount; ++cellIndex)
        {
            std::size_t const cellID = pointCellIDs[cellIndex];
            if(cellID == invalidCellID or not localCellIndexByID.emplace(cellID, cellIndex).second)
            {
                throw StormError("External source installation requires unique stable cell IDs");
            }
        }

        std::unordered_map<std::size_t, std::size_t> faceIndex;
        faceIndex.reserve(sources.size());
        std::vector<std::size_t> localSourceCellIndices(sources.size(), invalidCellID);
        std::unordered_set<std::size_t> localInteriorIDSet;
        localInteriorIDSet.reserve(sources.size());
        for(std::size_t sourceIndex = 0; sourceIndex < sources.size(); sourceIndex++)
        {
            PostProcessExternalSource &source = sources[sourceIndex];
            if(source.faceIndex == invalidCellID ||
                source.cellID == invalidCellID ||
                source.interiorCellID == invalidCellID ||
                source.cellID == source.interiorCellID ||
                !(source.luminosity >= 0.0) ||
                !std::isfinite(source.luminosity) ||
                !std::isfinite(source.location[0]) ||
                !std::isfinite(source.location[1]) ||
                !std::isfinite(source.location[2]))
            {
                throw StormError("External source face has invalid data");
            }
            double const normalMagnitude = fastabs(source.outwardNormal);
            if(not (normalMagnitude > 0.0) or not std::isfinite(normalMagnitude))
            {
                throw StormError("External source face has an invalid normal");
            }
            source.outwardNormal = source.outwardNormal / normalMagnitude;

            auto const cellIt = localCellIndexByID.find(source.cellID);
            if(cellIt == localCellIndexByID.end())
            {
                throw StormError("External source references a non-local transport cell");
            }
            std::size_t const cellIndex = cellIt->second;
            localSourceCellIndices[sourceIndex] = cellIndex;
            auto const &cellFaces = owner_.componentGrid().GetCellFaces(cellIndex);
            if(std::find(cellFaces.begin(), cellFaces.end(), source.faceIndex) == cellFaces.end())
            {
                throw StormError(
                    "External source face is not attached to its transport cell");
            }
            std::pair<std::size_t, std::size_t> const neighbors = owner_.componentGrid().GetFaceNeighbors(source.faceIndex);
            if(neighbors.first != cellIndex and neighbors.second != cellIndex)
            {
                throw StormError("External source face does not contain its transport cell");
            }
            std::size_t const interiorCellIndex = (neighbors.first == cellIndex)? neighbors.second : neighbors.first;
            if(owner_.componentGrid().IsPointOutsideBox(interiorCellIndex) or interiorCellIndex >= pointCellIDs.size() or
                pointCellIDs[interiorCellIndex] != source.interiorCellID)
            {
                throw StormError("External source interior-cell ID does not match the opposite face neighbor");
            }
            if(not faceIndex.emplace(source.faceIndex, sourceIndex).second)
            {
                throw StormError("Duplicate external source face");
            }
            localInteriorIDSet.insert(source.interiorCellID);
        }

        std::unordered_set<std::size_t> globalInteriorIDs = localInteriorIDSet;
    #ifdef STORM_WITH_MPI
        if(owner_.parameters_.withDDMC)
        {
            std::vector<std::uint64_t> localInteriorIDs;
            localInteriorIDs.reserve(localInteriorIDSet.size());
            for(std::size_t id : localInteriorIDSet)
            {
                localInteriorIDs.push_back(static_cast<std::uint64_t>(id));
            }
            std::vector<std::uint64_t> allInteriorIDs = MPI_All_cast(localInteriorIDs, MPI_COMM_WORLD);
            globalInteriorIDs.clear();
            globalInteriorIDs.reserve(allInteriorIDs.size());
            for(std::uint64_t id : allInteriorIDs)
            {
                globalInteriorIDs.insert(static_cast<std::size_t>(id));
            }
        }
    #endif

        owner_.postProcessExternalSources_ = std::move(sources);
        owner_.postProcessExternalSourceLocalCellIndices_ = std::move(localSourceCellIndices);
        owner_.postProcessExternalSourceFaceIndex_ = std::move(faceIndex);
        owner_.postProcessExternalSourceInteriorCellIDs_ = std::move(globalInteriorIDs);
        owner_.postProcessExternalSourceMode_ = true;
    }

    void clearPostProcessExternalSources()
    {
        owner_.postProcessExternalSources_.clear();
        owner_.postProcessExternalSourceLocalCellIndices_.clear();
        owner_.postProcessExternalSourceFaceIndex_.clear();
        owner_.postProcessExternalSourceInteriorCellIDs_.clear();
        owner_.postProcessExternalSourceMode_ = false;
    }

    PointT samplePostProcessExternalSourceDirection(const PointT &outwardNormal, MCParticle &particle)
    {
        PointT normal = outwardNormal;
        double const normalMagnitude = fastabs(normal);
        if(not (normalMagnitude > 0.0) or not std::isfinite(normalMagnitude))
        {
            throw StormError("External source face has an invalid normal");
        }
        normal = normal / normalMagnitude;
        PointT helper = (std::abs(normal.z) < 0.9)? PointT(0.0, 0.0, 1.0) : PointT(0.0, 1.0, 0.0);
        PointT tangent1 = helper - ScalarProd(helper, normal) * normal;
        double const tangentMagnitude = fastabs(tangent1);
        if(not (tangentMagnitude > 0.0) or not std::isfinite(tangentMagnitude))
        {
            throw StormError("External source face cannot construct a tangent basis");
        }
        tangent1 = tangent1 / tangentMagnitude;
        PointT tangent2 = CrossProduct(normal, tangent1);
        tangent2 = tangent2 / std::max(fastabs(tangent2), std::numeric_limits<double>::min());
        double const mu = std::sqrt(owner_.randomUnitOpen(particle));
        double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        double const phi = 2.0 * 3.14159265358979323846 * owner_.randomUnitOpen(particle);
        return mu * normal + sinTheta * (std::cos(phi) * tangent1 + std::sin(phi) * tangent2);
    }

    typename Owner::GroupArray buildPostProcessExternalSourcePlanckPdf(const CellT &cell) const
    {
        GroupArray pdf{};
        double const kT = units::k_boltz * cell.temperature;
        if(not (kT > 0.0) or not std::isfinite(kT))
        {
            throw StormError("External source Planck spectrum requires positive finite temperature");
        }
        double total = 0.0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const left = owner_.energyBoundaries_[group];
            double const right = owner_.energyBoundaries_[group + 1];
            double const mass = (left < right)? planck_integral::planck_integral(left / kT, right / kT) : 0.0;
            pdf[group] = (mass > 0.0 and std::isfinite(mass))? mass : 0.0;
            total += pdf[group];
        }
        if(total > 0.0 and std::isfinite(total))
        {
            for(double &value : pdf)
            {
                value /= total;
            }
            return pdf;
        }
        double const peakEnergy = 2.8214393721220789 * kT;
        std::size_t fallbackGroup = 0;
        while(fallbackGroup + 1 < NumGroups and peakEnergy >= owner_.energyBoundaries_[fallbackGroup + 1])
        {
            ++fallbackGroup;
        }
        pdf[fallbackGroup] = 1.0;
        return pdf;
    }

    double samplePostProcessExternalSourcePlanckFrequencyInGroup(const CellT &cell, std::size_t group)
    {
        group = std::min(group, NumGroups - 1);
        double const left = owner_.energyBoundaries_[group];
        double const right = owner_.energyBoundaries_[group + 1];
        double const kT = units::k_boltz * cell.temperature;
        double const groupMass = planck_integral::planck_integral(left / kT, right / kT);
        if(not (groupMass > 0.0) or not std::isfinite(groupMass))
        {
            return 0.5 * (left + right);
        }
        double const target = owner_.randomUnitOpen() * groupMass;
        double lo = left;
        double hi = right;
        for(int iteration = 0; iteration < 56; ++iteration)
        {
            double const mid = 0.5 * (lo + hi);
            double const mass = planck_integral::planck_integral(left / kT, mid / kT);
            if(mass < target)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        double frequency = 0.5 * (lo + hi);
        owner_.clampFrequencyToBounds(frequency);
        return frequency;
    }

    double samplePostProcessExternalSourcePlanckFrequency(const CellT &cell)
    {
        GroupArray const pdf = owner_.buildPostProcessExternalSourcePlanckPdf(cell);
        double const target = owner_.randomUnitOpen();
        double cumulative = 0.0;
        std::size_t selectedGroup = NumGroups - 1;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            cumulative += pdf[group];
            if(target <= cumulative)
            {
                selectedGroup = group;
                break;
            }
        }
        return owner_.samplePostProcessExternalSourcePlanckFrequencyInGroup(cell, selectedGroup);
    }

    typename Owner::MCParticle generatePostProcessExternalSourceParticle(std::size_t cellIndex, const CellT &cell, const PostProcessExternalSource &source)
    {
        MCParticle particle;
        owner_.initializeParticleRNG(particle);
        particle.id = std::numeric_limits<std::size_t>::max();
        particle.cellIndex = cellIndex;
        particle.velocity = units::clight * owner_.samplePostProcessExternalSourceDirection(source.outwardNormal, particle);
        static constexpr double nudge = 1.0e-8;
        particle.location = (1.0 - nudge) * source.location + nudge * owner_.componentGrid().GetMeshPoint(cellIndex);
        if(!owner_.componentGrid().IsPointInCell(particle.location, cellIndex))
        {
            throw StormError("External source face location did not nudge into its transport cell");
        }
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((owner_.parameters_.withHydro and !owner_.parameters_.MMC) or
                (owner_.parameters_.postProcess.enabled and
                owner_.parameters_.postProcess.useCellVelocities))
            {
                radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
            }
        }
        return particle;
    }

    bool handlePostProcessExternalSourceBoundary(MCParticle &particle, std::size_t cellIndex, std::size_t faceIndex, Functionality &functionality)
    {
        if(not owner_.postProcessExternalSourceMode_ or cellIndex >= owner_.cells_.size())
        {
            return false;
        }
        auto const faceIt = owner_.postProcessExternalSourceFaceIndex_.find(faceIndex);
        if(faceIt == owner_.postProcessExternalSourceFaceIndex_.end())
        {
            return false;
        }
        PostProcessExternalSource const &source = owner_.postProcessExternalSources_[faceIt->second];
        if(radiation_imc_detail::cellID(owner_.cells_[cellIndex]) != source.cellID)
        {
            return false;
        }

        PointT normal = source.outwardNormal / std::max(fastabs(source.outwardNormal), std::numeric_limits<double>::min());
        double const normalVelocity = ScalarProd(particle.velocity, normal);
        double const directionTolerance = 1.0e-12 * std::max(fastabs(particle.velocity), 1.0);
        if(normalVelocity > directionTolerance)
        {
            throw StormError("Packet reached the external-source face while moving away from the interior");
        }

        MCParticle materialParticle = particle;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((owner_.parameters_.withHydro and !owner_.parameters_.MMC) or
                (owner_.parameters_.postProcess.enabled and owner_.parameters_.postProcess.useCellVelocities))
            {
                radiation_imc_detail::lorentzTransformToComoving<PointT>(materialParticle, owner_.cells_[cellIndex]);
            }
        }
        materialParticle.velocity = units::clight * owner_.samplePostProcessExternalSourceDirection(normal, particle);
        if(owner_.parameters_.withMultigroupOpacity)
        {
            materialParticle.frequency = owner_.samplePostProcessExternalSourcePlanckFrequency(owner_.cells_[cellIndex]);
        }
    #ifdef MONTECARLO_POLARIZATION
        if(owner_.polarizationEnabled())
        {
            materialParticle.stokesQ = 0.0;
            materialParticle.stokesU = 0.0;
            materialParticle.polarizationInitialized = false;
        }
    #endif
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((owner_.parameters_.withHydro and !owner_.parameters_.MMC) or
                (owner_.parameters_.postProcess.enabled and owner_.parameters_.postProcess.useCellVelocities))
            {
                radiation_imc_detail::lorentzTransformToLab<PointT>(materialParticle, owner_.cells_[cellIndex]);
                owner_.clampFrequencyToBounds(materialParticle.frequency);
            }
        }
        static constexpr double nudge = 1.0e-8;
        materialParticle.location = (1.0 - nudge) * source.location + nudge * owner_.componentGrid().GetMeshPoint(cellIndex);
        materialParticle.initialWeight = std::abs(materialParticle.weight);
        materialParticle.radiationState.clearDDMC();
        particle = materialParticle;
        functionality.change = ParticleStatus::NO_CELL_MOVE;
        return true;
    }

    void recordObserverCrossing(const MCParticle &particle, const PointT &crossingPoint)
    {
        if(not owner_.observer_)
        {
            return;
        }
        ObserverCrossingRecord<PointT> record;
        record.crossingPoint = crossingPoint;
        record.direction = particle.velocity;
        record.weight = particle.weight;
        record.frequency = particle.frequency;
        record.sourceCellID = particle.sourceCellID;
    #ifdef MONTECARLO_POLARIZATION
        if(owner_.polarizationEnabled())
        {
            record.stokesQ = particle.stokesQ;
            record.stokesU = particle.stokesU;
            record.polarizationBasis = particle.polarizationInitialized
                ? polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity)
                : polarization::choosePerpendicularBasis(particle.velocity);
            record.polarizationInitialized = true;
            polarization::clampLinearPolarization(record.stokesQ, record.stokesU);
        }
    #endif
        owner_.observer_->recordCrossing(record);
    }

    void onBoundaryResult(const MCParticle &particle, ParticleStatus status, bool escaped)
    {
        if(escaped and status == ParticleStatus::REMOVE and owner_.observer_)
        {
            owner_.observer_->addBoxEscapeEnergy(particle.weight);
        }
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMCOBSERVER_PROCESS_HPP
