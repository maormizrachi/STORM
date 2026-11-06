#ifndef STORM_RADIATION_IMC_COMPONENT_BASE_HPP
#define STORM_RADIATION_IMC_COMPONENT_BASE_HPP

#include <cstddef>
#include <memory>

#include "../../StormError.hpp"

namespace STORM::radiation_imc_detail {

// Replaces a packet by an edited copy of itself. Draws taken during the edit
// must come from the copy: if the packet's counter-based stream has moved past
// the copy's (same key), the assignment would roll the stream back and the
// next draws would reuse numbers that already set the packet's state (the face
// re-emission once did this). A copy on another key (a fresh stream) is fine.
template<typename Packet>
void replaceWithEditedCopy(Packet &particle, const Packet &edited)
{
    if(edited.rngKey == particle.rngKey && edited.rngCounter < particle.rngCounter)
    {
        StormError eo("Replacing a packet by an edited copy would roll its RNG stream back");
        eo.addEntry("Counter of the packet", particle.rngCounter);
        eo.addEntry("Counter of the copy", edited.rngCounter);
        eo.addEntry("Particle", particle);
        throw eo;
    }
    particle = edited;
}

// Common non-owning handle used by the composed IMC processes.  The owner is
// deliberately opaque here: concrete processes receive the narrow aliases
// they need, while RadiationIMC remains the only object exposing the public
// MonteCarloPhysics interface.
template<typename Owner>
class IMCComponentBase
{
protected:
    using OwnerType = Owner;
    using PointT = typename Owner::PointType;
    using GridT = typename Owner::GridType;
    using CellT = typename Owner::CellType;
    using ExtensivesT = typename Owner::ExtensivesType;
    using EOST = typename Owner::EOSType;
    using OpacityT = typename Owner::OpacityType;
    using TraitsT = typename Owner::TraitsType;
    using PositionSamplerT = typename Owner::PositionSamplerType;
    using Parameters = typename Owner::Parameters;
    using MCParticle = typename Owner::MCParticle;
    using Functionality = typename Owner::Functionality;
    using BoundaryCond = typename Owner::BoundaryCond;
    using PositionDecomposition = typename Owner::PositionDecomposition;
    using GroupArray = typename Owner::GroupArray;
    using GroupBoundaries = typename Owner::GroupBoundaries;
    using GroupCdf = typename Owner::GroupCdf;
    using GroupMatrix = typename Owner::GroupMatrix;
    using GroupCdfMatrix = typename Owner::GroupCdfMatrix;
    using ComptonCellData = typename Owner::ComptonCellData;
    using Observer = typename Owner::Observer;
    using DDMCCellData = typename Owner::DDMCCellData;
    using DDMCFaceLeak = typename Owner::DDMCFaceLeak;
    using SourceAllocationSummary = typename Owner::SourceAllocationSummary;
    using GroupSamplingDiagnostics = typename Owner::GroupSamplingDiagnostics;
    using PostProcessExternalSource = typename Owner::PostProcessExternalSource;
    using ComptonProjectionResult = typename Owner::ComptonProjectionResult;
    using ComptonCorrectionResult = typename Owner::ComptonCorrectionResult;
    using ComptonCorrectionFailure = typename Owner::ComptonCorrectionFailure;
    static constexpr std::size_t NumGroups = Owner::kNumGroups;
    static constexpr bool kSamplerHasDecomposition = Owner::kSamplerHasDecomposition;

    explicit IMCComponentBase(Owner &owner) : owner_(owner)
    {}

    static void replaceWithEditedCopy(MCParticle &particle, const MCParticle &edited)
    {
        radiation_imc_detail::replaceWithEditedCopy<MCParticle>(particle, edited);
    }

    Owner &owner_;
};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_COMPONENT_BASE_HPP
