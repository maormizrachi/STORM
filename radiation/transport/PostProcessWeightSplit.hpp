#ifndef STORM_RADIATION_POSTPROCESS_WEIGHT_SPLIT_HPP
#define STORM_RADIATION_POSTPROCESS_WEIGHT_SPLIT_HPP

// Post-process weight window (importance splitting) as a function of the
// packet and the installed window, independent of the transport object so it
// can be tested directly. RadiationIMC::setPostProcessWeightSplit installs the
// scalar window, setPostProcessGroupWindow the per-group windows and the
// provenance candidates; IMCTransportProcess::step calls apply() before every
// step.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../../StormError.hpp"
#include "../../particle/PacketProvenance.hpp"

namespace STORM
{
namespace postprocess_split
{
// Per-group window on top of the scalar window. With `active`, a packet in
// lab-frame group g that starts an IMC step in local cell c splits when
// |w| > maxWeight[g] / I_eff(c), I_eff = max(importance[c], importanceFloor)
// for the cells that have an importance entry; the scalar window is then
// unused. With `provenance`, the shadow candidate windows only set packet
// provenance bits (PacketProvenance.hpp); they never split.
template<std::size_t NumGroups>
struct GroupWindow
{
    bool active = false;
    std::array<double, NumGroups> maxWeight{};
    double importanceFloor = 0.0;
    bool provenance = false;
    bool shadowValid = false;
    std::array<std::array<double, NumGroups>, provenance::kCandidates> shadowMaxWeight{};
    // Derived by finalize(): smallest active / shadow window.
    double minMaxWeight = 0.0;
    double shadowMinMaxWeight = 0.0;
};

// Since the last installation of the scalar window: split events, copies,
// events capped at maxCopies, events the copy budget reduced or refused, and
// steps a packet heavier than the smallest window took without a split
// opportunity (DDMC resident or pending flux; cell without importance).
struct ScalarCounters
{
    std::size_t events = 0;
    std::size_t copies = 0;
    std::size_t capped = 0;
    std::size_t refused = 0;
    std::size_t skippedResident = 0;
    std::size_t skippedNoImportance = 0;
};

// The same by the packet's lab-frame group (per-group window or provenance
// on), plus IMC steps of packets heavier than their group window in a cell
// without importance.
template<std::size_t NumGroups>
struct GroupCounters
{
    std::array<std::size_t, NumGroups> events{};
    std::array<std::size_t, NumGroups> copies{};
    std::array<std::size_t, NumGroups> capped{};
    std::array<std::size_t, NumGroups> refused{};
    std::array<std::size_t, NumGroups> skippedNoImportance{};
};

// Validates a group window against the installed scalar window and fills its
// derived fields. Throws StormError on a non-positive or non-finite window or
// an importance floor outside [0, 1].
template<std::size_t NumGroups>
GroupWindow<NumGroups> finalize(GroupWindow<NumGroups> window, double scalarWindow)
{
    if(!(scalarWindow > 0.0))
    {
        throw StormError("The per-group weight window and packet provenance need the scalar weight window");
    }
    if(NumGroups < 2)
    {
        throw StormError("The per-group weight window and packet provenance need multigroup transport");
    }
    if(!(window.importanceFloor >= 0.0) || !(window.importanceFloor <= 1.0))
    {
        throw StormError("Weight-window importance floor must be in [0, 1]");
    }
    auto smallest = [](std::array<double, NumGroups> const &values, char const *what)
    {
        double minimum = std::numeric_limits<double>::infinity();
        for(std::size_t g = 0; g < NumGroups; ++g)
        {
            if(!(values[g] > 0.0) || !std::isfinite(values[g]))
            {
                StormError eo("Per-group weight window must be positive and finite");
                eo.addEntry("Window", std::string(what));
                eo.addEntry("Group", g);
                eo.addEntry("Value", values[g]);
                throw eo;
            }
            minimum = std::min(minimum, values[g]);
        }
        return minimum;
    };
    window.minMaxWeight = window.active ? smallest(window.maxWeight, "active") : 0.0;
    window.shadowMinMaxWeight = 0.0;
    if(window.provenance && window.shadowValid)
    {
        window.shadowMinMaxWeight = std::numeric_limits<double>::infinity();
        for(unsigned c = 0; c < provenance::kCandidates; ++c)
        {
            window.shadowMinMaxWeight = std::min(window.shadowMinMaxWeight,
                smallest(window.shadowMaxWeight[c], "shadow"));
        }
    }
    if(!window.active)
    {
        window.importanceFloor = 0.0;
    }
    return window;
}

// The installed window (read only during transport).
template<std::size_t NumGroups>
struct Inputs
{
    double scalarWindow = 0.0;                         // <= 0: window off
    std::vector<double> const *importance = nullptr;   // per local cell
    std::size_t maxCopies = 1;
    std::size_t copyBudget = 0;                        // copies per installation
    GroupWindow<NumGroups> const *groupWindow = nullptr;
};

// One split check at the start of a step. A packet heavier than its window
// W / I(c) becomes ceil(|w| I / W) equal packets (at most maxCopies, within
// the copy budget): weight and initial weight divided, splitFactor multiplied,
// the extra copies appended to particlesToAdd with fresh RNG streams from
// initRng (the original keeps its stream). DDMC residents and packets with a
// pending DDMC flux never split. labGroup(frequency) gives the lab-frame
// group. With provenance on, the representation history and the candidate
// opportunities are recorded first (they never change a decision). Returns
// the number of copies made (0 = no split).
template<std::size_t NumGroups, typename ParticleT, typename LabGroup, typename InitRng>
std::size_t apply(ParticleT &particle,
                  std::vector<ParticleT> &particlesToAdd,
                  Inputs<NumGroups> const &in,
                  ScalarCounters &counters,
                  GroupCounters<NumGroups> &groupCounters,
                  LabGroup &&labGroup,
                  InitRng &&initRng)
{
    if(!(in.scalarWindow > 0.0))
    {
        return 0;
    }
    static GroupWindow<NumGroups> const noGroupWindow{};
    GroupWindow<NumGroups> const &groupWindow = in.groupWindow ? *in.groupWindow : noGroupWindow;
    bool const isResident = particle.radiationState.isDDMC();
    bool const hasPending = particle.radiationState.hasPendingFlux();
    if(groupWindow.provenance)
    {
        // Representation history, independent of the weight.
        if(isResident)
        {
            particle.provenance0 = static_cast<std::uint8_t>(particle.provenance0 | provenance::kEverDDMC);
        }
        if(hasPending)
        {
            particle.provenance0 = static_cast<std::uint8_t>(particle.provenance0 | provenance::kEverPendingFlux);
        }
    }
    // Smallest window that can split (importance <= 1): the scalar window, or
    // the smallest group window. Lighter packets can neither split nor set a
    // provenance bit, so they return before the group lookup.
    double const smallestWindow = groupWindow.active ? groupWindow.minMaxWeight : in.scalarWindow;
    bool const shadow = groupWindow.provenance && groupWindow.shadowValid;
    double const rejectBelow = shadow ? std::min(smallestWindow, groupWindow.shadowMinMaxWeight) : smallestWindow;
    double const weight = std::abs(particle.weight);
    if(!(weight > rejectBelow))
    {
        return 0;
    }
    // A DDMC resident is a cell-wide state, and a pending DDMC flux would be
    // credited once per copy; such packets split on a later IMC step.
    bool const overSmallestWindow = weight > smallestWindow;
    if(isResident || hasPending)
    {
        if(overSmallestWindow)
        {
            ++counters.skippedResident;
        }
        return 0;
    }
    std::size_t const cell = particle.cellIndex;
    bool const validCell = in.importance && cell < in.importance->size();
    double importance = validCell ? std::min(1.0, (*in.importance)[cell]) : 0.0;
    if(validCell && groupWindow.active)
    {
        importance = std::max(importance, groupWindow.importanceFloor);
    }
    bool const byGroup = groupWindow.active || groupWindow.provenance;
    std::size_t const group = byGroup
        ? std::min<std::size_t>(static_cast<std::size_t>(labGroup(particle.frequency)), NumGroups - 1) : 0;
    if(shadow)
    {
        for(unsigned c = 0; c < provenance::kCandidates; ++c)
        {
            double const candidate = groupWindow.shadowMaxWeight[c][group];
            if(validCell && importance > 0.0)
            {
                if(weight > candidate / importance)
                {
                    particle.provenance1 = static_cast<std::uint8_t>(particle.provenance1 | provenance::kOpportunity[c]);
                }
            }
            else if(validCell && weight > candidate)
            {
                particle.provenance0 = static_cast<std::uint8_t>(
                    particle.provenance0 | provenance::kZeroImportanceOverweight[c]);
            }
        }
    }
    double const maxWeight = groupWindow.active ? groupWindow.maxWeight[group] : in.scalarWindow;
    if(!validCell || !(importance > 0.0))
    {
        if(overSmallestWindow)
        {
            ++counters.skippedNoImportance;
        }
        if(byGroup && weight > maxWeight)
        {
            ++groupCounters.skippedNoImportance[group];
        }
        return 0;
    }
    double const window = maxWeight / importance;
    if(!(weight > window) || !std::isfinite(weight))
    {
        return 0;
    }
    double const wanted = std::ceil(weight / window);
    std::size_t copies = in.maxCopies;
    bool capped = false;
    if(wanted < static_cast<double>(copies))
    {
        copies = static_cast<std::size_t>(wanted);
    }
    else
    {
        ++counters.capped;
        capped = true;
    }
    if(copies < 2)
    {
        return 0;
    }
    std::size_t const room = in.copyBudget > counters.copies ? in.copyBudget - counters.copies : 0;
    if(copies - 1 > room)
    {
        ++counters.refused;
        if(byGroup)
        {
            ++groupCounters.refused[group];
        }
        if(groupWindow.provenance)
        {
            particle.provenance0 = static_cast<std::uint8_t>(particle.provenance0 | provenance::kRefusedAncestor);
        }
        if(room == 0)
        {
            return 0;
        }
        copies = room + 1;
    }
    if(groupWindow.active)
    {
        // splitFactor is a float: keep the product an exact integer.
        double const product = static_cast<double>(particle.splitFactor) * static_cast<double>(copies);
        if(!std::isfinite(product) || product > 16777216.0)
        {
            StormError eo("Weight-window split factor exceeds the exact float range");
            eo.addEntry("Split factor", particle.splitFactor);
            eo.addEntry("Copies", copies);
            eo.addEntry("Weight", particle.weight);
            eo.addEntry("Window", window);
            eo.addEntry("Group", group);
            eo.addEntry("Cell index", cell);
            throw eo;
        }
    }
    double const scale = 1.0 / static_cast<double>(copies);
    particle.weight *= scale;
    particle.initialWeight *= scale;
    particle.splitFactor *= static_cast<float>(copies);
    for(std::size_t copy = 1; copy < copies; ++copy)
    {
        ParticleT extra = particle;
        extra.id = std::numeric_limits<decltype(extra.id)>::max();
        initRng(extra);
        particlesToAdd.push_back(std::move(extra));
    }
    ++counters.events;
    counters.copies += copies - 1;
    if(byGroup)
    {
        ++groupCounters.events[group];
        groupCounters.copies[group] += copies - 1;
        if(capped)
        {
            ++groupCounters.capped[group];
        }
    }
    return copies;
}
} // namespace postprocess_split
} // namespace STORM

#endif // STORM_RADIATION_POSTPROCESS_WEIGHT_SPLIT_HPP
