#ifndef STORM_RADIATION_IMC_STATE_HPP
#define STORM_RADIATION_IMC_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../RandomWalk.hpp"
#include "../ddmc/DDMCTypes.hpp"
#include "IMCComptonState.hpp"
#include "IMCDDMCState.hpp"
#include "IMCRNG.hpp"
#include "IMCRandomWalkState.hpp"
#include "IMCSourceControl.hpp"
#include "IMCTransportTallies.hpp"

namespace STORM::radiation_imc_detail {

/// Storage shared by the IMC execution phases.
///
/// This is deliberately a passive state object: it owns no reference back to
/// RadiationIMC and contains no transport logic.  Keeping the state here
/// makes the public RadiationIMC declaration small while preserving the
/// existing field names and access patterns used by the composed processes.
template<typename ParametersT,
         typename TraitsT,
         typename PositionSamplerT,
         typename PositionDecompositionT,
         typename ComptonCellDataT,
         typename ComptonBackendT,
         typename ObserverT,
         typename RandomWalkT,
         typename PGRWCellDataT,
         typename DDMCCellDataT,
         typename PointT,
         std::size_t NumGroups>
struct IMCState
    : IMCRNG<PositionDecompositionT>,
      IMCTransportTallies<PointT, NumGroups>,
      IMCSourceControl<NumGroups>,
      IMCComptonState<ComptonCellDataT, ComptonBackendT, NumGroups>,
      IMCRandomWalkState<RandomWalkT, PGRWCellDataT>,
      IMCDDMCState<DDMCCellDataT, PointT>
{
    using GroupArray = std::array<double, NumGroups>;
    using RNGState = IMCRNG<PositionDecompositionT>;

    IMCState(ParametersT parameters,
             TraitsT traits,
             PositionSamplerT positionSampler,
             std::uint64_t seed) :
        RNGState(seed),
        parameters_(std::move(parameters)),
        traits_(std::move(traits)),
        positionSampler_(std::move(positionSampler))
    {
    }

    ParametersT parameters_;
    TraitsT traits_;
    PositionSamplerT positionSampler_;

    std::shared_ptr<ObserverT> observer_;
    bool preStepInitialized_ = false;

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMC_STATE_HPP
