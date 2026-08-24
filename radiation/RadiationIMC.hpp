#ifndef STORM_RADIATION_IMC_HPP
#define STORM_RADIATION_IMC_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef STORM_WITH_MPI
    #include <mpi.h>
    #include "../utils/MpiExchangeGrid.hpp"
#endif

#include <boost/math/special_functions/pow.hpp>

#include <units/units.hpp>
#include "StormError.hpp"
#include "boundary/BoundaryCondition.hpp"
#include "elementary/PointOps.hpp"
#include "particle/Particle.hpp"
#include "particle/StepResult.hpp"
#include "../utils/RandomInCell.hpp"
#include "physics/MonteCarloPhysics.hpp"
#include "radiation/RadiationIMCParameters.hpp"
#include "radiation/RadiationPressureGradient.hpp"
#include "radiation/RadiationIMCTraits.hpp"
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/Compton.hpp"
#include "radiation/compton/CMMCComptonBackend.hpp"
#include "radiation/Observer.hpp"
#include "radiation/Polarization.hpp"
#include "radiation/RandomWalk.hpp"
#include "radiation/ddmc/DDMCGeometry.hpp"
#include "radiation/ddmc/DDMCGhostExchange.hpp"
#include "radiation/ddmc/DDMCSampling.hpp"
#include "radiation/ddmc/DDMCTypes.hpp"
#include "radiation/ddmc/DDMCWollaegerInterface.hpp"
#include <planck_integral/planck_integral.hpp>
#include "../utils/LinearInterpolation.hpp"
#include "../utils/CounterRNG.hpp"
#include "../mesh_movement/MeshMovement.hpp"

#ifdef STORM_WITH_GPU
    #include "../gpu/GreyIMCData.hpp"
    #include "../gpu/KokkosRuntime.hpp"
#endif

namespace STORM {

namespace radiation_imc_detail {

template<typename T, typename = void>
struct has_member_ID : std::false_type {};

template<typename T>
struct has_member_ID<T, std::void_t<decltype(std::declval<const T &>().ID)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_density : std::false_type {};

template<typename T>
struct has_member_density<T, std::void_t<decltype(std::declval<const T &>().density)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_pressure : std::false_type {};

template<typename T>
struct has_member_pressure<T, std::void_t<decltype(std::declval<T &>().pressure)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_internal_energy_specific : std::false_type {};

template<typename T>
struct has_member_internal_energy_specific<T, std::void_t<decltype(std::declval<T &>().internal_energy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_internal_energy_density : std::false_type {};

template<typename T>
struct has_member_internal_energy_density<T, std::void_t<decltype(std::declval<T &>().internalEnergy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_total_energy : std::false_type {};

template<typename T>
struct has_member_total_energy<T, std::void_t<decltype(std::declval<T &>().energy)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_mass : std::false_type {};

template<typename T>
struct has_member_mass<T, std::void_t<decltype(std::declval<const T &>().mass)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_radiation_energy : std::false_type {};

template<typename T>
struct has_member_radiation_energy<T, std::void_t<decltype(std::declval<T &>().Erad)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_group_energy_mutable : std::false_type {};

template<typename T>
struct has_member_group_energy_mutable<T, std::void_t<decltype(std::declval<T &>().Eg)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_velocity : std::false_type {};

template<typename T>
struct has_member_velocity<T, std::void_t<decltype(std::declval<const T &>().velocity)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_momentum : std::false_type {};

template<typename T>
struct has_member_momentum<T, std::void_t<decltype(std::declval<T &>().momentum)>> : std::true_type {};

template<typename T, typename = void>
struct has_member_temperature : std::false_type {};

template<typename T>
struct has_member_temperature<T, std::void_t<decltype(std::declval<const T &>().temperature)>> : std::true_type {};

template<typename EOST, typename TracersT, typename TracerNamesT, typename = void>
struct has_dT2e : std::false_type {};

template<typename EOST, typename TracersT, typename TracerNamesT>
struct has_dT2e<EOST, TracersT, TracerNamesT, std::void_t<decltype(
    std::declval<const EOST &>().dT2e(
        std::declval<double>(), std::declval<double>(),
        std::declval<const TracersT &>(),
        std::declval<const TracerNamesT &>()))>> : std::true_type {};

template<typename CellT>
std::size_t cellID(const CellT &cell)
{
    if constexpr(has_member_ID<CellT>::value)
    {
        return cell.ID;
    }
    else
    {
        (void) cell;
        return std::numeric_limits<std::size_t>::max();
    }
}

template<typename T, typename = void>
struct has_get_self_index : std::false_type {};

template<typename T>
struct has_get_self_index<T, std::void_t<
    decltype(std::declval<const T &>().GetSelfIndex())>> : std::true_type {};

template<typename GridT, typename CellT>
std::size_t ddmcStableCellID(const GridT &grid,
                             std::size_t cellIndex,
                             const CellT &cell)
{
    if constexpr(has_member_ID<CellT>::value)
    {
        return static_cast<std::size_t>(cell.ID);
    }
    else if constexpr(has_get_self_index<GridT>::value)
    {
        const auto &selfIndex = grid.GetSelfIndex();
        if(cellIndex < selfIndex.size())
        {
            return selfIndex[cellIndex];
        }
    }
    return cellIndex;
}

template<typename ExtensivesT>
void addTotalEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_total_energy<ExtensivesT>::value)
    {
        extensives.energy += energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
void clearRadiationEnergyIfPresent(ExtensivesT &extensives)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad = 0.0;
    }
    else
    {
        (void) extensives;
    }
}

template<typename ExtensivesT>
void addRadiationEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad += energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
void setRadiationEnergyIfPresent(ExtensivesT &extensives, double energy)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        extensives.Erad = energy;
    }
    else
    {
        (void) extensives;
        (void) energy;
    }
}

template<typename ExtensivesT>
double radiationEnergyIfPresent(const ExtensivesT &extensives)
{
    if constexpr(has_member_radiation_energy<ExtensivesT>::value)
    {
        return extensives.Erad;
    }
    else
    {
        (void) extensives;
        return 0.0;
    }
}

template<typename ExtensivesT>
void clearGroupEnergyIfPresent(ExtensivesT &extensives)
{
    if constexpr(has_member_group_energy_mutable<ExtensivesT>::value)
    {
        std::fill(extensives.Eg.begin(), extensives.Eg.end(), 0.0);
    }
    else
    {
        (void) extensives;
    }
}

template<typename CellT>
void setCellRadiationEnergyIfPresent(CellT &cell, double value)
{
    if constexpr(has_member_radiation_energy<CellT>::value)
    {
        cell.Erad = value;
    }
    else
    {
        (void) cell;
        (void) value;
    }
}

template<typename CellT>
void setCellGroupEnergyIfPresent(CellT &cell, std::size_t group, double value)
{
    if constexpr(has_member_group_energy_mutable<CellT>::value)
    {
        cell.Eg[group] = value;
    }
    else
    {
        (void) cell;
        (void) group;
        (void) value;
    }
}

template<typename PointT, typename ParticleT, typename CellT>
double computeDopplerShift(const ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return 1.0;
        }
        double gamma = 1.0 / std::sqrt(1.0 - v2 * units::inv_clight2);
        return gamma * (1.0 - ScalarProd(cell.velocity, particle.velocity) * units::inv_clight2);
    }
    else
    {
        (void) particle;
        (void) cell;
    }
    return 1.0;
}

template<typename PointT, typename ParticleT, typename CellT>
void lorentzTransformToComoving(ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double const v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return;
        }
        double const gamma = 1.0 / std::sqrt(
            1.0 - v2 * units::inv_clight2);
        double const dopplerShift = gamma *
            (1.0 - ScalarProd(cell.velocity, particle.velocity) *
             units::inv_clight2);
        if(!(dopplerShift > 0.0) || !std::isfinite(dopplerShift))
        {
            throw StormError(
                "RadiationIMC received an invalid lab-to-comoving Doppler factor");
        }
        particle.frequency *= dopplerShift;
        particle.weight *= dopplerShift;
        double const vDotP = ScalarProd(particle.velocity, cell.velocity);
        particle.velocity = particle.velocity + cell.velocity *
            ((gamma - 1.0) * vDotP / v2 - gamma);
        double const newSpeed = fastabs(particle.velocity);
        if(newSpeed > 0.0)
        {
            particle.velocity *= units::clight / newSpeed;
        }
    }
    else
    {
        (void) particle;
        (void) cell;
    }
}

template<typename PointT, typename ParticleT, typename CellT>
void lorentzTransformToLab(ParticleT &particle, const CellT &cell)
{
    if constexpr(has_member_velocity<CellT>::value)
    {
        double v2 = ScalarProd(cell.velocity, cell.velocity);
        if(v2 < 1e-30)
        {
            return;
        }
        double gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
        PointT negV = cell.velocity * (-1.0);
        double dopplerShift = gamma * (1.0 - ScalarProd(negV, particle.velocity) * units::inv_clight2);
        particle.frequency *= dopplerShift;
        particle.weight *= dopplerShift;
        double vDotP = ScalarProd(particle.velocity, negV);
        particle.velocity = particle.velocity + negV * ((gamma - 1.0) * vDotP / v2 - gamma);
        double newSpeed = fastabs(particle.velocity);
        if(newSpeed > 0.0)
        {
            particle.velocity = particle.velocity * (units::clight / newSpeed);
        }
    }
    else
    {
        (void) particle;
        (void) cell;
    }
}

} // namespace radiation_imc_detail

template<typename PointT, typename GridT>
struct CellCenterPositionSampler
{
    using Decomposition = CellVolumeDecomposition;

    void BuildDecomposition(const GridT &grid,
                            std::size_t cellIndex,
                            Decomposition &out) const
    {
        (void) grid;
        (void) cellIndex;
        out.clear();
    }

    PointT Sample(const GridT &grid,
                  std::size_t cellIndex,
                  const Decomposition &decomp,
                  std::mt19937_64 &rng,
                  std::uniform_real_distribution<double> &dist) const
    {
        (void) decomp;
        (void) rng;
        (void) dist;
        return grid.GetMeshPoint(cellIndex);
    }

    PointT operator()(const GridT &grid,
                      std::size_t cellIndex,
                      std::mt19937_64 &rng,
                      std::uniform_real_distribution<double> &dist) const
    {
        (void) rng;
        (void) dist;
        return grid.GetMeshPoint(cellIndex);
    }
};

namespace radiation_imc_detail {

/// Detects position samplers exposing the reusable-decomposition API.
template<typename SamplerT, typename = void>
struct sampler_decomposition
{
    struct type {};
    static constexpr bool supported = false;
};

template<typename SamplerT>
struct sampler_decomposition<SamplerT, std::void_t<typename SamplerT::Decomposition>>
{
    using type = typename SamplerT::Decomposition;
    static constexpr bool supported = true;
};

} // namespace radiation_imc_detail

template<typename PointT,
         typename GridT,
         typename CellT,
         typename ExtensivesT,
         typename EOST,
         std::size_t NumGroups,
         typename TraitsT = DirectRadiationIMCTraits<PointT, CellT, ExtensivesT, NumGroups>,
         typename PositionSamplerT = RandomInCellPositionSampler<PointT, GridT>>
class RadiationIMC final : public MonteCarloPhysics<PointT, GridT>
{
public:
    static_assert(NumGroups > 0, "RadiationIMC requires at least one frequency group");

    using Base = MonteCarloPhysics<PointT, GridT>;
    using MCParticle = Particle<PointT>;
    using Functionality = StepResult;
    using BoundaryCond = BoundaryCondition<PointT, GridT>;
    using Parameters = RadiationIMCParameters<NumGroups>;
    using OpacityModel = RadiationOpacityModel<PointT, GridT, CellT, NumGroups>;
    using Traits = TraitsT;
    using PositionSampler = PositionSamplerT;
    using PositionDecomposition =
        typename radiation_imc_detail::sampler_decomposition<PositionSamplerT>::type;
    static constexpr bool kSamplerHasDecomposition =
        radiation_imc_detail::sampler_decomposition<PositionSamplerT>::supported;
    using GroupArray = std::array<double, NumGroups>;
    using GroupBoundaries = std::array<double, NumGroups + 1>;
    using GroupCdf = std::array<double, NumGroups + 1>;
    using GroupMatrix = std::array<GroupArray, NumGroups>;
    using GroupCdfMatrix = std::array<GroupCdf, NumGroups>;

    struct SourceAllocationSummary
    {
        bool adaptiveEnabled = false;
        std::size_t totalPhotons = 0;
        std::size_t sourceCells = 0;
        std::size_t boostedCells = 0;
        std::size_t learnedCells = 0;
        std::size_t learnedBoostedCells = 0;
        std::size_t learnedPhotons = 0;
        std::size_t learnedExtraPhotons = 0;
        std::size_t minPhotons = 0;
        std::size_t maxPhotons = 0;
        std::size_t learnedMinPhotons = 0;
        std::size_t learnedMaxPhotons = 0;
        double adaptiveScoreSum = 0.0;
        double adaptiveScoreP05 = 0.0;
        double adaptiveScoreP50 = 0.0;
        double adaptiveScoreP95 = 0.0;
        double adaptiveScoreMax = 0.0;
        double adaptiveScoreSpanLow = 0.0;
        double adaptiveScoreSpanHigh = 0.0;
        std::size_t learnedPhotonsAtLeast1000 = 0;
        std::size_t learnedPhotonsAtLeast2000 = 0;
    };

    struct GroupSamplingDiagnostics
    {
        std::size_t totalSampled = 0;
        std::size_t cellsWithGroupScores = 0;
        double weightCorrectionMin = 1.0;
        double weightCorrectionMax = 1.0;
        double weightCorrectionSum = 0.0;
        std::size_t weightCorrectionCount = 0;
        std::size_t weightCorrectionCapped = 0;
        std::size_t weightCorrectionFallback = 0;
        std::size_t invalidPdfFallback = 0;
        std::size_t invalidPdfFallbackPackets = 0;
        double sampledEnergy = 0.0;
        double cappedEnergy = 0.0;
        double cappedEnergyFraction = 0.0;
        bool estimatorPotentiallyBiased = false;
    };

    enum class ComptonCorrectionFailure : unsigned char
    {
        None = 0,
        DirectLinearSolveFailed,
        DirectNegativeMass,
        DirectMaterialCap,
        DirectProjectedResidual,
        AdaptiveLinearSolveFailed,
        AdaptiveMaximumSubsteps,
        AdaptiveMaximumRejectedTrials,
        AdaptiveFractionBelowMinimum,
        AdaptiveNoProgress,
        NonFiniteState,
        InvalidEnergyBudget,
        EnergyClosureFailure
    };

    using ComptonCellData = STORM::ComptonCellData<NumGroups>;
    using Observer = RadiationObserver<PointT>;

    using DDMCFaceLeak = ddmc::FaceLeak<PointT>;
    using DDMCCellData = ddmc::CellData<PointT>;

    struct PostProcessExternalSource
    {
        std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
        std::size_t cellID = std::numeric_limits<std::size_t>::max();
        std::size_t interiorCellID = std::numeric_limits<std::size_t>::max();
        PointT location{};
        PointT outwardNormal{};
        double luminosity = 0.0;
    };

    RadiationIMC(const GridT &grid,
                 const std::shared_ptr<BoundaryCond> &boundary,
                 std::vector<CellT> &cells,
                 std::vector<ExtensivesT> &extensives,
                 std::shared_ptr<EOST> eos,
                 std::shared_ptr<OpacityModel> opacity,
                 Parameters parameters,
                 TraitsT traits = TraitsT(),
                 PositionSamplerT positionSampler = PositionSamplerT(),
                 std::uint64_t seed = 42);

    std::vector<MCParticle> preStep(double fullDt) override;
    Functionality step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd) override;
    void postStep(const std::vector<MCParticle> &particles, double fullDt) override;
    void onBoundaryResult(const MCParticle &particle,
                          ParticleStatus status,
                          bool escaped) override;

    MCParticle generateSingleParticle(std::size_t cellIndex, const CellT &cell);
    /// `decomposition` may be null, in which case the sampler rebuilds it internally.
    MCParticle generateSingleParticle(std::size_t cellIndex, const CellT &cell,
                                      const PositionDecomposition *decomposition);
    std::vector<MCParticle> generateInitialParticles(std::size_t particlesPerCell);
    void adjustExistingParticles(std::vector<MCParticle> &particles, double fullDt);

    const std::vector<double> &getFactorFleck() const { return this->factorFleck_; }
    const std::vector<double> &getPlanckOpacities() const { return this->planckOpacities_; }
    const std::vector<double> &getScatteringOpacities() const { return this->scatteringOpacities_; }
    const std::vector<ComptonCellData> &getComptonData() const { return this->comptonData_; }
    const GroupArray &getComptonGroupCenters() const { return this->comptonGroupCenters_; }
    const GroupArray &getComptonGroupWidths() const { return this->comptonGroupWidths_; }
    const std::vector<double> &getEradTimeAvg() const { return this->Erad_time_avg_; }
    std::vector<double> &getEradTimeAvg() { return this->Erad_time_avg_; }
    const std::vector<GroupArray> &getEgTimeAvg() const { return this->Eg_time_avg_; }
    std::vector<GroupArray> &getEgTimeAvg() { return this->Eg_time_avg_; }
#ifdef STORM_WITH_GPU
    bool UsesDeviceTransport() const
    {
        return this->gpuTransportEnabled_;
    }

    gpu::GreyIMCViews<gpu::DeviceVec3> GetDeviceTransportViews() const
    {
        return this->gpuData_->Views(units::clight, !this->parameters_.noHydroFeedback);
    }
#endif
    const GroupBoundaries &getEnergyBoundaries() const { return this->energyBoundaries_; }
    const Parameters &getParameters() const { return this->parameters_; }
    const SourceAllocationSummary &getLastSourceAllocationSummary() const { return this->lastSourceAllocationSummary_; }
    const std::vector<std::size_t> &getLastSourcePhotonsPerCell() const { return this->lastSourcePhotonsPerCell_; }
    GroupSamplingDiagnostics getLastGroupSamplingDiagnostics() const { return this->lastGroupSamplingDiagnostics_; }
    void setObserver(std::shared_ptr<Observer> observer)
    {
        this->observer_ = std::move(observer);
        if(this->observer_)
        {
            this->observer_->setPolarizationEnabled(this->polarizationEnabled());
        }
    }
    const std::shared_ptr<Observer> &getObserver() const { return this->observer_; }

    void reseedRNG(std::uint64_t seed)
    {
        std::uint64_t rankOffset = 0;
#ifdef STORM_WITH_MPI
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            rankOffset = static_cast<std::uint64_t>(rank) * 104729ULL;
        }
#endif
        this->rng_.seed(seed + rankOffset);
        this->particleRngSeed_ = seed + rankOffset;
        this->sourceRngStreamCounter_ = 0;
        this->opacity_->reseed(seed + 1ULL);
        ReseedRandomInCell(seed + 2ULL);
    }

    std::size_t getRandomWalkStepCount() const override { return this->rwStepCount_; }
    std::size_t getDDMCStepCount() const override { return this->ddmcStepCount_; }
    std::size_t getDDMCLeakCount() const override { return this->ddmcLeakCount_; }
    std::size_t getDDMCCensusCount() const override { return this->ddmcCensusCount_; }
    std::size_t getDDMCUpscatterCount() const override { return this->ddmcUpscatterCount_; }
    std::size_t getDDMCFallbackCount() const override { return this->ddmcFallbackCount_; }
    std::string getAccelerationDebugInfo(std::size_t cellIndex,
                                         double frequency) const override;
    std::string getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const;
    std::string getDDMCInterfaceEventDiagnosticsTSV(double xMin,
                                                    double xMax) const;
    std::size_t getDDMCMovingInterfaceBypassCount() const
    {
        return this->ddmcMovingInterfaceBypassCount_;
    }
    double getDDMCMovingInterfaceMaxFactor() const
    {
        return this->ddmcMovingInterfaceMaxFactor_;
    }
    double getDDMCLeakReciprocityResidualMax() const
    {
        return this->ddmcLeakReciprocityResidualMax_;
    }
    std::size_t getDDMCLeakReciprocityCheckCount() const
    {
        return this->ddmcLeakReciprocityCheckCount_;
    }
    const std::vector<DDMCCellData> &getDDMCCellData() const
    {
        return this->ddmcCellData_;
    }
    const std::vector<PointT> &getDDMCFluxRhsIntegrated() const
    {
        return this->ddmcFluxRhsIntegrated_;
    }
    std::size_t getDDMCMomentumFeedbackCount() const
    {
        return this->ddmcMomentumFeedbackCount_;
    }
    std::size_t getDDMCMomentumMatrixFallbackCount() const
    {
        return this->ddmcMomentumMatrixFallbackCount_;
    }

    void setPostProcessExternalSources(
        std::vector<PostProcessExternalSource> sources);
    void clearPostProcessExternalSources();
    bool hasPostProcessExternalSources() const
    {
        return this->postProcessExternalSourceMode_;
    }

    void setNewPhotonsPerCell(std::size_t n);
    void setAdaptiveSourceCellScores(std::unordered_map<std::size_t, double> scores,
                                     double strength,
                                     double maxFactor,
                                     double learnedReserveFrac,
                                     double learnedMinFactor,
                                     double observerBudgetMultiplier,
                                     std::size_t learnedMinPhotons = 0,
                                     std::size_t learnedMaxPhotons = 0,
                                     double scorePower = 1.0);
    void clearAdaptiveSourceCellScores();
    void setAdaptiveSourceCellGroupScores(std::unordered_map<std::size_t, GroupArray> scores,
                                          double strength,
                                          double pdfFloor,
                                          double maxBias,
                                          double maxWeightCorrection);
    void clearAdaptiveSourceCellGroupScores();
    void setSourceEmissionControl(bool useLearnedScores,
                                  bool includeUniformBase,
                                  std::size_t baseMultiplier,
                                  std::size_t learnedBoostFactor = 20,
                                  std::size_t learnedExtraBudget = 0);
    void clearSourceEmissionControl();

private:
    enum class Event
    {
        Face,
        Scatter,
        Census
    };

#ifdef STORM_WITH_GPU
    bool GreyKernelEligible() const
    {
#if defined(STORM_DEBUG) || defined(STORM_WITH_TRACING_HISTORY)
        return false;
#else
        return !this->parameters_.withMultigroupOpacity &&
               !this->parameters_.withRandomWalk &&
               !this->parameters_.withDDMC &&
               !this->parameters_.withCompton &&
               !this->parameters_.withHydro &&
               !this->parameters_.postProcess.enabled &&
               !this->observer_ &&
               !this->polarizationEnabled();
#endif
    }

    gpu::GreyIMCViews<PointT> GetHostTransportViews()
    {
        gpu::GreyIMCViews<PointT> result;
        result.grid.cellFaceOffsets = this->gridData.cellFaceOffsets.data();
        result.grid.cellCenters = this->gridData.cellCenters.data();
        result.grid.normals = this->gridData.normals.data();
        result.grid.pointsOnFaces = this->gridData.pointsOnFaces.data();
        result.grid.nextCellIndices = this->gridData.nextCellIndices.data();
        result.grid.boundaryCrossings = this->gridData.boundaryCrossings.data();
        result.grid.deviceBoundaryBehaviors =
            this->gridData.deviceBoundaryBehaviors.data();
        result.grid.cellCount = this->grid.GetPointNo();
        result.absorptionOpacities = this->planckOpacities_.data();
        result.scatteringOpacities = this->scatteringOpacities_.data();
        result.fleckFactors = this->factorFleck_.data();
        result.pendingMaterialEnergy = this->pendingMaterialEnergy_.data();
        result.pendingRadiationEnergy = this->pendingRadiationEnergy_.data();
        result.speedOfLight = units::clight;
        result.depositMaterialEnergy = !this->parameters_.noHydroFeedback;
        return result;
    }
#endif

    struct ComptonProjectionResult
    {
        GroupArray endpoint{};
        double targetTotal = 0.0;
        double inputTotal = 0.0;
        double negativeMass = 0.0;
        double worstNegative = 0.0;
        std::size_t worstNegativeGroup = NumGroups;
        double maximumRelativeChange = 0.0;
        bool usedProjection = false;
        bool usedCapRepair = false;
        bool success = false;
    };

    struct ComptonCorrectionResult
    {
        GroupArray endpoint{};
        GroupArray delta{};
        double radiationTotal = 0.0;
        double materialEnergyBefore = 0.0;
        double materialEnergyAfter = 0.0;
        double budgetBefore = 0.0;
        double energyClosureResidual = 0.0;
        bool usedProjection = false;
        bool usedCapRepair = false;
        bool success = false;
        ComptonCorrectionFailure failure =
            ComptonCorrectionFailure::None;
    };

    std::vector<MCParticle> generateParticles(double fullDt);
    void validateGridSizedState() const;
    void validateEnergyBoundaries() const;
    void rejectUnsupportedParameters() const;
    void rejectUnsupportedParameter(const std::string &name) const;
    bool polarizationEnabled() const
    {
        return this->parameters_.withPolarization ||
               this->parameters_.postProcess.polarization.enabled;
    }
    void precomputeComptonData(double sourceDt);
    void initializeComptonMatrixGenerator();
    std::vector<double> buildComptonTemperatures() const;
    GroupCdf buildSafeComptonCdf(const GroupArray &weights) const;
    void buildComptonMatricesForCell(const CellT &cell,
                                     std::size_t cellIndex,
                                     ComptonOccupationMode occupationMode,
                                     ComptonCellData &data);
    double computeLteTemperature(const CellT &cell,
                                 std::size_t cellIndex) const;
    void recomputeComptonContractions(ComptonCellData &data) const;
    void buildComptonSources(double sourceDt, ComptonCellData &data) const;
    void buildComptonEventData(ComptonCellData &data) const;
    void computeComptonRiskForCell(double fullDt, ComptonCellData &data) const;
    std::vector<MCParticle> generateComptonParticles(double fullDt);
    void applyComptonEndOfStepCorrection(double fullDt);
    void reconcileComptonParticles(std::vector<MCParticle> &particles);
    void splitComptonRiskyParticles(std::vector<MCParticle> &particles,
                                    double fullDt);
    std::size_t sampleComptonCdf(const GroupCdf &cdf, double random) const;
    static double sumComptonGroups(const GroupArray &values);
    static double minComptonGroup(const GroupArray &values);
    static double maxAbsComptonGroup(const GroupArray &values);
    static double normComptonGroups(const GroupArray &values);
    static double compensatedSumComptonGroups(const GroupArray &values);
    static const char *comptonCorrectionFailureName(
        ComptonCorrectionFailure failure);
    static bool solveComptonGroupSystem(GroupMatrix matrix,
                                         GroupArray rhs,
                                         GroupArray &solution);
    static GroupArray multiplyComptonMatrix(const GroupMatrix &matrix,
                                            const GroupArray &values);
    static double relativeComptonResidual(const GroupMatrix &matrix,
                                          const GroupArray &solution,
                                          const GroupArray &rhs,
                                          double scale);
    static ComptonProjectionResult projectNonnegativeConservative(
        const GroupArray &candidate,
        double targetTotal,
        double energyScale,
        double perGroupNegativeTolerance,
        double totalNegativeTolerance);
    ComptonCorrectionResult solveComptonCorrection(
        std::size_t cellIndex,
        double fullDt,
        const ComptonCellData &data,
        const GroupArray &rawGroupEnergy,
        const GroupArray &timeAvgGroupEnergy,
        double budgetBefore,
        double materialFloor,
        double preStepRadiation) const;
    double frequencyForComptonGroup(std::size_t group) const;
    void setPacketFromComovingState(MCParticle &particle,
                                    const CellT &cell,
                                    double comovingFrequency,
                                    double comovingWeight) const;
    double applyComptonScatterEvent(std::size_t cellIndex,
                                    CellT &cell,
                                    std::size_t sourceGroup,
                                    MCParticle &particle,
                                    const PointT &oldVelocity,
                                    double oldWeight);
    std::size_t sampleComptonTarget(const ComptonCellData &data,
                                     std::size_t sourceGroup,
                                     MCParticle &particle);
    void addComptonMaterialExchange(std::size_t cellIndex, double energy);
    void recordObserverCrossing(const MCParticle &particle,
                                const PointT &crossingPoint);
    void setInitialWeightFromWeight(MCParticle &particle) const;
    void initializeParticleRNG(MCParticle &particle);
    double randomUnitOpen();
    double randomUnitOpen(MCParticle &particle);
    PointT sampleRandomVelocity(const CellT &cell, MCParticle &particle);
    PointT sampleScatterVelocity(const CellT &cell, MCParticle &particle);
    double density(std::size_t cellIndex) const;
    double specificInternalEnergy(std::size_t cellIndex) const;
    double totalRadiationEnergy(std::size_t cellIndex) const;
    void depositMaterialEnergy(std::size_t cellIndex, double energy);
    void resetTransportTallies(std::size_t cellCount);
    void tallyMaterialEnergy(std::size_t cellIndex, double energy,
                             bool addToTotalEnergy = false);
    void tallyMomentum(std::size_t cellIndex, const PointT &momentum);
    void tallyRadiationEnergy(std::size_t cellIndex, double integratedEnergy);
    void tallyGroupRadiationEnergy(std::size_t cellIndex,
                                   std::size_t group,
                                   double integratedEnergy);
    void applyTransportTallies();
    void synchronizeMaterialCell(std::size_t cellIndex);
    void throwIfNegativeInternalEnergy(std::size_t cellIndex, const std::string &where);

    void precomputeRandomWalkData();
    bool tryRandomWalkStep(MCParticle &particle, Functionality &functionality);
    double computeMinDistanceToFaces(std::size_t cellIndex, const PointT &location) const;
    double computeCellSurfaceArea(std::size_t cellIndex) const;

    void precomputeDDMCData();
    bool tryDDMCStep(MCParticle &particle, Functionality &functionality);
    void addDDMCFluxContribution(std::size_t cellIndex,
                                 const PointT &contribution);
    void applyDDMCMomentumFeedback(double fullDt);
    bool tryIMCToDDMCInterface(MCParticle &particle,
                               Functionality &functionality,
                               std::vector<MCParticle> &particlesToAdd,
                               std::size_t sourceCellIndex,
                               std::size_t targetCellIndex,
                               std::size_t faceIndex);

    enum class DDMCDiagnosticEventKind : unsigned char
    {
        IMCCandidate,
        IMCFrequencyReject,
        IMCIncident,
        IMCAdmitted,
        IMCReflected,
        IMCBypass,
        DDMCToDDMC,
        DDMCToIMC
    };

    static constexpr std::size_t DDMC_DIAGNOSTIC_GREY_GROUP =
        std::numeric_limits<std::size_t>::max();

    struct DDMCDiagnosticEventKey
    {
        DDMCDiagnosticEventKind kind = DDMCDiagnosticEventKind::IMCCandidate;
        std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
        std::size_t sourceCellID = std::numeric_limits<std::size_t>::max();
        std::size_t targetCellID = std::numeric_limits<std::size_t>::max();
        std::size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;

        bool operator<(DDMCDiagnosticEventKey const &other) const
        {
            return std::tie(kind, faceIndex, sourceCellID, targetCellID, group) <
                   std::tie(other.kind, other.faceIndex, other.sourceCellID,
                            other.targetCellID, other.group);
        }
    };

    struct DDMCDiagnosticEventAccumulator
    {
        std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
        std::size_t sourceCellID = std::numeric_limits<std::size_t>::max();
        std::size_t targetCellID = std::numeric_limits<std::size_t>::max();
        std::size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;
        std::size_t sourceGroupCutoff = 0;
        std::size_t targetGroupCutoff = 0;
        double faceX = std::numeric_limits<double>::quiet_NaN();
        double sourceGeneratorX = std::numeric_limits<double>::quiet_NaN();
        double targetGeneratorX = std::numeric_limits<double>::quiet_NaN();
        std::size_t count = 0;
        double signedEnergy = 0.0;
        double absoluteEnergy = 0.0;
        double muSum = 0.0;
        std::size_t muCount = 0;
        double admissionProbabilitySum = 0.0;
        std::size_t admissionProbabilityCount = 0;
    };

    void recordDDMCDiagnosticEvent(DDMCDiagnosticEventKind kind,
                                   std::size_t sourceCellIndex,
                                   std::size_t targetCellIndex,
                                   std::size_t faceIndex,
                                   std::size_t group,
                                   double energy,
                                   std::size_t sourceGroupCutoff,
                                   std::size_t targetGroupCutoff,
                                   double mu,
                                   double admissionProbability);

    void clampFrequencyToBounds(double &frequency) const;
    PointT samplePostProcessExternalSourceDirection(
        const PointT &outwardNormal, MCParticle &particle);
    GroupArray buildPostProcessExternalSourcePlanckPdf(
        const CellT &cell) const;
    double samplePostProcessExternalSourcePlanckFrequencyInGroup(
        const CellT &cell, std::size_t group);
    double samplePostProcessExternalSourcePlanckFrequency(const CellT &cell);
    MCParticle generatePostProcessExternalSourceParticle(
        std::size_t cellIndex, const CellT &cell,
        const PostProcessExternalSource &source);
    bool handlePostProcessExternalSourceBoundary(
        MCParticle &particle, std::size_t cellIndex,
        std::size_t faceIndex, Functionality &functionality);

    std::vector<CellT> &cells_;
    std::vector<ExtensivesT> &extensives_;
    std::shared_ptr<EOST> eos_;
    std::shared_ptr<OpacityModel> opacity_;
    Parameters parameters_;
    TraitsT traits_;
    PositionSamplerT positionSampler_;
    GroupBoundaries energyBoundaries_{};
    std::vector<double> factorFleck_;
    std::vector<double> planckOpacities_;
    std::vector<double> scatteringOpacities_;
    std::vector<double> Erad_time_avg_;
    std::vector<GroupArray> Eg_time_avg_;
    std::vector<std::size_t> lastSourcePhotonsPerCell_;
    SourceAllocationSummary lastSourceAllocationSummary_;
    GroupSamplingDiagnostics lastGroupSamplingDiagnostics_;
    std::vector<ComptonCellData> comptonData_;
    GroupArray comptonGroupCenters_{};
    GroupArray comptonGroupWidths_{};
    std::unique_ptr<CMMCComptonBackend<NumGroups>> comptonMatrixGen_;
    bool comptonGroupsInitialized_ = false;
    bool comptonDataReusableInPreStep_ = false;
    double comptonRiskPrecomputeDt_ = -1.0;
    std::shared_ptr<Observer> observer_;
    bool postProcessExternalSourceMode_ = false;
    std::vector<PostProcessExternalSource> postProcessExternalSources_;
    std::vector<std::size_t> postProcessExternalSourceLocalCellIndices_;
    std::unordered_map<std::size_t, std::size_t>
        postProcessExternalSourceFaceIndex_;
    std::unordered_set<std::size_t>
        postProcessExternalSourceInteriorCellIDs_;
    bool preStepInitialized_ = false;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_;
    std::uint64_t particleRngSeed_ = 42;
    std::uint64_t sourceRngStreamCounter_ = 0;
    std::uint64_t creationRank_ = 0;
    bool creationRankCached_ = false;
    /// Reused across cells so the per-cell decomposition keeps its capacity.
    PositionDecomposition scratchDecomposition_;
    std::vector<double> pendingMaterialEnergy_;
    std::vector<double> pendingTotalEnergy_;
    std::vector<PointT> pendingMomentum_;
    std::vector<double> pendingRadiationEnergy_;
    std::vector<GroupArray> pendingGroupRadiationEnergy_;
#ifdef STORM_WITH_GPU
    std::unique_ptr<gpu::KokkosRuntime> gpuRuntime_;
    std::unique_ptr<gpu::GreyIMCData> gpuData_;
    bool gpuTransportEnabled_ = false;
    std::size_t gpuGridBuildGeneration_ = std::numeric_limits<std::size_t>::max();
#endif

            // Use one source-group averaged multiplier for every sampled
            // target. This is the variance-reduced Monte Carlo split; the
            // target-dependent difference remains in residualKernel.
    std::unique_ptr<RandomWalk> randomWalk_;
    std::vector<bool> rwCellEligible_;
    std::vector<double> rwCellTotalOpacity_;
    std::vector<PGRWCellData> rwCellData_;
    std::size_t rwStepCount_ = 0;

    std::vector<DDMCCellData> ddmcCellData_;
    std::vector<int> ddmcPointEligible_;
    std::vector<double> ddmcPointDiffusionCoefficient_;
    std::vector<double> ddmcPointSigmaDiffusion_;
    std::vector<double> ddmcPointSigmaParticleGate_;
    std::vector<std::size_t> ddmcPointGroupCutoff_;
    std::vector<PointT> ddmcPointVelocity_;
    std::vector<std::size_t> ddmcPointCellID_;
    std::vector<PointT> ddmcFluxRhsIntegrated_;
    double ddmcLeakReciprocityResidualMax_ = 0.0;
    std::size_t ddmcLeakReciprocityCheckCount_ = 0;
    std::size_t ddmcMomentumFeedbackCount_ = 0;
    std::size_t ddmcMomentumMatrixFallbackCount_ = 0;
    std::size_t ddmcResidentLeakCount_ = 0;
    std::size_t ddmcTransportLeakCount_ = 0;
    std::size_t ddmcRemoteResidentLeakCount_ = 0;
    std::size_t ddmcMPIFaceFluxReductionCount_ = 0;
    std::size_t ddmcLeakInvalidGeometryCount_ = 0;
    std::size_t ddmcUnsupportedBoundaryFaceCount_ = 0;
    std::size_t ddmcInterfaceIncidentCount_ = 0;
    std::size_t ddmcInterfaceAdmittedCount_ = 0;
    std::size_t ddmcInterfaceReflectedCount_ = 0;
    std::size_t ddmcInterfaceGuAppliedCount_ = 0;
    std::size_t ddmcInterfaceGuFallbackCount_ = 0;
    std::size_t ddmcInterfaceBypassCount_ = 0;
    std::size_t ddmcInterfaceSplitPacketCount_ = 0;
    std::size_t ddmcInterfaceFluxTallyCount_ = 0;
    double ddmcInterfaceMinimumMu_ = std::numeric_limits<double>::infinity();
    std::map<DDMCDiagnosticEventKey, DDMCDiagnosticEventAccumulator>
        ddmcDiagnosticEvents_;
    std::size_t ddmcExternalSourceCandidateFaceCount_ = 0;
    std::size_t ddmcExternalSourceAcceleratedFaceCount_ = 0;
    std::size_t ddmcExternalSourceExplicitFallbackFaceCount_ = 0;
    std::size_t ddmcExternalSourceInteriorExcludedCellCount_ = 0;
    std::size_t ddmcExternalSourceThermalizationCount_ = 0;
    std::size_t ddmcExternalSourceStayDDMCCount_ = 0;
    std::size_t ddmcExternalSourceToIMCCount_ = 0;
    double ddmcExternalSourceThermalizedEnergy_ = 0.0;
    double ddmcExternalSourceToIMCEnergy_ = 0.0;
    double ddmcExternalSourceMinimumFaceOpticalDepth_ =
        std::numeric_limits<double>::infinity();
    std::size_t ddmcStepCount_ = 0;
    std::size_t ddmcLeakCount_ = 0;
    std::size_t ddmcCensusCount_ = 0;
    std::size_t ddmcUpscatterCount_ = 0;
    std::size_t ddmcFallbackCount_ = 0;
    std::size_t ddmcMovingInterfaceBypassCount_ = 0;
    double ddmcMovingInterfaceMaxFactor_ = 0.0;

    std::unordered_map<std::size_t, double> adaptiveSourceScores_;
    bool adaptiveSourceScoresEnabled_ = false;
    double adaptiveSourceStrength_ = 0.0;
    double adaptiveSourceMaxFactor_ = 1.0;
    double adaptiveSourceLearnedReserveFrac_ = 0.0;
    double adaptiveSourceLearnedMinFactor_ = 1.0;
    double adaptiveSourceObserverBudgetMultiplier_ = 1.0;
    std::size_t adaptiveSourceLearnedMinPhotons_ = 0;
    std::size_t adaptiveSourceLearnedMaxPhotons_ = 0;
    double adaptiveSourceScorePower_ = 1.0;

    std::unordered_map<std::size_t, GroupArray> adaptiveSourceCellGroupScores_;
    bool adaptiveSourceCellGroupScoresEnabled_ = false;
    double adaptiveGroupStrength_ = 0.0;
    double adaptiveGroupPdfFloor_ = 0.0;
    double adaptiveGroupMaxBias_ = 1.0;
    double adaptiveGroupMaxWeightCorrection_ = 1.0;

    bool sourceEmissionControlEnabled_ = false;
    bool sourceEmissionUseLearnedScores_ = false;
    bool sourceEmissionIncludeUniformBase_ = true;
    std::size_t sourceEmissionBaseMultiplier_ = 1;
    std::size_t sourceEmissionLearnedBoostFactor_ = 20;
    std::size_t sourceEmissionLearnedExtraBudget_ = 0;
};

// ============================================================
// Constructor
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::RadiationIMC(
    const GridT &grid,
    const std::shared_ptr<BoundaryCond> &boundary,
    std::vector<CellT> &cells,
    std::vector<ExtensivesT> &extensives,
    std::shared_ptr<EOST> eos,
    std::shared_ptr<OpacityModel> opacity,
    Parameters parameters,
    TraitsT traits,
    PositionSamplerT positionSampler,
    std::uint64_t seed):
    Base(grid, boundary),
    cells_(cells),
    extensives_(extensives),
    eos_(std::move(eos)),
    opacity_(std::move(opacity)),
    parameters_(std::move(parameters)),
    traits_(std::move(traits)),
    positionSampler_(std::move(positionSampler)),
    rng_(seed),
    dist_(0.0, 1.0)
{
    if(this->parameters_.newPhotonsPerCell == 0)
    {
        StormError eo("RadiationIMC requires newPhotonsPerCell > 0");
        throw eo;
    }
    if(!this->eos_)
    {
        StormError eo("RadiationIMC requires a non-null EOS");
        throw eo;
    }
    if(!this->opacity_)
    {
        StormError eo("RadiationIMC requires a non-null opacity model");
        throw eo;
    }

    this->validateGridSizedState();
    this->rejectUnsupportedParameters();

    if(this->parameters_.energyBoundariesProvided)
    {
        this->energyBoundaries_ = this->parameters_.energyBoundaries;
    }
    else if(!this->cells_.empty())
    {
        this->energyBoundaries_ = this->traits_.energyBoundaries(this->cells_.front());
    }
    else
    {
        for(std::size_t g = 0; g < NumGroups + 1; ++g)
        {
            this->energyBoundaries_[g] = static_cast<double>(g);
        }
    }
    this->validateEnergyBoundaries();

    if(this->parameters_.withRandomWalk)
    {
        this->randomWalk_ = std::make_unique<RandomWalk>();
    }

#ifdef STORM_WITH_MPI
    int rank = 0;
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    this->rng_.seed(seed + static_cast<std::uint64_t>(rank) * 104729ULL);
    this->particleRngSeed_ =
        seed + static_cast<std::uint64_t>(rank) * 104729ULL;
#else
    this->particleRngSeed_ = seed;
#endif
    this->opacity_->reseed(seed + 1ULL);

    const std::size_t Ncells = this->grid.GetPointNo();
    this->resetTransportTallies(Ncells);
    this->Erad_time_avg_.assign(Ncells, 0.0);
}

// ============================================================
// Validation helpers
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::validateGridSizedState() const
{
    const std::size_t Ncells = this->grid.GetPointNo();
    if(this->cells_.size() < Ncells)
    {
        StormError eo("RadiationIMC cells vector is smaller than the grid cell count");
        eo.addEntry("Grid cells", Ncells);
        eo.addEntry("Cells size", this->cells_.size());
        throw eo;
    }
    if(this->extensives_.size() < Ncells)
    {
        StormError eo("RadiationIMC extensives vector is smaller than the grid cell count");
        eo.addEntry("Grid cells", Ncells);
        eo.addEntry("Extensives size", this->extensives_.size());
        throw eo;
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::validateEnergyBoundaries() const
{
    for(std::size_t g = 0; g < NumGroups; ++g)
    {
        if(!std::isfinite(this->energyBoundaries_[g]) ||
           !std::isfinite(this->energyBoundaries_[g + 1]) ||
           this->energyBoundaries_[g + 1] <= this->energyBoundaries_[g])
        {
            StormError eo("RadiationIMC energy boundaries must be finite and strictly increasing");
            eo.addEntry("Group", g);
            eo.addEntry("Lower", this->energyBoundaries_[g]);
            eo.addEntry("Upper", this->energyBoundaries_[g + 1]);
            throw eo;
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::rejectUnsupportedParameter(const std::string &name) const
{
    StormError eo("RadiationIMC option is planned but not implemented in the initial STORM port");
    eo.addEntry("Unsupported option", name);
    throw eo;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::rejectUnsupportedParameters() const
{
    if(this->parameters_.withCompton && this->parameters_.withDDMC)
    {
        StormError eo("RadiationIMC configuration is invalid: Compton and DDMC are incompatible");
        eo.addEntry("withCompton", true);
        eo.addEntry("withDDMC", true);
        eo.addEntry("Reason", "Compton group-changing transport has no DDMC derivation");
        throw eo;
    }
    if(this->parameters_.withDDMC && !this->boundary)
    {
        StormError eo("RadiationIMC DDMC requires a boundary-condition object");
        eo.addEntry("Reason", "DDMC precompute must classify every external face");
        throw eo;
    }
    if(this->parameters_.withRandomWalk &&
       (!std::isfinite(this->parameters_.rwMinCellOpticalDepth) ||
        this->parameters_.rwMinCellOpticalDepth <= 0.0))
    {
        StormError eo("RadiationIMC random-walk cell optical-depth threshold must be finite and positive");
        eo.addEntry("rwMinCellOpticalDepth", this->parameters_.rwMinCellOpticalDepth);
        throw eo;
    }
    if(this->parameters_.withRandomWalk &&
       (!std::isfinite(this->parameters_.rwMinParticleOpticalDepth) ||
        this->parameters_.rwMinParticleOpticalDepth <= 0.0))
    {
        StormError eo("RadiationIMC random-walk particle optical-depth threshold must be finite and positive");
        eo.addEntry("rwMinParticleOpticalDepth", this->parameters_.rwMinParticleOpticalDepth);
        throw eo;
    }
    if(this->parameters_.withDDMC &&
       (!std::isfinite(this->parameters_.ddmcMinCellOpticalDepth) ||
        this->parameters_.ddmcMinCellOpticalDepth <= 0.0))
    {
        StormError eo("RadiationIMC DDMC cell optical-depth threshold must be finite and positive");
        eo.addEntry("ddmcMinCellOpticalDepth", this->parameters_.ddmcMinCellOpticalDepth);
        throw eo;
    }
    if(this->parameters_.withDDMC &&
       (!std::isfinite(this->parameters_.ddmcMinParticleOpticalDepth) ||
        this->parameters_.ddmcMinParticleOpticalDepth <= 0.0))
    {
        StormError eo("RadiationIMC DDMC particle optical-depth threshold must be finite and positive");
        eo.addEntry("ddmcMinParticleOpticalDepth", this->parameters_.ddmcMinParticleOpticalDepth);
        throw eo;
    }
    if(this->parameters_.withDDMC &&
       (!std::isfinite(
            this->parameters_.ddmcExternalSourceMinFaceOpticalDepth) ||
        this->parameters_.ddmcExternalSourceMinFaceOpticalDepth <= 0.0))
    {
        StormError eo(
            "RadiationIMC DDMC external-source face optical-depth threshold must be finite and positive");
        eo.addEntry("ddmcExternalSourceMinFaceOpticalDepth",
                    this->parameters_.ddmcExternalSourceMinFaceOpticalDepth);
        throw eo;
    }
    if(this->parameters_.withDDMC &&
       (!(this->parameters_.ddmcMaxInterfaceVelocityOverC > 0.0) ||
        !std::isfinite(this->parameters_.ddmcMaxInterfaceVelocityOverC) ||
        !(this->parameters_.ddmcInterfaceTargetWeightRatio > 0.0) ||
        !std::isfinite(this->parameters_.ddmcInterfaceTargetWeightRatio) ||
        this->parameters_.ddmcMaxInterfaceSplits == 0 ||
        this->parameters_.ddmcMaxGroupCutoff == 0 ||
        this->parameters_.ddmcMaxGroupCutoff > NumGroups))
    {
        throw StormError(
            "RadiationIMC DDMC interface controls are outside their valid ranges");
    }
    if(this->parameters_.withDDMC &&
       (!std::isfinite(this->parameters_.ddmcMaxMovingInterfaceWeightCorrection) ||
        this->parameters_.ddmcMaxMovingInterfaceWeightCorrection <= 0.0))
    {
        StormError eo("RadiationIMC DDMC moving-interface weight correction cap must be finite and positive");
        eo.addEntry("ddmcMaxMovingInterfaceWeightCorrection",
                    this->parameters_.ddmcMaxMovingInterfaceWeightCorrection);
        throw eo;
    }
    if(this->parameters_.withDDMC && this->parameters_.ddmcUseMultigroupPGRW &&
       !this->parameters_.withMultigroupOpacity)
    {
        rejectUnsupportedParameter("ddmcUseMultigroupPGRW requires withMultigroupOpacity");
    }
    if(this->parameters_.withCompton && !this->parameters_.withMultigroupOpacity)
    {
        StormError eo("RadiationIMC Compton transport requires multigroup opacity");
        eo.addEntry("withCompton", true);
        eo.addEntry("withMultigroupOpacity", false);
        throw eo;
    }
    if(this->parameters_.withCompton && this->parameters_.withRandomWalk)
    {
        StormError eo("RadiationIMC configuration is invalid: Compton and random walk are incompatible");
        eo.addEntry("withCompton", true);
        eo.addEntry("withRandomWalk", true);
        eo.addEntry("Reason", "Compton event kernels are not represented by the random-walk closure");
        throw eo;
    }
    if(this->parameters_.postProcess.polarization.enabled &&
       !this->parameters_.postProcess.enabled &&
       !this->parameters_.withPolarization)
    {
        throw StormError("RadiationIMC post-process polarization requires postProcess.enabled");
    }
    if(this->polarizationEnabled())
    {
        if(this->parameters_.withCompton)
        {
            throw StormError("RadiationIMC polarization does not support Compton transport yet");
        }
#ifndef MONTECARLO_POLARIZATION
        throw StormError("RadiationIMC polarization requires a build with MONTECARLO_POLARIZATION");
#else
        auto const &polarization = this->parameters_.postProcess.polarization;
        if(polarization.manualScatteringsAfterAcceleration < 0 ||
           polarization.manualScatteringsAfterAcceleration > 128)
        {
            StormError eo("RadiationIMC polarization manual scatter count must be in [0, 128]");
            eo.addEntry("manualScatteringsAfterAcceleration",
                        polarization.manualScatteringsAfterAcceleration);
            throw eo;
        }
        if(!std::isfinite(polarization.depolarizationScatterings) ||
           polarization.depolarizationScatterings <= 0.0)
        {
            StormError eo("RadiationIMC polarization depolarizationScatterings must be finite and positive");
            eo.addEntry("depolarizationScatterings", polarization.depolarizationScatterings);
            throw eo;
        }
        if(polarization.acceleratedClosure != "damped_last_scatterings")
        {
            StormError eo("RadiationIMC polarization acceleratedClosure is unsupported");
            eo.addEntry("acceleratedClosure", polarization.acceleratedClosure);
            throw eo;
        }
#endif
    }
    if(this->parameters_.postProcess.enabled)
    {
        if(!std::isfinite(this->parameters_.postProcess.sourceDt) ||
           this->parameters_.postProcess.sourceDt <= 0.0)
        {
            StormError eo("RadiationIMC post-process sourceDt must be finite and positive");
            eo.addEntry("sourceDt", this->parameters_.postProcess.sourceDt);
            throw eo;
        }
        if(!std::isfinite(this->parameters_.postProcess.transportTime) ||
           this->parameters_.postProcess.transportTime <= 0.0)
        {
            StormError eo("RadiationIMC post-process transportTime must be finite and positive");
            eo.addEntry("transportTime", this->parameters_.postProcess.transportTime);
            throw eo;
        }
    }
}

// ============================================================
// Small utilities
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::randomUnitOpen()
{
    double value = this->dist_(this->rng_);
    if(value <= 0.0)
    {
        return std::numeric_limits<double>::min();
    }
    if(value >= 1.0)
    {
        return std::nextafter(1.0, 0.0);
    }
    return value;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::initializeParticleRNG(MCParticle &particle)
{
    // Queried once: this runs per emitted particle, and the rank cannot change.
    if(!this->creationRankCached_)
    {
#ifdef STORM_WITH_MPI
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            this->creationRank_ = static_cast<std::uint64_t>(rank);
            this->creationRankCached_ = true;
        }
        else
        {
            this->creationRank_ = 0;
        }
#else
        this->creationRank_ = 0;
        this->creationRankCached_ = true;
#endif
    }
    particle.rngKey = CounterRNG::makeKey(
        this->particleRngSeed_, this->creationRank_, this->sourceRngStreamCounter_++);
    particle.rngCounter = 0;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::randomUnitOpen(MCParticle &particle)
{
    if(particle.rngKey == std::numeric_limits<std::uint64_t>::max())
    {
        std::uint64_t creationRank = 0;
#ifdef STORM_WITH_MPI
        creationRank = static_cast<std::uint64_t>(
            std::max<rank_t>(particle.rank, 0));
#endif
        particle.rngKey = CounterRNG::makeKey(
            this->particleRngSeed_, creationRank,
            static_cast<std::uint64_t>(particle.id));
        particle.rngCounter = 0;
    }
    return CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sampleRandomVelocity(
    const CellT &cell, MCParticle &particle)
{
    const double random1 = this->randomUnitOpen(particle);
    const double random2 = this->randomUnitOpen(particle);
    return this->opacity_->getRandomVelocity(cell, random1, random2);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sampleScatterVelocity(
    const CellT &cell, MCParticle &particle)
{
    const double random1 = this->randomUnitOpen(particle);
    const double random2 = this->randomUnitOpen(particle);
    return this->opacity_->getNewScatterVelocity(
        cell, particle.velocity, particle.frequency, random1, random2);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::resetTransportTallies(std::size_t cellCount)
{
    this->pendingMaterialEnergy_.assign(cellCount, 0.0);
    this->pendingTotalEnergy_.assign(cellCount, 0.0);
    this->pendingMomentum_.assign(cellCount, PointT{});
    this->pendingRadiationEnergy_.assign(cellCount, 0.0);
    this->pendingGroupRadiationEnergy_.assign(cellCount, GroupArray{});
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tallyMaterialEnergy(
    std::size_t cellIndex, double energy, bool addToTotalEnergy)
{
    this->pendingMaterialEnergy_[cellIndex] += energy;
    if(addToTotalEnergy)
    {
        this->pendingTotalEnergy_[cellIndex] += energy;
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tallyMomentum(
    std::size_t cellIndex, const PointT &momentum)
{
    this->pendingMomentum_[cellIndex] += momentum;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tallyRadiationEnergy(
    std::size_t cellIndex, double integratedEnergy)
{
    this->pendingRadiationEnergy_[cellIndex] += integratedEnergy;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tallyGroupRadiationEnergy(
    std::size_t cellIndex, std::size_t group, double integratedEnergy)
{
    this->pendingGroupRadiationEnergy_[cellIndex][group] += integratedEnergy;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::applyTransportTallies()
{
    for(std::size_t i = 0; i < this->pendingMaterialEnergy_.size(); ++i)
    {
        this->extensives_[i].internal_energy += this->pendingMaterialEnergy_[i];
        radiation_imc_detail::addTotalEnergyIfPresent(
            this->extensives_[i], this->pendingTotalEnergy_[i]);
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            this->extensives_[i].momentum += this->pendingMomentum_[i];
        }
        this->Erad_time_avg_[i] += this->pendingRadiationEnergy_[i];
        if((this->parameters_.withEgTimeAvg ||
            this->parameters_.withCompton) &&
           this->parameters_.withMultigroupOpacity)
        {
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                this->Eg_time_avg_[i][group] +=
                    this->pendingGroupRadiationEnergy_[i][group];
            }
        }
    }
    this->pendingMaterialEnergy_.clear();
    this->pendingTotalEnergy_.clear();
    this->pendingMomentum_.clear();
    this->pendingRadiationEnergy_.clear();
    this->pendingGroupRadiationEnergy_.clear();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setInitialWeightFromWeight(MCParticle &particle) const
{
    particle.initialWeight = std::abs(particle.weight);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::density(std::size_t cellIndex) const
{
    if constexpr(radiation_imc_detail::has_member_density<CellT>::value)
    {
        return this->cells_[cellIndex].density;
    }
    else
    {
        static_assert(radiation_imc_detail::has_member_mass<ExtensivesT>::value,
                      "RadiationIMC requires CellT::density or ExtensivesT::mass");
        return this->extensives_[cellIndex].mass / this->grid.GetVolume(cellIndex);
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::specificInternalEnergy(std::size_t cellIndex) const
{
    static_assert(radiation_imc_detail::has_member_mass<ExtensivesT>::value,
                  "RadiationIMC requires ExtensivesT::mass for specific internal energy");
    return this->extensives_[cellIndex].internal_energy / this->extensives_[cellIndex].mass;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::totalRadiationEnergy(std::size_t cellIndex) const
{
    if constexpr(radiation_imc_detail::has_member_radiation_energy<CellT>::value &&
                 radiation_imc_detail::has_member_density<CellT>::value)
    {
        return this->cells_[cellIndex].Erad * this->cells_[cellIndex].density * this->grid.GetVolume(cellIndex);
    }
    else
    {
        const double extensiveRadiation = radiation_imc_detail::radiationEnergyIfPresent(this->extensives_[cellIndex]);
        if(extensiveRadiation > 0.0)
        {
            return extensiveRadiation;
        }
        if constexpr(radiation_imc_detail::has_member_radiation_energy<CellT>::value &&
                     radiation_imc_detail::has_member_mass<ExtensivesT>::value)
        {
            return this->cells_[cellIndex].Erad * this->extensives_[cellIndex].mass;
        }
        else
        {
            return 0.0;
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::throwIfNegativeInternalEnergy(std::size_t cellIndex, const std::string &where)
{
    double &E = this->extensives_[cellIndex].internal_energy;
    if(E >= 0.0)
    {
        return;
    }
    double volume = this->grid.GetVolume(cellIndex);
    double thermalScale = units::arad * std::pow(this->cells_[cellIndex].temperature, 4) * volume;
    if(thermalScale < 1e-30)
    {
        thermalScale = 1e-30;
    }
    double ratio = std::abs(E) / thermalScale;
    if(ratio < 0.1)
    {
        // A small negative excursion is Monte-Carlo roundoff/noise.  Do not
        // turn it into a zero-temperature cell: that changes the opacity by
        // many orders of magnitude (the Marshak opacity is proportional to
        // T^-4.5) and creates isolated hot/cold points in profiles.
        // RadiationCell keeps the material energy from before this step, so
        // use it as the positivity floor when available.
        if constexpr(radiation_imc_detail::has_member_internal_energy_density<CellT>::value)
        {
            E = std::max(0.0, this->cells_[cellIndex].internalEnergy);
        }
        else
        {
            E = 0.0;
        }
        return;
    }

    StormError eo("Negative material internal energy in RadiationIMC");
    eo.addEntry("Where", where);
    eo.addEntry("Cell index", cellIndex);
    eo.addEntry("Cell ID", radiation_imc_detail::cellID(this->cells_[cellIndex]));
    eo.addEntry("Internal energy", E);
    if constexpr(radiation_imc_detail::has_member_mass<ExtensivesT>::value)
    {
        eo.addEntry("Mass", this->extensives_[cellIndex].mass);
    }
    eo.addEntry("Density", this->density(cellIndex));
    eo.addEntry("Temperature", this->cells_[cellIndex].temperature);
    throw eo;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::depositMaterialEnergy(std::size_t cellIndex, double energy)
{
    if(this->parameters_.noHydroFeedback)
    {
        return;
    }
    this->tallyMaterialEnergy(cellIndex, energy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::synchronizeMaterialCell(std::size_t cellIndex)
{
    CellT &cell = this->cells_[cellIndex];
    const double volume = this->grid.GetVolume(cellIndex);
    const double specificEnergy = this->specificInternalEnergy(cellIndex);

    if constexpr(radiation_imc_detail::has_member_internal_energy_specific<CellT>::value)
    {
        cell.internal_energy = specificEnergy;
    }
    else if constexpr(radiation_imc_detail::has_member_internal_energy_density<CellT>::value)
    {
        cell.internalEnergy = this->extensives_[cellIndex].internal_energy;
    }

    const auto &tracers = this->traits_.tracers(cell);
    const auto &tracerNames = this->traits_.tracerNames(cell);
    cell.temperature = this->eos_->de2T(this->density(cellIndex), specificEnergy, tracers, tracerNames);
    if constexpr(radiation_imc_detail::has_member_pressure<CellT>::value)
    {
        cell.pressure = this->eos_->de2p(this->density(cellIndex), specificEnergy, tracers, tracerNames);
    }
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value && radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
    {
        if(this->parameters_.withHydro)
        {
            cell.velocity = this->extensives_[cellIndex].momentum / this->extensives_[cellIndex].mass;
            if constexpr(radiation_imc_detail::has_member_total_energy<ExtensivesT>::value)
            {
                this->extensives_[cellIndex].energy = this->extensives_[cellIndex].internal_energy
                    + 0.5 * ScalarProd(this->extensives_[cellIndex].momentum, this->extensives_[cellIndex].momentum)
                    / this->extensives_[cellIndex].mass;
            }
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::clampFrequencyToBounds(double &frequency) const
{
    frequency = std::clamp(frequency, this->energyBoundaries_[0], this->energyBoundaries_[NumGroups]);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setNewPhotonsPerCell(std::size_t n)
{
    this->parameters_.newPhotonsPerCell = n;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setAdaptiveSourceCellScores(
    std::unordered_map<std::size_t, double> scores, double strength, double maxFactor,
    double learnedReserveFrac, double learnedMinFactor,
    double observerBudgetMultiplier, std::size_t learnedMinPhotons,
    std::size_t learnedMaxPhotons, double scorePower)
{
    this->adaptiveSourceScores_ = std::move(scores);
    this->adaptiveSourceStrength_ = std::clamp(strength, 0.0, 1.0);
    this->adaptiveSourceMaxFactor_ = std::max(1.0, maxFactor);
    this->adaptiveSourceLearnedReserveFrac_ = std::clamp(learnedReserveFrac, 0.0, 1.0);
    this->adaptiveSourceLearnedMinFactor_ = std::max(1.0, learnedMinFactor);
    this->adaptiveSourceObserverBudgetMultiplier_ = std::max(1.0, observerBudgetMultiplier);
    this->adaptiveSourceLearnedMinPhotons_ = learnedMinPhotons;
    this->adaptiveSourceLearnedMaxPhotons_ = learnedMaxPhotons;
    this->adaptiveSourceScorePower_ =
        (scorePower > 0.0 && std::isfinite(scorePower)) ? scorePower : 1.0;
    this->adaptiveSourceScoresEnabled_ = !this->adaptiveSourceScores_.empty();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::clearAdaptiveSourceCellScores()
{
    this->adaptiveSourceScores_.clear();
    this->adaptiveSourceScoresEnabled_ = false;
    this->adaptiveSourceStrength_ = 0.0;
    this->adaptiveSourceMaxFactor_ = 1.0;
    this->adaptiveSourceLearnedReserveFrac_ = 0.0;
    this->adaptiveSourceLearnedMinFactor_ = 1.0;
    this->adaptiveSourceObserverBudgetMultiplier_ = 1.0;
    this->adaptiveSourceLearnedMinPhotons_ = 0;
    this->adaptiveSourceLearnedMaxPhotons_ = 0;
    this->adaptiveSourceScorePower_ = 1.0;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setAdaptiveSourceCellGroupScores(
    std::unordered_map<std::size_t, GroupArray> scores, double strength, double pdfFloor, double maxBias, double maxWeightCorrection)
{
    this->adaptiveSourceCellGroupScores_ = std::move(scores);
    this->adaptiveGroupStrength_ = std::clamp(strength, 0.0, 1.0);
    this->adaptiveGroupPdfFloor_ = std::clamp(pdfFloor, 0.0, 1.0);
    this->adaptiveGroupMaxBias_ = std::max(1.0, maxBias);
    this->adaptiveGroupMaxWeightCorrection_ = std::max(1.0, maxWeightCorrection);
    this->adaptiveSourceCellGroupScoresEnabled_ = !this->adaptiveSourceCellGroupScores_.empty() && this->adaptiveGroupStrength_ > 0.0;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::clearAdaptiveSourceCellGroupScores()
{
    this->adaptiveSourceCellGroupScores_.clear();
    this->adaptiveSourceCellGroupScoresEnabled_ = false;
    this->adaptiveGroupStrength_ = 0.0;
    this->adaptiveGroupPdfFloor_ = 0.0;
    this->adaptiveGroupMaxBias_ = 1.0;
    this->adaptiveGroupMaxWeightCorrection_ = 1.0;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setSourceEmissionControl(
    bool useLearnedScores, bool includeUniformBase, std::size_t baseMultiplier, std::size_t learnedBoostFactor, std::size_t learnedExtraBudget)
{
    this->sourceEmissionControlEnabled_ = true;
    this->sourceEmissionUseLearnedScores_ = useLearnedScores;
    this->sourceEmissionIncludeUniformBase_ = includeUniformBase;
    this->sourceEmissionBaseMultiplier_ = std::max<std::size_t>(1, baseMultiplier);
    this->sourceEmissionLearnedBoostFactor_ = std::max<std::size_t>(1, learnedBoostFactor);
    this->sourceEmissionLearnedExtraBudget_ = learnedExtraBudget;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::clearSourceEmissionControl()
{
    this->sourceEmissionControlEnabled_ = false;
    this->sourceEmissionUseLearnedScores_ = false;
    this->sourceEmissionIncludeUniformBase_ = true;
    this->sourceEmissionBaseMultiplier_ = 1;
    this->sourceEmissionLearnedBoostFactor_ = 20;
    this->sourceEmissionLearnedExtraBudget_ = 0;
}

// ============================================================
// Random Walk helpers
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::computeMinDistanceToFaces(
    std::size_t cellIndex, const PointT &location) const
{
    double Ro = std::numeric_limits<double>::max();
    const std::size_t begin = this->gridData.cellFaceOffsets[cellIndex];
    const std::size_t end = this->gridData.cellFaceOffsets[cellIndex + 1];
    for(std::size_t f = begin; f < end; ++f)
    {
        double d = ScalarProd(location - this->gridData.pointsOnFaces[f],
                              this->gridData.normals[f]);
        Ro = std::min(Ro, d);
    }
    return (Ro > 0.0) ? Ro : 0.0;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::computeCellSurfaceArea(std::size_t cellIndex) const
{
    double surfaceArea = 0.0;
    for(std::size_t faceIdx : this->grid.GetCellFaces(cellIndex))
    {
        surfaceArea += this->grid.GetArea(faceIdx);
    }
    return surfaceArea;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::precomputeRandomWalkData()
{
    const std::size_t Ncells = this->grid.GetPointNo();
    this->rwCellEligible_.assign(Ncells, false);
    this->rwCellTotalOpacity_.assign(Ncells, 0.0);
    if(this->parameters_.withMultigroupOpacity)
    {
        this->rwCellData_.resize(Ncells);
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        const CellT &cell = this->cells_[i];
        double scatOp = this->scatteringOpacities_[i];
        if(!std::isfinite(scatOp) || scatOp < 0.0)
        {
            StormError eo("RadiationIMC random-walk precompute received an invalid scattering opacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Scattering opacity", scatOp);
            throw eo;
        }
        double sigmaT_gray = this->planckOpacities_[i] + scatOp;
        this->rwCellTotalOpacity_[i] = sigmaT_gray;

        double surfaceArea = this->computeCellSurfaceArea(i);
        double volume = this->grid.GetVolume(i);
        double meanChordLength = (surfaceArea > 0.0) ? 4.0 * volume / surfaceArea : 0.0;

        if(!this->parameters_.withMultigroupOpacity)
        {
            this->rwCellEligible_[i] = (sigmaT_gray * meanChordLength >= this->parameters_.rwMinCellOpticalDepth);
        }
        else
        {
            GroupArray energyCenters = this->opacity_->getEnergyCenters(this->energyBoundaries_);
            double kT = units::k_boltz * cell.temperature;

            double totalSigABgAll = 0.0;
            double totalBgDiff = 0.0, sumBgSigADiff = 0.0, sumBgSigTDiff = 0.0, sumBgOverSigTDiff = 0.0;
            std::size_t cutoff = 0;
            bool foundNonDiffusive = false;

            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                double Bg = ddmc::PlanckBandMass(
                    this->energyBoundaries_, kT, g, g + 1);
                double sigA_g = this->opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double scatOp_g = this->opacity_->CalcScatteringOpacity(
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

                if(!foundNonDiffusive && sigT_g * meanChordLength >= this->parameters_.rwMinCellOpticalDepth)
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

            PGRWCellData &data = this->rwCellData_[i];
            if(cutoff > 0 && totalBgDiff > 0.0)
            {
                data.groupCutoff = cutoff;
                data.sigmaA_bar = sumBgSigADiff / totalBgDiff;
                data.sigmaT_bar = sumBgSigTDiff / totalBgDiff;
                data.D = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                data.gamma = (totalSigABgAll > 0.0) ? sumBgSigADiff / totalSigABgAll : 1.0;
                this->rwCellTotalOpacity_[i] = data.sigmaT_bar;
                this->rwCellEligible_[i] = true;
            }
            else
            {
                data = PGRWCellData{};
                this->rwCellEligible_[i] = false;
            }
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tryRandomWalkStep(
    MCParticle &particle, Functionality &functionality)
{
    std::size_t cellIndex = particle.cellIndex;
    CellT &cell = this->cells_[cellIndex];

    double Ro = this->computeMinDistanceToFaces(cellIndex, particle.location);

    double sigmaT, sigma_a_eff, D_phys, gamma_rw;
    bool isPGRW = this->parameters_.withMultigroupOpacity;
    std::size_t groupCutoff = 0;

    if(isPGRW)
    {
        const PGRWCellData &rwd = this->rwCellData_[cellIndex];
        sigmaT = rwd.sigmaT_bar;
        sigma_a_eff = rwd.sigmaA_bar;
        D_phys = rwd.D;
        gamma_rw = rwd.gamma;
        groupCutoff = rwd.groupCutoff;
    }
    else
    {
        sigmaT = this->rwCellTotalOpacity_[cellIndex];
        sigma_a_eff = this->planckOpacities_[cellIndex];
        D_phys = (sigmaT > 0.0) ? units::clight / (3.0 * sigmaT) : 0.0;
        gamma_rw = 1.0;
    }

    bool doRW = (Ro > 0.0 && sigmaT > 0.0 && D_phys > 0.0
                 && Ro * sigmaT >= this->parameters_.rwMinParticleOpticalDepth);

    if(doRW && isPGRW)
    {
        double cutoffEnergy = this->energyBoundaries_[groupCutoff];
        double coFreq = particle.frequency;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                double dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                coFreq *= dopplerShift;
            }
        }
        this->clampFrequencyToBounds(coFreq);
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
    double f = this->factorFleck_[cellIndex];
    double tauLeak = this->randomWalk_->sampleLeakTime(this->randomUnitOpen(particle));
    double tLeak = tauLeak * Ro * Ro / D_phys;

    double tCensus = particle.timeLeft;

    double tUpscatter = std::numeric_limits<double>::max();
    if(isPGRW && gamma_rw < 1.0 && sigma_a_eff > 0.0 && f > 0.0)
    {
        double xiUp = this->randomUnitOpen(particle);
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
    if(!this->parameters_.noHydroFeedback)
    {
        this->tallyMaterialEnergy(cellIndex, -rwExp * particle.weight);
    }
    if(rwAbsRate > 0.0)
    {
        this->tallyRadiationEnergy(
            cellIndex, particle.weight * rwExp * (-1.0 / rwAbsRate));
        if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
        {
            std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
            if(g < NumGroups)
            {
                this->tallyGroupRadiationEnergy(
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
        if(!this->parameters_.noHydroFeedback)
        {
            this->tallyMaterialEnergy(cellIndex, particle.weight);
        }
        return true;
    }

    constexpr double RW_PI = 3.14159265358979323846;
    double cosTheta = 2.0 * this->randomUnitOpen(particle) - 1.0;
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    double phi = 2.0 * RW_PI * this->randomUnitOpen(particle);
    PointT posDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);

    double displacement;
    if(rwEvent == RW_LEAK)
    {
        displacement = Ro;
    }
    else
    {
        double tauPos = D_phys * dt / (Ro * Ro);
        displacement = Ro * this->randomWalk_->sampleRadius(tauPos, this->randomUnitOpen(particle));
    }

    if(displacement > Ro * (1.0 + 1e-12))
    {
        displacement = Ro;
    }

    PointT rwCenter = particle.location;
    particle.location = rwCenter + displacement * posDir;

    static constexpr double nudge = 1e-6;
    particle.location = particle.location * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

    const std::size_t faceBegin = this->gridData.cellFaceOffsets[cellIndex];
    const std::size_t faceEnd = this->gridData.cellFaceOffsets[cellIndex + 1];
    for(std::size_t fi = faceBegin; fi < faceEnd; ++fi)
    {
        double d = ScalarProd(
            particle.location - this->gridData.pointsOnFaces[fi],
            this->gridData.normals[fi]);
        if(d < 0.0)
        {
            displacement *= 0.99;
            particle.location = rwCenter + displacement * posDir;
            fi = faceBegin - 1;
        }
    }

    particle.velocity = this->sampleRandomVelocity(cell, particle);

#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
    {
        MCParticle polarizationParticle = particle;
        polarizationParticle.velocity = oldVelocity;
        double dtCo = dt;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
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
            this->parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
            this->parameters_.postProcess.polarization.depolarizationScatterings,
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
        GroupArray cumOp = this->opacity_->GetCumulativeOpacity(cell, this->energyBoundaries_);
        double cdfAtCutoff = cumOp[groupCutoff - 1];
        double cdfTotal = cumOp[NumGroups - 1];
        if(cdfTotal > cdfAtCutoff)
        {
            double lo = cdfAtCutoff / cdfTotal;
            double xi = this->randomUnitOpen(particle);
            particle.frequency = this->opacity_->GetThermalEnergy(cell, lo + xi * (1.0 - lo), this->energyBoundaries_);
        }
        else
        {
            particle.frequency = std::nextafter(
                this->energyBoundaries_[groupCutoff],
                std::numeric_limits<double>::max());
        }
        this->clampFrequencyToBounds(particle.frequency);
    }

    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if(this->parameters_.withHydro && !this->parameters_.MMC)
        {
            radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
            if(this->parameters_.withMultigroupOpacity)
            {
                this->clampFrequencyToBounds(particle.frequency);
            }
#ifdef MONTECARLO_POLARIZATION
            if(this->polarizationEnabled())
            {
                particle.polarizationBasis = polarization::projectBasisToDirection(
                    particle.polarizationBasis, particle.velocity);
            }
#endif
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                if(!this->parameters_.diffusionPressureGradient && !this->parameters_.noHydroFeedback)
                {
                    this->tallyMomentum(
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

// ============================================================
// DDMC helpers
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::precomputeDDMCData()
{
    const std::size_t Ncells = this->grid.GetPointNo();
    this->ddmcCellData_.assign(Ncells, DDMCCellData{});
    this->ddmcLeakReciprocityResidualMax_ = 0.0;
    this->ddmcLeakReciprocityCheckCount_ = 0;
    this->ddmcResidentLeakCount_ = 0;
    this->ddmcTransportLeakCount_ = 0;
    this->ddmcRemoteResidentLeakCount_ = 0;
    this->ddmcMPIFaceFluxReductionCount_ = 0;
    this->ddmcLeakInvalidGeometryCount_ = 0;
    this->ddmcUnsupportedBoundaryFaceCount_ = 0;
    this->ddmcInterfaceIncidentCount_ = 0;
    this->ddmcInterfaceAdmittedCount_ = 0;
    this->ddmcInterfaceReflectedCount_ = 0;
    this->ddmcInterfaceGuAppliedCount_ = 0;
    this->ddmcInterfaceGuFallbackCount_ = 0;
    this->ddmcInterfaceBypassCount_ = 0;
    this->ddmcInterfaceSplitPacketCount_ = 0;
    this->ddmcInterfaceFluxTallyCount_ = 0;
    this->ddmcInterfaceMinimumMu_ = std::numeric_limits<double>::infinity();
    this->ddmcDiagnosticEvents_.clear();
    this->ddmcExternalSourceCandidateFaceCount_ = 0;
    this->ddmcExternalSourceAcceleratedFaceCount_ = 0;
    this->ddmcExternalSourceExplicitFallbackFaceCount_ = 0;
    this->ddmcExternalSourceInteriorExcludedCellCount_ = 0;
    this->ddmcExternalSourceThermalizationCount_ = 0;
    this->ddmcExternalSourceStayDDMCCount_ = 0;
    this->ddmcExternalSourceToIMCCount_ = 0;
    this->ddmcExternalSourceThermalizedEnergy_ = 0.0;
    this->ddmcExternalSourceToIMCEnergy_ = 0.0;
    this->ddmcExternalSourceMinimumFaceOpticalDepth_ =
        std::numeric_limits<double>::infinity();

    // Eligibility is exchanged separately from the local cell data.  This
    // is important for Voronoi/MPI grids: a ghost index is not a local cell
    // index and must never index ddmcCellData_.
    const std::size_t pointCount = std::max(
        this->grid.GetTotalPointNumber(), this->grid.getMeshPoints().size());
    this->ddmcPointEligible_.assign(pointCount, 0);
    this->ddmcPointDiffusionCoefficient_.assign(pointCount, 0.0);
    this->ddmcPointSigmaDiffusion_.assign(pointCount, 0.0);
    this->ddmcPointSigmaParticleGate_.assign(pointCount, 0.0);
    this->ddmcPointGroupCutoff_.assign(pointCount, 0);
    this->ddmcPointVelocity_.assign(pointCount, PointT{});
    this->ddmcPointCellID_.assign(
        pointCount, std::numeric_limits<std::size_t>::max());

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData_[i];
        const CellT &cell = this->cells_[i];
        data.eligibilityReason = ddmc::EligibilityReason::InvalidGeometry;
        double scatOp = this->scatteringOpacities_[i];
        if(!std::isfinite(scatOp) || scatOp < 0.0)
        {
            StormError eo("RadiationIMC DDMC precompute received an invalid scattering opacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Scattering opacity", scatOp);
            throw eo;
        }
        double volume = this->grid.GetVolume(i);
        double surfaceArea = this->computeCellSurfaceArea(i);
        if(volume <= 0.0 || surfaceArea <= 0.0)
        {
            continue;
        }
        double meanChordLength = 4.0 * volume / surfaceArea;

        if(this->parameters_.ddmcUseMultigroupPGRW &&
           this->parameters_.withMultigroupOpacity)
        {
            GroupArray energyCenters =
                this->opacity_->getEnergyCenters(this->energyBoundaries_);
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
                double sigA_g = this->opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double scatOp_g = this->opacity_->CalcScatteringOpacity(
                    cell, energyCenters[g]);
                if(!std::isfinite(sigA_g) || sigA_g < 0.0 ||
                   !std::isfinite(scatOp_g) || scatOp_g < 0.0)
                {
                    StormError eo("RadiationIMC DDMC precompute received an invalid multigroup opacity");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Group", g);
                    eo.addEntry("Absorption opacity", sigA_g);
                    eo.addEntry("Scattering opacity", scatOp_g);
                    throw eo;
                }
                double sigT_g = sigA_g + scatOp_g;
                double Bg = ddmc::PlanckBandMass(
                    this->energyBoundaries_, kT, g, g + 1);
                totalSigABg += sigA_g * Bg;
                if(!foundNonDiffusive && sigT_g * meanChordLength >= this->parameters_.ddmcMinCellOpticalDepth)
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
            if(cutoff > 0 && totalBgDiff > 0.0)
            {
                data.groupCutoff = std::min(
                    cutoff, this->parameters_.ddmcMaxGroupCutoff);
                data.sigmaA = sumBgSigADiff / totalBgDiff;
                data.sigmaT = sumBgSigTDiff / totalBgDiff;
                data.sigmaEnergyAbs = data.sigmaA;
                data.sigmaMomentum = data.sigmaT;
                data.sigmaDiffusion = data.sigmaT;
                data.sigmaParticleGate = data.sigmaT;
                data.sigmaGroupExit = data.sigmaT;
                data.diffusionCoefficient =
                    (sumBgOverSigTDiff > 0.0)
                    ? (units::clight / 3.0) *
                        sumBgOverSigTDiff / totalBgDiff
                    : 0.0;
                data.gamma = totalSigABg > 0.0
                    ? sumBgSigADiff / totalSigABg : 1.0;
            data.eligible =
                    data.sigmaParticleGate > 0.0 &&
                    data.sigmaParticleGate * meanChordLength >=
                        this->parameters_.ddmcMinCellOpticalDepth &&
                    data.diffusionCoefficient > 0.0;
            }
        }
        else
        {
            data.sigmaA = this->planckOpacities_[i];
            data.sigmaT = data.sigmaA + scatOp;
            data.sigmaEnergyAbs = data.sigmaA;
            data.sigmaMomentum = data.sigmaT;
            data.sigmaDiffusion = data.sigmaT;
            data.sigmaParticleGate = data.sigmaT;
            data.sigmaGroupExit = data.sigmaT;
            data.diffusionCoefficient = (data.sigmaDiffusion > 0.0)
                ? units::clight / (3.0 * data.sigmaDiffusion) : 0.0;
            data.gamma = 1.0;
            data.eligible = (data.sigmaParticleGate * meanChordLength >= this->parameters_.ddmcMinCellOpticalDepth
                             && data.diffusionCoefficient > 0.0);
        }

        if(!data.eligible)
        {
            data.eligibilityReason = data.diffusionCoefficient > 0.0
                ? ddmc::EligibilityReason::OpticallyThin
                : ddmc::EligibilityReason::NoDiffusionCoefficient;
        }

        // External-face exclusions are local properties and must be applied
        // before eligibility is sent to a neighboring rank.  Otherwise a
        // ghost can incorrectly advertise DDMC eligibility even though the
        // owner later rejects the cell because of an unsupported boundary.
        if(data.eligible)
        {
            for(std::size_t faceIdx : this->grid.GetCellFaces(i))
            {
                const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
                std::size_t const next = (neighbors.first == i)
                    ? neighbors.second : neighbors.first;
                if(this->grid.IsPointOutsideBox(next))
                {
                    DDMCBoundaryFaceBehavior const behavior =
                        this->boundary->getDDMCBoundaryFaceBehavior(
                            faceIdx, i, next);
                    if(behavior == DDMCBoundaryFaceBehavior::ReflectingRigid)
                    {
                        ++data.rigidBoundaryFaceCount;
                    }
                    else
                    {
                        ++data.unsupportedBoundaryFaceCount;
                        ++this->ddmcUnsupportedBoundaryFaceCount_;
                        if(data.firstUnsupportedBoundaryFace ==
                           std::numeric_limits<std::size_t>::max())
                        {
                            data.firstUnsupportedBoundaryFace = faceIdx;
                        }
                        data.boundaryExcluded = true;
                        data.eligible = false;
                        data.eligibilityReason =
                            ddmc::EligibilityReason::BoundaryExcluded;
                    }
                }
            }
        }

        this->ddmcPointEligible_[i] = data.eligible ? 1 : 0;
        this->ddmcPointDiffusionCoefficient_[i] = data.diffusionCoefficient;
        this->ddmcPointSigmaDiffusion_[i] = data.sigmaDiffusion;
        this->ddmcPointSigmaParticleGate_[i] = data.sigmaParticleGate;
        this->ddmcPointGroupCutoff_[i] = data.groupCutoff;
        this->ddmcPointCellID_[i] = radiation_imc_detail::ddmcStableCellID(
            this->grid, i, cell);
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            this->ddmcPointVelocity_[i] = cell.velocity;
        }
    }

    if(this->postProcessExternalSourceMode_)
    {
        if(this->postProcessExternalSourceLocalCellIndices_.size() !=
           this->postProcessExternalSources_.size())
        {
            throw StormError(
                "DDMC external source-to-cell map has inconsistent size");
        }
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            DDMCCellData &data = this->ddmcCellData_[i];
            std::size_t const cellID = radiation_imc_detail::ddmcStableCellID(
                this->grid, i, this->cells_[i]);
            if(this->postProcessExternalSourceInteriorCellIDs_.count(cellID))
            {
                data.externalSourceInteriorExcluded = true;
                data.eligible = false;
                this->ddmcPointEligible_[i] = 0;
                ++this->ddmcExternalSourceInteriorExcludedCellCount_;
            }
        }
        for(std::size_t sourceIndex = 0;
            sourceIndex < this->postProcessExternalSources_.size();
            ++sourceIndex)
        {
            PostProcessExternalSource const &source =
                this->postProcessExternalSources_[sourceIndex];
            std::size_t const i =
                this->postProcessExternalSourceLocalCellIndices_[sourceIndex];
            if(i >= Ncells)
            {
                throw StormError("DDMC external source-to-cell map is stale");
            }
            DDMCCellData &data = this->ddmcCellData_[i];
            ++data.externalSourceBoundaryFaceCount;
            ++this->ddmcExternalSourceCandidateFaceCount_;
            PointT const normal = source.outwardNormal /
                std::max(fastabs(source.outwardNormal),
                         std::numeric_limits<double>::min());
            double const faceDistance = std::abs(ScalarProd(
                this->grid.FaceCM(source.faceIndex) -
                    this->grid.GetMeshPoint(i), normal));
            double const faceTau = data.sigmaDiffusion * faceDistance;
            double const diagnosticFaceTau =
                (faceTau >= 0.0 && std::isfinite(faceTau)) ? faceTau : 0.0;
            data.minExternalSourceFaceOpticalDepth = std::min(
                data.minExternalSourceFaceOpticalDepth, diagnosticFaceTau);
            this->ddmcExternalSourceMinimumFaceOpticalDepth_ = std::min(
                this->ddmcExternalSourceMinimumFaceOpticalDepth_,
                diagnosticFaceTau);
            if(!(faceTau >=
                 this->parameters_.ddmcExternalSourceMinFaceOpticalDepth) ||
               !std::isfinite(faceTau))
            {
                data.externalSourceFaceOpticalDepthExcluded = true;
                data.eligible = false;
                this->ddmcPointEligible_[i] = 0;
            }
        }
        for(DDMCCellData const &data : this->ddmcCellData_)
        {
            if(data.externalSourceBoundaryFaceCount > 0 && !data.eligible)
            {
                this->ddmcExternalSourceExplicitFallbackFaceCount_ +=
                    data.externalSourceBoundaryFaceCount;
            }
        }
    }

#ifdef STORM_WITH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized)
    {
        ddmc::ExchangePointMetadata(this->grid, this->ddmcPointEligible_);
        ddmc::ExchangePointMetadata(
            this->grid, this->ddmcPointDiffusionCoefficient_);
        ddmc::ExchangePointMetadata(this->grid, this->ddmcPointSigmaDiffusion_);
        ddmc::ExchangePointMetadata(
            this->grid, this->ddmcPointSigmaParticleGate_);
        ddmc::ExchangePointMetadata(this->grid, this->ddmcPointGroupCutoff_);
        ddmc::ExchangePointMetadata(this->grid, this->ddmcPointVelocity_);
        ddmc::ExchangePointMetadata(this->grid, this->ddmcPointCellID_);
    }
#endif

    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        // Compute the velocity-divergence operator after ghost velocities are
        // available.  Static cells simply compile this block away.
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            DDMCCellData &data = this->ddmcCellData_[i];
            double const volume = this->grid.GetVolume(i);
            if(!(volume > 0.0))
                continue;
            PointT const center = this->grid.GetMeshPoint(i);
            double divergence = 0.0;
            double maxJump = 0.0;
            for(std::size_t faceIndex : this->grid.GetCellFaces(i))
            {
                auto const &neighbors = this->grid.GetFaceNeighbors(faceIndex);
                std::size_t const next = neighbors.first == i
                    ? neighbors.second : neighbors.first;
                if(this->grid.IsPointOutsideBox(next))
                    continue;
                PointT normal = this->grid.Normal(faceIndex);
                double const normalMagnitude = fastabs(normal);
                if(!(normalMagnitude > 0.0))
                    continue;
                normal = normal / normalMagnitude;
                PointT const targetCenter = this->grid.GetMeshPoint(next);
                if(ScalarProd(normal, targetCenter - center) < 0.0)
                    normal = -normal;
                PointT targetVelocity = this->ddmcPointVelocity_[next];
                divergence += 0.5 * ScalarProd(
                    this->cells_[i].velocity + targetVelocity, normal) *
                    this->grid.GetArea(faceIndex);
                maxJump = std::max(maxJump,
                    fastabs(targetVelocity - this->cells_[i].velocity) *
                    units::inv_clight);
            }
            data.velocityDivergence = divergence / volume;
            data.maxFaceVelocityJumpOverC = maxJump;
        }
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData_[i];
        if(!data.eligible)
        {
            continue;
        }
        double volume = this->grid.GetVolume(i);
        PointT const cellCenter = this->grid.GetMeshPoint(i);
        double const sourceBandMass =
            (this->parameters_.ddmcUseMultigroupPGRW &&
             this->parameters_.withMultigroupOpacity)
            ? ddmc::PlanckBandMass(
                this->energyBoundaries_, units::k_boltz * this->cells_[i].temperature,
                0, data.groupCutoff) : 1.0;
        for(std::size_t faceIdx : this->grid.GetCellFaces(i))
        {
            auto const sourceFace =
                this->postProcessExternalSourceFaceIndex_.find(faceIdx);
            if(this->postProcessExternalSourceMode_ &&
               sourceFace != this->postProcessExternalSourceFaceIndex_.end())
            {
                if(sourceFace->second >=
                   this->postProcessExternalSources_.size())
                {
                    throw StormError(
                        "DDMC external-source face map contains an invalid source index");
                }
                PostProcessExternalSource const &source =
                    this->postProcessExternalSources_[sourceFace->second];
                if(radiation_imc_detail::ddmcStableCellID(
                       this->grid, i, this->cells_[i]) != source.cellID)
                {
                    throw StormError(
                        "DDMC attempted to build an external-source leak from the interior side");
                }
                PointT sourceNormal = source.outwardNormal /
                    std::max(fastabs(source.outwardNormal),
                             std::numeric_limits<double>::min());
                double const sourceDistanceToFace = std::abs(ScalarProd(
                    this->grid.FaceCM(faceIdx) - cellCenter, sourceNormal));
                double const area = this->grid.GetArea(faceIdx);
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
                ++this->ddmcExternalSourceAcceleratedFaceCount_;
                continue;
            }
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            std::size_t nextCellIndex = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(this->grid.IsPointOutsideBox(nextCellIndex))
            {
                if(this->boundary->getDDMCBoundaryFaceBehavior(
                       faceIdx, i, nextCellIndex) !=
                   DDMCBoundaryFaceBehavior::ReflectingRigid)
                {
                    data.boundaryExcluded = true;
                }
                continue;
            }
            PointT normal = this->grid.Normal(faceIdx);
            double const normalMag = fastabs(normal);
            double const area = this->grid.GetArea(faceIdx);
            if(!(normalMag > 0.0) || !std::isfinite(normalMag) ||
               !(area > 0.0) || !std::isfinite(area))
            {
                ++this->ddmcLeakInvalidGeometryCount_;
                continue;
            }
            normal = normal / normalMag;
            PointT faceCenter = this->grid.FaceCM(faceIdx);
            PointT outwardReference = this->grid.IsPointOutsideBox(nextCellIndex)
                ? faceCenter - cellCenter
                : this->grid.GetMeshPoint(nextCellIndex) - cellCenter;
            if(ScalarProd(normal, outwardReference) < 0.0)
            {
                normal = -normal;
            }
            double sourceDistance = std::abs(ScalarProd(
                faceCenter - cellCenter, normal));
            if(sourceDistance <= 0.0)
            {
                sourceDistance = 0.5 * std::abs(ScalarProd(
                    this->grid.GetMeshPoint(nextCellIndex) - cellCenter,
                    normal));
            }
            if(!(sourceDistance > 0.0) || !std::isfinite(sourceDistance))
            {
                ++this->ddmcLeakInvalidGeometryCount_;
                continue;
            }

            double targetDistance = 0.0;
            if(nextCellIndex < this->grid.getMeshPoints().size())
            {
                targetDistance = std::abs(ScalarProd(
                    this->grid.GetMeshPoint(nextCellIndex) - faceCenter,
                    normal));
            }

            bool const targetEligible =
                nextCellIndex < this->ddmcPointEligible_.size() &&
                this->ddmcPointEligible_[nextCellIndex] != 0;
            double internalRate = 0.0;
            double conductance = 0.0;
            if(targetEligible && targetDistance > 0.0)
            {
                conductance = ddmc::TwoSidedConductance(
                    area, sourceDistance,
                    data.diffusionCoefficient, targetDistance,
                    this->ddmcPointDiffusionCoefficient_[nextCellIndex]);
                internalRate = conductance / volume;
            }

            double boundaryRate = ddmc::BoundaryLeakRate(
                area, volume, data.sigmaDiffusion,
                sourceDistance, units::clight);
            std::size_t const targetCutoff =
                nextCellIndex < this->ddmcPointGroupCutoff_.size()
                ? this->ddmcPointGroupCutoff_[nextCellIndex] : 0;
            double ddmcFraction = 0.0;
            if(targetEligible && internalRate > 0.0)
            {
                if(!(this->parameters_.ddmcUseMultigroupPGRW &&
                     this->parameters_.withMultigroupOpacity) ||
                   targetCutoff >= data.groupCutoff)
                {
                    ddmcFraction = 1.0;
                }
                else if(targetCutoff > 0 && sourceBandMass > 0.0)
                {
                    ddmcFraction = std::clamp(
                        ddmc::PlanckBandMass(
                            this->energyBoundaries_,
                            units::k_boltz * this->cells_[i].temperature,
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
                this->ddmcPointEligible_.size() &&
                this->ddmcPointEligible_[face.nextCellIndex] != 0;
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
                continue;
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
        this->ddmcPointEligible_[i] =
            this->ddmcCellData_[i].eligible ? 1 : 0;
    }
    ddmc::ExchangePointMetadata(this->grid, this->ddmcPointEligible_);
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        refreshMixedFaceChannels(this->ddmcCellData_[i]);
    }
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        this->ddmcPointEligible_[i] =
            this->ddmcCellData_[i].eligible ? 1 : 0;
    }
    ddmc::ExchangePointMetadata(this->grid, this->ddmcPointEligible_);
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        refreshMixedFaceChannels(this->ddmcCellData_[i]);
    }

    // Deterministic local reciprocity check.  The same identity is used by
    // the distributed validation, where target coefficients come from the
    // exchanged ghost arrays above.
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        for(const DDMCFaceLeak &forward : this->ddmcCellData_[i].faceLeaks)
        {
            std::size_t const j = forward.nextCellIndex;
            if(forward.kind != ddmc::FaceKind::Internal || j >= Ncells || j <= i)
                continue;
            for(const DDMCFaceLeak &reverse : this->ddmcCellData_[j].faceLeaks)
            {
                if(reverse.faceIndex != forward.faceIndex ||
                   reverse.nextCellIndex != i ||
                   reverse.kind != ddmc::FaceKind::Internal)
                    continue;
                double const residual = ddmc::ReciprocityResidual(
                    this->grid.GetVolume(i), forward.internalRate,
                    this->grid.GetVolume(j), reverse.internalRate);
                this->ddmcLeakReciprocityResidualMax_ = std::max(
                    this->ddmcLeakReciprocityResidualMax_, residual);
                ++this->ddmcLeakReciprocityCheckCount_;
                break;
            }
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::addDDMCFluxContribution(
    std::size_t cellIndex, const PointT &contribution)
{
    if(cellIndex < this->ddmcFluxRhsIntegrated_.size())
    {
        this->ddmcFluxRhsIntegrated_[cellIndex] += contribution;
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::applyDDMCMomentumFeedback(double fullDt)
{
    (void)fullDt;
    if constexpr(!radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
    {
        return;
    }
    else
    {
        if(this->parameters_.noHydroFeedback || !this->parameters_.withHydro ||
           this->parameters_.diffusionPressureGradient)
            return;

#ifdef STORM_WITH_MPI
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized)
        {
            ddmc::ReducePointContributions(
                this->grid, this->ddmcFluxRhsIntegrated_);
            ++this->ddmcMPIFaceFluxReductionCount_;
        }
#endif

        for(std::size_t i = 0; i < this->grid.GetPointNo(); ++i)
        {
            if(i >= this->ddmcCellData_.size() ||
               i >= this->ddmcFluxRhsIntegrated_.size())
                continue;
            DDMCCellData const &data = this->ddmcCellData_[i];
            PointT const rhs = this->ddmcFluxRhsIntegrated_[i];
            if(!data.eligible || !(data.sigmaMomentum > 0.0) ||
               !(data.faceAreaSum > 0.0) || !(fastabs(rhs) > 0.0))
                continue;

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
                ++this->ddmcMomentumMatrixFallbackCount_;
            }

            PointT const deltaP = data.sigmaMomentum *
                this->grid.GetVolume(i) * units::inv_clight * fluxDt;
            if(!(std::isfinite(deltaP[0]) && std::isfinite(deltaP[1]) &&
                 std::isfinite(deltaP[2])))
                continue;
            this->extensives_[i].momentum += deltaP;
            ++this->ddmcMomentumFeedbackCount_;
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tryDDMCStep(
    MCParticle &particle, Functionality &functionality)
{
    std::size_t cellIndex = particle.cellIndex;
    bool const packetInDDMC = particle.radiationState.isDDMC();
    bool convertedIncomingToComoving = false;
    bool useComovingFrame = false;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        useComovingFrame =
            (this->parameters_.withHydro && !this->parameters_.MMC) ||
            (this->parameters_.postProcess.enabled &&
             this->parameters_.postProcess.useCellVelocities);
    }

    auto finalizePolarization = [&](const PointT &finalVelocity)
    {
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
        {
            ParticleCounterEngine polarizationEngine(
                particle.rngKey, particle.rngCounter);
            std::uniform_real_distribution<double> polarizationUnit(0.0, 1.0);
            polarization::finalizeAcceleratedPolarizationHistory<PointT>(
                particle, finalVelocity,
                this->parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
                this->parameters_.postProcess.polarization.depolarizationScatterings,
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
            return;

        bool const wasResident = particle.radiationState.isResident();
        bool const wasComoving = particle.radiationState.isComoving();
        if(wasResident || packetInDDMC)
        {
            particle.location = this->grid.GetMeshPoint(cellIndex);
            if(sampleDirection)
            {
                particle.velocity = this->sampleRandomVelocity(
                    this->cells_[cellIndex], particle);
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
                    particle, this->cells_[cellIndex]);
            }
        }
        particle.radiationState.clearDDMC();
        particle.initialWeight = std::abs(particle.weight);
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
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
    if(packetInDDMC && !this->parameters_.withDDMC)
    {
        exitDDMCToTransport(true);
        functionality.change = ParticleStatus::NO_CELL_MOVE;
        return true;
    }
    if(cellIndex >= this->ddmcCellData_.size())
    {
        exitDDMCToTransport(true);
        return false;
    }
    if(!particle.radiationState.invariantHolds())
    {
        exitDDMCToTransport(true);
        ++this->ddmcFallbackCount_;
        return false;
    }
    DDMCCellData const &data = this->ddmcCellData_[cellIndex];
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
            this->ddmcPointCellID_.size()
            ? this->ddmcPointCellID_[cellIndex]
            : std::numeric_limits<std::size_t>::max();
        std::size_t const currentCellID = exchangedCellID ==
            std::numeric_limits<std::size_t>::max()
            ? cellIndex : exchangedCellID;
        if(currentCellID == particle.radiationState.bypassCellID)
        {
            ++this->ddmcFallbackCount_;
            return false;
        }
    }

    double Ro = this->computeMinDistanceToFaces(cellIndex, particle.location);
    if(!particle.radiationState.isResident() &&
       Ro * data.sigmaParticleGate <
           this->parameters_.ddmcMinParticleOpticalDepth)
    {
        ++this->ddmcFallbackCount_;
        return false;
    }

    if(this->parameters_.ddmcUseMultigroupPGRW && this->parameters_.withMultigroupOpacity)
    {
        if(data.groupCutoff == 0 || data.groupCutoff > NumGroups)
        {
            exitDDMCToTransport(true);
            ++this->ddmcFallbackCount_;
            return false;
        }
        MCParticle frequencyProbe = particle;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(!particle.radiationState.isResident() && useComovingFrame)
            {
                radiation_imc_detail::lorentzTransformToComoving<PointT>(
                    frequencyProbe, this->cells_[cellIndex]);
            }
        }
        double coFreq = frequencyProbe.frequency;
        this->clampFrequencyToBounds(coFreq);
        if(coFreq >= this->energyBoundaries_[data.groupCutoff])
        {
            exitDDMCToTransport(true);
            ++this->ddmcFallbackCount_;
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
                    particle, this->cells_[cellIndex]);
                this->clampFrequencyToBounds(particle.frequency);
                convertedIncomingToComoving = true;
            }
        }
    }

#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
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
                    particle, this->cells_[cellIndex]);
                this->clampFrequencyToBounds(particle.frequency);
                particle.initialWeight = std::abs(particle.weight);
            }
        }
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
        {
            polarization::initializeIfNeeded<PointT>(particle);
            particle.polarizationBasis = polarization::projectBasisToDirection(
                particle.polarizationBasis, particle.velocity);
        }
#endif
    };

    double f = this->factorFleck_[cellIndex];
    double upscatterRate = 0.0;
    if(this->parameters_.ddmcUseMultigroupPGRW && data.gamma < 1.0 &&
       data.sigmaEnergyAbs > 0.0 &&
       (f > 0.0 || this->postProcessExternalSourceMode_))
    {
        upscatterRate = units::clight * (1.0 - f) * data.sigmaEnergyAbs *
            (1.0 - data.gamma);
    }
    double eventRate = data.totalLeakRate + upscatterRate;
    if(eventRate <= 0.0)
    {
        exitDDMCToTransport(true);
        ++this->ddmcFallbackCount_;
        return false;
    }

    double tEvent = -std::log(this->randomUnitOpen(particle)) / eventRate;
    double tCensus = particle.timeLeft;
    double tCutoff = std::numeric_limits<double>::max();
    if(this->parameters_.ddmcUseMultigroupPGRW &&
       data.groupCutoff > 0 && data.groupCutoff <= NumGroups &&
       data.velocityDivergence < 0.0)
    {
        double frequency = particle.frequency;
        this->clampFrequencyToBounds(frequency);
        double const cutoffFrequency =
            this->energyBoundaries_[data.groupCutoff];
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
    if(this->polarizationEnabled())
    {
        double const fHistory = this->factorFleck_[cellIndex];
        double const scatteringOpacity =
            this->scatteringOpacities_[cellIndex];
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
        if(this->parameters_.withHydro && !this->parameters_.MMC &&
           data.velocityDivergence != 0.0)
        {
            double const logShift = -data.velocityDivergence * dt / 3.0;
            if(std::isfinite(logShift) && logShift != 0.0)
            {
                double const boundedLogShift = std::clamp(logShift, -50.0, 50.0);
                double const shift = std::exp(boundedLogShift);
                particle.frequency *= shift;
                particle.weight *= shift;
                this->clampFrequencyToBounds(particle.frequency);
            }
        }
    }

    double absRate = data.sigmaEnergyAbs * f * units::clight;
    double oldWeight = particle.weight;
    double expFactor = std::expm1(-dt * absRate);

    if(!this->parameters_.noHydroFeedback)
    {
        double const absorbedEnergy = -expFactor * oldWeight;
        this->tallyMaterialEnergy(cellIndex, absorbedEnergy);
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                     radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro &&
               !this->parameters_.diffusionPressureGradient)
            {
                this->tallyMomentum(
                    cellIndex, absorbedEnergy * this->cells_[cellIndex].velocity *
                    units::inv_clight2);
            }
        }
    }

    double integratedForTally = (absRate > 0.0)
        ? oldWeight * expFactor * (-1.0 / absRate)
        : oldWeight * dt;
    this->tallyRadiationEnergy(cellIndex, integratedForTally);

    if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
    {
        if(this->parameters_.ddmcUseMultigroupPGRW && data.groupCutoff > 0 &&
           data.groupCutoff <= NumGroups)
        {
            double const kT = units::k_boltz *
                this->cells_[cellIndex].temperature;
            double const bandMass = ddmc::PlanckBandMass(
                this->energyBoundaries_, kT, 0, data.groupCutoff);
            if(bandMass > 0.0)
            {
                for(std::size_t g = 0; g < data.groupCutoff; ++g)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        this->energyBoundaries_, kT, g, g + 1);
                    this->tallyGroupRadiationEnergy(
                        cellIndex, g,
                        integratedForTally * groupMass / bandMass);
                }
            }
        }
        else
        {
            std::size_t g = this->opacity_->findGroup(
                particle.frequency, this->energyBoundaries_);
            if(g < NumGroups)
            {
                this->tallyGroupRadiationEnergy(
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
        if(!this->parameters_.noHydroFeedback)
        {
            this->tallyMaterialEnergy(cellIndex, particle.weight);
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                         radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(this->parameters_.withHydro &&
                   !this->parameters_.diffusionPressureGradient)
                {
                    this->tallyMomentum(
                        cellIndex, particle.weight * this->cells_[cellIndex].velocity *
                        units::inv_clight2);
                }
            }
        }
        ++this->ddmcStepCount_;
        return true;
    }

    ++this->ddmcStepCount_;

    if(cutoffEvent)
    {
        particle.frequency = std::nextafter(
            this->energyBoundaries_[data.groupCutoff],
            std::numeric_limits<double>::max());
        particle.velocity = this->sampleRandomVelocity(
            this->cells_[cellIndex], particle);
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
            this->addDDMCFluxContribution(
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
        particle.location = this->grid.GetMeshPoint(cellIndex);
        particle.velocity = this->sampleRandomVelocity(
            this->cells_[cellIndex], particle);
    }

    if(censusEvent)
    {
        // Census is a representation boundary.  Reconstruct a valid IMC
        // packet before returning it to the manager; the next time step must
        // not carry a stale DDMC direction or frame.
        particle.location = this->grid.GetMeshPoint(cellIndex);
        if(this->parameters_.withMultigroupOpacity)
        {
            bool sampledResidentBand = false;
            if(this->parameters_.ddmcUseMultigroupPGRW &&
               data.groupCutoff > 0 && data.groupCutoff <= NumGroups)
            {
                double const kT = units::k_boltz *
                    this->cells_[cellIndex].temperature;
                double const bandMass = ddmc::PlanckBandMass(
                    this->energyBoundaries_, kT, 0, data.groupCutoff);
                if(!(bandMass > 0.0))
                {
                    StormError eo("RadiationIMC DDMC census has no resident-band Planck mass");
                    eo.addEntry("Cell index", cellIndex);
                    eo.addEntry("Group cutoff", data.groupCutoff);
                    throw eo;
                }
                double remaining = this->randomUnitOpen(particle) * bandMass;
                for(std::size_t group = 0; group < data.groupCutoff; ++group)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        this->energyBoundaries_, kT, group, group + 1);
                    if(remaining <= groupMass || group + 1 == data.groupCutoff)
                    {
                        double const localRandom = groupMass > 0.0
                            ? std::clamp(remaining / groupMass, 0.0, 1.0)
                            : this->randomUnitOpen(particle);
                        particle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                            this->cells_[cellIndex], group, localRandom,
                            this->energyBoundaries_);
                        double const upperBand =
                            this->energyBoundaries_[data.groupCutoff];
                        particle.frequency = std::min(
                            particle.frequency,
                            std::nextafter(upperBand,
                                this->energyBoundaries_[0]));
                        sampledResidentBand = true;
                        break;
                    }
                    remaining -= groupMass;
                }
            }
            if(!sampledResidentBand)
            {
                particle.frequency = this->opacity_->GetThermalEnergy(
                    this->cells_[cellIndex], this->randomUnitOpen(particle),
                    this->energyBoundaries_);
            }
            this->clampFrequencyToBounds(particle.frequency);
        }
        particle.velocity = this->sampleRandomVelocity(
            this->cells_[cellIndex], particle);
        finalizePolarization(particle.velocity);
        convertResidentToLab();
        particle.radiationState.clearDDMC();
        functionality.change = ParticleStatus::DONE;
        ++this->ddmcCensusCount_;
        return true;
    }

    double eventPick = this->randomUnitOpen(particle) * eventRate;
    if(eventPick <= data.totalLeakRate)
    {
        double facePick = this->randomUnitOpen(particle) * data.totalLeakRate;
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

        PointT leakFaceCenter = this->grid.FaceCM(chosen->faceIndex);
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
                this->postProcessExternalSourceFaceIndex_.find(
                    chosen->faceIndex);
            if(sourceFace ==
                   this->postProcessExternalSourceFaceIndex_.end() ||
               sourceFace->second >=
                   this->postProcessExternalSources_.size())
            {
                throw StormError(
                    "DDMC selected an external-source face without an installed source");
            }
            PostProcessExternalSource const &source =
                this->postProcessExternalSources_[sourceFace->second];
            if(radiation_imc_detail::ddmcStableCellID(
                   this->grid, cellIndex, this->cells_[cellIndex]) !=
               source.cellID)
            {
                throw StormError(
                    "DDMC selected an external-source face from the wrong transport cell");
            }

            nOut = source.outwardNormal /
                std::max(fastabs(source.outwardNormal),
                         std::numeric_limits<double>::min());
            double const eventEnergy = std::abs(particle.weight);
            ++this->ddmcExternalSourceThermalizationCount_;
            this->ddmcExternalSourceThermalizedEnergy_ += eventEnergy;
            if(this->parameters_.withMultigroupOpacity)
            {
                particle.frequency =
                    this->samplePostProcessExternalSourcePlanckFrequency(
                        this->cells_[cellIndex]);
                this->clampFrequencyToBounds(particle.frequency);
            }

            bool const leaveDDMCBand =
                this->parameters_.ddmcUseMultigroupPGRW &&
                data.groupCutoff < NumGroups &&
                particle.frequency >=
                    this->energyBoundaries_[data.groupCutoff];
            if(leaveDDMCBand)
            {
                particle.velocity = units::clight *
                    this->samplePostProcessExternalSourceDirection(nOut, particle);
#ifdef MONTECARLO_POLARIZATION
                if(this->polarizationEnabled())
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
                            particle, this->cells_[cellIndex]);
                        this->clampFrequencyToBounds(particle.frequency);
                    }
                }
                static constexpr double nudge = 1.0e-8;
                particle.location = (1.0 - nudge) * source.location +
                    nudge * this->grid.GetMeshPoint(cellIndex);
                particle.radiationState.clearDDMC();
                particle.initialWeight = std::abs(particle.weight);
                functionality.change = ParticleStatus::NO_CELL_MOVE;
                ++this->ddmcExternalSourceToIMCCount_;
                this->ddmcExternalSourceToIMCEnergy_ += eventEnergy;
                ++this->ddmcLeakCount_;
                ++this->ddmcTransportLeakCount_;
                return true;
            }

            particle.location = this->grid.GetMeshPoint(cellIndex);
            particle.velocity = this->sampleRandomVelocity(
                this->cells_[cellIndex], particle);
#ifdef MONTECARLO_POLARIZATION
            if(this->polarizationEnabled())
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
            ++this->ddmcExternalSourceStayDDMCCount_;
            ++this->ddmcLeakCount_;
            ++this->ddmcResidentLeakCount_;
            return true;
        }

        constexpr double DDMC_PI = 3.14159265358979323846;
        bool const useDDMCChannel =
            chosen->ddmcRate > 0.0 &&
            this->randomUnitOpen(particle) < chosen->ddmcRate / chosen->rate;
        double mu = useDDMCChannel
            ? ddmc::SampleAsymptoticMu(this->randomUnitOpen(particle))
            : std::sqrt(this->randomUnitOpen(particle));
        double phiLeak = 2.0 * DDMC_PI * this->randomUnitOpen(particle);
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
        if(!targetDDMC && this->parameters_.ddmcUseMultigroupPGRW &&
           this->parameters_.withMultigroupOpacity)
        {
            std::size_t beginGroup = 0;
            if(chosen->targetDDMCEligible &&
               chosen->targetGroupCutoff > 0 &&
               chosen->targetGroupCutoff < data.groupCutoff)
            {
                beginGroup = chosen->targetGroupCutoff;
            }
            double const kT = units::k_boltz *
                this->cells_[cellIndex].temperature;
            double const bandMass = ddmc::PlanckBandMass(
                this->energyBoundaries_, kT, beginGroup, data.groupCutoff);
            if(bandMass > 0.0)
            {
                double remaining = this->randomUnitOpen(particle) * bandMass;
                for(std::size_t group = beginGroup;
                    group < data.groupCutoff; ++group)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        this->energyBoundaries_, kT, group, group + 1);
                    if(remaining <= groupMass || group + 1 == data.groupCutoff)
                    {
                        double const localRandom = groupMass > 0.0
                            ? std::clamp(remaining / groupMass, 0.0, 1.0)
                            : this->randomUnitOpen(particle);
                        particle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                            this->cells_[cellIndex], group, localRandom,
                            this->energyBoundaries_);
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
            if(this->parameters_.withHydro && !this->parameters_.MMC &&
               !targetDDMC)
            {
                CellT sourceCell = this->cells_[cellIndex];
                sourceCell.velocity = this->cells_[cellIndex].velocity;
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
            !this->parameters_.withMultigroupOpacity
            ? DDMC_DIAGNOSTIC_GREY_GROUP
            : this->opacity_->findGroup(
                particle.frequency, this->energyBoundaries_);
        this->recordDDMCDiagnosticEvent(
            targetDDMC ? DDMCDiagnosticEventKind::DDMCToDDMC
                       : DDMCDiagnosticEventKind::DDMCToIMC,
            cellIndex, chosen->nextCellIndex, chosen->faceIndex,
            diagnosticGroup, fluxWeightComoving, data.groupCutoff,
            chosen->targetGroupCutoff,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());

        PointT const fluxContribution = fluxWeightComoving * dir;
        this->addDDMCFluxContribution(cellIndex, fluxContribution);
        if(targetDDMC)
        {
            if(chosen->nextCellIndex < this->grid.GetPointNo())
            {
                this->addDDMCFluxContribution(
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
            ++this->ddmcResidentLeakCount_;
            if(chosen->nextCellIndex >= this->grid.GetPointNo())
            {
                ++this->ddmcRemoteResidentLeakCount_;
            }
        }
        else
        {
            ++this->ddmcTransportLeakCount_;
        }
        ++this->ddmcLeakCount_;
    }
    else
    {
        if(!this->parameters_.ddmcUseMultigroupPGRW)
        {
            finalizePolarization(particle.velocity);
            particle.radiationState.clearDDMC();
            functionality.change = ParticleStatus::DONE;
            ++this->ddmcCensusCount_;
            return true;
        }
        CellT &cell = this->cells_[cellIndex];
        double const kT = units::k_boltz * cell.temperature;
        double const upperBandMass = ddmc::PlanckBandMass(
            this->energyBoundaries_, kT, data.groupCutoff, NumGroups);
        if(!(upperBandMass > 0.0))
        {
            StormError eo("RadiationIMC DDMC upscatter has no representable upper frequency band");
            eo.addEntry("Cell index", cellIndex);
            eo.addEntry("Group cutoff", data.groupCutoff);
            eo.addEntry("Upper-band Planck mass", upperBandMass);
            throw eo;
        }
        double remaining = this->randomUnitOpen(particle) * upperBandMass;
        std::size_t selectedGroup = data.groupCutoff;
        for(std::size_t group = data.groupCutoff; group < NumGroups; ++group)
        {
            double const groupMass = ddmc::PlanckBandMass(
                this->energyBoundaries_, kT, group, group + 1);
            if(remaining <= groupMass || group + 1 == NumGroups)
            {
                selectedGroup = group;
                double const localRandom = groupMass > 0.0
                    ? std::clamp(remaining / groupMass, 0.0, 1.0)
                    : this->randomUnitOpen(particle);
                particle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                    cell, selectedGroup, localRandom, this->energyBoundaries_);
                break;
            }
            remaining -= groupMass;
        }
        this->clampFrequencyToBounds(particle.frequency);
        particle.velocity = this->sampleRandomVelocity(cell, particle);
        exitDDMCToTransport(false);
        functionality.change = ParticleStatus::NO_CELL_MOVE;
        ++this->ddmcUpscatterCount_;
        return true;
    }
    return true;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT,
                  PositionSamplerT>::recordDDMCDiagnosticEvent(
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
    if(!this->parameters_.withDDMC ||
       !this->parameters_.ddmcInterfaceDiagnostics)
    {
        return;
    }

    auto pointID = [this](std::size_t index)
    {
        if(index < this->ddmcPointCellID_.size() &&
           this->ddmcPointCellID_[index] !=
               std::numeric_limits<std::size_t>::max())
        {
            return this->ddmcPointCellID_[index];
        }
        if(index < this->cells_.size())
        {
            return radiation_imc_detail::ddmcStableCellID(
                this->grid, index, this->cells_[index]);
        }
        return std::numeric_limits<std::size_t>::max();
    };
    auto pointX = [this](std::size_t index)
    {
        if(index < this->grid.getMeshPoints().size())
            return static_cast<double>(this->grid.GetMeshPoint(index)[0]);
        return std::numeric_limits<double>::quiet_NaN();
    };

    std::size_t const sourceCellID = pointID(sourceCellIndex);
    std::size_t const targetCellID = pointID(targetCellIndex);
    DDMCDiagnosticEventKey const key{
        kind, faceIndex, sourceCellID, targetCellID, group};
    auto inserted = this->ddmcDiagnosticEvents_.emplace(
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
        entry.faceX = this->grid.FaceCM(faceIndex)[0];
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

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tryIMCToDDMCInterface(
    MCParticle &particle,
    Functionality &functionality,
    std::vector<MCParticle> &particlesToAdd,
    std::size_t sourceCellIndex,
    std::size_t targetCellIndex,
    std::size_t faceIndex)
{
    if(sourceCellIndex >= this->cells_.size() ||
       targetCellIndex >= this->ddmcPointEligible_.size() ||
       this->grid.IsPointOutsideBox(targetCellIndex) ||
       this->ddmcPointEligible_[targetCellIndex] == 0)
    {
        return false;
    }

    std::size_t const exchangedTargetID = this->ddmcPointCellID_[targetCellIndex];
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

    PointT normal = this->grid.Normal(faceIndex);
    double const normalMagnitude = fastabs(normal);
    if(!(normalMagnitude > 0.0) || !std::isfinite(normalMagnitude))
        return false;
    normal = normal / normalMagnitude;

    PointT const sourceCenter = this->grid.GetMeshPoint(sourceCellIndex);
    PointT const targetCenter = this->grid.GetMeshPoint(targetCellIndex);
    if(ScalarProd(normal, targetCenter - sourceCenter) < 0.0)
        normal = -normal;

    PointT faceVelocity{};
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        PointT targetVelocity = this->ddmcPointVelocity_[targetCellIndex];
        double const sourceDistance = std::abs(ScalarProd(
            this->grid.FaceCM(faceIndex) - sourceCenter, normal));
        double const targetDistance = std::abs(ScalarProd(
            targetCenter - this->grid.FaceCM(faceIndex), normal));
        double const distanceSum = sourceDistance + targetDistance;
        faceVelocity = distanceSum > 0.0
            ? (targetDistance * this->cells_[sourceCellIndex].velocity +
               sourceDistance * targetVelocity) / distanceSum
            : 0.5 * (this->cells_[sourceCellIndex].velocity + targetVelocity);
    }

    MCParticle faceComoving = particle;
    MCParticle targetComoving = particle;
#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
    {
        polarization::initializeIfNeeded<PointT>(faceComoving);
    }
#endif
    bool useVelocityFrames = false;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities))
        {
            CellT faceCell = this->cells_[sourceCellIndex];
            CellT targetCell = faceCell;
            faceCell.velocity = faceVelocity;
            targetCell.velocity = this->ddmcPointVelocity_[targetCellIndex];
            radiation_imc_detail::lorentzTransformToComoving<PointT>(
                faceComoving, faceCell);
#ifdef MONTECARLO_POLARIZATION
            if(this->polarizationEnabled())
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
        this->ddmcPointGroupCutoff_.size()
        ? this->ddmcPointGroupCutoff_[sourceCellIndex] : 0;
    std::size_t const targetGroupCutoff = targetCellIndex <
        this->ddmcPointGroupCutoff_.size()
        ? this->ddmcPointGroupCutoff_[targetCellIndex] : 0;
    std::size_t const diagnosticGroup = this->parameters_.withMultigroupOpacity
        ? this->opacity_->findGroup(
            targetComoving.frequency, this->energyBoundaries_)
        : DDMC_DIAGNOSTIC_GREY_GROUP;
    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCCandidate, sourceCellIndex,
        targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
        sourceGroupCutoff, targetGroupCutoff,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());

    if(this->parameters_.ddmcUseMultigroupPGRW &&
       this->parameters_.withMultigroupOpacity)
    {
        std::size_t const cutoff = this->ddmcPointGroupCutoff_[targetCellIndex];
        double frequency = targetComoving.frequency;
        this->clampFrequencyToBounds(frequency);
        if(cutoff == 0 || cutoff > NumGroups ||
           frequency >= this->energyBoundaries_[cutoff])
        {
            this->recordDDMCDiagnosticEvent(
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
        return false;
    double const mu = ScalarProd(faceComoving.velocity / speed, normal);
    if(!(mu > 0.0) || !std::isfinite(mu))
        return false;
    ++this->ddmcInterfaceIncidentCount_;
    this->ddmcInterfaceMinimumMu_ = std::min(
        this->ddmcInterfaceMinimumMu_, mu);

    double movingFactor = 1.0;
    auto bypassMovingInterface = [&]()
    {
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::IMCBypass, sourceCellIndex,
            targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
            sourceGroupCutoff, targetGroupCutoff, mu,
            std::numeric_limits<double>::quiet_NaN());
        ++this->ddmcMovingInterfaceBypassCount_;
        ++this->ddmcInterfaceBypassCount_;
        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = targetCellIndex;
        return true;
    };
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if(useVelocityFrames &&
           this->parameters_.ddmcUseMovingInterfaceCorrection)
        {
            double const betaNormal = -ScalarProd(faceVelocity, normal) *
                units::inv_clight;
            if(!std::isfinite(betaNormal) ||
               std::abs(betaNormal) >
                   this->parameters_.ddmcMaxInterfaceVelocityOverC)
            {
                ++this->ddmcInterfaceGuFallbackCount_;
                particle.radiationState.bypassCellID = targetID;
                return bypassMovingInterface();
            }
            movingFactor = ddmc::MovingFactor(mu, betaNormal);
            if(std::isfinite(movingFactor))
            {
                this->ddmcMovingInterfaceMaxFactor_ = std::max(
                    this->ddmcMovingInterfaceMaxFactor_, movingFactor);
            }
            if(!(movingFactor > 0.0) ||
               !std::isfinite(movingFactor) ||
               movingFactor > this->parameters_.ddmcMaxMovingInterfaceWeightCorrection)
            {
                ++this->ddmcInterfaceGuFallbackCount_;
                particle.radiationState.bypassCellID = targetID;
                return bypassMovingInterface();
            }
            ++this->ddmcInterfaceGuAppliedCount_;
        }
    }

    double const targetWeight =
        this->parameters_.ddmcInterfaceTargetWeightRatio *
        std::max(std::abs(particle.weight),
                 std::numeric_limits<double>::min());
    std::size_t requiredSplitCount = 1;
    if(targetWeight > 0.0)
    {
        requiredSplitCount = static_cast<std::size_t>(std::ceil(
            std::abs(faceComoving.weight * movingFactor) / targetWeight));
        requiredSplitCount = std::max<std::size_t>(1, requiredSplitCount);
    }
    if(requiredSplitCount > std::max<std::size_t>(
           1, this->parameters_.ddmcMaxInterfaceSplits))
    {
        particle.radiationState.bypassCellID = targetID;
        return bypassMovingInterface();
    }

    double const targetOpacity =
        this->ddmcPointSigmaDiffusion_[targetCellIndex];
    double const targetDistance = std::abs(ScalarProd(
        targetCenter - this->grid.FaceCM(faceIndex), normal));
    double const admission = ddmc::StaticAdmissionProbability(
        mu, targetOpacity, targetDistance);
    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCIncident, sourceCellIndex,
        targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
        sourceGroupCutoff, targetGroupCutoff, mu, admission);

    if(this->randomUnitOpen(particle) > admission)
    {
        ++this->ddmcInterfaceReflectedCount_;
        // Diffuse-albedo rejection stays in the source IMC cell.  The
        // incoming direction is not reflected specularly at a transport-
        // diffusion interface.
        constexpr double pi = 3.14159265358979323846;
        double const reflectedMu = std::sqrt(this->randomUnitOpen(particle));
        double const sinTheta = std::sqrt(
            std::max(0.0, 1.0 - reflectedMu * reflectedMu));
        double const phi = 2.0 * pi * this->randomUnitOpen(particle);
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
                CellT faceCell = this->cells_[sourceCellIndex];
                faceCell.velocity = faceVelocity;
                radiation_imc_detail::lorentzTransformToLab<PointT>(
                    faceComoving, faceCell);
            }
        }
        particle.velocity = faceComoving.velocity;
        particle.frequency = faceComoving.frequency;
        particle.weight = faceComoving.weight;
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
        {
            particle.stokesQ = faceComoving.stokesQ;
            particle.stokesU = faceComoving.stokesU;
            particle.polarizationBasis = polarization::projectBasisToDirection(
                faceComoving.polarizationBasis, particle.velocity);
            particle.polarizationInitialized = faceComoving.polarizationInitialized;
        }
#endif
        particle.location = (1.0 - 1.0e-10) * this->grid.FaceCM(faceIndex) +
            1.0e-10 * sourceCenter;
        functionality.change = ParticleStatus::NO_CELL_MOVE;
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::IMCReflected, sourceCellIndex,
            targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
            sourceGroupCutoff, targetGroupCutoff, mu, admission);
        return true;
    }

    faceComoving.weight *= movingFactor;
    targetComoving = faceComoving;
    ++this->ddmcInterfaceAdmittedCount_;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if(useVelocityFrames)
        {
            CellT faceCell = this->cells_[sourceCellIndex];
            CellT targetCell = faceCell;
            faceCell.velocity = faceVelocity;
            targetCell.velocity = this->ddmcPointVelocity_[targetCellIndex];
            targetComoving = faceComoving;
            radiation_imc_detail::lorentzTransformToLab<PointT>(
                targetComoving, faceCell);
            radiation_imc_detail::lorentzTransformToComoving<PointT>(
                targetComoving, targetCell);
#ifdef MONTECARLO_POLARIZATION
            if(this->polarizationEnabled())
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
    if(targetCellIndex >= this->grid.GetPointNo())
        splitCount = 1;
    targetComoving.weight /= static_cast<double>(splitCount);
    particle.weight = targetComoving.weight;
    particle.frequency = targetComoving.frequency;
    particle.initialWeight = std::abs(particle.weight);
#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
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
        if(targetCellIndex < this->grid.GetPointNo())
        {
            this->addDDMCFluxContribution(targetCellIndex, contribution);
            ++this->ddmcInterfaceFluxTallyCount_;
        }
        else
        {
            particle.radiationState.pendingFlux = contribution;
            particle.radiationState.set(
                RadiationTransportState<PointT>::PendingFlux);
        }
    }
    particle.location = this->grid.FaceCM(faceIndex);
    particle.velocity = this->sampleRandomVelocity(
        this->cells_[sourceCellIndex], particle);
#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
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
        ++this->ddmcInterfaceSplitPacketCount_;
    }
    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCAdmitted, sourceCellIndex,
        targetCellIndex, faceIndex, diagnosticGroup, admittedTargetWeight,
        sourceGroupCutoff, targetGroupCutoff, mu, admission);
    return true;
}

// ============================================================
// preStep
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::precomputeComptonData(double sourceDt)
{
    this->comptonRiskPrecomputeDt_ = sourceDt;
    this->comptonGroupCenters_ = this->opacity_->getEnergyCenters(this->energyBoundaries_);
    for(std::size_t g = 0; g < NumGroups; ++g)
    {
        this->comptonGroupWidths_[g] = this->energyBoundaries_[g + 1] - this->energyBoundaries_[g];
        if(!std::isfinite(this->comptonGroupCenters_[g]) ||
           !std::isfinite(this->comptonGroupWidths_[g]) ||
           this->comptonGroupWidths_[g] <= 0.0)
        {
            StormError eo("RadiationIMC Compton groups have invalid centers or widths");
            eo.addEntry("Group", g);
            eo.addEntry("Center", this->comptonGroupCenters_[g]);
            eo.addEntry("Width", this->comptonGroupWidths_[g]);
            throw eo;
        }
    }

    if(!this->parameters_.withCompton)
    {
        this->comptonData_.clear();
        return;
    }
    this->initializeComptonMatrixGenerator();
    const std::size_t Ncells = this->grid.GetPointNo();
    this->comptonData_.assign(Ncells, ComptonCellData{});
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT const &cell = this->cells_[i];
        ComptonCellData &data = this->comptonData_[i];
        data.volume = this->grid.GetVolume(i);
        data.temperature = cell.temperature;
        data.Um = units::arad * boost::math::pow<4>(cell.temperature);
        const auto &tracers = this->traits_.tracers(cell);
        const auto &tracerNames = this->traits_.tracerNames(cell);
        data.cv = this->eos_->dT2cv(
            this->density(i), cell.temperature, tracers, tracerNames);
        if(!std::isfinite(data.cv) || data.cv <= 0.0)
        {
            StormError eo("RadiationIMC Compton precompute requires positive heat capacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Heat capacity", data.cv);
            throw eo;
        }
        data.beta = 4.0 * units::arad *
            boost::math::pow<3>(cell.temperature) / data.cv;

        double planckIntegralTotal = 0.0;
        double const kT = units::k_boltz * cell.temperature;
        if(kT > 0.0 && std::isfinite(kT))
        {
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const lower = this->energyBoundaries_[group] / kT;
                double const upper = this->energyBoundaries_[group + 1] / kT;
                data.planckFraction[group] =
                    planck_integral::planck_integral(lower, upper);
                planckIntegralTotal += data.planckFraction[group];
            }
        }

        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(planckIntegralTotal > 0.0)
            {
                data.planckFraction[group] /= planckIntegralTotal;
            }
            else
            {
                data.planckFraction[group] = 0.0;
            }
            double absorptionOpacity = this->opacity_->CalcAbsorptionOpacity(
                cell, this->comptonGroupCenters_[group]);
            if(!std::isfinite(absorptionOpacity) || absorptionOpacity < 0.0)
            {
                StormError eo("RadiationIMC Compton precompute received invalid absorption opacity");
                eo.addEntry("Cell index", i);
                eo.addEntry("Group", group);
                eo.addEntry("Absorption opacity", absorptionOpacity);
                throw eo;
            }
            data.absorptionOpacity[group] = absorptionOpacity;
            data.planckOpacity += absorptionOpacity * data.planckFraction[group];
            data.oldRadiationEnergy[group] = std::max(
                0.0,
                this->traits_.groupEnergyPerMass(cell, group) * this->density(i));
        }
        this->planckOpacities_[i] = data.planckOpacity;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            data.baseSourceFraction[group] = data.planckOpacity > 0.0
                ? data.absorptionOpacity[group] *
                    data.planckFraction[group] / data.planckOpacity
                : 0.0;
        }
        data.planckCdf = this->buildSafeComptonCdf(data.planckFraction);
        data.baseSourceCdf =
            this->buildSafeComptonCdf(data.baseSourceFraction);
        data.groupCenters = this->comptonGroupCenters_;
        data.groupWidths = this->comptonGroupWidths_;

        ComptonOccupationMode occupationMode =
            this->parameters_.comptonUseInduced
                ? ComptonOccupationMode::RadiationField
                : ComptonOccupationMode::Zero;
        this->buildComptonMatricesForCell(
            cell, i, occupationMode, data);
        this->recomputeComptonContractions(data);
        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                gamma = 1.0 / std::sqrt(
                    1.0 - ScalarProd(cell.velocity, cell.velocity) *
                    units::inv_clight2);
            }
        }
        double const cdtEff = units::clight * sourceDt * gamma;
        double denominator = 1.0 + data.beta * cdtEff * data.Gamma;
        if((denominator <= 0.0 || data.Upsilon < 0.0) &&
           this->parameters_.comptonAllowNZeroFallback)
        {
            ComptonOccupationMode fallbackMode = ComptonOccupationMode::Zero;
            if(data.Upsilon < 0.0 &&
               this->parameters_.comptonUseInduced &&
               this->parameters_.comptonInducedMode ==
                   ComptonInducedMode::AdaptivePlanckFallback &&
               data.planckOpacity * units::clight * sourceDt >= 1.0)
            {
                fallbackMode = ComptonOccupationMode::PlanckFunction;
            }
            this->buildComptonMatricesForCell(
                cell, i, fallbackMode, data);
            data.useNZero = fallbackMode == ComptonOccupationMode::Zero;
            data.usePlanckInduced =
                fallbackMode == ComptonOccupationMode::PlanckFunction;
            this->recomputeComptonContractions(data);
            denominator = 1.0 + data.beta * cdtEff * data.Gamma;
        }
        if(!std::isfinite(denominator) || denominator <= 0.0)
        {
            StormError eo("Compton Fleck denominator is nonpositive");
            eo.addEntry("Cell index", i);
            eo.addEntry("Denominator", denominator);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            throw eo;
        }
        data.fleck = 1.0 / denominator;
        if(!std::isfinite(data.fleck) || data.fleck < 0.0 || data.fleck > 1.0)
        {
            StormError eo("Invalid Compton-modified Fleck factor");
            eo.addEntry("Cell index", i);
            eo.addEntry("Fleck", data.fleck);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            eo.addEntry("Planck opacity", data.planckOpacity);
            throw eo;
        }
        data.betaCdtF = data.beta * cdtEff * data.fleck;
        if(std::abs(data.Gamma) > 1e-200)
        {
            data.betaCdtF = (1.0 - data.fleck) / data.Gamma;
        }
        this->factorFleck_[i] = data.fleck;
        this->buildComptonEventData(data);
        this->buildComptonSources(sourceDt, data);
        this->computeComptonRiskForCell(sourceDt, data);
        data.active = true;
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::initializeComptonMatrixGenerator()
{
    if(this->comptonMatrixGen_)
    {
        return;
    }
#ifndef STORM_WITH_COMPTON
    throw StormError(
        "RadiationIMC requested Compton support, but STORM was built without CMMC");
#else
    this->comptonMatrixGen_ =
        std::make_unique<CMMCComptonBackend<NumGroups>>(
            this->comptonGroupCenters_,
            this->energyBoundaries_,
            this->parameters_.comptonMatrixSamples,
            true,
            1);
    this->comptonMatrixGen_->SetTables(this->buildComptonTemperatures());
    this->comptonGroupsInitialized_ = true;
#endif
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<double> RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::buildComptonTemperatures() const
{
    std::vector<double> temperatures;
    temperatures.reserve(109);
    temperatures.push_back(0.0001 * units::kev_kelvin);
    temperatures.push_back(0.001 * units::kev_kelvin);
    temperatures.push_back(0.005 * units::kev_kelvin);
    for(std::size_t index = 0; index < 128; ++index)
    {
        double const exponent = -2.0 + 6.0 *
            static_cast<double>(index) / 127.0;
        double const temperature =
            std::pow(10.0, exponent) * units::kev_kelvin;
        if(temperature > 1000.0 * units::kev_kelvin)
        {
            break;
        }
        temperatures.push_back(temperature);
    }
    return temperatures;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::GroupCdf
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::buildSafeComptonCdf(
    const GroupArray &weights) const
{
    GroupCdf cdf{};
    double total = 0.0;
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        total += std::max(0.0, weights[group]);
        cdf[group + 1] = total;
    }
    if(!(total > 0.0) || !std::isfinite(total))
    {
        for(std::size_t group = 0; group <= NumGroups; ++group)
        {
            cdf[group] = static_cast<double>(group) /
                static_cast<double>(NumGroups);
        }
        return cdf;
    }
    for(double &value : cdf)
    {
        value /= total;
    }
    cdf[NumGroups] = 1.0;
    return cdf;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::buildComptonMatricesForCell(
    const CellT &cell,
    std::size_t cellIndex,
    ComptonOccupationMode occupationMode,
    ComptonCellData &data)
{
    bool const usePlanckLTE =
        occupationMode == ComptonOccupationMode::PlanckFunction;
    double const lteTemperature = usePlanckLTE
        ? this->computeLteTemperature(cell, cellIndex) : cell.temperature;
    GroupArray ltePlanckFractions{};
    if(usePlanckLTE)
    {
        double const kT = units::k_boltz * lteTemperature;
        double total = 0.0;
        if(kT > 0.0)
        {
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                double const mass = planck_integral::planck_integral(
                    this->energyBoundaries_[group] / kT,
                    this->energyBoundaries_[group + 1] / kT);
                ltePlanckFractions[group] =
                    (mass > 0.0 && std::isfinite(mass)) ? mass : 0.0;
                total += ltePlanckFractions[group];
            }
        }
        if(total > 0.0)
        {
            for(double &fraction : ltePlanckFractions)
            {
                fraction /= total;
            }
        }
    }
    double const lteRadiationEnergyDensity = usePlanckLTE
        ? units::arad * boost::math::pow<4>(lteTemperature) : 0.0;
    double const pi = 3.141592653589793238462643383279502884;
    double const occupationFactor = boost::math::pow<3>(units::clight) /
        (8.0 * pi * units::planck_constant);
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const dnu = this->comptonGroupWidths_[group] /
            units::planck_constant;
        double const nu = this->comptonGroupCenters_[group] /
            units::planck_constant;
        double occupation = 0.0;
        if(occupationMode == ComptonOccupationMode::RadiationField)
        {
            occupation = occupationFactor * data.oldRadiationEnergy[group] /
                (boost::math::pow<3>(nu) * dnu);
        }
        else if(usePlanckLTE)
        {
            occupation = occupationFactor *
                ltePlanckFractions[group] * lteRadiationEnergyDensity /
                (boost::math::pow<3>(nu) * dnu);
        }
        data.occupation[group] = std::clamp(
            std::isfinite(occupation) ? occupation : 0.0, 0.0, 100.0);
    }

    double const minimumTemperature = 0.0001 * units::kev_kelvin;
    double const maximumTemperature =
        this->comptonMatrixGen_->GetMaximumTemperature() * 0.9999;
    double const temperature = std::clamp(
        lteTemperature, minimumTemperature, maximumTemperature);
    double lastGroupUpScatter = 0.0;
    double lastGroupDownScatter = 0.0;
    this->comptonMatrixGen_->GetTauMatrix(
        temperature,
        std::max(0.0, this->density(cellIndex)),
        1.0,
        1.0,
        data.tau,
        data.dtau_dUm,
        lastGroupUpScatter,
        lastGroupDownScatter);
    data.rates = data.tau;
    data.derivative = data.dtau_dUm;
    data.S = GroupMatrix{};
    data.dSdUm = GroupMatrix{};
    for(std::size_t source = 0; source < NumGroups; ++source)
    {
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            if(source + 1 == NumGroups && target + 1 == NumGroups)
            {
                data.S[source][source] +=
                    (lastGroupUpScatter - lastGroupDownScatter) *
                    (1.0 + data.occupation[source]);
                data.dSdUm[source][source] +=
                    data.dtau_dUm[source][source] *
                    (1.0 + data.occupation[source]);
                continue;
            }
            double const inFactor =
                this->comptonGroupCenters_[source] /
                this->comptonGroupCenters_[target] *
                (1.0 + data.occupation[source]);
            data.S[target][source] +=
                data.tau[target][source] * inFactor;
            data.dSdUm[target][source] +=
                data.dtau_dUm[target][source] * inFactor;
            double const outFactor = 1.0 + data.occupation[target];
            data.S[source][source] -=
                data.tau[source][target] * outFactor;
            data.dSdUm[source][source] -=
                data.dtau_dUm[source][target] * outFactor;
        }
    }
    double const derivativeScale = 1.0 /
        (4.0 * units::arad * boost::math::pow<3>(lteTemperature));
    for(std::size_t source = 0; source < NumGroups; ++source)
    {
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            data.dSdUm[source][target] *= derivativeScale;
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                    TraitsT, PositionSamplerT>::computeLteTemperature(
    const CellT &cell, std::size_t cellIndex) const
{
    double const rho = this->density(cellIndex);
    if(!(rho > 0.0) || !std::isfinite(rho))
    {
        return cell.temperature;
    }
    double radiationSpecificEnergy = 0.0;
    if constexpr(
        radiation_imc_detail::has_member_radiation_energy<CellT>::value)
    {
        radiationSpecificEnergy = std::max(0.0, cell.Erad);
    }
    double const totalSpecificEnergy =
        this->specificInternalEnergy(cellIndex) + radiationSpecificEnergy;
    if(!(totalSpecificEnergy > 0.0) || !std::isfinite(totalSpecificEnergy))
    {
        return cell.temperature;
    }

    double const maximumTemperature = this->comptonMatrixGen_
        ? this->comptonMatrixGen_->GetMaximumTemperature() * 0.9999
        : std::max(cell.temperature, 1.0);
    double const radiationTemperature = std::pow(
        std::max(radiationSpecificEnergy, 0.0) * rho / units::arad,
        0.25);
    double temperature = std::clamp(
        std::max(cell.temperature, radiationTemperature),
        1.0e-30, maximumTemperature);
    auto const &tracers = this->traits_.tracers(cell);
    auto const &tracerNames = this->traits_.tracerNames(cell);
    auto matterSpecificEnergy = [&](double candidateTemperature) -> double
    {
        using TracersType = std::decay_t<decltype(tracers)>;
        using TracerNamesType = std::decay_t<decltype(tracerNames)>;
        if constexpr(radiation_imc_detail::has_dT2e<
                         EOST, TracersType, TracerNamesType>::value)
        {
            return this->eos_->dT2e(
                rho, candidateTemperature, tracers, tracerNames);
        }
        else
        {
            throw StormError(
                "Planck-function Compton occupation requires an EOS dT2e method");
        }
    };
    for(int iteration = 0; iteration < 50; ++iteration)
    {
        double const matterEnergy = matterSpecificEnergy(temperature);
        double const radiationEnergy = units::arad *
            boost::math::pow<4>(temperature) / rho;
        double const residual =
            matterEnergy + radiationEnergy - totalSpecificEnergy;
        double const scale = std::max(
            std::abs(totalSpecificEnergy),
            std::numeric_limits<double>::min());
        if(std::abs(residual) <= 1.0e-10 * scale)
        {
            break;
        }
        double const cv = this->eos_->dT2cv(
            rho, temperature, tracers, tracerNames);
        double const derivative = cv + 4.0 * units::arad *
            boost::math::pow<3>(temperature) / rho;
        if(!(derivative > 0.0) || !std::isfinite(derivative))
        {
            break;
        }
        double const candidate = temperature - residual / derivative;
        if(!std::isfinite(candidate))
        {
            break;
        }
        temperature = std::clamp(
            candidate, 1.0e-30, maximumTemperature);
    }
    return temperature;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::recomputeComptonContractions(
    ComptonCellData &data) const
{
    data.Upsilon = 0.0;
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        data.D[group] = 0.0;
        for(std::size_t source = 0; source < NumGroups; ++source)
        {
            data.D[group] += data.dSdUm[source][group] *
                data.oldRadiationEnergy[source];
        }
        data.Upsilon += data.D[group];
        data.M[group] = data.absorptionOpacity[group] *
            data.planckFraction[group] + data.D[group];
    }
    for(std::size_t source = 0; source < NumGroups; ++source)
    {
        data.rowS[source] = 0.0;
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            data.rowS[source] += data.S[source][target];
        }
        data.Lambda[source] = data.absorptionOpacity[source] -
            data.rowS[source];
    }
    data.Gamma = data.planckOpacity + data.Upsilon;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::buildComptonSources(
    double sourceDt,
    ComptonCellData &data) const
{
    double const cdt = units::clight * sourceDt;
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const kgbg = data.absorptionOpacity[group] *
            data.planckFraction[group];
        data.Bbase[group] = data.volume * cdt * data.fleck *
            kgbg * data.Um;
        if(data.planckOpacity > 0.0)
        {
            data.Bcorr[group] = data.volume * cdt * data.planckOpacity *
                data.Um *
                ((kgbg / data.planckOpacity) *
                 (1.0 - (1.0 + data.beta * cdt * data.planckOpacity) *
                  data.fleck) -
                 data.beta * cdt * data.fleck * data.D[group]);
        }
        else
        {
            data.Bcorr[group] = 0.0;
        }
        data.Btotal[group] = data.Bbase[group] + data.Bcorr[group];
        data.Bpos[group] = std::max(0.0, data.Btotal[group]);
        data.Bres[group] = data.Btotal[group] - data.Bpos[group];
        data.residualSource[group] = data.Bcorr[group];
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::buildComptonEventData(
    ComptonCellData &data) const
{
    data.segmentKernel = GroupMatrix{};
    data.residualKernel = GroupMatrix{};
    data.Ktotal = GroupMatrix{};
    data.implicitKernel = GroupMatrix{};
    data.implicitEventRateMatrix = GroupMatrix{};
    for(std::size_t source = 0; source < NumGroups; ++source)
    {
        GroupArray weights{};
        double outRate = 0.0;
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            if(target == source)
            {
                continue;
            }
            weights[target] = std::max(
                0.0,
                data.tau[source][target] *
                (1.0 + data.occupation[target]));
            outRate += weights[target];
        }
        data.outRate[source] = outRate;
        data.comptonOutRate[source] = outRate;
        data.targetCdf[source] = this->buildSafeComptonCdf(weights);
        data.implicitEventCdf[source] = data.targetCdf[source];
        double baseOpacity = 0.0;
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            baseOpacity += data.betaCdtF * data.absorptionOpacity[source] *
                data.absorptionOpacity[target] *
                data.planckFraction[target];
        }
        data.baseEffectiveOpacity[source] = baseOpacity;

        double mu = 1.0;
        if(outRate > 0.0)
        {
            mu = 0.0;
            for(std::size_t target = 0; target < NumGroups; ++target)
            {
                if(target == source)
                {
                    continue;
                }
                mu += weights[target] / outRate *
                    this->comptonGroupCenters_[target] /
                    std::max(this->comptonGroupCenters_[source],
                             std::numeric_limits<double>::min());
            }
        }
        data.comptonMu[source] = mu;
        data.comptonMh[source] =
            1.0 + data.fleck * (mu - 1.0);
        data.meanEnergyRatio[source] = mu;
        data.modifiedFleck[source] = data.fleck;
        data.implicitEventRate[source] = outRate;
        data.implicitDiagonalCorrection[source] =
            outRate * (data.comptonMh[source] - 1.0);

        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            double const kgbg = data.absorptionOpacity[target] *
                data.planckFraction[target];
            double const kTotal = data.S[source][target] +
                data.betaCdtF * data.M[target] * data.Lambda[source];
            double const hBase = data.betaCdtF *
                data.absorptionOpacity[source] * kgbg;
            data.segmentKernel[source][target] = kTotal - hBase;
            data.Ktotal[source][target] =
                (source == target ? -data.absorptionOpacity[source] : 0.0) +
                kTotal;
            double const kImc = source == target
                ? -data.absorptionOpacity[source] +
                    (1.0 - data.fleck) *
                    data.absorptionOpacity[source] *
                    data.baseSourceFraction[source]
                : (1.0 - data.fleck) *
                    data.absorptionOpacity[source] *
                    data.baseSourceFraction[target];
            double const eventKernel = source == target
                ? -outRate
                : (outRate > 0.0
                    ? outRate * weights[target] / outRate *
                        data.comptonMh[source]
                    : 0.0);
            data.implicitKernel[source][target] = eventKernel;
            data.implicitEventRateMatrix[source][target] = eventKernel;
            data.residualKernel[source][target] =
                data.Ktotal[source][target] - kImc - eventKernel;
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::computeComptonRiskForCell(
    double fullDt,
    ComptonCellData &data) const
{
    GroupArray rhs{};
    GroupArray predicted{};
    GroupMatrix residualMatrix{};
    double totalOldExtensive = 0.0;

    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const oldExtensive =
            std::max(0.0, data.oldRadiationEnergy[group]) * data.volume;
        rhs[group] = oldExtensive + data.Bcorr[group];
        totalOldExtensive += oldExtensive;
        data.riskScore[group] = 0.0;
        data.riskTargetPackets[group] = 0;
    }
    if(!(totalOldExtensive > 0.0))
    {
        return;
    }

    for(std::size_t row = 0; row < NumGroups; ++row)
    {
        for(std::size_t column = 0; column < NumGroups; ++column)
        {
            residualMatrix[row][column] =
                (row == column ? 1.0 : 0.0) -
                fullDt * units::clight *
                    data.residualKernel[column][row];
        }
    }

    bool const solved = RadiationIMC::solveComptonGroupSystem(
        residualMatrix, rhs, predicted);
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const oldExtensive =
            std::max(0.0, data.oldRadiationEnergy[group]) * data.volume;
        double const groupFloor =
            std::max(1e-10 * totalOldExtensive, 1.0);
        double const scale = std::max(oldExtensive, groupFloor);
        double score = std::abs(data.Bcorr[group]) / scale;
        if(solved)
        {
            double const depletion = oldExtensive - predicted[group];
            if(depletion > 0.0)
            {
                score = std::max(score, depletion / scale);
            }
            if(predicted[group] < 0.0)
            {
                score = std::max(
                    score, 1.0 + std::abs(predicted[group]) / scale);
            }
        }
        else
        {
            score = std::max(score, 2.0);
        }

        if(oldExtensive <= 1e-8 * totalOldExtensive && score < 10.0)
        {
            continue;
        }
        if(score < 0.5)
        {
            continue;
        }

        data.riskScore[group] = score;
        data.riskTargetPackets[group] = score >= 10.0 ? 96
            : score >= 3.0 ? 64
            : score >= 1.0 ? 32 : 16;
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::size_t RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sampleComptonTarget(
    const ComptonCellData &data, std::size_t sourceGroup,
    MCParticle &particle)
{
    if(sourceGroup >= NumGroups || !(data.outRate[sourceGroup] > 0.0))
    {
        return sourceGroup;
    }
    return this->sampleComptonCdf(
        data.targetCdf[sourceGroup], this->randomUnitOpen(particle));
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::addComptonMaterialExchange(
    std::size_t cellIndex, double energy)
{
    if(this->parameters_.noHydroFeedback || this->parameters_.postProcess.enabled)
    {
        return;
    }
    this->tallyMaterialEnergy(cellIndex, energy, true);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateComptonParticles(double fullDt)
{
    std::vector<MCParticle> result;
    for(std::size_t cellIndex = 0; cellIndex < this->comptonData_.size(); ++cellIndex)
    {
        ComptonCellData const &data = this->comptonData_[cellIndex];
        GroupArray sourceEnergy{};
        GroupArray fractional{};
        std::array<std::size_t, NumGroups> groupCounts{};
        double totalSourceEnergy = 0.0;
        std::size_t activeGroups = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            sourceEnergy[group] = std::max(0.0, data.Bbase[group]);
            totalSourceEnergy += sourceEnergy[group];
            if(sourceEnergy[group] > 0.0)
            {
                ++activeGroups;
            }
        }
        std::size_t const packetCount = std::max(
            this->parameters_.newPhotonsPerCell, activeGroups);
        if(packetCount == 0 || !(totalSourceEnergy > 0.0))
        {
            continue;
        }

        std::size_t allocated = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(sourceEnergy[group] > 0.0)
            {
                groupCounts[group] = 1;
                ++allocated;
            }
        }
        std::size_t const remainingBudget = packetCount - allocated;
        std::size_t extraAllocated = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(!(sourceEnergy[group] > 0.0))
            {
                continue;
            }
            double const exactExtra = static_cast<double>(remainingBudget) *
                sourceEnergy[group] / totalSourceEnergy;
            std::size_t const extra =
                static_cast<std::size_t>(std::floor(exactExtra));
            groupCounts[group] += extra;
            fractional[group] = exactExtra - static_cast<double>(extra);
            extraAllocated += extra;
        }
        while(extraAllocated < remainingBudget)
        {
            std::size_t bestGroup = NumGroups;
            double bestFraction = -1.0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                if(sourceEnergy[group] > 0.0 &&
                   fractional[group] > bestFraction)
                {
                    bestGroup = group;
                    bestFraction = fractional[group];
                }
            }
            if(bestGroup == NumGroups)
            {
                break;
            }
            ++groupCounts[bestGroup];
            fractional[bestGroup] = 0.0;
            ++extraAllocated;
        }

        std::size_t riskBudget = std::max<std::size_t>(8, packetCount / 4);
        std::array<std::size_t, NumGroups> riskOrder{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            riskOrder[group] = group;
        }
        std::sort(riskOrder.begin(), riskOrder.end(),
            [&](std::size_t left, std::size_t right)
            {
                return data.riskScore[left] > data.riskScore[right];
            });
        for(std::size_t order = 0;
            order < NumGroups && riskBudget > 0; ++order)
        {
            std::size_t const group = riskOrder[order];
            std::size_t const target = data.riskTargetPackets[group];
            if(target == 0 || !(sourceEnergy[group] > 0.0) ||
               groupCounts[group] >= target)
            {
                continue;
            }
            std::size_t const add =
                std::min(target - groupCounts[group], riskBudget);
            groupCounts[group] += add;
            riskBudget -= add;
        }

        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                gamma = 1.0 / std::sqrt(
                    1.0 - ScalarProd(
                        this->cells_[cellIndex].velocity,
                        this->cells_[cellIndex].velocity) *
                    units::inv_clight2);
            }
        }

        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            std::size_t const groupPackets = groupCounts[group];
            if(groupPackets == 0 || !(sourceEnergy[group] > 0.0))
            {
                continue;
            }
            if(!this->parameters_.noHydroFeedback)
            {
                this->extensives_[cellIndex].internal_energy -=
                    sourceEnergy[group];
                radiation_imc_detail::addTotalEnergyIfPresent(
                    this->extensives_[cellIndex],
                    -sourceEnergy[group] * gamma);
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                    {
                        if(this->parameters_.withHydro &&
                           !this->parameters_.diffusionPressureGradient)
                        {
                            this->extensives_[cellIndex].momentum -=
                                sourceEnergy[group] *
                                this->cells_[cellIndex].velocity *
                                units::inv_clight2 * gamma;
                        }
                    }
                }
            }
            double const packetEnergy = sourceEnergy[group] /
                static_cast<double>(groupPackets);
            for(std::size_t packetIndex = 0;
                packetIndex < groupPackets; ++packetIndex)
            {
                MCParticle particle = this->generateSingleParticle(
                    cellIndex, this->cells_[cellIndex]);
                particle.timeLeft = fullDt * this->randomUnitOpen(particle);
                this->setPacketFromComovingState(
                    particle,
                    this->cells_[cellIndex],
                    this->frequencyForComptonGroup(group),
                    packetEnergy);
                this->setInitialWeightFromWeight(particle);
                if(particle.initialWeight > 0.0)
                {
                    result.push_back(particle);
                }
            }
        }
    }
    return result;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::size_t RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sampleComptonCdf(
    const GroupCdf &cdf, double random) const
{
    double const value = std::clamp(
        random, 0.0, std::nextafter(1.0, 0.0));
    auto iterator = std::upper_bound(cdf.begin(), cdf.end(), value);
    if(iterator == cdf.begin())
    {
        return 0;
    }
    std::size_t group = static_cast<std::size_t>(
        std::distance(cdf.begin(), iterator) - 1);
    return std::min(group, NumGroups - 1);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::frequencyForComptonGroup(
    std::size_t group) const
{
    if(group >= NumGroups)
    {
        throw StormError("RadiationIMC received an invalid Compton target group");
    }
    double frequency = this->comptonGroupCenters_[group];
    this->clampFrequencyToBounds(frequency);
    return frequency;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sumComptonGroups(
    const GroupArray &values)
{
    return RadiationIMC::compensatedSumComptonGroups(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::compensatedSumComptonGroups(
    const GroupArray &values)
{
    double sum = 0.0;
    double compensation = 0.0;
    for(double const value : values)
    {
        double const corrected = value - compensation;
        double const next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }
    return sum;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
const char *RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::comptonCorrectionFailureName(
    ComptonCorrectionFailure failure)
{
    switch(failure)
    {
        case ComptonCorrectionFailure::None:
            return "None";
        case ComptonCorrectionFailure::DirectLinearSolveFailed:
            return "DirectLinearSolveFailed";
        case ComptonCorrectionFailure::DirectNegativeMass:
            return "DirectNegativeMass";
        case ComptonCorrectionFailure::DirectMaterialCap:
            return "DirectMaterialCap";
        case ComptonCorrectionFailure::DirectProjectedResidual:
            return "DirectProjectedResidual";
        case ComptonCorrectionFailure::AdaptiveLinearSolveFailed:
            return "AdaptiveLinearSolveFailed";
        case ComptonCorrectionFailure::AdaptiveMaximumSubsteps:
            return "AdaptiveMaximumSubsteps";
        case ComptonCorrectionFailure::AdaptiveMaximumRejectedTrials:
            return "AdaptiveMaximumRejectedTrials";
        case ComptonCorrectionFailure::AdaptiveFractionBelowMinimum:
            return "AdaptiveFractionBelowMinimum";
        case ComptonCorrectionFailure::AdaptiveNoProgress:
            return "AdaptiveNoProgress";
        case ComptonCorrectionFailure::NonFiniteState:
            return "NonFiniteState";
        case ComptonCorrectionFailure::InvalidEnergyBudget:
            return "InvalidEnergyBudget";
        case ComptonCorrectionFailure::EnergyClosureFailure:
            return "EnergyClosureFailure";
    }
    return "Unknown";
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::minComptonGroup(
    const GroupArray &values)
{
    return *std::min_element(values.begin(), values.end());
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::maxAbsComptonGroup(
    const GroupArray &values)
{
    double maximum = 0.0;
    for(double const value : values)
    {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::normComptonGroups(
    const GroupArray &values)
{
    double sumSquares = 0.0;
    for(double const value : values)
    {
        sumSquares += value * value;
    }
    return std::sqrt(sumSquares);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::GroupArray
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::multiplyComptonMatrix(
    const GroupMatrix &matrix,
    const GroupArray &values)
{
    GroupArray result{};
    for(std::size_t row = 0; row < NumGroups; ++row)
    {
        for(std::size_t column = 0; column < NumGroups; ++column)
        {
            result[row] += matrix[row][column] * values[column];
        }
    }
    return result;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::relativeComptonResidual(
    const GroupMatrix &matrix,
    const GroupArray &solution,
    const GroupArray &rhs,
    double scale)
{
    GroupArray residual = RadiationIMC::multiplyComptonMatrix(matrix, solution);
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        residual[group] -= rhs[group];
    }
    return RadiationIMC::normComptonGroups(residual) /
        std::max(1.0, std::max(RadiationIMC::normComptonGroups(rhs), scale));
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::ComptonProjectionResult
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::projectNonnegativeConservative(
    const GroupArray &candidate,
    double targetTotal,
    double energyScale,
    double perGroupNegativeTolerance,
    double totalNegativeTolerance)
{
    ComptonProjectionResult result;
    result.inputTotal = RadiationIMC::compensatedSumComptonGroups(candidate);
    result.targetTotal = targetTotal;
    if(!std::isfinite(result.inputTotal) ||
       !std::isfinite(targetTotal) ||
       targetTotal < -totalNegativeTolerance)
    {
        return result;
    }
    if(targetTotal < 0.0)
    {
        result.targetTotal = 0.0;
    }

    GroupArray positive{};
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const value = candidate[group];
        if(!std::isfinite(value))
        {
            return result;
        }
        if(value < 0.0)
        {
            double const negative = -value;
            result.negativeMass += negative;
            if(negative > result.worstNegative)
            {
                result.worstNegative = negative;
                result.worstNegativeGroup = group;
            }
        }
        positive[group] = std::max(0.0, value);
    }
    double const positiveTotal =
        RadiationIMC::compensatedSumComptonGroups(positive);

    if(result.worstNegative > perGroupNegativeTolerance ||
       result.negativeMass > totalNegativeTolerance)
    {
        return result;
    }
    if(result.targetTotal > 0.0 && !(positiveTotal > 0.0))
    {
        return result;
    }

    result.usedProjection = result.negativeMass > 0.0 ||
        std::abs(result.inputTotal - result.targetTotal) >
            1e-14 * std::max(1.0, energyScale);
    if(result.targetTotal > 0.0)
    {
        // Project onto the nonnegative simplex in the Euclidean norm:
        //
        //     minimize ||x - candidate||_2
        //     subject to x_g >= 0 and sum_g x_g = targetTotal.
        //
        // The previous clip-and-rescale operation changed every positive bin
        // multiplicatively.  When many small negative bins were present, that
        // unnecessarily enlarged the correction residual and forced dozens of
        // path-dependent substeps.  The simplex projection is the smallest
        // conservative nonnegative change to the direct solution.
        GroupArray sorted = candidate;
        std::sort(sorted.begin(), sorted.end(),
            [](double left, double right) { return left > right; });
        static constexpr GroupArray inverseActiveCounts = []()
        {
            GroupArray values{};
            for(std::size_t index = 0; index < NumGroups; ++index)
            {
                values[index] =
                    1.0 / static_cast<double>(index + 1);
            }
            return values;
        }();
        double cumulative = 0.0;
        double theta = 0.0;
        std::size_t active = 0;
        for(std::size_t index = 0; index < NumGroups; ++index)
        {
            cumulative += sorted[index];
            // Keep this as a multiply by a compile-time reciprocal.  Intel LLVM
            // otherwise vectorizes the prefix scan with a speculative zero
            // divisor in a lane that is blended out later; trapping floating
            // point environments still raise SIGFPE on that discarded lane.
            double const trialTheta =
                (cumulative - result.targetTotal) *
                inverseActiveCounts[index];
            if(sorted[index] > trialTheta)
            {
                active = index + 1;
                theta = trialTheta;
            }
        }
        if(active == 0 || !std::isfinite(theta))
        {
            return result;
        }
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            result.endpoint[group] = std::max(0.0, candidate[group] - theta);
        }

        // Close the conserved radiation total at roundoff level.  Adjusting
        // the largest active component is the least fragile option because it
        // cannot turn a marginally active group negative.
        double simplexTotal =
            RadiationIMC::compensatedSumComptonGroups(result.endpoint);
        if(!std::isfinite(simplexTotal))
        {
            return result;
        }
        double const closure = result.targetTotal - simplexTotal;
        auto const largest = std::max_element(
            result.endpoint.begin(), result.endpoint.end());
        if(largest == result.endpoint.end() ||
           !std::isfinite(*largest + closure) ||
           *largest + closure < 0.0)
        {
            return result;
        }
        *largest += closure;

        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const scale = std::max(
                {1.0, std::abs(candidate[group]), std::abs(result.endpoint[group])});
            result.maximumRelativeChange = std::max(
                result.maximumRelativeChange,
                std::abs(result.endpoint[group] - candidate[group]) / scale);
        }
    }
    else
    {
        result.endpoint.fill(0.0);
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            result.maximumRelativeChange = std::max(
                result.maximumRelativeChange,
                std::abs(candidate[group]));
        }
    }

    double const projectedTotal =
        RadiationIMC::compensatedSumComptonGroups(result.endpoint);
    if(!std::isfinite(projectedTotal) ||
       std::abs(projectedTotal - result.targetTotal) >
           1e-12 * std::max(1.0, energyScale))
    {
        return result;
    }
    result.success = true;
    return result;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::solveComptonGroupSystem(
    GroupMatrix matrix,
    GroupArray rhs,
    GroupArray &solution)
{
    for(std::size_t row = 0; row < NumGroups; ++row)
    {
        for(std::size_t column = 0; column < NumGroups; ++column)
        {
            if(!std::isfinite(matrix[row][column]))
            {
                return false;
            }
        }
        if(!std::isfinite(rhs[row]))
        {
            return false;
        }
    }
    for(std::size_t column = 0; column < NumGroups; ++column)
    {
        std::size_t pivot = column;
        double pivotMagnitude = std::abs(matrix[column][column]);
        for(std::size_t row = column + 1; row < NumGroups; ++row)
        {
            double const candidate = std::abs(matrix[row][column]);
            if(candidate > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = candidate;
            }
        }
        if(!(pivotMagnitude > 1e-200) || !std::isfinite(pivotMagnitude))
        {
            return false;
        }
        if(pivot != column)
        {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(rhs[pivot], rhs[column]);
        }
        double const pivotValue = matrix[column][column];
        for(std::size_t row = column + 1; row < NumGroups; ++row)
        {
            double const factor = matrix[row][column] / pivotValue;
            if(factor == 0.0)
            {
                continue;
            }
            matrix[row][column] = 0.0;
            for(std::size_t j = column + 1; j < NumGroups; ++j)
            {
                matrix[row][j] -= factor * matrix[column][j];
                if(!std::isfinite(matrix[row][j]))
                {
                    return false;
                }
            }
            rhs[row] -= factor * rhs[column];
            if(!std::isfinite(rhs[row]))
            {
                return false;
            }
        }
    }

    solution.fill(0.0);
    for(std::size_t reverse = 0; reverse < NumGroups; ++reverse)
    {
        std::size_t const row = NumGroups - 1 - reverse;
        double value = rhs[row];
        for(std::size_t column = row + 1; column < NumGroups; ++column)
        {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
        if(!std::isfinite(solution[row]))
        {
            return false;
        }
    }
    return true;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::setPacketFromComovingState(
    MCParticle &particle,
    const CellT &cell,
    double comovingFrequency,
    double comovingWeight) const
{
    double dopplerShift = 1.0;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled &&
            this->parameters_.postProcess.useCellVelocities))
        {
            dopplerShift =
                radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
        }
    }
    if(!(dopplerShift > 0.0) || !std::isfinite(dopplerShift))
    {
        throw StormError(
            "RadiationIMC received an invalid source-packet Doppler factor");
    }
    particle.frequency = comovingFrequency / dopplerShift;
    particle.weight = comovingWeight / dopplerShift;
    this->clampFrequencyToBounds(particle.frequency);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::applyComptonScatterEvent(
    std::size_t cellIndex,
    CellT &cell,
    std::size_t sourceGroup,
    MCParticle &particle,
    const PointT &oldVelocity,
    double oldWeight)
{
    if(sourceGroup >= NumGroups)
    {
        return 0.0;
    }
    ComptonCellData &data = this->comptonData_[cellIndex];
    if(!(data.comptonOutRate[sourceGroup] > 0.0))
    {
        return 0.0;
    }
    std::size_t const targetGroup = this->sampleComptonTarget(
        data, sourceGroup, particle);
    if(targetGroup == sourceGroup || targetGroup >= NumGroups)
    {
        throw StormError("RadiationIMC Compton CDF returned an invalid target group");
    }

    if(this->parameters_.comptonAngleDependent)
    {
        std::vector<double> angleCdf;
        this->comptonMatrixGen_->GetAngleCdf(
            data.temperature, sourceGroup, targetGroup, angleCdf);
        std::size_t const angleBins =
            this->comptonMatrixGen_->GetAngleBinCount();
        if(angleBins == 0 || angleCdf.size() != angleBins + 1)
        {
            throw StormError("RadiationIMC Compton backend returned an invalid angle CDF");
        }
        double const random = this->randomUnitOpen(particle);
        std::size_t iteratorBin = 0;
        auto const iterator = std::upper_bound(
            angleCdf.begin(), angleCdf.end(), random);
        if(iterator != angleCdf.begin())
        {
            iteratorBin = static_cast<std::size_t>(
                std::distance(angleCdf.begin(), iterator) - 1);
        }
        iteratorBin = std::min(iteratorBin, angleBins - 1);
        double const lowerCdf = angleCdf[iteratorBin];
        double const upperCdf = angleCdf[iteratorBin + 1];
        double const fraction = upperCdf > lowerCdf
            ? (random - lowerCdf) / (upperCdf - lowerCdf) : 0.5;
        double const binWidth = 2.0 /
            static_cast<double>(angleBins);
        double const cosine = -1.0 +
            (static_cast<double>(iteratorBin) +
             std::clamp(fraction, 0.0, 1.0)) * binWidth;
        double const sine = std::sqrt(
            std::max(0.0, 1.0 - cosine * cosine));
        double const pi = 3.141592653589793238462643383279502884;
        double const phi = 2.0 * pi * this->randomUnitOpen(particle);
        double const speed = fastabs(oldVelocity);
        PointT const oldDirection = speed > 0.0
            ? oldVelocity / speed : PointT(0.0, 0.0, 1.0);
        PointT helper = std::abs(oldDirection.z) < 0.9
            ? PointT(0.0, 0.0, 1.0)
            : PointT(1.0, 0.0, 0.0);
        PointT perpendicular1 = helper -
            ScalarProd(helper, oldDirection) * oldDirection;
        if(fastabs(perpendicular1) > 0.0)
        {
            perpendicular1 = normalize(perpendicular1);
        }
        else
        {
            perpendicular1 = PointT(1.0, 0.0, 0.0);
        }
        PointT const perpendicular2 =
            normalize(CrossProduct(oldDirection, perpendicular1));
        PointT const newDirection = normalize(
            oldDirection * cosine +
            sine * (std::cos(phi) * perpendicular1 +
                    std::sin(phi) * perpendicular2));
        particle.velocity = newDirection * units::clight;
    }
    else
    {
        particle.velocity = this->sampleScatterVelocity(cell, particle);
    }

    // Deliberately use the source-group mean energy multiplier rather
    // than the sampled target ratio. This preserves the mean Compton
    // exchange while avoiding target-to-target packet-weight noise; the
    // deterministic residual correction carries the spectral difference.
    double const eventMultiplier = data.comptonMh[sourceGroup];
    if(!(eventMultiplier > 0.0) || !std::isfinite(eventMultiplier))
    {
        throw StormError("RadiationIMC produced an invalid averaged Compton event multiplier");
    }
    double const newWeight = oldWeight * eventMultiplier;
    particle.weight = newWeight;
    particle.frequency = this->frequencyForComptonGroup(targetGroup);
    return oldWeight - newWeight;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::ComptonCorrectionResult
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::solveComptonCorrection(
    std::size_t cellIndex,
    double fullDt,
    const ComptonCellData &data,
    const GroupArray &rawGroupEnergy,
    const GroupArray &timeAvgGroupEnergy,
    double budgetBefore,
    double materialFloor,
    double preStepRadiation) const
{
    ComptonCorrectionResult result;
    result.materialEnergyBefore =
        this->extensives_[cellIndex].internal_energy;
    result.budgetBefore = budgetBefore;

    auto fail = [&](ComptonCorrectionFailure failure)
    {
        result.failure = failure;
        result.success = false;
        return result;
    };

    double const rawTotal =
        RadiationIMC::compensatedSumComptonGroups(rawGroupEnergy);
    double const timeAverageTotal =
        RadiationIMC::compensatedSumComptonGroups(timeAvgGroupEnergy);
    if(!std::isfinite(rawTotal) ||
       !std::isfinite(timeAverageTotal) ||
       !std::isfinite(budgetBefore) ||
       !std::isfinite(materialFloor) ||
       materialFloor < 0.0)
    {
        return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
    }

    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        if(!std::isfinite(rawGroupEnergy[group]) ||
           rawGroupEnergy[group] < 0.0 ||
           !std::isfinite(timeAvgGroupEnergy[group]) ||
           timeAvgGroupEnergy[group] < 0.0 ||
           !std::isfinite(data.Bcorr[group]))
        {
            return fail(ComptonCorrectionFailure::NonFiniteState);
        }
    }
    if(rawTotal < 0.0)
    {
        return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
    }

    double const totalPostTransportRadiation = rawTotal;
    if(!std::isfinite(preStepRadiation) ||
       preStepRadiation < 0.0)
    {
        return fail(ComptonCorrectionFailure::NonFiniteState);
    }
    double const bcorrScale = preStepRadiation > 0.0
        ? std::clamp(
            totalPostTransportRadiation / preStepRadiation, 0.0, 1.0)
        : 1.0;
    double const materialCap = this->parameters_.noHydroFeedback
        ? std::numeric_limits<double>::infinity()
        : budgetBefore - materialFloor;
    if(!this->parameters_.noHydroFeedback &&
       this->extensives_[cellIndex].internal_energy < materialFloor)
    {
        return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
    }

    GroupArray oldGroupEnergy{};
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        oldGroupEnergy[group] = std::max(
            0.0, data.oldRadiationEnergy[group] * data.volume);
        if(!std::isfinite(oldGroupEnergy[group]))
        {
            return fail(ComptonCorrectionFailure::NonFiniteState);
        }
    }

    GroupArray drive{};
    GroupArray conservativeTimeAverageDrive{};
    GroupArray materialCouplingDrive{};
    GroupArray residualColumnSum{};
    GroupMatrix residualOperator{};
    GroupMatrix matrix{};
    for(std::size_t row = 0; row < NumGroups; ++row)
    {
        for(std::size_t column = 0; column < NumGroups; ++column)
        {
            double const Lrc = fullDt * units::clight *
                data.residualKernel[column][row];
            if(!std::isfinite(Lrc))
            {
                return fail(ComptonCorrectionFailure::NonFiniteState);
            }
            residualOperator[row][column] = Lrc;
            residualColumnSum[column] += Lrc;
            matrix[row][column] =
                (row == column ? 1.0 : 0.0) - Lrc;
        }
    }

    // The path-length estimator has far better group support than the
    // endpoint census, but applying it to the full residual operator also
    // changes the operator's column-sum component and therefore the net
    // radiation/material exchange.  That destroys the well-balanced state
    // used to build Bcorr and the modified Fleck factor.  Separate
    //
    //     L = C + diag(1^T L),       1^T C = 0,
    //
    // use E_avg only in C E_avg, and evaluate the net coupling with the
    // pre-step state used to construct the Compton coefficients.  Scale that
    // net source together with Bcorr so transport depletion cannot break the
    // balance between the two terms.
    for(std::size_t row = 0; row < NumGroups; ++row)
    {
        for(std::size_t column = 0; column < NumGroups; ++column)
        {
            double const Lrc = residualOperator[row][column];
            double const conservativeLrc = Lrc -
                (row == column ? residualColumnSum[column] : 0.0);
            conservativeTimeAverageDrive[row] +=
                conservativeLrc * timeAvgGroupEnergy[column];
        }
        materialCouplingDrive[row] = bcorrScale *
            (data.Bcorr[row] +
             residualColumnSum[row] * oldGroupEnergy[row]);
        drive[row] = conservativeTimeAverageDrive[row] +
            materialCouplingDrive[row];
        if(!std::isfinite(drive[row]) ||
           !std::isfinite(conservativeTimeAverageDrive[row]) ||
           !std::isfinite(materialCouplingDrive[row]) ||
           !std::isfinite(residualColumnSum[row]))
        {
            return fail(ComptonCorrectionFailure::NonFiniteState);
        }
    }
    double energyScale = std::max({
        1.0,
        std::abs(budgetBefore),
        std::abs(rawTotal),
        std::abs(timeAverageTotal),
        RadiationIMC::maxAbsComptonGroup(rawGroupEnergy),
        RadiationIMC::maxAbsComptonGroup(timeAvgGroupEnergy),
        RadiationIMC::maxAbsComptonGroup(oldGroupEnergy),
        RadiationIMC::normComptonGroups(conservativeTimeAverageDrive),
        RadiationIMC::normComptonGroups(materialCouplingDrive),
        RadiationIMC::normComptonGroups(drive),
        RadiationIMC::maxAbsComptonGroup(drive)});
    if(!std::isfinite(energyScale))
    {
        return fail(ComptonCorrectionFailure::NonFiniteState);
    }
    GroupArray directDelta{};
    bool const directOk = RadiationIMC::solveComptonGroupSystem(
        matrix, drive, directDelta);

    GroupArray directEndpoint{};
    double directTotal = 0.0;
    ComptonProjectionResult directProjection;
    bool directCandidateFinite = directOk;
    double directResidual = std::numeric_limits<double>::infinity();
    if(directOk)
    {
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            directEndpoint[group] =
                rawGroupEnergy[group] + directDelta[group];
            if(!std::isfinite(directEndpoint[group]))
            {
                directCandidateFinite = false;
            }
        }
        directResidual = RadiationIMC::relativeComptonResidual(
            matrix, directDelta, drive, energyScale);
        directTotal =
            RadiationIMC::compensatedSumComptonGroups(directEndpoint);
        if(!std::isfinite(directTotal))
        {
            directCandidateFinite = false;
        }
        if(directCandidateFinite)
        {
            energyScale = std::max(
                energyScale,
                std::max(
                    std::abs(directTotal),
                    RadiationIMC::maxAbsComptonGroup(directEndpoint)));
        }
    }

    // These are deliberately loose engineering tolerances for a noisy,
    // low-packet spectrum.  They are not machine-roundoff epsilons.  The
    // aggregate negative-mass limit must allow several individually small
    // negative bins to be removed in one conservative projection.  The
    // cycle-309 failure had 0.661% aggregate negative mass while every group
    // was below 0.01% of the cell energy, so the former 0.2% aggregate limit
    // forced many path-dependent projected substeps.  Permit up to a 1%
    // one-shot projection and require the projected equations to remain
    // accurate to the same 1% engineering scale.
    constexpr double perGroupNegativeToleranceFraction = 1e-3;
    constexpr double totalNegativeToleranceFraction = 1e-2;
    constexpr double capToleranceFraction = 1e-6;
    constexpr double relativeResidualTolerance = 1e-2;
    constexpr double grossResidualTolerance = 1e-2;
    double const perGroupNegativeTolerance =
        perGroupNegativeToleranceFraction * energyScale;
    double const totalNegativeTolerance =
        totalNegativeToleranceFraction * energyScale;
    double const capTolerance = capToleranceFraction * energyScale;
    bool const directCapWithinTolerance =
        directCandidateFinite &&
        (!std::isfinite(materialCap) ||
         directTotal <= materialCap + capTolerance);
    bool const directCapRepair =
        directCandidateFinite &&
        std::isfinite(materialCap) &&
        directTotal > materialCap &&
        directTotal <= materialCap + capTolerance;
    double directTargetTotal = directTotal;
    if(directCapRepair)
    {
        directTargetTotal = materialCap;
    }
    if(directCandidateFinite)
    {
        directProjection = RadiationIMC::projectNonnegativeConservative(
            directEndpoint,
            directTargetTotal,
            energyScale,
            perGroupNegativeTolerance,
            totalNegativeTolerance);
        directProjection.usedCapRepair = directCapRepair;
    }
    bool const directClampAcceptable = directCandidateFinite &&
        directProjection.worstNegative <= perGroupNegativeTolerance &&
        directProjection.negativeMass <= totalNegativeTolerance;

    if(directCandidateFinite && directProjection.success)
    {
        GroupArray projectedDelta{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            projectedDelta[group] =
                directProjection.endpoint[group] - rawGroupEnergy[group];
        }
        double const projectedResidual =
            RadiationIMC::relativeComptonResidual(
                matrix, projectedDelta, drive, energyScale);
        bool const residualAcceptable =
            directResidual <= grossResidualTolerance &&
            projectedResidual <= relativeResidualTolerance;
        bool const endpointAcceptable =
            directClampAcceptable &&
            directCapWithinTolerance &&
            residualAcceptable;
        if(endpointAcceptable)
        {
            result.endpoint = directProjection.endpoint;
            result.delta = projectedDelta;
            result.radiationTotal =
                RadiationIMC::compensatedSumComptonGroups(result.endpoint);
            result.materialEnergyBefore =
                this->extensives_[cellIndex].internal_energy;
            result.materialEnergyAfter = this->parameters_.noHydroFeedback
                ? result.materialEnergyBefore
                : budgetBefore - result.radiationTotal;
            result.budgetBefore = budgetBefore;
            result.usedProjection = directProjection.usedProjection;
            result.usedCapRepair = directProjection.usedCapRepair;
            result.energyClosureResidual =
                this->parameters_.noHydroFeedback
                ? 0.0
                : budgetBefore - (result.materialEnergyAfter +
                    result.radiationTotal);
            double const closureTolerance =
                128.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(budgetBefore));
            if(!this->parameters_.noHydroFeedback &&
               std::isfinite(result.energyClosureResidual) &&
               std::abs(result.energyClosureResidual) <= closureTolerance)
            {
                result.materialEnergyAfter +=
                    result.energyClosureResidual;
                result.energyClosureResidual =
                    budgetBefore - (result.materialEnergyAfter +
                        result.radiationTotal);
            }
            if(!std::isfinite(result.materialEnergyAfter) ||
               result.materialEnergyAfter < materialFloor - capTolerance)
            {
                result.failure =
                    ComptonCorrectionFailure::InvalidEnergyBudget;
            }
            else if(!std::isfinite(result.energyClosureResidual) ||
                    std::abs(result.energyClosureResidual) >
                        closureTolerance)
            {
                result.failure =
                    ComptonCorrectionFailure::EnergyClosureFailure;
            }
            else
            {
                result.success = true;
                result.failure = ComptonCorrectionFailure::None;
                return result;
            }
        }
    }

    GroupArray currentEndpoint = rawGroupEnergy;
    GroupArray currentDelta{};
    double tau = 0.0;
    double fraction = 1.0;
    std::size_t acceptedSubsteps = 0;
    std::size_t rejectedTrials = 0;
    double minimumAcceptedFraction =
        std::numeric_limits<double>::infinity();
    double maximumAcceptedFraction = 0.0;
    bool usedProjection = false;
    bool usedCapRepair = false;
    ComptonCorrectionFailure lastFailure;
    if(!directOk)
    {
        lastFailure = ComptonCorrectionFailure::DirectLinearSolveFailed;
    }
    else if(!directCandidateFinite)
    {
        lastFailure = ComptonCorrectionFailure::NonFiniteState;
    }
    else if(!directCapWithinTolerance)
    {
        lastFailure = ComptonCorrectionFailure::DirectMaterialCap;
    }
    else if(!directProjection.success || !directClampAcceptable)
    {
        lastFailure = ComptonCorrectionFailure::DirectNegativeMass;
    }
    else
    {
        lastFailure = ComptonCorrectionFailure::DirectProjectedResidual;
    }

    constexpr std::size_t maxAcceptedSubsteps = 32;
    // Rejected trial solves are line-search work, not physical substeps.  Keep
    // the requested 32 accepted-substep limit, but do not abort merely because
    // two trial reductions were needed per accepted step.
    constexpr std::size_t maxRejectedTrials = 128;
    constexpr double minimumFraction = 1e-12;
    constexpr double completionTolerance = 1e-12;

    while(tau < 1.0 - completionTolerance &&
          acceptedSubsteps < maxAcceptedSubsteps &&
          rejectedTrials < maxRejectedTrials)
    {
        fraction = std::min(fraction, 1.0 - tau);
        if(fraction < minimumFraction)
        {
            lastFailure = ComptonCorrectionFailure::AdaptiveFractionBelowMinimum;
            break;
        }

        GroupMatrix fractionalMatrix{};
        GroupArray fractionalRhs{};
        for(std::size_t row = 0; row < NumGroups; ++row)
        {
            fractionalRhs[row] =
                currentDelta[row] + fraction * drive[row];
            for(std::size_t column = 0; column < NumGroups; ++column)
            {
                double const Lrc = fullDt * units::clight *
                    data.residualKernel[column][row];
                fractionalMatrix[row][column] =
                    (row == column ? 1.0 : 0.0) - fraction * Lrc;
            }
        }

        GroupArray trialDelta{};
        bool const trialSolveOk = RadiationIMC::solveComptonGroupSystem(
            fractionalMatrix,
            fractionalRhs,
            trialDelta);
        if(!trialSolveOk)
        {
            ++rejectedTrials;
            lastFailure =
                ComptonCorrectionFailure::AdaptiveLinearSolveFailed;
            fraction *= 0.5;
            continue;
        }

        GroupArray trialEndpoint{};
        bool finiteTrial = true;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            trialEndpoint[group] =
                rawGroupEnergy[group] + trialDelta[group];
            if(!std::isfinite(trialEndpoint[group]))
            {
                finiteTrial = false;
            }
        }
        if(!finiteTrial)
        {
            ++rejectedTrials;
            lastFailure = ComptonCorrectionFailure::NonFiniteState;
            fraction *= 0.5;
            continue;
        }

        double const trialTotal =
            RadiationIMC::compensatedSumComptonGroups(trialEndpoint);
        double const capOvershoot = std::isfinite(materialCap)
            ? trialTotal - materialCap : 0.0;
        bool const capTooLarge = std::isfinite(materialCap) &&
            capOvershoot > capTolerance;
        bool const targetIsNegative =
            trialTotal < -totalNegativeTolerance;
        double trialTargetTotal = trialTotal;
        bool const trialCapRepair = std::isfinite(materialCap) &&
            capOvershoot > 0.0 && !capTooLarge;
        if(trialCapRepair)
        {
            trialTargetTotal = materialCap;
        }

        ComptonProjectionResult trialProjection =
            RadiationIMC::projectNonnegativeConservative(
                trialEndpoint,
                trialTargetTotal,
                energyScale,
                perGroupNegativeTolerance,
                totalNegativeTolerance);
        trialProjection.usedCapRepair = trialCapRepair;
        GroupArray projectedDelta{};
        if(trialProjection.success)
        {
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                projectedDelta[group] =
                    trialProjection.endpoint[group] -
                    rawGroupEnergy[group];
            }
        }
        double fractionalResidual = std::numeric_limits<double>::infinity();
        if(trialProjection.success)
        {
            fractionalResidual =
                RadiationIMC::relativeComptonResidual(
                    fractionalMatrix,
                    projectedDelta,
                    fractionalRhs,
                    energyScale);
        }
        bool const trialAcceptable =
            !capTooLarge &&
            !targetIsNegative &&
            trialProjection.success &&
            fractionalResidual <= relativeResidualTolerance &&
            trialProjection.worstNegative <=
                perGroupNegativeTolerance &&
            trialProjection.negativeMass <= totalNegativeTolerance;
        if(trialAcceptable)
        {
            currentEndpoint = trialProjection.endpoint;
            currentDelta = projectedDelta;
            tau += fraction;
            ++acceptedSubsteps;
            minimumAcceptedFraction = std::min(
                minimumAcceptedFraction, fraction);
            maximumAcceptedFraction = std::max(
                maximumAcceptedFraction, fraction);
            usedProjection = usedProjection ||
                trialProjection.usedProjection;
            usedCapRepair = usedCapRepair ||
                trialProjection.usedCapRepair;
            // Grow from the last successful fraction, but also request at
            // least the average fraction needed to finish in the remaining
            // accepted-step budget.  Retrying the complete remaining interval
            // after every success caused repeated reject/shrink cycles and
            // exhausted the rejected-trial guard without improving tau.
            double const remaining = std::max(0.0, 1.0 - tau);
            std::size_t const remainingSlots =
                maxAcceptedSubsteps - acceptedSubsteps;
            if(remainingSlots > 0)
            {
                double const requiredAverage = remaining /
                    static_cast<double>(remainingSlots);
                fraction = std::min(
                    remaining,
                    std::max(1.5 * fraction, requiredAverage));
            }
            else
            {
                fraction = remaining;
            }
            continue;
        }

        ++rejectedTrials;
        lastFailure = capTooLarge
            ? ComptonCorrectionFailure::DirectMaterialCap
            : ComptonCorrectionFailure::AdaptiveNoProgress;
        double theta = 1.0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(trialEndpoint[group] < 0.0)
            {
                double const denominator =
                    currentEndpoint[group] - trialEndpoint[group];
                theta = denominator > 0.0
                    ? std::min(theta,
                        currentEndpoint[group] / denominator)
                    : 0.0;
            }
        }
        if(std::isfinite(materialCap) && trialTotal > materialCap)
        {
            double const currentTotal =
                RadiationIMC::compensatedSumComptonGroups(currentEndpoint);
            double const totalChange = trialTotal - currentTotal;
            if(totalChange > 0.0)
            {
                theta = std::min(theta,
                    (materialCap - currentTotal) / totalChange);
            }
            else
            {
                theta = 0.0;
            }
        }
        double const newFraction = std::clamp(
            0.8 * theta * fraction,
            0.1 * fraction,
            0.7 * fraction);
        if(newFraction < minimumFraction ||
           newFraction >= 0.99 * fraction)
        {
            lastFailure = ComptonCorrectionFailure::AdaptiveNoProgress;
            break;
        }
        fraction = newFraction;
    }

    if(tau < 1.0 - completionTolerance)
    {
        if(acceptedSubsteps >= maxAcceptedSubsteps)
        {
            lastFailure =
                ComptonCorrectionFailure::AdaptiveMaximumSubsteps;
        }
        else if(rejectedTrials >= maxRejectedTrials)
        {
            lastFailure =
                ComptonCorrectionFailure::AdaptiveMaximumRejectedTrials;
        }
        else if(fraction < minimumFraction)
        {
            lastFailure =
                ComptonCorrectionFailure::AdaptiveFractionBelowMinimum;
        }
        return fail(lastFailure);
    }

    result.endpoint = currentEndpoint;
    result.delta = currentDelta;
    result.radiationTotal =
        RadiationIMC::compensatedSumComptonGroups(result.endpoint);
    result.materialEnergyBefore =
        this->extensives_[cellIndex].internal_energy;
    result.materialEnergyAfter = this->parameters_.noHydroFeedback
        ? result.materialEnergyBefore
        : budgetBefore - result.radiationTotal;
    result.budgetBefore = budgetBefore;
    result.usedProjection = usedProjection;
    result.usedCapRepair = usedCapRepair;
    result.energyClosureResidual =
        this->parameters_.noHydroFeedback
        ? 0.0
        : budgetBefore - (result.materialEnergyAfter +
            result.radiationTotal);
    double const closureTolerance =
        128.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(budgetBefore));
    if(!this->parameters_.noHydroFeedback &&
       std::isfinite(result.energyClosureResidual) &&
       std::abs(result.energyClosureResidual) <= closureTolerance)
    {
        result.materialEnergyAfter +=
            result.energyClosureResidual;
        result.energyClosureResidual =
            budgetBefore - (result.materialEnergyAfter +
                result.radiationTotal);
    }
    if(!std::isfinite(result.materialEnergyAfter) ||
       result.materialEnergyAfter < materialFloor - capTolerance)
    {
        return fail(ComptonCorrectionFailure::InvalidEnergyBudget);
    }
    if(!std::isfinite(result.energyClosureResidual) ||
       std::abs(result.energyClosureResidual) > closureTolerance)
    {
        return fail(ComptonCorrectionFailure::EnergyClosureFailure);
    }
    result.success = true;
    result.failure = ComptonCorrectionFailure::None;
    return result;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::applyComptonEndOfStepCorrection(
    double fullDt)
{
    if(!this->parameters_.withCompton)
    {
        return;
    }
    if constexpr(!radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
    {
        return;
    }
    else
    {
        const std::size_t Ncells = this->grid.GetPointNo();
        std::vector<ComptonCorrectionResult> results(Ncells);
        for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
        {
            ComptonCellData const &data = this->comptonData_[cellIndex];
            std::size_t cellID =
                radiation_imc_detail::cellID(this->cells_[cellIndex]);
            if(cellID == std::numeric_limits<std::size_t>::max())
            {
                cellID = cellIndex;
            }

            GroupArray rawGroupEnergy{};
            GroupArray timeAvgGroupEnergy{};
            double totalPreStepRadiation = 0.0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                rawGroupEnergy[group] =
                    this->extensives_[cellIndex].Eg[group];
                if(cellIndex < this->Eg_time_avg_.size())
                {
                    timeAvgGroupEnergy[group] = std::max(
                        0.0,
                        this->Eg_time_avg_[cellIndex][group] *
                            data.volume);
                }
                totalPreStepRadiation += data.oldRadiationEnergy[group];
            }
            double const rawTotal =
                RadiationIMC::compensatedSumComptonGroups(rawGroupEnergy);
            double const preStepExtensive =
                totalPreStepRadiation * data.volume;
            double const budgetBefore = this->parameters_.noHydroFeedback
                ? rawTotal
                : this->extensives_[cellIndex].internal_energy + rawTotal;
            ComptonCorrectionResult result =
                this->solveComptonCorrection(
                    cellIndex,
                    fullDt,
                    data,
                    rawGroupEnergy,
                    timeAvgGroupEnergy,
                    budgetBefore,
                    0.0,
                    preStepExtensive);

            if(!result.success)
            {
                StormError eo(
                    "Compton correction failed to integrate the complete timestep; "
                    "no partial correction was committed");
                eo.addEntry("Cell index", cellIndex);
                eo.addEntry("Cell ID", cellID);
                eo.addEntry("Full dt", fullDt);
                eo.addEntry(
                    "Failure",
                    RadiationIMC::comptonCorrectionFailureName(
                        result.failure));
                throw eo;
            }
            results[cellIndex] = result;
        }

        for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
        {
            ComptonCorrectionResult const &result = results[cellIndex];
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                this->extensives_[cellIndex].Eg[group] =
                    result.endpoint[group];
            }
            radiation_imc_detail::setRadiationEnergyIfPresent(
                this->extensives_[cellIndex], result.radiationTotal);
            if(!this->parameters_.noHydroFeedback)
            {
                this->extensives_[cellIndex].internal_energy =
                    result.materialEnergyAfter;
                radiation_imc_detail::addTotalEnergyIfPresent(
                    this->extensives_[cellIndex],
                    result.materialEnergyAfter -
                        result.materialEnergyBefore);
            }
        }
    } // else (has_member_group_energy_mutable)
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::reconcileComptonParticles(
    std::vector<MCParticle> &particles)
{
    if(!this->parameters_.withCompton)
    {
        return;
    }
    if constexpr(!radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
    {
        return;
    }
    else
    {
    const std::size_t Ncells = this->grid.GetPointNo();
    std::vector<GroupArray> raw(Ncells, GroupArray{});
    for(const MCParticle &particle : particles)
    {
        if(particle.cellIndex >= Ncells)
        {
            continue;
        }
        if(particle.weight < 0.0)
        {
            throw StormError(
                "Negative particle weight in positive-only Compton reconciliation");
        }
        double frequency = particle.frequency;
        this->clampFrequencyToBounds(frequency);
        std::size_t const group = this->opacity_->findGroup(
            frequency, this->energyBoundaries_);
        if(group < NumGroups)
        {
            raw[particle.cellIndex][group] += particle.weight;
        }
    }

    std::vector<GroupArray> scale(Ncells, GroupArray{});
    for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
    {
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const target = std::max(
                0.0, this->extensives_[cellIndex].Eg[group]);
            scale[cellIndex][group] =
                raw[cellIndex][group] > 0.0 && target < raw[cellIndex][group]
                ? target / raw[cellIndex][group] : 1.0;
        }
    }

    auto iterator = particles.begin();
    while(iterator != particles.end())
    {
        MCParticle &particle = *iterator;
        if(particle.cellIndex < Ncells)
        {
            double frequency = particle.frequency;
            this->clampFrequencyToBounds(frequency);
            std::size_t const group = this->opacity_->findGroup(
                frequency, this->energyBoundaries_);
            if(group < NumGroups)
            {
                particle.weight *= scale[particle.cellIndex][group];
                this->setInitialWeightFromWeight(particle);
            }
        }
        if(!(particle.weight > 0.0))
        {
            iterator = particles.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }

    for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
    {
        GroupArray deficits{};
        GroupArray fractional{};
        std::array<std::size_t, NumGroups> groupCounts{};
        double totalDeficit = 0.0;
        std::size_t activeDeficitGroups = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const target = std::max(
                0.0, this->extensives_[cellIndex].Eg[group]);
            double const represented = raw[cellIndex][group] > target
                ? target : raw[cellIndex][group];
            double const deficit = target - represented;
            if(deficit <= 1e-12 * std::max(1.0, target))
            {
                continue;
            }
            deficits[group] = deficit;
            totalDeficit += deficit;
            ++activeDeficitGroups;
        }
        if(!(totalDeficit > 0.0))
        {
            continue;
        }

        std::size_t const packetBudget = std::max(
            this->parameters_.newPhotonsPerCell,
            activeDeficitGroups);
        std::size_t allocated = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(deficits[group] > 0.0)
            {
                groupCounts[group] = 1;
                ++allocated;
            }
        }
        std::size_t const remainingBudget = packetBudget - allocated;
        std::size_t extraAllocated = 0;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            if(!(deficits[group] > 0.0))
            {
                continue;
            }
            double const exactExtra = static_cast<double>(remainingBudget) *
                deficits[group] / totalDeficit;
            std::size_t const extra =
                static_cast<std::size_t>(std::floor(exactExtra));
            groupCounts[group] += extra;
            fractional[group] = exactExtra - static_cast<double>(extra);
            extraAllocated += extra;
        }
        while(extraAllocated < remainingBudget)
        {
            std::size_t bestGroup = NumGroups;
            double bestFraction = -1.0;
            for(std::size_t group = 0; group < NumGroups; ++group)
            {
                if(deficits[group] > 0.0 &&
                   fractional[group] > bestFraction)
                {
                    bestGroup = group;
                    bestFraction = fractional[group];
                }
            }
            if(bestGroup == NumGroups)
            {
                break;
            }
            ++groupCounts[bestGroup];
            fractional[bestGroup] = 0.0;
            ++extraAllocated;
        }

        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            std::size_t const packetCount = groupCounts[group];
            if(packetCount == 0)
            {
                continue;
            }
            double const packetWeight = deficits[group] /
                static_cast<double>(packetCount);
            for(std::size_t packetIndex = 0;
                packetIndex < packetCount; ++packetIndex)
            {
                MCParticle particle = this->generateSingleParticle(
                    cellIndex, this->cells_[cellIndex]);
                this->setPacketFromComovingState(
                    particle,
                    this->cells_[cellIndex],
                    this->frequencyForComptonGroup(group),
                    packetWeight);
                this->setInitialWeightFromWeight(particle);
                particles.push_back(particle);
            }
        }
    }

    std::vector<GroupArray> representedAfter(Ncells, GroupArray{});
    for(const MCParticle &particle : particles)
    {
        if(particle.cellIndex >= Ncells)
        {
            continue;
        }
        double frequency = particle.frequency;
        this->clampFrequencyToBounds(frequency);
        std::size_t const group = this->opacity_->findGroup(
            frequency, this->energyBoundaries_);
        if(group < NumGroups)
        {
            representedAfter[particle.cellIndex][group] += particle.weight;
        }
    }
    constexpr double reconciliationTolerance = 1e-8;
    for(std::size_t cellIndex = 0; cellIndex < Ncells; ++cellIndex)
    {
        double maxRelativeError = 0.0;
        double targetTotal = 0.0;
        double actualTotal = 0.0;
        std::size_t worstGroup = NumGroups;
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const target = std::max(
                0.0, this->extensives_[cellIndex].Eg[group]);
            double const actual = representedAfter[cellIndex][group];
            targetTotal += target;
            actualTotal += actual;
            double const scale = std::max(
                {1.0, std::abs(target), std::abs(actual)});
            double const relativeError =
                std::abs(actual - target) / scale;
            if(relativeError > maxRelativeError)
            {
                maxRelativeError = relativeError;
                worstGroup = group;
            }
        }
        double const totalScale = std::max(
            {1.0, std::abs(targetTotal), std::abs(actualTotal)});
        double const totalRelativeError =
            std::abs(actualTotal - targetTotal) / totalScale;
        maxRelativeError = std::max(maxRelativeError, totalRelativeError);
        if(maxRelativeError > reconciliationTolerance)
        {
            StormError eo(
                "Compton particle reconciliation did not reproduce the "
                "accepted deterministic endpoint");
            eo.addEntry("Cell index", cellIndex);
            eo.addEntry("Cell ID",
                        radiation_imc_detail::cellID(this->cells_[cellIndex]));
            eo.addEntry("Worst group", worstGroup);
            eo.addEntry("Maximum relative error", maxRelativeError);
            eo.addEntry("Target total", targetTotal);
            eo.addEntry("Actual total", actualTotal);
            eo.addEntry("Tolerance", reconciliationTolerance);
            throw eo;
        }
    }
    } // else (has_member_group_energy_mutable)
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::recordObserverCrossing(
    const MCParticle &particle, const PointT &crossingPoint)
{
    if(!this->observer_)
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
    if(this->polarizationEnabled())
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
    this->observer_->recordCrossing(record);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::onBoundaryResult(
    const MCParticle &particle, ParticleStatus status, bool escaped)
{
    if(escaped && status == ParticleStatus::REMOVE && this->observer_)
    {
        this->observer_->addBoxEscapeEnergy(particle.weight);
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::preStep(double fullDt)
{
    if(!std::isfinite(fullDt) || fullDt <= 0.0)
    {
        StormError eo("RadiationIMC::preStep requires a finite, positive timestep");
        eo.addEntry("Full dt", fullDt);
        throw eo;
    }
    const std::size_t Ncells = this->grid.GetPointNo();
    this->resetTransportTallies(Ncells);
    double const fleckDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.sourceDt : fullDt;
    double const sourceDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.sourceDt : fullDt;
    bool const reuseComptonPrecompute =
        !this->parameters_.postProcess.enabled &&
        this->parameters_.withCompton &&
        this->parameters_.withMultigroupOpacity &&
        this->comptonDataReusableInPreStep_ &&
        this->comptonData_.size() == Ncells &&
        this->factorFleck_.size() == Ncells &&
        this->planckOpacities_.size() == Ncells &&
        this->comptonRiskPrecomputeDt_ == sourceDt;
    if(!reuseComptonPrecompute)
    {
        this->planckOpacities_.assign(Ncells, 0.0);
        this->factorFleck_.assign(Ncells, 1.0);
    }
    this->scatteringOpacities_.assign(Ncells, 0.0);
    this->Erad_time_avg_.assign(Ncells, 0.0);
    if((this->parameters_.withEgTimeAvg ||
        this->parameters_.withCompton) &&
       this->parameters_.withMultigroupOpacity)
    {
        GroupArray zeros{};
        this->Eg_time_avg_.assign(Ncells, zeros);
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT &cell = this->cells_[i];

        double const volume = this->grid.GetVolume(i);
        if(!std::isfinite(volume) || volume <= 0.0)
        {
            StormError eo("RadiationIMC::preStep requires finite, positive cell volumes");
            eo.addEntry("Cell index", i);
            eo.addEntry("Volume", volume);
            throw eo;
        }

        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((this->parameters_.withHydro && !this->parameters_.MMC) ||
               (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities))
            {
                gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
            }
        }

        if(this->parameters_.withCompton)
        {
            if(!reuseComptonPrecompute)
            {
                this->planckOpacities_[i] = 0.0;
                this->factorFleck_[i] = 1.0;
            }
            continue;
        }

        this->planckOpacities_[i] = this->opacity_->CalcPlanckOpacity(cell);
        if(!std::isfinite(this->planckOpacities_[i]) ||
           this->planckOpacities_[i] < 0.0)
        {
            StormError eo("RadiationIMC::preStep received an invalid Planck opacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Planck opacity", this->planckOpacities_[i]);
            throw eo;
        }
        this->scatteringOpacities_[i] =
            this->opacity_->CalcScatteringOpacity(cell);
        if(!std::isfinite(this->scatteringOpacities_[i]) ||
           this->scatteringOpacities_[i] < 0.0)
        {
            StormError eo("RadiationIMC::preStep received an invalid scattering opacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Scattering opacity", this->scatteringOpacities_[i]);
            throw eo;
        }
        const auto &tracers = this->traits_.tracers(cell);
        const auto &tracerNames = this->traits_.tracerNames(cell);
        double cv = this->eos_->dT2cv(this->density(i), cell.temperature, tracers, tracerNames);
        if(!std::isfinite(cv) || cv <= 0.0)
        {
            StormError eo("RadiationIMC::preStep requires a finite, positive heat capacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("cv", cv);
            throw eo;
        }
        this->factorFleck_[i] = 1.0 / (1.0 + (4.0 * units::arad * boost::math::pow<3>(cell.temperature) * this->planckOpacities_[i] * units::clight * fleckDt * gamma) / cv);
        if(!std::isfinite(this->factorFleck_[i]) ||
           this->factorFleck_[i] < 0.0 || this->factorFleck_[i] > 1.0)
        {
            StormError eo("Invalid factor fleck in RadiationIMC::preStep");
            eo.addEntry("Factor fleck", this->factorFleck_[i]);
            eo.addEntry("Planck opacity", this->planckOpacities_[i]);
            eo.addEntry("Temperature", cell.temperature);
            eo.addEntry("Density", this->density(i));
            eo.addEntry("Gamma", gamma);
            eo.addEntry("cv", cv);
            eo.addEntry("Full dt", fleckDt);
            throw eo;
        }
    }

    if(this->parameters_.withRandomWalk || this->parameters_.withDDMC)
    {
        this->updateGridData();
    }
    if(this->parameters_.withRandomWalk)
    {
        this->precomputeRandomWalkData();
        this->rwStepCount_ = 0;
    }
    if(this->parameters_.withDDMC)
    {
        this->ddmcFluxRhsIntegrated_.assign(Ncells, PointT{});
        this->ddmcMomentumFeedbackCount_ = 0;
        this->ddmcMomentumMatrixFallbackCount_ = 0;
        this->precomputeDDMCData();
        this->ddmcStepCount_ = 0;
        this->ddmcLeakCount_ = 0;
        this->ddmcCensusCount_ = 0;
        this->ddmcUpscatterCount_ = 0;
        this->ddmcFallbackCount_ = 0;
        this->ddmcMovingInterfaceBypassCount_ = 0;
        this->ddmcMovingInterfaceMaxFactor_ = 0.0;
    }
    else
    {
        this->ddmcFluxRhsIntegrated_.clear();
    }

    double const transportDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.transportTime : fullDt;
    if(this->parameters_.withCompton && !reuseComptonPrecompute)
    {
        this->precomputeComptonData(sourceDt);
    }
    this->comptonDataReusableInPreStep_ = false;
    std::vector<MCParticle> newParticles = this->generateParticles(sourceDt);
    if(this->parameters_.postProcess.enabled)
    {
        for(MCParticle &particle : newParticles)
        {
            particle.timeLeft = transportDt * this->randomUnitOpen(particle);
        }
    }
    if(this->boundary)
    {
        std::vector<MCParticle> boundaryParticles = this->boundary->generateNewBoundaryParticles(fullDt);
        for(MCParticle &particle : boundaryParticles)
        {
            this->setInitialWeightFromWeight(particle);
            if(this->parameters_.postProcess.enabled)
            {
                particle.timeLeft = transportDt * this->randomUnitOpen(particle);
            }
        }
        newParticles.insert(newParticles.end(), boundaryParticles.begin(), boundaryParticles.end());
    }
    if(this->observer_)
    {
        double emittedEnergy = 0.0;
        double emittedPositiveEnergy = 0.0;
        double emittedNegativeEnergy = 0.0;
        for(const MCParticle &particle : newParticles)
        {
            emittedEnergy += particle.weight;
            if(particle.weight >= 0.0)
            {
                emittedPositiveEnergy += particle.weight;
            }
            else
            {
                emittedNegativeEnergy -= particle.weight;
            }
        }
        this->observer_->addEmittedEnergy(emittedEnergy);
        this->observer_->addEmittedEnergyComponents(
            emittedPositiveEnergy, emittedNegativeEnergy);
    }
    this->preStepInitialized_ = true;
#ifdef STORM_WITH_GPU
    this->gpuTransportEnabled_ = this->GreyKernelEligible();
    if(this->gpuTransportEnabled_)
    {
        if(!this->gpuRuntime_)
        {
            this->gpuRuntime_ = std::make_unique<gpu::KokkosRuntime>();
        }
        if(!this->gpuData_)
        {
            this->gpuData_ = std::make_unique<gpu::GreyIMCData>();
        }
        const std::size_t buildGeneration = this->grid.GetBuildGeneration();
        if(this->gpuGridBuildGeneration_ != buildGeneration)
        {
            this->gpuData_->UploadGrid(
                this->gridData.cellFaceOffsets,
                this->gridData.cellCenters,
                this->gridData.normals,
                this->gridData.pointsOnFaces,
                this->gridData.nextCellIndices,
                this->gridData.boundaryCrossings,
                this->gridData.deviceBoundaryBehaviors);
            this->gpuGridBuildGeneration_ = buildGeneration;
        }
        this->gpuData_->UploadTables(
            this->planckOpacities_,
            this->scatteringOpacities_,
            this->factorFleck_);
    }
#endif
    return newParticles;
}

// ============================================================
// step
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::Functionality
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::step(
    MCParticle &particle,
    std::vector<MCParticle> &particlesToAdd)
{
    Functionality functionality;

    std::size_t cellIndex = particle.cellIndex;
    CellT &cell = this->cells_[cellIndex];

    if(particle.radiationState.hasPendingFlux() &&
       cellIndex < this->grid.GetPointNo())
    {
        this->addDDMCFluxContribution(
            cellIndex, particle.radiationState.pendingFlux);
        particle.radiationState.clearPendingFlux();
    }

    double dopplerShift = 1.0;
    bool useComovingTransportFrame = false;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities))
        {
            useComovingTransportFrame = true;
            dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
        }
    }

    if(!particle.radiationState.isDDMC() &&
       this->parameters_.withRandomWalk && !this->parameters_.withCompton &&
       this->randomWalk_ && this->rwCellEligible_[cellIndex])
    {
        if(this->tryRandomWalkStep(particle, functionality))
        {
            ++this->rwStepCount_;
            return functionality;
        }
    }

    if(particle.radiationState.isDDMC() ||
       (this->parameters_.withDDMC &&
        cellIndex < this->ddmcCellData_.size() &&
        this->ddmcCellData_[cellIndex].eligible))
    {
        if(this->tryDDMCStep(particle, functionality))
        {
            return functionality;
        }
    }

#ifdef STORM_WITH_GPU
    if(this->GreyKernelEligible())
    {
        gpu::TransportResult result = gpu::AdvanceOne(particle, this->GetHostTransportViews());
        if(result.error != gpu::TransportError::None)
        {
            StormError eo("RadiationIMC GPU-compatible grey transport failed");
            eo.addEntry("Cell index", particle.cellIndex);
            eo.addEntry("Transport error", static_cast<int>(result.error));
            throw eo;
        }
        return result.step;
    }
#endif

    auto [faceIntersect, timeIntersect, nextCellIndex] =
        this->getIntersectionDetails(particle);

    double shiftedFrequency = particle.frequency * dopplerShift;
    double absorptionOpacity;
    std::size_t group = std::numeric_limits<std::size_t>::max();
    if(this->parameters_.withMultigroupOpacity)
    {
        this->clampFrequencyToBounds(shiftedFrequency);
        group = this->opacity_->findGroup(shiftedFrequency, this->energyBoundaries_);
        absorptionOpacity = this->parameters_.withCompton
            ? this->comptonData_[cellIndex].absorptionOpacity[group]
            : this->opacity_->CalcAbsorptionOpacity(cell, shiftedFrequency);
    }
    else
    {
        absorptionOpacity = this->planckOpacities_[cellIndex];
    }
    double elasticScatteringOpacity = this->parameters_.withCompton
        ? 0.0
        : (this->parameters_.withMultigroupOpacity
            ? this->opacity_->CalcScatteringOpacity(cell, shiftedFrequency)
            : this->scatteringOpacities_[cellIndex]);
    if(!std::isfinite(absorptionOpacity) || absorptionOpacity < 0.0 ||
       !std::isfinite(elasticScatteringOpacity) || elasticScatteringOpacity < 0.0)
    {
        StormError eo("RadiationIMC transport received an invalid opacity");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Frequency", shiftedFrequency);
        eo.addEntry("Absorption opacity", absorptionOpacity);
        eo.addEntry("Scattering opacity", elasticScatteringOpacity);
        throw eo;
    }
    double const transportFleck = this->parameters_.withCompton && group < NumGroups
        ? this->comptonData_[cellIndex].fleck
        : this->factorFleck_[cellIndex];
    double effectiveAbsorptionOpacity =
        (1.0 - transportFleck) * absorptionOpacity;
    double comptonOpacity = 0.0;
    if(this->parameters_.withCompton && group < NumGroups)
    {
        comptonOpacity = this->comptonData_[cellIndex].comptonOutRate[group];
    }
    double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity + comptonOpacity;
    double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
    double _log1p = -std::log1p(this->randomUnitOpen(particle) - 1.0);
    double scatteringDistance = scatteringLength * _log1p / dopplerShift;
    double timeScattering = std::isfinite(scatteringDistance) ? scatteringDistance / fastabs(particle.velocity) : std::numeric_limits<double>::infinity();

    double timeLeft = particle.timeLeft;
    enum Events
    {
        INTERSECTION = 0,
        SCATTERING = 1,
        TIMELEFT = 2,
        OBSERVER = 3
    };
    std::array<std::pair<std::size_t, double>, 4> times;
    times[INTERSECTION] = {INTERSECTION, timeIntersect};
    times[SCATTERING] = {SCATTERING, timeScattering};
    times[TIMELEFT] = {TIMELEFT, timeLeft};
    times[OBSERVER] = {OBSERVER, std::numeric_limits<double>::infinity()};

    ObserverCrossing<PointT> observerCrossing;
    if(this->observer_)
    {
        observerCrossing = this->observer_->nextOutwardCrossing(
            particle.location, particle.velocity, particle.timeLeft);
        if(observerCrossing.hit)
        {
            times[OBSERVER] = {OBSERVER, observerCrossing.time};
        }
        else
        {
            times[OBSERVER] = {OBSERVER, std::numeric_limits<double>::infinity()};
        }
    }

    std::pair<std::size_t, double> min = *std::min_element(times.begin(), times.end(),
        [](const std::pair<std::size_t, double> &a, const std::pair<std::size_t, double> &b) { return a.second < b.second; });
    double dt = min.second;

    particle.timeLeft -= dt;
    double weightEvolutionOpacity = absorptionOpacity * transportFleck;
    double tmp2 = weightEvolutionOpacity * units::clight;
    double tmp = -dt * tmp2;
    double expFactor1 = std::expm1(tmp * dopplerShift);
    double expFactor2 = std::expm1(tmp);
    double integratedForTally = particle.weight * dt;
    if(std::abs(tmp2 * dt) >= 1e-12)
    {
        integratedForTally = particle.weight * expFactor2 * (-1.0 / tmp2);
    }
    particle.location += particle.velocity * dt;
    if(!this->parameters_.noHydroFeedback && !this->parameters_.postProcess.enabled)
    {
        double const materialDeposit = -expFactor2 * particle.weight;
        this->tallyMaterialEnergy(
            cellIndex, materialDeposit, this->parameters_.withCompton);
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.diffusionPressureGradient)
            {
                this->tallyMomentum(
                    cellIndex, -expFactor1 * particle.weight * particle.velocity *
                    units::inv_clight2);
            }
        }
    }
    this->tallyRadiationEnergy(cellIndex, integratedForTally);
    if((this->parameters_.withEgTimeAvg ||
        this->parameters_.withCompton) &&
       this->parameters_.withMultigroupOpacity)
    {
        std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
        if(g < NumGroups)
        {
            this->tallyGroupRadiationEnergy(
                cellIndex, g, integratedForTally);
        }
    }
    double const weightBeforeContinuousDecay = particle.weight;
    particle.weight *= 1.0 + expFactor1;
    if(this->parameters_.postProcess.enabled && this->observer_)
    {
        this->observer_->addAbsorbedEnergy(
            weightBeforeContinuousDecay - particle.weight);
    }
            MCParticle const labParticleBeforeCompton = particle;

    if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
    {
        if(this->observer_ && this->parameters_.postProcess.enabled)
        {
            this->observer_->addCutoffEnergy(particle.weight);
        }
        functionality.change = ParticleStatus::REMOVE;
        if(!this->parameters_.noHydroFeedback && !this->parameters_.postProcess.enabled)
        {
            this->tallyMaterialEnergy(
                cellIndex, particle.weight, this->parameters_.withCompton);
        }
        return functionality;
    }

    if(min.first == INTERSECTION)
    {
        if(this->handlePostProcessExternalSourceBoundary(
               particle, cellIndex, faceIntersect, functionality))
        {
            return functionality;
        }
        if(!particle.radiationState.isDDMC() &&
           this->tryIMCToDDMCInterface(
               particle, functionality, particlesToAdd, cellIndex, nextCellIndex,
               faceIntersect))
        {
            return functionality;
        }
        if(particle.radiationState.bypassCellID !=
           std::numeric_limits<std::size_t>::max())
        {
            std::size_t const exchangedCellID = cellIndex <
                this->ddmcPointCellID_.size()
                ? this->ddmcPointCellID_[cellIndex]
                : std::numeric_limits<std::size_t>::max();
            std::size_t const currentCellID = exchangedCellID ==
                std::numeric_limits<std::size_t>::max() ? cellIndex : exchangedCellID;
            if(currentCellID == particle.radiationState.bypassCellID)
            {
                particle.radiationState.bypassCellID =
                    std::numeric_limits<std::size_t>::max();
            }
        }
        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
        functionality.boundaryCrossing =
            this->grid.IsPointOutsideBox(nextCellIndex);
    }
    else if(min.first == SCATTERING)
    {
        PointT oldVelocity = particle.velocity;
        double D_lab_to_co = dopplerShift;
        double eventRandom = this->randomUnitOpen(particle) * eventOpacity;
        bool isEffectiveScatter = false;
        bool isComptonScatter = false;
        bool comptonTransformedToLab = false;
#ifdef MONTECARLO_POLARIZATION
        MCParticle polarizationMaterialParticle = particle;
        PointT polarizationOldVelocity = particle.velocity;
        if(this->polarizationEnabled())
        {
            polarization::initializeIfNeeded<PointT>(polarizationMaterialParticle);
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(this->parameters_.withHydro && !this->parameters_.MMC)
                {
                    radiation_imc_detail::lorentzTransformToComoving<PointT>(
                        polarizationMaterialParticle, cell);
                    polarizationOldVelocity = polarizationMaterialParticle.velocity;
                    polarizationMaterialParticle.polarizationBasis =
                        polarization::projectBasisToDirection(
                            polarizationMaterialParticle.polarizationBasis,
                            polarizationOldVelocity);
                }
            }
        }
#endif
        if(eventRandom < elasticScatteringOpacity)
        {
#ifdef MONTECARLO_POLARIZATION
            if(this->polarizationEnabled())
            {
                auto u01 = [&]() { return this->randomUnitOpen(particle); };
                PointT const newVelocity = polarization::samplePolarizedThomsonDirection(
                    polarizationMaterialParticle, polarizationOldVelocity, u01);
                polarization::applyThomsonScatter<PointT>(
                    polarizationMaterialParticle, polarizationOldVelocity, newVelocity);
                particle.velocity = polarizationMaterialParticle.velocity;
                particle.stokesQ = polarizationMaterialParticle.stokesQ;
                particle.stokesU = polarizationMaterialParticle.stokesU;
                particle.polarizationBasis = polarizationMaterialParticle.polarizationBasis;
                particle.polarizationInitialized = true;
            }
            else
#endif
            {
            particle.velocity = this->sampleScatterVelocity(cell, particle);
            }
        }
        else if((eventRandom -= elasticScatteringOpacity) < effectiveAbsorptionOpacity)
        {
            particle.velocity = this->sampleScatterVelocity(cell, particle);
            isEffectiveScatter = true;
        }
        else
        {
            if(!this->parameters_.withCompton || group >= NumGroups)
            {
                StormError eo("RadiationIMC selected a Compton event without Compton data");
                eo.addEntry("Cell index", cellIndex);
                eo.addEntry("Group", group);
                throw eo;
            }
            MCParticle comptonParticle = particle;
            if(useComovingTransportFrame)
            {
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    radiation_imc_detail::lorentzTransformToComoving<PointT>(
                        comptonParticle, cell);
                    this->clampFrequencyToBounds(comptonParticle.frequency);
                }
            }
            PointT const comptonOldVelocity = comptonParticle.velocity;
            double const comptonOldWeight = comptonParticle.weight;
            double const comovingMaterialDeposit =
                this->applyComptonScatterEvent(
                    cellIndex,
                    cell,
                    group,
                    comptonParticle,
                    comptonOldVelocity,
                    comptonOldWeight);
            if(useComovingTransportFrame)
            {
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    radiation_imc_detail::lorentzTransformToLab<PointT>(
                        comptonParticle, cell);
                    this->clampFrequencyToBounds(comptonParticle.frequency);
                }
            }
            particle = comptonParticle;
            if(!this->parameters_.noHydroFeedback &&
               !this->parameters_.postProcess.enabled)
            {
                this->tallyMaterialEnergy(
                    cellIndex, comovingMaterialDeposit);
                this->pendingTotalEnergy_[cellIndex] +=
                    labParticleBeforeCompton.weight - particle.weight;
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if(this->parameters_.withHydro &&
                       !this->parameters_.diffusionPressureGradient)
                    {
                        this->tallyMomentum(
                            cellIndex,
                            (labParticleBeforeCompton.weight *
                                 labParticleBeforeCompton.velocity -
                             particle.weight * particle.velocity) *
                                units::inv_clight2);
                    }
                }
            }
            isComptonScatter = true;
            comptonTransformedToLab = useComovingTransportFrame;
        }
        if(this->parameters_.withMultigroupOpacity)
        {
            if(!isComptonScatter)
            {
                particle.frequency *= dopplerShift;
                this->clampFrequencyToBounds(particle.frequency);
            }
            if(isEffectiveScatter)
            {
                if(this->parameters_.withCompton)
                {
                    std::size_t const targetGroup = this->sampleComptonCdf(
                        this->comptonData_[cellIndex].baseSourceCdf,
                        this->randomUnitOpen(particle));
                    particle.frequency =
                        this->frequencyForComptonGroup(targetGroup);
                }
                else
                {
                    double reemitRandom = this->randomUnitOpen(particle);
                    particle.frequency = this->opacity_->GetThermalEnergy(
                        cell, reemitRandom, this->energyBoundaries_);
                }
            }
        }
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC &&
               !isComptonScatter && !comptonTransformedToLab)
            {
                double weightBefore = particle.weight;
                particle.weight *= D_lab_to_co;
                radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
                if(this->parameters_.withMultigroupOpacity)
                {
                    this->clampFrequencyToBounds(particle.frequency);
                }
#ifdef MONTECARLO_POLARIZATION
                if(this->polarizationEnabled())
                {
                    particle.polarizationBasis = polarization::projectBasisToDirection(
                        particle.polarizationBasis, particle.velocity);
                }
#endif
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if(!this->parameters_.diffusionPressureGradient && !this->parameters_.noHydroFeedback)
                    {
                        this->tallyMomentum(
                            cellIndex,
                            (weightBefore * oldVelocity -
                             particle.weight * particle.velocity) *
                                units::inv_clight2);
                    }
                }
            }
        }
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
        {
            if(isEffectiveScatter || isComptonScatter)
            {
                polarization::resetUnpolarized<PointT>(particle);
            }
            polarization::initializeIfNeeded<PointT>(particle);
            particle.polarizationBasis = polarization::projectBasisToDirection(
                particle.polarizationBasis, particle.velocity);
            polarization::clampLinearPolarization(particle.stokesQ, particle.stokesU);
        }
#endif
        functionality.change = ParticleStatus::NO_CELL_MOVE;
    }
    else if(min.first == OBSERVER)
    {
        this->recordObserverCrossing(particle, observerCrossing.point);
        // Leave the packet outside the observer surface so the same positive
        // outward root cannot be selected again on the next step.
        particle.location = observerCrossing.point +
            normalize(particle.velocity) * std::max(1.0e-12, 1.0e-10 *
            std::max(1.0, fastabs(observerCrossing.point)));
        functionality.change = ParticleStatus::NO_CELL_MOVE;
    }
    else if(min.first == TIMELEFT)
    {
        if(this->observer_)
        {
            this->observer_->addTimedOutEnergy(particle.weight);
        }
        functionality.change = ParticleStatus::DONE;
    }

    return functionality;
}

// ============================================================
// postStep
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::postStep(
    const std::vector<MCParticle> &particles,
    double fullDt)
{
    const std::size_t Ncells = this->grid.GetPointNo();
#ifdef STORM_WITH_GPU
    if(this->gpuTransportEnabled_)
    {
        this->gpuData_->AddTallies(this->pendingMaterialEnergy_, this->pendingRadiationEnergy_);
    }
#endif
    this->applyTransportTallies();
    double const tallyDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.transportTime : fullDt;
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        this->Erad_time_avg_[i] /= (tallyDt * this->grid.GetVolume(i));
        if((this->parameters_.withEgTimeAvg ||
            this->parameters_.withCompton) &&
           this->parameters_.withMultigroupOpacity)
        {
            double norm = tallyDt * this->grid.GetVolume(i);
            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                this->Eg_time_avg_[i][g] /= norm;
            }
        }
    }

    // Post-processing is a diagnostic transport pass.  It must not feed
    // packet absorption, Compton residuals, or hydro synchronization back
    // into the snapshot that supplied the source.
    if(this->parameters_.postProcess.enabled)
    {
        return;
    }

    if(!this->parameters_.noHydroFeedback)
    {
        if(this->parameters_.diffusionPressureGradient && this->parameters_.withHydro)
        {
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
#ifdef STORM_WITH_MPI
                {
                    int mpiInit = 0;
                    MPI_Initialized(&mpiInit);
                    if(mpiInit)
                    {
                        ddmc::ExchangePointMetadata(
                            this->grid, this->Erad_time_avg_);
                    }
                }
#endif
            }
        }

        if(this->parameters_.withDDMC)
        {
            this->applyDDMCMomentumFeedback(fullDt);
        }

        for(std::size_t i = 0; i < Ncells; ++i)
        {
            this->throwIfNegativeInternalEnergy(i, "postStep");
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value && radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                if(this->parameters_.withHydro)
                {
                    if(this->parameters_.diffusionPressureGradient)
                    {
                        PointT const radiationEnergyGradient =
                            radiation_pressure_gradient_detail::
                                reconstructRadiationEnergyGradient<PointT>(
                                    this->grid, this->Erad_time_avg_, i);
                        this->extensives_[i].momentum +=
                            radiationEnergyGradient *
                            (-fullDt * this->grid.GetVolume(i) / 3.0);
                    }
                }
            }
            this->synchronizeMaterialCell(i);
        }
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        radiation_imc_detail::clearRadiationEnergyIfPresent(this->extensives_[i]);
        radiation_imc_detail::clearGroupEnergyIfPresent(this->extensives_[i]);
    }
    for(const MCParticle &particle : particles)
    {
        radiation_imc_detail::addRadiationEnergyIfPresent(this->extensives_[particle.cellIndex], particle.weight);
        if(this->parameters_.withMultigroupOpacity)
        {
            if constexpr(radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
            {
                std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
                if(g < NumGroups)
                {
                    this->extensives_[particle.cellIndex].Eg[g] += particle.weight;
                }
            }
        }
    }

    if(this->parameters_.withCompton)
    {
        this->applyComptonEndOfStepCorrection(fullDt);
        std::vector<MCParticle> &mutableParticles =
            const_cast<std::vector<MCParticle> &>(particles);
        this->reconcileComptonParticles(mutableParticles);
        if(!this->parameters_.noHydroFeedback)
        {
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                this->synchronizeMaterialCell(i);
            }
        }
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        const double mass = this->extensives_[i].mass;
        const double radiationPerMass = (mass > 0.0)
            ? radiation_imc_detail::radiationEnergyIfPresent(this->extensives_[i]) / mass
            : 0.0;
        radiation_imc_detail::setCellRadiationEnergyIfPresent(this->cells_[i], radiationPerMass);
        if constexpr(radiation_imc_detail::has_member_group_energy_mutable<CellT>::value)
        {
            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                double groupVal = (mass > 0.0 && radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
                    ? this->traits_.extensiveGroupEnergy(this->extensives_[i], g) / mass
                    : 0.0;
                radiation_imc_detail::setCellGroupEnergyIfPresent(this->cells_[i], g, groupVal);
            }
        }
    }
}

// ============================================================
// generateParticles
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateParticles(double fullDt)
{
    if(this->parameters_.withCompton)
    {
        if(this->postProcessExternalSourceMode_)
        {
            throw StormError(
                "External fixed-flux post-process sources do not support Compton yet");
        }
        return this->generateComptonParticles(fullDt);
    }
    std::vector<MCParticle> newParticles;
    const std::size_t Ncells = this->grid.GetPointNo();
    this->lastGroupSamplingDiagnostics_ = GroupSamplingDiagnostics{};

    std::vector<std::size_t> externalSourceOffsets(Ncells + 1, 0);
    std::vector<std::size_t> externalSourceIndices;
    if(this->postProcessExternalSourceMode_)
    {
        if(this->postProcessExternalSourceLocalCellIndices_.size() !=
           this->postProcessExternalSources_.size())
        {
            throw StormError("External source-to-cell map has inconsistent size");
        }
        for(std::size_t sourceIndex = 0;
            sourceIndex < this->postProcessExternalSources_.size();
            ++sourceIndex)
        {
            auto const &source = this->postProcessExternalSources_[sourceIndex];
            std::size_t const cellIndex =
                this->postProcessExternalSourceLocalCellIndices_[sourceIndex];
            if(source.luminosity > 0.0 && cellIndex < Ncells)
            {
                ++externalSourceOffsets[cellIndex + 1];
            }
        }
        for(std::size_t i = 1; i < externalSourceOffsets.size(); ++i)
        {
            externalSourceOffsets[i] += externalSourceOffsets[i - 1];
        }
        externalSourceIndices.resize(externalSourceOffsets.back());
        std::vector<std::size_t> cursor = externalSourceOffsets;
        for(std::size_t sourceIndex = 0;
            sourceIndex < this->postProcessExternalSources_.size();
            ++sourceIndex)
        {
            auto const &source = this->postProcessExternalSources_[sourceIndex];
            std::size_t const cellIndex =
                this->postProcessExternalSourceLocalCellIndices_[sourceIndex];
            if(source.luminosity > 0.0 && cellIndex < Ncells)
            {
                externalSourceIndices[cursor[cellIndex]++] = sourceIndex;
            }
        }
    }

    std::vector<double> energyToCreateVec(Ncells);
    std::vector<double> gammaVec(Ncells);
    double localTotalEnergy = 0.0;
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT &cell = this->cells_[i];
        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if((this->parameters_.withHydro && !this->parameters_.MMC) ||
               (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities))
            {
                gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
            }
        }
        gammaVec[i] = gamma;
        if(this->postProcessExternalSourceMode_)
        {
            energyToCreateVec[i] = 0.0;
            for(std::size_t offset = externalSourceOffsets[i];
                offset < externalSourceOffsets[i + 1]; ++offset)
            {
                energyToCreateVec[i] +=
                    this->postProcessExternalSources_[
                        externalSourceIndices[offset]].luminosity * fullDt;
            }
        }
        else
        {
            energyToCreateVec[i] = this->factorFleck_[i] *
                this->grid.GetVolume(i) * units::arad *
                boost::math::pow<4>(cell.temperature) *
                this->planckOpacities_[i] * fullDt * units::clight;
        }
        localTotalEnergy += energyToCreateVec[i];
    }

    double globalTotalEnergy = localTotalEnergy;
    std::size_t globalTotalCells = Ncells;
    std::size_t globalSourceCells = static_cast<std::size_t>(std::count_if(
        energyToCreateVec.begin(), energyToCreateVec.end(),
        [](double energy) { return energy > 0.0; }));
#ifdef STORM_WITH_MPI
    {
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Allreduce(MPI_IN_PLACE, &globalTotalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &globalTotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &globalSourceCells, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        }
    }
#endif

    std::size_t const budgetCells = this->postProcessExternalSourceMode_
        ? globalSourceCells : globalTotalCells;
    if(this->parameters_.newPhotonsPerCell >
           std::numeric_limits<std::size_t>::max() / 10 ||
       (this->parameters_.newPhotonsPerCell > 0 &&
        budgetCells > std::numeric_limits<std::size_t>::max() /
            (10 * this->parameters_.newPhotonsPerCell)))
    {
        throw StormError("External source particle budget overflow");
    }
    std::size_t totalParticles =
        budgetCells * this->parameters_.newPhotonsPerCell * 10;
    std::vector<std::size_t> nPhotonsVec(Ncells);
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        std::size_t proportionalShare = (globalTotalEnergy > 0)
            ? static_cast<std::size_t>(energyToCreateVec[i] / globalTotalEnergy * totalParticles)
            : this->parameters_.newPhotonsPerCell;
        nPhotonsVec[i] = energyToCreateVec[i] > 0.0
            ? std::max(this->parameters_.newPhotonsPerCell,
                       std::min(proportionalShare, this->parameters_.newPhotonsPerCell * 20))
            : 0;
    }

    if(this->sourceEmissionControlEnabled_)
    {
        double scoreSum = 0.0;
        for(auto const &kv : this->adaptiveSourceScores_)
        {
            if(std::isfinite(kv.second) && kv.second > 0.0)
            {
                scoreSum += std::pow(
                    kv.second, this->adaptiveSourceScorePower_);
            }
        }

        std::size_t const basePhotons = this->parameters_.newPhotonsPerCell * this->sourceEmissionBaseMultiplier_;
        std::size_t const maxPhotons = static_cast<std::size_t>(std::ceil(
            static_cast<double>(std::max<std::size_t>(1, this->parameters_.newPhotonsPerCell))
            * this->adaptiveSourceMaxFactor_ * this->adaptiveSourceObserverBudgetMultiplier_));
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            std::size_t cellId = radiation_imc_detail::cellID(this->cells_[i]);
            auto const it = this->adaptiveSourceScores_.find(cellId);
            bool const learned = this->adaptiveSourceScoresEnabled_ && it != this->adaptiveSourceScores_.end()
                && std::isfinite(it->second) && it->second > 0.0;

            std::size_t photons = this->sourceEmissionIncludeUniformBase_ ? basePhotons : 0;
            if(this->sourceEmissionUseLearnedScores_ && learned)
            {
                std::size_t learnedPhotons = this->parameters_.newPhotonsPerCell * this->sourceEmissionLearnedBoostFactor_;
                if(scoreSum > 0.0 && this->sourceEmissionLearnedExtraBudget_ > 0)
                {
                    learnedPhotons += static_cast<std::size_t>(std::ceil(
                        this->adaptiveSourceStrength_ * static_cast<double>(this->sourceEmissionLearnedExtraBudget_)
                        * std::pow(it->second,
                                   this->adaptiveSourceScorePower_) /
                          scoreSum));
                }
                std::size_t const minLearned = static_cast<std::size_t>(std::ceil(
                    static_cast<double>(std::max<std::size_t>(1, this->parameters_.newPhotonsPerCell))
                    * this->adaptiveSourceLearnedMinFactor_));
                learnedPhotons = std::max(learnedPhotons, minLearned);
                if(this->adaptiveSourceLearnedMinPhotons_ > 0)
                {
                    learnedPhotons = std::max(
                        learnedPhotons,
                        this->adaptiveSourceLearnedMinPhotons_);
                }
                if(this->adaptiveSourceLearnedMaxPhotons_ > 0)
                {
                    learnedPhotons = std::min(
                        learnedPhotons,
                        this->adaptiveSourceLearnedMaxPhotons_);
                }
                photons = std::max(photons, learnedPhotons);
            }
            nPhotonsVec[i] = std::min(photons, std::max<std::size_t>(1, maxPhotons));
        }
    }

    if(this->postProcessExternalSourceMode_)
    {
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            if(energyToCreateVec[i] > 0.0 && nPhotonsVec[i] == 0)
            {
                nPhotonsVec[i] = 1;
            }
        }
    }
    this->lastSourcePhotonsPerCell_ = nPhotonsVec;
    this->lastSourceAllocationSummary_ = SourceAllocationSummary{};
    this->lastSourceAllocationSummary_.adaptiveEnabled =
        this->sourceEmissionControlEnabled_ && this->sourceEmissionUseLearnedScores_ && this->adaptiveSourceScoresEnabled_;
    std::vector<double> adaptiveScores;
    adaptiveScores.reserve(this->adaptiveSourceScores_.size());
    for(auto const &entry : this->adaptiveSourceScores_)
    {
        if(entry.second > 0.0 && std::isfinite(entry.second))
        {
            adaptiveScores.push_back(entry.second);
        }
    }
    if(!adaptiveScores.empty())
    {
        std::sort(adaptiveScores.begin(), adaptiveScores.end());
        auto percentile = [&adaptiveScores](double quantile)
        {
            double const position = quantile *
                static_cast<double>(adaptiveScores.size() - 1);
            std::size_t const lower = static_cast<std::size_t>(position);
            std::size_t const upper = std::min(
                lower + 1, adaptiveScores.size() - 1);
            double const fraction = position - static_cast<double>(lower);
            return (1.0 - fraction) * adaptiveScores[lower] +
                fraction * adaptiveScores[upper];
        };
        this->lastSourceAllocationSummary_.adaptiveScoreP05 =
            percentile(0.05);
        this->lastSourceAllocationSummary_.adaptiveScoreP50 =
            percentile(0.50);
        this->lastSourceAllocationSummary_.adaptiveScoreP95 =
            percentile(0.95);
        this->lastSourceAllocationSummary_.adaptiveScoreMax =
            adaptiveScores.back();
        this->lastSourceAllocationSummary_.adaptiveScoreSpanLow =
            adaptiveScores.front();
        this->lastSourceAllocationSummary_.adaptiveScoreSpanHigh =
            adaptiveScores.back();
    }
    this->lastSourceAllocationSummary_.minPhotons = std::numeric_limits<std::size_t>::max();
    this->lastSourceAllocationSummary_.learnedMinPhotons = std::numeric_limits<std::size_t>::max();
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        std::size_t const photons = nPhotonsVec[i];
        if(photons == 0)
        {
            continue;
        }
        ++this->lastSourceAllocationSummary_.sourceCells;
        this->lastSourceAllocationSummary_.totalPhotons += photons;
        this->lastSourceAllocationSummary_.minPhotons = std::min(this->lastSourceAllocationSummary_.minPhotons, photons);
        this->lastSourceAllocationSummary_.maxPhotons = std::max(this->lastSourceAllocationSummary_.maxPhotons, photons);
        if(photons > this->parameters_.newPhotonsPerCell)
        {
            ++this->lastSourceAllocationSummary_.boostedCells;
        }

        std::size_t cellId = radiation_imc_detail::cellID(this->cells_[i]);
        auto const it = this->adaptiveSourceScores_.find(cellId);
        bool const learned = this->adaptiveSourceScoresEnabled_ && it != this->adaptiveSourceScores_.end()
            && std::isfinite(it->second) && it->second > 0.0;
        if(learned)
        {
            ++this->lastSourceAllocationSummary_.learnedCells;
            this->lastSourceAllocationSummary_.learnedPhotons += photons;
            this->lastSourceAllocationSummary_.adaptiveScoreSum += it->second;
            this->lastSourceAllocationSummary_.learnedMinPhotons =
                std::min(this->lastSourceAllocationSummary_.learnedMinPhotons, photons);
            this->lastSourceAllocationSummary_.learnedMaxPhotons =
                std::max(this->lastSourceAllocationSummary_.learnedMaxPhotons, photons);
            if(photons >= 1000)
            {
                ++this->lastSourceAllocationSummary_.learnedPhotonsAtLeast1000;
            }
            if(photons >= 2000)
            {
                ++this->lastSourceAllocationSummary_.learnedPhotonsAtLeast2000;
            }
            if(photons > this->parameters_.newPhotonsPerCell)
            {
                ++this->lastSourceAllocationSummary_.learnedBoostedCells;
                this->lastSourceAllocationSummary_.learnedExtraPhotons += photons - this->parameters_.newPhotonsPerCell;
            }
        }
    }
    if(this->lastSourceAllocationSummary_.minPhotons == std::numeric_limits<std::size_t>::max())
    {
        this->lastSourceAllocationSummary_.minPhotons = 0;
    }
    if(this->lastSourceAllocationSummary_.learnedMinPhotons == std::numeric_limits<std::size_t>::max())
    {
        this->lastSourceAllocationSummary_.learnedMinPhotons = 0;
    }

    newParticles.reserve(this->lastSourceAllocationSummary_.totalPhotons);

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT &cell = this->cells_[i];
        double energyToCreate = energyToCreateVec[i];
        double gamma = gammaVec[i];
        std::size_t nPhotonsCell = nPhotonsVec[i];
        if(nPhotonsCell == 0)
        {
            continue;
        }

        // The decomposition is identical for every photon emitted from this cell.
        const PositionDecomposition *cellDecomposition = nullptr;
        if constexpr(kSamplerHasDecomposition)
        {
            this->positionSampler_.BuildDecomposition(
                this->grid, i, this->scratchDecomposition_);
            cellDecomposition = &this->scratchDecomposition_;
        }

        if(!this->parameters_.noHydroFeedback)
        {
            this->extensives_[i].internal_energy -= energyToCreate;
            if constexpr(radiation_imc_detail::has_member_total_energy<ExtensivesT>::value)
            {
                this->extensives_[i].energy -= energyToCreate * gamma;
            }
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
            {
                if(this->parameters_.withHydro && !this->parameters_.diffusionPressureGradient)
                {
                    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                    {
                        this->extensives_[i].momentum -= energyToCreate * cell.velocity * units::inv_clight2 * gamma;
                    }
                }
            }
        }
        double energyPerPhoton = energyToCreate * gamma / nPhotonsCell;

        bool useGroupFreqSampling = this->adaptiveSourceCellGroupScoresEnabled_
            && this->parameters_.withMultigroupOpacity
            && !this->parameters_.withCompton;
        GroupArray physicalPdf{};
        GroupArray samplingPdf{};
        bool groupPdfValid = false;
        bool groupScoreAvailable = false;
        if(useGroupFreqSampling)
        {
            std::size_t cellId = radiation_imc_detail::cellID(cell);
            auto it = this->adaptiveSourceCellGroupScores_.find(cellId);
            if(it != this->adaptiveSourceCellGroupScores_.end())
            {
                groupScoreAvailable = true;
                physicalPdf = this->postProcessExternalSourceMode_
                    ? this->buildPostProcessExternalSourcePlanckPdf(cell)
                    : this->opacity_->GetThermalGroupPdf(
                        cell, this->energyBoundaries_);
                double totalPhys = 0.0;
                std::size_t nPhysGroups = 0;
                for(std::size_t g = 0; g < NumGroups; ++g)
                {
                    if(physicalPdf[g] > 0.0)
                    {
                        ++nPhysGroups;
                        totalPhys += physicalPdf[g];
                    }
                }
                if(totalPhys > 0.0 && nPhysGroups > 0)
                {
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        physicalPdf[g] = (physicalPdf[g] > 0.0) ? physicalPdf[g] / totalPhys : 0.0;
                    }
                    GroupArray const &learnedScoreRaw = it->second;
                    double const scoreFloor = 1e-12;
                    GroupArray learnedPdf{};
                    double learnedTotal = 0.0;
                    for(std::size_t g = 0; g < NumGroups; ++g)
                    {
                        if(physicalPdf[g] > 0.0)
                        {
                            learnedPdf[g] = std::max(learnedScoreRaw[g], scoreFloor);
                            learnedTotal += learnedPdf[g];
                        }
                    }
                    if(learnedTotal > 0.0)
                    {
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            learnedPdf[g] /= learnedTotal;
                        }
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            samplingPdf[g] = (1.0 - this->adaptiveGroupStrength_) * physicalPdf[g]
                                + this->adaptiveGroupStrength_ * learnedPdf[g];
                        }
                        double floorPerGroup = (nPhysGroups > 0) ? this->adaptiveGroupPdfFloor_ / static_cast<double>(nPhysGroups) : 0.0;
                        GroupArray lowerBound{};
                        GroupArray upperBound{};
                        double lowerTotal = 0.0;
                        double upperTotal = 0.0;
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            if(physicalPdf[g] > 0.0)
                            {
                                lowerBound[g] = std::max(floorPerGroup, physicalPdf[g] / this->adaptiveGroupMaxWeightCorrection_);
                                upperBound[g] = std::min(1.0, this->adaptiveGroupMaxBias_ * physicalPdf[g]);
                                lowerBound[g] = std::min(lowerBound[g], upperBound[g]);
                                lowerTotal += lowerBound[g];
                                upperTotal += upperBound[g];
                            }
                            else
                            {
                                samplingPdf[g] = 0.0;
                            }
                        }

                        if(lowerTotal <= 1.0 + 1e-12 && upperTotal >= 1.0 - 1e-12)
                        {
                            std::array<bool, NumGroups> fixed{};
                            double remaining = 1.0;
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                if(!(physicalPdf[g] > 0.0))
                                {
                                    fixed[g] = true;
                                    samplingPdf[g] = 0.0;
                                }
                            }

                            for(std::size_t iter = 0; iter < NumGroups + 2; ++iter)
                            {
                                double freeTotal = 0.0;
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    if(!fixed[g])
                                    {
                                        freeTotal += std::max(samplingPdf[g], 0.0);
                                    }
                                }
                                if(!(freeTotal > 0.0))
                                {
                                    groupPdfValid = false;
                                    break;
                                }

                                bool clamped = false;
                                double const scale = remaining / freeTotal;
                                for(std::size_t g = 0; g < NumGroups; ++g)
                                {
                                    if(fixed[g])
                                    {
                                        continue;
                                    }
                                    double const candidate = std::max(samplingPdf[g], 0.0) * scale;
                                    if(candidate < lowerBound[g])
                                    {
                                        samplingPdf[g] = lowerBound[g];
                                        fixed[g] = true;
                                        remaining -= lowerBound[g];
                                        clamped = true;
                                    }
                                    else if(candidate > upperBound[g])
                                    {
                                        samplingPdf[g] = upperBound[g];
                                        fixed[g] = true;
                                        remaining -= upperBound[g];
                                        clamped = true;
                                    }
                                }

                                if(!clamped)
                                {
                                    for(std::size_t g = 0; g < NumGroups; ++g)
                                    {
                                        if(!fixed[g])
                                        {
                                            samplingPdf[g] = std::max(samplingPdf[g], 0.0) * scale;
                                        }
                                    }
                                    remaining = 0.0;
                                    break;
                                }
                                if(remaining < 0.0)
                                {
                                    break;
                                }
                            }
                        }
                        else
                        {
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                samplingPdf[g] = 0.0;
                            }
                        }

                        double sampTotal = 0.0;
                        for(std::size_t g = 0; g < NumGroups; ++g)
                        {
                            sampTotal += samplingPdf[g];
                        }
                        if(sampTotal > 0.0)
                        {
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                samplingPdf[g] /= sampTotal;
                            }
                            groupPdfValid = true;
                            for(std::size_t g = 0; g < NumGroups; ++g)
                            {
                                if(physicalPdf[g] > 0.0)
                                {
                                    double const correction = physicalPdf[g] / samplingPdf[g];
                                    if(!(samplingPdf[g] > 0.0)
                                        || correction > this->adaptiveGroupMaxWeightCorrection_ * (1.0 + 1e-10)
                                        || samplingPdf[g] > this->adaptiveGroupMaxBias_ * physicalPdf[g] * (1.0 + 1e-10))
                                    {
                                        groupPdfValid = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if(useGroupFreqSampling && groupScoreAvailable && !groupPdfValid)
        {
            ++this->lastGroupSamplingDiagnostics_.invalidPdfFallback;
            this->lastGroupSamplingDiagnostics_.invalidPdfFallbackPackets += nPhotonsCell;
        }

        for(std::size_t j = 0; j < nPhotonsCell; ++j)
        {
            MCParticle particle;
            this->initializeParticleRNG(particle);
            if(this->postProcessExternalSourceMode_)
            {
                std::size_t const begin = externalSourceOffsets[i];
                std::size_t const end = externalSourceOffsets[i + 1];
                if(begin == end)
                {
                    throw StormError(
                        "External source cell has energy but no source faces");
                }
                double const totalLuminosity = energyToCreate / fullDt;
                double const target = this->randomUnitOpen(particle) * totalLuminosity;
                double cumulative = 0.0;
                std::size_t selectedSource = externalSourceIndices[end - 1];
                for(std::size_t offset = begin; offset < end; ++offset)
                {
                    std::size_t const sourceIndex =
                        externalSourceIndices[offset];
                    cumulative += this->postProcessExternalSources_[
                        sourceIndex].luminosity;
                    if(target <= cumulative)
                    {
                        selectedSource = sourceIndex;
                        break;
                    }
                }
                particle = this->generatePostProcessExternalSourceParticle(
                    i, cell,
                    this->postProcessExternalSources_[selectedSource]);
            }
            else
            {
                particle = this->generateSingleParticle(i, cell, cellDecomposition);
            }
            particle.cellID = radiation_imc_detail::cellID(cell);
            particle.sourceCellID = particle.cellID;
            particle.timeLeft = fullDt * this->randomUnitOpen(particle);

            double weightCorrection = 1.0;
            bool usedGroupFrequencySampling = false;

            if(groupPdfValid)
            {
                double rndGroup = this->randomUnitOpen(particle);
                double cumul = 0.0;
                std::size_t selectedGroup = NumGroups - 1;
                for(std::size_t g = 0; g < NumGroups; ++g)
                {
                    cumul += samplingPdf[g];
                    if(rndGroup <= cumul)
                    {
                        selectedGroup = g;
                        break;
                    }
                }
                double freqCo = 0.0;

                if(samplingPdf[selectedGroup] > 0.0)
                {
                    weightCorrection = physicalPdf[selectedGroup] / samplingPdf[selectedGroup];
                    if(weightCorrection > this->adaptiveGroupMaxWeightCorrection_)
                    {
                        ++this->lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                    }
                    else if(weightCorrection > 0.0 && std::isfinite(weightCorrection))
                    {
                        if(this->lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                        {
                            this->lastGroupSamplingDiagnostics_.weightCorrectionMin = weightCorrection;
                        }
                        else
                        {
                            this->lastGroupSamplingDiagnostics_.weightCorrectionMin = std::min(this->lastGroupSamplingDiagnostics_.weightCorrectionMin, weightCorrection);
                        }
                        if(this->lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                        {
                            this->lastGroupSamplingDiagnostics_.weightCorrectionMax = weightCorrection;
                        }
                        else
                        {
                            this->lastGroupSamplingDiagnostics_.weightCorrectionMax = std::max(this->lastGroupSamplingDiagnostics_.weightCorrectionMax, weightCorrection);
                        }
                        this->lastGroupSamplingDiagnostics_.weightCorrectionSum += weightCorrection;
                        ++this->lastGroupSamplingDiagnostics_.weightCorrectionCount;
                        ++this->lastGroupSamplingDiagnostics_.totalSampled;
                        this->lastGroupSamplingDiagnostics_.sampledEnergy += energyPerPhoton;
                        double rndFreq = this->randomUnitOpen(particle);
                        freqCo = this->postProcessExternalSourceMode_
                            ? this->samplePostProcessExternalSourcePlanckFrequencyInGroup(
                                cell, selectedGroup)
                            : this->opacity_->SampleThermalEnergyInGroup(
                                cell, selectedGroup, rndFreq,
                                this->energyBoundaries_);
                        usedGroupFrequencySampling = true;
                    }
                    else
                    {
                        ++this->lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                    }
                }
                else
                {
                    ++this->lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                }

                if(usedGroupFrequencySampling &&
                   ((this->parameters_.withHydro && !this->parameters_.MMC) ||
                    (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities)))
                {
                    double D = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                    particle.frequency = freqCo / D;
                    particle.weight = energyToCreate / (nPhotonsCell * D) * weightCorrection;
                }
                else if(usedGroupFrequencySampling)
                {
                    particle.frequency = freqCo;
                    particle.weight = energyPerPhoton * weightCorrection;
                }
            }

            if(!usedGroupFrequencySampling &&
               ((this->parameters_.withHydro && !this->parameters_.MMC) ||
                (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities)))
            {
                double D = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                if(this->parameters_.withMultigroupOpacity)
                {
                    double rnd = this->randomUnitOpen(particle);
                    double freqCo = this->postProcessExternalSourceMode_
                        ? this->samplePostProcessExternalSourcePlanckFrequency(cell)
                        : this->opacity_->GetThermalEnergy(
                            cell, rnd, this->energyBoundaries_);
                    particle.frequency = freqCo / D;
                }
                particle.weight = energyToCreate / (nPhotonsCell * D);
            }
            else if(!usedGroupFrequencySampling)
            {
                if(this->parameters_.withMultigroupOpacity)
                {
                    particle.frequency = this->postProcessExternalSourceMode_
                        ? this->samplePostProcessExternalSourcePlanckFrequency(cell)
                        : this->opacity_->GetThermalEnergy(
                            cell, this->randomUnitOpen(particle),
                            this->energyBoundaries_);
                }
                particle.weight = energyPerPhoton;
            }
            this->setInitialWeightFromWeight(particle);
            newParticles.push_back(particle);
        }
    }

    return newParticles;
}

// ============================================================
// generateSingleParticle
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateSingleParticle(
    std::size_t cellIndex,
    const CellT &cell)
{
    return this->generateSingleParticle(cellIndex, cell, nullptr);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateSingleParticle(
    std::size_t cellIndex,
    const CellT &cell,
    const PositionDecomposition *decomposition)
{
    MCParticle particle;
    this->initializeParticleRNG(particle);
    particle.id = std::numeric_limits<std::size_t>::max();
    particle.cellIndex = cellIndex;
    particle.cellID = radiation_imc_detail::cellID(cell);
    particle.sourceCellID = particle.cellID;
    particle.frequency = 0.0;
    if constexpr(kSamplerHasDecomposition)
    {
        particle.location = (decomposition != nullptr)
            ? this->positionSampler_.Sample(this->grid, cellIndex, *decomposition,
                                            this->rng_, this->dist_)
            : this->positionSampler_(this->grid, cellIndex, this->rng_, this->dist_);
    }
    else
    {
        (void) decomposition;
        particle.location = this->positionSampler_(this->grid, cellIndex, this->rng_, this->dist_);
    }
    if(this->grid.IsPointOutsideBox(particle.location))
    {
        PointT meshPoint = this->grid.GetMeshPoint(cellIndex);
        PointT original = particle.location;
        PointT direction = meshPoint - original;
        double t = 1e-8;
        while(this->grid.IsPointOutsideBox(particle.location) && t < 1.0)
        {
            particle.location = original + t * direction;
            t *= 2;
        }
        particle.location = particle.location + 1e-8 * (meshPoint - particle.location);
    }

    particle.velocity = this->sampleRandomVelocity(cell, particle);

#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
    {
        polarization::resetUnpolarized<PointT>(particle);
    }
#endif

    if((this->parameters_.withHydro && !this->parameters_.MMC) ||
       (this->parameters_.postProcess.enabled && this->parameters_.postProcess.useCellVelocities))
    {
        radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
#ifdef MONTECARLO_POLARIZATION
        if(this->polarizationEnabled())
        {
            particle.polarizationBasis = polarization::projectBasisToDirection(
                particle.polarizationBasis, particle.velocity);
        }
#endif
    }

    particle.timeLeft = 0.0;
    particle.steps = 0;
    return particle;
}

// ============================================================
// generateInitialParticles
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateInitialParticles(std::size_t particlesPerCell)
{
    if(particlesPerCell == 0)
    {
        return {};
    }

    std::vector<MCParticle> result;
    const std::size_t Ncells = this->grid.GetPointNo();
    result.reserve(Ncells * particlesPerCell);

    const std::size_t Ngroups = this->energyBoundaries_.empty() ? 0 : this->energyBoundaries_.size() - 1;

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        const double totalErad = this->totalRadiationEnergy(i);
        if(totalErad <= 0.0)
        {
            continue;
        }

        std::vector<double> cumulativePlanck;
        if(this->parameters_.withMultigroupOpacity && Ngroups > 0)
        {
            cumulativePlanck.resize(Ngroups + 1);
            cumulativePlanck[0] = 0.0;
            if(this->parameters_.withCompton)
            {
                for(std::size_t g = 1; g <= Ngroups; ++g)
                {
                    double const groupEnergy = std::max(
                        0.0,
                        this->traits_.groupEnergyPerMass(
                            this->cells_[i], g - 1) *
                        this->density(i) * this->grid.GetVolume(i));
                    cumulativePlanck[g] =
                        cumulativePlanck[g - 1] + groupEnergy;
                }
            }
            else
            {
                double const kT =
                    units::k_boltz * this->cells_[i].temperature;
                for(std::size_t g = 1; g <= Ngroups; ++g)
                {
                    double const a = this->energyBoundaries_[g - 1] / kT;
                    double const b = this->energyBoundaries_[g] / kT;
                    cumulativePlanck[g] =
                        planck_integral::planck_integral(a, b) +
                        cumulativePlanck[g - 1];
                }
            }
        }

        const double weightPerPhoton = totalErad / static_cast<double>(particlesPerCell);
        for(std::size_t j = 0; j < particlesPerCell; ++j)
        {
            MCParticle particle = this->generateSingleParticle(i, this->cells_[i]);
            double comovingFrequency = 0.0;
            if(this->parameters_.withMultigroupOpacity && !cumulativePlanck.empty())
            {
                double rnd = this->randomUnitOpen(particle);
                double const total = cumulativePlanck.back();
                if(this->parameters_.withCompton)
                {
                    rnd *= total;
                }
                comovingFrequency = STORM::LinearInterpolation(
                    cumulativePlanck, this->energyBoundaries_, rnd);
            }
            if(this->parameters_.withCompton)
            {
                this->setPacketFromComovingState(
                    particle,
                    this->cells_[i],
                    comovingFrequency,
                    weightPerPhoton);
            }
            else
            {
                particle.weight = weightPerPhoton;
                if(this->parameters_.withMultigroupOpacity &&
                   !cumulativePlanck.empty())
                {
                    particle.frequency = comovingFrequency;
                    this->clampFrequencyToBounds(particle.frequency);
                }
            }
            this->setInitialWeightFromWeight(particle);
            result.push_back(particle);
        }
    }
    return result;
}

// ============================================================
// Compton risk splitting
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::splitComptonRiskyParticles(
    std::vector<MCParticle> &particles,
    double fullDt)
{
    this->comptonDataReusableInPreStep_ = false;
    if(this->parameters_.postProcess.enabled ||
       !this->parameters_.withCompton ||
       !this->parameters_.withMultigroupOpacity)
    {
        return;
    }

    std::size_t const Ncells = this->grid.GetPointNo();
    this->factorFleck_.assign(Ncells, 1.0);
    this->planckOpacities_.assign(Ncells, 0.0);
    this->precomputeComptonData(fullDt);
    this->comptonDataReusableInPreStep_ = true;

    std::vector<std::vector<std::size_t>> bins(Ncells * NumGroups);
    for(std::size_t particleIndex = 0;
        particleIndex < particles.size(); ++particleIndex)
    {
        MCParticle &particle = particles[particleIndex];
        if(particle.cellIndex >= Ncells || !(particle.weight > 0.0))
        {
            continue;
        }
        double frequency = particle.frequency;
        this->clampFrequencyToBounds(frequency);
        std::size_t const group = this->opacity_->findGroup(
            frequency, this->energyBoundaries_);
        if(group >= NumGroups ||
           this->comptonData_[particle.cellIndex].riskTargetPackets[group] == 0)
        {
            continue;
        }
        bins[particle.cellIndex * NumGroups + group].push_back(
            particleIndex);
    }

    std::size_t const maxExtra =
        std::max<std::size_t>(1, particles.size() / 10);
    constexpr std::size_t maxExtraPerCell = 200;
    std::vector<std::size_t> extraPerCell(Ncells, 0);
    std::size_t extraCount = 0;
    particles.reserve(particles.size() + maxExtra);

    for(std::size_t cellIndex = 0;
        cellIndex < Ncells && extraCount < maxExtra; ++cellIndex)
    {
        std::array<std::size_t, NumGroups> riskOrder{};
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            riskOrder[group] = group;
        }
        std::sort(riskOrder.begin(), riskOrder.end(),
            [&](std::size_t left, std::size_t right)
            {
                return this->comptonData_[cellIndex].riskScore[left] >
                    this->comptonData_[cellIndex].riskScore[right];
            });

        for(std::size_t orderIndex = 0;
            orderIndex < NumGroups && extraCount < maxExtra;
            ++orderIndex)
        {
            std::size_t const group = riskOrder[orderIndex];
            std::size_t const target =
                this->comptonData_[cellIndex].riskTargetPackets[group];
            std::vector<std::size_t> const &bin =
                bins[cellIndex * NumGroups + group];
            std::size_t const count = bin.size();
            if(target == 0 || count == 0 || count >= target)
            {
                continue;
            }

            std::size_t allowed = target - count;
            allowed = std::min(allowed, maxExtra - extraCount);
            allowed = std::min(
                allowed, maxExtraPerCell -
                    std::min(extraPerCell[cellIndex], maxExtraPerCell));
            if(allowed == 0)
            {
                continue;
            }

            std::vector<std::size_t> copiesPerOriginal(count, 0);
            for(std::size_t copy = 0; copy < allowed; ++copy)
            {
                ++copiesPerOriginal[copy % count];
            }
            for(std::size_t index = 0; index < count; ++index)
            {
                std::size_t const copies = copiesPerOriginal[index];
                if(copies == 0)
                {
                    continue;
                }
                std::size_t const particleIndex = bin[index];
                std::size_t const pieces = copies + 1;
                double const splitWeight =
                    particles[particleIndex].weight /
                    static_cast<double>(pieces);
                particles[particleIndex].weight = splitWeight;
                this->setInitialWeightFromWeight(particles[particleIndex]);
                for(std::size_t copy = 0; copy < copies; ++copy)
                {
                    MCParticle duplicate = particles[particleIndex];
                    duplicate.weight = splitWeight;
                    duplicate.id = std::numeric_limits<std::size_t>::max();
                    duplicate.steps = 0;
                    this->setInitialWeightFromWeight(duplicate);
                    particles.push_back(duplicate);
                }
            }
            extraCount += allowed;
            extraPerCell[cellIndex] += allowed;
        }
    }
}

// ============================================================
// adjustExistingParticles (MMC)
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::adjustExistingParticles(
    std::vector<MCParticle> &particles,
    double fullDt)
{
    this->splitComptonRiskyParticles(particles, fullDt);
    if(!this->parameters_.MMC)
    {
        return;
    }

    const std::size_t Ncells = this->grid.GetPointNo();
    std::vector<double> divV(Ncells, 0.0);

    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            PointT r_i = this->grid.GetMeshPoint(i);
            for(std::size_t faceIdx : this->grid.GetCellFaces(i))
            {
                const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
                std::size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
                PointT neighborPoint;
                PointT neighborVelocity;
                if(neighborIdx < Ncells && !this->grid.IsPointOutsideBox(neighborIdx))
                {
                    neighborPoint = this->grid.GetMeshPoint(neighborIdx);
                    neighborVelocity = this->cells_[neighborIdx].velocity;
                }
                else
                {
                    neighborPoint = this->grid.FaceCM(faceIdx);
                    neighborVelocity = this->cells_[i].velocity;
                }
                PointT diff = r_i - neighborPoint;
                double distMag = fastabs(diff);
                if(distMag <= 0.0)
                {
                    continue;
                }
                PointT r_ij = diff / distMag;
                double A_ij = this->grid.GetArea(faceIdx);
                divV[i] -= 0.5 * ScalarProd(this->cells_[i].velocity + neighborVelocity, r_ij) * A_ij;
            }
            divV[i] /= this->grid.GetVolume(i);
        }

        const auto [ll, ur] = this->grid.GetBoxCoordinates();

        auto it = particles.begin();
        while(it != particles.end())
        {
            MCParticle &p = *it;
            std::size_t ci = p.cellIndex;
            if(ci < Ncells)
            {
                p.location += this->cells_[ci].velocity * fullDt;
                p.weight += -p.weight * fullDt * divV[ci] / 3.0;
            }

            if(this->grid.IsPointOutsideBox(p.location))
            {
                p.location.x = std::max(ll.x, std::min(ur.x, p.location.x));
                p.location.y = std::max(ll.y, std::min(ur.y, p.location.y));
                p.location.z = std::max(ll.z, std::min(ur.z, p.location.z));
                if(this->boundary)
                {
                    ParticleStatus status = this->boundary->apply(p);
                    if(status == ParticleStatus::REMOVE)
                    {
                        it = particles.erase(it);
                        continue;
                    }
                }
            }
            ++it;
        }
    }
    else
    {
        (void) particles;
        (void) fullDt;
    }

    UpdateNewCells<PointT>(this->grid, particles);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                  TraitsT, PositionSamplerT>::setPostProcessExternalSources(
    std::vector<PostProcessExternalSource> sources)
{
    if(!this->parameters_.postProcess.enabled)
    {
        throw StormError(
            "External sources require RadiationIMC post-process mode");
    }
    if(this->parameters_.withRandomWalk)
    {
        throw StormError(
            "External source surfaces require random-walk acceleration to be disabled");
    }

    std::size_t const localCellCount = this->grid.GetPointNo();
    if(this->cells_.size() < localCellCount)
    {
        throw StormError(
            "External source installation has fewer cells than local tessellation points");
    }

    std::size_t const invalidCellID =
        std::numeric_limits<std::size_t>::max();
    std::size_t const pointCount = std::max(
        this->grid.GetTotalPointNumber(), this->grid.getMeshPoints().size());
    std::vector<std::size_t> pointCellIDs(pointCount, invalidCellID);
    for(std::size_t cellIndex = 0; cellIndex < localCellCount; ++cellIndex)
    {
        pointCellIDs[cellIndex] = radiation_imc_detail::ddmcStableCellID(
            this->grid, cellIndex, this->cells_[cellIndex]);
    }
#ifdef STORM_WITH_MPI
    {
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized)
        {
            ddmc::ExchangePointMetadata(this->grid, pointCellIDs);
        }
    }
#endif

    std::unordered_map<std::size_t, std::size_t> localCellIndexByID;
    localCellIndexByID.reserve(localCellCount);
    for(std::size_t cellIndex = 0; cellIndex < localCellCount; ++cellIndex)
    {
        std::size_t const cellID = pointCellIDs[cellIndex];
        if(cellID == invalidCellID ||
           !localCellIndexByID.emplace(cellID, cellIndex).second)
        {
            throw StormError(
                "External source installation requires unique stable cell IDs");
        }
    }

    std::unordered_map<std::size_t, std::size_t> faceIndex;
    faceIndex.reserve(sources.size());
    std::vector<std::size_t> localSourceCellIndices(
        sources.size(), invalidCellID);
    std::unordered_set<std::size_t> localInteriorIDSet;
    localInteriorIDSet.reserve(sources.size());
    for(std::size_t sourceIndex = 0; sourceIndex < sources.size();
        ++sourceIndex)
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
        if(!(normalMagnitude > 0.0) || !std::isfinite(normalMagnitude))
        {
            throw StormError("External source face has an invalid normal");
        }
        source.outwardNormal = source.outwardNormal / normalMagnitude;

        auto const cellIt = localCellIndexByID.find(source.cellID);
        if(cellIt == localCellIndexByID.end())
        {
            throw StormError(
                "External source references a non-local transport cell");
        }
        std::size_t const cellIndex = cellIt->second;
        localSourceCellIndices[sourceIndex] = cellIndex;
        auto const &cellFaces = this->grid.GetCellFaces(cellIndex);
        if(std::find(cellFaces.begin(), cellFaces.end(), source.faceIndex) ==
           cellFaces.end())
        {
            throw StormError(
                "External source face is not attached to its transport cell");
        }
        auto const neighbors = this->grid.GetFaceNeighbors(source.faceIndex);
        if(neighbors.first != cellIndex && neighbors.second != cellIndex)
        {
            throw StormError(
                "External source face does not contain its transport cell");
        }
        std::size_t const interiorCellIndex = neighbors.first == cellIndex
            ? neighbors.second : neighbors.first;
        if(this->grid.IsPointOutsideBox(interiorCellIndex) ||
           interiorCellIndex >= pointCellIDs.size() ||
           pointCellIDs[interiorCellIndex] != source.interiorCellID)
        {
            throw StormError(
                "External source interior-cell ID does not match the opposite face neighbor");
        }
        if(!faceIndex.emplace(source.faceIndex, sourceIndex).second)
        {
            throw StormError("Duplicate external source face");
        }
        localInteriorIDSet.insert(source.interiorCellID);
    }

    std::unordered_set<std::size_t> globalInteriorIDs = localInteriorIDSet;
#ifdef STORM_WITH_MPI
    if(this->parameters_.withDDMC)
    {
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized)
        {
            std::vector<std::uint64_t> localInteriorIDs;
            localInteriorIDs.reserve(localInteriorIDSet.size());
            for(std::size_t id : localInteriorIDSet)
            {
                localInteriorIDs.push_back(static_cast<std::uint64_t>(id));
            }
            int mpiSize = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
            int const localCount = static_cast<int>(localInteriorIDs.size());
            std::vector<int> counts(static_cast<std::size_t>(mpiSize), 0);
            MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1,
                          MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displacements(
                static_cast<std::size_t>(mpiSize), 0);
            int totalCount = 0;
            for(int rank = 0; rank < mpiSize; ++rank)
            {
                displacements[static_cast<std::size_t>(rank)] = totalCount;
                totalCount += counts[static_cast<std::size_t>(rank)];
            }
            std::vector<std::uint64_t> allInteriorIDs(
                static_cast<std::size_t>(totalCount));
            MPI_Allgatherv(
                localInteriorIDs.empty() ? nullptr : localInteriorIDs.data(),
                localCount, MPI_UINT64_T,
                allInteriorIDs.empty() ? nullptr : allInteriorIDs.data(),
                counts.data(), displacements.data(), MPI_UINT64_T,
                MPI_COMM_WORLD);
            globalInteriorIDs.clear();
            globalInteriorIDs.reserve(allInteriorIDs.size());
            for(std::uint64_t id : allInteriorIDs)
            {
                globalInteriorIDs.insert(static_cast<std::size_t>(id));
            }
        }
    }
#endif

    this->postProcessExternalSources_ = std::move(sources);
    this->postProcessExternalSourceLocalCellIndices_ =
        std::move(localSourceCellIndices);
    this->postProcessExternalSourceFaceIndex_ = std::move(faceIndex);
    this->postProcessExternalSourceInteriorCellIDs_ =
        std::move(globalInteriorIDs);
    this->postProcessExternalSourceMode_ = true;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                  TraitsT, PositionSamplerT>::clearPostProcessExternalSources()
{
    this->postProcessExternalSources_.clear();
    this->postProcessExternalSourceLocalCellIndices_.clear();
    this->postProcessExternalSourceFaceIndex_.clear();
    this->postProcessExternalSourceInteriorCellIDs_.clear();
    this->postProcessExternalSourceMode_ = false;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                    TraitsT, PositionSamplerT>::
samplePostProcessExternalSourceDirection(const PointT &outwardNormal,
                                         MCParticle &particle)
{
    PointT normal = outwardNormal;
    double const normalMagnitude = fastabs(normal);
    if(!(normalMagnitude > 0.0) || !std::isfinite(normalMagnitude))
    {
        throw StormError("External source face has an invalid normal");
    }
    normal = normal / normalMagnitude;
    PointT helper = std::abs(ScalarProd(normal, PointT(0.0, 0.0, 1.0))) < 0.9
        ? PointT(0.0, 0.0, 1.0) : PointT(0.0, 1.0, 0.0);
    PointT tangent1 = helper - ScalarProd(helper, normal) * normal;
    double const tangentMagnitude = fastabs(tangent1);
    if(!(tangentMagnitude > 0.0) || !std::isfinite(tangentMagnitude))
    {
        throw StormError(
            "External source face cannot construct a tangent basis");
    }
    tangent1 = tangent1 / tangentMagnitude;
    PointT tangent2 = CrossProduct(normal, tangent1);
    tangent2 = tangent2 /
        std::max(fastabs(tangent2), std::numeric_limits<double>::min());
    double const mu = std::sqrt(this->randomUnitOpen(particle));
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    double const phi = 2.0 * 3.14159265358979323846 *
        this->randomUnitOpen(particle);
    return mu * normal + sinTheta *
        (std::cos(phi) * tangent1 + std::sin(phi) * tangent2);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      TraitsT, PositionSamplerT>::GroupArray
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
             TraitsT, PositionSamplerT>::
buildPostProcessExternalSourcePlanckPdf(const CellT &cell) const
{
    GroupArray pdf{};
    double const kT = units::k_boltz * cell.temperature;
    if(!(kT > 0.0) || !std::isfinite(kT))
    {
        throw StormError(
            "External source Planck spectrum requires positive finite temperature");
    }
    double total = 0.0;
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        double const left = this->energyBoundaries_[group];
        double const right = this->energyBoundaries_[group + 1];
        double const mass = (left < right)
            ? planck_integral::planck_integral(left / kT, right / kT) : 0.0;
        pdf[group] = (mass > 0.0 && std::isfinite(mass)) ? mass : 0.0;
        total += pdf[group];
    }
    if(total > 0.0 && std::isfinite(total))
    {
        for(double &value : pdf)
        {
            value /= total;
        }
        return pdf;
    }
    double const peakEnergy = 2.8214393721220789 * kT;
    std::size_t fallbackGroup = 0;
    while(fallbackGroup + 1 < NumGroups &&
          peakEnergy >= this->energyBoundaries_[fallbackGroup + 1])
    {
        ++fallbackGroup;
    }
    pdf[fallbackGroup] = 1.0;
    return pdf;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                    TraitsT, PositionSamplerT>::
samplePostProcessExternalSourcePlanckFrequencyInGroup(
    const CellT &cell, std::size_t group)
{
    group = std::min(group, NumGroups - 1);
    double const left = this->energyBoundaries_[group];
    double const right = this->energyBoundaries_[group + 1];
    double const kT = units::k_boltz * cell.temperature;
    double const groupMass = planck_integral::planck_integral(
        left / kT, right / kT);
    if(!(groupMass > 0.0) || !std::isfinite(groupMass))
    {
        return 0.5 * (left + right);
    }
    double const target = this->randomUnitOpen() * groupMass;
    double lo = left;
    double hi = right;
    for(int iteration = 0; iteration < 56; ++iteration)
    {
        double const mid = 0.5 * (lo + hi);
        double const mass = planck_integral::planck_integral(
            left / kT, mid / kT);
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
    this->clampFrequencyToBounds(frequency);
    return frequency;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                    TraitsT, PositionSamplerT>::
samplePostProcessExternalSourcePlanckFrequency(const CellT &cell)
{
    GroupArray const pdf =
        this->buildPostProcessExternalSourcePlanckPdf(cell);
    double const target = this->randomUnitOpen();
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
    return this->samplePostProcessExternalSourcePlanckFrequencyInGroup(
        cell, selectedGroup);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
             TraitsT, PositionSamplerT>::
generatePostProcessExternalSourceParticle(
    std::size_t cellIndex, const CellT &cell,
    const PostProcessExternalSource &source)
{
    MCParticle particle;
    this->initializeParticleRNG(particle);
    particle.id = std::numeric_limits<std::size_t>::max();
    particle.cellIndex = cellIndex;
    particle.velocity = units::clight *
        this->samplePostProcessExternalSourceDirection(
            source.outwardNormal, particle);
    static constexpr double nudge = 1.0e-8;
    particle.location = (1.0 - nudge) * source.location +
        nudge * this->grid.GetMeshPoint(cellIndex);
    if(!this->grid.IsPointInCell(particle.location, cellIndex))
    {
        throw StormError(
            "External source face location did not nudge into its transport cell");
    }
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled &&
            this->parameters_.postProcess.useCellVelocities))
        {
            radiation_imc_detail::lorentzTransformToLab<PointT>(
                particle, cell);
        }
    }
    return particle;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                  TraitsT, PositionSamplerT>::
handlePostProcessExternalSourceBoundary(
    MCParticle &particle, std::size_t cellIndex,
    std::size_t faceIndex, Functionality &functionality)
{
    if(!this->postProcessExternalSourceMode_ ||
       cellIndex >= this->cells_.size())
    {
        return false;
    }
    auto const faceIt =
        this->postProcessExternalSourceFaceIndex_.find(faceIndex);
    if(faceIt == this->postProcessExternalSourceFaceIndex_.end())
    {
        return false;
    }
    PostProcessExternalSource const &source =
        this->postProcessExternalSources_[faceIt->second];
    if(radiation_imc_detail::cellID(this->cells_[cellIndex]) != source.cellID)
    {
        return false;
    }

    PointT normal = source.outwardNormal /
        std::max(fastabs(source.outwardNormal),
                 std::numeric_limits<double>::min());
    double const normalVelocity = ScalarProd(particle.velocity, normal);
    double const directionTolerance =
        1.0e-12 * std::max(fastabs(particle.velocity), 1.0);
    if(normalVelocity > directionTolerance)
    {
        throw StormError(
            "Packet reached the external-source face while moving away from the interior");
    }

    MCParticle materialParticle = particle;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled &&
            this->parameters_.postProcess.useCellVelocities))
        {
            radiation_imc_detail::lorentzTransformToComoving<PointT>(
                materialParticle, this->cells_[cellIndex]);
        }
    }
    materialParticle.velocity = units::clight *
        this->samplePostProcessExternalSourceDirection(normal, particle);
    if(this->parameters_.withMultigroupOpacity)
    {
        materialParticle.frequency =
            this->samplePostProcessExternalSourcePlanckFrequency(
                this->cells_[cellIndex]);
    }
#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
    {
        materialParticle.stokesQ = 0.0;
        materialParticle.stokesU = 0.0;
        materialParticle.polarizationInitialized = false;
    }
#endif
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if((this->parameters_.withHydro && !this->parameters_.MMC) ||
           (this->parameters_.postProcess.enabled &&
            this->parameters_.postProcess.useCellVelocities))
        {
            radiation_imc_detail::lorentzTransformToLab<PointT>(
                materialParticle, this->cells_[cellIndex]);
            this->clampFrequencyToBounds(materialParticle.frequency);
        }
    }
    static constexpr double nudge = 1.0e-8;
    materialParticle.location = (1.0 - nudge) * source.location +
        nudge * this->grid.GetMeshPoint(cellIndex);
    materialParticle.initialWeight = std::abs(materialParticle.weight);
    materialParticle.radiationState.clearDDMC();
    particle = materialParticle;
    functionality.change = ParticleStatus::NO_CELL_MOVE;
    return true;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         TraitsT, PositionSamplerT>::
getAccelerationDebugInfo(std::size_t cellIndex, double frequency) const
{
    if(cellIndex >= this->ddmcCellData_.size())
    {
        return std::string();
    }
    DDMCCellData const &data = this->ddmcCellData_[cellIndex];
    double internalRate = 0.0;
    double internalConductance = 0.0;
    double boundaryRate = 0.0;
    double ddmcChannelRate = 0.0;
    double transportChannelRate = 0.0;
    std::size_t mixedFaces = 0;
    for(DDMCFaceLeak const &face : data.faceLeaks)
    {
        internalRate += face.internalRate;
        internalConductance += face.conductance;
        boundaryRate += face.boundaryRate;
        ddmcChannelRate += face.ddmcRate;
        transportChannelRate += face.transportRate;
        if(face.ddmcRate > 0.0 && face.transportRate > 0.0)
        {
            ++mixedFaces;
        }
    }
    std::ostringstream out;
    out << std::setprecision(17)
        << "eligible=" << (data.eligible ? 1 : 0)
        << " boundary_excluded=" << (data.boundaryExcluded ? 1 : 0)
        << " rigid_boundary_faces=" << data.rigidBoundaryFaceCount
        << " unsupported_boundary_faces="
        << data.unsupportedBoundaryFaceCount
        << " first_unsupported_boundary_face="
        << data.firstUnsupportedBoundaryFace
        << " group_cutoff=" << data.groupCutoff
        << " sigmaT=" << data.sigmaT
        << " sigmaA=" << data.sigmaA
        << " sigmaEnergyAbs=" << data.sigmaEnergyAbs
        << " sigmaDiffusion=" << data.sigmaDiffusion
        << " sigmaParticleGate=" << data.sigmaParticleGate
        << " sigmaGroupExit=" << data.sigmaGroupExit
        << " gamma=" << data.gamma
        << " D=" << data.diffusionCoefficient
        << " leak_rate=" << data.totalLeakRate
        << " faces=" << data.faceLeaks.size()
        << " frequency=" << frequency
        << " ddmc_eligible=" << (data.eligible ? 1 : 0)
        << " ddmc_boundary_excluded="
        << (data.boundaryExcluded ? 1 : 0)
        << " ddmc_rigid_boundary_faces=" << data.rigidBoundaryFaceCount
        << " ddmc_unsupported_boundary_faces="
        << data.unsupportedBoundaryFaceCount
        << " ddmc_first_unsupported_boundary_face="
        << data.firstUnsupportedBoundaryFace
        << " ddmc_group_cutoff=" << data.groupCutoff
        << " ddmc_sigmaT=" << data.sigmaT
        << " ddmc_sigmaA=" << data.sigmaA
        << " ddmc_sigmaEnergyAbs=" << data.sigmaEnergyAbs
        << " ddmc_sigmaDiffusion=" << data.sigmaDiffusion
        << " ddmc_sigmaParticleGate=" << data.sigmaParticleGate
        << " ddmc_sigmaGroupExit=" << data.sigmaGroupExit
        << " ddmc_gamma=" << data.gamma
        << " ddmc_D=" << data.diffusionCoefficient
        << " ddmc_total_leak_rate=" << data.totalLeakRate
        << " ddmc_face_count=" << data.faceLeaks.size()
        << " ddmc_internal_leak_rate_sum=" << internalRate
        << " ddmc_internal_conductance_sum=" << internalConductance
        << " ddmc_boundary_rate_sum=" << boundaryRate
        << " ddmc_channel_rate_sum=" << ddmcChannelRate
        << " ddmc_transport_channel_rate_sum=" << transportChannelRate
        << " ddmc_mixed_face_count=" << mixedFaces
        << " ddmc_resident_leaks=" << this->ddmcResidentLeakCount_
        << " ddmc_transport_leaks=" << this->ddmcTransportLeakCount_
        << " ddmc_remote_resident_leaks="
        << this->ddmcRemoteResidentLeakCount_
        << " ddmc_mpi_face_flux_reductions="
        << this->ddmcMPIFaceFluxReductionCount_
        << " ddmc_leak_invalid_geometry="
        << this->ddmcLeakInvalidGeometryCount_
        << " ddmc_leak_reciprocity_max="
        << this->ddmcLeakReciprocityResidualMax_
        << " ddmc_interface_incident="
        << this->ddmcInterfaceIncidentCount_
        << " ddmc_interface_admitted="
        << this->ddmcInterfaceAdmittedCount_
        << " ddmc_interface_reflected="
        << this->ddmcInterfaceReflectedCount_
        << " ddmc_interface_gu_applied="
        << this->ddmcInterfaceGuAppliedCount_
        << " ddmc_interface_gu_fallback="
        << this->ddmcInterfaceGuFallbackCount_
        << " ddmc_interface_bypass="
        << this->ddmcInterfaceBypassCount_
        << " ddmc_interface_split_packets="
        << this->ddmcInterfaceSplitPacketCount_
        << " ddmc_interface_min_mu=" << this->ddmcInterfaceMinimumMu_
        << " ddmc_interface_max_gu="
        << this->ddmcMovingInterfaceMaxFactor_
        << " ddmc_interface_flux_tallies="
        << this->ddmcInterfaceFluxTallyCount_
        << " ddmc_leak_reciprocity_checks="
        << this->ddmcLeakReciprocityCheckCount_;
    return out.str();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         TraitsT, PositionSamplerT>::
getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const
{
    std::ostringstream out;
    out << std::setprecision(17);
    if(!this->parameters_.withDDMC ||
       !this->parameters_.ddmcInterfaceDiagnostics)
    {
        return out.str();
    }
    for(std::size_t cellIndex = 0;
         cellIndex < this->ddmcCellData_.size(); ++cellIndex)
    {
        DDMCCellData const &data = this->ddmcCellData_[cellIndex];
        std::size_t const sourceID = cellIndex < this->ddmcPointCellID_.size()
            ? this->ddmcPointCellID_[cellIndex]
            : radiation_imc_detail::ddmcStableCellID(
                this->grid, cellIndex, this->cells_[cellIndex]);
        double const sourceGeneratorX =
            this->grid.GetMeshPoint(cellIndex)[0];
        double const sourceCellCMX = this->grid.GetCellCM(cellIndex)[0];
        double const volume = this->grid.GetVolume(cellIndex);
        for(DDMCFaceLeak const &face :
            data.faceLeaks)
        {
            PointT const center = this->grid.FaceCM(face.faceIndex);
            if(center[0] < xMin || center[0] > xMax)
            {
                continue;
            }

            std::size_t const target = face.nextCellIndex;
            std::size_t const targetID = target < this->ddmcPointCellID_.size()
                ? this->ddmcPointCellID_[target]
                : std::numeric_limits<std::size_t>::max();
            double const targetGeneratorX =
                target < this->grid.getMeshPoints().size()
                ? static_cast<double>(this->grid.GetMeshPoint(target)[0])
                : std::numeric_limits<double>::quiet_NaN();
            int const targetEligible = target < this->ddmcPointEligible_.size()
                ? this->ddmcPointEligible_[target] : 0;
            double const targetSigma =
                target < this->ddmcPointSigmaDiffusion_.size()
                ? this->ddmcPointSigmaDiffusion_[target] : 0.0;
            double const targetD =
                target < this->ddmcPointDiffusionCoefficient_.size()
                ? this->ddmcPointDiffusionCoefficient_[target] : 0.0;

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

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename TraitsT,
         typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         TraitsT, PositionSamplerT>::
getDDMCInterfaceEventDiagnosticsTSV(double xMin, double xMax) const
{
    std::ostringstream out;
    out << std::setprecision(17);
    if(!this->parameters_.withDDMC ||
       !this->parameters_.ddmcInterfaceDiagnostics)
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

    for(auto const &item : this->ddmcDiagnosticEvents_)
    {
        DDMCDiagnosticEventKey const &key = item.first;
        DDMCDiagnosticEventAccumulator const &entry = item.second;
        if(entry.faceX < xMin || entry.faceX > xMax)
            continue;
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

} // namespace STORM

#endif // STORM_RADIATION_IMC_HPP
