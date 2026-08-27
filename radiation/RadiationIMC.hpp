#ifndef STORM_RADIATION_IMC_HPP
#define STORM_RADIATION_IMC_HPP

// Define STORM_IMC_DIFF only for the host differential harness.  Production
// CPU and GPU transport must execute the shared kernel.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
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
#include "radiation/ddmc/DDMCSampling.hpp"
#include "radiation/ddmc/DDMCTypes.hpp"
#include "radiation/ddmc/DDMCWollaegerInterface.hpp"
#include "radiation/imc/IMCConcepts.hpp"
#include "radiation/imc/IMCMaterialAccess.hpp"
#include "radiation/imc/IMCComponentBase.hpp"
#include "radiation/imc/IMCState.hpp"
#include <planck_integral/planck_integral.hpp>
#include "../utils/LinearInterpolation.hpp"
#include "../utils/CounterRNG.hpp"
#include "../mesh_movement/MeshMovement.hpp"
#include "../gpu/GreyIMCKernel.hpp"

#ifdef STORM_WITH_GPU
    #include "../gpu/GreyIMCData.hpp"
    #include "../gpu/KokkosRuntime.hpp"
#endif

namespace STORM {

namespace radiation_imc_detail {
template<typename Owner> class IMCLifecycleProcess;
template<typename Owner> class IMCTransportProcess;
template<typename Owner> class IMCSourceProcess;
template<typename Owner> class IMCRandomWalkProcess;
template<typename Owner> class DDMCEngine;
template<typename Owner> class ComptonProcess;
template<typename Owner> class IMCObserverProcess;
template<typename Owner> class IMCDeviceExecutor;
}

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

template<typename PointT,
         typename GridT,
         typename CellT,
         typename ExtensivesT,
         typename EOST,
         std::size_t NumGroups,
         typename OpacityT = RadiationOpacityModel<PointT, GridT, CellT, NumGroups>,
         typename TraitsT = DirectRadiationIMCTraits<PointT, CellT, ExtensivesT, NumGroups>,
         typename PositionSamplerT = RandomInCellPositionSampler<PointT, GridT>>
class RadiationIMC final : public MonteCarloPhysics<PointT, GridT>,
                           private radiation_imc_detail::IMCState<
                               RadiationIMCParameters<NumGroups>,
                               TraitsT,
                               PositionSamplerT,
                               typename radiation_imc_detail::sampler_decomposition<PositionSamplerT>::type,
                               STORM::ComptonCellData<NumGroups>,
                               CMMCComptonBackend<NumGroups>,
                               RadiationObserver<PointT>,
                               RandomWalk,
                               PGRWCellData,
                               ddmc::CellData<PointT>,
                               PointT,
                               NumGroups>
{
public:
    using PointType = PointT;
    using GridType = GridT;
    using CellType = CellT;
    using ExtensivesType = ExtensivesT;
    using EOSType = EOST;
    using OpacityType = OpacityT;
    using TraitsType = TraitsT;
    using PositionSamplerType = PositionSamplerT;
    static constexpr std::size_t kNumGroups = NumGroups;
    static_assert(NumGroups > 0, "RadiationIMC requires at least one frequency group");
    static_assert(radiation_imc_detail::has_opacity_calc_planck<OpacityT, CellT>::value,
        "OpacityT must provide CalcPlanckOpacity(const CellT &)");
    static_assert(radiation_imc_detail::has_opacity_calc_absorption<OpacityT, CellT>::value,
        "OpacityT must provide CalcAbsorptionOpacity(const CellT &, double)");
    static_assert(radiation_imc_detail::has_opacity_calc_scattering<OpacityT, CellT>::value,
        "OpacityT must provide CalcScatteringOpacity(const CellT &)");
    static_assert(radiation_imc_detail::has_opacity_calc_scattering_frequency<OpacityT, CellT>::value,
        "OpacityT must provide CalcScatteringOpacity(const CellT &, double)");
    static_assert(radiation_imc_detail::has_opacity_random_velocity<OpacityT, PointT, CellT>::value,
        "OpacityT must provide getRandomVelocity(const CellT &, double, double)");
    static_assert(radiation_imc_detail::has_opacity_scatter_velocity<OpacityT, PointT, CellT>::value,
        "OpacityT must provide getNewScatterVelocity(const CellT &, const PointT &, double, double, double)");
    static_assert(radiation_imc_detail::has_opacity_find_group<OpacityT, NumGroups>::value,
        "OpacityT must provide findGroup(double, const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_thermal_energy<OpacityT, CellT, NumGroups>::value,
        "OpacityT must provide GetThermalEnergy(const CellT &, double, const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_sample_thermal_in_group<OpacityT, CellT, NumGroups>::value,
        "OpacityT must provide SampleThermalEnergyInGroup(const CellT &, size_t, double, const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_thermal_group_pdf<OpacityT, CellT, NumGroups>::value,
        "OpacityT must provide GetThermalGroupPdf(const CellT &, const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_cumulative_opacity<OpacityT, CellT, NumGroups>::value,
        "OpacityT must provide GetCumulativeOpacity(const CellT &, const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_energy_centers<OpacityT, NumGroups>::value,
        "OpacityT must provide getEnergyCenters(const GroupBoundaries &)");
    static_assert(radiation_imc_detail::has_opacity_reseed<OpacityT>::value,
        "OpacityT must provide reseed(uint64_t)");

    using Base = MonteCarloPhysics<PointT, GridT>;
    using MCParticle = Particle<PointT>;
    using Functionality = StepResult;
    using BoundaryCond = BoundaryCondition<PointT, GridT>;
    using Parameters = RadiationIMCParameters<NumGroups>;
    using OpacityModel = OpacityT;
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
    using State = radiation_imc_detail::IMCState<
        RadiationIMCParameters<NumGroups>,
        TraitsT,
        PositionSamplerT,
        PositionDecomposition,
        STORM::ComptonCellData<NumGroups>,
        CMMCComptonBackend<NumGroups>,
        RadiationObserver<PointT>,
        RandomWalk,
        PGRWCellData,
        ddmc::CellData<PointT>,
        PointT,
        NumGroups>;

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
    Functionality stepImpl(MCParticle &particle, std::vector<MCParticle> &particlesToAdd);
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

    const std::vector<double> &getFactorFleck() const
    {
        return this->factorFleck_;
    }
    const std::vector<double> &getPlanckOpacities() const
    {
        return this->planckOpacities_;
    }
    const std::vector<double> &getScatteringOpacities() const
    {
        return this->scatteringOpacities_;
    }
    const std::vector<ComptonCellData> &getComptonData() const
    {
        return this->comptonData_;
    }
    const GroupArray &getComptonGroupCenters() const
    {
        return this->comptonGroupCenters_;
    }
    const GroupArray &getComptonGroupWidths() const
    {
        return this->comptonGroupWidths_;
    }
    const std::vector<double> &getEradTimeAvg() const
    {
        return this->Erad_time_avg_;
    }
    std::vector<double> &getEradTimeAvg()
    {
        return this->Erad_time_avg_;
    }
    const std::vector<GroupArray> &getEgTimeAvg() const
    {
        return this->Eg_time_avg_;
    }
    std::vector<GroupArray> &getEgTimeAvg()
    {
        return this->Eg_time_avg_;
    }
#ifdef STORM_WITH_GPU
    bool UsesDeviceTransport() const;
    gpu::GreyIMCViews<gpu::DeviceVec3> GetDeviceTransportViews() const;
#endif
    const GroupBoundaries &getEnergyBoundaries() const
    {
        return this->energyBoundaries_;
    }
    const Parameters &getParameters() const
    {
        return this->parameters_;
    }
    const SourceAllocationSummary &getLastSourceAllocationSummary() const
    {
        return this->lastSourceAllocationSummary_;
    }
    const std::vector<std::size_t> &getLastSourcePhotonsPerCell() const
    {
        return this->lastSourcePhotonsPerCell_;
    }
    GroupSamplingDiagnostics getLastGroupSamplingDiagnostics() const
    {
        return this->lastGroupSamplingDiagnostics_;
    }
    void setObserver(std::shared_ptr<Observer> observer)
    {
        this->observer_ = std::move(observer);
        if(this->observer_)
        {
            this->observer_->setPolarizationEnabled(this->polarizationEnabled());
        }
    }
    const std::shared_ptr<Observer> &getObserver() const
    {
        return this->observer_;
    }

    void reseedRNG(std::uint64_t seed)
    {
        std::uint64_t rankOffset = 0;
#ifdef STORM_WITH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rankOffset = static_cast<std::uint64_t>(rank) * 104729ULL;
#endif
        this->rng_.seed(seed + rankOffset);
        this->particleRngSeed_ = seed + rankOffset;
        this->sourceRngStreamCounter_ = 0;
        this->opacity_->reseed(seed + 1ULL);
        ReseedRandomInCell(seed + 2ULL);
    }

    std::size_t getRandomWalkStepCount() const override
    {
        return this->rwStepCount_;
    }
    std::size_t getDDMCStepCount() const override
    {
        return this->ddmcStepCount_;
    }
    std::size_t getDDMCLeakCount() const override
    {
        return this->ddmcLeakCount_;
    }
    std::size_t getDDMCCensusCount() const override
    {
        return this->ddmcCensusCount_;
    }
    std::size_t getDDMCUpscatterCount() const override
    {
        return this->ddmcUpscatterCount_;
    }
    std::size_t getDDMCFallbackCount() const override
    {
        return this->ddmcFallbackCount_;
    }
    std::string getAccelerationDebugInfo(std::size_t cellIndex, double frequency) const override;
    std::string getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const;
    std::string getDDMCInterfaceEventDiagnosticsTSV(double xMin, double xMax) const;
    std::size_t getDDMCMovingInterfaceBypassCount(void) const
    {
        return this->ddmcMovingInterfaceBypassCount_;
    }
    inline double getDDMCMovingInterfaceMaxFactor(void) const
    {
        return this->ddmcMovingInterfaceMaxFactor_;
    }
    inline double getDDMCLeakReciprocityResidualMax(void) const
    {
        return this->ddmcLeakReciprocityResidualMax_;
    }
    inline std::size_t getDDMCLeakReciprocityCheckCount(void) const
    {
        return this->ddmcLeakReciprocityCheckCount_;
    }
    inline const std::vector<DDMCCellData> &getDDMCCellData(void) const
    {
        return this->ddmcCellData_;
    }
    inline const std::vector<PointT> &getDDMCFluxRhsIntegrated(void) const
    {
        return this->ddmcFluxRhsIntegrated_;
    }
    inline std::size_t getDDMCMomentumFeedbackCount(void) const
    {
        return this->ddmcMomentumFeedbackCount_;
    }
    inline std::size_t getDDMCMomentumMatrixFallbackCount(void) const
    {
        return this->ddmcMomentumMatrixFallbackCount_;
    }

    void setPostProcessExternalSources(std::vector<PostProcessExternalSource> sources);
    void clearPostProcessExternalSources();
    inline bool hasPostProcessExternalSources(void) const
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

public:
    enum class Event
    {
        Face,
        Scatter,
        Census
    };

    bool GreyKernelEligible() const;
    bool SharedRandomWalkKernelEligible() const;
    gpu::GreyIMCViews<PointT> GetHostTransportViews();

    // Set while the differential harness replays a step through the legacy
    // event code so both paths can be compared on identical input.
    bool imcDiffForceLegacy_ = false;
    std::size_t imcDiffReports_ = 0;

    bool SharedFullIMCKernelEligible() const;

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
        ComptonCorrectionFailure failure = ComptonCorrectionFailure::None;
    };

private:

    std::vector<MCParticle> generateParticles(double fullDt);
    void validateGridSizedState() const;
    void validateEnergyBoundaries() const;
    void rejectUnsupportedParameters() const;
    void rejectUnsupportedParameter(const std::string &name) const;
    inline bool polarizationEnabled(void) const
    {
        return this->parameters_.withPolarization or this->parameters_.postProcess.polarization.enabled;
    }
    void precomputeComptonData(double sourceDt);
    void initializeComptonMatrixGenerator();
    std::vector<double> buildComptonTemperatures() const;
    GroupCdf buildSafeComptonCdf(const GroupArray &weights) const;
    void buildComptonMatricesForCell(const CellT &cell, std::size_t cellIndex, ComptonOccupationMode occupationMode, ComptonCellData &data);
    double computeLteTemperature(const CellT &cell, std::size_t cellIndex) const;
    void recomputeComptonContractions(ComptonCellData &data) const;
    void buildComptonSources(double sourceDt, ComptonCellData &data) const;
    void buildComptonEventData(ComptonCellData &data) const;
    void computeComptonRiskForCell(double fullDt, ComptonCellData &data) const;
    std::vector<MCParticle> generateComptonParticles(double fullDt);
    void applyComptonEndOfStepCorrection(double fullDt);
    void reconcileComptonParticles(std::vector<MCParticle> &particles);
    void splitComptonRiskyParticles(std::vector<MCParticle> &particles, double fullDt);
    std::size_t sampleComptonCdf(const GroupCdf &cdf, double random) const;
    double sumComptonGroups(const GroupArray &values);
    double minComptonGroup(const GroupArray &values);
    double maxAbsComptonGroup(const GroupArray &values);
    double normComptonGroups(const GroupArray &values);
    double compensatedSumComptonGroups(const GroupArray &values);
    const char *comptonCorrectionFailureName(ComptonCorrectionFailure failure);
    bool solveComptonGroupSystem(GroupMatrix matrix, GroupArray rhs, GroupArray &solution);
    GroupArray multiplyComptonMatrix(const GroupMatrix &matrix, const GroupArray &values);
    double relativeComptonResidual(const GroupMatrix &matrix, const GroupArray &solution, const GroupArray &rhs, double scale);
    ComptonProjectionResult projectNonnegativeConservative(
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
    void setPacketFromComovingState(MCParticle &particle, const CellT &cell, double comovingFrequency, double comovingWeight) const;
    double applyComptonScatterEvent(std::size_t cellIndex, CellT &cell, std::size_t sourceGroup, MCParticle &particle, const PointT &oldVelocity, double oldWeight);
    std::size_t sampleComptonTarget(const ComptonCellData &data, std::size_t sourceGroup, MCParticle &particle);
    void addComptonMaterialExchange(std::size_t cellIndex, double energy);
    void recordObserverCrossing(const MCParticle &particle, const PointT &crossingPoint);
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
    void tallyMaterialEnergy(std::size_t cellIndex, double energy, bool addToTotalEnergy = false);
    void tallyMomentum(std::size_t cellIndex, const PointT &momentum);
    void tallyRadiationEnergy(std::size_t cellIndex, double integratedEnergy);
    void tallyGroupRadiationEnergy(std::size_t cellIndex, std::size_t group, double integratedEnergy);
    void applyTransportTallies();
    void synchronizeMaterialCell(std::size_t cellIndex);
    void throwIfNegativeInternalEnergy(std::size_t cellIndex, const std::string &where);

    void precomputeRandomWalkData();
    bool tryRandomWalkStep(MCParticle &particle, Functionality &functionality);
    double computeMinDistanceToFaces(std::size_t cellIndex, const PointT &location) const;
    double computeCellSurfaceArea(std::size_t cellIndex) const;

    void precomputeDDMCData();
    bool tryDDMCStep(MCParticle &particle, Functionality &functionality);
    void addDDMCFluxContribution(std::size_t cellIndex, const PointT &contribution);
    void applyDDMCMomentumFeedback(double fullDt);
    bool tryIMCToDDMCInterface(MCParticle &particle, Functionality &functionality, std::vector<MCParticle> &particlesToAdd, std::size_t sourceCellIndex, std::size_t targetCellIndex, std::size_t faceIndex);

public:
    using DDMCDiagnosticEventKind = radiation_imc_detail::DDMCDiagnosticEventKind;
    using DDMCDiagnosticEventKey = radiation_imc_detail::DDMCDiagnosticEventKey;
    using DDMCDiagnosticEventAccumulator = radiation_imc_detail::DDMCDiagnosticEventAccumulator;
    static constexpr std::size_t DDMC_DIAGNOSTIC_GREY_GROUP = radiation_imc_detail::DDMC_DIAGNOSTIC_GREY_GROUP;

private:

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
    PointT samplePostProcessExternalSourceDirection(const PointT &outwardNormal, MCParticle &particle);
    GroupArray buildPostProcessExternalSourcePlanckPdf(const CellT &cell) const;
    double samplePostProcessExternalSourcePlanckFrequencyInGroup(const CellT &cell, std::size_t group);
    double samplePostProcessExternalSourcePlanckFrequency(const CellT &cell);
    MCParticle generatePostProcessExternalSourceParticle(std::size_t cellIndex, const CellT &cell, const PostProcessExternalSource &source);
    bool handlePostProcessExternalSourceBoundary(MCParticle &particle, std::size_t cellIndex, std::size_t faceIndex, Functionality &functionality);

    std::vector<CellT> &cells_;
    std::vector<ExtensivesT> &extensives_;
    std::shared_ptr<EOST> eos_;
    std::shared_ptr<OpacityModel> opacity_;
    std::vector<std::size_t> lastSourcePhotonsPerCell_;
    SourceAllocationSummary lastSourceAllocationSummary_;
    GroupSamplingDiagnostics lastGroupSamplingDiagnostics_;
    bool postProcessExternalSourceMode_ = false;
    std::vector<PostProcessExternalSource> postProcessExternalSources_;
    std::vector<std::size_t> postProcessExternalSourceLocalCellIndices_;
    std::unordered_map<std::size_t, std::size_t> postProcessExternalSourceFaceIndex_;
    std::unordered_set<std::size_t> postProcessExternalSourceInteriorCellIDs_;

    const GridT &componentGrid() const
    {
        return this->grid;
    }
    const std::shared_ptr<BoundaryCond> &componentBoundary() const
    {
        return this->boundary;
    }
    const auto &componentGridData() const
    {
        return this->gridData;
    }
    auto &componentGridData()
    {
        return this->gridData;
    }
    auto componentIntersectionDetails(MCParticle &particle)
    {
        return this->getIntersectionDetails(particle);
    }

    template<typename> friend class radiation_imc_detail::IMCLifecycleProcess;
    template<typename> friend class radiation_imc_detail::IMCTransportProcess;
    template<typename> friend class radiation_imc_detail::IMCSourceProcess;
    template<typename> friend class radiation_imc_detail::IMCRandomWalkProcess;
    template<typename> friend class radiation_imc_detail::DDMCEngine;
    template<typename> friend class radiation_imc_detail::ComptonProcess;
    template<typename> friend class radiation_imc_detail::IMCObserverProcess;
    template<typename> friend class radiation_imc_detail::IMCDeviceExecutor;

    std::unique_ptr<radiation_imc_detail::IMCLifecycleProcess<RadiationIMC>> lifecycleProcess_;
    std::unique_ptr<radiation_imc_detail::IMCTransportProcess<RadiationIMC>> transportProcess_;
    std::unique_ptr<radiation_imc_detail::IMCSourceProcess<RadiationIMC>> sourceProcess_;
    std::unique_ptr<radiation_imc_detail::IMCRandomWalkProcess<RadiationIMC>> randomWalkProcess_;
    std::unique_ptr<radiation_imc_detail::DDMCEngine<RadiationIMC>> ddmcEngine_;
    std::unique_ptr<radiation_imc_detail::ComptonProcess<RadiationIMC>> comptonProcess_;
    std::unique_ptr<radiation_imc_detail::IMCObserverProcess<RadiationIMC>> observerProcess_;
    std::unique_ptr<radiation_imc_detail::IMCDeviceExecutor<RadiationIMC>> deviceExecutor_;
};

} // namespace STORM

#include "radiation/imc/IMCLifecycleProcess.hpp"
#include "radiation/transport/IMCTransportProcess.hpp"
#include "radiation/source/IMCSourceProcess.hpp"
#include "radiation/random_walk/IMCRandomWalkProcess.hpp"
#include "radiation/ddmc/DDMCEngine.hpp"
#include "radiation/compton/ComptonProcess.hpp"
#include "radiation/observer/IMCObserverProcess.hpp"
#include "radiation/gpu/IMCDeviceExecutor.hpp"
#include "RadiationIMCFacade.hpp"

#endif // STORM_RADIATION_IMC_HPP
