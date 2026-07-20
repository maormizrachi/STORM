#ifndef STORM_RADIATION_IMC_HPP
#define STORM_RADIATION_IMC_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
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
#include "radiation/RadiationIMCTraits.hpp"
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/Compton.hpp"
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
#include "../mesh_movement/MeshMovement.hpp"

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
        double beta2 = ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2;
        if(beta2 < 1e-20)
        {
            return;
        }
        double gamma = 1.0 / std::sqrt(1.0 - beta2);
        double vDotDir = ScalarProd(cell.velocity, particle.velocity);
        double speed = fastabs(particle.velocity);
        if(speed <= 0.0)
        {
            return;
        }
        PointT dir = particle.velocity / speed;
        double vDotDirNorm = ScalarProd(cell.velocity, dir);
        double D = 1.0 - vDotDirNorm / units::clight;
        particle.frequency *= D;
        particle.weight *= D;
        PointT newDir = dir - cell.velocity / units::clight
            + (gamma - 1.0) * vDotDirNorm / (beta2 * units::clight2) * cell.velocity;
        double newDirMag = fastabs(newDir);
        if(newDirMag > 0.0)
        {
            particle.velocity = newDir * (units::clight / newDirMag);
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
         typename TraitsT = DirectRadiationIMCTraits<PointT, CellT, ExtensivesT, NumGroups>,
         typename PositionSamplerT = RandomInCellPositionSampler<PointT, GridT>>
class RadiationIMC : public MonteCarloPhysics<PointT, GridT>
{
public:
    static_assert(NumGroups > 0, "RadiationIMC requires at least one frequency group");

    using Base = MonteCarloPhysics<PointT, GridT>;
    using MCParticle = Particle<PointT, GridT>;
    using Functionality = StepResult<PointT, GridT>;
    using BoundaryCond = BoundaryCondition<PointT, GridT>;
    using Parameters = RadiationIMCParameters<NumGroups>;
    using OpacityModel = RadiationOpacityModel<PointT, GridT, CellT, NumGroups>;
    using GroupArray = std::array<double, NumGroups>;
    using GroupBoundaries = std::array<double, NumGroups + 1>;

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

    using ComptonKernel = ComptonKernelModel<PointT, GridT, CellT, NumGroups>;
    using ComptonResult = ComptonKernelResult<NumGroups>;
    using ComptonCellData = STORM::ComptonCellData<NumGroups>;
    using Observer = RadiationObserver<PointT>;

    using DDMCFaceLeak = ddmc::FaceLeak<PointT>;
    using DDMCCellData = ddmc::CellData<PointT>;

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
    std::vector<MCParticle> generateInitialParticles(std::size_t particlesPerCell);
    void adjustExistingParticles(std::vector<MCParticle> &particles, double fullDt);

    const std::vector<double> &getFactorFleck() const { return this->factorFleck_; }
    const std::vector<double> &getPlanckOpacities() const { return this->planckOpacities_; }
    const std::vector<double> &getEradTimeAvg() const { return this->Erad_time_avg_; }
    std::vector<double> &getEradTimeAvg() { return this->Erad_time_avg_; }
    const std::vector<GroupArray> &getEgTimeAvg() const { return this->Eg_time_avg_; }
    std::vector<GroupArray> &getEgTimeAvg() { return this->Eg_time_avg_; }
    const GroupBoundaries &getEnergyBoundaries() const { return this->energyBoundaries_; }
    const Parameters &getParameters() const { return this->parameters_; }
    const SourceAllocationSummary &getLastSourceAllocationSummary() const { return this->lastSourceAllocationSummary_; }
    const std::vector<std::size_t> &getLastSourcePhotonsPerCell() const { return this->lastSourcePhotonsPerCell_; }
    GroupSamplingDiagnostics getLastGroupSamplingDiagnostics() const { return this->lastGroupSamplingDiagnostics_; }
    const std::vector<ComptonCellData> &getComptonData() const { return this->comptonData_; }
    const GroupArray &getComptonGroupCenters() const { return this->comptonGroupCenters_; }
    const GroupArray &getComptonGroupWidths() const { return this->comptonGroupWidths_; }
    void setComptonKernel(std::shared_ptr<const ComptonKernel> kernel)
    {
        if(this->preStepInitialized_)
        {
            throw StormError("RadiationIMC::setComptonKernel must be called before preStep");
        }
        this->comptonKernel_ = std::move(kernel);
    }
    const std::shared_ptr<const ComptonKernel> &getComptonKernel() const
    {
        return this->comptonKernel_;
    }
    void setObserver(std::shared_ptr<Observer> observer)
    {
        this->observer_ = std::move(observer);
        if(this->observer_)
        {
            this->observer_->setPolarizationEnabled(this->polarizationEnabled());
        }
    }
    const std::shared_ptr<Observer> &getObserver() const { return this->observer_; }

    std::size_t getRandomWalkStepCount() const override { return this->rwStepCount_; }
    std::size_t getDDMCStepCount() const override { return this->ddmcStepCount_; }
    std::size_t getDDMCLeakCount() const override { return this->ddmcLeakCount_; }
    std::size_t getDDMCCensusCount() const { return this->ddmcCensusCount_; }
    std::size_t getDDMCUpscatterCount() const { return this->ddmcUpscatterCount_; }
    std::size_t getDDMCFallbackCount() const { return this->ddmcFallbackCount_; }
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

    void setNewPhotonsPerCell(std::size_t n);
    void setAdaptiveSourceCellScores(std::unordered_map<std::size_t, double> scores,
                                     double strength,
                                     double maxFactor,
                                     double learnedReserveFrac,
                                     double learnedMinFactor,
                                     double observerBudgetMultiplier);
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
    std::vector<MCParticle> generateComptonResidualParticles(double transportDt);
    std::size_t sampleComptonTarget(const ComptonCellData &data, std::size_t sourceGroup);
    void addComptonMaterialExchange(std::size_t cellIndex, double energy);
    void recordObserverCrossing(const MCParticle &particle,
                                const PointT &crossingPoint);
    void setInitialWeightFromWeight(MCParticle &particle) const;
    double randomUnitOpen();
    double density(std::size_t cellIndex) const;
    double specificInternalEnergy(std::size_t cellIndex) const;
    double totalRadiationEnergy(std::size_t cellIndex) const;
    void depositMaterialEnergy(std::size_t cellIndex, double energy);
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
                               std::size_t sourceCellIndex,
                               std::size_t targetCellIndex,
                               std::size_t faceIndex);

    void clampFrequencyToBounds(double &frequency) const;

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
    std::vector<double> Erad_time_avg_;
    std::vector<GroupArray> Eg_time_avg_;
    std::vector<std::size_t> lastSourcePhotonsPerCell_;
    SourceAllocationSummary lastSourceAllocationSummary_;
    GroupSamplingDiagnostics lastGroupSamplingDiagnostics_;
    std::vector<ComptonCellData> comptonData_;
    GroupArray comptonGroupCenters_{};
    GroupArray comptonGroupWidths_{};
    std::shared_ptr<const ComptonKernel> comptonKernel_;
    std::shared_ptr<Observer> observer_;
    bool preStepInitialized_ = false;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_;

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
#endif
    this->opacity_->reseed(seed + 1ULL);

    const std::size_t Ncells = this->grid.GetPointNo();
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
    if(this->parameters_.withCompton && this->parameters_.comptonAngleDependent)
    {
        rejectUnsupportedParameter(
            "comptonAngleDependent=true (the current Compton kernel is group-only)");
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
    this->extensives_[cellIndex].internal_energy += energy;
    this->throwIfNegativeInternalEnergy(cellIndex, "depositMaterialEnergy");
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
    double learnedReserveFrac, double learnedMinFactor, double observerBudgetMultiplier)
{
    this->adaptiveSourceScores_ = std::move(scores);
    this->adaptiveSourceStrength_ = std::clamp(strength, 0.0, 1.0);
    this->adaptiveSourceMaxFactor_ = std::max(1.0, maxFactor);
    this->adaptiveSourceLearnedReserveFrac_ = std::clamp(learnedReserveFrac, 0.0, 1.0);
    this->adaptiveSourceLearnedMinFactor_ = std::max(1.0, learnedMinFactor);
    this->adaptiveSourceObserverBudgetMultiplier_ = std::max(1.0, observerBudgetMultiplier);
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
    const auto &normals = this->gridData.normalsOfCells[cellIndex];
    const auto &facePoints = this->gridData.pointsOnFaces[cellIndex];
    double Ro = std::numeric_limits<double>::max();
    for(std::size_t f = 0; f < normals.size(); ++f)
    {
        double d = ScalarProd(location - facePoints[f], normals[f]);
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
        double scatOp = this->opacity_->CalcScatteringOpacity(cell);
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
    double tauLeak = this->randomWalk_->sampleLeakTime(this->randomUnitOpen());
    double tLeak = tauLeak * Ro * Ro / D_phys;

    double tCensus = particle.timeLeft;

    double tUpscatter = std::numeric_limits<double>::max();
    if(isPGRW && gamma_rw < 1.0 && sigma_a_eff > 0.0 && f > 0.0)
    {
        double xiUp = this->randomUnitOpen();
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
        this->extensives_[cellIndex].internal_energy += -rwExp * particle.weight;
    }
    if(rwAbsRate > 0.0)
    {
        this->Erad_time_avg_[cellIndex] += particle.weight * rwExp * (-1.0 / rwAbsRate);
        if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
        {
            std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
            if(g < NumGroups)
            {
                this->Eg_time_avg_[cellIndex][g] += particle.weight * rwExp * (-1.0 / rwAbsRate);
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
            this->extensives_[cellIndex].internal_energy += particle.weight;
        }
        return true;
    }

    constexpr double RW_PI = 3.14159265358979323846;
    double cosTheta = 2.0 * this->randomUnitOpen() - 1.0;
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    double phi = 2.0 * RW_PI * this->randomUnitOpen();
    PointT posDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);

    double displacement;
    if(rwEvent == RW_LEAK)
    {
        displacement = Ro;
    }
    else
    {
        double tauPos = D_phys * dt / (Ro * Ro);
        displacement = Ro * this->randomWalk_->sampleRadius(tauPos, this->randomUnitOpen());
    }

    if(displacement > Ro * (1.0 + 1e-12))
    {
        displacement = Ro;
    }

    PointT rwCenter = particle.location;
    particle.location = rwCenter + displacement * posDir;

    static constexpr double nudge = 1e-6;
    particle.location = particle.location * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

    const auto &normals = this->gridData.normalsOfCells[cellIndex];
    const auto &facePoints = this->gridData.pointsOnFaces[cellIndex];
    for(std::size_t fi = 0; fi < normals.size(); ++fi)
    {
        double d = ScalarProd(particle.location - facePoints[fi], normals[fi]);
        if(d < 0.0)
        {
            displacement *= 0.99;
            particle.location = rwCenter + displacement * posDir;
            fi = static_cast<std::size_t>(-1);
        }
    }

    particle.velocity = this->opacity_->getRandomVelocity(cell, this->rng_, this->dist_);

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
        polarization::applyAcceleratedPolarizationHistory<PointT>(
            polarizationParticle, dtCo,
            std::max(0.0, sigmaT - sigma_a_eff),
            std::max(0.0, (1.0 - f) * sigma_a_eff),
            particle.velocity,
            this->parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
            this->parameters_.postProcess.polarization.depolarizationScatterings,
            this->rng_, this->dist_);
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
            double xi = this->randomUnitOpen();
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
                    this->extensives_[cellIndex].momentum += (oldWeight * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
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
        double scatOp = this->opacity_->CalcScatteringOpacity(cell);
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
                data.groupCutoff = cutoff;
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
                if(this->grid.IsPointOutsideBox(next) &&
                   this->boundary->getDDMCBoundaryFaceBehavior(
                       faceIdx, i, next) !=
                       DDMCBoundaryFaceBehavior::ReflectingRigid)
                {
                    data.boundaryExcluded = true;
                    data.eligible = false;
                    data.eligibilityReason = ddmc::EligibilityReason::BoundaryExcluded;
                    break;
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
            double normalMag = fastabs(normal);
            if(normalMag <= 0.0)
            {
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
            if(sourceDistance <= 0.0)
            {
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
                    this->grid.GetArea(faceIdx), sourceDistance,
                    data.diffusionCoefficient, targetDistance,
                    this->ddmcPointDiffusionCoefficient_[nextCellIndex]);
                internalRate = conductance / volume;
            }

            double boundaryRate = ddmc::BoundaryLeakRate(
                this->grid.GetArea(faceIdx), volume, data.sigmaDiffusion,
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
                faceLeak.area = this->grid.GetArea(faceIdx);
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
            polarization::finalizeAcceleratedPolarizationHistory<PointT>(
                particle, finalVelocity,
                this->parameters_.postProcess.polarization.manualScatteringsAfterAcceleration,
                this->parameters_.postProcess.polarization.depolarizationScatterings,
                this->rng_, this->dist_);
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
                particle.velocity = this->opacity_->getRandomVelocity(
                    this->cells_[cellIndex], this->rng_, this->dist_);
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
       data.sigmaEnergyAbs > 0.0 && f > 0.0)
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

    double tEvent = -std::log(this->randomUnitOpen()) / eventRate;
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
        double const scatteringOpacity = std::max(0.0,
            this->opacity_->CalcScatteringOpacity(this->cells_[cellIndex]));
        double const explicitResetOpacity = upscatterRate / units::clight;
        polarization::accumulateAcceleratedPolarizationHistory<PointT>(
            particle, dt,
            scatteringOpacity,
            std::max(0.0, (1.0 - fHistory) * data.sigmaEnergyAbs -
                              explicitResetOpacity),
            this->rng_, this->dist_);
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
        this->extensives_[cellIndex].internal_energy += absorbedEnergy;
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                     radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro &&
               !this->parameters_.diffusionPressureGradient)
            {
                this->extensives_[cellIndex].momentum +=
                    absorbedEnergy * this->cells_[cellIndex].velocity *
                    units::inv_clight2;
            }
        }
    }

    double integratedForTally = (absRate > 0.0)
        ? oldWeight * expFactor * (-1.0 / absRate)
        : oldWeight * dt;
    this->Erad_time_avg_[cellIndex] += integratedForTally;

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
                    this->Eg_time_avg_[cellIndex][g] += integratedForTally *
                        groupMass / bandMass;
                }
            }
        }
        else
        {
            std::size_t g = this->opacity_->findGroup(
                particle.frequency, this->energyBoundaries_);
            if(g < NumGroups)
            {
                this->Eg_time_avg_[cellIndex][g] += integratedForTally;
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
            this->extensives_[cellIndex].internal_energy += particle.weight;
            if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value &&
                         radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if(this->parameters_.withHydro &&
                   !this->parameters_.diffusionPressureGradient)
                {
                    this->extensives_[cellIndex].momentum +=
                        particle.weight * this->cells_[cellIndex].velocity *
                        units::inv_clight2;
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
        particle.velocity = this->opacity_->getRandomVelocity(
            this->cells_[cellIndex], this->rng_, this->dist_);
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
        particle.velocity = this->opacity_->getRandomVelocity(
            this->cells_[cellIndex], this->rng_, this->dist_);
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
                double remaining = this->randomUnitOpen() * bandMass;
                for(std::size_t group = 0; group < data.groupCutoff; ++group)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        this->energyBoundaries_, kT, group, group + 1);
                    if(remaining <= groupMass || group + 1 == data.groupCutoff)
                    {
                        double const localRandom = groupMass > 0.0
                            ? std::clamp(remaining / groupMass, 0.0, 1.0)
                            : this->randomUnitOpen();
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
                    this->cells_[cellIndex], this->randomUnitOpen(),
                    this->energyBoundaries_);
            }
            this->clampFrequencyToBounds(particle.frequency);
        }
        particle.velocity = this->opacity_->getRandomVelocity(
            this->cells_[cellIndex], this->rng_, this->dist_);
        finalizePolarization(particle.velocity);
        convertResidentToLab();
        particle.radiationState.clearDDMC();
        functionality.change = ParticleStatus::DONE;
        ++this->ddmcCensusCount_;
        return true;
    }

    double eventPick = this->randomUnitOpen() * eventRate;
    if(eventPick <= data.totalLeakRate)
    {
        double facePick = this->randomUnitOpen() * data.totalLeakRate;
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

        constexpr double DDMC_PI = 3.14159265358979323846;
        bool const useDDMCChannel =
            chosen->ddmcRate > 0.0 &&
            this->randomUnitOpen() < chosen->ddmcRate / chosen->rate;
        double mu = useDDMCChannel
            ? ddmc::SampleAsymptoticMu(this->randomUnitOpen())
            : std::sqrt(this->randomUnitOpen());
        double phiLeak = 2.0 * DDMC_PI * this->randomUnitOpen();
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
                double remaining = this->randomUnitOpen() * bandMass;
                for(std::size_t group = beginGroup;
                    group < data.groupCutoff; ++group)
                {
                    double const groupMass = ddmc::PlanckBandMass(
                        this->energyBoundaries_, kT, group, group + 1);
                    if(remaining <= groupMass || group + 1 == data.groupCutoff)
                    {
                        double const localRandom = groupMass > 0.0
                            ? std::clamp(remaining / groupMass, 0.0, 1.0)
                            : this->randomUnitOpen();
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
        double remaining = this->randomUnitOpen() * upperBandMass;
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
                    : this->randomUnitOpen();
                particle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                    cell, selectedGroup, localRandom, this->energyBoundaries_);
                break;
            }
            remaining -= groupMass;
        }
        this->clampFrequencyToBounds(particle.frequency);
        particle.velocity = this->opacity_->getRandomVelocity(cell, this->rng_, this->dist_);
        exitDDMCToTransport(false);
        functionality.change = ParticleStatus::NO_CELL_MOVE;
        ++this->ddmcUpscatterCount_;
        return true;
    }
    return true;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tryIMCToDDMCInterface(
    MCParticle &particle,
    Functionality &functionality,
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

    if(this->parameters_.ddmcUseMultigroupPGRW &&
       this->parameters_.withMultigroupOpacity)
    {
        std::size_t const cutoff = this->ddmcPointGroupCutoff_[targetCellIndex];
        double frequency = targetComoving.frequency;
        this->clampFrequencyToBounds(frequency);
        if(cutoff == 0 || cutoff > NumGroups ||
           frequency >= this->energyBoundaries_[cutoff])
        {
            return false;
        }
    }

    double const speed = fastabs(faceComoving.velocity);
    if(!(speed > 0.0) || !std::isfinite(speed))
        return false;
    double const mu = ScalarProd(faceComoving.velocity / speed, normal);
    if(!(mu > 0.0) || !std::isfinite(mu))
        return false;

    double movingFactor = 1.0;
    auto bypassMovingInterface = [&]()
    {
        ++this->ddmcMovingInterfaceBypassCount_;
        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = targetCellIndex;
        return true;
    };
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if(useVelocityFrames)
        {
            double const betaNormal = -ScalarProd(faceVelocity, normal) *
                units::inv_clight;
            if(!std::isfinite(betaNormal) || std::abs(betaNormal) > 0.5)
            {
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
                particle.radiationState.bypassCellID = targetID;
                return bypassMovingInterface();
            }
        }
    }

    double const targetOpacity =
        this->ddmcPointSigmaDiffusion_[targetCellIndex];
    double const targetDistance = std::abs(ScalarProd(
        targetCenter - this->grid.FaceCM(faceIndex), normal));
    double const admission = ddmc::StaticAdmissionProbability(
        mu, targetOpacity, targetDistance);

    if(this->randomUnitOpen() > admission)
    {
        // Diffuse-albedo rejection stays in the source IMC cell.  The
        // incoming direction is not reflected specularly at a transport-
        // diffusion interface.
        constexpr double pi = 3.14159265358979323846;
        double const reflectedMu = std::sqrt(this->randomUnitOpen());
        double const sinTheta = std::sqrt(
            std::max(0.0, 1.0 - reflectedMu * reflectedMu));
        double const phi = 2.0 * pi * this->randomUnitOpen();
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
        return true;
    }

    faceComoving.weight *= movingFactor;
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
        PointT const contribution = particle.weight *
            (targetComoving.velocity / admittedSpeed);
        if(targetCellIndex < this->grid.GetPointNo())
        {
            this->addDDMCFluxContribution(targetCellIndex, contribution);
        }
        else
        {
            particle.radiationState.pendingFlux = contribution;
            particle.radiationState.set(
                RadiationTransportState<PointT>::PendingFlux);
        }
    }
    particle.location = this->grid.FaceCM(faceIndex);
    particle.velocity = this->opacity_->getRandomVelocity(
        this->cells_[sourceCellIndex], this->rng_, this->dist_);
#ifdef MONTECARLO_POLARIZATION
    if(this->polarizationEnabled())
    {
        particle.polarizationBasis = polarization::projectBasisToDirection(
            particle.polarizationBasis, particle.velocity);
    }
#endif
    functionality.change = ParticleStatus::CELL_MOVE;
    functionality.nextCellIndex = targetCellIndex;
    return true;
}

// ============================================================
// preStep
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::precomputeComptonData(double sourceDt)
{
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
    if(!this->comptonKernel_)
    {
        StormError eo("RadiationIMC withCompton requires a ComptonKernelModel");
        eo.addEntry("Reason", "Compton matrices must be frozen in preStep");
        throw eo;
    }

    const std::size_t Ncells = this->grid.GetPointNo();
    this->comptonData_.assign(Ncells, ComptonCellData{});
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT const &cell = this->cells_[i];
        ComptonResult result = this->comptonKernel_->build(
            cell, this->density(i), cell.temperature, this->energyBoundaries_,
            this->comptonGroupCenters_, sourceDt,
            this->parameters_.comptonMatrixSamples, this->rng_);
        if(!isValidComptonResult(result))
        {
            StormError eo("RadiationIMC Compton kernel returned invalid frozen data");
            eo.addEntry("Cell index", i);
            throw eo;
        }

        ComptonCellData &data = this->comptonData_[i];
        data.active = true;
        data.fleck = this->factorFleck_[i];
        data.groupCenters = this->comptonGroupCenters_;
        data.groupWidths = this->comptonGroupWidths_;
        data.rates = result.rates;
        data.derivative = result.derivative;
        data.residualSource = result.residualSource;
        data.signedSourceActive = result.hasResidualSource;

        for(std::size_t source = 0; source < NumGroups; ++source)
        {
            double outRate = 0.0;
            for(std::size_t target = 0; target < NumGroups; ++target)
            {
                if(target != source)
                {
                    outRate += result.rates[source][target];
                }
            }
            if(!std::isfinite(outRate) || outRate < 0.0)
            {
                StormError eo("RadiationIMC Compton kernel returned invalid outgoing rate");
                eo.addEntry("Cell index", i);
                eo.addEntry("Source group", source);
                eo.addEntry("Outgoing rate", outRate);
                throw eo;
            }
            data.outRate[source] = outRate;
            double total = 0.0;
            for(std::size_t target = 0; target < NumGroups; ++target)
            {
                if(target != source)
                {
                    total += result.rates[source][target];
                }
                data.targetCdf[source][target] = total;
            }
            if(outRate > 0.0)
            {
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    data.targetCdf[source][target] /= outRate;
                }
            }
            data.meanEnergyRatio[source] = 1.0;
            if(outRate > 0.0)
            {
                data.meanEnergyRatio[source] = 0.0;
                for(std::size_t target = 0; target < NumGroups; ++target)
                {
                    if(target == source) continue;
                    data.meanEnergyRatio[source] +=
                        result.rates[source][target] / outRate *
                        this->comptonGroupCenters_[target] /
                        std::max(this->comptonGroupCenters_[source], std::numeric_limits<double>::min());
                }
            }
        }
        double minimumModifiedFleck = 1.0;
        for(std::size_t source = 0; source < NumGroups; ++source)
        {
            double const kernelFleck = result.hasFleckCorrection
                ? result.fleckCorrection[source] : 1.0;
            data.modifiedFleck[source] = std::clamp(
                data.fleck * kernelFleck, 0.0, 1.0);
            minimumModifiedFleck = std::min(
                minimumModifiedFleck, data.modifiedFleck[source]);
        }
        // Keep the scalar as a conservative summary for diagnostics and
        // legacy callers; transport uses modifiedFleck[sourceGroup].
        data.fleck = minimumModifiedFleck;
        for(double source : data.residualSource)
        {
            data.signedSourceNet += source;
            data.signedSourceL1 += std::abs(source);
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::size_t RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::sampleComptonTarget(
    const ComptonCellData &data, std::size_t sourceGroup)
{
    if(sourceGroup >= NumGroups || !(data.outRate[sourceGroup] > 0.0))
    {
        return sourceGroup;
    }
    double const target = this->randomUnitOpen();
    for(std::size_t group = 0; group < NumGroups; ++group)
    {
        if(group == sourceGroup) continue;
        if(target <= data.targetCdf[sourceGroup][group])
        {
            return group;
        }
    }
    for(std::size_t group = NumGroups; group-- > 0;)
    {
        if(group != sourceGroup && data.rates[sourceGroup][group] > 0.0) return group;
    }
    return sourceGroup;
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::addComptonMaterialExchange(
    std::size_t cellIndex, double energy)
{
    if(this->parameters_.noHydroFeedback || this->parameters_.postProcess.enabled)
    {
        return;
    }
    this->extensives_[cellIndex].internal_energy += energy;
    radiation_imc_detail::addTotalEnergyIfPresent(this->extensives_[cellIndex], energy);
    this->throwIfNegativeInternalEnergy(cellIndex, "Compton transport");
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::generateComptonResidualParticles(double transportDt)
{
    std::vector<MCParticle> result;
    if(!this->parameters_.withCompton || this->comptonData_.empty())
    {
        return result;
    }
    for(std::size_t cellIndex = 0; cellIndex < this->comptonData_.size(); ++cellIndex)
    {
        ComptonCellData const &data = this->comptonData_[cellIndex];
        for(std::size_t group = 0; group < NumGroups; ++group)
        {
            double const sourceEnergy = data.residualSource[group];
            if(!data.signedSourceActive || sourceEnergy == 0.0)
            {
                continue;
            }
            std::size_t const packetCount = std::max<std::size_t>(1, this->parameters_.newPhotonsPerCell);
            this->addComptonMaterialExchange(cellIndex, -sourceEnergy);
            for(std::size_t packetIndex = 0; packetIndex < packetCount; ++packetIndex)
            {
                MCParticle particle = this->generateSingleParticle(cellIndex, this->cells_[cellIndex]);
                particle.timeLeft = transportDt * this->randomUnitOpen();
                particle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                    this->cells_[cellIndex], group, this->randomUnitOpen(), this->energyBoundaries_);
                particle.weight = sourceEnergy / static_cast<double>(packetCount);
                this->setInitialWeightFromWeight(particle);
                result.push_back(particle);
            }
        }
    }
    return result;
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
    double const fleckDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.sourceDt : fullDt;
    this->planckOpacities_.assign(Ncells, 0.0);
    this->factorFleck_.assign(Ncells, 1.0);
    this->Erad_time_avg_.assign(Ncells, 0.0);
    if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
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

        this->planckOpacities_[i] = this->opacity_->CalcPlanckOpacity(cell);
        if(!std::isfinite(this->planckOpacities_[i]) ||
           this->planckOpacities_[i] < 0.0)
        {
            StormError eo("RadiationIMC::preStep received an invalid Planck opacity");
            eo.addEntry("Cell index", i);
            eo.addEntry("Planck opacity", this->planckOpacities_[i]);
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

    double const sourceDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.sourceDt : fullDt;
    double const transportDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.transportTime : fullDt;
    this->precomputeComptonData(sourceDt);
    std::vector<MCParticle> newParticles = this->generateParticles(sourceDt);
    if(this->parameters_.postProcess.enabled)
    {
        for(MCParticle &particle : newParticles)
        {
            particle.timeLeft = transportDt * this->randomUnitOpen();
        }
    }
    std::vector<MCParticle> residualParticles =
        this->generateComptonResidualParticles(transportDt);
    newParticles.insert(newParticles.end(), residualParticles.begin(), residualParticles.end());
    if(this->boundary)
    {
        std::vector<MCParticle> boundaryParticles = this->boundary->generateNewBoundaryParticles(fullDt);
        for(MCParticle &particle : boundaryParticles)
        {
            this->setInitialWeightFromWeight(particle);
            if(this->parameters_.postProcess.enabled)
            {
                particle.timeLeft = transportDt * this->randomUnitOpen();
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
    (void) particlesToAdd;
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

    auto [faceIntersect, timeIntersect, nextCellIndex] =
        this->getIntersectionDetails(particle);

    double shiftedFrequency = particle.frequency * dopplerShift;
    double absorptionOpacity;
    std::size_t group = std::numeric_limits<std::size_t>::max();
    if(this->parameters_.withMultigroupOpacity)
    {
        this->clampFrequencyToBounds(shiftedFrequency);
        group = this->opacity_->findGroup(shiftedFrequency, this->energyBoundaries_);
        absorptionOpacity = this->opacity_->CalcAbsorptionOpacity(cell, shiftedFrequency);
    }
    else
    {
        absorptionOpacity = this->planckOpacities_[cellIndex];
    }
    double elasticScatteringOpacity = this->parameters_.withMultigroupOpacity
        ? this->opacity_->CalcScatteringOpacity(cell, shiftedFrequency)
        : this->opacity_->CalcScatteringOpacity(cell);
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
        ? this->comptonData_[cellIndex].modifiedFleck[group]
        : this->factorFleck_[cellIndex];
    double effectiveAbsorptionOpacity = (1.0 - transportFleck) * absorptionOpacity;
    double comptonOpacity = 0.0;
    if(this->parameters_.withCompton && group < NumGroups)
    {
        comptonOpacity = this->comptonData_[cellIndex].outRate[group];
    }
    double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity + comptonOpacity;
    double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
    double _log1p = -std::log1p(this->randomUnitOpen() - 1.0);
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
        this->extensives_[cellIndex].internal_energy += materialDeposit;
        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.diffusionPressureGradient)
            {
                this->extensives_[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
            }
        }
    }
    this->Erad_time_avg_[cellIndex] += integratedForTally;
    if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
    {
        std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
        if(g < NumGroups)
        {
            this->Eg_time_avg_[cellIndex][g] += integratedForTally;
        }
    }
    double const weightBeforeContinuousDecay = particle.weight;
    particle.weight *= 1.0 + expFactor1;
    if(this->parameters_.postProcess.enabled && this->observer_)
    {
        this->observer_->addAbsorbedEnergy(
            weightBeforeContinuousDecay - particle.weight);
    }

    if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
    {
        if(this->observer_ && this->parameters_.postProcess.enabled)
        {
            this->observer_->addCutoffEnergy(particle.weight);
        }
        functionality.change = ParticleStatus::REMOVE;
        if(!this->parameters_.noHydroFeedback && !this->parameters_.postProcess.enabled)
        {
            this->extensives_[cellIndex].internal_energy += particle.weight;
        }
        return functionality;
    }

    if(min.first == INTERSECTION)
    {
        if(!particle.radiationState.isDDMC() &&
           this->tryIMCToDDMCInterface(
               particle, functionality, cellIndex, nextCellIndex,
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
        double oldWeight = particle.weight;
        double D_lab_to_co = dopplerShift;
        double eventRandom = this->randomUnitOpen() * eventOpacity;
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
                auto u01 = [&]() { return this->randomUnitOpen(); };
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
            particle.velocity = this->opacity_->getNewScatterVelocity(cell, particle.velocity, particle.frequency, this->rng_, this->dist_);
            }
        }
        else if((eventRandom -= elasticScatteringOpacity) < effectiveAbsorptionOpacity)
        {
            particle.velocity = this->opacity_->getNewScatterVelocity(cell, particle.velocity, particle.frequency, this->rng_, this->dist_);
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
            std::size_t const targetGroup = this->sampleComptonTarget(
                this->comptonData_[cellIndex], group);
            if(targetGroup == group || targetGroup >= NumGroups)
            {
                StormError eo("RadiationIMC Compton event has no valid target group");
                eo.addEntry("Cell index", cellIndex);
                eo.addEntry("Source group", group);
                throw eo;
            }
            // The Compton kernel is defined in the material-comoving frame.
            // Transform the incoming packet into that frame, perform the
            // group-changing event there, and transform the result back
            // before applying the lab-frame material energy ledger.
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
            double const meanRatio = this->comptonData_[cellIndex].meanEnergyRatio[group];
            comptonParticle.weight *= meanRatio;
            comptonParticle.frequency = this->opacity_->SampleThermalEnergyInGroup(
                cell, targetGroup, this->randomUnitOpen(), this->energyBoundaries_);
            comptonParticle.velocity = this->opacity_->getNewScatterVelocity(
                cell, comptonParticle.velocity, comptonParticle.frequency,
                this->rng_, this->dist_);
            if(useComovingTransportFrame)
            {
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    radiation_imc_detail::lorentzTransformToLab<PointT>(
                        comptonParticle, cell);
                    this->clampFrequencyToBounds(comptonParticle.frequency);
                }
            }
            particle.weight = comptonParticle.weight;
            particle.frequency = comptonParticle.frequency;
            particle.velocity = comptonParticle.velocity;
            this->addComptonMaterialExchange(cellIndex, oldWeight - particle.weight);
            ++this->comptonData_[cellIndex].implicitEvents;
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
                double reemitRandom = this->randomUnitOpen();
                particle.frequency = this->opacity_->GetThermalEnergy(cell, reemitRandom, this->energyBoundaries_);
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
                        this->extensives_[cellIndex].momentum += (weightBefore * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
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
    double const tallyDt = this->parameters_.postProcess.enabled
        ? this->parameters_.postProcess.transportTime : fullDt;
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        this->Erad_time_avg_[i] /= (tallyDt * this->grid.GetVolume(i));
        if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
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
        std::vector<double> Erad_time_avg_grad(Ncells, 0.0);
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
                for(std::size_t i = 0; i < Ncells; ++i)
                {
                    const PointT &point = this->grid.GetMeshPoint(i);
                    std::size_t neighbor_right = std::numeric_limits<std::size_t>::max();
                    std::size_t neighbor_left = std::numeric_limits<std::size_t>::max();
                    for(std::size_t faceIdx : this->grid.GetCellFaces(i))
                    {
                        const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
                        std::size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
                        PointT diff = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                        if(diff.x > 0.99)
                        {
                            neighbor_right = neighborIdx;
                        }
                        else if(diff.x < -0.99)
                        {
                            neighbor_left = neighborIdx;
                        }
                    }
                    if(neighbor_right == std::numeric_limits<std::size_t>::max())
                    {
                        StormError eo("No right neighbor found in RadiationIMC::postStep");
                        throw eo;
                    }
                    if(neighbor_left == std::numeric_limits<std::size_t>::max())
                    {
                        StormError eo("No left neighbor found in RadiationIMC::postStep");
                        throw eo;
                    }
                    const PointT &neighbor_right_point = this->grid.GetMeshPoint(neighbor_right);
                    const PointT &neighbor_left_point = this->grid.GetMeshPoint(neighbor_left);
                    double grad;
                    if(this->grid.IsPointOutsideBox(neighbor_left))
                    {
                        grad = (this->Erad_time_avg_[neighbor_right] - this->Erad_time_avg_[i]) / (neighbor_right_point.x - point.x);
                    }
                    else if(this->grid.IsPointOutsideBox(neighbor_right))
                    {
                        grad = (this->Erad_time_avg_[i] - this->Erad_time_avg_[neighbor_left]) / (point.x - neighbor_left_point.x);
                    }
                    else
                    {
                        grad = (this->Erad_time_avg_[neighbor_right] - this->Erad_time_avg_[neighbor_left]) / (neighbor_right_point.x - neighbor_left_point.x);
                    }
                    Erad_time_avg_grad[i] = grad;
                }
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
                        this->extensives_[i].momentum.x -= fullDt * this->grid.GetVolume(i) * Erad_time_avg_grad[i] / 3.0;
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
    std::vector<MCParticle> newParticles;
    const std::size_t Ncells = this->grid.GetPointNo();
    this->lastGroupSamplingDiagnostics_ = GroupSamplingDiagnostics{};

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
        energyToCreateVec[i] = this->factorFleck_[i] * this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(cell.temperature) * this->planckOpacities_[i] * fullDt * units::clight;
        localTotalEnergy += energyToCreateVec[i];
    }

    double globalTotalEnergy = localTotalEnergy;
    std::size_t globalTotalCells = Ncells;
#ifdef STORM_WITH_MPI
    {
        int mpiInit = 0;
        MPI_Initialized(&mpiInit);
        if(mpiInit)
        {
            MPI_Allreduce(MPI_IN_PLACE, &globalTotalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &globalTotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        }
    }
#endif

    std::size_t totalParticles = globalTotalCells * this->parameters_.newPhotonsPerCell * 10;
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
                scoreSum += kv.second;
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
                        * it->second / scoreSum));
                }
                std::size_t const minLearned = static_cast<std::size_t>(std::ceil(
                    static_cast<double>(std::max<std::size_t>(1, this->parameters_.newPhotonsPerCell))
                    * this->adaptiveSourceLearnedMinFactor_));
                learnedPhotons = std::max(learnedPhotons, minLearned);
                photons = std::max(photons, learnedPhotons);
            }
            nPhotonsVec[i] = std::min(photons, std::max<std::size_t>(1, maxPhotons));
        }
    }

    this->lastSourcePhotonsPerCell_ = nPhotonsVec;
    this->lastSourceAllocationSummary_ = SourceAllocationSummary{};
    this->lastSourceAllocationSummary_.adaptiveEnabled =
        this->sourceEmissionControlEnabled_ && this->sourceEmissionUseLearnedScores_ && this->adaptiveSourceScoresEnabled_;
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
                physicalPdf = this->opacity_->GetThermalGroupPdf(cell, this->energyBoundaries_);
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
            MCParticle particle = this->generateSingleParticle(i, cell);
            particle.timeLeft = fullDt * this->randomUnitOpen();

            double weightCorrection = 1.0;
            bool usedGroupFrequencySampling = false;

            if(groupPdfValid)
            {
                double rndGroup = this->randomUnitOpen();
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
                        double rndFreq = this->randomUnitOpen();
                        freqCo = this->opacity_->SampleThermalEnergyInGroup(cell, selectedGroup, rndFreq, this->energyBoundaries_);
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
                    double rnd = this->randomUnitOpen();
                    double freqCo = this->opacity_->GetThermalEnergy(cell, rnd, this->energyBoundaries_);
                    particle.frequency = freqCo / D;
                }
                particle.weight = energyToCreate / (nPhotonsCell * D);
            }
            else if(!usedGroupFrequencySampling)
            {
                if(this->parameters_.withMultigroupOpacity)
                {
                    particle.frequency = this->opacity_->GetThermalEnergy(cell, this->randomUnitOpen(), this->energyBoundaries_);
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
    MCParticle particle;
    particle.id = std::numeric_limits<std::size_t>::max();
    particle.cellIndex = cellIndex;
    particle.cellID = radiation_imc_detail::cellID(cell);
    particle.sourceCellID = particle.cellID;
    particle.frequency = 0.0;
    particle.location = this->positionSampler_(this->grid, cellIndex, this->rng_, this->dist_);
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

    particle.velocity = this->opacity_->getRandomVelocity(cell, this->rng_, this->dist_);

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
            double kT = units::k_boltz * this->cells_[i].temperature;
            cumulativePlanck.resize(Ngroups + 1);
            cumulativePlanck[0] = 0.0;
            for(std::size_t g = 1; g <= Ngroups; ++g)
            {
                double a = this->energyBoundaries_[g - 1] / kT;
                double b = this->energyBoundaries_[g] / kT;
                cumulativePlanck[g] = planck_integral::planck_integral(a, b) + cumulativePlanck[g - 1];
            }
        }

        const double weightPerPhoton = totalErad / static_cast<double>(particlesPerCell);
        for(std::size_t j = 0; j < particlesPerCell; ++j)
        {
            MCParticle particle = this->generateSingleParticle(i, this->cells_[i]);
            particle.weight = weightPerPhoton;
            if(this->parameters_.withMultigroupOpacity && !cumulativePlanck.empty())
            {
                double rnd = this->randomUnitOpen();
                particle.frequency = STORM::LinearInterpolation(cumulativePlanck, this->energyBoundaries_, rnd);
                this->clampFrequencyToBounds(particle.frequency);
            }
            this->setInitialWeightFromWeight(particle);
            result.push_back(particle);
        }
    }
    return result;
}

// ============================================================
// adjustExistingParticles (MMC)
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::adjustExistingParticles(
    std::vector<MCParticle> &particles,
    double fullDt)
{
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

} // namespace STORM

#endif // STORM_RADIATION_IMC_HPP
