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
#include "radiation/RandomWalk.hpp"
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

    struct ComptonCellData {};

    struct DDMCFaceLeak
    {
        std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
        std::size_t nextCellIndex = std::numeric_limits<std::size_t>::max();
        double rate = 0.0;
    };

    struct DDMCCellData
    {
        bool eligible = false;
        double sigmaA = 0.0;
        double sigmaT = 0.0;
        double diffusionCoefficient = 0.0;
        double gamma = 1.0;
        double totalLeakRate = 0.0;
        std::size_t groupCutoff = 0;
        std::vector<DDMCFaceLeak> faceLeaks;
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

    MCParticle generateSingleParticle(std::size_t cellIndex, const CellT &cell);
    std::vector<MCParticle> generateInitialParticles(std::size_t particlesPerCell);
    void adjustExistingParticles(std::vector<MCParticle> &particles, double fullDt);

    const std::vector<double> &getFactorFleck() const { return this->factorFleck_; }
    const std::vector<double> &getPlanckOpacities() const { return this->planckOpacities_; }
    const std::vector<double> &getEradTimeAvg() const { return this->Erad_time_avg_; }
    std::vector<double> &getEradTimeAvg() { return this->Erad_time_avg_; }
    const std::vector<double> &getDebugMaterialEmission() const { return this->debugMaterialEmission_; }
    const std::vector<double> &getDebugMaterialDeposition() const { return this->debugMaterialDeposition_; }
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

    std::size_t getRandomWalkStepCount() const override { return this->rwStepCount_; }
    std::size_t getDDMCStepCount() const override { return this->ddmcStepCount_; }
    std::size_t getDDMCLeakCount() const override { return this->ddmcLeakCount_; }

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
    std::vector<double> debugMaterialEmission_;
    std::vector<double> debugMaterialDeposition_;
    std::vector<std::size_t> lastSourcePhotonsPerCell_;
    SourceAllocationSummary lastSourceAllocationSummary_;
    GroupSamplingDiagnostics lastGroupSamplingDiagnostics_;
    std::vector<ComptonCellData> comptonData_;
    GroupArray comptonGroupCenters_{};
    GroupArray comptonGroupWidths_{};
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_;

    std::unique_ptr<RandomWalk> randomWalk_;
    std::vector<bool> rwCellEligible_;
    std::vector<double> rwCellTotalOpacity_;
    std::vector<PGRWCellData> rwCellData_;
    std::size_t rwStepCount_ = 0;

    std::vector<DDMCCellData> ddmcCellData_;
    std::size_t ddmcStepCount_ = 0;
    std::size_t ddmcLeakCount_ = 0;
    std::size_t ddmcCensusCount_ = 0;
    std::size_t ddmcUpscatterCount_ = 0;
    std::size_t ddmcFallbackCount_ = 0;

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
        this->validateEnergyBoundaries();
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
        if(this->energyBoundaries_[g + 1] <= this->energyBoundaries_[g])
        {
            StormError eo("RadiationIMC energy boundaries are not strictly increasing");
            eo.addEntry("Group", g);
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
    if(this->parameters_.withCompton) this->rejectUnsupportedParameter("withCompton");
    if(this->parameters_.postProcess.enabled) this->rejectUnsupportedParameter("postProcess.enabled");
    if(this->parameters_.postProcess.polarization.enabled) this->rejectUnsupportedParameter("postProcess.polarization.enabled");
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
                double a = this->energyBoundaries_[g] / kT;
                double b = this->energyBoundaries_[g + 1] / kT;
                double Bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 1.0 / static_cast<double>(NumGroups);
                double sigA_g = this->opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double sigT_g = sigA_g + scatOp;

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
        const double materialDeposit = -rwExp * particle.weight;
        this->extensives_[cellIndex].internal_energy += materialDeposit;
        this->debugMaterialDeposition_[cellIndex] += materialDeposit;
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
            this->debugMaterialDeposition_[cellIndex] += particle.weight;
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

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData_[i];
        const CellT &cell = this->cells_[i];
        double scatOp = this->opacity_->CalcScatteringOpacity(cell);
        double volume = this->grid.GetVolume(i);
        double surfaceArea = this->computeCellSurfaceArea(i);
        if(volume <= 0.0 || surfaceArea <= 0.0)
        {
            continue;
        }
        double meanChordLength = 4.0 * volume / surfaceArea;

        if(this->parameters_.ddmcUseMultigroupPGRW && this->parameters_.withMultigroupOpacity)
        {
            GroupArray energyCenters = this->opacity_->getEnergyCenters(this->energyBoundaries_);
            double kT = units::k_boltz * cell.temperature;
            if(kT <= 0.0)
            {
                continue;
            }
            double totalBgDiff = 0.0, sumBgSigADiff = 0.0, sumBgSigTDiff = 0.0, sumBgOverSigTDiff = 0.0;
            std::size_t cutoff = 0;
            bool foundNonDiffusive = false;
            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                double sigA_g = this->opacity_->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double sigT_g = sigA_g + scatOp;
                double Bg = 1.0 / static_cast<double>(NumGroups);
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
                data.diffusionCoefficient = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                data.gamma = 1.0;
                data.eligible = (data.sigmaT > 0.0 && data.diffusionCoefficient > 0.0);
            }
        }
        else
        {
            data.sigmaA = this->planckOpacities_[i];
            data.sigmaT = data.sigmaA + scatOp;
            data.diffusionCoefficient = (data.sigmaT > 0.0) ? units::clight / (3.0 * data.sigmaT) : 0.0;
            data.gamma = 1.0;
            data.eligible = (data.sigmaT * meanChordLength >= this->parameters_.ddmcMinCellOpticalDepth
                             && data.diffusionCoefficient > 0.0);
        }
    }

    PointT cellCenter;
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData_[i];
        if(!data.eligible)
        {
            continue;
        }
        double volume = this->grid.GetVolume(i);
        cellCenter = this->grid.GetMeshPoint(i);
        for(std::size_t faceIdx : this->grid.GetCellFaces(i))
        {
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            std::size_t nextCellIndex = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(this->grid.IsPointOutsideBox(nextCellIndex))
            {
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
            double faceDistance = std::abs(ScalarProd(faceCenter - cellCenter, normal));
            if(faceDistance <= 0.0)
            {
                faceDistance = 0.5 * std::abs(ScalarProd(this->grid.GetMeshPoint(nextCellIndex) - cellCenter, normal));
            }
            if(faceDistance <= 0.0)
            {
                continue;
            }
            double diffusionFace = data.diffusionCoefficient;
            if(nextCellIndex < Ncells && this->ddmcCellData_[nextCellIndex].diffusionCoefficient > 0.0)
            {
                double D1 = data.diffusionCoefficient;
                double D2 = this->ddmcCellData_[nextCellIndex].diffusionCoefficient;
                diffusionFace = (D1 > 0.0 && D2 > 0.0) ? (2.0 * D1 * D2 / (D1 + D2)) : std::max(D1, D2);
            }
            double rate = diffusionFace * this->grid.GetArea(faceIdx) / (volume * faceDistance);
            if(rate > 0.0 && std::isfinite(rate))
            {
                DDMCFaceLeak faceLeak;
                faceLeak.faceIndex = faceIdx;
                faceLeak.nextCellIndex = nextCellIndex;
                faceLeak.rate = rate;
                data.faceLeaks.push_back(faceLeak);
                data.totalLeakRate += rate;
            }
        }
        if(data.totalLeakRate <= 0.0)
        {
            data.eligible = false;
            data.faceLeaks.clear();
        }
    }
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::tryDDMCStep(
    MCParticle &particle, Functionality &functionality)
{
    std::size_t cellIndex = particle.cellIndex;
    if(cellIndex >= this->ddmcCellData_.size())
    {
        return false;
    }
    const DDMCCellData &data = this->ddmcCellData_[cellIndex];
    if(!data.eligible || data.totalLeakRate <= 0.0 || data.faceLeaks.empty())
    {
        return false;
    }

    double Ro = this->computeMinDistanceToFaces(cellIndex, particle.location);
    if(Ro * data.sigmaT < this->parameters_.ddmcMinParticleOpticalDepth)
    {
        ++this->ddmcFallbackCount_;
        return false;
    }

    if(this->parameters_.ddmcUseMultigroupPGRW && this->parameters_.withMultigroupOpacity)
    {
        if(data.groupCutoff == 0 || data.groupCutoff > NumGroups)
        {
            ++this->ddmcFallbackCount_;
            return false;
        }
        double coFreq = particle.frequency;
        this->clampFrequencyToBounds(coFreq);
        if(coFreq >= this->energyBoundaries_[data.groupCutoff])
        {
            ++this->ddmcFallbackCount_;
            return false;
        }
    }

    double f = this->factorFleck_[cellIndex];
    double upscatterRate = 0.0;
    if(this->parameters_.ddmcUseMultigroupPGRW && data.gamma < 1.0 && data.sigmaA > 0.0 && f > 0.0)
    {
        upscatterRate = units::clight * (1.0 - f) * data.sigmaA * (1.0 - data.gamma);
    }
    double eventRate = data.totalLeakRate + upscatterRate;
    if(eventRate <= 0.0)
    {
        ++this->ddmcFallbackCount_;
        return false;
    }

    double tEvent = -std::log(this->randomUnitOpen()) / eventRate;
    double tCensus = particle.timeLeft;
    double dt = std::min(tEvent, tCensus);
    bool censusEvent = (tCensus <= tEvent);

    double absRate = data.sigmaA * f * units::clight;
    double oldWeight = particle.weight;
    double expFactor = std::expm1(-dt * absRate);

    if(!this->parameters_.noHydroFeedback)
    {
        const double materialDeposit = -expFactor * oldWeight;
        this->extensives_[cellIndex].internal_energy += materialDeposit;
        this->debugMaterialDeposition_[cellIndex] += materialDeposit;
    }

    double integratedForTally = (absRate > 0.0)
        ? oldWeight * expFactor * (-1.0 / absRate)
        : oldWeight * dt;
    this->Erad_time_avg_[cellIndex] += integratedForTally;

    if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
    {
        std::size_t g = this->opacity_->findGroup(particle.frequency, this->energyBoundaries_);
        if(g < NumGroups)
        {
            this->Eg_time_avg_[cellIndex][g] += integratedForTally;
        }
    }

    particle.weight *= 1.0 + expFactor;
    particle.timeLeft -= dt;

    if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
    {
        functionality.change = ParticleStatus::REMOVE;
        if(!this->parameters_.noHydroFeedback)
        {
            this->extensives_[cellIndex].internal_energy += particle.weight;
            this->debugMaterialDeposition_[cellIndex] += particle.weight;
        }
        ++this->ddmcStepCount_;
        return true;
    }

    ++this->ddmcStepCount_;

    if(censusEvent)
    {
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
            ++this->ddmcFallbackCount_;
            functionality.change = ParticleStatus::DONE;
            return true;
        }

        PointT leakFaceCenter = this->grid.FaceCM(chosen->faceIndex);
        PointT nOut = this->grid.Normal(chosen->faceIndex);
        double nOutMag = fastabs(nOut);
        if(nOutMag <= 0.0)
        {
            ++this->ddmcFallbackCount_;
            functionality.change = ParticleStatus::DONE;
            return true;
        }
        nOut = nOut / nOutMag;

        PointT towardNeighbor = this->grid.GetMeshPoint(chosen->nextCellIndex) - this->grid.GetMeshPoint(cellIndex);
        if(ScalarProd(nOut, towardNeighbor) < 0.0)
        {
            nOut = nOut * (-1.0);
        }

        constexpr double DDMC_PI = 3.14159265358979323846;
        double xiMu = std::min(1.0, this->randomUnitOpen());
        double mu = std::sqrt(xiMu);
        double phiLeak = 2.0 * DDMC_PI * this->randomUnitOpen();
        double sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

        PointT helper = (std::abs(nOut.x) < 0.9)
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

        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = chosen->nextCellIndex;
        ++this->ddmcLeakCount_;
    }
    else
    {
        if(!this->parameters_.ddmcUseMultigroupPGRW)
        {
            functionality.change = ParticleStatus::DONE;
            ++this->ddmcCensusCount_;
            return true;
        }
        double lo = static_cast<double>(data.groupCutoff) / static_cast<double>(NumGroups);
        double xi = this->randomUnitOpen();
        CellT &cell = this->cells_[cellIndex];
        particle.frequency = this->opacity_->GetThermalEnergy(cell, lo + xi * (1.0 - lo), this->energyBoundaries_);
        this->clampFrequencyToBounds(particle.frequency);
        particle.velocity = this->opacity_->getRandomVelocity(cell, this->rng_, this->dist_);
        ++this->ddmcUpscatterCount_;
    }
    return true;
}

// ============================================================
// preStep
// ============================================================

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT, typename EOST, std::size_t NumGroups, typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, TraitsT, PositionSamplerT>::preStep(double fullDt)
{
    const std::size_t Ncells = this->grid.GetPointNo();
    this->planckOpacities_.assign(Ncells, 0.0);
    this->factorFleck_.assign(Ncells, 1.0);
    this->Erad_time_avg_.assign(Ncells, 0.0);
    this->debugMaterialEmission_.assign(Ncells, 0.0);
    this->debugMaterialDeposition_.assign(Ncells, 0.0);
    if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
    {
        GroupArray zeros{};
        this->Eg_time_avg_.assign(Ncells, zeros);
    }

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        CellT &cell = this->cells_[i];

        double gamma = 1.0;
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
            }
        }

        this->planckOpacities_[i] = this->opacity_->CalcPlanckOpacity(cell);
        const auto &tracers = this->traits_.tracers(cell);
        const auto &tracerNames = this->traits_.tracerNames(cell);
        double cv = this->eos_->dT2cv(this->density(i), cell.temperature, tracers, tracerNames);
        this->factorFleck_[i] = 1.0 / (1.0 + (4.0 * units::arad * boost::math::pow<3>(cell.temperature) * this->planckOpacities_[i] * units::clight * fullDt * gamma) / cv);
        if(this->factorFleck_[i] < 0.0 || this->factorFleck_[i] > 1.0)
        {
            StormError eo("Invalid factor fleck in RadiationIMC::preStep");
            eo.addEntry("Factor fleck", this->factorFleck_[i]);
            eo.addEntry("Planck opacity", this->planckOpacities_[i]);
            eo.addEntry("Temperature", cell.temperature);
            eo.addEntry("Density", this->density(i));
            eo.addEntry("Gamma", gamma);
            eo.addEntry("cv", cv);
            eo.addEntry("Full dt", fullDt);
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
        this->precomputeDDMCData();
        this->ddmcStepCount_ = 0;
        this->ddmcLeakCount_ = 0;
        this->ddmcCensusCount_ = 0;
        this->ddmcUpscatterCount_ = 0;
        this->ddmcFallbackCount_ = 0;
    }

    std::vector<MCParticle> newParticles = this->generateParticles(fullDt);
    if(this->boundary)
    {
        std::vector<MCParticle> boundaryParticles = this->boundary->generateNewBoundaryParticles(fullDt);
        for(MCParticle &particle : boundaryParticles)
        {
            this->setInitialWeightFromWeight(particle);
        }
        newParticles.insert(newParticles.end(), boundaryParticles.begin(), boundaryParticles.end());
    }
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

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);

    double dopplerShift = 1.0;
    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
    {
        if(this->parameters_.withHydro && !this->parameters_.MMC)
        {
            dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
        }
    }

    if(this->parameters_.withRandomWalk && this->randomWalk_ && this->rwCellEligible_[cellIndex])
    {
        if(this->tryRandomWalkStep(particle, functionality))
        {
            ++this->rwStepCount_;
            return functionality;
        }
    }

    if(this->parameters_.withDDMC && cellIndex < this->ddmcCellData_.size() && this->ddmcCellData_[cellIndex].eligible)
    {
        if(this->tryDDMCStep(particle, functionality))
        {
            return functionality;
        }
    }

    double absorptionOpacity;
    std::size_t group = std::numeric_limits<std::size_t>::max();
    if(this->parameters_.withMultigroupOpacity)
    {
        double shiftedFrequency = particle.frequency * dopplerShift;
        this->clampFrequencyToBounds(shiftedFrequency);
        group = this->opacity_->findGroup(shiftedFrequency, this->energyBoundaries_);
        absorptionOpacity = this->opacity_->CalcAbsorptionOpacity(cell, shiftedFrequency);
    }
    else
    {
        absorptionOpacity = this->planckOpacities_[cellIndex];
    }
    double elasticScatteringOpacity = this->opacity_->CalcScatteringOpacity(cell);
    double effectiveAbsorptionOpacity = (1.0 - this->factorFleck_[cellIndex]) * absorptionOpacity;
    double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity;
    double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
    double _log1p = -std::log1p(this->randomUnitOpen() - 1.0);
    double scatteringDistance = scatteringLength * _log1p / dopplerShift;
    double timeScattering = std::isfinite(scatteringDistance) ? scatteringDistance / fastabs(particle.velocity) : std::numeric_limits<double>::infinity();

    double timeLeft = particle.timeLeft;
    enum Events
    {
        INTERSECTION = 0,
        SCATTERING = 1,
        TIMELEFT = 2
    };
    std::array<std::pair<std::size_t, double>, 3> times;
    times[INTERSECTION] = {INTERSECTION, timeIntersect};
    times[SCATTERING] = {SCATTERING, timeScattering};
    times[TIMELEFT] = {TIMELEFT, timeLeft};

    std::pair<std::size_t, double> min = *std::min_element(times.begin(), times.end(),
        [](const std::pair<std::size_t, double> &a, const std::pair<std::size_t, double> &b) { return a.second < b.second; });
    double dt = min.second;

    particle.timeLeft -= dt;
    double weightEvolutionOpacity = absorptionOpacity * this->factorFleck_[cellIndex];
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
    if(!this->parameters_.noHydroFeedback)
    {
        double const materialDeposit = -expFactor2 * particle.weight;
        this->extensives_[cellIndex].internal_energy += materialDeposit;
        this->debugMaterialDeposition_[cellIndex] += materialDeposit;
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
    particle.weight *= 1.0 + expFactor1;

    if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
    {
        functionality.change = ParticleStatus::REMOVE;
        if(!this->parameters_.noHydroFeedback)
        {
            this->extensives_[cellIndex].internal_energy += particle.weight;
            this->debugMaterialDeposition_[cellIndex] += particle.weight;
        }
        return functionality;
    }

    if(min.first == INTERSECTION)
    {
        functionality.change = ParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
    }
    else if(min.first == SCATTERING)
    {
        PointT oldVelocity = particle.velocity;
        double oldWeight = particle.weight;
        double D_lab_to_co = dopplerShift;
        double eventRandom = this->randomUnitOpen() * eventOpacity;
        bool isEffectiveScatter = false;
        if(eventRandom < elasticScatteringOpacity)
        {
            particle.velocity = this->opacity_->getNewScatterVelocity(cell, particle.velocity, particle.frequency, this->rng_, this->dist_);
        }
        else
        {
            particle.velocity = this->opacity_->getNewScatterVelocity(cell, particle.velocity, particle.frequency, this->rng_, this->dist_);
            isEffectiveScatter = true;
        }
        if(this->parameters_.withMultigroupOpacity)
        {
            particle.frequency *= dopplerShift;
            this->clampFrequencyToBounds(particle.frequency);
            if(isEffectiveScatter)
            {
                double reemitRandom = this->randomUnitOpen();
                particle.frequency = this->opacity_->GetThermalEnergy(cell, reemitRandom, this->energyBoundaries_);
            }
        }
        if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
        {
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                double weightBefore = particle.weight;
                particle.weight *= D_lab_to_co;
                radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
                if(this->parameters_.withMultigroupOpacity)
                {
                    this->clampFrequencyToBounds(particle.frequency);
                }
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if(!this->parameters_.diffusionPressureGradient && !this->parameters_.noHydroFeedback)
                    {
                        this->extensives_[cellIndex].momentum += (weightBefore * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
                    }
                }
            }
        }
        functionality.change = ParticleStatus::NO_CELL_MOVE;
    }
    else if(min.first == TIMELEFT)
    {
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
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        this->Erad_time_avg_[i] /= (fullDt * this->grid.GetVolume(i));
        if(this->parameters_.withEgTimeAvg && this->parameters_.withMultigroupOpacity)
        {
            double norm = fullDt * this->grid.GetVolume(i);
            for(std::size_t g = 0; g < NumGroups; ++g)
            {
                this->Eg_time_avg_[i][g] /= norm;
            }
        }
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
                        STORM::MPI_exchange_data(this->grid, this->Erad_time_avg_, true);
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
            if(this->parameters_.withHydro && !this->parameters_.MMC)
            {
                gamma = 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2);
            }
        }
        gammaVec[i] = gamma;
        energyToCreateVec[i] = this->factorFleck_[i] * this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(cell.temperature) * this->planckOpacities_[i] * fullDt * units::clight;
        if(std::getenv("STORM_DISABLE_MATERIAL_EMISSION")
           || (std::getenv("STORM_SUPPRESS_COLD_EMISSION") && cell.temperature <= 2.0e4))
        {
            energyToCreateVec[i] = 0.0;
        }
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
            this->debugMaterialEmission_[i] += energyToCreate;
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

                if(usedGroupFrequencySampling && this->parameters_.withHydro && !this->parameters_.MMC)
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

            if(!usedGroupFrequencySampling && this->parameters_.withHydro && !this->parameters_.MMC)
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

    if(this->parameters_.withHydro && !this->parameters_.MMC)
    {
        radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
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
