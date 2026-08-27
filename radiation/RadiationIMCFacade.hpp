#ifndef STORM_RADIATION_IMC_FACADE_HPP
#define STORM_RADIATION_IMC_FACADE_HPP

namespace STORM {

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::RadiationIMC(
    const GridT &grid,
    const std::shared_ptr<BoundaryCond> &boundary,
    std::vector<CellT> &cells,
    std::vector<ExtensivesT> &extensives,
    std::shared_ptr<EOST> eos,
    std::shared_ptr<OpacityModel> opacity,
    Parameters parameters,
    TraitsT traits,
    PositionSamplerT positionSampler,
    std::uint64_t seed) :
    Base(grid, boundary),
    State(std::move(parameters), std::move(traits), std::move(positionSampler), seed),
    cells_(cells),
    extensives_(extensives),
    eos_(std::move(eos)),
    opacity_(std::move(opacity))
{
    lifecycleProcess_ = std::make_unique<
        radiation_imc_detail::IMCLifecycleProcess<RadiationIMC>>(*this);
    transportProcess_ = std::make_unique<
        radiation_imc_detail::IMCTransportProcess<RadiationIMC>>(*this);
    sourceProcess_ = std::make_unique<
        radiation_imc_detail::IMCSourceProcess<RadiationIMC>>(*this);
    randomWalkProcess_ = std::make_unique<
        radiation_imc_detail::IMCRandomWalkProcess<RadiationIMC>>(*this);
    ddmcEngine_ = std::make_unique<
        radiation_imc_detail::DDMCEngine<RadiationIMC>>(*this);
    comptonProcess_ = std::make_unique<
        radiation_imc_detail::ComptonProcess<RadiationIMC>>(*this);
    observerProcess_ = std::make_unique<
        radiation_imc_detail::IMCObserverProcess<RadiationIMC>>(*this);
    deviceExecutor_ = std::make_unique<
        radiation_imc_detail::IMCDeviceExecutor<RadiationIMC>>(*this);

    if(this->parameters_.newPhotonsPerCell == 0)
    {
        throw StormError("RadiationIMC requires newPhotonsPerCell > 0");
    }
    if(!this->eos_)
    {
        throw StormError("RadiationIMC requires a non-null EOS");
    }
    if(!this->opacity_)
    {
        throw StormError("RadiationIMC requires a non-null opacity model");
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    this->rng_.seed(seed + static_cast<std::uint64_t>(rank) * 104729ULL);
    this->particleRngSeed_ = seed + static_cast<std::uint64_t>(rank) * 104729ULL;
#else
    this->particleRngSeed_ = seed;
#endif
    this->opacity_->reseed(seed + 1ULL);

    const std::size_t Ncells = this->grid.GetPointNo();
    this->resetTransportTallies(Ncells);
    this->Erad_time_avg_.assign(Ncells, 0.0);
}

#ifdef STORM_WITH_GPU
template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                   TraitsT, PositionSamplerT>::UsesDeviceTransport() const
{
    return this->deviceExecutor_->UsesDeviceTransport();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
gpu::GreyIMCViews<gpu::DeviceVec3>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::GetDeviceTransportViews() const
{
    return this->deviceExecutor_->GetDeviceTransportViews();
}
#endif

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                   TraitsT, PositionSamplerT>::GreyKernelEligible() const
{
    return this->deviceExecutor_->GreyKernelEligible();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                   TraitsT, PositionSamplerT>::SharedRandomWalkKernelEligible() const
{
    return this->deviceExecutor_->SharedRandomWalkKernelEligible();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
gpu::GreyIMCViews<PointT>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::GetHostTransportViews()
{
    return this->deviceExecutor_->GetHostTransportViews();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                   TraitsT, PositionSamplerT>::SharedFullIMCKernelEligible() const
{
    return this->deviceExecutor_->SharedFullIMCKernelEligible();
}

// Public MonteCarloPhysics façade.  The implementation lives in the process
// objects; these methods preserve the source-compatible RadiationIMC API.
template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST,
                                  NumGroups, OpacityT, TraitsT,
                                  PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::preStep(double fullDt)
{
    return this->lifecycleProcess_->preStep(fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::Functionality
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::step(
    MCParticle &particle, std::vector<MCParticle> &particlesToAdd)
{
    return this->transportProcess_->step(particle, particlesToAdd);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::Functionality
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::stepImpl(
    MCParticle &particle, std::vector<MCParticle> &particlesToAdd)
{
    return this->transportProcess_->stepImpl(particle, particlesToAdd);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::postStep(
    const std::vector<MCParticle> &particles, double fullDt)
{
    this->transportProcess_->postStep(particles, fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::onBoundaryResult(
    const MCParticle &particle, ParticleStatus status, bool escaped)
{
    this->observerProcess_->onBoundaryResult(particle, status, escaped);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST,
                                  NumGroups, OpacityT, TraitsT,
                                  PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generateParticles(double fullDt)
{
    return this->sourceProcess_->generateParticles(fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generateSingleParticle(
    std::size_t cellIndex, const CellT &cell)
{
    return this->sourceProcess_->generateSingleParticle(cellIndex, cell);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generateSingleParticle(
    std::size_t cellIndex, const CellT &cell,
    const PositionDecomposition *decomposition)
{
    return this->sourceProcess_->generateSingleParticle(
        cellIndex, cell, decomposition);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST,
                                  NumGroups, OpacityT, TraitsT,
                                  PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generateInitialParticles(
    std::size_t particlesPerCell)
{
    return this->sourceProcess_->generateInitialParticles(particlesPerCell);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::adjustExistingParticles(
    std::vector<MCParticle> &particles, double fullDt)
{
    this->sourceProcess_->adjustExistingParticles(particles, fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::splitComptonRiskyParticles(
    std::vector<MCParticle> &particles, double fullDt)
{
    this->sourceProcess_->splitComptonRiskyParticles(particles, fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setPostProcessExternalSources(
    std::vector<PostProcessExternalSource> sources)
{
    this->observerProcess_->setPostProcessExternalSources(std::move(sources));
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::clearPostProcessExternalSources()
{
    this->observerProcess_->clearPostProcessExternalSources();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setNewPhotonsPerCell(std::size_t n)
{
    this->lifecycleProcess_->setNewPhotonsPerCell(n);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setAdaptiveSourceCellScores(
    std::unordered_map<std::size_t, double> scores, double strength,
    double maxFactor, double learnedReserveFrac, double learnedMinFactor,
    double observerBudgetMultiplier, std::size_t learnedMinPhotons,
    std::size_t learnedMaxPhotons, double scorePower)
{
    this->lifecycleProcess_->setAdaptiveSourceCellScores(
        std::move(scores), strength, maxFactor, learnedReserveFrac,
        learnedMinFactor, observerBudgetMultiplier, learnedMinPhotons,
        learnedMaxPhotons, scorePower);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::clearAdaptiveSourceCellScores()
{
    this->lifecycleProcess_->clearAdaptiveSourceCellScores();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setAdaptiveSourceCellGroupScores(
    std::unordered_map<std::size_t, GroupArray> scores, double strength,
    double pdfFloor, double maxBias, double maxWeightCorrection)
{
    this->lifecycleProcess_->setAdaptiveSourceCellGroupScores(
        std::move(scores), strength, pdfFloor, maxBias, maxWeightCorrection);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::clearAdaptiveSourceCellGroupScores()
{
    this->lifecycleProcess_->clearAdaptiveSourceCellGroupScores();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setSourceEmissionControl(
    bool useLearnedScores, bool includeUniformBase, std::size_t baseMultiplier,
    std::size_t learnedBoostFactor, std::size_t learnedExtraBudget)
{
    this->lifecycleProcess_->setSourceEmissionControl(
        useLearnedScores, includeUniformBase, baseMultiplier,
        learnedBoostFactor, learnedExtraBudget);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::clearSourceEmissionControl()
{
    this->lifecycleProcess_->clearSourceEmissionControl();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::
getAccelerationDebugInfo(std::size_t cellIndex, double frequency) const
{
    return this->transportProcess_->getAccelerationDebugInfo(cellIndex, frequency);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::validateGridSizedState() const
{
    this->lifecycleProcess_->validateGridSizedState();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::validateEnergyBoundaries() const
{
    this->lifecycleProcess_->validateEnergyBoundaries();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::rejectUnsupportedParameter(
    const std::string &name) const
{
    this->lifecycleProcess_->rejectUnsupportedParameter(name);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::rejectUnsupportedParameters() const
{
    this->lifecycleProcess_->rejectUnsupportedParameters();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::randomUnitOpen()
{
    return this->lifecycleProcess_->randomUnitOpen();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::initializeParticleRNG(
    MCParticle &particle)
{
    this->lifecycleProcess_->initializeParticleRNG(particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::randomUnitOpen(
    MCParticle &particle)
{
    return this->lifecycleProcess_->randomUnitOpen(particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                     TraitsT, PositionSamplerT>::sampleRandomVelocity(
    const CellT &cell, MCParticle &particle)
{
    return this->lifecycleProcess_->sampleRandomVelocity(cell, particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                     TraitsT, PositionSamplerT>::sampleScatterVelocity(
    const CellT &cell, MCParticle &particle)
{
    return this->lifecycleProcess_->sampleScatterVelocity(cell, particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::resetTransportTallies(
    std::size_t cellCount)
{
    this->lifecycleProcess_->resetTransportTallies(cellCount);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tallyMaterialEnergy(
    std::size_t cellIndex, double energy, bool addToTotalEnergy)
{
    this->lifecycleProcess_->tallyMaterialEnergy(
        cellIndex, energy, addToTotalEnergy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tallyMomentum(
    std::size_t cellIndex, const PointT &momentum)
{
    this->lifecycleProcess_->tallyMomentum(cellIndex, momentum);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tallyRadiationEnergy(
    std::size_t cellIndex, double integratedEnergy)
{
    this->lifecycleProcess_->tallyRadiationEnergy(cellIndex, integratedEnergy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tallyGroupRadiationEnergy(
    std::size_t cellIndex, std::size_t group, double integratedEnergy)
{
    this->lifecycleProcess_->tallyGroupRadiationEnergy(
        cellIndex, group, integratedEnergy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::applyTransportTallies()
{
    this->lifecycleProcess_->applyTransportTallies();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setInitialWeightFromWeight(
    MCParticle &particle) const
{
    this->lifecycleProcess_->setInitialWeightFromWeight(particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::density(
    std::size_t cellIndex) const
{
    return this->lifecycleProcess_->density(cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::specificInternalEnergy(
    std::size_t cellIndex) const
{
    return this->lifecycleProcess_->specificInternalEnergy(cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::totalRadiationEnergy(
    std::size_t cellIndex) const
{
    return this->lifecycleProcess_->totalRadiationEnergy(cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::throwIfNegativeInternalEnergy(
    std::size_t cellIndex, const std::string &where)
{
    this->lifecycleProcess_->throwIfNegativeInternalEnergy(cellIndex, where);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::depositMaterialEnergy(
    std::size_t cellIndex, double energy)
{
    this->lifecycleProcess_->depositMaterialEnergy(cellIndex, energy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::synchronizeMaterialCell(
    std::size_t cellIndex)
{
    this->lifecycleProcess_->synchronizeMaterialCell(cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::clampFrequencyToBounds(
    double &frequency) const
{
    this->lifecycleProcess_->clampFrequencyToBounds(frequency);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::computeMinDistanceToFaces(
    std::size_t cellIndex, const PointT &location) const
{
    return this->randomWalkProcess_->computeMinDistanceToFaces(
        cellIndex, location);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::computeCellSurfaceArea(
    std::size_t cellIndex) const
{
    return this->randomWalkProcess_->computeCellSurfaceArea(cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::precomputeRandomWalkData()
{
    this->randomWalkProcess_->precomputeRandomWalkData();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tryRandomWalkStep(
    MCParticle &particle, Functionality &functionality)
{
    return this->randomWalkProcess_->tryRandomWalkStep(particle, functionality);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::precomputeDDMCData()
{
    this->ddmcEngine_->precomputeDDMCData();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tryDDMCStep(
    MCParticle &particle, Functionality &functionality)
{
    return this->ddmcEngine_->tryDDMCStep(particle, functionality);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::addDDMCFluxContribution(
    std::size_t cellIndex, const PointT &contribution)
{
    this->ddmcEngine_->addDDMCFluxContribution(cellIndex, contribution);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::applyDDMCMomentumFeedback(
    double fullDt)
{
    this->ddmcEngine_->applyDDMCMomentumFeedback(fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::tryIMCToDDMCInterface(
    MCParticle &particle, Functionality &functionality,
    std::vector<MCParticle> &particlesToAdd, std::size_t sourceCellIndex,
    std::size_t targetCellIndex, std::size_t faceIndex)
{
    return this->ddmcEngine_->tryIMCToDDMCInterface(
        particle, functionality, particlesToAdd, sourceCellIndex,
        targetCellIndex, faceIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::recordDDMCDiagnosticEvent(
    DDMCDiagnosticEventKind kind, std::size_t sourceCellIndex,
    std::size_t targetCellIndex, std::size_t faceIndex, std::size_t group,
    double energy, std::size_t sourceGroupCutoff,
    std::size_t targetGroupCutoff, double mu, double admissionProbability)
{
    this->ddmcEngine_->recordDDMCDiagnosticEvent(
        kind, sourceCellIndex, targetCellIndex, faceIndex, group, energy,
        sourceGroupCutoff, targetGroupCutoff, mu, admissionProbability);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::
getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const
{
    return this->ddmcEngine_->getDDMCFaceDiagnosticsTSV(xMin, xMax);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::string RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::
getDDMCInterfaceEventDiagnosticsTSV(double xMin, double xMax) const
{
    return this->ddmcEngine_->getDDMCInterfaceEventDiagnosticsTSV(xMin, xMax);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::precomputeComptonData(double dt)
{
    this->comptonProcess_->precomputeComptonData(dt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::initializeComptonMatrixGenerator()
{
    this->comptonProcess_->initializeComptonMatrixGenerator();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::vector<double> RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST,
                                 NumGroups, OpacityT, TraitsT,
                                 PositionSamplerT>::buildComptonTemperatures() const
{
    return this->comptonProcess_->buildComptonTemperatures();
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::GroupCdf
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::buildSafeComptonCdf(
    const GroupArray &weights) const
{
    return this->comptonProcess_->buildSafeComptonCdf(weights);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::buildComptonMatricesForCell(
    const CellT &cell, std::size_t cellIndex,
    ComptonOccupationMode occupationMode, ComptonCellData &data)
{
    this->comptonProcess_->buildComptonMatricesForCell(
        cell, cellIndex, occupationMode, data);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::computeLteTemperature(
    const CellT &cell, std::size_t cellIndex) const
{
    return this->comptonProcess_->computeLteTemperature(cell, cellIndex);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::recomputeComptonContractions(
    ComptonCellData &data) const
{
    this->comptonProcess_->recomputeComptonContractions(data);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::buildComptonSources(
    double sourceDt, ComptonCellData &data) const
{
    this->comptonProcess_->buildComptonSources(sourceDt, data);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::buildComptonEventData(
    ComptonCellData &data) const
{
    this->comptonProcess_->buildComptonEventData(data);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::computeComptonRiskForCell(
    double fullDt, ComptonCellData &data) const
{
    this->comptonProcess_->computeComptonRiskForCell(fullDt, data);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::vector<typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST,
                                  NumGroups, OpacityT, TraitsT,
                                  PositionSamplerT>::MCParticle>
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generateComptonParticles(double fullDt)
{
    return this->comptonProcess_->generateComptonParticles(fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::size_t RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::sampleComptonTarget(
    const ComptonCellData &data, std::size_t sourceGroup,
    MCParticle &particle)
{
    return this->comptonProcess_->sampleComptonTarget(
        data, sourceGroup, particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::addComptonMaterialExchange(
    std::size_t cellIndex, double energy)
{
    this->comptonProcess_->addComptonMaterialExchange(cellIndex, energy);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
std::size_t RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::sampleComptonCdf(
    const GroupCdf &cdf, double random) const
{
    return this->comptonProcess_->sampleComptonCdf(cdf, random);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::frequencyForComptonGroup(
    std::size_t group) const
{
    return this->comptonProcess_->frequencyForComptonGroup(group);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::sumComptonGroups(
    const GroupArray &values)
{
    return this->comptonProcess_->sumComptonGroups(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::compensatedSumComptonGroups(
    const GroupArray &values)
{
    return this->comptonProcess_->compensatedSumComptonGroups(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
const char *RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                         OpacityT, TraitsT, PositionSamplerT>::
comptonCorrectionFailureName(ComptonCorrectionFailure failure)
{
    return this->comptonProcess_->comptonCorrectionFailureName(failure);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::minComptonGroup(
    const GroupArray &values)
{
    return this->comptonProcess_->minComptonGroup(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::maxAbsComptonGroup(
    const GroupArray &values)
{
    return this->comptonProcess_->maxAbsComptonGroup(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::normComptonGroups(
    const GroupArray &values)
{
    return this->comptonProcess_->normComptonGroups(values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::solveComptonGroupSystem(
    GroupMatrix matrix, GroupArray rhs, GroupArray &solution)
{
    return this->comptonProcess_->solveComptonGroupSystem(
        matrix, rhs, solution);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::GroupArray
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::multiplyComptonMatrix(
    const GroupMatrix &matrix, const GroupArray &values)
{
    return this->comptonProcess_->multiplyComptonMatrix(matrix, values);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::relativeComptonResidual(
    const GroupMatrix &matrix, const GroupArray &solution,
    const GroupArray &rhs, double scale)
{
    return this->comptonProcess_->relativeComptonResidual(
        matrix, solution, rhs, scale);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::ComptonProjectionResult
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::projectNonnegativeConservative(
    const GroupArray &candidate, double targetTotal, double energyScale,
    double perGroupNegativeTolerance, double totalNegativeTolerance)
{
    return this->comptonProcess_->projectNonnegativeConservative(
        candidate, targetTotal, energyScale, perGroupNegativeTolerance,
        totalNegativeTolerance);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::ComptonCorrectionResult
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::solveComptonCorrection(
    std::size_t cellIndex, double fullDt, const ComptonCellData &data,
    const GroupArray &rawGroupEnergy, const GroupArray &timeAvgGroupEnergy,
    double budgetBefore, double materialFloor, double preStepRadiation) const
{
    return this->comptonProcess_->solveComptonCorrection(
        cellIndex, fullDt, data, rawGroupEnergy, timeAvgGroupEnergy,
        budgetBefore, materialFloor, preStepRadiation);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::setPacketFromComovingState(
    MCParticle &particle, const CellT &cell, double comovingFrequency,
    double comovingWeight) const
{
    this->comptonProcess_->setPacketFromComovingState(
        particle, cell, comovingFrequency, comovingWeight);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::applyComptonScatterEvent(
    std::size_t cellIndex, CellT &cell, std::size_t sourceGroup,
    MCParticle &particle, const PointT &oldVelocity, double oldWeight)
{
    return this->comptonProcess_->applyComptonScatterEvent(
        cellIndex, cell, sourceGroup, particle, oldVelocity, oldWeight);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::applyComptonEndOfStepCorrection(
    double fullDt)
{
    this->comptonProcess_->applyComptonEndOfStepCorrection(fullDt);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::reconcileComptonParticles(
    std::vector<MCParticle> &particles)
{
    this->comptonProcess_->reconcileComptonParticles(particles);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
void RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::recordObserverCrossing(
    const MCParticle &particle, const PointT &crossingPoint)
{
    this->observerProcess_->recordObserverCrossing(particle, crossingPoint);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
PointT RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                     TraitsT, PositionSamplerT>::samplePostProcessExternalSourceDirection(
    const PointT &normal, MCParticle &particle)
{
    return this->observerProcess_->samplePostProcessExternalSourceDirection(
        normal, particle);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::GroupArray
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::buildPostProcessExternalSourcePlanckPdf(
    const CellT &cell) const
{
    return this->observerProcess_->buildPostProcessExternalSourcePlanckPdf(cell);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::
samplePostProcessExternalSourcePlanckFrequencyInGroup(
    const CellT &cell, std::size_t group)
{
    return this->observerProcess_->samplePostProcessExternalSourcePlanckFrequencyInGroup(
        cell, group);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
double RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                    TraitsT, PositionSamplerT>::
samplePostProcessExternalSourcePlanckFrequency(const CellT &cell)
{
    return this->observerProcess_->samplePostProcessExternalSourcePlanckFrequency(cell);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
typename RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups,
                      OpacityT, TraitsT, PositionSamplerT>::MCParticle
RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
             TraitsT, PositionSamplerT>::generatePostProcessExternalSourceParticle(
    std::size_t cellIndex, const CellT &cell,
    const PostProcessExternalSource &source)
{
    return this->observerProcess_->generatePostProcessExternalSourceParticle(
        cellIndex, cell, source);
}

template<typename PointT, typename GridT, typename CellT, typename ExtensivesT,
         typename EOST, std::size_t NumGroups, typename OpacityT,
         typename TraitsT, typename PositionSamplerT>
bool RadiationIMC<PointT, GridT, CellT, ExtensivesT, EOST, NumGroups, OpacityT,
                  TraitsT, PositionSamplerT>::handlePostProcessExternalSourceBoundary(
    MCParticle &particle, std::size_t cellIndex, std::size_t faceIndex,
    Functionality &functionality)
{
    return this->observerProcess_->handlePostProcessExternalSourceBoundary(
        particle, cellIndex, faceIndex, functionality);
}

} // namespace STORM

#endif // STORM_RADIATION_IMC_FACADE_HPP
