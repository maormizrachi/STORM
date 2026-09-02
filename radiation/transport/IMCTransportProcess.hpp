#ifndef STORM_RADIATION_IMCTRANSPORT_PROCESS_HPP
#define STORM_RADIATION_IMCTRANSPORT_PROCESS_HPP

#include "../imc/IMCComponentBase.hpp"

namespace STORM::radiation_imc_detail {

template<typename Owner>
class IMCTransportProcess final : public IMCComponentBase<Owner>
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

    // Host-side opacity policy for the shared portable event kernel.  Keeping
    // this policy with the transport process avoids making RadiationIMC own
    // transport implementation details while preserving the device-safe
    // interface used by AdvanceIMC.
    struct SpectralOpacityPolicy
    {
        Owner *owner = nullptr;

        template<typename ParticleU, typename ViewsU>
        STORM_TRANSPORT_INLINE
        transport::IMCOpacityState Evaluate(
            const ParticleU &particle,
            const ViewsU &,
            const std::size_t cellIndex,
            const double transportFrequency) const
        {
            (void) particle;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            (void) cellIndex;
            (void) transportFrequency;
            return {};
#else
            transport::IMCOpacityState result;
            CellT &cell = this->owner->cells_[cellIndex];
            double frequency = transportFrequency;
            this->owner->clampFrequencyToBounds(frequency);
            result.group = this->owner->opacity_->findGroup(
                frequency, this->owner->energyBoundaries_);
            result.absorption = this->owner->opacity_->CalcAbsorptionOpacity(
                cell, frequency);
            result.scattering = this->owner->opacity_->CalcScatteringOpacity(
                cell, frequency);
            result.fleck = this->owner->factorFleck_[cellIndex];
            return result;
#endif
        }

        template<typename ParticleU, typename ViewsU>
        STORM_TRANSPORT_INLINE
        void Scatter(ParticleU &particle,
                     const ViewsU &,
                     const std::size_t cellIndex,
                     const transport::IMCOpacityState &,
                     const bool effectiveScatter,
                     const double dopplerShift) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            (void) particle;
            (void) cellIndex;
            (void) effectiveScatter;
            (void) dopplerShift;
#else
            CellT &cell = this->owner->cells_[cellIndex];
            particle.velocity =
                this->owner->sampleScatterVelocity(cell, particle);
            particle.frequency *= dopplerShift;
            this->owner->clampFrequencyToBounds(particle.frequency);
            if(effectiveScatter)
            {
                const double reemitRandom =
                    this->owner->randomUnitOpen(particle);
                particle.frequency = this->owner->opacity_->GetThermalEnergy(
                    cell, reemitRandom, this->owner->energyBoundaries_);
            }
#endif
        }

        template<typename ParticleU, typename ViewsU>
        STORM_TRANSPORT_INLINE
        void TallyGroupRadiation(
            const ParticleU &,
            const ViewsU &,
            const std::size_t cellIndex,
            const transport::IMCOpacityState &opacityState,
            const double integratedEnergy) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            (void) cellIndex;
            (void) opacityState;
            (void) integratedEnergy;
#else
            if(this->owner->parameters_.withEgTimeAvg &&
               opacityState.group < NumGroups)
            {
                STORM_TRANSPORT_ACCUMULATE(
                    this->owner->pendingGroupRadiationEnergy_[
                        cellIndex * NumGroups + opacityState.group],
                    integratedEnergy);
            }
#endif
        }
    };
public:
    explicit IMCTransportProcess(Owner &owner) : Base(owner)
    {}

    typename Owner::Functionality
    step(
        MCParticle &particle,
        std::vector<MCParticle> &particlesToAdd)
    {

        #ifndef STORM_IMC_DIFF
            return owner_.stepImpl(particle, particlesToAdd);
        #else
            if(!owner_.SharedFullIMCKernelEligible())
            {
                return owner_.stepImpl(particle, particlesToAdd);
            }

            const std::size_t cellIndex = particle.cellIndex;
            struct CellTally
            {
                double material = 0.0;
                double radiation = 0.0;
                double total = 0.0;
                GroupArray group{};
            };
            auto snapshot = [&]()
            {
                CellTally t;
                t.material = owner_.pendingMaterialEnergy_[cellIndex];
                t.radiation = owner_.pendingRadiationEnergy_[cellIndex];
                t.total = owner_.pendingTotalEnergy_[cellIndex];
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    t.group[group] =
                        owner_.pendingGroupRadiationEnergy_[
                            cellIndex * NumGroups + group];
                }
                return t;
            };
            auto restore = [&](const CellTally &t)
            {
                owner_.pendingMaterialEnergy_[cellIndex] = t.material;
                owner_.pendingRadiationEnergy_[cellIndex] = t.radiation;
                owner_.pendingTotalEnergy_[cellIndex] = t.total;
                for(std::size_t group = 0; group < NumGroups; ++group)
                {
                    owner_.pendingGroupRadiationEnergy_[
                        cellIndex * NumGroups + group] = t.group[group];
                }
            };

            const CellTally before = snapshot();
            const MCParticle particleBefore = particle;

            MCParticle sharedParticle = particle;
            std::vector<MCParticle> sharedAdd;
            const Functionality sharedResult =
                owner_.stepImpl(sharedParticle, sharedAdd);
            const CellTally sharedTally = snapshot();

            restore(before);
            MCParticle legacyParticle = particle;
            std::vector<MCParticle> legacyAdd;
            owner_.imcDiffForceLegacy_ = true;
            Functionality legacyResult;
            try
            {
                legacyResult = owner_.stepImpl(legacyParticle, legacyAdd);
            }
            catch(...)
            {
                owner_.imcDiffForceLegacy_ = false;
                throw;
            }
            owner_.imcDiffForceLegacy_ = false;
            const CellTally legacyTally = snapshot();

            auto differs = [](double a, double b)
            {
                const double scale = std::max({1.0e-300, std::abs(a), std::abs(b)});
                return std::abs(a - b) > 1.0e-11 * scale;
            };
            std::string mismatch;
            auto checkScalar = [&](const char *name, double a, double b)
            {
                if(mismatch.empty() && differs(a, b))
                {
                    std::ostringstream os;
                    os << name << ": shared=" << std::setprecision(17) << a
                       << " legacy=" << std::setprecision(17) << b;
                    mismatch = os.str();
                }
            };
            if(sharedResult.change != legacyResult.change)
            {
                std::ostringstream os;
                os << "change: shared=" << static_cast<int>(sharedResult.change)
                   << " legacy=" << static_cast<int>(legacyResult.change);
                mismatch = os.str();
            }
            if(mismatch.empty() &&
               sharedResult.change == ParticleStatus::CELL_MOVE)
            {
                if(sharedResult.nextCellIndex != legacyResult.nextCellIndex)
                {
                    std::ostringstream os;
                    os << "nextCellIndex: shared=" << sharedResult.nextCellIndex
                       << " legacy=" << legacyResult.nextCellIndex;
                    mismatch = os.str();
                }
                else if(sharedResult.boundaryCrossing !=
                        legacyResult.boundaryCrossing)
                {
                    std::ostringstream os;
                    os << "boundaryCrossing: shared="
                       << sharedResult.boundaryCrossing
                       << " legacy=" << legacyResult.boundaryCrossing;
                    mismatch = os.str();
                }
            }
            checkScalar("weight", sharedParticle.weight, legacyParticle.weight);
            checkScalar("timeLeft", sharedParticle.timeLeft, legacyParticle.timeLeft);
            checkScalar("frequency", sharedParticle.frequency, legacyParticle.frequency);
            checkScalar("location.x", sharedParticle.location.x, legacyParticle.location.x);
            checkScalar("location.y", sharedParticle.location.y, legacyParticle.location.y);
            checkScalar("location.z", sharedParticle.location.z, legacyParticle.location.z);
            checkScalar("velocity.x", sharedParticle.velocity.x, legacyParticle.velocity.x);
            checkScalar("velocity.y", sharedParticle.velocity.y, legacyParticle.velocity.y);
            checkScalar("velocity.z", sharedParticle.velocity.z, legacyParticle.velocity.z);
            checkScalar("tally.material", sharedTally.material, legacyTally.material);
            checkScalar("tally.radiation", sharedTally.radiation, legacyTally.radiation);
            checkScalar("tally.total", sharedTally.total, legacyTally.total);
            if(mismatch.empty() &&
               sharedParticle.rngCounter != legacyParticle.rngCounter)
            {
                std::ostringstream os;
                os << "rngCounter: shared=" << sharedParticle.rngCounter
                   << " legacy=" << legacyParticle.rngCounter;
                mismatch = os.str();
            }

            if(!mismatch.empty() && owner_.imcDiffReports_ < 20)
            {
                ++owner_.imcDiffReports_;
                std::ostringstream os;
                os << std::setprecision(17)
                   << "[IMC-DIFF] " << mismatch << "\n"
                   << "  cell=" << cellIndex
                   << " in.weight=" << particleBefore.weight
                   << " in.initialWeight=" << particleBefore.initialWeight
                   << " in.timeLeft=" << particleBefore.timeLeft
                   << " in.frequency=" << particleBefore.frequency << "\n"
                   << "  in.loc=(" << particleBefore.location.x << ","
                   << particleBefore.location.y << "," << particleBefore.location.z
                   << ") in.vel=(" << particleBefore.velocity.x << ","
                   << particleBefore.velocity.y << "," << particleBefore.velocity.z
                   << ") in.rngCounter=" << particleBefore.rngCounter << "\n"
                   << "  shared: change=" << static_cast<int>(sharedResult.change)
                   << " next=" << sharedResult.nextCellIndex
                   << " bx=" << sharedResult.boundaryCrossing
                   << " w=" << sharedParticle.weight
                   << " tl=" << sharedParticle.timeLeft
                   << " nu=" << sharedParticle.frequency
                   << " dMat=" << (sharedTally.material - before.material)
                   << " dRad=" << (sharedTally.radiation - before.radiation) << "\n"
                   << "  legacy: change=" << static_cast<int>(legacyResult.change)
                   << " next=" << legacyResult.nextCellIndex
                   << " bx=" << legacyResult.boundaryCrossing
                   << " w=" << legacyParticle.weight
                   << " tl=" << legacyParticle.timeLeft
                   << " nu=" << legacyParticle.frequency
                   << " dMat=" << (legacyTally.material - before.material)
                   << " dRad=" << (legacyTally.radiation - before.radiation)
                   << std::endl;
                std::cerr << os.str();
            }

            // The legacy path is the reference: keep its state so the run stays on
            // the known-good trajectory while divergences are collected.
            restore(legacyTally);
            particle = legacyParticle;
            particlesToAdd = legacyAdd;
            return legacyResult;
        #endif
    }

    typename Owner::Functionality stepImpl(MCParticle &particle, std::vector<MCParticle> &particlesToAdd)
    {

            Functionality functionality;

            std::size_t cellIndex = particle.cellIndex;
            CellT &cell = owner_.cells_[cellIndex];

            if(particle.radiationState.hasPendingFlux() and cellIndex < owner_.componentGrid().GetPointNo())
            {
                owner_.addDDMCFluxContribution(cellIndex, particle.radiationState.pendingFlux);
                particle.radiationState.clearPendingFlux();
            }

            double dopplerShift = 1.0;
            bool useComovingTransportFrame = false;
            if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
            {
                if((owner_.parameters_.withHydro && !owner_.parameters_.MMC) ||
                   (owner_.parameters_.postProcess.enabled && owner_.parameters_.postProcess.useCellVelocities))
                {
                    useComovingTransportFrame = true;
                    dopplerShift = radiation_imc_detail::computeDopplerShift<PointT>(particle, cell);
                }
            }

            if(!particle.radiationState.isDDMC() &&
               owner_.parameters_.withRandomWalk && !owner_.parameters_.withCompton &&
               owner_.randomWalk_ && owner_.rwCellEligible_[cellIndex])
            {
                if(owner_.tryRandomWalkStep(particle, functionality))
                {
                    ++owner_.rwStepCount_;
                    return functionality;
                }
            }

            if(owner_.SharedDDMCKernelEligible() or particle.radiationState.isDDMC() or
                (owner_.parameters_.withDDMC and cellIndex < owner_.ddmcCellData_.size() and owner_.ddmcCellData_[cellIndex].eligible))
            {
                const bool sharedDDMC = owner_.SharedDDMCKernelEligible();
                const bool sharedDDMCEvent =
                    owner_.SharedDDMCEventKernelEligible();
                ddmc::ColdState<PointT> cold;
                cold.pendingFlux = particle.radiationState.pendingFlux;
                if(sharedDDMCEvent)
                {
                    const ddmc::AdvanceResult<PointT> result = ddmc::AdvanceDDMC(particle, cold, owner_.GetHostTransportViews());
                    particle.radiationState.pendingFlux = cold.pendingFlux;
                    if(result.error != ddmc::AdvanceError::None)
                    {
                        if(result.error == ddmc::AdvanceError::HostFallback)
                        {
                            if(owner_.tryDDMCStep(particle, functionality))
                            {
                                return functionality;
                            }
                        }
                        StormError eo("RadiationIMC shared DDMC transport failed");
                        eo.addEntry("Cell index", particle.cellIndex);
                        eo.addEntry("DDMC error", static_cast<int>(result.error));
                        throw eo;
                    }
                    if(result.taken)
                    {
                        ++owner_.ddmcStepCount_;
                        if(result.event == ddmc::AdvanceEvent::Census)
                        {
                            ++owner_.ddmcCensusCount_;
                        }
                        else if(result.event == ddmc::AdvanceEvent::DDMCLeak)
                        {
                            ++owner_.ddmcLeakCount_;
                            ++owner_.ddmcResidentLeakCount_;
                            if(result.remotePendingFlux)
                                ++owner_.ddmcRemoteResidentLeakCount_;
                        }
                        else if(result.event == ddmc::AdvanceEvent::IMCLeak)
                        {
                            ++owner_.ddmcLeakCount_;
                            ++owner_.ddmcTransportLeakCount_;
                        }
                        else if(result.event == ddmc::AdvanceEvent::Upscatter)
                        {
                            ++owner_.ddmcUpscatterCount_;
                        }
                        return result.step;
                    }
                }
                if(sharedDDMC)
                {
                    gpu::TransportResult imcResult = transport::AdvanceIMC(particle, owner_.GetHostTransportViews());
                    const ddmc::InterfaceResult interface = gpu::ApplyDDMCInterface(particle, cold, imcResult, owner_.GetHostTransportViews());
                    particle.radiationState.pendingFlux = cold.pendingFlux;
                    if(interface.taken and interface.event == ddmc::InterfaceEvent::Admitted and interface.extraSplitCount > 0)
                    {
                        const cell_index_t targetCellIndex = imcResult.step.nextCellIndex;
                        const std::size_t targetCell = static_cast<std::size_t>(targetCellIndex);
                        cell_id_t targetID = std::numeric_limits<cell_id_t>::max();
                        if(targetCell < owner_.ddmcPointCellID_.size())
                        {
                            targetID = static_cast<cell_id_t>(
                                owner_.ddmcPointCellID_[targetCell]);
                        }
                        const PointT targetCenter = owner_.componentGrid().GetMeshPoint(targetCell);
                        for(std::size_t copy = 0; copy < interface.extraSplitCount; ++copy)
                        {
                            MCParticle extra = particle;
                            extra.id = std::numeric_limits<std::size_t>::max();
                            extra.cellID = targetID;
                            extra.cellIndex = targetCellIndex;
                            extra.location = targetCenter;
                            particlesToAdd.push_back(std::move(extra));
                        }
                    }
                    if(imcResult.error == gpu::TransportError::HostFallback)
                    {
                        if(owner_.tryDDMCStep(particle, functionality))
                        {
                            return functionality;
                        }
                    }
                    if(imcResult.error != gpu::TransportError::None)
                    {
                        StormError eo("RadiationIMC shared DDMC/IMC transport failed");
                        eo.addEntry("Cell index", particle.cellIndex);
                        eo.addEntry("IMC transport error", static_cast<int>(imcResult.error));
                        throw eo;
                    }
                    return imcResult.step;
                }
                else if(owner_.tryDDMCStep(particle, functionality))
                {
                    return functionality;
                }
            }

            if(owner_.GreyKernelEligible())
            {
                gpu::TransportResult result = transport::AdvanceIMC(particle, owner_.GetHostTransportViews());
                if(result.error != gpu::TransportError::None)
                {
                    StormError eo("RadiationIMC GPU-compatible grey transport failed");
                    eo.addEntry("Cell index", particle.cellIndex);
                    eo.addEntry("Transport error", static_cast<int>(result.error));
                    throw eo;
                }
                return result.step;
            }
            if(owner_.SharedFullIMCKernelEligible())
            {
                const SpectralOpacityPolicy opacityPolicy{&owner_};
                const transport::TransportResult result = transport::AdvanceIMC(particle, owner_.GetHostTransportViews(), opacityPolicy);
                if(result.error != transport::TransportError::None)
                {
                    StormError eo("RadiationIMC shared full transport failed");
                    eo.addEntry("Cell index", particle.cellIndex);
                    eo.addEntry("Transport error", static_cast<int>(result.error));
                    throw eo;
                }
                return result.step;
            }

            auto [faceIntersect, timeIntersect, nextCellIndex] = owner_.componentIntersectionDetails(particle);

            double shiftedFrequency = particle.frequency * dopplerShift;
            double absorptionOpacity;
            std::size_t group = std::numeric_limits<std::size_t>::max();
            if(owner_.parameters_.withMultigroupOpacity)
            {
                owner_.clampFrequencyToBounds(shiftedFrequency);
                group = owner_.opacity_->findGroup(shiftedFrequency, owner_.energyBoundaries_);
                absorptionOpacity = owner_.parameters_.withCompton
                    ? owner_.comptonData_[cellIndex].absorptionOpacity[group]
                    : owner_.opacity_->CalcAbsorptionOpacity(cell, shiftedFrequency);
            }
            else
            {
                absorptionOpacity = owner_.planckOpacities_[cellIndex];
            }
            double elasticScatteringOpacity = owner_.parameters_.withCompton
                ? 0.0
                : (owner_.parameters_.withMultigroupOpacity
                    ? owner_.opacity_->CalcScatteringOpacity(cell, shiftedFrequency)
                    : owner_.scatteringOpacities_[cellIndex]);
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
            double const transportFleck = (owner_.parameters_.withCompton and group < NumGroups)? owner_.comptonData_[cellIndex].fleck : owner_.factorFleck_[cellIndex];
            double effectiveAbsorptionOpacity = (1.0 - transportFleck) * absorptionOpacity;
            double comptonOpacity = 0.0;
            if(owner_.parameters_.withCompton && group < NumGroups)
            {
                comptonOpacity = owner_.comptonData_[cellIndex].comptonOutRate[group];
            }
            double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity + comptonOpacity;
            double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
            double _log1p = -std::log1p(owner_.randomUnitOpen(particle) - 1.0);
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
            if(owner_.observer_)
            {
                observerCrossing = owner_.observer_->nextOutwardCrossing(particle.location, particle.velocity, particle.timeLeft);
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
                [](const std::pair<std::size_t, double> &a, const std::pair<std::size_t, double> &b)
                {
                    return a.second < b.second;
                });
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
            if(not owner_.parameters_.noHydroFeedback and not owner_.parameters_.postProcess.enabled)
            {
                double const materialDeposit = -expFactor2 * particle.weight;
                owner_.tallyMaterialEnergy(cellIndex, materialDeposit, owner_.parameters_.withCompton);
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if(owner_.parameters_.withHydro and not owner_.parameters_.diffusionPressureGradient)
                    {
                        owner_.tallyMomentum(cellIndex, -expFactor1 * particle.weight * particle.velocity * units::inv_clight2);
                    }
                }
            }
            owner_.tallyRadiationEnergy(cellIndex, integratedForTally);
            if((owner_.parameters_.withEgTimeAvg or owner_.parameters_.withCompton) and owner_.parameters_.withMultigroupOpacity)
            {
                std::size_t g = owner_.opacity_->findGroup(particle.frequency, owner_.energyBoundaries_);
                if(g < NumGroups)
                {
                    owner_.tallyGroupRadiationEnergy(cellIndex, g, integratedForTally);
                }
            }
            double const weightBeforeContinuousDecay = particle.weight;
            particle.weight *= 1.0 + expFactor1;
            if(owner_.parameters_.postProcess.enabled and owner_.observer_)
            {
                owner_.observer_->addAbsorbedEnergy(weightBeforeContinuousDecay - particle.weight);
            }

            MCParticle const labParticleBeforeCompton = particle;

            if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
            {
                if(owner_.observer_ and owner_.parameters_.postProcess.enabled)
                {
                    owner_.observer_->addCutoffEnergy(particle.weight);
                }
                functionality.change = ParticleStatus::REMOVE;
                if(not owner_.parameters_.noHydroFeedback and not owner_.parameters_.postProcess.enabled)
                {
                    owner_.tallyMaterialEnergy(cellIndex, particle.weight, owner_.parameters_.withCompton);
                }
                return functionality;
            }

            if(min.first == INTERSECTION)
            {
                if(owner_.handlePostProcessExternalSourceBoundary(particle, cellIndex, faceIntersect, functionality))
                {
                    return functionality;
                }
                if(not particle.radiationState.isDDMC() and owner_.tryIMCToDDMCInterface(particle, functionality, particlesToAdd, cellIndex, nextCellIndex, faceIntersect))
                {
                    return functionality;
                }
                if(particle.radiationState.bypassCellID != std::numeric_limits<std::size_t>::max())
                {
                    std::size_t const exchangedCellID = (cellIndex < owner_.ddmcPointCellID_.size())? owner_.ddmcPointCellID_[cellIndex] : std::numeric_limits<std::size_t>::max();
                    std::size_t const currentCellID = (exchangedCellID == std::numeric_limits<std::size_t>::max())? cellIndex : exchangedCellID;
                    if(currentCellID == particle.radiationState.bypassCellID)
                    {
                        particle.radiationState.bypassCellID = std::numeric_limits<std::size_t>::max();
                    }
                }
                functionality.change = ParticleStatus::CELL_MOVE;
                functionality.nextCellIndex = nextCellIndex;
                functionality.boundaryCrossing = owner_.componentGrid().IsPointOutsideBox(nextCellIndex);
            }
            else if(min.first == SCATTERING)
            {
                PointT oldVelocity = particle.velocity;
                double D_lab_to_co = dopplerShift;
                double eventRandom = owner_.randomUnitOpen(particle) * eventOpacity;
                bool isEffectiveScatter = false;
                bool isComptonScatter = false;
                bool comptonTransformedToLab = false;
        #ifdef MONTECARLO_POLARIZATION
                MCParticle polarizationMaterialParticle = particle;
                PointT polarizationOldVelocity = particle.velocity;
                if(owner_.polarizationEnabled())
                {
                    polarization::initializeIfNeeded<PointT>(polarizationMaterialParticle);
                    if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                    {
                        if(owner_.parameters_.withHydro && !owner_.parameters_.MMC)
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
                    if(owner_.polarizationEnabled())
                    {
                        auto u01 = [&]()
                        {
                            return owner_.randomUnitOpen(particle);
                        };
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
                    particle.velocity = owner_.sampleScatterVelocity(cell, particle);
                    }
                }
                else if((eventRandom -= elasticScatteringOpacity) < effectiveAbsorptionOpacity)
                {
                    particle.velocity = owner_.sampleScatterVelocity(cell, particle);
                    isEffectiveScatter = true;
                }
                else
                {
                    if(!owner_.parameters_.withCompton || group >= NumGroups)
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
                            owner_.clampFrequencyToBounds(comptonParticle.frequency);
                        }
                    }
                    PointT const comptonOldVelocity = comptonParticle.velocity;
                    double const comptonOldWeight = comptonParticle.weight;
                    double const comovingMaterialDeposit =
                        owner_.applyComptonScatterEvent(
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
                            owner_.clampFrequencyToBounds(comptonParticle.frequency);
                        }
                    }
                    particle = comptonParticle;
                    if(!owner_.parameters_.noHydroFeedback &&
                       !owner_.parameters_.postProcess.enabled)
                    {
                        owner_.tallyMaterialEnergy(
                            cellIndex, comovingMaterialDeposit);
                        owner_.pendingTotalEnergy_[cellIndex] +=
                            labParticleBeforeCompton.weight - particle.weight;
                        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                        {
                            if(owner_.parameters_.withHydro &&
                               !owner_.parameters_.diffusionPressureGradient)
                            {
                                owner_.tallyMomentum(
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
                if(owner_.parameters_.withMultigroupOpacity)
                {
                    if(!isComptonScatter)
                    {
                        particle.frequency *= dopplerShift;
                        owner_.clampFrequencyToBounds(particle.frequency);
                    }
                    if(isEffectiveScatter)
                    {
                        if(owner_.parameters_.withCompton)
                        {
                            std::size_t const targetGroup = owner_.sampleComptonCdf(owner_.comptonData_[cellIndex].baseSourceCdf, owner_.randomUnitOpen(particle));
                            particle.frequency = owner_.frequencyForComptonGroup(targetGroup);
                        }
                        else
                        {
                            double reemitRandom = owner_.randomUnitOpen(particle);
                            particle.frequency = owner_.opacity_->GetThermalEnergy(
                                cell, reemitRandom, owner_.energyBoundaries_);
                        }
                    }
                }
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value)
                {
                    if(owner_.parameters_.withHydro && !owner_.parameters_.MMC &&
                       !isComptonScatter && !comptonTransformedToLab)
                    {
                        double weightBefore = particle.weight;
                        particle.weight *= D_lab_to_co;
                        radiation_imc_detail::lorentzTransformToLab<PointT>(particle, cell);
                        if(owner_.parameters_.withMultigroupOpacity)
                        {
                            owner_.clampFrequencyToBounds(particle.frequency);
                        }
        #ifdef MONTECARLO_POLARIZATION
                        if(owner_.polarizationEnabled())
                        {
                            particle.polarizationBasis = polarization::projectBasisToDirection(
                                particle.polarizationBasis, particle.velocity);
                        }
        #endif
                        if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                        {
                            if(not owner_.parameters_.diffusionPressureGradient and not owner_.parameters_.noHydroFeedback)
                            {
                                owner_.tallyMomentum(cellIndex, (weightBefore * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2);
                            }
                        }
                    }
                }
        #ifdef MONTECARLO_POLARIZATION
                if(owner_.polarizationEnabled())
                {
                    if(isEffectiveScatter or isComptonScatter)
                    {
                        polarization::resetUnpolarized<PointT>(particle);
                    }
                    polarization::initializeIfNeeded<PointT>(particle);
                    particle.polarizationBasis = polarization::projectBasisToDirection(particle.polarizationBasis, particle.velocity);
                    polarization::clampLinearPolarization(particle.stokesQ, particle.stokesU);
                }
        #endif
                functionality.change = ParticleStatus::NO_CELL_MOVE;
            }
            else if(min.first == OBSERVER)
            {
                owner_.recordObserverCrossing(particle, observerCrossing.point);
                // Leave the packet outside the observer surface so the same positive
                // outward root cannot be selected again on the next step.
                particle.location = observerCrossing.point + normalize(particle.velocity) * std::max(1.0e-12, 1.0e-10 * std::max(1.0, fastabs(observerCrossing.point)));
                functionality.change = ParticleStatus::NO_CELL_MOVE;
            }
            else if(min.first == TIMELEFT)
            {
                if(owner_.observer_)
                {
                    owner_.observer_->addTimedOutEnergy(particle.weight);
                }
                functionality.change = ParticleStatus::DONE;
            }

            return functionality;
    }

    void addCensusEnergyToExtensive(std::size_t cellIndex, double energy)
    {
            radiation_imc_detail::addRadiationEnergyIfPresent(owner_.extensives_[cellIndex], energy);
    }

    void addCensusGroupEnergyToExtensive(std::size_t cellIndex, std::size_t group, double energy)
    {
            if constexpr(radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
            {
                if(group < NumGroups)
                {
                    owner_.extensives_[cellIndex].Eg[group] += energy;
                }
            }
            else
            {
                (void) cellIndex;
                (void) group;
                (void) energy;
            }
    }

    void postStep(const std::vector<MCParticle> &particles, double fullDt)
    {
        this->postStepImpl(particles, fullDt, false);
    }

    void postStepWithDeviceCensus(const std::vector<MCParticle> &hostParticles, double fullDt)
    {
        this->postStepImpl(hostParticles, fullDt, true);
    }

    void postStepImpl(const std::vector<MCParticle> &particles, double fullDt, bool includeDeviceCensus)
    {
        const std::size_t Ncells = owner_.componentGrid().GetPointNo();
        owner_.deviceExecutor_->addTallies(
            owner_.pendingMaterialEnergy_,
            owner_.pendingRadiationEnergy_,
            owner_.pendingGroupRadiationEnergy_,
            owner_.pendingMomentum_,
            owner_.ddmcFluxRhsIntegrated_,
            owner_.rwStepCount_,
            owner_.ddmcStepCount_,
            owner_.ddmcLeakCount_,
            owner_.ddmcResidentLeakCount_,
            owner_.ddmcTransportLeakCount_,
            owner_.ddmcRemoteResidentLeakCount_,
            owner_.ddmcCensusCount_);
        owner_.deviceExecutor_->addDDMCDiagnostics();
        owner_.applyTransportTallies();
        double const tallyDt = owner_.parameters_.postProcess.enabled? owner_.parameters_.postProcess.transportTime : fullDt;
        for(std::size_t i = 0; i < Ncells; ++i)
        {
            owner_.Erad_time_avg_[i] /= (tallyDt * owner_.componentGrid().GetVolume(i));
            if((owner_.parameters_.withEgTimeAvg ||
                owner_.parameters_.withCompton) &&
                owner_.parameters_.withMultigroupOpacity)
            {
                double norm = tallyDt * owner_.componentGrid().GetVolume(i);
                for(std::size_t g = 0; g < NumGroups; ++g)
                {
                    owner_.Eg_time_avg_[i][g] /= norm;
                }
            }
        }

        // Post-processing is a diagnostic transport pass.  It must not feed
        // packet absorption, Compton residuals, or hydro synchronization back
        // into the snapshot that supplied the source.
        if(owner_.parameters_.postProcess.enabled)
        {
            return;
        }

        if(not owner_.parameters_.noHydroFeedback)
        {
            if(owner_.parameters_.diffusionPressureGradient and owner_.parameters_.withHydro)
            {
                if constexpr(radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
    #ifdef STORM_WITH_MPI
                    STORM::MPI_exchange_data(owner_.componentGrid(), owner_.Erad_time_avg_, true);
    #endif
                }
            }

            if(owner_.parameters_.withDDMC)
            {
                owner_.applyDDMCMomentumFeedback(fullDt);
            }

            for(std::size_t i = 0; i < Ncells; ++i)
            {
                owner_.throwIfNegativeInternalEnergy(i, "postStep");
                if constexpr(radiation_imc_detail::has_member_velocity<CellT>::value && radiation_imc_detail::has_member_momentum<ExtensivesT>::value)
                {
                    if(owner_.parameters_.withHydro)
                    {
                        if(owner_.parameters_.diffusionPressureGradient)
                        {
                            PointT const radiationEnergyGradient =
                                radiation_pressure_gradient_detail::
                                    reconstructRadiationEnergyGradient<PointT>(
                                        owner_.componentGrid(), owner_.Erad_time_avg_, i);
                            owner_.extensives_[i].momentum +=
                                radiationEnergyGradient *
                                (-fullDt * owner_.componentGrid().GetVolume(i) / 3.0);
                        }
                    }
                }
                owner_.synchronizeMaterialCell(i);
            }
        }

        for(std::size_t i = 0; i < Ncells; ++i)
        {
            radiation_imc_detail::clearRadiationEnergyIfPresent(owner_.extensives_[i]);
            radiation_imc_detail::clearGroupEnergyIfPresent(owner_.extensives_[i]);
        }
        if(includeDeviceCensus)
        {
            std::vector<double> censusRadiationEnergy;
            std::vector<double> censusGroupRadiationEnergy;
            if(not owner_.deviceExecutor_->copyCensusTallies(censusRadiationEnergy, censusGroupRadiationEnergy))
            {
                throw StormError("Device census post-step requested without device census tallies");
            }
            if(censusRadiationEnergy.size() != Ncells)
            {
                throw StormError("Device census radiation-energy tally size mismatch");
            }
            for(std::size_t i = 0; i < Ncells; ++i)
            {
                this->addCensusEnergyToExtensive(i, censusRadiationEnergy[i]);
                if(owner_.parameters_.withMultigroupOpacity and censusGroupRadiationEnergy.size() == Ncells * NumGroups)
                {
                    const std::size_t offset = i * NumGroups;
                    for(std::size_t group = 0; group < NumGroups; ++group)
                    {
                        this->addCensusGroupEnergyToExtensive(i, group, censusGroupRadiationEnergy[offset + group]);
                    }
                }
            }
        }
        for(const MCParticle &particle : particles)
        {
            this->addCensusEnergyToExtensive(particle.cellIndex, particle.weight);
            if(owner_.parameters_.withMultigroupOpacity)
            {
                std::size_t g = owner_.opacity_->findGroup(particle.frequency, owner_.energyBoundaries_);
                this->addCensusGroupEnergyToExtensive(particle.cellIndex, g, particle.weight);
            }
        }

        if(owner_.parameters_.withCompton)
        {
            owner_.applyComptonEndOfStepCorrection(fullDt);
            std::vector<MCParticle> &mutableParticles = const_cast<std::vector<MCParticle> &>(particles);
            owner_.reconcileComptonParticles(mutableParticles);
            if(not owner_.parameters_.noHydroFeedback)
            {
                for(std::size_t i = 0; i < Ncells; ++i)
                {
                    owner_.synchronizeMaterialCell(i);
                }
            }
        }

        for(std::size_t i = 0; i < Ncells; ++i)
        {
            const double mass = owner_.extensives_[i].mass;
            const double radiationPerMass = (mass > 0.0)? radiation_imc_detail::radiationEnergyIfPresent(owner_.extensives_[i]) / mass : 0.0;
            radiation_imc_detail::setCellRadiationEnergyIfPresent(owner_.cells_[i], radiationPerMass);
            if constexpr(radiation_imc_detail::has_member_group_energy_mutable<CellT>::value)
            {
                for(std::size_t g = 0; g < NumGroups; ++g)
                {
                    double groupVal = (mass > 0.0 && radiation_imc_detail::has_member_group_energy_mutable<ExtensivesT>::value)
                        ? owner_.traits_.extensiveGroupEnergy(owner_.extensives_[i], g) / mass
                        : 0.0;
                    radiation_imc_detail::setCellGroupEnergyIfPresent(owner_.cells_[i], g, groupVal);
                }
            }
        }
    }

    std::string getAccelerationDebugInfo(std::size_t cellIndex, double frequency) const
    {
        if(cellIndex >= owner_.ddmcCellData_.size())
        {
            return std::string();
        }
        DDMCCellData const &data = owner_.ddmcCellData_[cellIndex];
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
            << " unsupported_boundary_faces=" << data.unsupportedBoundaryFaceCount
            << " first_unsupported_boundary_face=" << data.firstUnsupportedBoundaryFace
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
            << " ddmc_boundary_excluded=" << (data.boundaryExcluded ? 1 : 0)
            << " ddmc_rigid_boundary_faces=" << data.rigidBoundaryFaceCount
            << " ddmc_unsupported_boundary_faces=" << data.unsupportedBoundaryFaceCount
            << " ddmc_first_unsupported_boundary_face=" << data.firstUnsupportedBoundaryFace
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
            << " ddmc_resident_leaks=" << owner_.ddmcResidentLeakCount_
            << " ddmc_transport_leaks=" << owner_.ddmcTransportLeakCount_
            << " ddmc_remote_resident_leaks=" << owner_.ddmcRemoteResidentLeakCount_
            << " ddmc_mpi_face_flux_reductions=" << owner_.ddmcMPIFaceFluxReductionCount_
            << " ddmc_leak_invalid_geometry=" << owner_.ddmcLeakInvalidGeometryCount_
            << " ddmc_leak_reciprocity_max=" << owner_.ddmcLeakReciprocityResidualMax_
            << " ddmc_interface_incident=" << owner_.ddmcInterfaceIncidentCount_
            << " ddmc_interface_admitted=" << owner_.ddmcInterfaceAdmittedCount_
            << " ddmc_interface_reflected=" << owner_.ddmcInterfaceReflectedCount_
            << " ddmc_interface_gu_applied=" << owner_.ddmcInterfaceGuAppliedCount_
            << " ddmc_interface_gu_fallback=" << owner_.ddmcInterfaceGuFallbackCount_
            << " ddmc_interface_bypass=" << owner_.ddmcInterfaceBypassCount_
            << " ddmc_interface_split_packets=" << owner_.ddmcInterfaceSplitPacketCount_
            << " ddmc_interface_min_mu=" << owner_.ddmcInterfaceMinimumMu_
            << " ddmc_interface_max_gu=" << owner_.ddmcMovingInterfaceMaxFactor_
            << " ddmc_interface_flux_tallies=" << owner_.ddmcInterfaceFluxTallyCount_
            << " ddmc_leak_reciprocity_checks=" << owner_.ddmcLeakReciprocityCheckCount_;
        return out.str();
    }

};

} // namespace STORM::radiation_imc_detail

#endif // STORM_RADIATION_IMCTRANSPORT_PROCESS_HPP
