#include "particle/RadiationTransportState.hpp"
#include "radiation/ddmc/DDMCGeometry.hpp"
#include "radiation/ddmc/DDMCSampling.hpp"
#include "radiation/ddmc/DDMCWollaegerInterface.hpp"
#include "TestSupport.hpp"

#include <array>
#include <cmath>
#include <type_traits>

int main()
{
    using State = STORM::RadiationTransportState<std::array<double, 3>>;
    static_assert(std::is_trivially_copyable_v<State>);

    std::array<double, 3> flux{1.0, -2.0, 3.0};
    State state;
    state.set(State::DDMCMode);
    state.set(State::DDMCCellResident);
    state.pendingFlux = flux;
    state.set(State::PendingFlux);
    STORM_TEST_CHECK(state.isDDMC() && state.isResident() && state.hasPendingFlux());
    state.clearPendingFlux();
    STORM_TEST_CHECK(!state.hasPendingFlux());
    state.clearDDMC();
    STORM_TEST_CHECK(!state.isDDMC() && !state.isResident());

    double const conductance = STORM::ddmc::TwoSidedConductance(
        1.0, 0.5, 2.0, 0.5, 2.0);
    STORM_TEST_CHECK(std::abs(conductance - 2.0) < 1.0e-14);
    double const forward = STORM::ddmc::LeakageRate(
        1.0, 1.0, 0.5, 2.0, 0.5, 2.0);
    double const reverse = STORM::ddmc::LeakageRate(
        1.0, 1.0, 0.5, 2.0, 0.5, 2.0);
    STORM_TEST_CHECK(STORM::ddmc::ReciprocityResidual(1.0, forward, 1.0, reverse) == 0.0);

    std::array<double, 4> boundaries{1.0e-3, 1.0, 10.0, 100.0};
    double const lowMass = STORM::ddmc::PlanckBandMass(
        boundaries, 2.0, 0, 1);
    double const totalMass = STORM::ddmc::PlanckBandMass(
        boundaries, 2.0, 0, 3);
    STORM_TEST_CHECK(lowMass > 0.0 && totalMass > lowMass);
    double const fraction = STORM::ddmc::PlanckBandFraction(
        boundaries, 2.0, 0, 1, 3);
    STORM_TEST_CHECK(fraction > 0.0 && fraction < 1.0);

    for(double mu : {1.0e-8, 1.0e-3, 0.1, 0.5, 0.99})
    {
        double const kernel = STORM::ddmc::Kernel(mu);
        STORM_TEST_CHECK(std::isfinite(kernel));
        double const factor = STORM::ddmc::MovingFactor(mu, 0.01);
        STORM_TEST_CHECK(std::isfinite(factor) && factor > 0.0);
        double const admission = STORM::ddmc::StaticAdmissionProbability(
            mu, 10.0, 0.5);
        STORM_TEST_CHECK(admission >= 0.0 && admission <= 1.0);
    }

    for(double random : {0.0, 0.1, 0.5, 0.9, 1.0})
    {
        double const mu = STORM::ddmc::SampleAsymptoticMu(random);
        double const cdf = 0.5 * mu * mu * (1.0 + mu);
        STORM_TEST_CHECK(cdf >= 0.0 && cdf <= 1.0);
    }
    return 0;
}
