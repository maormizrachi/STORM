#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "radiation/ddmc/DDMCWollaegerInterface.hpp"

namespace {

void require(bool condition, const char *message)
{
    if(!condition)
        throw std::runtime_error(message);
}

void requireClose(double actual,
                  double expected,
                  double relativeTolerance,
                  const char *message)
{
    double const scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::abs(actual - expected) <= relativeTolerance * scale, message);
}

void testPaperReferencePoint()
{
    auto const coefficients = STORM::ddmc::Densmore2006Coefficients(15.0, 0.9);
    require(coefficients.valid, "Densmore reference point is not probabilistic");
    requireClose(coefficients.analyticEmissivity, 0.5257329470522999,
                 2.0e-15, "analytic emissivity does not match Eq. (19)");
    requireClose(coefficients.conversionCoefficient, 0.5754275279498156,
                 2.0e-15, "conversion coefficient does not match Eq. (48)");
    requireClose(STORM::ddmc::Densmore2006AdmissionProbability(
                     1.0, coefficients),
                 1.25 * coefficients.conversionCoefficient,
                 2.0e-15, "normal-incidence probability does not match Eq. (26)");
}

void testAsymptoticLimits()
{
    constexpr double omega = 0.9;
    auto const thin = STORM::ddmc::Densmore2006Coefficients(1.0e-6, omega);
    double const standardThin = 8.0 /
        (3.0e-6 + 6.0 * STORM::ddmc::ExtrapolationLength);
    requireClose(thin.conversionCoefficient, standardThin, 2.0e-12,
                 "thin-cell limit does not recover the standard boundary");

    auto const thick = STORM::ddmc::Densmore2006Coefficients(1.0e6, omega);
    require(thick.valid, "optically thick Densmore coefficient is invalid");
    requireClose(thick.conversionCoefficient, thick.analyticEmissivity,
                 2.0e-6, "large-tau coefficient does not approach emissivity");

    auto const pureScattering =
        STORM::ddmc::Densmore2006Coefficients(15.0, 1.0);
    double const pureScatteringLimit = 8.0 /
        (45.0 + 6.0 * STORM::ddmc::ExtrapolationLength);
    requireClose(pureScattering.conversionCoefficient, pureScatteringLimit,
                 2.0e-15, "omega-to-one limit is discontinuous");
}

void testProbabilityBoundsAndReciprocity()
{
    auto const invalid = STORM::ddmc::Densmore2006Coefficients(15.0, 0.5);
    require(!invalid.valid,
            "coefficient above the paper's four-fifths bound was accepted");

    auto const coefficients = STORM::ddmc::Densmore2006Coefficients(15.0, 0.99);
    require(coefficients.valid, "valid high-albedo coefficient was rejected");
    double const normalAdmission =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, coefficients);
    require(normalAdmission >= 0.0 && normalAdmission <= 1.0,
            "angular admission probability left the unit interval");

    // Flux-weighted hemispheric average of Eq. (26):
    // integral_0^1 2*mu*P(mu) dmu = P.
    constexpr int bins = 10000;
    double average = 0.0;
    for(int i = 0; i < bins; ++i)
    {
        double const mu = (static_cast<double>(i) + 0.5) / bins;
        average += 2.0 * mu *
            STORM::ddmc::Densmore2006AdmissionProbability(mu, coefficients) /
            bins;
    }
    requireClose(average, coefficients.conversionCoefficient, 2.0e-9,
                 "angular conversion is not reciprocal with boundary leakage");

    double const rate = STORM::ddmc::Densmore2006BoundaryLeakRate(
        2.0, 5.0, 7.0, coefficients);
    requireClose(rate,
                 7.0 * 2.0 * coefficients.conversionCoefficient / (4.0 * 5.0),
                 2.0e-15, "boundary leak rate does not match Eq. (29)");
}

void testCellMappingAndRoundoff()
{
    double const albedo = STORM::ddmc::Densmore2006SingleScatterAlbedo(
        5.0, 0.4);
    requireClose(albedo, 0.92, 2.0e-15,
                 "Fleck effective absorption is not mapped into albedo");

    auto const fromCell = STORM::ddmc::Densmore2006CellCoefficients(
        5.0, albedo, 1.5);
    auto const fromTau = STORM::ddmc::Densmore2006Coefficients(15.0, albedo);
    require(fromCell.valid == fromTau.valid,
            "cell mapping changed coefficient validity");
    requireClose(fromCell.conversionCoefficient,
                 fromTau.conversionCoefficient, 2.0e-15,
                 "cell mapping does not use twice the center-face distance");

    auto const fallback = STORM::ddmc::Densmore2006CellCoefficients(
        5.0, 0.5, 1.5);
    require(!fallback.valid,
            "cell mapping did not expose the legacy-fallback condition");

    double const atOne =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, fromCell);
    double const aboveOne = STORM::ddmc::Densmore2006AdmissionProbability(
        std::nextafter(1.0, 2.0), fromCell);
    requireClose(aboveOne, atOne, 0.0,
                 "roundoff above unit cosine changes interface admission");
}

} // namespace

int main()
{
    try
    {
        testPaperReferencePoint();
        testAsymptoticLimits();
        testProbabilityBoundsAndReciprocity();
        testCellMappingAndRoundoff();
    }
    catch(std::exception const &error)
    {
        std::cerr << "storm_ddmc_interface_test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
