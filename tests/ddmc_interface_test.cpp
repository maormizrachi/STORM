#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "radiation/ddmc/DDMCWollaegerInterface.hpp"

namespace {

void require(bool condition, const char *message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

void requireClose(double actual,
                  double expected,
                  double relativeTolerance,
                  const char *message)
{
    double const scale =
        std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::abs(actual - expected) <= relativeTolerance * scale, message);
}

void testCoefficient()
{
    const double coefficient =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 0.9);
    require(STORM::ddmc::IsProbabilisticDensmore2006Coefficient(coefficient),
            "Densmore reference point is not probabilistic");
    requireClose(coefficient, 0.5754275279498156, 2.0e-15,
                 "conversion coefficient does not match Eq. (48)");
    requireClose(STORM::ddmc::Densmore2006AdmissionProbability(
                     1.0, coefficient),
                 1.25 * coefficient,
                 2.0e-15,
                 "normal-incidence probability does not match Eq. (26)");
    const double thin =
        STORM::ddmc::Densmore2006ConversionCoefficient(1.0e-6, 0.9);
    double const standardThin =
        8.0 / (3.0e-6 + 6.0 * STORM::ddmc::ExtrapolationLength);
    requireClose(thin, standardThin, 2.0e-12,
                 "thin-cell limit does not recover the standard boundary");
    const double thick =
        STORM::ddmc::Densmore2006ConversionCoefficient(1.0e6, 0.9);
    requireClose(thick, 0.5257329470522999, 2.0e-6,
                 "large-tau coefficient does not approach Eq. (19)");
    const double pureScattering =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 1.0);
    requireClose(pureScattering,
                 8.0 / (45.0 + 6.0 * STORM::ddmc::ExtrapolationLength),
                 2.0e-15, "omega-to-one limit is discontinuous");
}

void testProbabilityBoundsAndReciprocity()
{
    const double invalid =
        STORM::ddmc::Densmore2006CellCoefficient(5.0, 0.5, 1.5);
    require(!std::isfinite(invalid),
            "coefficient above the paper's four-fifths bound was accepted");

    const double coefficient =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 0.99);
    double const normalAdmission =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, coefficient);
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
            STORM::ddmc::Densmore2006AdmissionProbability(
                mu, coefficient) / bins;
    }
    requireClose(average, coefficient, 2.0e-9,
                 "angular conversion is not reciprocal with boundary leakage");

    double const rate = STORM::ddmc::Densmore2006BoundaryLeakRate(
        2.0, 5.0, 7.0, coefficient);
    requireClose(
        rate,
        7.0 * 2.0 * coefficient / (4.0 * 5.0),
        2.0e-15, "boundary leak rate does not match Eq. (29)");
}

void testCellMappingAndRoundoff()
{
    double const albedo =
        STORM::ddmc::Densmore2006SingleScatterAlbedo(5.0, 0.4);
    requireClose(albedo, 0.92, 2.0e-15,
                 "Fleck effective absorption is not mapped into albedo");

    const double fromCell =
        STORM::ddmc::Densmore2006CellCoefficient(5.0, albedo, 1.5);
    const double fromTau =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, albedo);
    requireClose(fromCell, fromTau, 2.0e-15,
                 "cell mapping does not use twice the center-face distance");

    double const atOne =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, fromCell);
    double const aboveOne =
        STORM::ddmc::Densmore2006AdmissionProbability(
            std::nextafter(1.0, 2.0), fromCell);
    requireClose(aboveOne, atOne, 0.0,
                 "roundoff above unit cosine changes interface admission");
}

} // namespace

int main()
{
    try
    {
        testCoefficient();
        testProbabilityBoundsAndReciprocity();
        testCellMappingAndRoundoff();
    }
    catch(const std::exception &error)
    {
        std::cerr << "storm_ddmc_interface_test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
