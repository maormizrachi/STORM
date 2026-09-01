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

void requireClose(double actual, double expected, double tolerance,
                  const char *message)
{
    double const scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::abs(actual - expected) <= tolerance * scale, message);
}

void testCoefficient()
{
    double const reference =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 0.9);
    require(STORM::ddmc::IsProbabilisticDensmore2006Coefficient(reference),
            "paper reference point is not probabilistic");
    requireClose(reference, 0.5754275279498156, 2.0e-15,
                 "conversion coefficient does not match Eq. (48)");
    requireClose(STORM::ddmc::Densmore2006AdmissionProbability(1.0, reference),
                 1.25 * reference, 2.0e-15,
                 "normal-incidence probability does not match Eq. (26)");

    double const thin =
        STORM::ddmc::Densmore2006ConversionCoefficient(1.0e-6, 0.9);
    double const standardThin = 8.0 /
        (3.0e-6 + 6.0 * STORM::ddmc::ExtrapolationLength);
    requireClose(thin, standardThin, 2.0e-12,
                 "thin-cell limit does not recover the standard boundary");

    double const thick =
        STORM::ddmc::Densmore2006ConversionCoefficient(1.0e6, 0.9);
    requireClose(thick, 0.5257329470522999, 2.0e-6,
                 "large-tau coefficient does not approach Eq. (19)");

    double const pureScattering =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 1.0);
    requireClose(pureScattering,
                 8.0 / (45.0 + 6.0 * STORM::ddmc::ExtrapolationLength),
                 2.0e-15, "omega-to-one limit is discontinuous");
}

void testProbabilityAndReciprocity()
{
    double const invalid =
        STORM::ddmc::Densmore2006CellCoefficient(5.0, 0.5, 1.5);
    require(!std::isfinite(invalid),
            "coefficient above the paper's four-fifths bound was accepted");

    double const coefficient =
        STORM::ddmc::Densmore2006ConversionCoefficient(15.0, 0.99);
    double const normalAdmission =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, coefficient);
    require(normalAdmission >= 0.0 && normalAdmission <= 1.0,
            "angular admission probability left the unit interval");

    constexpr int bins = 10000;
    double average = 0.0;
    for(int i = 0; i < bins; ++i)
    {
        double const mu = (static_cast<double>(i) + 0.5) / bins;
        average += 2.0 * mu *
            STORM::ddmc::Densmore2006AdmissionProbability(mu, coefficient) /
            bins;
    }
    requireClose(average, coefficient, 2.0e-9,
                 "angular admission is not reciprocal with boundary leakage");
    requireClose(STORM::ddmc::Densmore2006BoundaryLeakRate(
                     2.0, 5.0, 7.0, coefficient),
                 7.0 * 2.0 * coefficient / (4.0 * 5.0), 2.0e-15,
                 "boundary leak rate does not match Eq. (29)");
}

void testCellMappingFallbackAndRoundoff()
{
    double const albedo =
        STORM::ddmc::Densmore2006SingleScatterAlbedo(5.0, 0.4);
    requireClose(albedo, 0.92, 2.0e-15,
                 "Fleck effective absorption is not mapped into albedo");

    double const fromCell =
        STORM::ddmc::Densmore2006CellCoefficient(5.0, albedo, 1.5);
    requireClose(fromCell,
                 STORM::ddmc::Densmore2006ConversionCoefficient(15.0, albedo),
                 2.0e-15,
                 "cell mapping does not use twice the center-face distance");

    double const atOne =
        STORM::ddmc::Densmore2006AdmissionProbability(1.0, fromCell);
    requireClose(STORM::ddmc::Densmore2006AdmissionProbability(
                     std::nextafter(1.0, 2.0), fromCell),
                 atOne, 0.0,
                 "roundoff above unit cosine changes interface admission");
}

} // namespace

int main()
{
    try
    {
        testCoefficient();
        testProbabilityAndReciprocity();
        testCellMappingFallbackAndRoundoff();
    }
    catch(std::exception const &error)
    {
        std::cerr << "storm_ddmc_interface_test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
