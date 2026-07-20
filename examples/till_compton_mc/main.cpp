/*
 * Till, McGraw & Warsa Compton equilibration case.
 *
 * This is the STORM equivalent of regression_tests/cases/till_compton_mc.
 * It uses the direct STORM IMC driver, the original RICH free-free opacity
 * formula, and an adapter around STORM's CMMC matrix generator.
 *
 * The default settings intentionally follow the old case:
 *   one cell, 32 groups, T_mat = 1 keV, T_rad = 10 keV,
 *   10,000 newly emitted packets per step, 4,000 initial packets,
 *   2,000,000 samples per CMMC matrix, and t_final = 3e-8 s.
 *
 * Useful overrides include --matrix-samples, --new-photons,
 * --initial-photons, --tf, --dt, --output-dir, and the plasma-cutoff
 * switches.  See README.md in this directory.
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "examples/Vector3D.hpp"
#if __has_include("MadCart/CartesianMesh3D.hpp")
#include "MadCart/CartesianMesh3D.hpp"
#elif __has_include("cartesian/CartesianMesh3D.hpp")
#include "cartesian/CartesianMesh3D.hpp"
#else
#error "Till-Compton example requires the MadCart CartesianMesh3D header"
#endif
#include "boundary/RigidBoundary.hpp"
#include "manager/MonteCarloManagerSerial.hpp"
#include "population/CombPopulationControl.hpp"
#include "radiation/RadiationIMC.hpp"
#include <units/units.hpp>

#include "TillComptonOpacity.hpp"

#ifndef STORM_DATA_DIR
#define STORM_DATA_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::size_t groups = 32;
constexpr double protonMass = 1.6726231e-24;
using Grid = MadCart::CartesianMesh3D<Vector3D>;

struct TillCell
{
    std::size_t ID = 0;
    double density = 0.0;
    double temperature = 0.0;
    double internal_energy = 0.0; // specific internal energy [erg/g]
    double Erad = 0.0;            // radiation energy per mass [erg/g]
    std::array<double, groups> Eg{};
};

struct TillExtensives
{
    double internal_energy = 0.0; // total material energy [erg]
    double mass = 0.0;            // total mass [g]
    double Erad = 0.0;            // total radiation energy [erg]
    std::array<double, groups> Eg{};
};

class TillEOS
{
public:
    TillEOS()
        : cvPerMass_(1.3 * 3.0 * units::k_boltz / protonMass)
    {}

    double dT2cv(double, double, const std::vector<double> &,
                 const std::vector<std::string> &) const
    {
        return cvPerMass_;
    }

    double de2T(double, double specificEnergy, const std::vector<double> &,
                const std::vector<std::string> &) const
    {
        return specificEnergy / cvPerMass_;
    }

private:
    double cvPerMass_;
};

struct RuntimeOptions
{
    std::optional<double> forcedDt;
    bool includePlasmaCutoff = true;
    double tf = 3e-8;
    std::size_t newPhotonsPerCell = 10000;
    std::size_t initialPhotonsPerCell = 4000;
    std::size_t matrixSamples = 2000000;
    fs::path outputDir = ".";
    bool help = false;
};

template<typename T>
std::optional<T> parsePositive(std::string_view value);

template<>
std::optional<double> parsePositive<double>(std::string_view value)
{
    double parsed = 0.0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if(result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
       !(parsed > 0.0) || !std::isfinite(parsed))
    {
        return {};
    }
    return parsed;
}

template<>
std::optional<std::size_t> parsePositive<std::size_t>(std::string_view value)
{
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if(result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0)
    {
        return {};
    }
    return parsed;
}

std::string shellQuote(const std::string &value)
{
    std::string quoted = "'";
    for(char character : value)
    {
        if(character == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

void printUsage(const char *program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "  --dt=SECONDS | --dt SECONDS   fixed timestep\n"
              << "  --tf=SECONDS                   final time (default 3e-8)\n"
              << "  --new-photons=N                emitted packets per cell/step\n"
              << "  --initial-photons=N            initial packets per cell\n"
              << "  --matrix-samples=N             CMMC samples per matrix\n"
              << "  --output-dir=PATH              profile/plot output directory\n"
              << "  --plasma-cutoff / --no-plasma-cutoff\n"
              << "  --help\n";
}

bool consumeValue(std::string_view argument,
                  std::string_view prefix,
                  std::string_view &value)
{
    if(argument.size() <= prefix.size() || argument.substr(0, prefix.size()) != prefix)
    {
        return false;
    }
    value = argument.substr(prefix.size());
    return true;
}

RuntimeOptions parseOptions(int argc, char **argv)
{
    RuntimeOptions options;
    for(int index = 1; index < argc; ++index)
    {
        const std::string_view argument = argv[index];
        if(argument == "--help" || argument == "-h")
        {
            options.help = true;
            continue;
        }
        if(argument == "--plasma-cutoff")
        {
            options.includePlasmaCutoff = true;
            continue;
        }
        if(argument == "--no-plasma-cutoff")
        {
            options.includePlasmaCutoff = false;
            continue;
        }

        std::string_view value;
        if(consumeValue(argument, "--dt=", value))
        {
            options.forcedDt = parsePositive<double>(value);
        }
        else if(argument == "--dt" && index + 1 < argc)
        {
            options.forcedDt = parsePositive<double>(argv[++index]);
        }
        else if(consumeValue(argument, "--tf=", value))
        {
            if(const auto parsed = parsePositive<double>(value)) options.tf = *parsed;
        }
        else if(consumeValue(argument, "--new-photons=", value))
        {
            if(const auto parsed = parsePositive<std::size_t>(value)) options.newPhotonsPerCell = *parsed;
        }
        else if(consumeValue(argument, "--initial-photons=", value))
        {
            if(value == "0")
            {
                options.initialPhotonsPerCell = 0;
            }
            else if(const auto parsed = parsePositive<std::size_t>(value))
            {
                options.initialPhotonsPerCell = *parsed;
            }
        }
        else if(consumeValue(argument, "--matrix-samples=", value))
        {
            if(const auto parsed = parsePositive<std::size_t>(value)) options.matrixSamples = *parsed;
        }
        else if(consumeValue(argument, "--output-dir=", value))
        {
            if(!value.empty()) options.outputDir = std::string(value);
        }
        else if(const auto parsed = parsePositive<double>(argument))
        {
            // Preserve the old case's convenient bare-dt form.
            options.forcedDt = *parsed;
        }
        else
        {
            std::cerr << "Ignoring unrecognized argument: " << argument << '\n';
        }
    }
    return options;
}

double radiationTemperature(const TillCell &cell)
{
    const double radiationEnergyDensity = std::max(0.0, cell.Erad * cell.density);
    return std::pow(radiationEnergyDensity / units::arad, 0.25);
}

double totalEnergy(const TillExtensives &extensives)
{
    return extensives.internal_energy + extensives.Erad;
}

struct ReferencePoint
{
    double time = 0.0;
    double gasTemperature = 0.0;
    double radiationTemperature = 0.0;
};

std::vector<ReferencePoint> loadReference(const fs::path &path)
{
    std::vector<ReferencePoint> reference;
    std::ifstream input(path);
    std::string line;
    while(std::getline(input, line))
    {
        if(line.empty() || line.front() == '#') continue;
        std::istringstream row(line);
        ReferencePoint point;
        if(row >> point.time >> point.gasTemperature >> point.radiationTemperature)
        {
            reference.push_back(point);
        }
    }
    return reference;
}

double interpolate(const std::vector<double> &x,
                   const std::vector<double> &y,
                   double query)
{
    if(x.empty() || y.empty() || x.size() != y.size()) return std::numeric_limits<double>::quiet_NaN();
    if(query <= x.front()) return y.front();
    if(query >= x.back()) return y.back();
    const auto upper = std::upper_bound(x.begin(), x.end(), query);
    const std::size_t right = static_cast<std::size_t>(upper - x.begin());
    const std::size_t left = right - 1;
    const double fraction = (query - x[left]) / (x[right] - x[left]);
    return y[left] + fraction * (y[right] - y[left]);
}

bool compareProfile(const fs::path &referencePath,
                    const std::vector<double> &time,
                    const std::vector<double> &gasTemperature,
                    const std::vector<double> &radiationTemperatureValues,
                    const std::vector<double> &energy,
                    double initialEnergy)
{
    const std::vector<ReferencePoint> reference = loadReference(referencePath);
    if(reference.empty())
    {
        std::cout << "No reference data at " << referencePath << "; skipping comparison\n";
        return true;
    }

    double gasError = 0.0;
    double radiationError = 0.0;
    std::size_t compared = 0;
    for(const ReferencePoint &point : reference)
    {
        if(point.time > time.back()) break;
        const double gas = interpolate(time, gasTemperature, point.time);
        const double radiation = interpolate(time, radiationTemperatureValues, point.time);
        gasError += std::abs(gas - point.gasTemperature) /
                    std::max(std::abs(point.gasTemperature), 1.0);
        radiationError += std::abs(radiation - point.radiationTemperature) /
                          std::max(std::abs(point.radiationTemperature), 1.0);
        ++compared;
    }

    if(compared == 0)
    {
        std::cout << "Reference starts after the simulated interval; skipping comparison\n";
        return true;
    }

    gasError /= static_cast<double>(compared);
    radiationError /= static_cast<double>(compared);
    const double energyDrift = initialEnergy != 0.0
        ? (energy.back() - initialEnergy) /
            std::max(std::abs(initialEnergy), 1.0) : 0.0;

    std::cout << std::scientific
              << "TILL_COMPTON_TGAS_REL_L1 = " << gasError << '\n'
              << "TILL_COMPTON_TRAD_REL_L1 = " << radiationError << '\n'
              << "TILL_COMPTON_TOTAL_ENERGY_REL_DRIFT = " << energyDrift << '\n'
              << "Compared reference points = " << compared << '\n';
    constexpr double temperatureErrorLimit = 0.25;
    bool const accepted = gasError <= temperatureErrorLimit &&
        radiationError <= temperatureErrorLimit;
    std::cout << "TILL_COMPTON_REFERENCE_PASS = " << accepted << '\n';
    return accepted;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const RuntimeOptions options = parseOptions(argc, argv);
        if(options.help)
        {
            printUsage(argv[0]);
            return 0;
        }
        if(options.matrixSamples < 4)
        {
            throw std::runtime_error("--matrix-samples must be at least 4 for CMMC progress reporting");
        }
        if(options.forcedDt && *options.forcedDt > options.tf)
        {
            std::cerr << "The forced timestep is larger than the final time; using one truncated step.\n";
        }

        const double initialTemperature = units::kev_kelvin;
        const double radiationInitialTemperature = 10.0 * units::kev_kelvin;
        const double density = 1.0;
        const double cvPerMass = 1.3 * 3.0 * units::k_boltz / protonMass;
        const double initialRadiationPerMass = units::arad *
            std::pow(radiationInitialTemperature, 4) / density;

        std::array<double, groups + 1> energyBoundaries{};
        const double minimumEnergy = units::kev * 1e-4;
        const double maximumEnergy = units::kev * 1e3;
        energyBoundaries[0] = minimumEnergy;
        const double ratio = std::pow(maximumEnergy / minimumEnergy, 1.0 / groups);
        for(std::size_t group = 0; group < groups; ++group)
        {
            energyBoundaries[group + 1] = energyBoundaries[group] * ratio;
        }

        const Vector3D lower(0.0, -0.5, -0.5);
        const Vector3D upper(1.0, 0.5, 0.5);
        Grid grid(lower, upper, 1, 1, 1);
        const std::size_t cellCount = grid.GetPointNo();

        std::vector<TillCell> cells(cellCount);
        std::vector<TillExtensives> extensives(cellCount);
        for(std::size_t cellIndex = 0; cellIndex < cellCount; ++cellIndex)
        {
            TillCell &cell = cells[cellIndex];
            TillExtensives &extensive = extensives[cellIndex];
            cell.ID = cellIndex;
            cell.density = density;
            cell.temperature = initialTemperature;
            cell.internal_energy = cvPerMass * initialTemperature;
            cell.Erad = initialRadiationPerMass;
            extensive.mass = density * grid.GetVolume(cellIndex);
            extensive.internal_energy = cell.internal_energy * extensive.mass;
            extensive.Erad = cell.Erad * extensive.mass;
            for(std::size_t group = 0; group < groups; ++group)
            {
                const double groupEnergy =
                    planck_integral::planck_energy_density_group_integral(
                        energyBoundaries[group], energyBoundaries[group + 1],
                        radiationInitialTemperature) / density;
                cell.Eg[group] = std::max(groupEnergy, cell.Erad * 1e-8);
                extensive.Eg[group] = cell.Eg[group] * extensive.mass;
            }
        }

        using Opacity = STORM::examples::TillComptonOpacity<
            Vector3D, Grid, TillCell, groups>;
        using IMC = STORM::RadiationIMC<Vector3D, Grid, TillCell,
                                        TillExtensives, TillEOS, groups>;

        STORM::RadiationIMCParameters<groups> parameters;
        parameters.newPhotonsPerCell = options.newPhotonsPerCell;
        parameters.withMultigroupOpacity = true;
        parameters.withCompton = true;
        parameters.comptonUseInduced = true;
        parameters.comptonAllowNZeroFallback = true;
        parameters.comptonMatrixSamples = options.matrixSamples;
        parameters.comptonAngleDependent = true;
        parameters.withEgTimeAvg = true;
        parameters.energyBoundaries = energyBoundaries;
        parameters.energyBoundariesProvided = true;

        auto boundary = std::make_shared<STORM::RigidBoundary<Vector3D, Grid>>(grid);
        auto eos = std::make_shared<TillEOS>();
        auto opacity = std::make_shared<Opacity>(energyBoundaries,
                                                 options.includePlasmaCutoff);
        auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives,
                                             eos, opacity, parameters);
        auto populationControl = std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(
            grid, 200, 5.0);
        STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(
            grid, physics, populationControl, boundary);

        std::cout << "Running case: Till MC (STORM)\n"
                  << "T_mat = 1 keV, T_rad = 10 keV, Compton = ON, absorption = ON\n"
                  << "groups = " << groups
                  << ", matrix samples = " << options.matrixSamples
                  << ", new photons/cell = " << options.newPhotonsPerCell
                  << ", initial photons/cell = " << options.initialPhotonsPerCell << '\n'
                  << "plasma cutoff = " << (options.includePlasmaCutoff ? "ON" : "OFF")
                  << ", t_final = " << options.tf << " s\n";

        fs::create_directories(options.outputDir);
        const fs::path referencePath = fs::path(STORM_DATA_DIR) / "data" / "in_fbc_reference.txt";
        const fs::path profilePath = options.outputDir / "till_compton_profile.txt";
        const fs::path plotBasePath = options.outputDir / "till_compton_mc";

        std::vector<double> time{0.0};
        std::vector<double> gasTemperature{cells.front().temperature};
        std::vector<double> radiationTemperatureValues{radiationTemperature(cells.front())};
        std::vector<double> energy{totalEnergy(extensives.front())};
        const double initialEnergy = energy.front();

        std::vector<STORM::Particle<Vector3D, Grid>> particles =
            physics->generateInitialParticles(options.initialPhotonsPerCell);
        const double initialDt = 1e-13;
        double timestep = options.forcedDt.value_or(initialDt);
        while(time.back() < options.tf)
        {
            const double stepDt = std::min(timestep, options.tf - time.back());
            if(!(stepDt > 0.0)) break;
            particles = manager.step(std::move(particles), stepDt);
            time.push_back(time.back() + stepDt);
            gasTemperature.push_back(cells.front().temperature);
            radiationTemperatureValues.push_back(radiationTemperature(cells.front()));
            energy.push_back(totalEnergy(extensives.front()));
            std::cout << "time = " << std::scientific << time.back()
                      << ", particles = " << particles.size()
                      << ", Tgas = " << gasTemperature.back() / units::kev_kelvin
                      << " keV\n";
            timestep = options.forcedDt.value_or(std::min(timestep * 1.2, 1e-10));
        }

        {
            std::ofstream output(profilePath);
            if(!output)
            {
                throw std::runtime_error("Unable to open profile output: " + profilePath.string());
            }
            output << "# time[s] Tgas[K] Trad[K] Etotal[erg]\n";
            output << std::setprecision(17);
            for(std::size_t index = 0; index < time.size(); ++index)
            {
                output << time[index] << ' ' << gasTemperature[index] << ' '
                       << radiationTemperatureValues[index] << ' ' << energy[index] << '\n';
            }
        }
        std::cout << "Wrote profile: " << profilePath << '\n';

        bool const referencePass = compareProfile(
            referencePath,
            time,
            gasTemperature,
            radiationTemperatureValues,
            energy,
            initialEnergy);
        const std::size_t comptonEvents = physics->getComptonEventCount();
        const std::size_t angleEvents = physics->getComptonAngleEventCount();
        std::cout << "TILL_COMPTON_EVENTS = " << comptonEvents << '\n'
                  << "TILL_COMPTON_ANGLE_EVENTS = " << angleEvents << '\n'
                  << "TILL_COMPTON_INDUCED_EXERCISED = "
                  << physics->getComptonInducedTermsExercised() << '\n';
        if(!referencePass)
        {
            throw std::runtime_error(
                "Till Compton temperature curve exceeded the reference tolerance");
        }
        bool const requireTransportCoverage = options.tf >= 1e-10;
        if(requireTransportCoverage &&
           (comptonEvents == 0 || angleEvents == 0 ||
            !physics->getComptonInducedTermsExercised()))
        {
            throw std::runtime_error(
                "Till Compton regression did not exercise the required transport paths");
        }
        if(!requireTransportCoverage)
        {
            std::cout << "Short smoke interval: event-coverage threshold skipped\n";
        }
        double const totalEnergyDrift = initialEnergy != 0.0
            ? (energy.back() - initialEnergy) /
                std::max(std::abs(initialEnergy), 1.0) : 0.0;
        if(!std::isfinite(totalEnergyDrift) ||
           std::abs(totalEnergyDrift) > 1e-8)
        {
            throw std::runtime_error(
                "Till Compton total-energy drift exceeded the strict threshold");
        }

        const fs::path plotScript = fs::path(STORM_DATA_DIR) / "plot_till_compton.py";
        const std::string plotCommand =
            "python3 " + shellQuote(plotScript.string()) +
            " --profile " + shellQuote(profilePath.string()) +
            " --reference " + shellQuote(referencePath.string()) +
            " --output " + shellQuote(plotBasePath.string());
        std::cout << "Generating profile comparison: " << plotCommand << '\n';
        const int plotStatus = std::system(plotCommand.c_str());
        if(plotStatus != 0)
        {
            std::cerr << "Warning: profile comparison plot failed with status " << plotStatus << '\n';
        }
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cerr << "Till Compton example failed: " << error.what() << '\n';
        return 1;
    }
}
