#include "radiation/compton/MACEComptonBackend.hpp"
#include "radiation/compton/ComptonDetailedBalance.hpp"

#include "compton_table/compton_table.hpp"
#include "utilities/units.hpp"

#ifdef STORM_WITH_MPI
#include "compton_table/compton_table_mpi.hpp"
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace STORM {

struct MACEComptonBackendCore::Impl
{
    std::size_t groups = 0;
    std::size_t angles = 0;
    std::vector<double> centers;
    std::vector<double> boundaries;
    std::vector<double> temperatures;
    std::vector<double> logTemperatures;
    // Per table temperature, summed over angle bins: [i][g][g'].
    std::vector<double> sigma;
    std::vector<double> dsigmaDT;
    std::vector<double> logSigma;
    std::vector<double> logSlope; // d ln sigma / d ln T at the nodes
    // Energy-weighted total scattering rate per table temperature, summed over angle bins: [i][g], and its
    // angle CDF [i][g][0..angles].
    std::vector<double> total;
    std::vector<double> logTotal;
    std::vector<double> logTotalSlope;
    std::vector<double> totalCdf;
    // Angle CDFs [i][g][g'][0..angles]; the diagonal holds the direction-only remainder.
    std::vector<double> angleCdf;
    mutable std::atomic<bool> warnedClamp{false};
    mutable std::atomic<bool> warnedRemainder{false};
    bool reportRank = true;

    std::size_t pair(std::size_t i, std::size_t g, std::size_t gp) const
    {
        return (i * groups + g) * groups + gp;
    }

    std::size_t row(std::size_t i, std::size_t g) const
    {
        return i * groups + g;
    }

    // Acceptance tolerance for the remainder total - sum_{t != g} tau_gt: below -remainderTolerance * total no
    // nonnegative diagonal keeps the total with these off-diagonal rates, and the run stops; smaller negative values
    // are taken as round-off and clipped to zero.  (The production table never comes near it: its smallest
    // remainder is +6.5% of the total.)
    static constexpr double remainderTolerance = 1e-6;

    struct RemainderReport
    {
        std::size_t clipped = 0;  // negative within the tolerance, set to zero
        std::size_t material = 0; // beyond the tolerance (also set to zero)
        double worstFraction = 1.0;
        std::size_t worstGroup = 0;
    };

    // Per-electron rates at T (inside the table): interpolated, detailed balance enforced, and the diagonal set
    // to the direction-only remainder total - sum_{t != g} tau_gt (negative remainders set to zero and counted
    // in report); totals, if given, receives the interpolated total.
    ComptonMatrixData Interpolate(double T,
                                  bool derivativesValid,
                                  RemainderReport &report,
                                  std::vector<double> *totals = nullptr) const;
};

namespace {

int mpiSize()
{
#ifdef STORM_WITH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized)
    {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
#endif
    return 1;
}

compton::table::Table loadTable(const compton::table::TableSpec &spec, const std::string &cacheDir, bool &reportRank)
{
#ifdef STORM_WITH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        reportRank = rank == 0;
        return compton::table::load_or_build_mpi(spec, cacheDir, MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr);
    }
#endif
    // Without MPI: the library's serial load-or-build, with the same lock file as the MPI path.
    return compton::table::load_or_build(spec, cacheDir, &std::cout);
}

// Cubic Hermite in (ln T, ln y) with the tabulated slopes d ln y / d ln T, linear in y where a node vanishes.
struct HermiteBasis
{
    double h = 0.0;
    double T = 0.0;
    double h00 = 0.0, h10 = 0.0, h01 = 0.0, h11 = 0.0;
    double d00 = 0.0, d10 = 0.0, d01 = 0.0, d11 = 0.0;
};

HermiteBasis MakeHermiteBasis(double h, double u, double T)
{
    HermiteBasis b;
    double const u2 = u * u;
    double const u3 = u2 * u;
    b.h = h;
    b.T = T;
    b.h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    b.h10 = u3 - 2.0 * u2 + u;
    b.h01 = -2.0 * u3 + 3.0 * u2;
    b.h11 = u3 - u2;
    b.d00 = 6.0 * u2 - 6.0 * u;
    b.d10 = 3.0 * u2 - 4.0 * u + 1.0;
    b.d01 = -6.0 * u2 + 6.0 * u;
    b.d11 = 3.0 * u2 - 2.0 * u;
    return b;
}

void HermiteInterpolate(const HermiteBasis &b,
                        double s0,
                        double s1,
                        double logS0,
                        double logS1,
                        double logSlope0,
                        double logSlope1,
                        double u,
                        double &value,
                        double &slope)
{
    if(s0 > 0.0 && s1 > 0.0)
    {
        double const m0 = logSlope0 * b.h;
        double const m1 = logSlope1 * b.h;
        double const y = b.h00 * logS0 + b.h10 * m0 + b.h01 * logS1 + b.h11 * m1;
        double const dyDu = b.d00 * logS0 + b.d10 * m0 + b.d01 * logS1 + b.d11 * m1;
        value = std::exp(y);
        slope = value * dyDu / (b.h * b.T);
    }
    else
    {
        // A node where the entry vanishes: linear in the value.
        value = (1.0 - u) * s0 + u * s1;
        slope = (s1 - s0) / (b.h * b.T);
    }
}

} // namespace

ComptonMatrixData MACEComptonBackendCore::Impl::Interpolate(double T,
                                                           bool derivativesValid,
                                                           RemainderReport &report,
                                                           std::vector<double> *totals) const
{
    std::size_t const G = groups;
    std::size_t const NT = temperatures.size();
    std::size_t const upper = static_cast<std::size_t>(
        std::upper_bound(temperatures.begin(), temperatures.end(), T) - temperatures.begin());
    std::size_t const i = std::min(upper == 0 ? 0 : upper - 1, NT - 2);
    double const h = logTemperatures[i + 1] - logTemperatures[i];
    double const u = std::clamp((std::log(T) - logTemperatures[i]) / h, 0.0, 1.0);
    HermiteBasis const basis = MakeHermiteBasis(h, u, T);

    ComptonMatrixData result;
    result.tau.assign(G, std::vector<double>(G, 0.0));
    result.dtau_dUm.assign(G, std::vector<double>(G, 0.0));
    for(std::size_t g = 0; g < G; ++g)
    {
        for(std::size_t gp = 0; gp < G; ++gp)
        {
            std::size_t const p0 = pair(i, g, gp);
            std::size_t const p1 = pair(i + 1, g, gp);
            double value = 0.0;
            double slope = 0.0;
            HermiteInterpolate(basis, sigma[p0], sigma[p1], logSigma[p0], logSigma[p1], logSlope[p0], logSlope[p1], u,
                               value, slope);
            result.tau[g][gp] = value;
            result.dtau_dUm[g][gp] = derivativesValid ? slope : 0.0;
        }
    }

    // Detailed balance on the interpolated rates and their derivatives (as CMMC, plus the derivative).
    EnforceComptonDetailedBalance(centers, boundaries, T, derivativesValid, true, result);

    // Direction-only rate: the energy-weighted total minus the balanced effective off-diagonal rates, so that the
    // total direction-changing rate (before occupation factors and the pair throttle) is the table's total.
    report = RemainderReport{};
    if(totals != nullptr)
    {
        totals->assign(G, 0.0);
    }
    for(std::size_t g = 0; g < G; ++g)
    {
        std::size_t const r0 = row(i, g);
        std::size_t const r1 = row(i + 1, g);
        double totalValue = 0.0;
        double totalSlope = 0.0;
        HermiteInterpolate(basis, total[r0], total[r1], logTotal[r0], logTotal[r1], logTotalSlope[r0],
                           logTotalSlope[r1], u, totalValue, totalSlope);
        if(totals != nullptr)
        {
            (*totals)[g] = totalValue;
        }
        double remainder = totalValue;
        double remainderSlope = totalSlope;
        for(std::size_t t = 0; t < G; ++t)
        {
            if(t != g)
            {
                remainder -= result.tau[g][t];
                remainderSlope -= result.dtau_dUm[g][t];
            }
        }
        if(totalValue > 0.0 && remainder / totalValue < report.worstFraction)
        {
            report.worstFraction = remainder / totalValue;
            report.worstGroup = g;
        }
        if(remainder < 0.0)
        {
            if(remainder < -remainderTolerance * totalValue)
            {
                ++report.material;
            }
            else
            {
                ++report.clipped;
            }
            remainder = 0.0;
            remainderSlope = 0.0;
        }
        result.tau[g][g] = remainder;
        result.dtau_dUm[g][g] = derivativesValid ? remainderSlope : 0.0;
    }
    return result;
}

MACEComptonBackendCore::MACEComptonBackendCore(std::vector<double> centers,
                                               std::vector<double> boundaries,
                                               const MACEComptonTableOptions &options):
    impl_(std::make_unique<Impl>())
{
    Impl &impl = *this->impl_;
    compton::table::TableSpec spec;
    spec.boundaries = boundaries;
    spec.centers = centers;
    spec.temperatures = compton::table::default_temperatures();
    spec.num_angle_bins = static_cast<int>(options.angleBins);
    spec.solver = options.exactSolver ? compton::table::SolverKind::exact : compton::table::SolverKind::approximate;

#ifdef _OPENMP
    int const savedThreads = omp_get_max_threads();
    int const size = mpiSize();
    if(options.threads > 0)
    {
        omp_set_num_threads(options.threads);
    }
    else if(size > 1)
    {
        omp_set_num_threads(1);
    }
#endif
    std::string cacheDir = options.cacheDir;
    if(cacheDir.empty())
    {
        if(char const *environment = std::getenv("STORM_COMPTON_TABLE_DIR"))
        {
            cacheDir = environment;
        }
    }
    compton::table::Table table = loadTable(spec, cacheDir, impl.reportRank);
#ifdef _OPENMP
    omp_set_num_threads(savedThreads);
#endif

    impl.groups = centers.size();
    impl.angles = options.angleBins;
    impl.centers = std::move(centers);
    impl.boundaries = std::move(boundaries);
    impl.temperatures = table.spec.temperatures;
    std::size_t const G = impl.groups;
    std::size_t const A = impl.angles;
    std::size_t const NT = impl.temperatures.size();
    if(NT < 2 || table.sigma.size() != NT * G * G * A)
    {
        throw std::runtime_error("MACE Compton table has the wrong size for this grid");
    }
    impl.logTemperatures.resize(NT);
    for(std::size_t i = 0; i < NT; ++i)
    {
        impl.logTemperatures[i] = std::log(impl.temperatures[i]);
    }
    impl.sigma.assign(NT * G * G, 0.0);
    impl.dsigmaDT.assign(NT * G * G, 0.0);
    impl.logSigma.assign(NT * G * G, 0.0);
    impl.logSlope.assign(NT * G * G, 0.0);
    impl.angleCdf.assign(NT * G * G * (A + 1), 0.0);
    for(std::size_t i = 0; i < NT; ++i)
    {
        for(std::size_t g = 0; g < G; ++g)
        {
            for(std::size_t gp = 0; gp < G; ++gp)
            {
                std::size_t const p = impl.pair(i, g, gp);
                double total = 0.0;
                double derivative = 0.0;
                double *cdf = impl.angleCdf.data() + p * (A + 1);
                cdf[0] = 0.0;
                for(std::size_t a = 0; a < A; ++a)
                {
                    std::size_t const index = table.index(i, g, gp, a);
                    total += table.sigma[index];
                    derivative += table.dsigma_dT[index];
                    cdf[a + 1] = total;
                }
                for(std::size_t b = 0; b <= A; ++b)
                {
                    cdf[b] = total > 0.0 ? cdf[b] / total : static_cast<double>(b) / static_cast<double>(A);
                }
                cdf[A] = 1.0;
                impl.sigma[p] = total;
                impl.dsigmaDT[p] = derivative;
                if(total > 0.0)
                {
                    impl.logSigma[p] = std::log(total);
                    impl.logSlope[p] = impl.temperatures[i] * derivative / total;
                }
            }
        }
    }

    // Energy-weighted totals and, per table temperature, the angle distribution of the direction-only remainder
    // total_a - sum_{t != g} q_gt p_gt,a with the balanced node rates q_gt and the off-diagonal angle bins p_gt,a
    // (what the off-diagonal events sample); negative bins are projected away (sum kept).
    impl.total.assign(NT * G, 0.0);
    impl.logTotal.assign(NT * G, 0.0);
    impl.logTotalSlope.assign(NT * G, 0.0);
    impl.totalCdf.assign(NT * G * (A + 1), 0.0);
    std::vector<double> bins(A, 0.0);
    for(std::size_t i = 0; i < NT; ++i)
    {
        ComptonMatrixData node;
        node.tau.assign(G, std::vector<double>(G, 0.0));
        node.dtau_dUm.assign(G, std::vector<double>(G, 0.0));
        for(std::size_t g = 0; g < G; ++g)
        {
            for(std::size_t gp = 0; gp < G; ++gp)
            {
                node.tau[g][gp] = impl.sigma[impl.pair(i, g, gp)];
            }
        }
        EnforceComptonDetailedBalance(impl.centers, impl.boundaries, impl.temperatures[i], false, true, node);
        for(std::size_t g = 0; g < G; ++g)
        {
            std::size_t const r = impl.row(i, g);
            double total = 0.0;
            double derivative = 0.0;
            for(std::size_t a = 0; a < A; ++a)
            {
                std::size_t const index = table.total_index(i, g, a);
                bins[a] = table.total[index];
                total += table.total[index];
                derivative += table.dtotal_dT[index];
            }
            impl.total[r] = total;
            if(total > 0.0)
            {
                impl.logTotal[r] = std::log(total);
                impl.logTotalSlope[r] = impl.temperatures[i] * derivative / total;
            }
            double *totalCdf = impl.totalCdf.data() + r * (A + 1);
            totalCdf[0] = 0.0;
            for(std::size_t a = 0; a < A; ++a)
            {
                totalCdf[a + 1] = totalCdf[a] + bins[a];
            }
            for(std::size_t b = 0; b <= A; ++b)
            {
                totalCdf[b] = total > 0.0 ? totalCdf[b] / total : static_cast<double>(b) / static_cast<double>(A);
            }
            totalCdf[A] = 1.0;
            for(std::size_t t = 0; t < G; ++t)
            {
                if(t == g)
                {
                    continue;
                }
                const double *cdf = impl.angleCdf.data() + impl.pair(i, g, t) * (A + 1);
                for(std::size_t a = 0; a < A; ++a)
                {
                    bins[a] -= node.tau[g][t] * (cdf[a + 1] - cdf[a]);
                }
            }
            double remainder = 0.0;
            double positive = 0.0;
            for(std::size_t a = 0; a < A; ++a)
            {
                remainder += bins[a];
                positive += std::max(0.0, bins[a]);
            }
            double *cdf = impl.angleCdf.data() + impl.pair(i, g, g) * (A + 1);
            cdf[0] = 0.0;
            double running = 0.0;
            for(std::size_t a = 0; a < A; ++a)
            {
                // Without a positive remainder there are no direction-only events: keep the total's shape.
                running += remainder > 0.0 && positive > 0.0 ? std::max(0.0, bins[a]) : table.total[table.total_index(i, g, a)];
                cdf[a + 1] = running;
            }
            for(std::size_t b = 0; b <= A; ++b)
            {
                cdf[b] = running > 0.0 ? cdf[b] / running : static_cast<double>(b) / static_cast<double>(A);
            }
            cdf[A] = 1.0;
        }
    }

    // Audit at the nodes and 15 temperatures between each pair: the scalar remainder after interpolation and
    // balance, and the angular closure of all scatterings, sum_a |R d_a - R_a| / total, with R_a the exact
    // remainder per bin (total and off-diagonal angle bins interpolated linearly in T, as the events sample them)
    // and d_a the direction-only CDF used at run time; at the nodes this is the projection of negative bins.
    double worstFraction = 1.0;
    double worstFractionTemperature = 0.0;
    std::size_t worstFractionGroup = 0;
    std::size_t clipped = 0;
    std::size_t material = 0;
    double worstAngular = 0.0;
    double worstMoment = 0.0;
    double worstAngularTemperature = 0.0;
    std::size_t worstAngularGroup = 0;
    std::size_t samples = 0;
    std::vector<double> totals;
    std::vector<double> exact(A, 0.0);
    double const binWidth = 2.0 / static_cast<double>(A);
    for(std::size_t i = 0; i + 1 < NT; ++i)
    {
        for(int k = (i == 0 ? 0 : 1); k <= 16; ++k)
        {
            double const u = static_cast<double>(k) / 16.0;
            double const T = std::exp((1.0 - u) * impl.logTemperatures[i] + u * impl.logTemperatures[i + 1]);
            Impl::RemainderReport report;
            ComptonMatrixData const rates = impl.Interpolate(T, true, report, &totals);
            ++samples;
            clipped += report.clipped;
            material += report.material;
            if(report.worstFraction < worstFraction)
            {
                worstFraction = report.worstFraction;
                worstFractionTemperature = T;
                worstFractionGroup = report.worstGroup;
            }
            double const x = std::clamp(
                (T - impl.temperatures[i]) / (impl.temperatures[i + 1] - impl.temperatures[i]), 0.0, 1.0);
            auto bin = [&](const std::vector<double> &cdfs, std::size_t offsetLow, std::size_t offsetHigh, std::size_t a)
            {
                return (1.0 - x) * (cdfs[offsetLow + a + 1] - cdfs[offsetLow + a]) +
                       x * (cdfs[offsetHigh + a + 1] - cdfs[offsetHigh + a]);
            };
            for(std::size_t g = 0; g < G; ++g)
            {
                double const remainder = rates.tau[g][g];
                if(!(totals[g] > 0.0) || !(remainder > 0.0))
                {
                    continue;
                }
                for(std::size_t a = 0; a < A; ++a)
                {
                    exact[a] = totals[g] * bin(impl.totalCdf, impl.row(i, g) * (A + 1), impl.row(i + 1, g) * (A + 1), a);
                }
                for(std::size_t t = 0; t < G; ++t)
                {
                    if(t == g)
                    {
                        continue;
                    }
                    for(std::size_t a = 0; a < A; ++a)
                    {
                        exact[a] -= rates.tau[g][t] * bin(impl.angleCdf, impl.pair(i, g, t) * (A + 1),
                                                          impl.pair(i + 1, g, t) * (A + 1), a);
                    }
                }
                double distance = 0.0;
                double moment = 0.0;
                for(std::size_t a = 0; a < A; ++a)
                {
                    double const used = remainder * bin(impl.angleCdf, impl.pair(i, g, g) * (A + 1),
                                                        impl.pair(i + 1, g, g) * (A + 1), a);
                    distance += std::abs(used - exact[a]);
                    moment += (-1.0 + (static_cast<double>(a) + 0.5) * binWidth) * (used - exact[a]);
                }
                worstMoment = std::max(worstMoment, std::abs(moment) / totals[g]);
                if(distance / totals[g] > worstAngular)
                {
                    worstAngular = distance / totals[g];
                    worstAngularTemperature = T;
                    worstAngularGroup = g;
                }
            }
        }
    }
    if(impl.reportRank)
    {
        std::cout << "[mace] direction-only rate = energy-weighted total - balanced off-diagonal rates, checked at "
                  << samples << " temperatures (the nodes and 15 between each pair): remainder/total >= "
                  << worstFraction << " (T = " << worstFractionTemperature << " K, group " << worstFractionGroup
                  << "), " << clipped << " rows clipped within " << Impl::remainderTolerance << ", " << material
                  << " beyond; angular closure sum_a |R d_a - R_a| / total <= " << worstAngular << " (T = "
                  << worstAngularTemperature << " K, group " << worstAngularGroup << "), first-moment error <= "
                  << worstMoment << " of the total" << std::endl;
    }
    if(material > 0)
    {
        throw std::runtime_error("MACE Compton table: the balanced off-diagonal rates exceed the energy-weighted "
                                 "total by more than the tolerance in " + std::to_string(material) +
                                 " sampled rows (worst remainder/total " + std::to_string(worstFraction) +
                                 "); the direction-only rate cannot keep the total on this grid");
    }
}

MACEComptonBackendCore::~MACEComptonBackendCore() = default;

ComptonMatrixData MACEComptonBackendCore::GetTauMatrix(double temperature,
                                                       double density,
                                                       double atomicWeight,
                                                       double ionization) const
{
    const Impl &impl = *this->impl_;
    std::size_t const G = impl.groups;
    double const T = std::clamp(temperature, impl.temperatures.front(), impl.temperatures.back());
    bool const clamped = T != temperature;
    if(clamped && impl.reportRank && !impl.warnedClamp.exchange(true))
    {
        std::cerr << "[mace] WARNING: temperature " << temperature << " K outside the Compton table ["
                  << impl.temperatures.front() << ", " << impl.temperatures.back()
                  << "] K; clamped (reported once)" << std::endl;
    }
    Impl::RemainderReport report;
    ComptonMatrixData result = impl.Interpolate(T, !clamped, report);
    if(report.material > 0)
    {
        std::string const message =
            "MACE Compton table: at T = " + std::to_string(T) + " K the balanced off-diagonal rates of group " +
            std::to_string(report.worstGroup) + " exceed the energy-weighted total (remainder/total " +
            std::to_string(report.worstFraction) + ", tolerance " + std::to_string(Impl::remainderTolerance) + ")";
        std::cerr << "[mace] ERROR: " << message << std::endl;
        throw std::runtime_error(message);
    }
    if(report.clipped > 0 && impl.reportRank && !impl.warnedRemainder.exchange(true))
    {
        std::cerr << "[mace] note: direction-only rate clipped to zero within the tolerance in " << report.clipped
                  << " groups at T = " << T << " K (worst group " << report.worstGroup << ", remainder/total "
                  << report.worstFraction << "; reported once)" << std::endl;
    }

    double const electrons = density * units::Navogadro / atomicWeight * ionization;
    for(std::size_t g = 0; g < G; ++g)
    {
        for(std::size_t gp = 0; gp < G; ++gp)
        {
            result.tau[g][gp] *= electrons;
            result.dtau_dUm[g][gp] *= electrons;
        }
    }
    return result;
}

double MACEComptonBackendCore::GetMinimumTemperature() const
{
    return this->impl_->temperatures.front();
}

double MACEComptonBackendCore::GetMaximumTemperature() const
{
    return this->impl_->temperatures.back();
}

void MACEComptonBackendCore::GetAngleCdf(double temperature,
                                         std::size_t sourceGroup,
                                         std::size_t targetGroup,
                                         std::vector<double> &cdf) const
{
    // Linear in T between the bracketing table temperatures, as CMMC.
    const Impl &impl = *this->impl_;
    std::size_t const A = impl.angles;
    std::size_t const NT = impl.temperatures.size();
    cdf.resize(A + 1);
    std::size_t const upper = static_cast<std::size_t>(
        std::lower_bound(impl.temperatures.begin(), impl.temperatures.end(), temperature) -
        impl.temperatures.begin());
    std::size_t const i = std::min(upper == 0 ? 0 : upper - 1, NT - 2);
    double const x = std::clamp(
        (temperature - impl.temperatures[i]) / (impl.temperatures[i + 1] - impl.temperatures[i]), 0.0, 1.0);
    const double *low = impl.angleCdf.data() + impl.pair(i, sourceGroup, targetGroup) * (A + 1);
    const double *high = impl.angleCdf.data() + impl.pair(i + 1, sourceGroup, targetGroup) * (A + 1);
    for(std::size_t b = 0; b <= A; ++b)
    {
        cdf[b] = (1.0 - x) * low[b] + x * high[b];
    }
    cdf[0] = 0.0;
    cdf[A] = 1.0;
}

std::size_t MACEComptonBackendCore::GetAngleBinCount() const
{
    return this->impl_->angles;
}

} // namespace STORM
