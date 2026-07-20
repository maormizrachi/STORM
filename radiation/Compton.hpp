#ifndef STORM_RADIATION_COMPTON_HPP
#define STORM_RADIATION_COMPTON_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <functional>
#include <stdexcept>
#include <utility>

namespace STORM {

/*
 * These result and model types are retained as an advanced test seam.  The
 * normal RadiationIMC workflow owns and lazily initializes its CMMC backend;
 * applications do not provide a matrix builder.
 *
 * Rates are macroscopic inverse lengths.  If supplied, fleckCorrection is a
 * kernel-derived multiplicative Fleck correction in (0, 1].  residualSource is an extensive
 * signed energy correction for the current source interval.  A positive
 * entry creates radiation and a negative entry creates a negative-weight
 * packet.  The transport class accounts for the equal-and-opposite material
 * ledger change when it creates those packets.
 */
template<std::size_t NumGroups>
struct ComptonKernelResult
{
    using GroupArray = std::array<double, NumGroups>;
    using GroupMatrix = std::array<GroupArray, NumGroups>;

    GroupMatrix rates{};
    GroupMatrix derivative{};
    GroupArray fleckCorrection{};
    GroupArray residualSource{};
    bool hasFleckCorrection = false;
    bool hasResidualSource = false;
};

template<typename PointT,
         typename GridT,
         typename CellT,
         std::size_t NumGroups>
class ComptonKernelModel
{
public:
    using Result = ComptonKernelResult<NumGroups>;
    using GroupBoundaries = std::array<double, NumGroups + 1>;
    using GroupArray = std::array<double, NumGroups>;

    virtual ~ComptonKernelModel() = default;

    virtual Result build(const CellT &cell,
                         double density,
                         double temperature,
                         const GroupBoundaries &boundaries,
                         const GroupArray &centers,
                         double sourceDt,
                         std::size_t matrixSamples,
                         std::mt19937_64 &rng) const = 0;
};

// Advanced adapter for isolated kernel tests and alternate transport
// experiments.  It is not required by RadiationIMC's normal Compton path.
template<typename PointT,
         typename GridT,
         typename CellT,
         std::size_t NumGroups>
class FunctionalComptonKernel final
    : public ComptonKernelModel<PointT, GridT, CellT, NumGroups>
{
public:
    using Base = ComptonKernelModel<PointT, GridT, CellT, NumGroups>;
    using Result = typename Base::Result;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupArray = typename Base::GroupArray;
    using Builder = std::function<Result(const CellT&, double, double,
                                         const GroupBoundaries&, const GroupArray&,
                                         double, std::size_t, std::mt19937_64&)>;

    explicit FunctionalComptonKernel(Builder builder): builder_(std::move(builder))
    {
        if(!builder_)
        {
            throw std::invalid_argument("FunctionalComptonKernel requires a builder");
        }
    }

    Result build(const CellT &cell,
                 double density,
                 double temperature,
                 const GroupBoundaries &boundaries,
                 const GroupArray &centers,
                 double sourceDt,
                 std::size_t matrixSamples,
                 std::mt19937_64 &rng) const override
    {
        return builder_(cell, density, temperature, boundaries, centers,
                        sourceDt, matrixSamples, rng);
    }

private:
    Builder builder_;
};

template<std::size_t NumGroups>
struct ComptonCellData
{
    using GroupArray = std::array<double, NumGroups>;
    using GroupMatrix = std::array<GroupArray, NumGroups>;
    using GroupCdf = std::array<double, NumGroups + 1>;
    using GroupCdfMatrix = std::array<GroupCdf, NumGroups>;

    bool active = false;
    bool signedSourceActive = false;
    double planckOpacity = 0.0;
    double volume = 0.0;
    double temperature = 0.0;
    double Um = 0.0;
    double beta = 0.0;
    double cv = 0.0;
    double fleck = 1.0;
    double Upsilon = 0.0;
    double Gamma = 0.0;
    double betaCdtF = 0.0;
    bool useNZero = false;
    bool usePlanckInduced = false;
    GroupArray absorptionOpacity{};
    GroupArray planckFraction{};
    GroupArray baseSourceFraction{};
    GroupCdf planckCdf{};
    GroupCdf baseSourceCdf{};
    GroupArray oldRadiationEnergy{};
    GroupArray occupation{};
    GroupArray D{};
    GroupArray M{};
    GroupArray rowS{};
    GroupArray Lambda{};
    GroupArray Bbase{};
    GroupArray Bcorr{};
    GroupArray Btotal{};
    GroupArray Bpos{};
    GroupArray Bres{};
    GroupArray baseEffectiveOpacity{};
    GroupMatrix rates{};
    GroupMatrix derivative{};
    GroupMatrix tau{};
    GroupMatrix dtau_dUm{};
    GroupMatrix S{};
    GroupMatrix dSdUm{};
    GroupMatrix segmentKernel{};
    GroupMatrix residualKernel{};
    GroupMatrix Ktotal{};
    GroupMatrix implicitKernel{};
    GroupMatrix implicitEventRateMatrix{};
    GroupArray outRate{};
    GroupCdfMatrix targetCdf{};
    GroupCdfMatrix implicitEventCdf{};
    GroupArray groupCenters{};
    GroupArray groupWidths{};
    GroupArray meanEnergyRatio{};
    GroupArray modifiedFleck{};
    GroupArray residualSource{};
    GroupArray comptonOutRate{};
    GroupArray comptonMu{};
    GroupArray comptonMh{};
    GroupArray implicitEventRate{};
    GroupArray implicitDiagonalCorrection{};
    GroupArray riskScore{};
    std::array<std::size_t, NumGroups> riskTargetPackets{};
    double signedSourceL1 = 0.0;
    double signedSourceNet = 0.0;
    std::size_t implicitEvents = 0;
    std::size_t angleDependentEvents = 0;
    std::size_t residualPackets = 0;
};

template<std::size_t NumGroups>
inline bool isValidComptonResult(const ComptonKernelResult<NumGroups> &result)
{
    for(std::size_t source = 0; source < NumGroups; ++source)
    {
        for(std::size_t target = 0; target < NumGroups; ++target)
        {
            if(!std::isfinite(result.rates[source][target]) ||
               result.rates[source][target] < 0.0 ||
               !std::isfinite(result.derivative[source][target]))
            {
                return false;
            }
        }
        if(!std::isfinite(result.residualSource[source]))
        {
            return false;
        }
        if(result.hasFleckCorrection &&
           (!std::isfinite(result.fleckCorrection[source]) ||
            result.fleckCorrection[source] <= 0.0 ||
            result.fleckCorrection[source] > 1.0))
        {
            return false;
        }
    }
    return true;
}

} // namespace STORM

#endif // STORM_RADIATION_COMPTON_HPP
