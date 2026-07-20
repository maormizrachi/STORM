#ifndef STORM_RADIATION_COMPTON_HPP
#define STORM_RADIATION_COMPTON_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <functional>
#include <stdexcept>
#include <utility>

namespace STORM {

/*
 * A frozen Compton kernel is deliberately kept separate from the opacity
 * model.  The opacity model describes ordinary material opacities; this
 * interface describes the group-changing Compton operator and lets an
 * application supply its preferred matrix generator (Monte Carlo tables,
 * analytic Kompaneets rates, or a tabulated kernel).
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

// Small adapter for existing matrix generators.  The callback is invoked
// once per cell in preStep, so callers can wrap CMMC or a table lookup
// without making the standalone transport library depend on that generator.
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

    bool active = false;
    bool signedSourceActive = false;
    double fleck = 1.0;
    GroupMatrix rates{};
    GroupMatrix derivative{};
    GroupArray outRate{};
    GroupMatrix targetCdf{};
    GroupArray groupCenters{};
    GroupArray groupWidths{};
    GroupArray meanEnergyRatio{};
    GroupArray modifiedFleck{};
    GroupArray residualSource{};
    double signedSourceL1 = 0.0;
    double signedSourceNet = 0.0;
    std::size_t implicitEvents = 0;
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
