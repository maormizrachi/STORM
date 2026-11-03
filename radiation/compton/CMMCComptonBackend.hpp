#ifndef STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP
#define STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "radiation/compton/ComptonTableBackend.hpp"

namespace STORM {

// CMMC (Monte Carlo sampled) Compton tables, built by SetTables.
class CMMCComptonBackendCore final : public ComptonTableBackendCore
{
public:
    CMMCComptonBackendCore(std::vector<double> centers,
                           std::vector<double> boundaries,
                           std::size_t samples,
                           bool detailedBalance,
                           int seed);
    ~CMMCComptonBackendCore() override;

    CMMCComptonBackendCore(const CMMCComptonBackendCore &) = delete;
    CMMCComptonBackendCore &operator=(const CMMCComptonBackendCore &) = delete;
    CMMCComptonBackendCore(CMMCComptonBackendCore &&) noexcept;
    CMMCComptonBackendCore &operator=(CMMCComptonBackendCore &&) noexcept;

    void SetTables(const std::vector<double> &temperatures);
    ComptonMatrixData GetTauMatrix(double temperature,
                                   double density,
                                   double atomicWeight,
                                   double ionization) const override;
    double GetMinimumTemperature() const override;
    double GetMaximumTemperature() const override;
    void GetAngleCdf(double temperature,
                     std::size_t sourceGroup,
                     std::size_t targetGroup,
                     std::vector<double> &cdf) const override;
    std::size_t GetAngleBinCount() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    double minimumTemperature_ = 0.0;
};

// The CMMC tables behind the ComptonTableBackend interface (kept for tools
// that construct CMMC tables directly).
template<std::size_t NumGroups>
class CMMCComptonBackend : public ComptonTableBackend<NumGroups>
{
public:
    using GroupArray = std::array<double, NumGroups>;

    CMMCComptonBackend(const GroupArray &centers,
                       const std::array<double, NumGroups + 1> &boundaries,
                       std::size_t samples,
                       bool detailedBalance,
                       int seed):
        CMMCComptonBackend(std::make_unique<CMMCComptonBackendCore>(
            std::vector<double>(centers.begin(), centers.end()),
            std::vector<double>(boundaries.begin(), boundaries.end()),
            samples,
            detailedBalance,
            seed))
    {}

    void SetTables(const std::vector<double> &temperatures)
    {
        this->cmmc_->SetTables(temperatures);
    }

private:
    explicit CMMCComptonBackend(std::unique_ptr<CMMCComptonBackendCore> core):
        ComptonTableBackend<NumGroups>(nullptr == core ? nullptr : std::unique_ptr<ComptonTableBackendCore>(core.get())),
        cmmc_(core.release())
    {}

    CMMCComptonBackendCore *cmmc_ = nullptr;
};

} // namespace STORM

#endif // STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP
