#ifndef STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP
#define STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace STORM {

struct CMMCComptonMatrixData
{
    std::vector<std::vector<double>> tau;
    std::vector<std::vector<double>> dtau_dUm;
    double lastGroupUpScatter = 0.0;
    double lastGroupDownScatter = 0.0;
};

class CMMCComptonBackendCore
{
public:
    CMMCComptonBackendCore(std::vector<double> centers,
                           std::vector<double> boundaries,
                           std::size_t samples,
                           bool detailedBalance,
                           int seed);
    ~CMMCComptonBackendCore();

    CMMCComptonBackendCore(const CMMCComptonBackendCore &) = delete;
    CMMCComptonBackendCore &operator=(const CMMCComptonBackendCore &) = delete;
    CMMCComptonBackendCore(CMMCComptonBackendCore &&) noexcept;
    CMMCComptonBackendCore &operator=(CMMCComptonBackendCore &&) noexcept;

    void SetTables(const std::vector<double> &temperatures);
    CMMCComptonMatrixData GetTauMatrix(double temperature,
                                       double density,
                                       double atomicWeight,
                                       double ionization) const;
    double GetMaximumTemperature() const;
    void GetAngleCdf(double temperature,
                     std::size_t sourceGroup,
                     std::size_t targetGroup,
                     std::vector<double> &cdf) const;
    std::size_t GetAngleBinCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template<std::size_t NumGroups>
class CMMCComptonBackend
{
public:
    using GroupArray = std::array<double, NumGroups>;
    using GroupMatrix = std::array<GroupArray, NumGroups>;

    CMMCComptonBackend(const GroupArray &centers,
                       const std::array<double, NumGroups + 1> &boundaries,
                       std::size_t samples,
                       bool detailedBalance,
                       int seed):
        core_(std::vector<double>(centers.begin(), centers.end()),
              std::vector<double>(boundaries.begin(), boundaries.end()),
              samples,
              detailedBalance,
              seed)
    {}

    void SetTables(const std::vector<double> &temperatures)
    {
        this->core_.SetTables(temperatures);
    }

    void GetTauMatrix(double temperature,
                      double density,
                      double atomicWeight,
                      double ionization,
                      GroupMatrix &tau,
                      GroupMatrix &dtau_dUm,
                      double &lastGroupUpScatter,
                      double &lastGroupDownScatter) const
    {
        CMMCComptonMatrixData const result = this->core_.GetTauMatrix(
            temperature, density, atomicWeight, ionization);
        if(result.tau.size() != NumGroups ||
           result.dtau_dUm.size() != NumGroups)
        {
            throw std::runtime_error("CMMC returned a matrix with the wrong group count");
        }
        for(std::size_t source = 0; source < NumGroups; ++source)
        {
            if(result.tau[source].size() != NumGroups ||
               result.dtau_dUm[source].size() != NumGroups)
            {
                throw std::runtime_error("CMMC returned a non-square group matrix");
            }
            for(std::size_t target = 0; target < NumGroups; ++target)
            {
                tau[source][target] = result.tau[source][target];
                dtau_dUm[source][target] = result.dtau_dUm[source][target];
            }
        }
        lastGroupUpScatter = result.lastGroupUpScatter;
        lastGroupDownScatter = result.lastGroupDownScatter;
    }

    double GetMaximumTemperature() const
    {
        return this->core_.GetMaximumTemperature();
    }

    void GetAngleCdf(double temperature,
                     std::size_t sourceGroup,
                     std::size_t targetGroup,
                     std::vector<double> &cdf) const
    {
        this->core_.GetAngleCdf(
            temperature, sourceGroup, targetGroup, cdf);
    }

    std::size_t GetAngleBinCount() const
    {
        return this->core_.GetAngleBinCount();
    }

private:
    CMMCComptonBackendCore core_;
};

} // namespace STORM

#endif // STORM_RADIATION_CMMC_COMPTON_BACKEND_HPP
