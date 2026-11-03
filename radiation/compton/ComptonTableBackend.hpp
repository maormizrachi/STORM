#ifndef STORM_RADIATION_COMPTON_TABLE_BACKEND_HPP
#define STORM_RADIATION_COMPTON_TABLE_BACKEND_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace STORM {

// Group-to-group Compton rates at one temperature: tau[source][target] and
// dtau/dT per unit length (the electron density folded in), and the last
// group's in-group up/down-scatter rates (CMMC only; zero otherwise).
struct ComptonMatrixData
{
    std::vector<std::vector<double>> tau;
    std::vector<std::vector<double>> dtau_dUm;
    double lastGroupUpScatter = 0.0;
    double lastGroupDownScatter = 0.0;
};

using CMMCComptonMatrixData = ComptonMatrixData;

// A source of Compton tables (CMMC or MACE).  Implementations live in their
// own translation units, so STORM sees only this interface.
class ComptonTableBackendCore
{
public:
    virtual ~ComptonTableBackendCore() = default;

    virtual ComptonMatrixData GetTauMatrix(double temperature,
                                           double density,
                                           double atomicWeight,
                                           double ionization) const = 0;

    virtual double GetMinimumTemperature() const = 0;

    virtual double GetMaximumTemperature() const = 0;

    virtual void GetAngleCdf(double temperature,
                             std::size_t sourceGroup,
                             std::size_t targetGroup,
                             std::vector<double> &cdf) const = 0;

    virtual std::size_t GetAngleBinCount() const = 0;

    // True when GetTauMatrix accepts any positive temperature and clamps it to its table itself (zero derivative
    // outside); callers then pass the physical temperature.  False: callers clamp into the table's interior.
    virtual bool ClampsTemperature() const
    {
        return false;
    }
};

template<std::size_t NumGroups>
class ComptonTableBackend
{
public:
    using GroupArray = std::array<double, NumGroups>;
    using GroupMatrix = std::array<GroupArray, NumGroups>;

    explicit ComptonTableBackend(std::unique_ptr<ComptonTableBackendCore> core):
        core_(std::move(core))
    {
        if(!core_)
        {
            throw std::runtime_error("ComptonTableBackend needs a table source");
        }
    }

    virtual ~ComptonTableBackend() = default;

    void GetTauMatrix(double temperature,
                      double density,
                      double atomicWeight,
                      double ionization,
                      GroupMatrix &tau,
                      GroupMatrix &dtau_dUm,
                      double &lastGroupUpScatter,
                      double &lastGroupDownScatter) const
    {
        ComptonMatrixData const result = this->core_->GetTauMatrix(
            temperature, density, atomicWeight, ionization);
        if(result.tau.size() != NumGroups ||
           result.dtau_dUm.size() != NumGroups)
        {
            throw std::runtime_error("Compton table returned a matrix with the wrong group count");
        }
        for(std::size_t source = 0; source < NumGroups; ++source)
        {
            if(result.tau[source].size() != NumGroups ||
               result.dtau_dUm[source].size() != NumGroups)
            {
                throw std::runtime_error("Compton table returned a non-square group matrix");
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

    double GetMinimumTemperature() const
    {
        return this->core_->GetMinimumTemperature();
    }

    double GetMaximumTemperature() const
    {
        return this->core_->GetMaximumTemperature();
    }

    void GetAngleCdf(double temperature,
                     std::size_t sourceGroup,
                     std::size_t targetGroup,
                     std::vector<double> &cdf) const
    {
        this->core_->GetAngleCdf(
            temperature, sourceGroup, targetGroup, cdf);
    }

    std::size_t GetAngleBinCount() const
    {
        return this->core_->GetAngleBinCount();
    }

    bool ClampsTemperature() const
    {
        return this->core_->ClampsTemperature();
    }

private:
    std::unique_ptr<ComptonTableBackendCore> core_;
};

} // namespace STORM

#endif // STORM_RADIATION_COMPTON_TABLE_BACKEND_HPP
