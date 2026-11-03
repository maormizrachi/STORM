#ifndef STORM_RADIATION_MACE_COMPTON_BACKEND_HPP
#define STORM_RADIATION_MACE_COMPTON_BACKEND_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "radiation/compton/ComptonTableBackend.hpp"

namespace STORM {

struct MACEComptonTableOptions
{
    std::string cacheDir;          // "" = $STORM_COMPTON_TABLE_DIR, else the working directory
    std::size_t angleBins = 32;
    bool exactSolver = true;       // false: MACE's approximate kernel solver
    int threads = 0;               // OpenMP threads per rank while building (0: see RadiationIMCParameters)
};

// MACE number-weighted effective-rate Compton tables (deps/MACE, branch
// cpp-library-mpi): read from the cache file for this grid or built with all
// MPI ranks and cached.  Rates are interpolated with a cubic Hermite in
// (log T, log sigma) using the tabulated derivatives as slopes (linear in
// sigma where a node is zero); temperatures outside the table are clamped
// (one warning, zero derivative); detailed balance is enforced on the
// interpolated rates as CMMC does.  The diagonal (direction-only) rate is the
// table's interpolated energy-weighted total scattering rate (its Planck-
// weighted thermal Klein-Nishina rate up to quadrature, taper, the grid edges
// and interpolation) minus the balanced effective off-diagonal rates, so that
// the rates of a row sum to that total before occupation factors and the pair
// throttle.  Negative remainders within a round-off tolerance are clipped to
// zero; larger ones stop the run (a load-time audit samples the table first).
// The diagonal angle CDF is that remainder per angle bin at the table
// temperatures, with negative bins projected away, interpolated linearly in T
// as all the angle CDFs (an approximation between the nodes, measured by the
// audit).
class MACEComptonBackendCore final : public ComptonTableBackendCore
{
public:
    MACEComptonBackendCore(std::vector<double> centers,
                           std::vector<double> boundaries,
                           const MACEComptonTableOptions &options);
    ~MACEComptonBackendCore() override;

    MACEComptonBackendCore(const MACEComptonBackendCore &) = delete;
    MACEComptonBackendCore &operator=(const MACEComptonBackendCore &) = delete;

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
    bool ClampsTemperature() const override
    {
        return true;
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace STORM

#endif // STORM_RADIATION_MACE_COMPTON_BACKEND_HPP
