#ifndef STORM_DENSMORE_BOUNDARY_HPP
#define STORM_DENSMORE_BOUNDARY_HPP

#include "examples/marshak_wave/MarshakBoundary.hpp"
#include "DensmoreOpacity.hpp"
#include <planck_integral/planck_integral.hpp>

namespace STORM {
namespace examples {

template<typename T, typename Grid>
class DensmoreBoundary : public MarshakBoundary<T, Grid>
{
public:
    using GroupBoundaries = std::array<double, N_DENSMORE_GROUPS + 1>;

    DensmoreBoundary(const Grid &grid, double temperature, size_t Npercell,
                     const GroupBoundaries &boundaries)
        : MarshakBoundary<T, Grid>(grid, temperature, Npercell),
          boundaries_(boundaries)
    {}

    std::vector<Particle<T, Grid>> generateNewBoundaryParticles(double fullDt) override
    {
        std::vector<Particle<T, Grid>> particles = MarshakBoundary<T, Grid>::generateNewBoundaryParticles(fullDt);
        double kT = constants::k_boltz * this->GetTemperature();
        std::uniform_real_distribution<double> unif(0, 1);
        static std::mt19937_64 re(42);

        std::array<double, N_DENSMORE_GROUPS + 1> cdf{};
        cdf[0] = 0.0;
        for(size_t g = 0; g < N_DENSMORE_GROUPS; ++g)
        {
            double a = boundaries_[g] / kT;
            double b = boundaries_[g + 1] / kT;
            double bg = (a > 0.0 && b > a) ? planck_integral::planck_integral(a, b) : 0.0;
            cdf[g + 1] = cdf[g] + bg;
        }
        double total = cdf[N_DENSMORE_GROUPS];

        for(Particle<T, Grid> &p : particles)
        {
            if(total > 0.0)
            {
                double r = unif(re) * total;
                size_t g = 0;
                while(g + 1 < N_DENSMORE_GROUPS && cdf[g + 1] < r)
                {
                    ++g;
                }
                double frac = unif(re);
                p.frequency = boundaries_[g] + frac * (boundaries_[g + 1] - boundaries_[g]);
            }
            else
            {
                p.frequency = 0.5 * (boundaries_[0] + boundaries_[N_DENSMORE_GROUPS]);
            }
        }
        return particles;
    }

private:
    GroupBoundaries boundaries_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_DENSMORE_BOUNDARY_HPP
