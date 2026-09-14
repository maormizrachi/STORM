#pragma once
#include <random>
#include "boundary/RigidBoundary.hpp"

// The boundary hook can emit interior packets. The boundary itself remains
// reflecting; these are isotropic VOLUME-source packets, not a surface bath.
template <class P, class Grid>
class OlsonSource : public STORM::RigidBoundary<P, Grid>
{
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0., 1.};
    size_t count_;
    double injected_ = 0;
    double dx_;
    std::vector<double> planckCDF_;

  public:
    double GetInjectedEnergy() const
    {
        return injected_;
    }
    double BlackbodyEnergy(double temperature = .5)
    {
        // Exact Planck mixture: choose n with probability 1/(zeta(4)n^4),
        // then a gamma(shape=4,scale=1/n) deviate. Tail n>4096 <5e-12.
        double u = uniform_(rng_);
        size_t n = std::lower_bound(planckCDF_.begin(), planckCDF_.end(), u) - planckCDF_.begin() + 1;
        double x = 0;
        for(int k = 0; k < 4; ++k)
        {
            x -= std::log(std::max(uniform_(rng_), 1e-300));
        }
        return temperature * units::kev * x / n;
    }

  public:
    OlsonSource(const Grid &g, size_t count, double dx, int rank) : STORM::RigidBoundary<P, Grid>(g), rng_(9173 + rank), count_(count), dx_(dx)
    {
        double sum = 0;
        for(int n = 1; n <= 4096; ++n)
        {
            sum += 1 / std::pow(double(n), 4);
            planckCDF_.push_back(sum);
        }
        for(double &q : planckCDF_)
        {
            q /= sum;
        }
    }
    std::vector<STORM::Particle<P>> generateNewBoundaryParticles(double dt) override
    {
        std::vector<STORM::Particle<P>> packets;
        const double drive = units::clight * units::arad * std::pow(.5 * units::kev_kelvin, 4);
        for(size_t i = 0; i < this->grid.GetPointNo(); ++i)
        {
            P center = this->grid.GetCellCM(i);
            // No source truncation: stochastic rounding allocates packets with
            // probability proportional to the largest Q in the cell.
            double nearR = std::hypot(std::max(0., center.x - dx_ / 2), std::max(0., center.y - dx_ / 2));
            double qmax = std::exp(-18.7 * nearR * nearR * nearR);
            double expected = count_ * qmax;
            size_t n = size_t(expected);
            if(uniform_(rng_) < expected - n)
            {
                ++n;
            }
            for(size_t j = 0; j < n; ++j)
            {
                STORM::Particle<P> p;
                p.location = P(center.x + dx_ * (uniform_(rng_) - .5), center.y + dx_ * (uniform_(rng_) - .5), uniform_(rng_));
                double r = std::hypot(p.location.x, p.location.y);
                double mu = 2 * uniform_(rng_) - 1, phi = 2 * M_PI * uniform_(rng_);
                p.velocity = P(std::sqrt(1 - mu * mu) * std::cos(phi), std::sqrt(1 - mu * mu) * std::sin(phi), mu) * units::clight;
                p.frequency = BlackbodyEnergy();
                p.timeLeft = dt * uniform_(rng_);
                p.cellIndex = i;
                p.weight = drive * dt * this->grid.GetVolume(i) * std::exp(-18.7 * r * r * r) / expected;
                p.initialWeight = p.weight;
                injected_ += p.weight;
                packets.push_back(p);
            }
        }
        return packets;
    }
};
