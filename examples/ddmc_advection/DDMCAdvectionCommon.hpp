#ifndef STORM_EXAMPLES_DDMC_ADVECTION_COMMON_HPP
#define STORM_EXAMPLES_DDMC_ADVECTION_COMMON_HPP

/*
 * Shared pieces of the DDMC advection examples (ddmc_advection_pulse and
 * ddmc_advection_interface): the cell types, a grey pure-scattering opacity,
 * a constant-cv EOS, the comoving <-> lab packet boost, a deterministic
 * random stream for the initial packets, and the key=value metrics writer
 * that examples/ddmc_advection/check_ddmc_advection.py reads.
 *
 * This directory has no REGRESSION_INFO, so it is not built on its own.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "examples/Vector3D.hpp"
#include "StormError.hpp"
#include "particle/Particle.hpp"
#include "utils/CounterRNG.hpp"
#include "radiation/RadiationOpacityModel.hpp"

namespace STORM {
namespace examples {
namespace ddmc_advection {

constexpr double kPi = 3.14159265358979323846;

// Material state of one cell.  RadiationIMC finds these members by name:
// `velocity` (with withHydro) makes the transport comoving, `ID` is the
// stable DDMC cell identity and `density` the mass density.
struct AdvectionCell
{
    std::size_t ID = 0;
    double temperature = 0.0;        // [K]
    double internalEnergy = 0.0;     // [erg]
    double Erad = 0.0;               // [erg/g], census energy written back by RadiationIMC
    double density = 0.0;            // [g/cm^3]
    Vector3D velocity;               // [cm/s]
    double scatteringOpacity = 0.0;  // [1/cm], read by ScatteringOpacity
};

struct AdvectionExtensives
{
    double internal_energy = 0.0;    // [erg]
    double mass = 0.0;               // [g]
    double Erad = 0.0;               // [erg]
};

// Constant specific heat.  With no absorption the material never exchanges
// energy with the radiation, but RadiationIMC still needs a positive heat
// capacity to form the Fleck factor.
class ConstantCvEOS
{
public:
    explicit ConstantCvEOS(double cvPerMass)
        : cvPerMass_(cvPerMass)
    {
        if(!(cvPerMass > 0.0) || !std::isfinite(cvPerMass))
        {
            StormError eo("ConstantCvEOS requires a finite, positive specific heat");
            eo.addEntry("cv per mass", cvPerMass);
            throw eo;
        }
    }

    double dT2cv(double /*density*/, double /*temperature*/,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return cvPerMass_;
    }

    double de2T(double /*density*/, double specificEnergy,
                const std::vector<double> &, const std::vector<std::string> &) const
    {
        return specificEnergy / cvPerMass_;
    }

private:
    double cvPerMass_;
};

// Grey, purely scattering material: no absorption (hence no emission) and
// isotropic scattering with each cell's own opacity.
template<typename GridT>
class ScatteringOpacity final : public RadiationOpacityModel<Vector3D, GridT, AdvectionCell, 1>
{
public:
    using RadiationOpacityModel<Vector3D, GridT, AdvectionCell, 1>::CalcScatteringOpacity;

    double CalcPlanckOpacity(const AdvectionCell & /*cell*/) override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const AdvectionCell &cell) override
    {
        return cell.scatteringOpacity;
    }
};

inline double Dot(const Vector3D &a, const Vector3D &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Deterministic uniform stream for the initial packets.  The standard
// library distributions are implementation defined, so they would make the
// initial condition depend on the compiler.
class OpenUnitStream
{
public:
    explicit OpenUnitStream(std::uint64_t seed)
        : state_(seed)
    {}

    // Uniform on (0, 1): 53 random bits offset by half a bin, so neither end
    // point is ever returned.
    double Next()
    {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        z ^= z >> 31U;
        return (static_cast<double>(z >> 11U) + 0.5) * 0x1.0p-53;
    }

    // Box-Muller standard normal deviate.
    double NextNormal()
    {
        double const radius = std::sqrt(-2.0 * std::log(this->Next()));
        return radius * std::cos(2.0 * kPi * this->Next());
    }

private:
    std::uint64_t state_;
};

inline Vector3D IsotropicDirection(OpenUnitStream &stream)
{
    double const mu = 1.0 - 2.0 * stream.Next();
    double const phi = 2.0 * kPi * stream.Next();
    double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return Vector3D(sinTheta * std::cos(phi), sinTheta * std::sin(phi), mu);
}

// Packets placed directly into the manager bypass the sources, which give
// every packet its own counter-RNG stream; without this, all packets share the
// default key and draw identical random numbers.
inline void SetPacketIdentity(Particle<Vector3D> &particle, std::uint64_t id, std::uint64_t seed)
{
    particle.id = static_cast<decltype(particle.id)>(id);
    particle.rngKey = STORM::CounterRNG::makeKey(seed, 0, id);
    particle.rngCounter = 0;
}

// Lab-frame packet from a comoving one.  A packet of comoving energy e0 and
// direction n0 in gas moving with velocity v = beta c vhat has
//   e = gamma e0 (1 + beta vhat.n0),
//   n = [n0 + ((gamma - 1) vhat.n0 + gamma beta) vhat] / (gamma (1 + beta vhat.n0)),
// the inverse of the Doppler factor gamma (1 - v.u / c^2) that RadiationIMC
// applies when it takes a lab packet into the comoving frame.
inline void SetLabPacketFromComoving(Particle<Vector3D> &particle,
                                     const Vector3D &comovingDirection,
                                     double comovingEnergy,
                                     const Vector3D &gasVelocity,
                                     double lightSpeed)
{
    double const speed = std::sqrt(Dot(gasVelocity, gasVelocity));
    if(!(speed > 0.0))
    {
        particle.velocity = comovingDirection * lightSpeed;
        particle.weight = comovingEnergy;
        return;
    }
    double const beta = speed / lightSpeed;
    if(!(beta < 1.0))
    {
        StormError eo("SetLabPacketFromComoving requires a subluminal gas velocity");
        eo.addEntry("beta", beta);
        throw eo;
    }
    Vector3D const vHat = gasVelocity / speed;
    double const gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    double const muHat = Dot(vHat, comovingDirection);
    double const doppler = gamma * (1.0 + beta * muHat);
    Vector3D direction = comovingDirection + vHat * ((gamma - 1.0) * muHat + gamma * beta);
    direction = direction / std::sqrt(Dot(direction, direction));
    particle.velocity = direction * lightSpeed;
    particle.weight = comovingEnergy * doppler;
}

// Comoving-frame energy of a packet in gas moving with `gasVelocity`.
// DDMC residents already carry it; a lab packet is boosted with the same
// Doppler factor gamma (1 - v.u / c^2) the engine uses.  With no absorption
// and a spatially uniform v, every packet keeps this energy to roundoff:
// comoving scattering is elastic, DDMC leaks and advective hops keep the
// comoving weight, and a transverse wall reflection leaves v.u unchanged.
inline double ComovingEnergy(const Particle<Vector3D> &particle,
                             const Vector3D &gasVelocity,
                             double lightSpeed)
{
    if(particle.radiationState.isComoving())
    {
        return particle.weight;
    }
    double const v2 = Dot(gasVelocity, gasVelocity);
    if(!(v2 > 0.0))
    {
        return particle.weight;
    }
    double const inverseC2 = 1.0 / (lightSpeed * lightSpeed);
    double const gamma = 1.0 / std::sqrt(1.0 - v2 * inverseC2);
    return particle.weight * gamma * (1.0 - Dot(gasVelocity, particle.velocity) * inverseC2);
}

// One line of space-separated key=value pairs, the metrics format read by
// check_ddmc_advection.py.  Values must not contain whitespace.
class MetricsLine
{
public:
    explicit MetricsLine(const std::string &variant)
    {
        this->AddText("variant", variant);
    }

    MetricsLine &AddReal(const std::string &key, double value)
    {
        std::ostringstream formatted;
        formatted << std::scientific << std::setprecision(10) << value;
        return this->AddText(key, formatted.str());
    }

    MetricsLine &AddCount(const std::string &key, std::size_t value)
    {
        return this->AddText(key, std::to_string(value));
    }

    MetricsLine &AddFlag(const std::string &key, bool value)
    {
        return this->AddText(key, value ? "1" : "0");
    }

    MetricsLine &AddReals(const std::string &key, const std::vector<double> &values)
    {
        std::ostringstream formatted;
        formatted << std::scientific << std::setprecision(6);
        for(std::size_t i = 0; i < values.size(); ++i)
        {
            formatted << (i > 0 ? "," : "") << values[i];
        }
        return this->AddText(key, formatted.str());
    }

    MetricsLine &AddText(const std::string &key, const std::string &value)
    {
        if(value.empty() || value.find_first_of(" \t\n=") != std::string::npos)
        {
            StormError eo("Metrics values must be non-empty and contain no whitespace or '='");
            eo.addEntry("Key", key);
            eo.addEntry("Value", value);
            throw eo;
        }
        if(!this->text_.empty())
        {
            this->text_ += ' ';
        }
        this->text_ += key + "=" + value;
        return *this;
    }

    const std::string &str() const
    {
        return this->text_;
    }

private:
    std::string text_;
};

inline std::vector<std::string> SplitList(const std::string &text, char separator)
{
    std::vector<std::string> items;
    std::string current;
    for(char character : text)
    {
        if(character == separator)
        {
            items.push_back(current);
            current.clear();
        }
        else
        {
            current += character;
        }
    }
    items.push_back(current);
    return items;
}

// Positive integer command-line argument, with an actionable message.
inline std::size_t ParsePositiveCount(const std::string &text, const std::string &name)
{
    std::size_t parsed = 0;
    std::size_t consumed = 0;
    try
    {
        unsigned long long const value = std::stoull(text, &consumed);
        parsed = static_cast<std::size_t>(value);
    }
    catch(const std::exception &)
    {
        consumed = 0;
    }
    if(consumed != text.size() || parsed == 0 || text.find('-') != std::string::npos)
    {
        StormError eo("Command-line argument must be a positive integer");
        eo.addEntry("Argument", name);
        eo.addEntry("Value", text);
        throw eo;
    }
    return parsed;
}

} // namespace ddmc_advection
} // namespace examples
} // namespace STORM

#endif // STORM_EXAMPLES_DDMC_ADVECTION_COMMON_HPP
