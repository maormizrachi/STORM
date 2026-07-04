#ifndef STORM_RADIATION_CELL_HPP
#define STORM_RADIATION_CELL_HPP

namespace STORM {

struct RadiationCell
{
    double temperature = 0;       // material temperature [K]
    double cv = 0;                // total heat capacity [erg/K] (unused by RadiationIMC)
    double internalEnergy = 0;    // total material internal energy [erg]
    double Erad = 0;              // radiation energy per unit mass [erg/g]
};

struct SimpleExtensives
{
    double internal_energy = 0;   // total material internal energy [erg]
    double mass = 0;              // total mass [g]
    double Erad = 0;              // total radiation energy [erg]
};

} // namespace STORM

#endif // STORM_RADIATION_CELL_HPP
