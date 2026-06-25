#ifndef RDMONT_RADIATION_CELL_HPP
#define RDMONT_RADIATION_CELL_HPP

namespace RDMont {

struct RadiationCell
{
    double temperature = 0;       // material temperature [K]
    double cv = 0;                // volumetric heat capacity [erg/K/cm^3]
    double internalEnergy = 0;    // material internal energy density [erg/cm^3]
    double Erad = 0;              // radiation energy density [erg/cm^3] (output)
};

} // namespace RDMont

#endif // RDMONT_RADIATION_CELL_HPP
