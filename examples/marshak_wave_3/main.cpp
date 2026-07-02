/*
 * Marshak wave problem 3: Derei et al. (2024) Test 1
 *
 * Non-uniform density, power-law EOS and opacity.
 * kappa_R = 40*(T/keV)^{-1.5}*rho^{1.2}, kappa_P = 0.0025*kappa_R
 * u(T,rho) = 1e14*(T/keV)^{3.4}*rho^{0.86} erg/cm^3
 * rho(x) = x^{20/19}, uniform grid
 * T_bath(t) = 1.0470478*(t/ns)^{86/57} keV
 * Domain [0, 1] cm, t_final = 1 ns
 *
 * 1D Cartesian mesh (CartesianMesh3D with ny=nz=1).
 * MC result is compared against a reference diffusion profile.
 *
 * Usage:
 *   ./marshak_wave_3 [Nx] [new_per_cell] [boundary_per_cell]
 */

#include "examples/marshak_wave/MarshakCommon.hpp"

int main(int argc, char *argv[])
{
    return STORM::examples::RunMarshakWave(3, argc, argv);
}
