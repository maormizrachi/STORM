/*
 * Marshak wave problem 4: Derei et al. (2024) Test 3
 *
 * Divergent density on a stretched grid.
 * kappa_R = 2*(T/keV)^{-4.5}*rho^{1.9}, kappa_P = 5e-4*kappa_R
 * u(T,rho) = 1e14*(T/keV)^6*rho^{0.7} erg/cm^3
 * rho(x) = x^{-40/139}, geometrically stretched grid
 * T_bath(t) = 1.01008116*(t/ns)^{14/139} keV
 * Domain [1e-5, 1+1e-5] cm, t_final = 1 ns
 *
 * 1D Cartesian mesh (CartesianMesh3D with ny=nz=1).
 * MC result is compared against a reference diffusion profile.
 *
 * Usage:
 *   ./marshak_wave_4 [Nx] [new_per_cell] [boundary_per_cell]
 */

#include "examples/marshak_wave/MarshakCommon.hpp"

int main(int argc, char *argv[])
{
    return STORM::examples::RunMarshakWave(4, argc, argv);
}
