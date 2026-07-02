/*
 * Marshak wave problem 2: Krief & McClarren (2024) Test 3
 *
 * Equilibrium radiation diffusion benchmark (kappa_P = kappa_R).
 * kappa_R = kappa_P = 100*(T/keV)^{-3} cm^{-1}
 * u(T) = 6.860085e14*(T/keV)^4 erg/cm^3, rho = 1, uniform grid
 * T_bath(t) = 1.014565*(t/ns)^{1/3} keV
 * Domain [0, 0.2] cm, t_final = 1 ns
 *
 * 1D Cartesian mesh (CartesianMesh3D with ny=nz=1).
 * MC result is compared against a reference diffusion profile.
 *
 * Usage:
 *   ./marshak_wave_2 [Nx] [new_per_cell] [boundary_per_cell]
 */

#include "examples/marshak_wave/MarshakCommon.hpp"

int main(int argc, char *argv[])
{
    return STORM::examples::RunMarshakWave(2, argc, argv);
}
