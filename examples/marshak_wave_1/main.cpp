/*
 * Marshak wave problem 1: Krief & McClarren (2024) Test 2
 *
 * Non-equilibrium radiation diffusion benchmark.
 * kappa_R = 100*(T/keV)^{-3} cm^{-1}, kappa_P = 0.001*kappa_R
 * u(T) = 6.860085e14*(T/keV)^4 erg/cm^3, rho = 1, uniform grid
 * T_bath(t) = 1.008038*(t/ns)^{1/3} keV
 * Domain [0, 0.2] cm, t_final = 1 ns
 *
 * 1D Cartesian mesh (CartesianMesh3D with ny=nz=1).
 * MC result is compared against a reference diffusion profile.
 *
 * Usage:
 *   ./marshak_wave_1 [Nx] [new_per_cell] [boundary_per_cell]
 */

#include "examples/marshak_wave/MarshakCommon.hpp"

int main(int argc, char *argv[])
{
    return STORM::examples::RunMarshakWave(1, argc, argv);
}
