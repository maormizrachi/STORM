# Marshak Wave Benchmarks (Problems 1--4)

Self-similar Marshak wave benchmarks on a 1D Cartesian mesh, comparing
IMC Monte Carlo transport against reference diffusion solutions.

| Problem | Directory | Reference | Opacity |
|---|---|---|---|
| 1 | `marshak_wave_1/` | Krief & McClarren (2024), Test 2 | Non-equilibrium ($\kappa_P = 0.001\,\kappa_R$) |
| 2 | `marshak_wave_2/` | Krief & McClarren (2024), Test 3 | Equilibrium ($\kappa_P = \kappa_R$) |
| 3 | `marshak_wave_3/` | Derei et al. (2024), Test 1 | Non-uniform $\rho(x)$, power-law EOS |
| 4 | `marshak_wave_4/` | Derei et al. (2024), Test 3 | Stretched grid, divergent density |

A thermal radiation wave propagates into cold material driven by a
time-dependent Planck boundary source at $x = 0$. The non-linear EOS
is handled by recomputing $T$ from internal energy after each step.
The scattering opacity returns $\kappa_R - \kappa_P$ so the transport
mean free path matches the Rosseland mean.

## Usage

```bash
./marshak_wave_N [Nx] [new_per_cell] [boundary_per_cell]
```

- `Nx` -- number of cells (default 128)
- `new_per_cell` -- photon packets per cell per step (default 5)
- `boundary_per_cell` -- boundary source photon packets per cell (default 100)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `Nx`, `new_per_cell`, `boundary_per_cell` | CLI args | Resolution and statistical quality |
| `imcParams.withRandomWalk` | `MarshakCommon.hpp` | Random walk acceleration (default on) |
| `CombPopulationControl(grid, 15, 6.0)` | `MarshakCommon.hpp` | Population control target and ratio |
| `dt` schedule | `MarshakCommon.hpp` | Time stepping (starts at 1e-15 s, ramps to 5e-11 s) |

These problems are extremely optically thick and benefit from random walk
acceleration. Without it, use small Nx (16--64) for demonstration.

## Comparison

At the end of each run, `plot_marshak.py` is invoked automatically. It
plots the MC temperature profile against the reference diffusion solution
and produces:

- `marshak_wave_N.png` / `.pdf` -- temperature profile comparison

The reference data (`reference.txt` in each problem directory) contains
RICH diffusion solutions (512 cells) in the format `x Tgas Trad` (Kelvin).

The L1 relative error against the reference is printed to stdout.

To generate the plot manually:

```bash
python3 ../marshak_wave/plot_marshak.py N
```
