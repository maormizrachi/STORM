# Moving Slab Benchmark

McClarren & Gentile (2021) moving radiating slab benchmark, adapted from
`RICH/regression_tests/cases/moving_slab_mc`.

A uniform slab of material (rho = 0.1 g/cm^3, T = 1 keV, L = 0.4 cm)
moves at v = 0.5994 cm/ns and radiates into vacuum. The simulation uses
124-group frequency-dependent transport with Doppler shifts and DDMC
acceleration. An observer at x = 12 cm records the time-averaged energy
density spectrum, which is compared against a semi-analytic solution.

**Important:** This is a parallel benchmark that consumes a large amount
of memory. Each MPI process should have at least **20 GB** of available
memory. On SLURM clusters, use `--ntasks-per-node=` to limit processes
per node accordingly, and `--distribution=cyclic` is recommended to
spread ranks across nodes.

## Usage

```bash
mpirun -np N ./moving_slab [newPhotonsPerCell]
```

- `newPhotonsPerCell` -- photon packets emitted per cell per step (default: 30000 / 9 ~ 3333)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `newPhotonsPerCell` | CLI arg or `main.cpp` | Controls statistical resolution |
| `N_SLAB_PTS` / `N_VAC_PTS` | `main.cpp` | Mesh resolution in the slab / vacuum |
| `NYZ` | `main.cpp` | Transverse mesh resolution (default 3) |
| `dtMax` | `RunSimulation()` | Maximum time step (default 0.1 ns) |
| `tO` | `main.cpp` | Observation time (default 10 ns) |
| `zO` | `main.cpp` | Observer position (default 12 cm) |
| `imcParams.withDDMC` | `main.cpp` | Enable/disable DDMC acceleration |
| Manager type | `CreateMonteCarloManager()` call | `ManagerType::Legacy`, `RDMA`, or `P2P` |

The opacity table is defined in `MovingSlabOpacity.hpp` with 124 energy
groups. The opacity model weights the Planck emission CDF by absorption
opacity for correct frequency sampling.

## Example output

<img src="storm_moving_slab_comparison.png?raw=true" alt="STORM vs semi-analytic spectrum comparison" width="600"/>

## Comparison

At the end of the run, `check_spectrum.py` is invoked automatically. It
computes the semi-analytic solution from McClarren & Gentile (2021) and
compares it against the simulation output, producing:

- `storm_moving_slab_comparison.png` / `.pdf` -- spectrum plot
- F-error and L1 norm printed to stdout

The comparison reads `moving_slab_mc_spectrum.txt` (written by the C++ code).

To run the comparison manually:

```bash
python3 check_spectrum.py
```
