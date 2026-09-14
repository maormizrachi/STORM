# Su–Olson Marshak-Wave Benchmark

Boundary-driven, non-equilibrium Marshak-wave benchmark from Su & Olson
(1996), with $\epsilon=1$. Gray implicit Monte Carlo (IMC) transport is compared
with an independently evaluated half-space diffusion solution.

A 1 keV blackbody bath illuminates the left boundary of a cold slab with
absorption opacity $\sigma=1\;\mathrm{cm}^{-1}$ and material energy density
$aT^4$. The dimensionless coordinates are $X=\sqrt{3}\sigma x$ and
$\tau=c\sigma t$. The domain is $X\in[0,40]$, and profiles are saved at
$\tau=1,10,100$. The far right and transverse boundaries reflect.

Transport and diffusion differ at early times; the regression checks their
late-time agreement. DDMC, random walk, physical scattering, and Compton
scattering are disabled. See [PHYSICS.md](PHYSICS.md) for the equations,
normalizations, packet allocation, and reference construction.

## Usage

From this directory:

```bash
./build.sh
./su_olson_serial 128 8 4000 0.1 output_serial
mpirun -np 4 ./su_olson 128 8 4000 0.1 output_mpi
python3 compare.py output_mpi --check
```

Both executables use the same `main.cpp`. `DUAL_MODE=1` lets THUNDER discover
`su_olson` and `su_olson_serial`; each writes to its own results directory.
To launch both through THUNDER, from the RICH root:

```bash
./regression_tests/run_all.sh --mode all --test su_olson
```

For a standalone SLURM run, submit `run.slurm` from this directory. Its default
settings use 512 cells and a smaller timestep than the regression run.

## Parameters you can modify

```text
./su_olson [Nx] [thermal_packet_parameter] [boundary_packets] [delta_tau] [output_dir]
```

| Parameter | Default | Regression value | Description |
|---|---:|---:|---|
| `Nx` | 512 | 128 | Slab cell count |
| `thermal_packet_parameter` | 8 | 8 | Thermal sampling budget parameter |
| `boundary_packets` | 10000 | 4000 | Incident packets per illuminated face per step |
| `delta_tau` | 0.05 | 0.1 | Dimensionless timestep |
| `output_dir` | `output` | `output_regression` | Profile and comparison directory |

The thermal parameter is a sampling-budget input, not an exact packet count
per cell. Details are in [PHYSICS.md](PHYSICS.md).

## Comparison

`compare.py` requires NumPy, Matplotlib, and mpmath. It evaluates the diffusion
solution by numerical Laplace inversion and generates `comparison.png` and
`metrics.json`. The check requires radiation and material energy relative
$L_1$ errors below 6% at $\tau=100$.

The figure below uses the latest completed regression simulation, campaign
`20260910_192213`, MPI job **10175644**, with four ranks and the regression
parameters above. Its late-time errors are **1.661% radiation energy** and
**1.268% material energy**; the comparison **passes**. The original campaign
reported a checker-path failure; the corrected checker was rerun against the
saved simulation output without rerunning the simulation.

The plotted inputs, generated metrics, and provenance are retained in
[results/regression_20260910_192213](results/regression_20260910_192213).
Regenerate the figure with:

```bash
python3 compare.py results/regression_20260910_192213 --check
cp results/regression_20260910_192213/comparison.png su_olson.png
```

## Example output

<img src="su_olson.png?raw=true" alt="Su–Olson radiation and material profiles and material temperature at tau 1, 10, and 100, compared with the diffusion reference" width="900"/>

The upper panels compare normalized radiation and material energies; the lower
panels compare material temperature. Early-time transport/diffusion differences
are expected and are not part of the late-time acceptance criterion.

[RESULTS.md](RESULTS.md) retains earlier serial/MPI validation runs.
