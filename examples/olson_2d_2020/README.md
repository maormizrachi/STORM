# Olson 2020 Two-Dimensional Benchmark

Frequency-dependent radiative-transfer benchmark from Olson (2020), with
aluminum blocks embedded in foam and a central thermal volume source.
The reference comparison uses the published PN markers and IMC material
curves in Figure 6 of Steinberg & Heizler (2023).

The first-quadrant domain is $[0,3.8]\times[0,3.8]$ cm with reflecting
boundaries. Both materials have density $0.001\;\mathrm{g\,cm}^{-3}$ and
initial temperature 0.01 keV. A continuous isotropic source has a 0.5 keV
Planck spectrum and radial strength proportional to $\exp(-18.7r^3)$.
Temperature- and frequency-dependent absorption and material heat capacity
follow the benchmark formulas.

The calculation uses continuous-frequency IMC with three-dimensional angular
sampling on a spatially two-dimensional mesh. DDMC, random walk, physical
scattering, Compton scattering, and hydrodynamics are disabled.
[PHYSICS.md](PHYSICS.md) documents the geometry, opacity and EOS conventions,
source normalization, and reference provenance.

## Usage

From this directory:

```bash
./build.sh
./olson_2d_2020_serial 38 4 32 1e-12 output_serial 50
mpirun -np 4 ./olson_2d_2020 38 4 32 1e-12 output_mpi 50
python3 compare.py output_mpi --check
```

Both executables compile the same `main.cpp`. THUNDER uses `DUAL_MODE=1` to
discover `olson_2d_2020` and `olson_2d_2020_serial`, with separate output
directories. From the RICH root, launch both with:

```bash
./regression_tests/run_all.sh --mode all --test olson_2d_2020
```

For a standalone SLURM run, submit `run.slurm` from this directory. Its default
380 × 380 mesh is finer than the regression mesh shown below.

## Parameters you can modify

```text
./olson_2d_2020 [Nxy] [thermal_packet_parameter] [source_sampling_density] [dt] [output_dir] [census_target_per_cell]
```

| Parameter | Default | Regression value | Description |
|---|---:|---:|---|
| `Nxy` | 380 | 38 | Cells along each axis; must be a multiple of 38 |
| `thermal_packet_parameter` | 4 | 4 | Thermal sampling budget parameter |
| `source_sampling_density` | 512 | 32 | Source sampling in the strongest source cells |
| `dt` | `1e-13` s | `1e-12` s | Maximum timestep |
| `output_dir` | `output` | `output_regression` | Profiles, fields, and energy ledger |
| `census_target_per_cell` | 80 | 50 | Comb population-control parameter |

Snapshots land at $ct=2,2.5,3$ cm. Sampling and census parameters control packet
budgets; they do not impose exact packet counts. See [PHYSICS.md](PHYSICS.md).

## Comparison

`compare.py` requires NumPy and Matplotlib. It generates `comparison.png`,
`radiation_map.png`, and `metrics.json`. It checks energy conservation, source
normalization, the smooth foam region against PN, and material temperature
against published IMC. Interface neighborhoods are excluded from the error
norms; all reference markers remain visible. The reference files and their
extraction provenance are in `reference/`.

The figure below uses the latest completed regression simulation, campaign
`20260910_192213`, MPI job **10175643**, with four ranks and the 38 × 38 mesh.
The corrected comparison **passes**:

| $ct$ (cm) | Material temperature relative $L_1$ vs IMC | Foam material $\log_{10}(T^4)$ RMSE vs PN | Foam radiation $\log_{10}(E)$ RMSE vs PN |
|---|---:|---:|---:|
| 2 | 3.001% | 0.0378 | 0.0261 |
| 2.5 | 3.268% | 0.0162 | 0.0144 |
| 3 | 3.191% | 0.0198 | 0.0202 |

Maximum energy-budget residual is **1.93 × 10⁻¹⁵**; source-normalization
relative error is **0.0579%**. The original campaign reported a checker-path
failure; these comparisons were rerun on the saved output after fixing that
path, without rerunning the simulation.

Inputs, metrics, and provenance are retained in
[results/regression_20260910_192213](results/regression_20260910_192213).
Regenerate the figure with:

```bash
python3 compare.py results/regression_20260910_192213 --check
cp results/regression_20260910_192213/comparison.png olson_2d_2020.png
```

## Example output

<img src="olson_2d_2020.png?raw=true" alt="Olson 2020 diagonal material-temperature and radiation profiles at ct 2, 2.5, and 3 cm, compared with published PN markers and IMC material curves" width="900"/>

Solid curves show this run, open circles show published PN markers, and dashed
material curves show published IMC. Differences in the faint radiation tail
and at material interfaces are reported separately from the checked foam
region. This figure uses the regression mesh; earlier finer-grid results are
retained in [RESULTS.md](RESULTS.md).
