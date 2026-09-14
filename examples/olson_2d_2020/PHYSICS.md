# Olson 2020 complex 2D benchmark

G. L. Olson, *Stretched and Filtered Multigroup Pn Transport for Improved
Positivity and Accuracy*, JCTT **49**, 215–232 (2020),
[DOI](https://doi.org/10.1080/23324309.2020.1800745).

The input formulas and geometry are also reproduced in Sec. IV, Eqs. (5),
(6), (9) and Fig. 4 of Steinberg & Heizler,
[arXiv:2303.06634v1](https://arxiv.org/abs/2303.06634v1), *Frequency-dependent
Discrete Implicit Monte-Carlo Scheme for the Radiative Transfer Equation*.
The reference markers in that paper's Fig. 6 are the Olson PN results.

## Physical inputs

- First-quadrant square `[0,3.8] × [0,3.8]` cm, all boundaries reflecting.
  The 1 cm extrusion in z also has reflecting walls: the solution is spatially
  2D but directions are isotropic on the full sphere, not a planar circle.
- Aluminum rectangles: `[0.5,1.5] × [0.5,1.5]`,
  `[1.5,2.5] × [1.5,2.5]`, and `[1.5,2.5] × [0,0.5]` cm.
  The last is a half-block cut by the reflecting symmetry axis.
- Both materials have density **0.001 g cm⁻³**. Material identity is carried
  explicitly in a cell tracer, never inferred from density.
- Initial material and radiation temperature: **0.01 keV**, in equilibrium.
  Initial packet frequencies are explicitly Planck sampled; STORM's generic
  within-group linear interpolation is unsuitable for this broad band.
- Isotropic volume source, on throughout: `Q(r)=B(0.5 keV)*exp(-18.7*r^3)`.
  Its energy source density is `c*a*(0.5*TkeV)^4*exp(-18.7*r^3)` erg cm⁻³ s⁻¹.
  It is not a boundary source and has no extra opacity multiplier.
- Source packets sample positions throughout cells, emission times uniformly
  within each timestep, and the true Planck spectrum. Stochastic allocation
  and packet weights give the spatial source without a hard cutoff.
- Exact temperature- and energy-dependent absorption formulas are in
  `OlsonPhysics.hpp`, including every edge and the temperature caps.
  Physical scattering, DDMC, random walk, Compton and hydrodynamics are off.
- Default mesh: **380 × 380**; default dt: **1e-13 s**. Mesh resolution must
  be a multiple of 38 so material interfaces coincide with cell faces.
  Steps are shortened to land exactly at **ct=2, 2.5, 3 cm**.

## Critical interface conventions

STORM's transport kernel and emission formula consume **macroscopic** opacity
in cm⁻¹. The input formulas provide mass opacity in cm²/g, so `sigma` multiplies
by 0.001 exactly once. The current Fleck-factor expression divides by the
raw `dT2cv` result without a density multiplier. The benchmark's explicit EOS
adapter therefore returns **volumetric** Cv there. Separately, `de2T` accepts
specific energy and multiplies by density before inverting the EOS. This
local adapter is deliberate; do not replace it with an ordinary ideal-gas EOS.

For temperature in keV, the integrated EOS is
`e = a*TkeV^4*H*(T+(T+chi)*alpha)` with zero energy at T=0,
where `(H,chi)=(0.1,0.1)` for foam and `(0.5,0.3)` for aluminum.
The ionization fraction is evaluated with a numerically stable expression.

`withMultigroupOpacity=true` is essential. One **bookkeeping** group spans
1e-7–100 keV, but this is **continuous-frequency transport**, not a gray
opacity approximation. Each absorption uses the exact formula at the packet
energy. Thermal emission and effective scattering sample **sigma(E,T) B(E,T)**,
whereas the imposed source and initial field sample **B(E,T)**.

The weighted emission CDF and Planck mean use 2049 temperature rows and about
1542 energy edges. Quadrature splits at every absorption edge and uses two
Gauss points per logarithmic interval. The T=0.1 keV cap transition is a table
node. Temperatures outside the verified [0.0001,2] keV table fail explicitly.

## Build, run, and compare

```
./build.sh
mpirun -np 128 ./olson_2d_2020 380 4 512 1e-13 output 80
./olson_2d_2020_serial 38 4 32 1e-12 smoke_serial 50
sbatch --nodes=16 --ntasks-per-node=8 --time=01:00:00 run.slurm
python3 compare.py output
```

`build.sh` builds both executables from the same main, using
`#ifdef STORM_WITH_MPI`. MPI uses the distributed Cartesian mesh and
STORM's two-sided MPI transport manager. Source-weighted static mesh
ownership reduces load imbalance without changing the grid or physical inputs.
No RDMA hardware is required.
CMMC provides units/Planck headers; Compton transport stays off.
The script's resource default is 32 ranks; the command above requests 128.

Arguments: `Nxy thermal_packet_parameter source_sampling_density dt
output_directory census_target_per_cell`. STORM allocates a nominal global
thermal budget `10*Ncells*thermal_packet_parameter`, subject to per-cell
minimum `parameter` and maximum `20*parameter`. The source parameter controls
expected packets in the strongest source cells. Comb population control uses the census argument as a per-cell minimum,
a nominal global budget `5*Ncells*census`, and maximum `20*census` per cell;
this is **not** the paper's exact global 25-million hard cap. Particle counts are printed in the solver log. The physical inputs,
mesh and dt match the paper; sampling budgets are tunable.

Submit `run.slurm` from this directory, or use `sbatch --chdir=...`.
It specifies partition `bigrun`. Use separate output directories for runs.

Each snapshot writes the full field and diagonal profile. The diagonal
coordinate is **r=sqrt(x²+y²)**, not x. Radiation energy is the instantaneous
census sum divided by volume and `a*TkeV^4`. The energy ledger compares total
material plus census energy against initial energy plus actual injected packet
energy; all walls reflect. The comparison also checks injected energy against
the analytic source-area integral `pi*Gamma(2/3)/(6*18.7^(2/3))`. It is written to `energy_budget.txt`.

## Verification and reference provenance

`tests/verify.cpp` and `tests/verify.py` independently check EOS inversion,
Cv against a finite-difference derivative, Planck means against adaptive SciPy
quadrature, and the sampled emission quantiles and mean energy. Across both materials and
0.001–1 keV the maximum relative discrepancy in the Planck mean and emitted
mean energy was **1.04e-4**. The maximum absolute CDF discrepancy at the
10%, 50% and 90% emission quantiles was **6.28e-4** (tolerance 1e-3).
Run `tests/verify.sh` to compile and run these checks. NumPy and SciPy are
required for the Python verification.

`extract_reference.py` extracts the actual open-circle reference markers from
page 15, Fig. 6 of the published vector PDF. It excludes legend samples and keeps PN markers separate from the additional
published IMC material curves extracted from dashed vector paths. Reproduce with:

```
python3 extract_reference.py /path/to/2303.06634v1.pdf
```

This requires PyMuPDF. The checked-in CSV files and `provenance.json` preserve
the source URL, SHA256, axis mapping and extraction uncertainty. No simulation
values are used to construct the reference.

`compare.py` (NumPy/Matplotlib) generates `comparison.png`,
`radiation_map.png`, and `metrics.json`. Axes follow the paper:
`log10((T/keV)^4)` and `log10(E/(a*TkeV^4))`. Quantitative comparisons exclude
one diagonal cell spacing around discontinuous interfaces; all reference
markers remain visible on the plots. The full PN discrepancies are reported,
but pass/fail checks use the smooth foam region before r=sqrt(0.5), where
Sec. IV reports PN/MC agreement. Beyond it the paper explicitly notes
interface differences. An additional full material-temperature comparison
uses the published IMC curves, which are drawn as dashed lines. In noisy tails, cells with zero census
energy cannot supply a finite logarithmic error and are omitted from the
metric, with the number of compared markers reported. The reference PN tails
and MC tails should not be assumed identical to arbitrarily small intensity.
Full fields may be stored as `field_ct*.txt.gz`; the comparison reads either
plain or gzip files. `--coarse PATH` additionally compares volume-averaged
fine fields with a coarser run and writes `mesh_comparison.png`. This includes
both spatial discretization and Monte Carlo sampling differences.

SLURM launches use a unique executable copy in `.job_binaries/` for each job,
so a subsequent rebuild cannot invalidate an executable mapped over NFS.

See [RESULTS.md](RESULTS.md) for recorded runs, numerical comparisons,
sampling limitations, and retained profiles and figures.
