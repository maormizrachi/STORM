# Recorded verification

Both MPI and serial executables compiled. All four final benchmark smoke
runs (including Su–Olson) passed on `bigrun` in job 10170860 (3m28s, exit 0:0).
Executable hashes and the compressed smoke log are retained in `results/`.

The full 380×380 calculation completed in job **10170859**, using **128 MPI
ranks**, in **00:46:26**, with exit **0:0**. Arguments:

```
380 4 512 1e-13 output_high_statistics 80
```

The full reference/energy comparison **PASSED** with the original tolerances.
The physical inputs, mesh and timestep follow the paper. STORM's sampling
parameters and population-control rules differ from the paper's exact
2-million-new/25-million-total particle prescription; see README.md.

| ct (cm) | Material T relative L1 vs published IMC | Foam material log10(T⁴) RMSE vs PN | Foam radiation log10(E) RMSE vs PN |
|---|---:|---:|---:|
| 2 | 1.348% | 0.0331 | 0.0322 |
| 2.5 | 1.381% | 0.0348 | 0.0406 |
| 3 | 1.497% | 0.0323 | 0.0265 |

The material L1 comparison uses r≤1.4 cm and excludes one diagonal cell
spacing around the material interface. Foam PN comparisons use
r<sqrt(0.5) minus one diagonal cell spacing. Reference digitization uncertainty
is approximately 0.01 cm / 0.03 dex. Full-profile PN errors and all raw
reference markers are also retained; the above numbers do not establish
agreement throughout the faint radiation tail.

Maximum energy-budget residual: **1.29e-14**.
Maximum source-normalization relative discrepancy: **4.06e-07**.
Aluminum area independently checked: **2.5 cm²**.

## Sampling and mesh sensitivity

Two lower-statistics runs completed cleanly: 190×190 on 32 ranks (job
10170856, 7m59s) and 380×380 on 64 ranks (job 10170855, 20m02s), both with
thermal parameter 2, source density 128, census parameter 20 and dt=1e-13 s.
Their material profiles agree with published IMC to about 2%, but their
radiation comparisons **failed** the 0.1-dex foam threshold at one or more
times. These failures are preserved in `results/coarse` and
`results/low_statistics`; the tolerance was not relaxed.

The higher-statistics run increases source sampling and the census parameter
by four, and the thermal parameter by two. Radiation tails remain noisy and
are plotted without smoothing. The published MC radiation profiles are also
noisy at low intensity. Increasing sampling is not a substitute for checking
source normalization, opacity units, or EOS conventions.

At ct=3, volume-averaging the fine field to the 190×190 grid gives a
heated-region temperature difference of
**1.472%**
and radiation-field difference of
**27.056%**. These include
sampling differences between independent runs; they are not isolated spatial
convergence errors.

## Retained results

All three diagonal profiles, compressed full fields, energy ledger,
comparison figures, metrics and solver log are in `results/fine`. Reproduce:

```
python3 compare.py results/fine --coarse results/coarse --check
```

Independent EOS/spectrum checks pass (`tests/verify.sh`): maximum relative
Planck-mean/emission-mean discrepancy 1.04e-4; maximum absolute emission-CDF
quantile discrepancy 6.28e-4, below its 1e-3 tolerance.

The full job used the same effective physics settings and explicit runtime
arguments as the final source. A subsequent edit made disabled options
explicit and printed startup settings; final smoke tests cover that build.

![Published-reference comparison](results/fine/comparison.png)
![Radiation field](results/fine/radiation_map.png)
