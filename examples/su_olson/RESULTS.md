# Recorded verification

Both executable variants compiled and completed on SLURM partition `bigrun`.
These are actual runs with `512 8 10000 0.05 OUTPUT`, through tau=100.
After explicitly writing the disabled physics settings and adding startup
diagnostics, both final binaries passed an additional smoke run in job
10170860. `results/final_build_and_smoke.json` records their hashes.

| Mode | Job | Ranks | Elapsed | Exit | Comparison |
|---|---:|---:|---|---|---|
| Serial | 10170853 | 1 | 15m44s | 0:0 | PASS |
| MPI | 10170854 | 4 | 14m57s | 0:0 | PASS |

At tau=100, the relative L1 discrepancies from the half-space diffusion
solution are:

| Mode | Radiation energy | Material energy | Material temperature |
|---|---:|---:|---:|
| Serial | 1.763% | 1.427% | 0.451% |
| MPI | 1.800% | 1.327% | 0.395% |

The serial/MPI material-temperature difference is 0.200% in the same norm.
Independent random histories need not agree pointwise. No MPI speedup is
claimed for this small slab.

At tau=1 the radiation discrepancy is 23.6% and material-temperature
discrepancy 46.0%: this IMC transport calculation has a causal front, whereas
the diffusion reference does not. The early-time comparison is deliberately
plotted and reported, but is not a diffusion-limit pass criterion.

`results/mpi` and `results/serial` retain all three profiles, semianalytic
reference samples, figures, metrics and comparison logs. Reproduce with:

```
python3 compare.py results/mpi --check
python3 compare.py results/serial --check
```

![MPI comparison](results/mpi/comparison.png)
