# Marshak Wave -- Problem 4 (MPI Parallel)

Derei et al. (2024), Test 3: stretched grid with divergent density profile
rho(x) = x^{-40/139}.

This problem uses a **geometrically stretched mesh** following the formula from
the reference paper: x_i = 10^{-5} + 2.24 x 10^{-2} (1.0075^i - 1), producing
~512 cells with fine resolution near x=0.

Unlike the other Marshak benchmarks, this test uses **MPI parallel transport**
via `CreateMonteCarloManager` (RDMA by default).

See [`../marshak_wave/README.md`](../marshak_wave/README.md) for full description
of the Marshak wave physics, EOS, and opacity models.

## Usage

```bash
mpirun -np N ./marshak_wave_4 [new_per_cell] [boundary_per_cell]
```

- `new_per_cell` -- photon packets per cell per step (default 15)
- `boundary_per_cell` -- boundary source photon packets per cell (default 100)
