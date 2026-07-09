# Marshak Wave -- Problem 4

Derei et al. (2024), Test 3: stretched grid with divergent density profile
$\rho(x) = x^{-40/139}$.

This problem uses a **geometrically stretched mesh** following the formula from
the reference paper: $x_i = 10^{-5} + 2.24 \times 10^{-2}(1.0075^i - 1)$, producing
~512 cells with fine resolution near $x = 0$.

Supports both serial and MPI-parallel execution. When built with MPI, uses
`CreateMonteCarloManager` (RDMA by default).

See [`../marshak_wave/README.md`](../marshak_wave/README.md) for full description
of the Marshak wave physics, EOS, and opacity models.

## Usage

```bash
./marshak_wave_4 [new_per_cell] [boundary_per_cell]
mpirun -np 16 ./marshak_wave_4 [new_per_cell] [boundary_per_cell]
```

Running with 16 MPI ranks is recommended for this problem.

- `new_per_cell` -- photon packets per cell per step (default 15)
- `boundary_per_cell` -- boundary source photon packets per cell (default 100)

## Example output

<img src="marshak_wave_4.png?raw=true" alt="Marshak wave problem 4 temperature profile" width="600"/>
