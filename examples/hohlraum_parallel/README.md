# Hohlraum Benchmark (MPI Parallel)

Same physics and geometry as `examples/hohlraum/` (McClarren & Urbatsch 2009),
but using MadVoro's parallel Voronoi construction and STORM's distributed
Monte Carlo transport via `CreateMonteCarloManager()` (default: RDMA with IBV,
falling back to MPI RMA).

## Usage

```bash
mpirun -np N ./hohlraum_parallel [N_base] [new_per_cell] [min_per_cell] [options]
```

### Positional arguments

- `N_base` -- total Voronoi mesh generator points (default 2000)
- `new_per_cell` -- new photon packets per cell per step (default 5)
- `min_per_cell` -- target packets per cell after population control (default 15)

### Options

- `--output-profile <file>` -- write temperature profile CSV to file
- `--output-vtk <dir>` -- write VTK mesh output per step (requires VTK build)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `N_base`, `new_per_cell`, `min_per_cell` | CLI args | Mesh resolution and statistics |
| `dt` | `main.cpp` | Time step (default 1e-11 s) |
| `numSteps` | `main.cpp` | Number of time steps (default 100) |
| `boundaryPhotonsPerFace` | `main.cpp` | Boundary source intensity (default 1000) |
| Manager type | `CreateMonteCarloManager()` call | `ManagerType::RDMA`, `Legacy`, or `P2P` |
| RDMA engine | `CreateMonteCarloManager()` call | `RDMAEngine::IBV`, `MPI`, or `Auto` |

## Output

- Temperature profile CSV (via `--output-profile`)
- VTK files per step (via `--output-vtk`, requires `STORM_WITH_VTK=ON`)
- An existing `plot_profile.py` can plot the CSV:

```bash
python3 plot_profile.py profile.txt --save hohlraum.png
```
