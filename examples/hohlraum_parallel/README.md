# Hohlraum Benchmark (MPI Parallel)

2D cylindrical hohlraum from McClarren & Urbatsch (2009), using MadVoro's
parallel Voronoi construction and STORM's distributed Monte Carlo transport
via `CreateMonteCarloManager()` (default: native RDMA through OFI/libfabric,
falling back to two-sided MPI).

## Usage

```bash
mpirun -np 512 ./hohlraum_parallel [N_base] [new_per_cell] [min_per_cell] [options]
```

> [!TIP]
> This benchmark is recommended to be run on **512 MPI processes** (`mpirun -np 512`).

### Positional arguments

- `N_base` -- total Voronoi mesh generator points (default 15000)
- `new_per_cell` -- new photon packets per cell per step (default 5)
- `min_per_cell` -- target packets per cell after population control (default 15)

### Options

- `--output-profile <file>` -- write temperature profile CSV to file (default: `profile.txt`)
- `--output-vtk <dir>` -- write VTK mesh output per step (requires VTK build)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `N_base`, `new_per_cell`, `min_per_cell` | CLI args | Mesh resolution and statistics |
| `dt` | `main.cpp` | Time step (default 1e-11 s) |
| `numSteps` | `main.cpp` | Number of time steps (default 100) |
| `boundaryPhotonsPerCell` | `main.cpp` | Boundary source intensity (default 1000) |
| Manager type | `CreateMonteCarloManager()` call | `ManagerType::RDMA`, `Legacy`, or `P2P` |
| RDMA engine | `CreateMonteCarloManager()` call | `RDMAEngine::OFI`, `IBV`, `MPI`, or `Auto` |

## Output

At the end of the run, `plot_profile.py` is invoked automatically to produce:

- `hohlraum_profile.png` / `.pdf` -- temperature profile along the x-axis

To regenerate the plot manually:

```bash
python3 plot_profile.py profile.txt --save hohlraum_profile.png
```

## Example output

<img src="hohlraum_profile.png?raw=true" alt="Hohlraum temperature profile" width="600"/>
