# STORM - Scalable Three-dimensional Optimized RDMA Monte Carlo

![C++ project](https://img.shields.io/badge/C++-2874a6)
![Linux](https://img.shields.io/badge/Linux-0e6655)
![macOS](https://img.shields.io/badge/macOS-27ae60)

STORM provides a templated Monte Carlo particle transport engine used by the [RICH](https://gitlab.com/eladtan/RICH) astrophysical simulation code. It can also be used as a standalone library for radiation transport and other Monte Carlo applications.

<img src="examples/hohlraum/hohlraum_3D.png?raw=true" alt="3D Hohlraum radiation temperature field computed with STORM's gray IMC solver on a Voronoi mesh." width="600"/>

## Features

- **Templated on `PointT` and `Grid`** — works with any 3D point type and tessellation (e.g. [MadVoro](https://github.com/maormizrachi/MadVoro), [MadCart](https://github.com/maormizrachi/MadCart))
- **Pluggable physics** — `MonteCarloPhysics` interface for custom particle interactions
- **Pluggable boundaries** — rigid, temperature-driven (single/two-sided), or custom BCs
- **Population control** — comb algorithm for photon packet management
- **Serial and MPI managers** — scale from laptop to supercomputer
- **Gray IMC radiation** — `SimpleRadiationPhysics` for standalone radiation transport
- **Benchmark suite** — Hohlraum, Marshak wave (1–4), Densmore 2012 with reference comparisons
- **Header-heavy** — most code is inline in headers for easy integration

## Directory Structure

```
STORM/
├── StormError.hpp              Error handling
├── types.hpp                   Type aliases
├── PhysicalConstants.hpp       Physical constants (CGS)
├── elementary/                 ADL fallback point operators (PointOps.hpp)
├── particle/                   Particle, status, step result
├── physics/                    Physics interface + NoPhysics
├── radiation/                  SimpleRadiationPhysics, OpacityModel, RadiationCell
├── boundary/                   Boundary conditions (rigid, temperature)
├── utils/                      RandomOnFace, LinearInterpolation
├── population/                 Population control (comb, no-op)
├── manager/                    Serial + MPI managers
├── examples/                   Standalone examples and benchmarks
├── install_deps.sh             Script to clone external dependencies
└── CMakeLists.txt              Standalone build configuration
```

## Requirements

- **C++17** compiler (GCC >= 8, Clang >= 7)
- **Boost** >= 1.74
- **OpenMP** (optional, used if available)
- **MPI** >= 2.0 (optional, for distributed-memory parallel builds)
- **HDF5** >= 1.8 with CXX and HL components (optional, for I/O)

## Building Standalone (Serial)

```bash
git clone git@github.com:maormizrachi/STORM.git
cd STORM
./install_deps.sh
mkdir build && cd build
cmake .. -DSTORM_BUILD_EXAMPLES=ON
make -j$(nproc)
```

### Running the Examples

```bash
./examples/serial_cartesian                    # 100 particles in a 10×10×10 Cartesian box
./examples/hohlraum [N_base] [new] [min]       # Gray IMC hohlraum on Voronoi mesh (serial)
./examples/marshak_wave_1 [Nx] [new] [bdy]     # Marshak wave problem 1 (Krief & McClarren Test 2)
./examples/marshak_wave_2 [Nx] [new] [bdy]     # Marshak wave problem 2 (Krief & McClarren Test 3)
./examples/marshak_wave_3 [Nx] [new] [bdy]     # Marshak wave problem 3 (Derei et al. Test 1)
./examples/marshak_wave_4 [Nx] [new] [bdy]     # Marshak wave problem 4 (Derei et al. Test 3)
./examples/densmore2012 [Nx] [new] [bdy]       # Densmore 2012 step-opacity on Cartesian mesh
```

#### Hohlraum

Matches the setup from `RICH/runs/Elad_paper_hohlraum` — McClarren & Urbatsch (2009) cylindrical hohlraum with vacuum BCs, 1 keV Planck drive, and 1000 boundary photons per face.

#### Marshak Wave (4 problems)

Self-similar Marshak wave benchmarks on a 1D Cartesian mesh, comparing MC transport against reference diffusion solutions:

| Problem | Reference | Opacity |
|---|---|---|
| 1 | Krief & McClarren (2024) Test 2 | Non-equilibrium (κ_P = 0.001 κ_R) |
| 2 | Krief & McClarren (2024) Test 3 | Equilibrium (κ_P = κ_R) |
| 3 | Derei et al. (2024) Test 1 | Non-uniform ρ(x), power-law EOS |
| 4 | Derei et al. (2024) Test 3 | Stretched grid, divergent density |

Non-linear EOS is handled by recomputing T from internal energy after each step. The ScatteringOpacity returns κ_R − κ_P so the transport mean free path matches the Rosseland mean.

**Note:** These problems are extremely optically thick and require random walk or DDMC acceleration for practical runtimes. Without acceleration, use small Nx (16–64) for demonstration.

#### Densmore 2012

Heterogeneous step-opacity benchmark from Densmore et al. (2012), Fig. 4. Uses a gray Planck-mean opacity approximation — the original problem requires 30 energy groups for accurate frequency-dependent transport. Results are compared against digitized reference data from the Milagro IMC code.

## Building with MPI

STORM's MPI build enables distributed-memory parallel transport via the managers in `manager/parallel/` (e.g. `TwoSidedMonteCarloManager`, `RDMAMonteCarloManager`). It compiles additional source files (`utils/RankSync.cpp`, `manager/parallel/ReallocationAgent.cpp`, `mpi_utils/AmountManager.cpp`) and defines `STORM_WITH_MPI`, `MADVORO_WITH_MPI`, `RICH_MPI`, `SPATIAL_DS_WITH_MPI`, and `__WITH_MPI`.

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DSTORM_WITH_MPI=ON
make -j$(nproc)
```

The MPI build requires the `mpi_utils`, `MeshDecomposer3D`, and `EasyRMA` dependencies (all cloned by `install_deps.sh`).

### Parallel Hohlraum Example

When `STORM_WITH_MPI=ON`, a parallel hohlraum example (`hohlraum_parallel`) is built automatically. It uses MadVoro's `BuildParallel()` for distributed Voronoi construction and STORM's `TwoSidedMonteCarloManager` for MPI particle transport:

```bash
mpirun -np 4 ./examples/hohlraum_parallel [N_base] [new_per_cell] [min_per_cell]
```

The parallel example generates points identically on all ranks (same seed), lets MadVoro partition the domain via Hilbert-curve load balancing, then runs the same IMC physics as the serial hohlraum with `MPI_Reduce`-based diagnostics on rank 0.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `STORM_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `STORM_WITH_MPI` | `OFF` | Enable MPI distributed-memory support |
| `STORM_WITH_HDF5` | `OFF` | Enable HDF5 I/O |
| `STORM_DEPS_DIR` | `./deps` | Path to external dependencies |

### Build Types

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release   # optimized
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Debug     # debug symbols + assertions
```

## External Dependencies

The `install_deps.sh` script clones external dependencies into `deps/` (override with `cmake .. -DSTORM_DEPS_DIR=/path/to/deps`):

- **[MadCart](https://github.com/maormizrachi/MadCart)** — 3D Cartesian mesh (used by the examples)
- **[MadVoro](https://github.com/maormizrachi/MadVoro)** — 3D Voronoi tessellation (includes a `range/finders` submodule, initialized automatically)
- **[mpi_utils](https://github.com/maormizrachi/mpi_utils)** — MPI serialization and exchange
- **[spatial_ds](https://github.com/maormizrachi/spatial_ds)** — Spatial data structures (OctTree, KDTree)
- **[MeshDecomposer3D](https://github.com/maormizrachi/MeshDecomposer3D)** — Domain decomposition, Hilbert ordering
- **[EasyRMA](https://github.com/maormizrachi/EasyRMA)** — One-sided MPI communication (MPI builds only)
- **[planck_integral](https://github.com/menahemkrief/planck_integral)** — Planck function integrals (Clark 1987)
- **[units](https://github.com/menahemkrief/units)** — Physical constants in CGS (required by planck_integral)

## Using Inside RICH

When used as a submodule inside RICH (at `source/monte/`), the CMakeLists.txt is not used. RICH's own build system compiles the source files directly via `GLOB_RECURSE`.

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
