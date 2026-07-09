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
- **Full IMC radiation** — `RadiationIMC` with Random Walk, DDMC, and multigroup support
- **Benchmark suite** — Hohlraum, Marshak wave (1–4), Densmore 2012, Moving slab
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
├── radiation/                  RadiationIMC, RadiationOpacityModel, RadiationCell
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
- **libibverbs** / **rdma-core** (optional, recommended for MPI builds on InfiniBand clusters)
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

### Examples

Executables are placed inside each example's subdirectory. Each example has its own README with problem description, parameters, and validation instructions.

| Example | Description | Mesh | Mode |
|---|---|---|---|
| [`serial_cartesian`](examples/serial_cartesian/) | Minimal particle transport on a Cartesian grid | Cartesian | Serial |
| [`hohlraum_parallel`](examples/hohlraum_parallel/) | McClarren & Urbatsch (2009) cylindrical hohlraum | Voronoi | MPI |
| [`marshak_wave_1`](examples/marshak_wave_1/) | Marshak wave, Krief & McClarren Test 2 | Cartesian | Serial |
| [`marshak_wave_2`](examples/marshak_wave_2/) | Marshak wave, Krief & McClarren Test 3 | Cartesian | Serial |
| [`marshak_wave_3`](examples/marshak_wave_3/) | Marshak wave, Derei et al. Test 1 | Cartesian | Serial |
| [`marshak_wave_4`](examples/marshak_wave_4/) | Marshak wave, Derei et al. Test 3 (geometric mesh) | Cartesian | Serial / MPI |
| [`densmore2012`](examples/densmore2012/) | Densmore et al. (2012) step-opacity benchmark | Cartesian | Serial |
| [`moving_slab`](examples/moving_slab/) | McClarren & Gentile (2021) 124-group moving slab | Voronoi | MPI |

## Building with MPI

STORM's MPI build enables distributed-memory parallel transport via the managers in `manager/parallel/` (e.g. `TwoSidedMonteCarloManager`, `RDMAMonteCarloManager`). It compiles additional source files (`utils/RankSync.cpp`, `manager/parallel/ReallocationAgent.cpp`, `mpi_utils/AmountManager.cpp`) and defines `STORM_WITH_MPI`, `MADVORO_WITH_MPI`, `RICH_MPI`, `SPATIAL_DS_WITH_MPI`, and `__WITH_MPI`.

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DSTORM_WITH_MPI=ON
make -j$(nproc)
```

The MPI build requires the `mpi_utils`, `MeshDecomposer3D`, and `EasyRMA` dependencies (all cloned by `install_deps.sh`).

### Building with IBV (Recommended for InfiniBand Clusters)

On systems with InfiniBand hardware, enabling IBV allows `RDMAMonteCarloManager` to use native InfiniBand Verbs for one-sided RDMA communication, which significantly reduces latency compared to MPI two-sided or MPI RMA fallbacks. Install `libibverbs-dev` (Debian/Ubuntu) or `rdma-core-devel` (RHEL/Fedora), then:

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DSTORM_WITH_MPI=ON -DSTORM_WITH_IBV=ON
make -j$(nproc)
```

When `STORM_WITH_IBV=OFF` (default), the `RDMAMonteCarloManager` falls back to MPI RMA windows automatically.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `STORM_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `STORM_WITH_MPI` | `OFF` | Enable MPI distributed-memory support |
| `STORM_WITH_IBV` | `OFF` | Enable IBV (InfiniBand Verbs) RDMA transport (requires MPI) |
| `STORM_WITH_VTK` | `OFF` | Enable VTK mesh output (requires VTK >= 9.3) |
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

## Contact

For questions, suggestions, or help getting started, feel free to reach out — I'm always happy to help:

**Maor Mizrachi** — [maormiz@cs.huji.ac.il](mailto:maormiz@cs.huji.ac.il)

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
