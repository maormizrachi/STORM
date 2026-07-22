# STORM - Scalable Three-dimensional Optimized RDMA Monte Carlo

![C++ project](https://img.shields.io/badge/C++-2874a6)
![Linux](https://img.shields.io/badge/Linux-0e6655)
![macOS](https://img.shields.io/badge/macOS-27ae60)

STORM provides a templated Monte Carlo particle transport engine used by the [RICH](https://gitlab.com/eladtan/RICH) astrophysical simulation code. It can also be used as a standalone library for radiation transport and other Monte Carlo applications.

<img src="examples/hohlraum_parallel/hohlraum_3D.png?raw=true" alt="3D Hohlraum radiation temperature field computed with STORM's gray IMC solver on a Voronoi mesh." width="600"/>

## Features

- **Templated on `PointT` and `Grid`** - works with any 3D point type and tessellation (e.g. [MadVoro](https://github.com/maormizrachi/MadVoro), [MadCart](https://github.com/maormizrachi/MadCart))
- **Pluggable physics** - `MonteCarloPhysics` interface for custom particle interactions
- **Pluggable boundaries** - rigid, temperature-driven (single/two-sided), or custom BCs
- **Population control** - comb algorithm for photon packet management
- **Serial and MPI managers** - scale from laptop to supercomputer
- **Full IMC radiation** - `RadiationIMC` with Random Walk, DDMC, and multigroup support
- **Benchmark suite** - Hohlraum, Marshak wave (1–4), Densmore 2012, Moving slab, Till-Compton
- **Header-heavy** - most code is inline in headers for easy integration

## Directory Structure

```
STORM/
├── StormError.hpp              Error handling
├── types.hpp                   Type aliases
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
- **libfabric** / OFI (optional, recommended for native RDMA on Slingshot, InfiniBand verbs, and other RDMA fabrics)
- **libibverbs** / **rdma-core** (optional, only for the legacy explicit IBV backend)
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
| [`densmore2012`](examples/densmore2012/) | Densmore et al. (2012) step-opacity benchmark | Cartesian | Serial / MPI |
| [`moving_slab`](examples/moving_slab/) | McClarren & Gentile (2021) 124-group moving slab | Voronoi | MPI |
| [`till_compton_mc`](examples/till_compton_mc/) | Till, McGraw & Warsa Compton equilibration | Cartesian | Serial |

The Till-Compton example uses the CMMC submodule at `deps/CMMC`. Clone STORM
with submodules, or initialize it explicitly with
`git submodule update --init --recursive`.

## Building with MPI

STORM's MPI build enables distributed-memory parallel transport via the managers in `manager/parallel/` (e.g. `TwoSidedMonteCarloManager`, `RDMAMonteCarloManager`). It compiles additional source files (`utils/RankSync.cpp`, `manager/parallel/ReallocationAgent.cpp`, `mpi_utils/AmountManager.cpp`) and defines `STORM_WITH_MPI`, `MADVORO_WITH_MPI`, `RICH_MPI`, `SPATIAL_DS_WITH_MPI`, and `__WITH_MPI`.

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DSTORM_WITH_MPI=ON
make -j$(nproc)
```

The MPI build requires the `mpi_utils`, `MeshDecomposer3D`, and `EasyRMA` dependencies (all cloned by `install_deps.sh`).

`RDMAMonteCarloManager` uses OFI/libfabric as the default native one-sided RDMA backend. On Slingshot it selects hardware providers such as CXI; on InfiniBand it uses the OFI verbs provider rather than the older direct IBV implementation. Software transports such as TCP or sockets are rejected for the native RDMA path. If OFI is unavailable, the high-level auto factory falls back to two-sided MPI transport.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `STORM_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `STORM_WITH_MPI` | `OFF` | Enable MPI distributed-memory support |
| `STORM_WITH_VTK` | `OFF` | Enable VTK mesh output (requires VTK >= 9.3) |
| `STORM_WITH_HDF5` | `OFF` | Enable HDF5 I/O |
| `STORM_DEPS_DIR` | `./deps` | Path to external dependencies |
| `STORM_CMMC_DIR` | `deps/CMMC/src` | Override the pinned CMMC source directory |

### Build Types

```bash
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release   # optimized
cmake .. -DSTORM_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Debug     # debug symbols + assertions
```

## Regression Tests

The regression test suite is provided by the THUNDER submodule, which builds
STORM, runs benchmarks, and validates results against reference data. Tests
are auto-discovered from directories under the configured roots containing a
`REGRESSION_INFO` file.

### Quick Start

```bash
# Serial tests only
./regression_tests/run_all.sh

# All tests (serial + MPI)
./regression_tests/run_all.sh --with-mpi

# Single test
./regression_tests/run_all.sh --with-mpi --test hohlraum_parallel

# Inspect resolved tests and commands
./regression_tests/run_all.sh --list-tests
./regression_tests/run_all.sh --dry-run
```

### SLURM Integration

All tests are submitted as SLURM batch jobs via `sbatch`. Serial tests run with a single task (`-n 1`); MPI tests run with their configured task count. By default the system's default partition is used. Override with:

```bash
./regression_tests/run_all.sh --with-mpi --partition=my_partition
```

### Test Inventory

| Test | Mode | SLURM Resources |
|---|---|---|
| `densmore2012` | MPI | 4 tasks |
| `marshak_wave_1` | serial | 1 task |
| `marshak_wave_2` | serial | 1 task |
| `marshak_wave_3` | serial | 1 task |
| `serial_cartesian` | serial | 1 task |
| `marshak_wave_4` | MPI | 16 tasks |
| `moving_slab` | MPI | 48 tasks, cyclic distribution |
| `hohlraum_parallel` | MPI | 512 tasks |
| `cartesian_parallel_check` | MPI | 4 tasks |
| `till_compton_mc` | serial | 1 task |

### Options

| Flag | Description |
|---|---|
| `--with-mpi` | Include MPI tests and compile with MPI support |
| `--mode <serial\|mpi\|all>` | Choose which test category to run (default: serial) |
| `--partition <name>` | SLURM partition for node allocation (default: system default) |
| `--test <id>` | Run only a specific test |
| `--build-type <type>` | CMake build type: Release, Debug, RelWithDebInfo |
| `--nproc <N>` | Override `make -j` parallelism |
| `--keep-artifacts` | Retain logs even when all tests pass |
| `--verbose` | Stream test output to terminal |
| `--recheck` | Re-run only the check step for `--test` (no build/run) |
| `--clean-results` | Delete `regression_results/` and exit |
| `--dry-run` | Print configure, build, and run commands without executing |

By default, pressing Ctrl+C (or sending SIGTERM/SIGHUP) cancels all running SLURM jobs. Use `--nohup` to let submitted jobs continue running after the script is interrupted.

A `summary.txt` file with the pass/fail status of every test is written to the results directory (`regression_results/<timestamp>/summary.txt`).

For instructions on adding a new regression test, see
[`regression_tests/THUNDER/README.md`](regression_tests/THUNDER/README.md) and
[`examples/REGRESSION_TESTS.md`](examples/REGRESSION_TESTS.md).

## External Dependencies

The `install_deps.sh` script clones external dependencies into `deps/` (override with `cmake .. -DSTORM_DEPS_DIR=/path/to/deps`):

- **[MadCart](https://github.com/maormizrachi/MadCart)** - 3D Cartesian mesh (used by the examples)
- **[MadVoro](https://github.com/maormizrachi/MadVoro)** - 3D Voronoi tessellation (includes a `range/finders` submodule, initialized automatically)
- **[mpi_utils](https://github.com/maormizrachi/mpi_utils)** - MPI serialization and exchange
- **[spatial_ds](https://github.com/maormizrachi/spatial_ds)** - Spatial data structures (OctTree, KDTree)
- **[MeshDecomposer3D](https://github.com/maormizrachi/MeshDecomposer3D)** - Domain decomposition, Hilbert ordering
- **[EasyRMA](https://github.com/maormizrachi/EasyRMA)** - One-sided MPI communication (MPI builds only)
- **[planck_integral](https://github.com/menahemkrief/planck_integral)** - Planck function integrals (Clark 1987)
- **[units](https://github.com/menahemkrief/units)** - Physical constants in CGS (required by planck_integral)
- **[CMMC](https://github.com/menahemkrief/CMMC)** - pinned at `091281f09257c4a5127fa5c3af237d4caf8ebfad`

## Using Inside RICH

When used as a submodule inside RICH (at `source/monte/`), the CMakeLists.txt is not used. RICH's own build system compiles the source files directly via `GLOB_RECURSE`.

## Contact

For questions, suggestions, or help getting started, feel free to reach out - I'm always happy to help:

**Maor Mizrachi** - [maormiz@cs.huji.ac.il](mailto:maormiz@cs.huji.ac.il)

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
