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
- **Gray IMC radiation** — `SimpleRadiationPhysics` for standalone radiation transport (Hohlraum, Marshak wave)
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
├── examples/                   Standalone examples (serial_cartesian, hohlraum)
├── install_deps.sh             Script to clone external dependencies
└── CMakeLists.txt              Standalone build configuration
```

## External Dependencies

- **Boost** >= 1.74
- **C++17** compiler
- **MPI** (optional, for parallel builds)
- **HDF5** (optional, for I/O)

## Building Standalone

```bash
git clone git@github.com:maormizrachi/STORM.git
cd STORM
./install_deps.sh
mkdir build && cd build
cmake .. -DSTORM_BUILD_EXAMPLES=ON
make -j$(nproc)
```

The `install_deps.sh` script clones into `deps/` (the default `STORM_DEPS_DIR`):
- **[MadCart](https://github.com/maormizrachi/MadCart)** — 3D Cartesian mesh (used by the examples)
- **[MadVoro](https://github.com/maormizrachi/MadVoro)** — 3D Voronoi tessellation
- **[mpi_utils](https://github.com/maormizrachi/mpi_utils)** — MPI serialization and exchange
- **[spatial_ds](https://github.com/maormizrachi/spatial_ds)** — Spatial data structures (OctTree, KDTree)
- **[MeshDecomposer3D](https://github.com/maormizrachi/MeshDecomposer3D)** — Domain decomposition, Hilbert ordering
- **[EasyRMA](https://github.com/maormizrachi/EasyRMA)** — One-sided MPI communication (MPI builds only)
- **[planck_integral](https://github.com/menahemkrief/planck_integral)** — Planck function integrals (Clark 1987)
- **[units](https://github.com/menahemkrief/units)** — Physical constants in CGS (required by planck_integral)

To use a custom deps location: `cmake .. -DSTORM_DEPS_DIR=/path/to/deps`

### Running the Examples

```bash
./examples/serial_cartesian   # 100 particles in a 10x10x10 Cartesian box
./examples/hohlraum           # Radiation-driven hohlraum (gray IMC)
```

## Using Inside RICH

When used as a submodule inside RICH (at `source/monte/`), the CMakeLists.txt is not used. RICH's own build system compiles the source files directly via `GLOB_RECURSE`.

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
