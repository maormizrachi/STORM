# RDMont - Monte Carlo Particle Transport Library

![C++ project](https://img.shields.io/badge/C++-2874a6)
![Linux](https://img.shields.io/badge/Linux-0e6655)
![macOS](https://img.shields.io/badge/macOS-27ae60)

RDMont provides a templated Monte Carlo particle transport engine used by the [RICH](https://github.com/maormizrachi/RICH) astrophysical simulation code. It can also be used as a standalone library for radiation transport and other Monte Carlo applications.

## Features

- **Templated on `PointT` and `TessT`** — works with any 3D point type and tessellation
- **Pluggable physics** — `MonteCarloPhysics` interface for custom particle interactions
- **Pluggable boundaries** — rigid, temperature-driven (single/two-sided), or custom BCs
- **Population control** — comb algorithm for photon packet management
- **Serial and MPI managers** — scale from laptop to supercomputer
- **Gray IMC radiation** — `SimpleRadiationPhysics` for standalone radiation transport (Hohlraum, Marshak wave)
- **Header-heavy** — most code is inline in headers for easy integration

## Directory Structure

```
RDMont/
├── RDMontError.hpp             Error handling
├── types.hpp                   Type aliases
├── PhysicalConstants.hpp       Physical constants (CGS)
├── particle/                   Particle, status, step result
├── physics/                    Physics interface + NoPhysics
├── radiation/                  SimpleRadiationPhysics, OpacityModel, RadiationCell
├── boundary/                   Boundary conditions (rigid, temperature)
├── utils/                      RandomOnFace, PlanckIntegral, LinearInterpolation
├── population/                 Population control (comb, no-op)
├── manager/                    Serial + MPI managers
├── io/                         HDF5 particle I/O (optional)
├── examples/                   Standalone examples
├── install_deps.sh             Script to clone external dependencies
└── CMakeLists.txt              Standalone build configuration
```

## External Dependencies

- **Boost** >= 1.74
- **MPI** (optional, for parallel builds)
- **HDF5** (optional, for I/O)

For standalone usage, run:

```bash
./install_deps.sh
```

This clones into `deps/`:
- **[mpi_utils](https://github.com/maormizrachi/mpi_utils)** — MPI serialization and exchange
- **[MadCart](https://github.com/maormizrachi/MadCart)** — Cartesian mesh (for examples)

## Building Standalone

```bash
./install_deps.sh
mkdir build && cd build
cmake .. -DRDMONT_DEPS_DIR=../deps \
         -DRDMONT_BUILD_EXAMPLES=ON
make -j$(nproc)
```

## Using Inside RICH

When used as a submodule inside RICH (at `source/monte/`), the CMakeLists.txt is not used. RICH's own build system compiles the source files directly via `GLOB_RECURSE`.

## License

See the [RICH repository](https://github.com/maormizrachi/RICH) for license information.
