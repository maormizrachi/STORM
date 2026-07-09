# Serial Cartesian (Minimal Example)

Minimal example demonstrating STORM's particle transport API on a
10x10x10 Cartesian grid with no physics.

100 particles are initialized at the center of the domain and
transported for 10 steps using `MonteCarloManagerSerial` with
`NoPhysics` (free streaming) and `RigidBoundary` (reflecting walls).

## Usage

```bash
./serial_cartesian
```

No command-line arguments. This example is intended as a starting point
for understanding the STORM API: grid construction, boundary conditions,
physics interface, population control, and the manager step loop.
