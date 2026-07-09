# Hohlraum Benchmark (Serial)

2D cylindrical hohlraum radiation transport benchmark from McClarren &
Urbatsch (2009), as presented in Steinberg & Heizler (2021),
arXiv:2108.13453, Section 4.2. Matches the setup in `RICH/runs/Elad_paper_hohlraum`.

A cylindrical cavity (x-axis is the symmetry axis) is driven by a 1 keV
Planck source at the left opening. Absorbing material walls and a capsule
heat up as radiation propagates through the vacuum channel.

## Geometry

- Domain: x in [0, 1.4], y in [-0.65, 0.65], z in [-0.65, 0.65] cm
- Left wall: x in [0.10, 0.15], r <= 0.45
- Capsule: x in [0.55, 0.95], r <= 0.45
- Right end cap: x in [1.35, 1.40], r <= 0.65
- Outer wall: x in [0.10, 1.40], r in [0.60, 0.65]

## Material properties

- Material: sigma_a = 300 * (T/keV)^{-3} cm^{-1}, Cv = 3e15 erg/keV/cm^3
- Vacuum: sigma_a ~ 0, Cv = 1e15 erg/keV/cm^3
- Initial temperature: 300 K

## Usage

```bash
./hohlraum [N_base] [new_per_cell] [min_per_cell]
```

- `N_base` -- total Voronoi mesh generator points (default 2000)
- `new_per_cell` -- new photon packets per cell per step (default 5)
- `min_per_cell` -- target packets per cell after population control (default 15)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `N_base`, `new_per_cell`, `min_per_cell` | CLI args | Mesh resolution and statistics |
| `dt` | `main.cpp` | Time step (default 1e-11 s) |
| `numSteps` | `main.cpp` | Number of time steps (default 100) |
| `boundaryPhotonsPerFace` | `main.cpp` | Boundary source intensity (default 1000) |
| `T_drive` | `main.cpp` | Drive temperature (default 1 keV) |
