# Densmore 2012 Step-Opacity Benchmark

Heterogeneous step-opacity benchmark from Densmore et al. (2012), Figure 4.

A 1D slab (x in [0, 3] cm) is driven by a 1 keV Planck source at the left
boundary. Two material regions have different opacity strengths:

- x < 2 cm: sigma0 = 10 keV^{3.5}/cm (optically thin)
- x >= 2 cm: sigma0 = 1000 keV^{3.5}/cm (optically thick)

The opacity has strong frequency dependence: sigma(E) = sigma0 / (sqrt(kT) * E^3).
This example uses 30-group frequency-dependent transport with opacity-weighted
Planck emission sampling, matching the original problem specification.

## Usage

```bash
./densmore2012 [Nx] [new_per_cell] [boundary_per_cell]
```

- `Nx` -- number of cells (default 256)
- `new_per_cell` -- photon packets per cell per step (default 50)
- `boundary_per_cell` -- boundary source photon packets per cell (default 100)

## Parameters you can modify

| Parameter | Location | Description |
|---|---|---|
| `Nx`, `new_per_cell`, `boundary_per_cell` | CLI args | Resolution and statistics |
| `dt` | `main.cpp` | Time step size (default 5e-12 s) |
| `tf` | `main.cpp` | Final time (default 1e-9 s) |
| `cvPerVolume` | `main.cpp` | Heat capacity (default 1e15 / keV_K) |
| `CombPopulationControl(grid, 200, 5.0)` | `main.cpp` | Population control target and ratio |

## Comparison

At the end of the run, `plot_densmore.py` is invoked automatically. It
plots the gray MC result against digitized reference data from the Milagro
IMC code (30-group multigroup) and produces:

- `densmore2012.png` / `.pdf` -- temperature profile comparison

The reference data is in `data/densmore2012_fig4_mc.csv` (x in cm, T in keV).

To generate the plot manually:

```bash
python3 plot_densmore.py
```
