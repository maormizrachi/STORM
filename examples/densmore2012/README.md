# Densmore 2012 Step-Opacity Benchmark

Heterogeneous step-opacity benchmark from Densmore et al. (2012), Figure 4.

A 1D slab ($x \in [0, 3]$ cm) is driven by a 1 keV Planck source at the left
boundary. Two material regions have different opacity strengths:

- $x < 2$ cm: $\sigma_0 = 10\;\mathrm{keV}^{3.5}/\mathrm{cm}$ (optically thin)
- $x \geq 2$ cm: $\sigma_0 = 1000\;\mathrm{keV}^{3.5}/\mathrm{cm}$ (optically thick)

The opacity has strong frequency dependence: $\sigma(E) = \sigma_0 / (\sqrt{kT} \cdot E^3)$.
This example uses 30-group frequency-dependent transport. CPU thermal
re-emission uses opacity-weighted Planck sampling. The current GPU transport
selects the group from the thermal CDF but samples uniformly within that group;
this pre-existing difference must be resolved before claiming CPU/GPU physics
parity. The slab and communication optimizations do not change that sampling.

## Usage

The example uses analytical transverse reflection by default. The material
and source are uniform in y/z, so transport can fold those coordinates at the
next x-crossing, scattering, or census event instead of visiting every narrow
box wall. Three-dimensional directions and the existing conservative random
walk remain enabled. Set `STORM_DENSMORE_EXPLICIT_WALLS=1` to compare with the
original wall-by-wall transport. This changes random-number consumption;
compare conserved quantities and temperature statistics, not packet identities.

For a 100-cycle comparison at the default timestep (final time 0.5 ns):

```bash
./densmore2012 1024 16 100 comb 100 1.0 1.0 0
```

Both non-MPI CPU and non-MPI GPU builds instantiate the concrete IMC physics
type. The GPU build uses one device. The literature reference plotted by the
example is at 1 ns and must not be used as a correctness threshold for this
shortened run.

Multi-rank GPU communication overlap is opt-in through
`MonteCarloConfig::gpuOverlapCommunication`. Set `STORM_DENSMORE_OVERLAP=1`
in this example to enable it; `STORM_DENSMORE_NO_OVERLAP=1` overrides that
setting for a synchronous comparison. The four-rank Densmore check showed
about 1.5% overhead, so overlap is not enabled globally by default.
Serial runs disable network overlap because there are no remote ranks.

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

## Example output

<img src="densmore2012.png?raw=true" alt="Densmore 2012 temperature profile comparison" width="600"/>
