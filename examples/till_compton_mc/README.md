# Till-Compton Equilibration

Till, McGraw & Warsa Compton equilibration case: a single cell with
$T_\mathrm{mat} = 1\;\mathrm{keV}$, $T_\mathrm{rad} = 10\;\mathrm{keV}$,
fully ionized hydrogen at $\rho = 1\;\mathrm{g/cm^3}$.

The gas and radiation temperatures equilibrate through 32-group
Compton scattering (with induced scattering) and free-free absorption.
The Compton kernel uses the CMMC submodule to build microscopic
group-changing matrices, tabulated over a temperature grid and
interpolated at runtime.

## Usage

```bash
./till_compton_mc [options]
```

| Option | Description |
|---|---|
| `--dt=SECONDS` | Fixed timestep |
| `--tf=SECONDS` | Final time (default 3e-8) |
| `--new-photons=N` | Emitted packets per cell per step (default 10000) |
| `--initial-photons=N` | Initial packets per cell (default 4000) |
| `--matrix-samples=N` | CMMC samples per matrix (default 2000000) |
| `--output-dir=PATH` | Profile and plot output directory |
| `--plasma-cutoff` / `--no-plasma-cutoff` | Enable or disable plasma frequency cutoff |

## Comparison

At the end of the run, `plot_till_compton.py` is invoked automatically. It
plots the gas and radiation temperature evolution against the digitized
IN-FBC reference and produces:

- `till_compton_mc.png` / `.pdf` -- temperature equilibration comparison

The comparison is diagnostic: Monte Carlo noise and the different
transport driver can produce deviations from the reference.

## Example output

<img src="till_compton_mc.png?raw=true" alt="Till-Compton temperature equilibration" width="600"/>
