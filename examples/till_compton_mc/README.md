# Till-Compton STORM example

This example ports `regression_tests/cases/till_compton_mc` to the direct
STORM IMC interface. It retains the original 32 logarithmic energy groups,
free-free absorption formula, Compton/induced-scattering settings, rigid
boundary, and timestep schedule.

The Compton kernel is an adapter around RICH's CMMC implementation. CMMC
produces microscopic group-changing matrices; the adapter applies the
fully-ionized-hydrogen electron density and supplies the resulting rates to
STORM. The matrix is rebuilt for each cell and timestep, so the default
2,000,000 samples is intentionally expensive.

Build from the repository's normal CMake configuration with
`STORM_BUILD_EXAMPLES=ON`, then run:

```sh
./source/monte/examples/till_compton_mc/till_compton_mc
```

For a quick smoke test, use a small matrix and packet population:

```sh
./source/monte/examples/till_compton_mc/till_compton_mc \
  --matrix-samples=64 --new-photons=4 --initial-photons=4 \
  --tf=1e-13 --dt=1e-13 --output-dir=/tmp/till-compton-smoke
```

The run writes `till_compton_profile.txt`, compares it with the digitized
IN-FBC reference in `data/in_fbc_reference.txt`, and generates
`till_compton_mc.png` and `till_compton_mc.pdf` in the selected output
directory. The comparison is diagnostic because it includes Monte Carlo
noise and compares two different transport drivers.

The plasma-cutoff behavior from the old case can be selected with
`--plasma-cutoff` or `--no-plasma-cutoff`. The fixed timestep can be supplied
as `--dt=SECONDS`; a bare positive argument is also accepted for compatibility
with the original test.
