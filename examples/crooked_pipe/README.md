# Crooked Pipe

This example implements Graziani's cylindrical Crooked Pipe radiation-transfer
benchmark with STORM's standalone MPI IMC API. It follows section 4.3 of
Steinberg and Heizler, *Astrophysical Journal Supplement Series* **258**, 14
(2022), [arXiv:2108.13453](https://arxiv.org/abs/2108.13453).

The domain is represented in three dimensions with cylindrical material
regions. The thin pipe has absorption opacity `0.2 cm^-1` and volumetric heat
capacity `1e13 erg/(keV cm^3)`. The thick surrounding material uses
`2000 cm^-1` and `1e16 erg/(keV cm^3)`. The initial material temperature is
`0.05 keV`; a `0.5 keV` blackbody drives the thin inlet.

The simulation records material temperature at the five probe locations from
Fig. 6:

1. `(r, z) = (0, 0.25) cm`
2. `(r, z) = (0, 2.75) cm`
3. `(r, z) = (1.25, 3.5) cm`
4. `(r, z) = (0, 4.25) cm`
5. `(r, z) = (0, 6.75) cm`

## Build and run

Configure STORM with MPI and examples enabled:

```bash
cmake -S ../.. -B ../../build/storm -DSTORM_WITH_MPI=ON -DSTORM_BUILD_EXAMPLES=ON
cmake --build ../../build/storm --target crooked_pipe -j
mpirun -np 32 ./crooked_pipe 2000 2 10 --output-probes crookedpipe_probes.txt
```

Run `./crooked_pipe --help` for production-size and communication-manager
options. The physical endpoint is `1000 ns`; reduced mesh and packet counts are
intended only for regression testing.

Accuracy here is limited by mesh resolution rather than by packet counts, and
the resolution that matters is inside the optically thick wall. Radiation that
enters the wall is absorbed in a Marshak boundary layer a few hundredths of a
centimetre deep, so a background-sized cell at the interface spreads that
energy over far too much heat capacity and holds the wave back. `--wall-layers`
places geometrically graded shells on the thick side of every interface, each
twice as thick as the previous one and holding half as many points;
`--wall-width` sets the thickness of the first shell and `--wall-points` its
point count. `--max-dt` caps the time step, which keeps the Fleck factor in the
thin channel close to one.

Going from one shell to four graded shells speeds the heat wave up by roughly a
factor of three at the downstream probes and is by far the most effective knob;
raising `--points` helps as well but converges slowly.

## Published comparison

The files in `data/` were digitized from Steinberg and Heizler Fig. 8(a).
They contain the DIMC, IMC, and overlaid Gentile (2001) material-temperature
histories in keV against logarithmic time in ns. They are approximate plot
digitizations, not author-supplied tables; their expected vertical uncertainty
is about `0.005-0.01 keV`.

Compare a run with the DIMC profile using:

```bash
python3 compare_reference.py crookedpipe_probes.txt
```

The comparison interpolates in logarithmic time and checks each probe's RMSE,
maximum temperature error, and `0.08 keV` heat-arrival time. Limits can be
overridden with `CROOKED_PIPE_MAX_RMSE_KEV`,
`CROOKED_PIPE_MAX_ERROR_KEV`, and
`CROOKED_PIPE_MAX_ARRIVAL_LOG10_ERROR`.
