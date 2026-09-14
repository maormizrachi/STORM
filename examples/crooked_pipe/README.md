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
cmake -S ../.. -B ../../build/storm -DSTORM_WITH_MPI=ON -DSTORM_BUILD_EXAMPLES=ON \
      -DSTORM_WITH_VTK=ON -DSTORM_WITH_HDF5=ON
cmake --build ../../build/storm --target crooked_pipe -j
mpirun -np 32 ./crooked_pipe 2000 2 10 \
    --output-probes crookedpipe_probes.txt \
    --output-dir crooked_pipe_out \
    --snapshot-interval 50
```

Run `./crooked_pipe --help` for production-size, output, and communication-manager
options. The physical endpoint is `1000 ns`; reduced mesh and packet counts are
intended only for regression testing.

Transport tuning options keep the mesh, packet counts, and physical time steps
unchanged:

| Option | Meaning | Default |
| --- | --- | --- |
| `--transport-work-scale` | Multiplier on measured per-cell transport work in mesh balancing | `1` |
| `--rebalance-interval` | Cycles between balance checks after the checks at cycles 1 and 2 | `10` |
| `--transport-inner-steps` | Maximum events per particle in a Kokkos transport wave | `64` |
| `--transport-min-launch` | Target active particles before launching a Kokkos wave; `0` disables holding | `1024` |
| `--transport-hold-skips` | Maximum consecutive polls holding a small batch before forcing progress | `64` |

The last three options require a Kokkos build. The batch target is not a fixed
number of batches or a hard minimum: a small tail still launches after the hold
limit. Increasing the hold limit allows more arrivals to accumulate but can
delay communication-dependent work. Its units are polls, not elapsed time.

For the tested 16-node hybrid configuration, use a Kokkos OpenMP build
(`STORM_WITH_GPU=ON` with an OpenMP-enabled Kokkos installation), allocate one
MPI rank and 16 CPU cores per node, and run:

```bash
export OMP_NUM_THREADS=16 KOKKOS_NUM_THREADS=16
export OMP_PROC_BIND=spread OMP_PLACES=cores GOMP_SPINCOUNT=1000
mpirun --map-by ppr:1:node:PE=16 --bind-to core -np 16 ./crooked_pipe \
    20000 2 10 --boundary-photons 100 --manager p2p --max-steps 120 \
    --transport-work-scale 0.125 --transport-min-launch 4096 \
    --transport-hold-skips 1024 --rebalance-interval 20 \
    --output-probes crookedpipe_probes.txt
```

The MPI reference uses a separate `STORM_WITH_GPU=OFF` build, 16 ranks per
node, one core per rank, and the same mesh, photon counts, and 120-cycle limit.
Keep its default transport work weight and balance interval when reproducing
this comparison.

For RDMA, replace `--manager p2p` with `--manager rdma --rdma-engine ofi`
and use `--transport-work-scale 0.25`. Keep the other settings above. On the
same 16-node, 120-cycle problem, this took 276.317 s with the default 64 inner
steps; adding `--transport-inner-steps 256` took 269.402 s. The MPI RDMA
reference took 243.812 s. These are application cycle-loop times, excluding
initial setup. Both hybrid settings completed using OFI verbs/ibv and reported
zero packet H2D/D2H staging bytes. Network transfers between ranks are still
required. These settings are measured choices for this problem, not new defaults.

For IMC paper runs, `--output-dir` writes a final cell-centered spatial profile
(`crooked_pipe_profile.csv`), periodic 3D VTK snapshots under `vtk/` every
`--snapshot-interval` cycles, and `latest.h5` (overwritten at each snapshot and
at the end). Probe histories remain independent via `--output-probes`. VTK and
HDF5 require a STORM build with `STORM_WITH_VTK=ON` and `STORM_WITH_HDF5=ON`.

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

## Experimental fixed-step RDMA queues

`--manager rdma --rdma-engine ofi --optimized-rdma` enables fixed receive-ring
allocations within each transport step, partial sends with cached credits, and
fair CPU receive scheduling. The experiment uses 65,536 physics events per source
slice and a 5 ms maximum outgoing-batch age checked at cooperative progress calls.
It is **off by default**: the 128-rank measurements did not reach the intended
15–20% reduction in total runtime. See the RICH workspace report at
`docs/rdma128_20260912/` for timings and protocol review.

`--rdma-ring-size N` sets initial receive capacity and `--rdma-ring-limit N` caps
future growth requests. The limit must be at least the allocator's minimum
capacity (50 by default); existing larger allocations are not reduced by this
limit. Rings may grow or shrink only between globally completed steps. A full
ring retains unsent packets and retries later; it does not trigger a mid-step
resize. This bounds ring capacity, not all outgoing host storage.

For controlled experiments, put overrides after `--optimized-rdma`:
`--transport-event-budget N`, `--send-max-age-us N` (zero disables age flushing),
`--rdma-fixed-queues-off`, and `--receive-scheduling-off`. `--rdma-fixed-queues-off` keeps the scheduling experiment;
`--receive-scheduling-off --send-max-age-us 0` keeps only the queue experiment. Scheduling changes particle and
floating-point accumulation order; bitwise-identical probe histories are not
promised.

The targeted queue test can be built with `-DSTORM_BUILD_RDMA_TESTS=ON` and
`cmake --build <build-dir> --target storm_rdma_fixed_queue_test`. Run it on two
ranks with argument `ofi` for native OFI or `mpi` for MPI RMA. It exercises delayed
consumption, wrap, partial sends, repeated grow/shrink epochs, source registration,
and rejection of allocation changes during transport.

## Optional CPU transport optimization

For CPU nodes supporting x86-64-v3 (including AVX2), configure STORM with
`-DSTORM_OPTIMIZE_CPU_TRANSPORT=ON`. This enables x86-64-v3 instructions
and a four-face vector intersection loop while retaining the configured
optimization level (standalone release builds use `-O2`). Floating-point
contraction is explicitly disabled: the architecture-tuned build with the
compiler's default contraction setting failed Voronoi mesh construction.
Fast math and approximate reciprocals are not enabled. The option is off by
default and is independent of the runtime `--optimized-rdma` flag above.

The vector path preserves scalar face order for equal-time hits, handles
parallel faces without dividing by zero, and uses the scalar loop for the
remaining one to three faces. It adds no geometry allocation or RDMA metadata.
Stationary-frame IMC events also reuse their identical attenuation exponential;
moving-frame events retain separate factors when the Doppler shift differs
from one.

Build the differential intersection test with
`-DSTORM_BUILD_CPU_TRANSPORT_TESTS=ON` together with the CPU optimization option,
then build and run `storm_cpu_intersection_test`. The test compares 800,000
scalar/vector intersections, including grazing, parallel, slab, and corner-hit
cases. The follow-up benchmark and review are recorded in the RICH workspace at
`docs/rdma128_followup/`; the subsequent `-O2` comparison is in
`docs/rdma128_o2/`.

### Optional direct-log sampling

`-DSTORM_CPU_DIRECT_LOG=ON` replaces `-log1p(u-1)` with `-log(u)` in the
shared CPU IMC kernel. STORM's 52-bit half-bin uniform format makes `u-1`
exactly representable, so the formulas specify the same exponential variate.
The math-library results can differ in their last rounding bit; this option
is **off by default** and does not promise bitwise-identical histories. It
retains the configured optimization level and introduces no fast-math flags.
GPU builds reject this option because they have not been validated for it.

The current 128-rank investigation, numerical checks, full-invocation timings,
and fixed-queue reallocation review are in the RICH workspace at
`docs/rdma128_55/`. During a fixed transport epoch, RDMA now skips resize
bookkeeping while continuing provider progress; resizing resumes only after
all ranks close the epoch. Payload completion, tail publication, and source
registration lifetimes retain their existing ordering.
