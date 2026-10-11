# DDMC advection: uniform radiation across a moving IMC-DDMC interface

Checks that uniformly moving gas carries uniform radiation from a DDMC region
into an IMC region without a jump or dip at the interface.

## Problem

Gas fills $x \in [0, 70]$ cm, one cell of $1 \times 2 \times 2$ cm across
(70 Cartesian cells, mean chord $4V/A = 1$ cm), and moves along $+x$ at
$v = \beta c$ with $\beta = 0.03$. It scatters isotropically and does not absorb.

| Region | Cells | $\sigma_s$ | $\sigma_s \cdot 4V/A$ | Transport |
|---|---|---|---|---|
| $x < 30$ cm | 30 | 8 cm$^{-1}$ | 8 | DDMC (`ddmcMinCellOpticalDepth = 4`) |
| $x > 30$ cm | 40 | 1 cm$^{-1}$ | 1 | IMC |

The gas carries radiation from the DDMC region into the IMC region. Transport is
in the comoving frame (`withHydro = true`, `noHydroFeedback = true`) and the walls
are rigid (`RigidBoundary`).

The radiation starts uniform and isotropic in the comoving frame,
$E_0 = 10^{12}$ erg cm$^{-3}$, with 4000 packets per cell. Uniform comoving
radiation in uniform flow is a steady state of the $O(v/c)$ comoving equations,
so the profile must stay flat. Inside the DDMC region the upwind advection leaks
keep it flat. DDMC-to-IMC leaks carry no advection, so at the interface the
$O(v/c)$ balance comes from the moving-interface correction applied to
IMC-to-DDMC packets.

The run lasts $t = 5\ \mathrm{cm}/v = 5.56$ ns in 20 census steps. Uniform flow
through a rigid wall depletes the radiation at the upstream wall ($x = 0$) and
piles it up at the downstream wall ($x = 70$ cm). Neither reaches the
interface:

- The upstream depletion front is at $vt = 5$ cm with width
  $\sqrt{2D_\mathrm{thick}t} = 3.7$ cm, more than five widths from $x = 26$ cm.
- The downstream pile-up spreads by $\sqrt{2D_\mathrm{thin}t} = 10.5$ cm, so it
  is about $10^{-3}$ of its wall value at $x = 34$ cm.

## Variants

| Variant | $\beta$ | `ddmcAdvection` | `ddmcUseMovingInterfaceCorrection` | Gated |
|---|---|---|---|---|
| `moving_correction` | 0.03 | true | true (defaults) | yes |
| `no_moving_correction` | 0.03 | true | false | no, recorded |
| `no_advection` | 0.03 | false | true | no, recorded |
| `static_control` | 0 | true | true | no, recorded |

`static_control` separates any static interface artifact from the advection
terms.

## Measurement and expected values

The profile is the time-averaged radiation energy density tally
(`RadiationIMC::getEradTimeAvg`), averaged over the steps after 40 % of the run
(12 of 20 steps) and divided by the initial $E_0$. The tally is comoving in
DDMC cells and lab-frame in IMC cells; they differ at $O(\beta^2)$, about
0.1 %. The comoving census energy density is reported alongside.

The metrics are:

- `max_dev_window`: $\max|E/E_0 - 1|$ over the 4 cells on each side of the
  interface
- `e_ratio_interface`: $E$ in the last DDMC cell over $E$ in the first IMC cell
- `window_profile`: the 8 window values
- `thick_bulk_rel`, `thin_bulk_rel`: bulk reference means away from the
  interface and the walls

| Variant | Expected |
|---|---|
| `moving_correction` | flat: `max_dev_window` at the noise level (about 1 %), ratio about 1 |
| `static_control` | flat |
| `no_advection` | a dip of tens of percent: the IMC side carries the radiation away but the DDMC side does not supply it |
| `no_moving_correction` | an $O(\beta)$ interface imbalance, likely a few to tens of percent |

## Usage

```bash
./ddmc_advection_interface [variants] [packetsPerCell] [steps]
```

- `variants`: `all` (default) or a comma-separated list, e.g.
  `moving_correction,static_control`
- `packetsPerCell`: initial packets in every cell (default 4000)
- `steps`: census steps (default 20)

Serial only. The run takes a few minutes on one core; most of it is IMC in the
thin region.

## Output and check

- stdout: one `DDMC_ADVECTION_INTERFACE_METRICS variant=... key=value ...` line
  per variant, the window values after every step, and a summary
- `ddmc_advection_interface_metrics.txt`: the same metrics lines
- `ddmc_advection_interface_profiles.txt`: averaged tally and census profiles,
  with the thick/thin flag and DDMC eligibility of every cell

```bash
python3 ../ddmc_advection/check_ddmc_advection.py interface ddmc_advection_interface_metrics.txt
```

The check requires:

- in every variant, DDMC eligibility exactly matches the thick region, so the
  interface is where the test says it is
- for `moving_correction`, `max_dev_window` $\le 0.03$

The other variants are recorded. The moving-interface bypass count and largest
correction factor are reported too. Like `advective_leaks`, they are counted on
the host DDMC path only, so they can read zero when the shared DDMC kernel runs.

The regression harness runs the same check through
`check_ddmc_advection_interface_case` in `../ddmc_advection/regression_checks.sh`
(selected by `CHECK_LIBRARY` in `REGRESSION_INFO`).
