# DDMC advection: radiation pulse in moving gas

Checks that DDMC radiation is carried by moving gas
(`RadiationIMCParameters::ddmcAdvection`), against IMC on the same problem.

## Problem

Gas fills $x \in [0, 60]$ cm on a $60 \times 3 \times 3$ Voronoi lattice of
1 cm cubes and moves along $+x$ at $v = \beta c$ with $\beta = 0.03$. It
scatters isotropically with $\sigma_s = 8\ \mathrm{cm^{-1}}$, 8 mean free paths
per cell ($\sigma_s \cdot 4V/A = 5.3$, above `ddmcMinCellOpticalDepth = 4`, so
every cell is DDMC eligible). It does not absorb, so nothing is emitted. The
transport is in the comoving frame (`withHydro = true`) and the gas does not
feel the radiation (`noHydroFeedback = true`). All walls are rigid
(`RigidBoundary`), which DDMC supports.

The only energy is a pulse of radiation that is isotropic in the comoving frame:
a Gaussian in $x$ centred at 26 cm with $\sigma_0 = 2$ cm (truncated at
$4\sigma_0$), uniform in $y$ and $z$, $10^{12}$ erg in total.

With $\nabla\cdot v = 0$, the $O(v/c)$ comoving diffusion equation

$$\frac{\partial E_0}{\partial t} + \nabla\cdot(v E_0) + \tfrac13 E_0\,\nabla\cdot v + \nabla\cdot F_0 = 0$$

reduces to $\partial_t E_0 + v\,\partial_x E_0 = D\,\partial_x^2 E_0$ with
$D = c/(3\sigma_s) = 1.25\times10^9\ \mathrm{cm^2\,s^{-1}}$. So the comoving
energy is conserved, the centroid moves with the gas, $x_c(t) = x_c(0) + v t$,
and the variance grows as $\sigma_0^2 + 2 D t$.

The run lasts $t = 4\ \mathrm{cm}/v = 4.45$ ns, in 20 census steps: the gas moves
4 cells while diffusion spreads the pulse by $\sqrt{2Dt} = 3.3$ cm. The pulse
stays more than six standard deviations from the $x$ walls.

Every packet keeps its comoving energy $e_0 = \gamma e\,(1 - v\cdot u/c^2)$ to
roundoff: comoving scattering is elastic, DDMC leaks and advective hops keep the
comoving weight, and reflection at a transverse wall leaves $v\cdot u$ unchanged.
The centroid and variance are $e_0$-weighted moments of the census packet
positions, and the total $e_0$ is the conserved quantity.

## Variants

| Variant | Transport | `ddmcAdvection` | Mesh | Gated |
|---|---|---|---|---|
| `imc` | IMC (`withDDMC = false`) | n/a | static | yes |
| `ddmc_static` | DDMC | true | static | yes |
| `ddmc_moving_mesh` | DDMC | true | lattice and box moved to $x + v t$ before every step | yes |
| `ddmc_no_advection` | DDMC | false | static | no, recorded |

The moving mesh reuses the remesh pattern of `examples/moving_slab`
(`SetBox` + `Build`, then `MeshMovement::UpdateNewCells`). Packets keep their
lab positions, so moving the mesh does not carry radiation by itself; the
advective DDMC leaks do. `ddmc_static` and `ddmc_moving_mesh` must therefore agree.

## Expected values

| Quantity | `imc` | `ddmc_static` | `ddmc_moving_mesh` | `ddmc_no_advection` |
|---|---|---|---|---|
| $\Delta x_c / (v t)$ | 1 | 1 | 1 | about 0 |
| comoving energy change | about $10^{-14}$ | about $10^{-14}$ | about $10^{-14}$ | about $10^{-14}$ |
| variance / $(\sigma_0^2 + 2Dt)$ | about 1.0 | about 1.27 | about 1.48 | about 1.0 |

The statistical error of $\Delta x_c/(vt)$ is about 0.5 % for IMC (40 000
packets) and 0.35 % for DDMC (100 000 packets).

The variance is reported but not gated. $\sigma_0^2 + 2Dt \approx 4.0 + 11.1$
cm$^2$. First-order upwind advection adds the numerical diffusion $v h/2$,
$1.5\,\beta\,\tau_\mathrm{cell} = 0.36$ of $D$, so $v h t = 4$ cm$^2$ more. On the
moving mesh, census positions are resampled in cells that have moved by
$s = v\,\Delta t$ since the previous census, which adds $s(h - s)$ per step,
about 3.2 cm$^2$ in total. The ratio against $\sigma_0^2 + 2(D + vh/2)t$ is also
printed.

## Usage

```bash
./ddmc_advection_pulse [variants] [ddmcPackets] [imcPackets] [steps]
```

- `variants`: `all` (default) or a comma-separated list, e.g. `imc,ddmc_static`
- `ddmcPackets`: packets per DDMC variant (default 100000)
- `imcPackets`: packets in the IMC variant (default 40000)
- `steps`: census steps (default 20)

Serial only. The run takes a few minutes on one core; most of it is the IMC
variant, which scatters each packet about 1100 times.

## Output and check

- stdout: one `DDMC_ADVECTION_PULSE_METRICS variant=... key=value ...` line per
  variant, then a summary table
- `ddmc_advection_pulse_metrics.txt`: the same metrics lines
- `ddmc_advection_pulse_history.txt`: centroid, variance and energy change after
  every step
- `ddmc_advection_pulse_profiles.txt`: final comoving energy fraction in 1 cm
  lab-frame bins

```bash
python3 ../ddmc_advection/check_ddmc_advection.py pulse ddmc_advection_pulse_metrics.txt
```

The check requires, for `imc`, `ddmc_static` and `ddmc_moving_mesh`:

- $|\Delta x_c/(vt) - 1| \le 0.05$
- $|\Delta E_0/E_0| \le 10^{-10}$, counting energy dropped by the moving box
  (expected 0)
- for DDMC, every cell DDMC eligible

and $|\Delta x_c/(vt)|$ of `ddmc_static` and `ddmc_moving_mesh` to agree within
0.03, or within 5 statistical standard deviations if that is larger.
`ddmc_no_advection`, the variances, the cell Peclet numbers and the interior
$|\nabla\cdot v|$ are recorded. `advective_leaks` counts only hops taken on the
host DDMC path; the shared CPU/GPU DDMC kernel, which grey DDMC normally uses,
does not count them.

The regression harness runs the same check through
`check_ddmc_advection_pulse_case` in `../ddmc_advection/regression_checks.sh`
(selected by `CHECK_LIBRARY` in `REGRESSION_INFO`).
