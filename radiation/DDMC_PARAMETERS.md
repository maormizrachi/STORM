# DDMC parameters in `RadiationIMCParameters`

These flags live in `radiation/RadiationIMCParameters.hpp` and are read by `RadiationIMC`. They do nothing unless `withDDMC` is true. Compton and DDMC cannot be combined: construction throws if both are on. DDMC also requires a boundary-condition object so every external face can be classified.

Grey DDMC is the default. Multigroup (partial-group random walk / group-cutoff) DDMC is a separate opt-in.

---

## `withDDMC` (default `false`)

**What it does.** Master switch. When true, optically thick cells can treat a packet as a cell-resident diffusion particle instead of tracing a microscopic IMC ray. Leak, census, and IMC–DDMC conversion use Densmore/Wollaeger interface probabilities.

**When to change.** Turn on for cells whose optical depth is large enough that IMC (even with random walk) spends most of its time on effective scatters. Leave off for streaming, optically thin, or unresolved radiation fronts: DDMC leak into a much thicker neighbor can stall a Marshak-like wave. Incompatible with Compton.

---

## `ddmcMinCellOpticalDepth` (default `15`)

**What it does.** Cell-level eligibility. After precompute, a cell is DDMC-eligible only if

\[
\sigma_{\mathrm{gate}}\,\ell \ge \texttt{ddmcMinCellOpticalDepth},
\]

where \(\ell = 4V/A\) is the mean chord length and \(\sigma_{\mathrm{gate}}\) is the grey total opacity, or the Planck-weighted total opacity of the diffusive group band when multigroup DDMC is on. The cell must also have a positive diffusion coefficient \(c/(3\sigma)\). Unsupported domain-boundary faces and (in post-process) interior source cells can still veto eligibility.

**When to change.**
- **Lower** (e.g. 5–10) to put more of the mesh in DDMC and cut cost. Accuracy at IMC–DDMC interfaces and at a sharp front gets worse.
- **Raise** (e.g. 25–50) if the front is under-resolved or DDMC is admitting cells that should still be IMC. The usual 15 is the Cleveland/Wollaeger-style “several mean free paths per cell” rule.

---

## `ddmcExternalSourceMinFaceOpticalDepth` (default `5`)

**What it does.** Used only in **post-process external-source** mode (photosphere / imposed flux faces). For each source face, \(\tau = \sigma_{\mathrm{diff}}\,d\) is the optical depth from the generator to that face. If \(\tau\) is below this value (or not finite), that cell is stripped of DDMC eligibility so the source is not thermalized inside a thin DDMC cell.

**When to change.** Irrelevant for ordinary hydro+IMC (including converging Marshak). In post-process, **raise** if surface sources are being swallowed by DDMC; **lower** only if the emitting layer is well resolved in optical depth and you want DDMC there.

---

## `ddmcUseMovingInterfaceCorrection` (default `true`)

**What it does.** At an IMC→DDMC face, if hydro (or post-process cell velocities) is on, apply the Gentile-style moving-interface weight factor \(G(\mu,\beta_n)\) before the static Densmore admission probability. \(\beta_n\) is the face-normal velocity over \(c\).

**When to change.** Leave true whenever material is moving. Set **false** for a static mesh, or to isolate whether a bias comes from the moving correction. Marshak-with-`withHydro=false` never takes this branch.

---

## `ddmcMaxInterfaceVelocityOverC` (default `0.1`)

**What it does.** If \(|\beta_n|\) exceeds this, the moving correction is **not** applied. The packet keeps IMC transport across that face (`bypassCellID`) instead of a possibly huge or invalid \(G\).

**When to change.** Only with hydro. **Lower** (e.g. 0.01) if even modest face speeds produce wild weights. **Raise** only if you have validated \(G\) at larger \(\beta\) and need DDMC across faster interfaces. Relativistic faces should stay in IMC.

---

## `ddmcMaxMovingInterfaceWeightCorrection` (default `10`)

**What it does.** After \(G\) is computed, if \(G\le 0\), non-finite, or \(G >\) this bound, the crossing is the same IMC bypass as above. Caps how much the moving correction may amplify packet weight.

**When to change.** **Lower** if you see rare but huge interface weights. **Raise** only with a reason: larger \(G\) is unbiased in the formula but noisy. Static problems never hit this.

---

## `ddmcInterfaceTargetWeightRatio` (default `2`)

**What it does.** After the moving factor, the admitted energy may be split into several DDMC packets so that each copy’s weight is about `ratio * |original weight|`. Required copy count is

\[
N = \left\lceil\frac{|w_{\mathrm{face}}G|}{\texttt{ratio}\cdot|w|}\right\rceil.
\]

**When to change.** **Lower** toward 1 to split more (flatter weights, more packets). **Raise** to split less (cheaper, heavier DDMC particles). Coupled to `ddmcMaxInterfaceSplits`.

---

## `ddmcMaxInterfaceSplits` (default `64`)

**What it does.** If the required copy count above exceeds this (minimum 1), the interface **bypasses DDMC** and the packet stays IMC rather than spawning tens of copies.

**When to change.** **Lower** if interface splits explode the particle count. **Raise** only if you must stay in DDMC for large \(G\) and can afford the extra packets. If splits always hit the cap, fix the moving-interface bounds instead of raising this blindly.

---

## `withMultigroupDDMC` (default `false`)

**What it does.** Enables **partial-group** DDMC: the lowest consecutive groups whose cells are optically thick form a diffusive band with a `groupCutoff`. Packets above the cutoff stay IMC; upscatter out of the band returns to IMC. Requires `withMultigroupOpacity`. If multigroup opacity is on and this flag is off, construction throws. Grey problems (`withMultigroupOpacity=false`) must leave this false.

**When to change.** Set **true** for frequency-dependent opacities where only the low groups are diffusive (Densmore-style, moving slab). Leave **false** for grey IMC/DDMC, including the converging Marshak driver.

---

## `ddmcMaxGroupCutoff` (default `NumGroups`)

**What it does.** Upper bound on the diffusive band length. Precompute walks groups from the bottom until a group fails the cell optical-depth test, then `groupCutoff = min(that, ddmcMaxGroupCutoff)`. Packets with frequency \(\ge E_{\mathrm{cutoff}}\) are not admitted to DDMC.

**When to change.** Only with `withMultigroupDDMC`. **Lower** if high groups are numerically “thick” but physically streaming, or to keep DDMC on a known IR subset. Leave at `NumGroups` to use the automatic cutoff.

---

## `ddmcInterfaceDiagnostics` (default `false`)

**What it does.** Records per-face IMC–DDMC events (incident, admitted, reflected, bypass, frequency reject, DDMC↔DDMC, DDMC→IMC) with energy and \(\mu\) sums, and can dump them as a TSV. Production transport is unchanged aside from the extra bookkeeping.

**When to change.** Turn **on** only when debugging an interface (admission too small, stalling front, weight spikes). Leave **off** in production; the maps and dumps are not free at scale.
