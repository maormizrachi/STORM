#!/usr/bin/env python3
"""Plot the Hillier polarization benchmark: every observer against the thin limit.

Reads the per-observer dumps written by the example and renders an SVG showing
q for all observer directions versus inclination, with the analytic
single-scattering curve

    q(i) = -(3/13) tau_bar sin^2 i

overlaid.  The spherical control is drawn on the same axes, where it should lie
flat on zero.

SVG is emitted directly rather than through matplotlib: STORM regression runs on
machines where matplotlib is not installed, and a plot that only works on some of
them is worse than one with no dependencies.  numpy is the only import.

Usage:
    plot_hillier.py [out.svg] [prolate.txt] [spherical.txt]
"""

import glob
import os
import sys

import numpy as np

W, H = 900, 600
L, R, T, B = 84, 34, 92, 114
PW, PH = W - L - R, H - T - B

PROLATE_COLOUR = "#256abf"
SPHERICAL_COLOUR = "#eb6834"
CURVE_COLOUR = "#0b0b0b"
# The ring mean must not read as just another detector, so it gets its own
# hue rather than a larger version of the scatter colour.
MEAN_COLOUR = "#7b2d8e"


def resolve(path, pattern):
    """Return \a path if it exists, else the first file matching \a pattern."""
    if os.path.exists(path):
        return path
    matches = sorted(glob.glob(os.path.join(os.path.dirname(path) or ".", pattern)))
    return matches[0] if matches else None



def ring_means(inc, q, energy, tol=0.5):
    """Pooled q and its shot noise for each distinct inclination present.

    Individual detectors scatter widely -- each is one direction's worth of
    photons -- so the quantity the test actually gates on, the pooled ring value,
    is not visible in the scatter alone.  Returns [(inclination, q, sigma, n)].
    """
    out = []
    for centre in sorted(set(np.round(inc / tol) * tol)):
        sel = np.abs(inc - centre) <= tol
        w = energy[sel]
        if not sel.any() or w.sum() <= 0.0:
            continue
        big_q = q[sel] * w
        n = int(sel.sum())
        var = max(0.0, (big_q ** 2).sum() - big_q.sum() ** 2 / n)
        out.append((centre, big_q.sum() / w.sum(), np.sqrt(var) / w.sum(), n))
    return out


def read_dump(path):
    """Return (inclination_deg, q, u, energy, tau_bar, geometry) from one dump."""
    tau_bar, geometry = float("nan"), "unknown"
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") and "tau_bar" in line:
                for tok in line.replace("=", " ").split():
                    pass
                parts = line.split()
                try:
                    tau_bar = float(parts[parts.index("tau_bar") + 2])
                    geometry = parts[parts.index("geometry") + 2]
                except (ValueError, IndexError):
                    pass
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], tau_bar, geometry


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "hillier_polarization.svg"
    prolate_path = sys.argv[2] if len(sys.argv) > 2 else "hillier_observers_prolate.txt"
    spherical_path = (sys.argv[3] if len(sys.argv) > 3
                      else "hillier_observers_spherical.txt")

    # The dumps carry a --tag suffix once a sweep is involved, so fall back to
    # whatever is actually on disk rather than insisting on the bare name.
    prolate_path = resolve(prolate_path, "hillier_observers_prolate*.txt")
    spherical_path = resolve(spherical_path, "hillier_observers_spherical*.txt")
    if prolate_path is None:
        print("no hillier_observers_prolate*.txt found; run the example first")
        return 1
    inc_p, q_p, _u_p, energy_p, tau_bar, _geom = read_dump(prolate_path)
    have_sph = spherical_path is not None
    if have_sph:
        inc_s, q_s, _u_s, _e_s, _tb, _g = read_dump(spherical_path)

    # Percent throughout; the analytic curve sets the y range.
    q_p = 100.0 * q_p
    if have_sph:
        inc_s, q_s = inc_s, 100.0 * q_s
    curve_x = np.linspace(0.0, 90.0, 181)
    curve_y = -100.0 * (3.0 / 13.0) * tau_bar * np.sin(np.radians(curve_x)) ** 2

    ymin = min(curve_y.min(), q_p.min()) * 1.25
    ymax = max(0.0, q_p.max()) + 0.12 * abs(ymin)

    def sx(x):
        return L + x / 90.0 * PW

    def sy(y):
        return T + (ymax - y) / (ymax - ymin) * PH

    s = []
    add = s.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="Inter, Helvetica, Arial, sans-serif">')
    add("""<style>
  .bg{fill:#fcfcfb;} .ttl{font-size:16px;font-weight:600;fill:#0b0b0b;}
  .sub{font-size:12px;fill:#4a4a48;} .ax{font-size:11.5px;fill:#4a4a48;}
  .axl{font-size:12.5px;font-weight:500;fill:#4a4a48;}
  .leg{font-size:11.5px;fill:#4a4a48;} .note{font-size:11px;fill:#8a8a86;}
  .grid{stroke:#e6e6e2;stroke-width:1;} .axs{stroke:#c9c9c4;stroke-width:1;}
  .zero{stroke:#9a9a95;stroke-width:1.4;}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19;} .ttl{fill:#ffffff;} .sub,.ax,.axl,.leg{fill:#c9c9c4;}
    .note{fill:#8a8a86;} .grid{stroke:#2e2e2c;} .axs{stroke:#3d3d3a;}
    .zero{stroke:#6a6a66;}
  }
</style>""")
    add(f'<rect class="bg" width="{W}" height="{H}"/>')
    add(f'<text class="ttl" x="{L}" y="26">Polarization versus inclination, '
        f'all observer directions</text>')
    add(f'<text class="sub" x="{L}" y="45">STORM hillier_polarization benchmark; '
        f'line is the analytic single-scattering limit, not a fit.</text>')

    # legend
    add(f'<circle cx="{L+6}" cy="62" r="3.6" fill="{PROLATE_COLOUR}"/>')
    add(f'<text class="leg" x="{L+18}" y="66">prolate envelope</text>')
    if have_sph:
        add(f'<circle cx="{L+156}" cy="62" r="3.6" fill="{SPHERICAL_COLOUR}"/>')
        add(f'<text class="leg" x="{L+168}" y="66">spherical control</text>')
    add(f'<circle cx="{L+300}" cy="62" r="5" fill="{MEAN_COLOUR}" '
        f'stroke="#fcfcfb" stroke-width="2"/>')
    add(f'<text class="leg" x="{L+312}" y="66">ring mean ±1σ</text>')
    add(f'<line x1="{L+430}" y1="62" x2="{L+458}" y2="62" stroke="{CURVE_COLOUR}" '
        f'stroke-width="2"/>')
    add(f'<text class="leg" x="{L+466}" y="66">'
        f'q = −(3/13)·τ̄·sin²i</text>')

    # axes
    step = 1.0 if (ymax - ymin) < 8 else 2.0
    yv = np.ceil(ymin / step) * step
    while yv <= ymax + 1e-9:
        y = sy(yv)
        add(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}"/>')
        add(f'<text class="ax" x="{L-9}" y="{y+4:.1f}" text-anchor="end">'
            f'{yv:g}</text>')
        yv += step
    for xv in (0, 15, 30, 45, 60, 75, 90):
        x = sx(xv)
        add(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+PH}"/>')
        add(f'<text class="ax" x="{x:.1f}" y="{T+PH+19}" text-anchor="middle">'
            f'{xv}</text>')
    add(f'<line class="zero" x1="{L}" y1="{sy(0):.1f}" x2="{L+PW}" y2="{sy(0):.1f}"/>')
    add(f'<line class="axs" x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}"/>')
    add(f'<text class="axl" x="{L+PW/2:.0f}" y="{T+PH+44}" text-anchor="middle">'
        f'inclination i (degrees)</text>')
    add(f'<text class="axl" transform="translate(24,{T+PH/2:.0f}) rotate(-90)" '
        f'text-anchor="middle">polarization q (%)</text>')

    # analytic curve first, so the points read on top of it
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(curve_x, curve_y)
                   if ymin <= y <= ymax)
    add(f'<polyline points="{pts}" fill="none" stroke="{CURVE_COLOUR}" '
        f'stroke-width="2" opacity="0.85"/>')

    if have_sph:
        for x, y in zip(inc_s, q_s):
            if ymin <= y <= ymax:
                add(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" '
                    f'fill="{SPHERICAL_COLOUR}" opacity="0.55"/>')
    for x, y in zip(inc_p, q_p):
        if ymin <= y <= ymax:
            add(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" '
                f'fill="{PROLATE_COLOUR}" opacity="0.72"/>')

    # Ring means on top, with capped +-1 sigma bars: the gated quantity.
    for centre, qb, sig, _n in ring_means(inc_p, q_p / 100.0, energy_p):
        qb, sig = 100.0 * qb, 100.0 * sig
        if not (ymin <= qb <= ymax):
            continue
        x, top, bottom = sx(centre), sy(qb + sig), sy(qb - sig)
        add(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{MEAN_COLOUR}" stroke-width="1.8"/>')
        for cap in (top, bottom):
            add(f'<line x1="{x-4:.1f}" y1="{cap:.1f}" x2="{x+4:.1f}" '
                f'y2="{cap:.1f}" stroke="{MEAN_COLOUR}" stroke-width="1.8"/>')
        add(f'<circle cx="{x:.1f}" cy="{sy(qb):.1f}" r="5" '
            f'fill="{MEAN_COLOUR}" stroke="#fcfcfb" stroke-width="2"/>')

    add(f'<text class="note" x="{L}" y="{H-30}">'
        f'τ̄ = {tau_bar:.4f}; {len(inc_p)} observer directions. Scatter '
        f'about the curve is Monte Carlo noise on a single direction.</text>')
    add(f'<text class="note" x="{L}" y="{H-14}">The test averages these into '
        f'inclination bands before asserting; it does not assert per direction.</text>')
    add("</svg>")

    with open(out, "w") as fh:
        fh.write("\n".join(s))
    print(f"wrote {out} from {os.path.basename(prolate_path)}"
          + (f" + {os.path.basename(spherical_path)}" if have_sph else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
