#!/usr/bin/env python3
"""Hillier (1994) Fig. 5: published curves with STORM's observers over them.

Collects the per-observer dumps written by a chi0 sweep
(hillier_observers_prolate_<tag>.txt), averages the observers into inclination
bands exactly as main.cpp does, and plots the band polarization against
log10 chi0 on top of Hillier's published curves.

Points are drawn without connecting lines: the comparison is against the
reference curves, not against a STORM interpolation.

Open markers mark a band whose <sin^2 i> departs from sin^2 of its nominal
inclination by more than BIAS_LIMIT -- which happens with Fibonacci directions,
where nothing lands on the paper's inclinations and a +-7 deg window has to stand
in for one.  With ring detectors placed on the inclinations themselves there is no
such departure and no open markers appear.

That distinction mattered: read with Fibonacci directions and a +-7 deg band,
i = 22.5 deg came out at about half Hillier's value, consistently across chi0.
With rings on 22.5 deg it agrees.  The deficit was the estimator, not the
transport.

The Hillier curves are digitised and are a visual reference only; the regression
gate is the analytic thin limit asserted in main.cpp, which needs no external
data.  See hillier_reference.py.

SVG is emitted directly rather than through matplotlib: STORM regression runs on
machines where matplotlib is not installed, and a plot that only works on some of
them is worse than one with no dependencies.  numpy is the only import.

Usage:
    plot_hillier_fig5.py [out.svg] [dump_dir]
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hillier_reference import HILLIER_DIGITIZED, RAMP, RAMP_DARK  # noqa: E402

# Must track main.cpp: the band geometry is what makes the points comparable.
BAND_CENTRES = [22.5, 45.0, 67.5, 90.0]
BAND_HALF_WIDTH_DEG = 7.0
GAMMA = 7.0 / 13.0
THIN_COEFF = 3.0 / 8.0 * (3.0 * GAMMA - 1.0)      # = 3/13
BIAS_LIMIT = 0.05        # bands whose <sin^2 i> is off by more than this: open
OBSERVER_NOTE = ""       # filled from the dump header

W, H = 1000, 720
L, R, T, B = 78, 150, 100, 160
PW, PH = W - L - R, H - T - B
XMIN, XMAX = -2.10, 0.06
YMIN, YMAX = -4.6, 2.4


def sx(x):
    return L + (x - XMIN) / (XMAX - XMIN) * PW


def sy(y):
    return T + (YMAX - y) / (YMAX - YMIN) * PH


# Only the thin end is a fair comparison for a single-scattering benchmark run at
# this packet count; past the turnover the envelope is optically thick.
THIN_MAX_LOG_CHI = -0.67


# Notes run the full width of the canvas, not just the plot area.  At 11.5px in a
# system sans the mean advance is a shade under 5.7px, so this is the character
# budget per line; wrapping on it keeps long captions inside the viewBox instead
# of letting them run off the right edge, which SVG will not tell you about.
NOTE_CHARS = int((W - L - 20) / 5.7)


def wrap(text, width=None):
    """Greedy word wrap; returns a list of lines."""
    width = width or NOTE_CHARS
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def summarise(series, centres):
    """Mean, sd and worst |ratio-1| of STORM/Hillier over the thin end."""
    ratios = []
    for centre in centres:
        ref = np.array(HILLIER_DIGITIZED[f"{centre}"])
        for x, v, _sig, _biased in series.get(centre, []):
            if x > THIN_MAX_LOG_CHI or not (ref[:, 0].min() <= x <= ref[:, 0].max()):
                continue
            href = np.interp(x, ref[:, 0], ref[:, 1])
            if abs(href) > 1e-9:
                ratios.append(v / href)
    if not ratios:
        return None
    ratios = np.array(ratios)
    return ratios.mean(), ratios.std(), np.abs(ratios - 1.0).max()


def read_dump(path):
    """Return (chi0, inclination_deg, q, u, energy) from one per-observer dump."""
    chi0 = float("nan")
    global OBSERVER_NOTE
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"chi0\s*=\s*([0-9.eE+-]+)", line)
            if m:
                chi0 = float(m.group(1))
            if "observer =" in line:
                OBSERVER_NOTE = line.lstrip("# ").rstrip()
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return chi0, data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def band(inc, q, energy, centre):
    """Pooled Stokes ratio over one inclination band.

    Mirrors BandAverage in main.cpp: sum(Q)/sum(I) weighted by energy, not a mean
    of per-observer ratios, which would be biased.  Returns (q, sigma, <sin^2 i>,
    n) or None when the band is empty.
    """
    sel = np.abs(inc - centre) <= BAND_HALF_WIDTH_DEG
    if not sel.any():
        return None
    w = energy[sel]
    big_q = q[sel] * w                      # dump stores q = Q/I per observer
    w_sum = w.sum()
    if w_sum <= 0.0:
        return None
    q_bar = big_q.sum() / w_sum
    n = int(sel.sum())
    var = max(0.0, (big_q ** 2).sum() - big_q.sum() ** 2 / n)
    sigma = np.sqrt(var) / w_sum
    sin2 = (w * np.sin(np.radians(inc[sel])) ** 2).sum() / w_sum
    return q_bar, sigma, sin2, n


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "hillier_fig5.svg"
    directory = sys.argv[2] if len(sys.argv) > 2 else "."

    paths = sorted(glob.glob(os.path.join(directory,
                                          "hillier_observers_prolate*.txt")))
    if not paths:
        print(f"no hillier_observers_prolate*.txt in {directory}; run the sweep first")
        return 1

    # centre -> list of (log10 chi0, p percent, sigma percent, biased)
    series = {c: [] for c in BAND_CENTRES}
    for path in paths:
        chi0, inc, q, _u, energy = read_dump(path)
        if not np.isfinite(chi0) or chi0 <= 0.0:
            continue
        for centre in BAND_CENTRES:
            got = band(inc, q, energy, centre)
            if got is None:
                continue
            q_bar, sigma, sin2, _n = got
            biased = abs(sin2 / np.sin(np.radians(centre)) ** 2 - 1.0) > BIAS_LIMIT
            series[centre].append((np.log10(chi0), 100.0 * q_bar,
                                   100.0 * sigma, biased))
    for c in BAND_CENTRES:
        series[c].sort()

    s = []
    add = s.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="Inter, Helvetica, Arial, sans-serif">')
    add("""<style>
  .bg{fill:#fcfcfb;} .ttl{font-size:17px;font-weight:600;fill:#0b0b0b;}
  .sub{font-size:12.5px;fill:#4a4a48;} .ax{font-size:12px;fill:#4a4a48;}
  .axl{font-size:12.5px;font-weight:500;fill:#4a4a48;}
  .lab{font-size:12px;font-weight:600;} .note{font-size:11.5px;fill:#8a8a86;}
  .leg{font-size:11.5px;fill:#4a4a48;}
  .grid{stroke:#e6e6e2;stroke-width:1;} .axs{stroke:#c9c9c4;stroke-width:1;}
  .zero{stroke:#9a9a95;stroke-width:1.5;}
  .hl{stroke-width:2;fill:none;} .mk{stroke:#fcfcfb;stroke-width:1.6;}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19;} .ttl{fill:#ffffff;} .sub,.ax,.axl,.leg{fill:#c9c9c4;}
    .note{fill:#8a8a86;} .grid{stroke:#2e2e2c;} .axs{stroke:#3d3d3a;}
    .zero{stroke:#6a6a66;} .mk{stroke:#1a1a19;}
  }
</style>""")
    series_css = ["<style>"]
    for i, colour in enumerate(RAMP):
        series_css.append(f"  .s{i}{{stroke:{colour};}} .f{i}{{fill:{colour};}}")
    series_css.append("  @media (prefers-color-scheme: dark){")
    for i, colour in enumerate(RAMP_DARK):
        series_css.append(f"    .s{i}{{stroke:{colour};}} .f{i}{{fill:{colour};}}")
    series_css.append("  }")
    series_css.append("</style>")
    add("\n".join(series_css))
    add(f'<rect class="bg" width="{W}" height="{H}"/>')
    add(f'<text class="ttl" x="{L}" y="28">STORM against Hillier (1994), '
        f'polarization versus scattering depth</text>')
    add(f'<text class="sub" x="{L}" y="48">Lines are Hillier as published; '
        f'points are STORM observers band-averaged. Nothing is fitted or '
        f'rescaled.</text>')

    lx, ly = L, 74
    add(f'<line class="hl" x1="{lx}" y1="{ly-4}" x2="{lx+26}" y2="{ly-4}" '
        f'stroke="#4a4a48"/>')
    add(f'<text class="leg" x="{lx+34}" y="{ly}">Hillier (1994)</text>')
    add(f'<circle cx="{lx+150}" cy="{ly-4}" r="4" fill="#4a4a48" '
        f'stroke="#fcfcfb" stroke-width="1.6"/>')
    add(f'<text class="leg" x="{lx+162}" y="{ly}">STORM (±1σ)</text>')
    if any(biased for pts in series.values() for *_r, biased in pts):
        add(f'<circle cx="{lx+238}" cy="{ly-4}" r="4" fill="#fcfcfb" '
            f'stroke="#4a4a48" stroke-width="1.6"/>')
        add(f'<text class="leg" x="{lx+250}" y="{ly}">open: band ⟨sin²i⟩ off its '
            f'nominal inclination</text>')

    for yv in range(-4, 3):
        y = sy(yv)
        add(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}"/>')
        add(f'<text class="ax" x="{L-10}" y="{y+4:.1f}" text-anchor="end">{yv}</text>')
    for xv in (-2.0, -1.5, -1.0, -0.5, 0.0):
        x = sx(xv)
        add(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+PH}"/>')
        add(f'<text class="ax" x="{x:.1f}" y="{T+PH+20}" text-anchor="middle">'
            f'{xv:.1f}</text>')
    add(f'<line class="zero" x1="{L}" y1="{sy(0):.1f}" x2="{L+PW}" y2="{sy(0):.1f}"/>')
    add(f'<line class="axs" x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}"/>')
    add(f'<text class="axl" x="{L+PW/2:.0f}" y="{T+PH+44}" text-anchor="middle">'
        f'log₁₀ χ₀</text>')
    add(f'<text class="axl" transform="translate(22,{T+PH/2:.0f}) rotate(-90)" '
        f'text-anchor="middle">polarization p (%)</text>')

    # Reference curves first, so the points read on top of them.
    for idx, centre in enumerate(BAND_CENTRES):
        pts = [(x, v) for x, v in HILLIER_DIGITIZED[f"{centre}"]
               if XMIN <= x <= XMAX and YMIN <= v <= YMAX]
        add(f'<polyline class="hl s{idx}" points="'
            + " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in pts) + '"/>')

    anchors, meta = [], []
    for idx, centre in enumerate(BAND_CENTRES):
        for x, v, sig, biased in series[centre]:
            if not (YMIN <= v <= YMAX):
                continue
            # Capped +-1 sigma bar.  Caps matter here: with a 2 deg acceptance
            # cone the bars are short, and a bare whisker at that length is hard
            # to tell from a tick or from the marker itself.
            top, bottom = sy(v + sig), sy(v - sig)
            add(f'<line class="s{idx}" x1="{sx(x):.1f}" y1="{top:.1f}" '
                f'x2="{sx(x):.1f}" y2="{bottom:.1f}" stroke-width="1.4" '
                f'opacity="0.9"/>')
            for cap in (top, bottom):
                add(f'<line class="s{idx}" x1="{sx(x)-3.2:.1f}" y1="{cap:.1f}" '
                    f'x2="{sx(x)+3.2:.1f}" y2="{cap:.1f}" stroke-width="1.4" '
                    f'opacity="0.9"/>')
            if biased:
                add(f'<circle class="s{idx}" cx="{sx(x):.1f}" cy="{sy(v):.1f}" '
                    f'r="3.8" fill="#fcfcfb" stroke-width="1.8"/>')
            else:
                add(f'<circle class="mk f{idx}" cx="{sx(x):.1f}" '
                    f'cy="{sy(v):.1f}" r="4"/>')
        ref = HILLIER_DIGITIZED[f"{centre}"]
        anchors.append(sy(ref[-1][1]))
        meta.append((sx(ref[-1][0]), idx, centre))

    # Nudge collided end labels apart, keeping their order.
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    placed = list(anchors)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if placed[b] - placed[a] < 16.0:
            placed[b] = placed[a] + 16.0
    for y, (x, idx, centre) in zip(placed, meta):
        add(f'<line class="s{idx}" x1="{x+6:.1f}" y1="{y:.1f}" x2="{x+16:.1f}" '
            f'y2="{y:.1f}" stroke-width="1.2" opacity="0.6"/>')
        add(f'<text class="lab f{idx}" x="{x+21:.1f}" y="{y+4:.1f}">'
            f'i = {centre}°</text>')

    npts = sum(len(v) for v in series.values())

    notes = [f"Envelope \u03c1 \u221d \u03c7\u2080 (R_min/r)\u2074 "
             f"(1 + 10cos\u00b2\u03b2), R_max = 30 R_min, "
             f"\u03c4\u0304 = 2.8887 \u03c7\u2080 \u2014 Hillier\u2019s geometry, "
             f"no adjustable parameter. {len(paths)} \u03c7\u2080 values, "
             f"{npts} band points."
             + (f" Detectors: {OBSERVER_NOTE.replace('observer = ', '')}."
                if OBSERVER_NOTE else "")]

    per_band = []
    for centre in BAND_CENTRES:
        one = summarise(series, [centre])
        if one is not None:
            per_band.append(f"{centre:g}\u00b0 {one[0]:.2f}")
    if per_band:
        notes.append(f"STORM / Hillier for log\u2081\u2080 \u03c7\u2080 "
                     f"\u2264 {THIN_MAX_LOG_CHI:g}, by inclination: "
                     + ", ".join(per_band) + ".")
    gate = summarise(series, [45.0, 67.5, 90.0])
    if gate is not None:
        notes.append(f"Over 45\u00b0, 67.5\u00b0 and 90\u00b0 together: "
                     f"{gate[0]:.3f} \u00b1 {gate[1]:.3f}, worst {gate[2]:.0%}. "
                     f"Past the turnover the envelope is optically thick and this "
                     f"run\u2019s statistics do not resolve it.")
    notes.append("Hillier curves are digitised (\u00b10.05 in p) and are a visual "
                 "reference; the regression gate is the analytic thin limit, which "
                 "uses no external data.")

    wrapped = [line for note in notes for line in wrap(note)]
    y = H - 16 - 15 * (len(wrapped) - 1)
    for line in wrapped:
        add(f'<text class="note" x="{L}" y="{y}">{line}</text>')
        y += 15
    add("</svg>")

    with open(out_path, "w") as fh:
        fh.write("\n".join(s))
    print(f"wrote {out_path}  ({len(paths)} chi0 values, {npts} band points)")

    # Numbers alongside the figure: a ratio is easier to check than a plot.
    print("\n  log10 chi0    i      p_STORM %   sigma %   p_Hillier %   ratio")
    for centre in BAND_CENTRES:
        ref = np.array(HILLIER_DIGITIZED[f"{centre}"])
        for x, v, sig, biased in series[centre]:
            if x < ref[:, 0].min() or x > ref[:, 0].max():
                continue
            href = np.interp(x, ref[:, 0], ref[:, 1])
            ratio = v / href if abs(href) > 1e-9 else float("nan")
            flag = "  (band biased)" if biased else ""
            print(f"  {x:9.3f}  {centre:5.1f}   {v:9.4f}  {sig:8.4f}   "
                  f"{href:9.4f}   {ratio:6.3f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
