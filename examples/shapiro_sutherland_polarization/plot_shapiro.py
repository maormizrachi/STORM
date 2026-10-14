#!/usr/bin/env python3
"""Figures comparing the STORM plane-parallel runs with Shapiro & Sutherland
(1982, ApJ 263, 902): Table 1 and Figures 2 and 3.

  table1.png      I/F and Q/I versus mu: the paper's Table 1 (circles), the
                  deterministic solution (line) and the STORM bins (bars, with
                  errors from the scatter between generations).
  fig2.png        Pi_net versus asphericity, oblate and prolate, equator view.
  fig3.png        log10(L/L_ss) versus asphericity, equator and pole views.
  comparison.txt  the numbers behind them.

In Figures 2 and 3 the lines integrate the paper's Table 1 over the spheroid as
the paper does (spline in mu), so they reproduce the published curves; dotted
lines integrate the deterministic solution; points integrate the STORM bin
tallies; open squares integrate the deterministic solution through the same
bins, the like-for-like reference for the points (the bin quadrature shifts the
most extreme asphericities by ~1%).

Usage: plot_shapiro.py RUN_DIR [--prefix ss_table1] [--out DIR]
Needs numpy, scipy and matplotlib (not a regression gate).
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planeparallel_reference as ref  # noqa: E402
import spheroid  # noqa: E402

LABELS = {"scattering": r"$\lambda=0$", "lambda16": r"$\lambda=1/6$", "lambdatau": r"$\lambda=\tau/(1+\tau)$"}
STYLES = {"scattering": "-", "lambda16": "--", "lambdatau": "-."}
COLORS = {"scattering": "#1b6ca8", "lambda16": "#c0392b", "lambdatau": "#2e8b57"}
GRID = dict(n_theta=600, n_phi=600)
SUBTITLE = ("lines from Table 1 (as published), dotted from the deterministic solution, points STORM,\n"
            "open squares the deterministic solution through the STORM $\\mu$ bins")


def load_run(paths):
    """Bin edges and per-generation tallies (generations x bins) of one case,
    pooled over the files of independent seeds, which must share their bins."""
    edges, E, Q = None, [], []
    for path in paths:
        d = np.atleast_2d(np.loadtxt(path))
        nb = int(d[:, 1].max()) + 1
        if len(d) % nb or not np.array_equal(d[:, 1], np.tile(np.arange(nb), len(d) // nb)):
            raise ValueError("%s: rows are not whole generations of %d bins" % (path, nb))
        e = np.append(d[:nb, 2], d[nb - 1, 3])
        if edges is None:
            edges = e
        elif not np.array_equal(e, edges):
            raise ValueError("%s: mu bins differ from %s" % (path, paths[0]))
        E.append(d[:, 4].reshape(-1, nb))
        Q.append(d[:, 5].reshape(-1, nb))
    E, Q = np.vstack(E), np.vstack(Q)
    if len(E) < 2:
        raise ValueError("%s: need at least 2 generations for the errors" % paths[0])
    return edges, E, Q


def ratio(num, den):
    """num / den, nan where den is not positive (no light or no scatter)."""
    num, den = np.broadcast_arrays(np.asarray(num, float), np.asarray(den, float))
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def mc_bins(edges, E, Q):
    """Bin I/F and Q/I (Shapiro & Sutherland's sign) with generation-scatter errors."""
    n = E.shape[0]
    Es, tot = E.sum(0), E.sum()
    i_f = Es / (0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)) / (2.0 * tot)
    p = ratio(-Q.sum(0), Es)   # nan in a bin without light
    corr = n / (n - 1.0)       # n >= 2 (load_run)
    di = ratio(E, Es) - E.sum(1)[:, None] / tot
    dq = ratio(-Q - p * E, Es)
    return i_f, i_f * np.sqrt(corr * (di ** 2).sum(0)), p, np.sqrt(corr * (dq ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", default="ss_table1")
    ap.add_argument("--out", default="")
    ap.add_argument("--xi-points", type=int, default=12)
    args = ap.parse_args()
    out = args.out or args.run_dir
    os.makedirs(out, exist_ok=True)

    mu_t = ref.TABLE1[:, 0]
    mus = np.linspace(0.0, 1.0, 101)
    paper, det, det_bins, runs = {}, {}, {}, {}
    for case in ref.CASES:
        col = ref.TABLE1_COLUMN[case]
        paper[case] = (CubicSpline(mu_t, ref.TABLE1[:, col]), CubicSpline(mu_t, ref.TABLE1[:, col + 1]))
        em = ref.deterministic(case)
        det[case] = tuple(CubicSpline(mus, v) for v in ref.at_mu(em, mus))
        det_bins[case] = ref.binned(em)
        paths = sorted(glob.glob(os.path.join(args.run_dir, "%s_%s.txt" % (args.prefix, case))) +
                       glob.glob(os.path.join(args.run_dir, "%s_%s_s*.txt" % (args.prefix, case))))
        if paths:
            runs[case] = load_run(paths)
            if len(runs[case][0]) - 1 != len(det_bins[case]):
                raise ValueError("%s: STORM has %d mu bins, the reference %d" % (
                    case, len(runs[case][0]) - 1, len(det_bins[case])))

    def save(fig, name, title):
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=150)
        plt.close(fig)

    # ---------------- Table 1 ----------------
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    summary = ["# Table 1 bins (STORM vs deterministic):"]
    for j, case in enumerate(ref.CASES):
        c, col = COLORS[case], ref.TABLE1_COLUMN[case]
        ax[0, j].plot(mus, det[case][0](mus), color=c, lw=1.2, label="deterministic")
        ax[1, j].plot(mus, 100 * det[case][1](mus), color=c, lw=1.2)
        ax[0, j].plot(mu_t, ref.TABLE1[:, col], "ko", ms=4, mfc="none", label="S&S Table 1")
        ax[1, j].plot(mu_t, 100 * ref.TABLE1[:, col + 1], "ko", ms=4, mfc="none")
        if case in runs:
            edges, E, Q = runs[case]
            i_f, si, p, sp = mc_bins(edges, E, Q)
            mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
            ax[0, j].errorbar(mid, i_f, xerr=half, yerr=si, fmt="none", ecolor="k", elinewidth=1.4,
                              label="STORM (%d gen.)" % E.shape[0])
            ax[1, j].errorbar(mid, 100 * p, xerr=half, yerr=100 * sp, fmt="none", ecolor="k", elinewidth=1.4)
            tab = det_bins[case]
            summary.append("%-10s  max|dI/F|/ref %.4f  max|dQ/I| %.5f  chi2/dof I %.2f  Q %.2f  limb Q/I "
                           "%.4f+-%.4f (ref bin %.4f)" % (
                               case, np.max(np.abs(i_f / tab[:, 0] - 1)), np.max(np.abs(p - tab[:, 1])),
                               np.mean(ratio(i_f - tab[:, 0], si) ** 2), np.mean(ratio(p - tab[:, 1], sp) ** 2),
                               p[0], sp[0], tab[0, 1]))
        ax[0, j].set_title(LABELS[case])
        ax[1, j].set_xlabel(r"$\mu$")
        ax[0, j].grid(alpha=0.3)
        ax[1, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel(r"$I(0,\mu)/F$")
    ax[1, 0].set_ylabel(r"$Q/I$ (%)")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    save(fig, "table1.png", "Plane-parallel atmosphere: Shapiro & Sutherland (1982) Table 1 vs STORM")

    # ---------------- Figures 2 and 3 ----------------
    # Columns of the spheroid values: Pi_net, L/L_ss (quantity 0, 1); for the
    # points also their errors (2, 3) and the binned deterministic values (4, 5).
    xis_line = np.concatenate([np.linspace(0, 0.9, 19), np.linspace(0.92, 0.995, 8)])
    xis_mc = np.linspace(0.05, 0.95, args.xi_points)
    curves = {(shape, view, case): [np.zeros((len(xis_line), 2)) for _ in range(2)]
              for shape in ("oblate", "prolate") for view in ("equator", "pole") for case in ref.CASES}
    points = {(shape, view, case): np.zeros((len(xis_mc), 6))
              for shape in ("oblate", "prolate") for view in ("equator", "pole") for case in runs}
    # Deterministic bin tallies in the form of the STORM ones (unit total energy).
    det_tallies = {}
    for case in runs:
        e = det_bins[case][:, 0] * (runs[case][0][1:] ** 2 - runs[case][0][:-1] ** 2)
        det_tallies[case] = (e, -det_bins[case][:, 1] * e)
    for shape in ("oblate", "prolate"):
        for i, x in enumerate(xis_line):   # one surface per (xi, shape), shared by views and laws
            surf = spheroid.surface(x, shape, **GRID)
            for view in ("equator", "pole"):
                for case in ref.CASES:
                    for j, law in enumerate((paper[case], det[case])):
                        curves[shape, view, case][j][i] = spheroid.continuous(*law, surf, view)
        for i, x in enumerate(xis_mc):
            surf = spheroid.surface(x, shape, **GRID)
            for view in ("equator", "pole"):
                for case, (edges, E, Q) in runs.items():
                    A, C, area = spheroid.bin_weights(surf, view, edges)
                    points[shape, view, case][i] = (spheroid.binned(E, Q, A, C, area) +
                                                    spheroid.binned_errors(E, Q, A, C, area) +
                                                    spheroid.binned(*det_tallies[case], A, C, area))

    def panel(a, shape, view, quantity, title):
        scale = (lambda v: 100.0 * v) if quantity == 0 else np.log10
        for case in ref.CASES:
            c = COLORS[case]
            paper_v, det_v = curves[shape, view, case]
            a.plot(xis_line, scale(paper_v[:, quantity]), color=c, ls=STYLES[case], lw=1.0,
                   label=LABELS[case] + " (Table 1)")
            a.plot(xis_line, scale(det_v[:, quantity]), color=c, ls=":", lw=1.5)
            if case in runs:
                p = points[shape, view, case]
                v, err = p[:, quantity], p[:, 2 + quantity]
                a.errorbar(xis_mc, scale(v), yerr=100 * err if quantity == 0 else err / (v * np.log(10)),
                           fmt="o", ms=3.5, color=c, label=LABELS[case] + " STORM")
                a.plot(xis_mc, scale(p[:, 4 + quantity]), ls="none", marker="s", ms=7, mfc="none", mec=c,
                       mew=0.8, label="deterministic, same $\\mu$ bins as STORM" if case == "scattering" else None)
        a.set_xlabel(r"asphericity $\xi$")
        a.set_ylabel(r"$\Pi_{\rm net}$ (%)" if quantity == 0 else r"$\log_{10}(L/L_{\rm ss})$")
        a.set_title(title)
        a.set_xlim(0, 1)
        a.grid(alpha=0.3)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for k, shape in enumerate(("oblate", "prolate")):
        panel(ax[0, k], shape, "equator", 0, "%s spheroid, equator view" % shape.capitalize())
        panel(ax[1, k], shape, "equator", 0, "%s spheroid, equator view (enlarged)" % shape.capitalize())
        ax[1, k].set_xlim(0, 0.75 if shape == "oblate" else 1.0)
        ax[1, k].set_ylim(0, 2.2 if shape == "oblate" else 1.0)
    ax[0, 0].legend(fontsize=7)
    save(fig, "fig2.png", "Shapiro & Sutherland Fig. 2: " + SUBTITLE)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for i, view in enumerate(("equator", "pole")):
        for k, shape in enumerate(("oblate", "prolate")):
            panel(ax[i, k], shape, view, 1, "%s spheroid, %s view" % (shape.capitalize(), view))
    ax[0, 0].legend(fontsize=7)
    save(fig, "fig3.png", "Shapiro & Sutherland Fig. 3: " + SUBTITLE)

    # ---------------- Numbers ----------------
    def compare(p, idx, quantity):
        value, err, binned = p[idx, quantity], p[idx, 2 + quantity], p[idx, 4 + quantity]
        s = 100.0 if quantity == 0 else 1.0
        return "binned %.6g STORM %.6g+-%.2g (%+.1f sigma)" % (s * binned, s * value, s * err,
                                                               ratio(value - binned, err))

    lines = summary + ["# Fig. 2/3 at three asphericities: Pi_net (%) from Table 1 as published, then the",
                       "# deterministic solution through the STORM bins (binned) against STORM; L/L_ss likewise"]
    for shape in ("oblate", "prolate"):
        for case in runs:
            pe, pp = points[shape, "equator", case], points[shape, "pole", case]
            for idx in (0, len(xis_mc) // 2, len(xis_mc) - 1):
                surf = spheroid.surface(xis_mc[idx], shape, **GRID)
                table1_pi = spheroid.continuous(*paper[case], surf, "equator")[0]
                lines.append("%-7s %-10s xi=%.3f  Pi_net: table1 %.4f %s  |  L/Lss equator: %s  |  pole: %s" % (
                    shape, case, xis_mc[idx], 100 * table1_pi, compare(pe, idx, 0), compare(pe, idx, 1),
                    compare(pp, idx, 1)))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "comparison.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
