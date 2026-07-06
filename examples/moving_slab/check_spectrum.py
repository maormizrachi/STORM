#!/usr/bin/env python3
"""
Compare STORM moving-slab MC spectrum against the semi-analytic solution.

Reads moving_slab_mc_spectrum.txt produced by the C++ example, computes
the McClarren & Gentile (2021) semi-analytic reference for the
"original vacuum" benchmark with the 124-group aluminum opacity table,
and reports error metrics + comparison plots.

Usage:
    python3 check_spectrum.py [--spectrum moving_slab_mc_spectrum.txt] [--plot-dir .]
"""
import argparse
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CGS, keV, ns)
# ---------------------------------------------------------------------------
K_BOLTZMANN_GJ_PER_KEV = 1.6021917e-25
PLANCK_GJ_NS = 6.6262e-34
C_LIGHT = 29.9792458  # cm/ns
ERG_PER_GJ = 1e16

# ---------------------------------------------------------------------------
# 124-group aluminum opacity table (Tables 3-7 of the 2026 paper)
# ---------------------------------------------------------------------------
_OPACITY_TABLE = [
    (1.000e-03, 1.229e-03, 1.000e+04), (1.229e-03, 1.510e-03, 1.000e+04),
    (1.510e-03, 1.856e-03, 1.000e+04), (1.856e-03, 2.281e-03, 1.000e+04),
    (2.281e-03, 2.804e-03, 1.000e+04), (2.804e-03, 3.446e-03, 1.000e+04),
    (3.446e-03, 4.234e-03, 1.000e+04), (4.234e-03, 5.204e-03, 1.000e+04),
    (5.204e-03, 6.396e-03, 1.000e+04), (6.396e-03, 7.860e-03, 1.000e+04),
    (7.860e-03, 9.660e-03, 1.000e+04), (9.660e-03, 1.187e-02, 1.000e+04),
    (1.187e-02, 1.459e-02, 1.000e+04), (1.459e-02, 1.793e-02, 1.000e+04),
    (1.793e-02, 2.204e-02, 1.000e+04), (2.204e-02, 2.708e-02, 8.933e+03),
    (2.708e-02, 3.328e-02, 8.569e+03), (3.328e-02, 4.090e-02, 7.335e+03),
    (4.090e-02, 5.027e-02, 5.656e+03), (5.027e-02, 6.178e-02, 4.031e+03),
    (6.178e-02, 7.593e-02, 2.710e+03), (7.593e-02, 9.331e-02, 1.770e+03),
    (9.331e-02, 1.147e-01, 1.184e+03), (1.147e-01, 1.409e-01, 7.924e+02),
    (1.409e-01, 1.732e-01, 5.061e+02), (1.732e-01, 2.129e-01, 3.230e+02),
    (2.129e-01, 2.616e-01, 2.062e+02), (2.616e-01, 3.215e-01, 2.100e+02),
    (3.215e-01, 3.951e-01, 1.229e+02), (3.951e-01, 4.856e-01, 7.579e+01),
    (4.856e-01, 5.968e-01, 4.905e+01), (5.968e-01, 7.334e-01, 3.110e+01),
    (7.334e-01, 9.014e-01, 1.947e+01), (9.014e-01, 1.000e+00, 1.196e+01),
    (1.000e+00, 1.014e+00, 1.187e+01), (1.014e+00, 1.028e+00, 1.149e+01),
    (1.028e+00, 1.042e+00, 1.112e+01), (1.042e+00, 1.057e+00, 1.076e+01),
    (1.057e+00, 1.072e+00, 1.041e+01), (1.072e+00, 1.087e+00, 1.007e+01),
    (1.087e+00, 1.102e+00, 9.740e+00), (1.102e+00, 1.117e+00, 9.416e+00),
    (1.117e+00, 1.133e+00, 9.098e+00), (1.133e+00, 1.149e+00, 8.785e+00),
    (1.149e+00, 1.165e+00, 8.477e+00), (1.165e+00, 1.181e+00, 8.180e+00),
    (1.181e+00, 1.198e+00, 7.900e+00), (1.198e+00, 1.214e+00, 7.635e+00),
    (1.214e+00, 1.231e+00, 7.381e+00), (1.231e+00, 1.248e+00, 7.138e+00),
    (1.248e+00, 1.266e+00, 6.902e+00), (1.266e+00, 1.283e+00, 6.674e+00),
    (1.283e+00, 1.301e+00, 6.452e+00), (1.301e+00, 1.319e+00, 6.237e+00),
    (1.319e+00, 1.338e+00, 6.029e+00), (1.338e+00, 1.357e+00, 5.827e+00),
    (1.357e+00, 1.375e+00, 5.631e+00), (1.375e+00, 1.395e+00, 5.438e+00),
    (1.395e+00, 1.414e+00, 5.250e+00), (1.414e+00, 1.434e+00, 5.066e+00),
    (1.434e+00, 1.454e+00, 4.886e+00), (1.454e+00, 1.474e+00, 4.709e+00),
    (1.474e+00, 1.495e+00, 4.542e+00), (1.495e+00, 1.516e+00, 4.387e+00),
    (1.516e+00, 1.537e+00, 4.243e+00), (1.537e+00, 1.558e+00, 4.117e+00),
    (1.558e+00, 1.580e+00, 4.310e+00), (1.580e+00, 1.602e+00, 1.572e+01),
    (1.602e+00, 1.625e+00, 4.834e+00), (1.625e+00, 1.647e+00, 3.726e+00),
    (1.647e+00, 1.670e+00, 3.758e+00), (1.670e+00, 1.694e+00, 4.706e+00),
    (1.694e+00, 1.717e+00, 3.394e+01), (1.717e+00, 1.741e+00, 9.034e+02),
    (1.741e+00, 1.765e+00, 1.615e+01), (1.765e+00, 1.790e+00, 4.098e+00),
    (1.790e+00, 1.815e+00, 3.420e+00), (1.815e+00, 1.840e+00, 3.389e+00),
    (1.840e+00, 1.866e+00, 3.986e+00), (1.866e+00, 1.892e+00, 4.350e+00),
    (1.892e+00, 1.919e+00, 3.933e+00), (1.919e+00, 1.945e+00, 4.258e+00),
    (1.945e+00, 1.972e+00, 4.861e+00), (1.972e+00, 1.995e+00, 6.836e+00),
    (1.995e+00, 2.089e+00, 4.674e+01), (2.089e+00, 2.188e+00, 2.108e+01),
    (2.188e+00, 2.291e+00, 2.281e+01), (2.291e+00, 2.399e+00, 1.963e+01),
    (2.399e+00, 2.512e+00, 1.749e+01), (2.512e+00, 2.630e+00, 1.590e+01),
    (2.630e+00, 2.754e+00, 1.442e+01), (2.754e+00, 2.884e+00, 1.294e+01),
    (2.884e+00, 3.020e+00, 1.144e+01), (3.020e+00, 3.162e+00, 1.014e+01),
    (3.162e+00, 3.311e+00, 9.047e+00), (3.311e+00, 3.467e+00, 8.057e+00),
    (3.467e+00, 3.631e+00, 7.118e+00), (3.631e+00, 3.802e+00, 6.219e+00),
    (3.802e+00, 3.981e+00, 5.474e+00), (3.981e+00, 4.169e+00, 4.861e+00),
    (4.169e+00, 4.365e+00, 4.311e+00), (4.365e+00, 4.571e+00, 3.792e+00),
    (4.571e+00, 4.786e+00, 3.296e+00), (4.786e+00, 5.012e+00, 2.888e+00),
    (5.012e+00, 5.248e+00, 2.555e+00), (5.248e+00, 5.495e+00, 2.258e+00),
    (5.495e+00, 5.754e+00, 1.978e+00), (5.754e+00, 6.026e+00, 1.713e+00),
    (6.026e+00, 6.310e+00, 1.496e+00), (6.310e+00, 6.607e+00, 1.320e+00),
    (6.607e+00, 6.918e+00, 1.163e+00), (6.918e+00, 7.244e+00, 1.016e+00),
    (7.244e+00, 7.586e+00, 8.770e-01), (7.586e+00, 7.943e+00, 7.641e-01),
    (7.943e+00, 8.318e+00, 6.729e-01), (8.318e+00, 8.710e+00, 5.919e-01),
    (8.710e+00, 9.120e+00, 5.160e-01), (9.120e+00, 9.550e+00, 4.442e-01),
    (9.550e+00, 1.070e+01, 3.862e-01), (1.070e+01, 1.315e+01, 2.385e-01),
    (1.315e+01, 1.616e+01, 1.309e-01), (1.616e+01, 1.986e+01, 7.143e-02),
    (1.986e+01, 2.441e+01, 3.867e-02), (2.441e+01, 3.000e+01, 2.076e-02),
]


# ---------------------------------------------------------------------------
# Benchmark parameters (McClarren & Gentile 2021, "original vacuum")
# ---------------------------------------------------------------------------
RHO_SLAB = 0.1       # g/cm^3
L_SLAB = 0.4         # cm
T_SLAB = 1.0         # keV
V_SLAB = 0.5994      # cm/ns
Z_OBS = 12.0         # cm
T_OBS = 10.0         # ns
MU_ORDER = 256
NU_ORDER = 32


# ---------------------------------------------------------------------------
# Semi-analytic solution
# ---------------------------------------------------------------------------

def planck_energy_form(nu_keV, T_keV):
    if nu_keV <= 0.0 or T_keV <= 0.0:
        return 0.0
    x = nu_keV / T_keV
    bose = math.exp(-x) if x > 700.0 else 1.0 / math.expm1(x)
    prefac = 2.0 * (K_BOLTZMANN_GJ_PER_KEV ** 4) / (C_LIGHT ** 2 * PLANCK_GJ_NS ** 3)
    return prefac * (nu_keV ** 3) * bose


def gamma_D(mu, beta):
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return gamma * (1.0 - mu * beta)


class PiecewiseOpacity:
    def __init__(self, table, rho_slab):
        self.nu_min = np.array([r[0] for r in table])
        self.nu_max = np.array([r[1] for r in table])
        self.kappa = np.array([r[2] for r in table])
        self.sigma = rho_slab * self.kappa
        self.boundaries = np.concatenate([self.nu_min[:1], self.nu_max])

    def sigma_at(self, nu_keV):
        if nu_keV < self.nu_min[0] or nu_keV > self.nu_max[-1]:
            return 0.0
        if nu_keV == self.nu_max[-1]:
            return float(self.sigma[-1])
        idx = np.searchsorted(self.nu_max, nu_keV, side="right")
        return float(self.sigma[idx])

    def breakpoints_in_lab_interval(self, nu_lo, nu_hi, gD):
        pts = [nu_lo, nu_hi]
        if gD <= 0.0:
            return pts
        mapped = self.boundaries[1:-1] / gD
        for x in mapped:
            if nu_lo < x < nu_hi:
                pts.append(float(x))
        return sorted(set(round(p, 15) for p in pts))


def gauss_legendre_integrate(func, a, b, order):
    x, w = np.polynomial.legendre.leggauss(order)
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    vals = np.array([func(xm + xr * xi) for xi in x])
    return xr * float(np.dot(w, vals))


def intensity_original(mu, nu_lab, opacity):
    """Eq. (13)-(14) of McClarren & Gentile (2021)."""
    beta = V_SLAB / C_LIGHT
    denom = mu * C_LIGHT - V_SLAB
    if denom <= 0.0:
        return 0.0
    tb = max(0.0, (mu * C_LIGHT * T_OBS - Z_OBS) / denom)
    tf = max(0.0, (L_SLAB + mu * C_LIGHT * T_OBS - Z_OBS) / denom)
    s = max(0.0, C_LIGHT * (tf - tb))
    if s <= 0.0:
        return 0.0
    gD = gamma_D(mu, beta)
    nu_fluid = gD * nu_lab
    sigma = opacity.sigma_at(nu_fluid)
    if sigma <= 0.0:
        return 0.0
    source = planck_energy_form(nu_fluid, T_SLAB) / (gD ** 3)
    return source * (1.0 - math.exp(-gD * sigma * s))


def group_average_energy_density(nu_lo, nu_hi, opacity):
    """E_{r,g} in GJ/(cm^3 keV) for one group."""
    beta = V_SLAB / C_LIGHT

    def mu_integrand(mu):
        pts = opacity.breakpoints_in_lab_interval(nu_lo, nu_hi, gamma_D(mu, beta))
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            if b <= a:
                continue
            total += gauss_legendre_integrate(
                lambda nu: intensity_original(mu, nu, opacity), a, b, NU_ORDER)
        return total

    mu_int = gauss_legendre_integrate(mu_integrand, 0.0, 1.0, MU_ORDER)
    return (2.0 * math.pi / C_LIGHT) * mu_int / (nu_hi - nu_lo)


def compute_reference():
    """Compute semi-analytic reference for all 124 groups."""
    opacity = PiecewiseOpacity(_OPACITY_TABLE, RHO_SLAB)
    ref = np.zeros(len(_OPACITY_TABLE))
    for g, (nu_lo, nu_hi, _) in enumerate(_OPACITY_TABLE):
        ref[g] = group_average_energy_density(nu_lo, nu_hi, opacity)
        if (g + 1) % 10 == 0:
            print(f"  computed {g + 1}/{len(_OPACITY_TABLE)} groups", flush=True)
    return ref


# ---------------------------------------------------------------------------
# Read STORM output
# ---------------------------------------------------------------------------

def read_spectrum(path):
    meta = {}
    data_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("# ").split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                    if key == "columns:":
                        continue
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        meta[key] = val
                continue
            data_lines.append(line)
    cols = np.loadtxt(data_lines, ndmin=2)
    return meta, cols


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def energy_weighted_fractional_error(code, ref, floor_frac=1e-4):
    """Eq. 20 of the 2026 paper."""
    ref_max = np.max(ref)
    if ref_max <= 0:
        return 0.0
    mask = ref > floor_frac * ref_max
    if not np.any(mask):
        return 0.0
    c, r = code[mask], ref[mask]
    num = np.sum(c * np.abs(c - r) / r)
    den = np.sum(c)
    return float(num / den) if den > 0 else float("inf")


def relative_l1(code, ref, floor_frac=1e-4):
    ref_max = np.max(ref)
    if ref_max <= 0:
        return 0.0
    mask = ref > floor_frac * ref_max
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(code[mask] - ref[mask])) / np.sum(ref[mask]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare STORM moving-slab MC spectrum to semi-analytic solution")
    parser.add_argument("--spectrum", default="moving_slab_mc_spectrum.txt",
                        help="Path to moving_slab_mc_spectrum.txt")
    parser.add_argument("--max-ferror", type=float, default=0.30,
                        help="Maximum allowed energy-weighted fractional error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)
    G = cols.shape[0]
    nu_min_keV = cols[:, 1]
    nu_max_keV = cols[:, 2]
    eg_time_avg = cols[:, 3]

    code_erad = eg_time_avg / ((nu_max_keV - nu_min_keV) * ERG_PER_GJ)

    print(f"Read {G} groups from {args.spectrum}")
    print(f"Computing semi-analytic reference ({MU_ORDER}-pt mu x {NU_ORDER}-pt nu quadrature)...")
    ref_erad = compute_reference()

    ferr = energy_weighted_fractional_error(code_erad, ref_erad)
    l1 = relative_l1(code_erad, ref_erad)

    print()
    print(f"FERROR  = {ferr:.8e}")
    print(f"L1      = {l1:.8e}")
    print(f"Moving slab MC check:")
    print(f"  Groups              = {G}")
    print(f"  Observer x          = {meta.get('observer_x_cm', '?')} cm")
    print(f"  F-error             = {ferr:.6f}")
    print(f"  Rel. L1             = {l1:.6f}")
    print(f"  Threshold           = {args.max_ferror}")

    plot_dir = args.plot_dir or os.path.dirname(os.path.abspath(args.spectrum))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nu_center = np.sqrt(nu_min_keV * nu_max_keV)
        pos_code = code_erad > 0
        pos_ref = ref_erad > 0

        fig, ax = plt.subplots(figsize=(8, 5))
        if np.any(pos_ref):
            ax.loglog(nu_center[pos_ref], ref_erad[pos_ref],
                      "-", lw=1.5, color="C0", label="Semi-analytic")
        if np.any(pos_code):
            ax.loglog(nu_center[pos_code], code_erad[pos_code],
                      "x", ms=4, mew=1.0, color="C1", label="STORM MC")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(r"$E_{r,g}$ (GJ/cm$^3$/keV)")
        ax.set_title(f"Moving slab MC (STORM)\n"
                     f"F-error = {ferr:.4f}, L1 = {l1:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim(1e-2, 20)
        ax.set_ylim(1e-7, 2e-3)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        base = os.path.join(plot_dir, "storm_moving_slab_comparison")
        fig.savefig(base + ".png", dpi=200)
        fig.savefig(base + ".pdf")
        plt.close(fig)
        print(f"  Plot: {base}.png")
        print(f"  Plot: {base}.pdf")
    except ImportError:
        print("  matplotlib not available -- skipping plots")

    if ferr > args.max_ferror:
        print(f"\nFAIL: F-error {ferr:.6f} exceeds threshold {args.max_ferror}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
