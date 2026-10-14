#!/usr/bin/env python3
"""Deterministic reference for the polarized plane-parallel atmosphere.

Solves the azimuth-independent vector transfer equation of Shapiro & Sutherland
(1982, ApJ 263, 902; their eqs. 12a,b, Code 1950) for the two linear
polarization components I_l (electric vector in the meridian plane) and I_r
(perpendicular to it) in a semi-infinite atmosphere with Thomson scattering, a
fraction lambda(tau) of true absorption and LTE emission lambda B(tau):

  mu dI_l/dtau = I_l - (1-lam)(3/8) int [2(1-mu^2)(1-mu'^2) I_l' + mu^2 mu'^2 I_l'
                                          + mu^2 I_r'] dmu' - lam B/2
  mu dI_r/dtau = I_r - (1-lam)(3/8) int [mu'^2 I_l' + I_r'] dmu' - lam B/2

The source functions depend on mu only through mu^2 and three moments of the
field (M1 = int (1-mu^2) I_l, M2 = int mu^2 I_l, M3 = int I_r), so the problem
is a linear system for the moments on a depth grid.  Formal solutions integrate
piecewise-linear sources exactly along discrete ordinates.  The lower boundary
is the diffusion-limit intensity: (B + mu dB/dtau)/2 per component with
absorption, the constant-flux field (source + 3F mu/8) for pure scattering.

Q/I = (I_r - I_l)/(I_r + I_l) is Shapiro & Sutherland's sign (positive for the
electric vector parallel to the surface); F = 2 int_0^1 I(0,mu) mu dmu.

Run as a script: compares each case with the paper's Table 1 and prints the
20-bin averages stored in ReferenceTables.hpp.
"""

import numpy as np

CASES = ("scattering", "lambda16", "lambdatau")

# Shapiro & Sutherland (1982) Table 1: mu, then (I/F, Q/I) for lambda = 0
# (Chandrasekhar 1960), lambda = 1/6 and lambda = tau/(1+tau) (Harrington 1969).
TABLE1 = np.array([
    [0.00, 0.41441, 0.117130, 0.309, 0.2290, 0.354, 0.2833],
    [0.05, 0.47490, 0.089790, 0.350, 0.2057, 0.378, 0.2624],
    [0.10, 0.52397, 0.074480, 0.385, 0.1910, 0.391, 0.2516],
    [0.15, 0.57001, 0.063110, 0.419, 0.1785, 0.402, 0.2421],
    [0.20, 0.61439, 0.054100, 0.455, 0.1669, 0.415, 0.2316],
    [0.25, 0.65770, 0.046670, 0.492, 0.1556, 0.430, 0.2192],
    [0.30, 0.70029, 0.040410, 0.532, 0.1444, 0.451, 0.2045],
    [0.40, 0.78398, 0.030330, 0.621, 0.1217, 0.509, 0.1691],
    [0.50, 0.86637, 0.022520, 0.724, 0.0989, 0.602, 0.1294],
    [0.60, 0.94789, 0.016270, 0.847, 0.0765, 0.738, 0.0912],
    [0.70, 1.02882, 0.011123, 0.991, 0.0551, 0.927, 0.0585],
    [0.80, 1.10931, 0.006818, 1.161, 0.0351, 1.178, 0.0329],
    [0.90, 1.18947, 0.003155, 1.360, 0.0167, 1.501, 0.0137],
    [1.00, 1.26938, 0.0, 1.592, 0.0, 1.907, 0.0],
])
TABLE1_COLUMN = {"scattering": 1, "lambda16": 3, "lambdatau": 5}


def _linear_weights(tau, mu):
    """Per interval: weights (a, b) with int_0^d S(t) e^{-t} dt = a S(near) +
    b S(far) for S linear, and the attenuation e = e^{-d}, d = dtau/mu."""
    d = np.diff(tau) / mu
    e = np.exp(-d)
    small = d < 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(small, d / 2 - d * d / 3, 1.0 - (1.0 - e) / d)
        b = np.where(small, d / 2 - d * d / 6, (1.0 - e) / d - e)
    return a, b, e


def lambda_matrices(tau, mu):
    """Matrices (N x N) giving the upward (towards tau = 0) and downward
    intensities at each node from a source linear between nodes, for one
    mu > 0, and the attenuation e^{-(tau_max - tau)/mu} of the lower boundary."""
    n = len(tau)
    a, b, e = _linear_weights(tau, mu)
    up, dn = np.zeros((n, n)), np.zeros((n, n))
    att = np.zeros(n)
    att[-1] = 1.0
    for k in range(n - 2, -1, -1):
        up[k] = e[k] * up[k + 1]
        up[k, k] += a[k]
        up[k, k + 1] += b[k]
        att[k] = e[k] * att[k + 1]
    for k in range(n - 1):
        dn[k + 1] = e[k] * dn[k]
        dn[k + 1, k + 1] += a[k]
        dn[k + 1, k] += b[k]
    return up, dn, att


def surface_weights(tau, mu):
    """Row 0 of the upward lambda matrix and the attenuation of the lower
    boundary at tau = 0: the emergent intensity is w @ S + att0 I(tau_max)."""
    a, b, e = _linear_weights(tau, mu)
    p = np.concatenate(([1.0], np.cumprod(e)))   # e^{-tau_k/mu}
    w = np.zeros(len(tau))
    w[:-1] += p[:-1] * a
    w[1:] += p[:-1] * b
    return w, p[-1]


def solve(lam, planck=None, tau_max=60.0, n_tau=300, n_mu=24):
    """Emergent (I_l, I_r) as a function of mu.  planck=None: pure scattering
    with unit flux from below; otherwise LTE emission lam B, B = planck(tau)."""
    tau = np.concatenate(([0.0], np.geomspace(1e-5, tau_max, n_tau - 1)))
    n = len(tau)
    x, w = np.polynomial.legendre.leggauss(n_mu)
    lam_v = np.array([lam(t) for t in tau])
    c = (1.0 - lam_v) * 3.0 / 8.0
    if planck is None:
        bv = np.zeros(n)

        def bottom(mu):   # (weight of the bottom-node source, constant)
            return 1.0, mu * 0.375
    else:
        bv = np.array([planck(t) for t in tau]) * lam_v / 2.0
        h = 1e-4
        db = (planck(tau_max + h) - planck(tau_max - h)) / ((tau_max + h) - (tau_max - h))

        def bottom(mu):
            return 0.0, (planck(tau_max) + mu * db) / 2.0

    # Moments over mu' in [-1, 1] (each Gauss node carries both directions):
    # M = A M + rhs, with S_l = c [2(1-mu^2) M1 + mu^2 (M2 + M3)] + bv and
    # S_r = c (M2 + M3) + bv.
    A = np.zeros((3 * n, 3 * n))
    rhs = np.zeros(3 * n)
    for mu, wt in zip(0.5 * (x + 1.0), 0.5 * w):
        up, dn, att = lambda_matrices(tau, mu)
        k_src, k_const = bottom(mu)
        L = up + dn
        L[:, -1] += k_src * att
        const = L @ bv + att * k_const
        m2 = mu * mu
        cl1, cl23 = L * (c * 2.0 * (1.0 - m2)), L * (c * m2)
        for row, f in ((0, wt * (1.0 - m2)), (1, wt * m2)):
            rows = slice(row * n, (row + 1) * n)
            A[rows, :n] += f * cl1
            A[rows, n:2 * n] += f * cl23
            A[rows, 2 * n:] += f * cl23
            rhs[rows] += f * const
        A[2 * n:, n:2 * n] += wt * (L * c)
        A[2 * n:, 2 * n:] += wt * (L * c)
        rhs[2 * n:] += wt * const
    M1, M2, M3 = np.split(np.linalg.solve(np.eye(3 * n) - A, rhs), 3)

    def emergent(mu):
        m2 = mu * mu
        sl = c * (2.0 * (1.0 - m2) * M1 + m2 * (M2 + M3)) + bv
        sr = c * (M2 + M3) + bv
        if mu <= 0.0:
            return sl[0], sr[0]
        w, att0 = surface_weights(tau, mu)
        k_src, k_const = bottom(mu)
        return (w @ sl + att0 * (k_src * sl[-1] + k_const),
                w @ sr + att0 * (k_src * sr[-1] + k_const))

    return emergent


def harrington_planck(u0_teff=20.0):
    """Harrington's B_nu(tau) up to a constant: u0 = h nu / k Teff and the grey
    T^4 = (3/4) Teff^4 (tau + q(tau)), q a fit to the Hopf function (q(0) =
    1/sqrt 3, q(inf) = 0.7104).  Harrington's paper was not at hand; this is the
    form that reproduces the paper's absorption columns best."""
    def q(t):
        return 0.710446 - 0.133311 * np.exp(-3.4488 * t)
    u_surface = u0_teff * (4.0 / (3.0 * q(0.0))) ** 0.25
    return lambda t: 1.0 / np.expm1(u_surface * (q(0.0) / (t + q(t))) ** 0.25)


def deterministic(case):
    """Emergent (I_l, I_r)(mu) of one Table 1 case."""
    if case == "scattering":
        return solve(lambda t: 0.0)
    lam = (lambda t: 1.0 / 6.0) if case == "lambda16" else (lambda t: t / (1.0 + t))
    return solve(lam, harrington_planck())


def at_mu(emergent, mus):
    """I/F and Q/I at the given mu."""
    x, w = np.polynomial.legendre.leggauss(64)
    flux = 2.0 * sum(wt * m * sum(emergent(m)) for m, wt in zip(0.5 * (x + 1.0), 0.5 * w))
    v = np.array([emergent(m) for m in mus])
    i = v.sum(1)
    return i / flux, (v[:, 1] - v[:, 0]) / i


def binned(emergent, nbins=20, nsub=16):
    """Bin averages as the Monte Carlo plane observer counts them (energy
    weighted): I/F = int I mu dmu / int mu dmu / F, Q/I = int Q mu / int I mu.
    Returns an (nbins, 2) array."""
    x, w = np.polynomial.legendre.leggauss(nsub)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    e, q, m = (np.zeros(nbins) for _ in range(3))
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        for mu, wt in zip(lo + (hi - lo) * 0.5 * (x + 1.0), (hi - lo) * 0.5 * w):
            il, ir = emergent(mu)
            e[j] += wt * mu * (il + ir)
            q[j] += wt * mu * (ir - il)
            m[j] += wt * mu
    return np.column_stack([e / m / (2.0 * e.sum()), q / e])


def main():
    for case in CASES:
        em = deterministic(case)
        col = TABLE1_COLUMN[case]
        i_f, q_i = at_mu(em, TABLE1[:, 0])
        print("# %s: mu, I/F (Table 1), Q/I (Table 1)" % case)
        for row, a, b in zip(TABLE1, i_f, q_i):
            print("%5.2f  %8.5f (%8.5f)  %8.5f (%8.5f)" % (row[0], a, row[col], b, row[col + 1]))
        print("# rms difference: I/F %.5f  Q/I %.5f" % (np.sqrt(np.mean((i_f - TABLE1[:, col]) ** 2)),
                                                       np.sqrt(np.mean((q_i - TABLE1[:, col + 1]) ** 2))))
        print("# 20-bin averages {I/F, Q/I} (ReferenceTables.hpp)")
        for a, b in binned(em):
            print("    {%.6f, %.6f}," % (a, b))


if __name__ == "__main__":
    main()
