#!/usr/bin/env python3
"""Net polarization and luminosity of a spheroidal scattering atmosphere
(Shapiro & Sutherland 1982, ApJ 263, 902, section IIIb-c, eqs. 19-39).

Each surface element of an ellipsoid of revolution radiates like the
semi-infinite plane-parallel atmosphere, with law of darkening (I/F)(mu) and
polarization Pi(mu) = Q/I (positive for the electric vector parallel to the
surface, i.e. perpendicular to the plane containing the normal and the line
of sight), mu the cosine between the local normal and the line of sight, and
uniform surface flux F.  For a view direction k the disc-integrated Stokes
parameters, referred to the projected symmetry axis, are

    I_net = int (I/F)(mu) F mu dA,
    Q_net = int (I/F)(mu) F (-Pi(mu)) cos 2 chi mu dA,

chi the angle on the sky between the projected symmetry axis and the
projected normal.  Pi_net = |Q_net| / I_net.  An observer who assumes a sphere
infers L_ss = 4 F int (I/F) mu dA (same law), so L / L_ss = A_total / (4 int
(I/F)(mu) mu dA) (eq. 39; it uses 2 int_0^1 (I/F) mu dmu = 1).

Asphericity: oblate xi = 1 - c/a, prolate xi = 1 - a/c (eq. 24), a and c the
semi-axes perpendicular and parallel to the symmetry axis.  "equator" views
perpendicular to the axis, "pole" along it.

Two forms of the law are supported:
  continuous: callables (I/F)(mu) and Pi(mu), e.g. splines through Table 1;
  binned: Monte Carlo plane-observer tallies per mu bin (energy E_b and
  meridian-frame Q_b = I_l - I_r), for which Pi_net and L/L_ss are ratios of
  sums linear in the tallies, so per-generation tallies give the errors.
"""

import numpy as np


def surface(xi, shape, n_theta=1200, n_phi=1200):
    """Surface elements of the spheroid: outward unit normals (3, N), areas (N)
    and the total area, for asphericity xi.  Build it once per (xi, shape) and
    pass it to continuous() and bin_weights() for every view and law."""
    if shape == "oblate":
        a, c = 1.0, 1.0 - xi
    elif shape == "prolate":
        a, c = 1.0 - xi, 1.0
    else:
        raise ValueError(shape)
    c = max(c, 1e-9)
    a = max(a, 1e-9)
    # Midpoint grid in the parametric angles; the cos(theta) grid concentrates
    # points where the limb of an oblate disc lies.
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    ph = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    T, P = np.meshgrid(th, ph, indexing="ij")
    st, ct, sp, cp = np.sin(T), np.cos(T), np.sin(P), np.cos(P)
    # dP/dtheta x dP/dphi for P = (a st cp, a st sp, c ct)
    vx = a * c * st * st * cp
    vy = a * c * st * st * sp
    vz = a * a * st * ct
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    dA = norm * (np.pi / n_theta) * (2.0 * np.pi / n_phi)
    n = np.stack([vx / norm, vy / norm, vz / norm]).reshape(3, -1)
    return n, dA.ravel(), dA.sum()


def view_geometry(n, view):
    """mu = n.k and cos 2 chi for the visible elements."""
    if view == "equator":
        mu = n[0]
        ly, lz = n[1], n[2]   # projection of n on the sky (y, z); reference axis z
    else:
        mu = n[2]
        ly, lz = n[1], n[0]   # reference axis x (any: Q_net = 0 by symmetry)
    s = ly * ly + lz * lz
    with np.errstate(invalid="ignore", divide="ignore"):
        cos2chi = np.where(s > 0.0, (lz * lz - ly * ly) / s, 0.0)
    vis = mu > 0.0
    return mu[vis], cos2chi[vis], vis


def continuous(law_i, law_p, surf, view):
    """Pi_net and L/L_ss for a continuous law over the surface surf."""
    n, dA, area = surf
    mu, c2, vis = view_geometry(n, view)
    w = law_i(mu) * mu * dA[vis]
    inet = w.sum()
    qnet = (-law_p(mu) * c2 * w).sum()
    return abs(qnet) / inet, area / (4.0 * inet)


def bin_weights(surf, view, edges):
    """Per-bin geometric weights: A_b = sum mu dA / m_b and C_b = sum mu dA
    cos 2 chi / m_b over the visible elements whose mu falls in bin b, with
    m_b = int mu dmu over the bin.  Then, with tallies E_b and meridian Q_b,
    Pi_net = |sum Q_b C_b| / sum E_b A_b and L/L_ss = A_tot sum E / (2 sum E_b A_b)."""
    n, dA, area = surf
    mu, c2, vis = view_geometry(n, view)
    b = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, len(edges) - 2)
    m = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
    w = mu * dA[vis]
    A = np.bincount(b, weights=w, minlength=len(m)) / m
    C = np.bincount(b, weights=w * c2, minlength=len(m)) / m
    return A, C, area


def binned(E, Q, A, C, area):
    """Pi_net and L/L_ss from bin tallies (arrays over bins, or over
    generations x bins: the tallies are summed first)."""
    E = np.asarray(E, float)
    Q = np.asarray(Q, float)
    if E.ndim == 2:
        E, Q = E.sum(0), Q.sum(0)
    den = (E * A).sum()
    return abs((Q * C).sum()) / den, area * E.sum() / (2.0 * den)


def binned_errors(Eg, Qg, A, C, area):
    """Generation-scatter errors (delta method) of Pi_net and L/L_ss from
    per-generation tallies Eg, Qg (generations x bins)."""
    Eg = np.asarray(Eg, float)
    Qg = np.asarray(Qg, float)
    n = Eg.shape[0]
    num_g = (Qg * C).sum(1)
    den_g = (Eg * A).sum(1)
    tot_g = Eg.sum(1)
    num, den, tot = num_g.sum(), den_g.sum(), tot_g.sum()
    pi = abs(num) / den
    l = area * tot / (2.0 * den)
    dpi = (np.sign(num) * num_g - pi * den_g) / den
    dl = l * (tot_g / tot - den_g / den)
    corr = n / (n - 1.0) if n > 1 else 0.0
    return np.sqrt(corr * (dpi ** 2).sum()), np.sqrt(corr * (dl ** 2).sum())
