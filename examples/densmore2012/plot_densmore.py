#!/usr/bin/env python3
"""
Plot STORM Densmore 2012 benchmark against the digitized reference.

Usage:
    python3 plot_densmore.py [profile.txt] [reference.csv]

Defaults:
    densmore2012_profile.txt  (in cwd)
    <script_dir>/data/densmore2012_fig4_mc.csv
"""

import sys
import os
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    profile_path = sys.argv[1] if len(sys.argv) >= 2 else "densmore2012_profile.txt"
    ref_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(script_dir, "data", "densmore2012_fig4_mc.csv")

    keV_K = 1.16045250e7

    if not os.path.exists(profile_path):
        print(f"Profile not found: {profile_path}")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot")
        sys.exit(0)

    sim = np.loadtxt(profile_path)
    sim_x = sim[:, 0]
    sim_T_keV = sim[:, 1] / keV_K

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sim_x, sim_T_keV, "-", lw=1, label="STORM MC (gray)")

    if os.path.exists(ref_path):
        ref = np.loadtxt(ref_path, delimiter=",", comments="#")
        ref_x = ref[:, 0]
        ref_T = ref[:, 1]
        ax.plot(ref_x, ref_T, "ks", ms=5, label="Densmore 2012 (Milagro MC)")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Temperature [keV]")
    ax.set_title("Densmore et al. (2012) Heterogeneous Step-Opacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)

    base = "densmore2012"
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"Saved {base}.png and {base}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
