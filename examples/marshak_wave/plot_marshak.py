#!/usr/bin/env python3
"""
Plot STORM Marshak wave MC profile against the reference diffusion solution.

Usage:
    python3 plot_marshak.py <problem_number> [profile.txt] [reference.txt]

If paths are omitted, defaults to:
    marshak_wave_<N>_profile.txt   (in cwd)
    <script_dir>/../marshak_wave_<N>/reference.txt
"""

import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_marshak.py <problem_number> [profile.txt] [reference.txt]")
        sys.exit(1)

    problem = int(sys.argv[1])
    script_dir = os.path.dirname(os.path.abspath(__file__))

    profile_path = sys.argv[2] if len(sys.argv) >= 3 else f"marshak_wave_{problem}_profile.txt"
    ref_path = (sys.argv[3] if len(sys.argv) >= 4
                else os.path.join(script_dir, "..", f"marshak_wave_{problem}", "reference.txt"))

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
    sim_Tgas_keV = sim[:, 1] / keV_K
    sim_Trad_keV = sim[:, 2] / keV_K if sim.shape[1] >= 3 else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sim_x, sim_Tgas_keV, "o-", ms=3, lw=1, label="STORM $T_{gas}$")
    if sim_Trad_keV is not None:
        ax.plot(sim_x, sim_Trad_keV, "s-", ms=3, lw=1, label="STORM $T_{rad}$")

    if os.path.exists(ref_path):
        ref = np.loadtxt(ref_path)
        ref_x = ref[:, 0]
        ref_Tgas_keV = ref[:, 1] / keV_K
        ref_Trad_keV = ref[:, 2] / keV_K
        ax.plot(ref_x, ref_Tgas_keV, "k-", lw=1.5, label="Reference $T_{gas}$")
        ax.plot(ref_x, ref_Trad_keV, "k--", lw=1.5, label="Reference $T_{rad}$")
    else:
        print(f"Reference not found: {ref_path} -- plotting MC only")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Temperature [keV]")
    ax.set_title(f"Marshak Wave Problem {problem}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    base = f"marshak_wave_{problem}"
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"Saved {base}.png and {base}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
