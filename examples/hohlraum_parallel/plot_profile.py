#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile.txt> [--save <output.png>]")
        sys.exit(1)

    filename = sys.argv[1]
    save_path = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            save_path = sys.argv[idx + 1]

    data = np.loadtxt(filename, delimiter=",", comments="#")
    x_cm = data[:, 0]
    T_keV = data[:, 2]

    x_mm = x_cm * 10.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_mm, T_keV, "o-", markersize=3, linewidth=1)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Temperature [keV]")
    ax.set_title("Hohlraum Temperature Profile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
