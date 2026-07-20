#!/usr/bin/env python3
"""Plot a STORM Till-Compton profile against the digitized IN-FBC curve."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_table(path: Path, columns: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0 or data.shape[1] < columns:
        raise ValueError(f"{path} does not contain {columns} columns")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="output path without the .png/.pdf suffix")
    args = parser.parse_args()

    profile = load_table(args.profile, 4)
    reference = load_table(args.reference, 3)
    time = profile[:, 0]
    gas = profile[:, 1]
    radiation = profile[:, 2]
    energy = profile[:, 3]
    relative_drift = (energy - energy[0]) / energy[0] if energy[0] != 0.0 else energy * 0.0

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    axes[0].plot(time, gas, label="STORM $T_{gas}$", color="tab:blue")
    axes[0].plot(time, radiation, label="STORM $T_{rad}$", color="tab:orange")
    axes[0].plot(reference[:, 0], reference[:, 1], "o", ms=3,
                 label="IN-FBC reference $T_{gas}$", color="tab:blue", alpha=0.65)
    axes[0].plot(reference[:, 0], reference[:, 2], "o", ms=3,
                 label="IN-FBC reference $T_{rad}$", color="tab:orange", alpha=0.65)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(time, relative_drift, color="tab:green")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Relative total-energy drift")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.suptitle("Till-Compton equilibration: STORM profile comparison")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
