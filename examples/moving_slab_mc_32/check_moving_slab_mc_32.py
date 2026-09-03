#!/usr/bin/env python3
"""Validate the 32-group moving-slab spectrum against its reference solution."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def ReadSpectrum(path):
    metadata = {}
    dataLines = []
    with open(path, encoding="utf-8") as stream:
        for rawLine in stream:
            line = rawLine.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("# ").split(None, 1)
                if len(parts) == 2 and parts[0] != "columns:":
                    try:
                        metadata[parts[0]] = float(parts[1])
                    except ValueError:
                        metadata[parts[0]] = parts[1]
                continue
            dataLines.append(line)
    return metadata, np.loadtxt(dataLines, ndmin=2)


def EnergyWeightedFractionalError(code, reference, floorFraction=1.0e-4):
    referenceMaximum = np.max(reference)
    if referenceMaximum <= 0.0:
        return 0.0
    mask = reference > floorFraction * referenceMaximum
    if not np.any(mask):
        return 0.0
    codeSignificant = code[mask]
    referenceSignificant = reference[mask]
    denominator = np.sum(codeSignificant)
    if denominator <= 0.0:
        return float("inf")
    numerator = np.sum(codeSignificant * np.abs(codeSignificant - referenceSignificant) / referenceSignificant)
    return float(numerator / denominator)


def RelativeL1(code, reference, floorFraction=1.0e-4):
    referenceMaximum = np.max(reference)
    if referenceMaximum <= 0.0:
        return 0.0
    mask = reference > floorFraction * referenceMaximum
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(code[mask] - reference[mask])) / np.sum(reference[mask]))


def WritePlots(plotDirectory, groupLower, groupUpper, opacity, code, reference, fError, l1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    groupCenter = np.sqrt(groupLower * groupUpper)
    figure, axes = plt.subplots(figsize=(8, 5))
    positiveCode = code > 0.0
    positiveReference = reference > 0.0
    axes.loglog(groupCenter[positiveReference], reference[positiveReference], "-", lw=1.5, color="C0", label="Semi-analytic (32-group)")
    axes.loglog(groupCenter[positiveCode], code[positiveCode], "x", ms=5, mew=1.0, color="C1", label="MC simulation (32-group)")
    axes.set_xlabel("Energy (keV)")
    axes.set_ylabel(r"$E_{r,g}$ (GJ/cm$^3$/keV)")
    axes.set_title(f"Moving slab MC (32-group, original vacuum)\nF-error = {fError:.4f}, L1 = {l1:.4f}")
    axes.legend(fontsize=9)
    axes.set_xlim(1.0e-2, 20.0)
    axes.set_ylim(1.0e-7, 2.0e-3)
    axes.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    comparisonBase = os.path.join(plotDirectory, "moving_slab_mc_32_comparison")
    figure.savefig(comparisonBase + ".png", dpi=200)
    figure.savefig(comparisonBase + ".pdf")
    plt.close(figure)
    print(f"  Plot: {comparisonBase}.png")
    print(f"  Plot: {comparisonBase}.pdf")

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.step(groupLower, opacity, where="post", lw=1.2, color="C3")
    axes.scatter(groupCenter, opacity, s=24, marker="x", linewidths=1.0, color="C3", zorder=3)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Energy (keV)")
    axes.set_ylabel(r"$\kappa$ (cm$^2$/g)")
    axes.set_title("32-group collapsed aluminum opacity (Planck-weighted at 1 keV)")
    axes.set_xlim(groupLower[0], groupUpper[-1])
    axes.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    opacityBase = os.path.join(plotDirectory, "moving_slab_mc_32_opacity")
    figure.savefig(opacityBase + ".png", dpi=200)
    figure.savefig(opacityBase + ".pdf")
    plt.close(figure)
    print(f"  Plot: {opacityBase}.png")
    print(f"  Plot: {opacityBase}.pdf")


def Main():
    parser = argparse.ArgumentParser(description="Moving slab MC 32-group spectrum check")
    parser.add_argument("--spectrum", required=True, help="Path to moving_slab_mc_32_spectrum.txt")
    parser.add_argument("--max-ferror", type=float, default=0.30, help="Maximum allowed energy-weighted fractional error")
    parser.add_argument("--plot-dir", default=None, help="Directory for plots (default: same as spectrum)")
    parser.add_argument("--no-plots", action="store_true", help="Skip comparison plot generation")
    arguments = parser.parse_args()

    metadata, columns = ReadSpectrum(arguments.spectrum)
    if columns.shape[0] != 32 or columns.shape[1] < 5:
        print(f"FAIL: expected a 32-row, 5-column spectrum; found {columns.shape}")
        return 1

    groupLower = columns[:, 1]
    groupUpper = columns[:, 2]
    opacity = columns[:, 3]
    code = columns[:, 4] / ((groupUpper - groupLower) * 1.0e16)
    referencePath = Path(__file__).with_name("moving_slab_mc_32_reference.txt")
    reference = np.loadtxt(referencePath)
    if reference.shape != code.shape:
        print(f"FAIL: reference shape {reference.shape} does not match spectrum shape {code.shape}")
        return 1

    fError = EnergyWeightedFractionalError(code, reference)
    l1 = RelativeL1(code, reference)
    print(f"MOVING_SLAB_MC_32_FERROR={fError:.8e}")
    print(f"MOVING_SLAB_MC_32_L1={l1:.8e}")
    print(f"MOVING_SLAB_MC_32_MAX_FERROR={arguments.max_ferror:.8e}")
    print("Moving slab MC 32-group check:")
    print(f"  Groups              = {columns.shape[0]}")
    print(f"  Observer x          = {metadata.get('observer_x_cm', '?')} cm")
    print(f"  F-error             = {fError:.6f}")
    print(f"  Rel. L1             = {l1:.6f}")
    print(f"  Threshold           = {arguments.max_ferror}")
    if not arguments.no_plots:
        WritePlots(arguments.plot_dir or os.path.dirname(arguments.spectrum), groupLower, groupUpper, opacity, code, reference, fError, l1)

    if fError > arguments.max_ferror:
        print(f"FAIL: F-error {fError:.6f} exceeds threshold {arguments.max_ferror}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
