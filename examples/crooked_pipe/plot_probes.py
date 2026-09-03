#!/usr/bin/env python3
"""Plot crooked-pipe probe histories against the digitized Fig. 8(a) curves.

Usage: plot_probes.py output.png probes.txt[=label] [more probe files ...]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBE_COUNT = 5
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COLORS = ["C0", "C1", "C2", "C3", "C4"]
STYLES = ["-", ":", "-.", (0, (3, 1, 1, 1))]


def ReadTable(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(value) for value in line.split(",")])
    return rows


def Main(arguments):
    if len(arguments) < 2:
        print(__doc__.strip())
        return 1

    outputPath = arguments[0]
    series = []
    for argument in arguments[1:]:
        path, separator, label = argument.partition("=")
        series.append((path, label if separator else os.path.basename(path)))

    reference = ReadTable(os.path.join(DATA_DIR, "fig8_dimc.csv"))
    plt.figure(figsize=(7.5, 4.8))
    for probe in range(PROBE_COUNT):
        plt.plot([row[0] for row in reference], [row[probe + 1] for row in reference],
                 color=COLORS[probe], linestyle="--", linewidth=1.1)
    for order, (path, label) in enumerate(series):
        rows = ReadTable(path)
        for probe in range(PROBE_COUNT):
            plt.plot([row[0] for row in rows], [row[probe + 2] for row in rows],
                     color=COLORS[probe], linestyle=STYLES[order % len(STYLES)], linewidth=1.4,
                     label=f"{label} P{probe + 1}" if probe == 0 else None)

    plt.xscale("log")
    plt.xlim(1.0, 1000.0)
    plt.xlabel("time [ns]")
    plt.ylabel("material temperature [keV]")
    plt.title("Crooked pipe probes, dashed = Steinberg & Heizler DIMC")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=130)
    print(outputPath)
    return 0


if __name__ == "__main__":
    sys.exit(Main(sys.argv[1:]))
