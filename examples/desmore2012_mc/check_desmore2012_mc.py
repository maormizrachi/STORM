#!/usr/bin/env python3
"""Compare a Densmore 2012 temperature profile with the reference curve."""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare Densmore 2012 MC gas temperature profile against reference."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--max-tgas-l1", type=float, default=0.05)
    args = parser.parse_args()

    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_sim = raw[:, 0]
    temperature_sim = raw[:, 1]

    kevKelvin = 1e3 * 1.602176634e-12 / 1.380649e-16
    temperatureSimKeV = temperature_sim / kevKelvin

    reference = np.loadtxt(args.reference, delimiter=",", comments="#")
    x_reference = reference[:, 0]
    temperatureReferenceKeV = reference[:, 1]
    temperatureReferenceInterp = np.interp(
        x_sim, x_reference, temperatureReferenceKeV)

    l1 = float(np.mean(
        np.abs(temperatureSimKeV - temperatureReferenceInterp)))
    print(f"DESMORE2012_MC_TGAS_L1={l1:.8e}")
    print(f"DESMORE2012_MC_MAX_TGAS_L1={args.max_tgas_l1:.8e}")

    if l1 > args.max_tgas_l1:
        print(
            f"Densmore 2012 MC gas temperature L1 exceeds tolerance "
            f"({l1:.6e} > {args.max_tgas_l1:.6e})",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: L1 = {l1:.6e} < {args.max_tgas_l1:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
