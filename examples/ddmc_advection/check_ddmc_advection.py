#!/usr/bin/env python3
"""Pass/fail criteria for the DDMC advection regression examples.

    check_ddmc_advection.py pulse     ddmc_advection_pulse_metrics.txt
    check_ddmc_advection.py interface ddmc_advection_interface_metrics.txt

Each metrics file holds one line per variant of space-separated key=value
pairs, written by the example's main.cpp; lines starting with '#' are
comments.  Every criterion prints one line: "ok:" for a gated criterion that
holds, "FAIL:" for one that does not, "RECORD:" for a value that is reported
but not gated.  The last line is "PASS: ..." or "FAIL: ...", and the exit
status is non-zero if any gated criterion failed.

Only the standard library is used, so the check runs wherever python3 does.
"""

import argparse
import math
import sys


PULSE_GATED_VARIANTS = ("imc", "ddmc_static", "ddmc_moving_mesh")
PULSE_RECORDED_VARIANTS = ("ddmc_no_advection",)
INTERFACE_GATED_VARIANTS = ("moving_correction", "static_control")
INTERFACE_RECORDED_VARIANTS = ("no_moving_correction", "no_advection")


class Report:
    def __init__(self):
        self.failures = []

    def Gate(self, passed, message):
        if passed:
            print(f"ok: {message}")
        else:
            print(f"FAIL: {message}")
            self.failures.append(message)

    def Record(self, message):
        print(f"RECORD: {message}")

    def Finish(self, passSummary):
        if self.failures:
            extra = f" (+{len(self.failures) - 1} more)" if len(self.failures) > 1 else ""
            print(f"FAIL: {self.failures[0]}{extra}")
            return 1
        print(f"PASS: {passSummary}")
        return 0


def ReadMetrics(path):
    variants = {}
    with open(path, encoding="utf-8") as stream:
        for lineNumber, rawLine in enumerate(stream, 1):
            line = rawLine.strip()
            if not line or line.startswith("#"):
                continue
            entry = {}
            for token in line.split():
                key, separator, value = token.partition("=")
                if not separator or not key:
                    raise ValueError(f"{path}:{lineNumber}: expected key=value, found {token!r}")
                entry[key] = value
            name = entry.get("variant")
            if name is None:
                raise ValueError(f"{path}:{lineNumber}: line has no variant= field")
            if name in variants:
                raise ValueError(f"{path}:{lineNumber}: variant {name!r} appears twice")
            variants[name] = entry
    if not variants:
        raise ValueError(f"{path}: no metrics lines")
    return variants


def Number(entry, key):
    if key not in entry:
        raise KeyError(f"variant {entry.get('variant')!r} has no {key!r} metric")
    return float(entry[key])


def Count(entry, key):
    if key not in entry:
        raise KeyError(f"variant {entry.get('variant')!r} has no {key!r} metric")
    return int(entry[key])


def Finite(*values):
    return all(math.isfinite(value) for value in values)


def CheckPulse(variants, args, report):
    for name in PULSE_GATED_VARIANTS:
        if name not in variants:
            report.Gate(False, f"{name}: variant missing from the metrics file")
            continue
        entry = variants[name]
        ratio = Number(entry, "dx_over_vt")
        report.Gate(Finite(ratio) and abs(ratio - 1.0) <= args.max_centroid_error,
                    f"{name}: centroid displacement / (v t) = {ratio:.4f} "
                    f"(+- {Number(entry, 'centroid_err_over_vt'):.4f} statistical; "
                    f"allowed 1 +- {args.max_centroid_error})")
        drift = Number(entry, "e0_rel_change")
        report.Gate(Finite(drift) and abs(drift) <= args.max_energy_drift,
                    f"{name}: comoving energy relative change = {drift:.3e} "
                    f"(allowed {args.max_energy_drift:.1e}; removed by mesh motion "
                    f"{Number(entry, 'e0_removed'):.3e} erg)")
        if Count(entry, "ddmc"):
            eligible = Count(entry, "ddmc_eligible_cells")
            cells = Count(entry, "cells")
            report.Gate(eligible == cells,
                        f"{name}: {eligible} of {cells} cells DDMC eligible (all required)")
        report.Record(f"{name}: variance / (var0 + 2 D t) = {Number(entry, 'var_ratio_diffusion'):.3f}, "
                      f"/ (var0 + 2 (D + v h / 2) t) = {Number(entry, 'var_ratio_upwind'):.3f} "
                      f"(var0 = {Number(entry, 'var0_cm2'):.3f} cm^2, "
                      f"var1 = {Number(entry, 'var1_cm2'):.3f} cm^2)")

    if "ddmc_static" in variants and "ddmc_moving_mesh" in variants:
        static = variants["ddmc_static"]
        moving = variants["ddmc_moving_mesh"]
        difference = abs(Number(static, "dx_over_vt") - Number(moving, "dx_over_vt"))
        noise = math.hypot(Number(static, "centroid_err_over_vt"), Number(moving, "centroid_err_over_vt"))
        tolerance = max(args.max_mesh_disagreement, 5.0 * noise)
        report.Gate(Finite(difference) and difference <= tolerance,
                    f"ddmc_static vs ddmc_moving_mesh: |difference| of dx / (v t) = {difference:.4f} "
                    f"(allowed {tolerance:.4f}: max of {args.max_mesh_disagreement} and 5 sigma)")

    for name in PULSE_RECORDED_VARIANTS:
        if name not in variants:
            report.Record(f"{name}: not run")
            continue
        entry = variants[name]
        report.Record(f"{name}: centroid displacement / (v t) = {Number(entry, 'dx_over_vt'):.4f} "
                      f"(expected near 0: without advection DDMC radiation does not move with the gas), "
                      f"comoving energy relative change = {Number(entry, 'e0_rel_change'):.3e}")

    for name, entry in variants.items():
        if Count(entry, "ddmc"):
            report.Record(f"{name}: max cell Peclet = {Number(entry, 'max_cell_peclet'):.3f}, "
                          f"Peclet histogram = {entry.get('peclet_histogram', '?')}, "
                          f"advective leaks (host path only) = {entry.get('advective_leaks', '?')}, "
                          f"max interior |div v| h / v = {Number(entry, 'max_interior_divv_h_over_v'):.2e}")

    return "DDMC pulse moves with the gas (IMC, static and moving mesh) and conserves comoving energy"


def CheckInterface(variants, args, report):
    for name in INTERFACE_GATED_VARIANTS + INTERFACE_RECORDED_VARIANTS:
        if name not in variants:
            if name in INTERFACE_GATED_VARIANTS:
                report.Gate(False, f"{name}: variant missing from the metrics file")
            else:
                report.Record(f"{name}: not run")
            continue
        entry = variants[name]
        mismatches = Count(entry, "eligibility_mismatches")
        report.Gate(mismatches == 0,
                    f"{name}: DDMC eligibility is exactly the thick region "
                    f"({Count(entry, 'ddmc_eligible_cells')} eligible of "
                    f"{Count(entry, 'expected_ddmc_cells')} thick cells, {mismatches} mismatches)")
        deviation = Number(entry, "max_dev_window")
        ratio = Number(entry, "e_ratio_interface")
        half = Count(entry, "window_half_width")
        summary = (f"{name}: max |E/E0 - 1| within {half} cells of the interface = {deviation:.4f}, "
                   f"E(last DDMC) / E(first IMC) = {ratio:.4f}, "
                   f"window = [{entry.get('window_profile', '?')}], "
                   f"bulk thick/thin = {Number(entry, 'thick_bulk_rel'):.4f}/{Number(entry, 'thin_bulk_rel'):.4f}, "
                   f"census max dev = {Number(entry, 'max_dev_window_census'):.4f}")
        if name == "static_control":
            report.Gate(Finite(deviation) and deviation <= args.max_flat_deviation,
                        f"{summary} (allowed {args.max_flat_deviation})")
        elif name in INTERFACE_GATED_VARIANTS:
            # A fixed IMC-DDMC interface with gas flowing through it keeps an
            # O(beta sigma h) jump: the asymptotic interface condition and the
            # moving-interface correction G_U are Lagrangian (interface moving
            # with the gas). The advective leak factor removes part of it
            # (0.27 -> 0.21 here; 0.38 without advection). The gate guards
            # against regressing; it is not the physical answer.
            report.Gate(Finite(deviation) and deviation <= args.max_moving_deviation,
                        f"{summary} (allowed {args.max_moving_deviation}: known fixed-interface error)")
        else:
            report.Record(summary)

    return "uniform radiation stays flat across a static IMC-DDMC interface and within the known bound across a moving one"


def Main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=("pulse", "interface"))
    parser.add_argument("metrics", help="metrics file written by the example")
    parser.add_argument("--max-centroid-error", type=float, default=0.05,
                        help="pulse: allowed |dx / (v t) - 1| for the gated variants")
    parser.add_argument("--max-energy-drift", type=float, default=1.0e-10,
                        help="pulse: allowed |relative change| of the total comoving energy")
    parser.add_argument("--max-mesh-disagreement", type=float, default=0.03,
                        help="pulse: allowed |dx / (v t)| difference between static and moving mesh")
    parser.add_argument("--max-flat-deviation", type=float, default=0.03,
                        help="interface: allowed max |E / E0 - 1| next to a static interface")
    parser.add_argument("--max-moving-deviation", type=float, default=0.25,
                        help="interface: allowed max |E / E0 - 1| next to the interface with gas flowing through it")
    args = parser.parse_args()

    report = Report()
    try:
        variants = ReadMetrics(args.metrics)
        if args.example == "pulse":
            summary = CheckPulse(variants, args, report)
        else:
            summary = CheckInterface(variants, args, report)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: cannot evaluate {args.metrics}: {error}")
        return 1
    return report.Finish(summary)


if __name__ == "__main__":
    sys.exit(Main())
