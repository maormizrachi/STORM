#!/usr/bin/env python3
"""Compare STORM Crooked Pipe probes with Steinberg & Heizler Fig. 8(a)."""

import argparse
import csv
import math
import os
import sys


PROBE_COUNT = 5
HEAT_THRESHOLD_KEV = 0.08


def ReadRows(path, expectedColumns):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for rawRow in csv.reader(line for line in stream if not line.lstrip().startswith("#")):
            if not rawRow:
                continue
            if len(rawRow) != expectedColumns:
                raise ValueError(f"{path}: expected {expectedColumns} columns, found {len(rawRow)}")
            row = [float(value.strip()) for value in rawRow]
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"{path}: non-finite value")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two samples")
    if any(rows[index][0] <= 0 or rows[index][0] <= rows[index - 1][0] for index in range(1, len(rows))):
        raise ValueError(f"{path}: times must be positive and strictly increasing")
    return rows


def InterpolateLogTime(rows, timeNs, column):
    if timeNs < rows[0][0] or timeNs > rows[-1][0]:
        return None
    lower = 0
    upper = len(rows) - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if rows[middle][0] <= timeNs:
            lower = middle
        else:
            upper = middle
    x0 = math.log10(rows[lower][0])
    x1 = math.log10(rows[upper][0])
    fraction = (math.log10(timeNs) - x0) / (x1 - x0)
    return rows[lower][column] + fraction * (rows[upper][column] - rows[lower][column])


def ArrivalTime(rows, column, threshold):
    for index in range(1, len(rows)):
        before = rows[index - 1][column]
        after = rows[index][column]
        if before < threshold <= after:
            fraction = (threshold - before) / (after - before)
            logTime = math.log10(rows[index - 1][0]) + fraction * (
                math.log10(rows[index][0]) - math.log10(rows[index - 1][0])
            )
            return 10**logTime
    return None


def Compare(simulation, reference, rmseLimit, maxLimit, arrivalLimit):
    failures = []
    for probe in range(PROBE_COUNT):
        simulationColumn = probe + 2
        referenceColumn = probe + 1
        differences = []
        for row in simulation:
            referenceTemperature = InterpolateLogTime(reference, row[0], referenceColumn)
            if referenceTemperature is None or referenceTemperature < HEAT_THRESHOLD_KEV:
                continue
            differences.append(row[simulationColumn] - referenceTemperature)
        if len(differences) < 3:
            failures.append(f"P{probe + 1}: insufficient post-heating overlap")
            continue
        rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
        maximum = max(abs(value) for value in differences)
        simulationArrival = ArrivalTime(simulation, simulationColumn, HEAT_THRESHOLD_KEV)
        referenceArrival = ArrivalTime(reference, referenceColumn, HEAT_THRESHOLD_KEV)
        arrivalError = math.inf
        if simulationArrival is not None and referenceArrival is not None:
            arrivalError = abs(math.log10(simulationArrival / referenceArrival))
        print(
            f"CROOKED_PIPE_P{probe + 1}_RMSE_KEV={rmse:.8g} "
            f"MAX_ERROR_KEV={maximum:.8g} ARRIVAL_LOG10_ERROR={arrivalError:.8g}"
        )
        if rmse > rmseLimit:
            failures.append(f"P{probe + 1}: RMSE {rmse:.4g} > {rmseLimit:.4g} keV")
        if maximum > maxLimit:
            failures.append(f"P{probe + 1}: max error {maximum:.4g} > {maxLimit:.4g} keV")
        if arrivalError > arrivalLimit:
            failures.append(f"P{probe + 1}: arrival error {arrivalError:.4g} > {arrivalLimit:.4g} decades")
    return failures


def Main():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", help="STORM probe CSV: t_ns, cycle, T1...T5")
    parser.add_argument(
        "--reference",
        default=os.path.join(scriptDirectory, "data", "fig8_dimc.csv"),
        help="digitized Fig. 8 reference CSV",
    )
    parser.add_argument("--rmse-limit", type=float, default=float(os.getenv("CROOKED_PIPE_MAX_RMSE_KEV", "0.05")))
    parser.add_argument("--max-limit", type=float, default=float(os.getenv("CROOKED_PIPE_MAX_ERROR_KEV", "0.10")))
    parser.add_argument(
        "--arrival-limit",
        type=float,
        default=float(os.getenv("CROOKED_PIPE_MAX_ARRIVAL_LOG10_ERROR", "0.30")),
    )
    args = parser.parse_args()

    try:
        simulation = ReadRows(args.probes, PROBE_COUNT + 2)
        reference = ReadRows(args.reference, PROBE_COUNT + 1)
        if any(row[column] < 0 or row[column] > 0.6 for row in simulation for column in range(2, PROBE_COUNT + 2)):
            raise ValueError("simulation temperatures must lie in [0, 0.6] keV")
        failures = Compare(simulation, reference, args.rmse_limit, args.max_limit, args.arrival_limit)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: Crooked Pipe probes match Steinberg & Heizler Fig. 8(a)")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
