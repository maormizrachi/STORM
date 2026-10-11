#!/usr/bin/env bash
# examples/ddmc_advection/regression_checks.sh
#
# Check functions for the DDMC advection examples (ddmc_advection_pulse and
# ddmc_advection_interface).  Their REGRESSION_INFO selects this file with
# CHECK_LIBRARY; THUNDER sources it (it is not executed) and calls
#   check_function run_dir start_epoch stdout_log stderr_log
# The shared helpers come from regression_tests/lib/regression_checks.sh.
# The pass/fail criteria live in check_ddmc_advection.py next to this file.

DDMC_ADVECTION_CHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../regression_tests/lib/regression_checks.sh
source "${DDMC_ADVECTION_CHECK_DIR}/../../regression_tests/lib/regression_checks.sh"

# $1 = pulse|interface, $2 = metrics file, $3 = start epoch,
# $4 = stdout log, $5 = stderr log, $6 = label for messages
run_ddmc_advection_check() {
    local example="$1"
    local metrics="$2"
    local start_epoch="$3"
    local stdout_log="$4"
    local stderr_log="$5"
    local label="$6"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1
    is_nonempty_and_newer "$metrics" "$start_epoch" || return 1

    local output
    if ! output="$(python3 "${DDMC_ADVECTION_CHECK_DIR}/check_ddmc_advection.py" \
        "$example" "$metrics" 2>&1)"; then
        printf '%s\n' "$output"
        REGRESSION_CHECK_MSG="$(printf '%s\n' "$output" | grep '^FAIL:' | tail -n 1 || true)"
        REGRESSION_CHECK_MSG="${REGRESSION_CHECK_MSG:-${label} check failed}"
        return 1
    fi

    printf '%s\n' "$output"
    REGRESSION_CHECK_MSG="$(printf '%s\n' "$output" | grep -m1 '^PASS:' || true)"
    REGRESSION_CHECK_MSG="${REGRESSION_CHECK_MSG:-PASS (${label})}"
    return 0
}

# DDMC radiation pulse advected by uniformly moving gas: centroid moves by
# v t for IMC and for DDMC on a static and a moving mesh, comoving energy is
# conserved; DDMC without advection is recorded only.
check_ddmc_advection_pulse_case() {
    local run_dir="$1"
    run_ddmc_advection_check pulse "${run_dir}/ddmc_advection_pulse_metrics.txt" \
        "$2" "$3" "$4" "DDMC advection pulse"
}

# Uniform radiation in uniform flow across an IMC-DDMC interface: flat to 3 %
# next to the interface with advection and the moving-interface correction;
# the other variants are recorded only.
check_ddmc_advection_interface_case() {
    local run_dir="$1"
    run_ddmc_advection_check interface "${run_dir}/ddmc_advection_interface_metrics.txt" \
        "$2" "$3" "$4" "DDMC advection interface"
}
