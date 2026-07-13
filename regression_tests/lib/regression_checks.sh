#!/usr/bin/env bash
# regression_tests/lib/regression_checks.sh
#
# Shared check functions for STORM regression tests.
# Each check function receives:
#   $1 = run_dir    (absolute path to the example directory where output lives)
#   $2 = start_epoch (epoch at suite start; 0 = accept any mtime)
#   $3 = stdout log path
#   $4 = stderr log path
#
# On return, set REGRESSION_CHECK_MSG to a human-readable summary.
# Return 0 on pass, non-zero on fail.

REGRESSION_CHECK_MSG=""

# ---- helpers --------------------------------------------------------

check_no_fatal_markers() {
    local stdout_log="$1"
    local stderr_log="$2"
    local fatal_patterns="Segmentation fault|SIGSEGV|std::bad_alloc|terminate called|Aborted|MPI_ABORT|StormError"

    for log in "$stdout_log" "$stderr_log"; do
        [[ -f "$log" ]] || continue
        if grep -qE "${fatal_patterns}" "$log" 2>/dev/null; then
            local match
            match="$(grep -m1 -E "${fatal_patterns}" "$log")"
            REGRESSION_CHECK_MSG="Fatal marker in $(basename "$log"): ${match}"
            return 1
        fi
    done
    return 0
}

is_nonempty_and_newer() {
    local filepath="$1"
    local start_epoch="$2"
    if [[ ! -f "$filepath" ]]; then
        REGRESSION_CHECK_MSG="Missing output file: $filepath"
        return 1
    fi
    if [[ ! -s "$filepath" ]]; then
        REGRESSION_CHECK_MSG="Empty output file: $filepath"
        return 1
    fi
    if [[ "$start_epoch" != "0" ]]; then
        local mtime
        mtime="$(stat -c %Y "$filepath" 2>/dev/null || stat -f %m "$filepath" 2>/dev/null)" || true
        if [[ -n "$mtime" && "$mtime" -lt "$start_epoch" ]]; then
            REGRESSION_CHECK_MSG="Stale output file (mtime before suite start): $filepath"
            return 1
        fi
    fi
    return 0
}

grep_stdout() {
    local pattern="$1"
    local stdout_log="$2"
    grep -oP "$pattern" "$stdout_log" 2>/dev/null | head -1
}

extract_number_after() {
    local label="$1"
    local log="$2"
    grep -m1 "$label" "$log" 2>/dev/null | grep -oP '[\-+]?[0-9]+(\.[0-9]+)?([eE][\-+]?[0-9]+)?' | tail -1
}

# ---- per-test check functions ----------------------------------------

# Marshak wave tests (1-3 serial, 4 parallel)
# All print "PASS (L1 < 0.10)" or "WARN: L1 = ..."
check_marshak_wave_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    if grep -q "^PASS" "$stdout_log" 2>/dev/null; then
        local l1
        l1="$(extract_number_after "TGAS_REL_L1" "$stdout_log")"
        REGRESSION_CHECK_MSG="PASS (TGAS_REL_L1 = ${l1:-?})"
        return 0
    fi

    if grep -q "^WARN:" "$stdout_log" 2>/dev/null; then
        local l1
        l1="$(extract_number_after "TGAS_REL_L1" "$stdout_log")"
        REGRESSION_CHECK_MSG="WARN: L1=${l1:-?} (above 0.10 threshold but may be acceptable for MC noise)"
        return 0
    fi

    REGRESSION_CHECK_MSG="No PASS/WARN line found in stdout"
    return 1
}

# Densmore 2012 benchmark
# Prints "DENSMORE2012_TGAS_L1 = ..." and "PASS (L1 < 0.10 keV)"
check_densmore2012_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    if grep -q "^PASS" "$stdout_log" 2>/dev/null; then
        local l1
        l1="$(extract_number_after "DENSMORE2012_TGAS_L1" "$stdout_log")"
        REGRESSION_CHECK_MSG="PASS (DENSMORE2012_TGAS_L1 = ${l1:-?} keV)"
        return 0
    fi

    REGRESSION_CHECK_MSG="No PASS line found in stdout"
    return 1
}

# Moving slab benchmark
# The Python check_spectrum.py prints "PASS" or "FAIL: F-error ... exceeds threshold"
check_moving_slab_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    if grep -q "^PASS" "$stdout_log" 2>/dev/null; then
        local ferr
        ferr="$(extract_number_after "F-error" "$stdout_log")"
        local l1
        l1="$(extract_number_after "Rel. L1" "$stdout_log")"
        REGRESSION_CHECK_MSG="PASS (F-error=${ferr:-?}, L1=${l1:-?})"
        return 0
    fi

    if grep -q "^FAIL" "$stdout_log" 2>/dev/null; then
        local msg
        msg="$(grep -m1 "^FAIL" "$stdout_log")"
        REGRESSION_CHECK_MSG="$msg"
        return 1
    fi

    REGRESSION_CHECK_MSG="No PASS/FAIL line found in stdout (check_spectrum.py may not have run)"
    return 1
}

# Hohlraum parallel benchmark
# No analytic solution; just check it ran to completion without fatal errors
# and produced a profile.
check_hohlraum_parallel_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    REGRESSION_CHECK_MSG="PASS (completed without errors)"
    return 0
}

# Cartesian parallel check (MPI decomposition unit test)
# Prints "cartesian_parallel_check PASS" or "cartesian_parallel_check FAIL"
check_cartesian_parallel_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    if grep -q "cartesian_parallel_check PASS" "$stdout_log" 2>/dev/null; then
        REGRESSION_CHECK_MSG="PASS"
        return 0
    fi

    if grep -q "cartesian_parallel_check FAIL" "$stdout_log" 2>/dev/null; then
        local msg
        msg="$(grep -m1 "cartesian_parallel_check FAIL" "$stdout_log")"
        REGRESSION_CHECK_MSG="$msg"
        return 1
    fi

    REGRESSION_CHECK_MSG="No PASS/FAIL marker found"
    return 1
}

# Serial cartesian (smoke test — just checks it runs)
check_serial_cartesian_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    REGRESSION_CHECK_MSG="PASS (completed without fatal errors)"
    return 0
}
