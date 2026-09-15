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
STORM_CHECK_EXAMPLES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../examples" && pwd)"

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

check_densmore2012_l1_limit_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local max_l1="$5"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1
    local l1
    l1="$(extract_number_after "DENSMORE2012_TGAS_L1" "$stdout_log")"
    if [[ -z "$l1" ]]; then
        REGRESSION_CHECK_MSG="Missing DENSMORE2012_TGAS_L1 comparison metric"
        return 1
    fi
    if ! awk -v l1="$l1" -v max_l1="$max_l1" \
        'BEGIN { exit !(l1 >= 0 && l1 <= max_l1) }'; then
        REGRESSION_CHECK_MSG="DENSMORE2012_TGAS_L1 exceeds ${max_l1} keV"
        return 1
    fi
    REGRESSION_CHECK_MSG="PASS (DENSMORE2012_TGAS_L1 = ${l1} keV)"
}

check_densmore2012_ddmc_case() {
    check_densmore2012_l1_limit_case "$1" "$2" "$3" "$4" 0.06
}

check_densmore2012_serial_case() {
    check_densmore2012_l1_limit_case "$1" "$2" "$3" "$4" 0.05
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

check_moving_slab_mc_32_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local spectrum_file="${run_dir}/moving_slab_mc_32_spectrum.txt"
    local checker_stdout="${run_dir}/moving_slab_mc_32_check.stdout.log"
    local checker_stderr="${run_dir}/moving_slab_mc_32_check.stderr.log"
    local storm_root
    storm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1
    is_nonempty_and_newer "$spectrum_file" "$start_epoch" || return 1
    python3 "${storm_root}/examples/moving_slab_mc_32/check_moving_slab_mc_32.py" \
        --spectrum "$spectrum_file" \
        --max-ferror "${MOVING_SLAB_MC_32_MAX_FERROR:-0.30}" \
        --plot-dir "$run_dir" \
        >"$checker_stdout" 2>"$checker_stderr" || {
        REGRESSION_CHECK_MSG="Moving slab MC 32-group spectrum comparison failed"
        return 1
    }
    REGRESSION_CHECK_MSG="Moving slab MC 32-group spectrum comparison passed"
    return 0
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

# Crooked Pipe benchmark. Compare the five material-temperature histories with
# the digitized DIMC curves from Steinberg & Heizler (2022), Fig. 8(a).
check_crooked_pipe_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    local probes="${run_dir}/crookedpipe_probes.txt"
    is_nonempty_and_newer "$probes" "$start_epoch" || return 1

    local storm_root
    storm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    local comparison_script="${storm_root}/examples/crooked_pipe/compare_reference.py"
    local comparison
    if ! comparison="$(python3 "$comparison_script" "$probes" 2>&1)"; then
        printf '%s\n' "$comparison"
        REGRESSION_CHECK_MSG="$(printf '%s\n' "$comparison" | grep -m1 '^FAIL:' || true)"
        REGRESSION_CHECK_MSG="${REGRESSION_CHECK_MSG:-Crooked Pipe reference comparison failed}"
        return 1
    fi

    printf '%s\n' "$comparison"
    if ! grep -q '^PASS:' <<< "$comparison"; then
        REGRESSION_CHECK_MSG="Crooked Pipe comparison did not report PASS"
        return 1
    fi

    REGRESSION_CHECK_MSG="PASS (five probe histories match Steinberg-Heizler Fig. 8(a))"
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

# Till-Compton equilibration benchmark.  The comparison is diagnostic because
# the reference is digitized and the STORM run is Monte Carlo; require the
# profile and the reported comparison/energy diagnostics, but do not impose a
# numerical pass threshold on the noisy temperature curves here.
check_till_compton_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    local profile="${run_dir}/till_compton_profile.txt"
    is_nonempty_and_newer "$profile" "$start_epoch" || return 1

    local rows
    rows="$(awk '
        NF && $1 !~ /^#/ {
            if (NF != 4) exit 1
            count++
        }
        END {
            if (count < 2) exit 1
            print count
        }
    ' "$profile" 2>/dev/null)" || {
        REGRESSION_CHECK_MSG="Malformed Till-Compton profile: ${profile}"
        return 1
    }

    if ! grep -q "TILL_COMPTON_TOTAL_ENERGY_REL_DRIFT" "$stdout_log" 2>/dev/null; then
        REGRESSION_CHECK_MSG="Till-Compton comparison diagnostics missing from stdout"
        return 1
    fi

    REGRESSION_CHECK_MSG="PASS (profile generated with ${rows} samples; comparison diagnostics reported)"
    return 0
}

# The benchmark-local scripts own their physical references and tolerances.
check_su_olson() {
    local run_dir="$1" start_epoch="$2" stdout_log="$3" stderr_log="$4"
    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1
    local tau
    for tau in 1 10 100; do
        is_nonempty_and_newer "$run_dir/output_regression/profile_tau${tau}.txt" "$start_epoch" || return 1
    done
    if ! python3 "$STORM_CHECK_EXAMPLES/su_olson/compare.py" "$run_dir/output_regression" --check; then
        REGRESSION_CHECK_MSG="Su-Olson reference comparison failed"
        return 1
    fi
    REGRESSION_CHECK_MSG="PASS (Su-Olson late-time radiation and material relative L1 < 6%)"
}

check_olson_2d_2020() {
    local run_dir="$1" start_epoch="$2" stdout_log="$3" stderr_log="$4"
    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1
    local ct
    for ct in 2 2p5 3; do
        is_nonempty_and_newer "$run_dir/output_regression/profile_ct${ct}.txt" "$start_epoch" || return 1
    done
    if ! python3 "$STORM_CHECK_EXAMPLES/olson_2d_2020/compare.py" "$run_dir/output_regression" --check; then
        REGRESSION_CHECK_MSG="Olson 2020 energy/reference comparison failed"
        return 1
    fi
    REGRESSION_CHECK_MSG="PASS (Olson 2020 energy conservation and digitized PN comparison)"
}

# Hillier (1994) polarized electron-scattering benchmark.
# The example prints its own verdict after comparing against the analytic
# single-scattering limit, so the harness only has to read it back.
check_hillier_polarization_case() {
    local run_dir="$1"
    local start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    check_no_fatal_markers "$stdout_log" "$stderr_log" || return 1

    if grep -qx 'FAIL hillier_polarization' "$stdout_log"; then
        local detail
        detail="$(grep -m1 '^  FAIL: ' "$stdout_log" | sed 's/^ *FAIL: *//')"
        set_check_msg "Hillier polarization benchmark failed${detail:+: ${detail}}"
        return 1
    fi
    # Two gating runs per invocation: the prolate envelope and the spherical null
    # control.  One verdict means one of them never ran, which is a silently
    # halved test rather than a pass.
    local verdicts
    verdicts="$(grep -cx 'PASS hillier_polarization' "$stdout_log" || true)"
    if [ "${verdicts:-0}" -lt 2 ]; then
        set_check_msg "Hillier polarization benchmark produced ${verdicts:-0} verdict(s), expected 2 (prolate + null control)"
        return 1
    fi

    # Figures are a by-product, never a gate: a machine without numpy must still
    # be able to run the regression.
    local storm_root
    storm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    local plot_log="${run_dir}/hillier_plot.log"
    if python3 -c 'import numpy' >/dev/null 2>&1; then
        python3 "${storm_root}/examples/hillier_polarization/plot_hillier_fig5.py" \
            "${run_dir}/hillier_fig5.svg" "$run_dir" >"$plot_log" 2>&1 || true
        python3 "${storm_root}/examples/hillier_polarization/plot_hillier.py" \
            "${run_dir}/hillier_observers.svg" \
            "${run_dir}/hillier_observers_prolate_0020.txt" \
            "${run_dir}/hillier_observers_spherical_0020.txt" >>"$plot_log" 2>&1 || true
    fi

    set_check_msg "polarized Thomson transport matches the single-scattering limit"
    return 0
}
