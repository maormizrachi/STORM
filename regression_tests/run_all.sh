#!/usr/bin/env bash
#
# STORM Regression Test Suite
#
# Builds STORM, runs each benchmark, and checks pass/fail.
# Tests are auto-discovered from examples/*/REGRESSION_INFO files.
#
# Usage:
#   ./regression_tests/run_all.sh [options]
#
# Options are documented in usage() below.

set -u

# ==================== Colors ====================
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
ORANGE=$'\033[0;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKS_SCRIPT="${ROOT_DIR}/regression_tests/lib/regression_checks.sh"
EXAMPLES_DIR="${ROOT_DIR}/examples"

if [[ ! -f "${CHECKS_SCRIPT}" ]]; then
    echo "Missing checks script: ${CHECKS_SCRIPT}" >&2
    exit 2
fi
source "${CHECKS_SCRIPT}"

# ==================== Defaults ====================
WITH_MPI=0
NO_IBV=0
MPI_NP_OVERRIDE=""
KEEP_ARTIFACTS=0
VERBOSE=0
TEST_FILTER=""
CLEAN_RESULTS=0
RECHECK_MODE=0
MODE="serial"
SEQUENTIAL=0
BUILD_TYPE="Release"
NPROC=""
PARTITION=""
NOHUP=0

ARTIFACT_ROOT="${ROOT_DIR}/regression_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"

# ==================== Discover valid test IDs ====================
discover_test_ids() {
    local ids=""
    for info_file in "${EXAMPLES_DIR}"/*/REGRESSION_INFO; do
        [[ -f "$info_file" ]] || continue
        local dir_name
        dir_name="$(basename "$(dirname "$info_file")")"
        ids+="${dir_name}|"
    done
    echo "${ids%|}"
}
VALID_TEST_IDS="$(discover_test_ids)"

# ==================== Usage ====================
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

STORM Regression Test Suite

Tests are auto-discovered from examples/*/REGRESSION_INFO files.
To add a new regression test, create a REGRESSION_INFO file in the
example directory with test configuration (TAGS, CHECK_FUNCTION, etc.).

Options:
  --with-mpi               Include MPI tests and compile with MPI support
  --no-ibv                 Disable IBV auto-detection (libibverbs)
  --mode <mode>            Run mode: serial, mpi, all (default: serial; --with-mpi sets to all)
  --mpi-np <N>             Override default MPI ranks for parallel tests
  --test <id>              Run only one test (${VALID_TEST_IDS//|/, })
  --build-type <type>      CMake build type: Release, Debug, RelWithDebInfo (default: Release)
  --nproc <N>              Override make -j parallelism (default: \$(nproc))
  --partition <name>       SLURM partition for MPI tests (default: system default)
  --sequential             Run tests one at a time (default: parallel where possible)
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal
  --recheck                Re-run the CHECK step only for --test (no build/run)
  --clean-results          Delete regression_results and exit
  --nohup                  Do not cancel SLURM jobs on Ctrl+C / signals
  -h, --help               Show this help

Modes:
  serial           Run tests tagged "serial" only  (no MPI build needed)
  mpi              Run tests tagged "mpi" only     (requires --with-mpi)
  all              Run all tests                   (requires --with-mpi for MPI tests)

REGRESSION_INFO format (shell-sourceable key=value):
  Required:  TAGS            "serial" or "mpi"
             CHECK_FUNCTION  check function from regression_checks.sh
  Optional:  BUILD_TARGET    CMake target (default: directory name)
             RUN_COMMAND     run command (default: binary for serial, mpirun for mpi)
             TIMEOUT         max seconds (default: 3600)
             MPI_NP          MPI ranks (default: 4)
             SBATCH_ARGS     sbatch flags (default: -n 1 for serial, -n \${MPI_NP} for mpi)
             MEM_PER_CPU     memory per CPU for auto ntasks-per-node

Examples:
  ./regression_tests/run_all.sh                          # serial tests only
  ./regression_tests/run_all.sh --with-mpi               # all tests (serial + MPI)
  ./regression_tests/run_all.sh --with-mpi --no-ibv      # all tests, force no IBV
  ./regression_tests/run_all.sh --test marshak_wave_1    # single test
  ./regression_tests/run_all.sh --recheck --test densmore2012
  ./regression_tests/run_all.sh --clean-results

Results are saved to: regression_results/<YYYYMMDD_HHMMSS>/
A summary.txt file with pass/fail status is written to the results directory.
EOF
}

# ==================== Parse arguments ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-mpi)      WITH_MPI=1; shift ;;
        --no-ibv)        NO_IBV=1; shift ;;
        --mode)          MODE="${2:-}"; shift 2 ;;
        --mpi-np)        MPI_NP_OVERRIDE="${2:-}"; shift 2 ;;
        --test)          TEST_FILTER="${2:-}"; shift 2 ;;
        --build-type)    BUILD_TYPE="${2:-}"; shift 2 ;;
        --nproc)         NPROC="${2:-}"; shift 2 ;;
        --partition)     PARTITION="${2:-}"; shift 2 ;;
        --sequential)    SEQUENTIAL=1; shift ;;
        --keep-artifacts) KEEP_ARTIFACTS=1; shift ;;
        --verbose)       VERBOSE=1; shift ;;
        --recheck)       RECHECK_MODE=1; shift ;;
        --clean-results) CLEAN_RESULTS=1; shift ;;
        --nohup)         NOHUP=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# ==================== Auto-derive mode ====================
if [[ "${WITH_MPI}" -eq 1 && "${MODE}" == "serial" ]]; then
    MODE="all"
fi

# ==================== Clean ====================
if [[ "${CLEAN_RESULTS}" -eq 1 ]]; then
    if [[ -d "${ARTIFACT_ROOT}" ]]; then
        rm -rf "${ARTIFACT_ROOT}"
        echo "Removed ${ARTIFACT_ROOT}"
    else
        echo "No results directory to clean (${ARTIFACT_ROOT})"
    fi
    exit 0
fi

# ==================== Validate ====================
case "${MODE}" in
    serial|mpi|all) ;;
    *)
        echo "--mode must be one of: serial, mpi, all" >&2
        exit 2
        ;;
esac

if [[ "${MODE}" == "mpi" || "${MODE}" == "all" ]] && [[ "${WITH_MPI}" -eq 0 ]]; then
    echo "${RED}Error: --mode ${MODE} requires --with-mpi${NC}" >&2
    exit 2
fi

if [[ -n "${TEST_FILTER}" ]]; then
    if ! echo "${TEST_FILTER}" | grep -qE "^(${VALID_TEST_IDS})$"; then
        echo "--test must be one of: ${VALID_TEST_IDS//|/, }" >&2
        exit 2
    fi
fi

if [[ -z "${NPROC}" ]]; then
    NPROC="$(nproc 2>/dev/null || echo 4)"
fi

PARTITION_FLAG=""
if [[ -n "${PARTITION}" ]]; then
    PARTITION_FLAG="--partition=${PARTITION}"
fi

# ==================== Signal handling ====================
cleanup_on_signal() {
    trap - INT TERM HUP

    echo ""
    echo "${RED}${BOLD}Signal received — cancelling all SLURM jobs...${NC}"

    # Cancel sbatch jobs found in artifact directory
    if [[ -d "${RUN_ARTIFACT_DIR}" ]]; then
        for f in "${RUN_ARTIFACT_DIR}"/*/slurm_job_id.txt; do
            [[ -f "$f" ]] || continue
            local jid
            jid="$(< "$f")"
            if [[ -n "$jid" ]]; then
                scancel "$jid" 2>/dev/null && echo "  Cancelled SLURM job $jid"
            fi
        done
    fi

    # Kill background processes without waiting
    local bg_pids
    bg_pids="$(jobs -p 2>/dev/null)" || true
    if [[ -n "$bg_pids" ]]; then
        kill $bg_pids 2>/dev/null
        sleep 0.5
        kill -9 $bg_pids 2>/dev/null
    fi

    exit 130
}

if [[ "${NOHUP}" -eq 0 ]]; then
    trap cleanup_on_signal INT TERM HUP
fi

# ==================== Recheck mode ====================
if [[ "${RECHECK_MODE}" -eq 1 ]]; then
    if [[ -z "${TEST_FILTER}" ]]; then
        echo "--recheck requires --test <id>" >&2
        exit 2
    fi

    info_file="${EXAMPLES_DIR}/${TEST_FILTER}/REGRESSION_INFO"
    if [[ ! -f "$info_file" ]]; then
        echo "Missing REGRESSION_INFO: ${info_file}" >&2
        exit 2
    fi

    latest_dir=""
    if [[ -d "${ARTIFACT_ROOT}" ]]; then
        for ts_dir in $(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort -r); do
            case_dir="${ts_dir}/${TEST_FILTER}"
            if [[ -d "$case_dir" && -f "$case_dir/run.stdout.log" ]]; then
                latest_dir="$case_dir"
                break
            fi
        done
    fi

    if [[ -z "$latest_dir" ]]; then
        echo "No regression artifact for test '${TEST_FILTER}' under ${ARTIFACT_ROOT}." >&2
        exit 1
    fi

    CHECK_FUNCTION=""
    source "$info_file"

    run_dir_abs="${EXAMPLES_DIR}/${TEST_FILTER}"

    echo "${BOLD}Recheck analysis for '${TEST_FILTER}'${NC}"
    echo "  Artifact: ${latest_dir}"

    if "${CHECK_FUNCTION}" "${run_dir_abs}" "0" "${latest_dir}/run.stdout.log" "${latest_dir}/run.stderr.log"; then
        echo "${GREEN}CHECK PASS: ${REGRESSION_CHECK_MSG}${NC}"
        exit 0
    fi
    echo "${RED}CHECK FAIL: ${REGRESSION_CHECK_MSG}${NC}"
    exit 1
fi

# ==================== Helper functions ====================
test_matches_mode() {
    local tags="$1"
    case "${MODE}" in
        all) return 0 ;;
        serial)
            [[ " ${tags} " == *" serial "* ]] && return 0
            return 1
            ;;
        mpi)
            [[ " ${tags} " == *" mpi "* ]] && return 0
            return 1
            ;;
    esac
    return 1
}

print_status() {
    local phase="$1"
    local test_id="$2"
    local msg="$3"
    local color="${4:-${NC}}"
    printf "${color}[%-7s] %-28s %s${NC}\n" "${phase}" "${test_id}" "${msg}"
}

write_check_result() {
    local case_dir="$1"
    local name="$2"
    local status="$3"
    local detail="$4"
    mkdir -p "${case_dir}"
    printf '%s\t%s\t%s\n' "${name}" "${status}" "${detail}" > "${case_dir}/check_result.txt"
    update_summary_file
}

format_duration() {
    local secs="$1"
    if [[ $secs -ge 3600 ]]; then
        printf '%dh %dm %ds' $((secs / 3600)) $((secs % 3600 / 60)) $((secs % 60))
    elif [[ $secs -ge 60 ]]; then
        printf '%dm %ds' $((secs / 60)) $((secs % 60))
    else
        printf '%ds' "$secs"
    fi
}

update_summary_file() {
    [[ -d "${RUN_ARTIFACT_DIR}" ]] || return 0
    [[ ${#TEST_IDS[@]} -gt 0 ]] || return 0

    (
        flock -w 10 200 || return

        local s_total=0 s_passed=0 s_failed=0 s_pending=0
        local s_lines=""

        for tid in "${TEST_IDS[@]}"; do
            local case_dir="${RUN_ARTIFACT_DIR}/${tid}"
            local result_file="${case_dir}/check_result.txt"

            s_total=$((s_total + 1))

            if [[ -f "$result_file" ]]; then
                local r_name r_status r_detail
                IFS=$'\t' read -r r_name r_status r_detail < "$result_file"

                local duration_str="--"
                if [[ -f "${case_dir}/run_start_epoch.txt" && -f "${case_dir}/run_end_epoch.txt" ]]; then
                    local t_start t_end elapsed
                    t_start="$(< "${case_dir}/run_start_epoch.txt")"
                    t_end="$(< "${case_dir}/run_end_epoch.txt")"
                    elapsed=$((t_end - t_start))
                    duration_str="$(format_duration $elapsed)"
                fi

                if [[ "$r_status" == "PASS" ]]; then
                    s_passed=$((s_passed + 1))
                else
                    s_failed=$((s_failed + 1))
                fi
                s_lines+="$(printf '%-8s %-28s %-40s [%s]' "${r_status}" "${r_name}" "${r_detail}" "${duration_str}")"$'\n'
            else
                s_pending=$((s_pending + 1))
                s_lines+="$(printf '%-8s %-28s' "PENDING" "${tid}")"$'\n'
            fi
        done

        {
            echo "STORM Regression Test Results"
            echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Mode: ${MODE}"
            echo ""
            printf '%s' "${s_lines}"
            echo ""
            echo "Total: ${s_total}   Passed: ${s_passed}   Failed: ${s_failed}   Pending: ${s_pending}"
        } > "${RUN_ARTIFACT_DIR}/summary.txt.tmp"
        mv "${RUN_ARTIFACT_DIR}/summary.txt.tmp" "${RUN_ARTIFACT_DIR}/summary.txt"

    ) 200>"${RUN_ARTIFACT_DIR}/.summary.lock"
}

# ==================== Memory helper ====================
parse_mem_mb() {
    local val="$1"
    local num="${val%[GgMm]}"
    local suffix="${val: -1}"
    case "${suffix}" in
        G|g) echo $(( num * 1024 )) ;;
        M|m) echo "${num}" ;;
        *)   echo "${val}" ;;
    esac
}

compute_tasks_per_node() {
    local mem_per_cpu="$1"
    local partition_flag="$2"
    local mem_per_cpu_mb
    mem_per_cpu_mb="$(parse_mem_mb "${mem_per_cpu}")"

    local node_mem_mb
    node_mem_mb="$(sinfo ${partition_flag} -h -o "%m" | sort -n | head -1)"
    if [[ -z "${node_mem_mb}" || "${node_mem_mb}" -eq 0 ]]; then
        echo ""
        return 1
    fi

    echo $(( node_mem_mb / mem_per_cpu_mb ))
}

# ==================== Build STORM ====================

BUILD_DIR="${ROOT_DIR}/build_regression"

build_storm() {
    echo "${BOLD}=== Building STORM ===${NC}"
    echo "  Build type:  ${BUILD_TYPE}"
    echo "  With MPI:    ${WITH_MPI}"
    echo "  No IBV:      ${NO_IBV}"
    echo "  Build dir:   ${BUILD_DIR}"
    echo "  Parallelism: -j${NPROC}"

    local -A unique_targets=()
    for t in "$@"; do
        unique_targets["${t}"]=1
    done
    local targets=("${!unique_targets[@]}")

    if [[ ${#targets[@]} -gt 0 ]]; then
        echo "  Targets:     ${targets[*]}"
    fi
    echo

    local cmake_args=(
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
        -DSTORM_BUILD_EXAMPLES=ON
    )

    if [[ "${WITH_MPI}" -eq 1 ]]; then
        cmake_args+=(-DSTORM_WITH_MPI=ON)
    else
        cmake_args+=(-DSTORM_WITH_MPI=OFF)
    fi

    if [[ "${NO_IBV}" -eq 1 ]]; then
        cmake_args+=(-DSTORM_NO_IBV=ON)
    fi

    mkdir -p "${BUILD_DIR}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    local build_log="${RUN_ARTIFACT_DIR}/build.log"

    # --- Configure ---
    printf "  Configuring..."
    {
        echo "=== CMake configure ==="
        cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${cmake_args[@]}" 2>&1
    } > "${build_log}" 2>&1
    local cmake_rc=$?

    if [[ $cmake_rc -ne 0 ]]; then
        printf " ${RED}FAILED${NC}\n"
        print_status "BUILD" "STORM" "Configure failed (see ${build_log})" "${RED}"
        echo ""
        echo "${RED}Configure failed. Last 30 lines:${NC}"
        tail -30 "${build_log}"
        return 1
    fi
    printf " ${GREEN}OK${NC}\n"

    # --- Build with progress ---
    echo "" >> "${build_log}"
    echo "=== CMake build ===" >> "${build_log}"

    local build_cmd=(cmake --build "${BUILD_DIR}" -j "${NPROC}")
    for t in "${targets[@]}"; do
        build_cmd+=(--target "${t}")
    done

    "${build_cmd[@]}" >> "${build_log}" 2>&1 &
    local build_pid=$!

    printf "  Building...  [  0%%]"
    local prev_line=""
    while kill -0 $build_pid 2>/dev/null; do
        local last_progress
        last_progress="$(grep -E '^\[[ ]*[0-9]+%\]' "${build_log}" 2>/dev/null | tail -1)" || true
        if [[ -n "$last_progress" && "$last_progress" != "$prev_line" ]]; then
            prev_line="$last_progress"
            local pct desc
            pct="$(echo "$last_progress" | sed 's/.*\[ *\([0-9]*\)%.*/\1/')"
            desc="${last_progress#*] }"
            desc="${desc##*/}"
            printf '\r  Building...  [%3d%%] %-55s' "$pct" "${desc:0:55}"
        fi
        sleep 1
    done

    wait $build_pid
    local build_rc=$?

    if [[ $build_rc -ne 0 ]]; then
        printf '\r  Building...  %s%s\n' "${RED}FAILED${NC}" "$(printf '%55s' '')"
        print_status "BUILD" "STORM" "FAILED (see ${build_log})" "${RED}"
        echo ""
        echo "${RED}Build failed. Last 30 lines:${NC}"
        tail -30 "${build_log}"
        return 1
    fi

    printf '\r  Building...  [100%%] %-55s\n' "${GREEN}done${NC}"
    print_status "BUILD" "STORM" "OK" "${GREEN}"
    return 0
}

# ==================== Collect and filter tests ====================
declare -a TEST_IDS=()
declare -a TEST_TAGS=()
declare -a TEST_BUILD_TARGETS=()
declare -a TEST_RUN_DIR_RELS=()
declare -a TEST_RUN_COMMANDS=()
declare -a TEST_CHECK_FUNCTIONS=()
declare -a TEST_TIMEOUTS=()
declare -a TEST_MPI_NPS=()
declare -a TEST_SBATCH_ARGS=()
declare -a TEST_MEM_PER_CPUS=()

load_tests() {
    for info_file in "${EXAMPLES_DIR}"/*/REGRESSION_INFO; do
        [[ -f "$info_file" ]] || continue

        local dir_name
        dir_name="$(basename "$(dirname "$info_file")")"

        # Reset to defaults before sourcing
        TAGS=""
        BUILD_TARGET="${dir_name}"
        RUN_COMMAND=""
        CHECK_FUNCTION=""
        TIMEOUT="3600"
        MPI_NP="4"
        SBATCH_ARGS=""
        MEM_PER_CPU=""

        source "${info_file}"

        local test_id="${dir_name}"

        if [[ -z "${TAGS}" || -z "${CHECK_FUNCTION}" ]]; then
            echo "${ORANGE}Skipping malformed REGRESSION_INFO: ${info_file} (missing TAGS or CHECK_FUNCTION)${NC}" >&2
            continue
        fi

        # Apply defaults based on test type
        if [[ " ${TAGS} " == *" serial "* ]]; then
            [[ -z "${SBATCH_ARGS}" ]] && SBATCH_ARGS="-n 1"
            [[ -z "${RUN_COMMAND}" ]] && RUN_COMMAND='"${STORM_BIN}"'
        elif [[ " ${TAGS} " == *" mpi "* ]]; then
            [[ -z "${SBATCH_ARGS}" ]] && SBATCH_ARGS='-n ${MPI_NP}'
            [[ -z "${RUN_COMMAND}" ]] && RUN_COMMAND='mpirun "${STORM_BIN}"'
        fi

        # Apply filters
        if [[ -n "${TEST_FILTER}" && "${test_id}" != "${TEST_FILTER}" ]]; then
            continue
        fi

        if ! test_matches_mode "${TAGS}"; then
            continue
        fi

        if [[ " ${TAGS} " == *" mpi "* ]] && [[ "${WITH_MPI}" -eq 0 ]]; then
            continue
        fi

        TEST_IDS+=("${test_id}")
        TEST_TAGS+=("${TAGS}")
        TEST_BUILD_TARGETS+=("${BUILD_TARGET}")
        TEST_RUN_DIR_RELS+=("examples/${dir_name}")
        TEST_RUN_COMMANDS+=("${RUN_COMMAND}")
        TEST_CHECK_FUNCTIONS+=("${CHECK_FUNCTION}")
        TEST_TIMEOUTS+=("${TIMEOUT}")
        TEST_MPI_NPS+=("${MPI_NP}")
        TEST_SBATCH_ARGS+=("${SBATCH_ARGS}")
        TEST_MEM_PER_CPUS+=("${MEM_PER_CPU}")
    done
}

# ==================== Run a single test ====================
run_single_test() {
    local idx="$1"
    local test_id="${TEST_IDS[$idx]}"
    local build_target="${TEST_BUILD_TARGETS[$idx]}"
    local run_dir_rel="${TEST_RUN_DIR_RELS[$idx]}"
    local run_cmd_template="${TEST_RUN_COMMANDS[$idx]}"
    local check_fn="${TEST_CHECK_FUNCTIONS[$idx]}"
    local timeout_sec="${TEST_TIMEOUTS[$idx]}"
    local mpi_np="${TEST_MPI_NPS[$idx]}"
    local sbatch_args_template="${TEST_SBATCH_ARGS[$idx]}"
    local mem_per_cpu="${TEST_MEM_PER_CPUS[$idx]}"

    if [[ -n "${MPI_NP_OVERRIDE}" ]]; then
        mpi_np="${MPI_NP_OVERRIDE}"
    fi

    local case_artifact_dir="${RUN_ARTIFACT_DIR}/${test_id}"
    mkdir -p "${case_artifact_dir}"

    local run_dir_abs="${ROOT_DIR}/${run_dir_rel}"

    # Locate binary: check the example's own dir first (set_target_properties),
    # then the build dir.
    local STORM_BIN=""
    if [[ -x "${run_dir_abs}/${build_target}" ]]; then
        STORM_BIN="${run_dir_abs}/${build_target}"
    elif [[ -x "${BUILD_DIR}/examples/${build_target}" ]]; then
        STORM_BIN="${BUILD_DIR}/examples/${build_target}"
    elif [[ -x "${BUILD_DIR}/examples/${run_dir_rel#examples/}/${build_target}" ]]; then
        STORM_BIN="${BUILD_DIR}/examples/${run_dir_rel#examples/}/${build_target}"
    else
        print_status "RUN" "${test_id}" "Binary not found: ${build_target}" "${RED}"
        write_check_result "${case_artifact_dir}" "${test_id}" "FAIL" "Binary not found"
        return 1
    fi

    local stdout_log="${case_artifact_dir}/run.stdout.log"
    local stderr_log="${case_artifact_dir}/run.stderr.log"

    local MPI_NP="${mpi_np}"
    local run_cmd
    run_cmd="$(eval "cat <<STORM_EVAL_EOF
${run_cmd_template}
STORM_EVAL_EOF")"

    print_status "RUN" "${test_id}" "timeout=${timeout_sec}s binary=${STORM_BIN}"

    date +%s > "${case_artifact_dir}/run_start_epoch.txt"

    local TASKS_PER_NODE=""
    if [[ -n "${mem_per_cpu}" ]]; then
        TASKS_PER_NODE="$(compute_tasks_per_node "${mem_per_cpu}" "${PARTITION_FLAG}")"
        if [[ -z "${TASKS_PER_NODE}" || "${TASKS_PER_NODE}" -le 0 ]]; then
            print_status "RUN" "${test_id}" "Failed to compute tasks-per-node from MEM_PER_CPU=${mem_per_cpu}" "${RED}"
            write_check_result "${case_artifact_dir}" "${test_id}" "FAIL" "Cannot determine tasks-per-node"
            return 1
        fi
        print_status "RUN" "${test_id}" "MEM_PER_CPU=${mem_per_cpu} -> ${TASKS_PER_NODE} tasks/node" "${CYAN}"
    fi

    local run_rc=0
    if [[ -n "${sbatch_args_template}" ]]; then
        # Batch submission: SLURM manages output and time limit
        local sbatch_args
        sbatch_args="$(eval "cat <<STORM_EVAL_EOF
${sbatch_args_template}
STORM_EVAL_EOF")"
        if [[ -n "${TASKS_PER_NODE}" ]]; then
            sbatch_args+=" --ntasks-per-node=${TASKS_PER_NODE} --exclusive"
        fi
        local timeout_min=$(( (timeout_sec + 59) / 60 ))

        if [[ "${SEQUENTIAL}" -eq 1 ]]; then
            sbatch --wait ${PARTITION_FLAG} ${sbatch_args} \
                --output="${stdout_log}" --error="${stderr_log}" \
                --chdir="${run_dir_abs}" --time="${timeout_min}" \
                --job-name="storm_${test_id}" \
                --wrap="${run_cmd}" || run_rc=$?
        else
            local job_id
            job_id=$(sbatch --parsable ${PARTITION_FLAG} ${sbatch_args} \
                --output="${stdout_log}" --error="${stderr_log}" \
                --chdir="${run_dir_abs}" --time="${timeout_min}" \
                --job-name="storm_${test_id}" \
                --wrap="${run_cmd}" 2>/dev/null) || true

            if [[ -z "${job_id}" ]]; then
                print_status "SUBMIT" "${test_id}" "sbatch failed" "${RED}"
                run_rc=1
            else
                echo "${job_id}" > "${case_artifact_dir}/slurm_job_id.txt"
                print_status "SUBMIT" "${test_id}" "job ${job_id}" "${CYAN}"

                while squeue -j "${job_id}" -h -o "%T" 2>/dev/null | grep -qE .; do
                    sleep 10
                done

                local sacct_rc=""
                for _retry in 1 2 3 4 5; do
                    sacct_rc=$(sacct -j "${job_id}" -n -o ExitCode -X 2>/dev/null | tr -d ' ' | cut -d: -f1)
                    [[ -n "${sacct_rc}" ]] && break
                    sleep 2
                done
                run_rc="${sacct_rc:-1}"
            fi
        fi
    else
        # Direct execution (no SLURM — fallback if SBATCH_ARGS is empty)
        if [[ "${VERBOSE}" -eq 1 ]]; then
            (cd "${run_dir_abs}" && timeout "${timeout_sec}" bash -c "${run_cmd}" 2>&1 | tee "${stdout_log}") 2>"${stderr_log}" || run_rc=$?
        else
            (cd "${run_dir_abs}" && timeout "${timeout_sec}" bash -c "${run_cmd}" > "${stdout_log}" 2>"${stderr_log}") || run_rc=$?
        fi
    fi

    date +%s > "${case_artifact_dir}/run_end_epoch.txt"
    echo "${run_rc}" > "${case_artifact_dir}/run_exit_code.txt"

    if [[ ${run_rc} -eq 124 ]]; then
        print_status "RUN" "${test_id}" "TIMEOUT after ${timeout_sec}s" "${RED}"
        write_check_result "${case_artifact_dir}" "${test_id}" "FAIL" "Timeout after ${timeout_sec}s"
        return 1
    fi

    if [[ ${run_rc} -ne 0 ]]; then
        print_status "RUN" "${test_id}" "Exit code ${run_rc}" "${RED}"
        write_check_result "${case_artifact_dir}" "${test_id}" "FAIL" "Run exit code ${run_rc}"
        return 1
    fi

    # Check phase
    local suite_start
    suite_start="$(< "${case_artifact_dir}/run_start_epoch.txt")"

    if "${check_fn}" "${run_dir_abs}" "${suite_start}" "${stdout_log}" "${stderr_log}"; then
        print_status "CHECK" "${test_id}" "${REGRESSION_CHECK_MSG}" "${GREEN}"
        write_check_result "${case_artifact_dir}" "${test_id}" "PASS" "${REGRESSION_CHECK_MSG}"
        return 0
    fi

    print_status "CHECK" "${test_id}" "${REGRESSION_CHECK_MSG}" "${RED}"
    write_check_result "${case_artifact_dir}" "${test_id}" "FAIL" "${REGRESSION_CHECK_MSG}"
    return 1
}

# ==================== Main ====================

load_tests

if [[ ${#TEST_IDS[@]} -eq 0 ]]; then
    echo "${ORANGE}No tests matched the current mode/filter.${NC}"
    if [[ "${MODE}" != "all" ]]; then
        echo "  Hint: use --with-mpi to include MPI tests, or --mode all"
    fi
    exit 0
fi

echo "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
echo "${BOLD}║           STORM Regression Test Suite                    ║${NC}"
echo "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Mode:         ${MODE}"
echo "  MPI:          $(if [[ ${WITH_MPI} -eq 1 ]]; then echo "yes"; else echo "no"; fi)"
echo "  IBV:          $(if [[ ${NO_IBV} -eq 1 ]]; then echo "disabled"; else echo "auto-detect"; fi)"
echo "  Partition:    $(if [[ -n "${PARTITION}" ]]; then echo "${PARTITION}"; else echo "(default)"; fi)"
echo "  Build type:   ${BUILD_TYPE}"
echo "  Tests:        ${#TEST_IDS[@]}"
echo "  Artifacts:    ${RUN_ARTIFACT_DIR}"
echo "  Signal:       $(if [[ ${NOHUP} -eq 1 ]]; then echo "nohup (jobs survive Ctrl+C)"; else echo "cancel jobs on Ctrl+C"; fi)"
echo ""

# Phase 1: Build
if ! build_storm "${TEST_BUILD_TARGETS[@]}"; then
    echo ""
    echo "${RED}${BOLD}BUILD FAILED — cannot run tests.${NC}"

    mkdir -p "${RUN_ARTIFACT_DIR}"

    {
        echo "STORM Regression Test Results"
        echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Mode: ${MODE}"
        echo ""
        printf '%-6s  %-28s %s\n' "FAIL" "BUILD" "CMake build failed"
        echo ""
        echo "Total: 1   Passed: 0   Failed: 1"
    } > "${RUN_ARTIFACT_DIR}/summary.txt"

    echo ""
    echo "${BOLD}════════════════════════ Summary ════════════════════════${NC}"
    printf "  ${RED}%-28s %s${NC}\n" "BUILD" "FAIL"
    echo ""
    echo "Summary saved to: ${RUN_ARTIFACT_DIR}/summary.txt"
    exit 1
fi
echo ""

# Phase 2: Run tests
update_summary_file

echo "${BOLD}=== Running ${#TEST_IDS[@]} test(s) ===${NC}"
echo ""

declare -a RUN_PIDS=()
declare -a RUN_INDICES=()

for i in "${!TEST_IDS[@]}"; do
    if [[ "${SEQUENTIAL}" -eq 1 ]]; then
        run_single_test "$i"
    else
        run_single_test "$i" &
        RUN_PIDS+=($!)
        RUN_INDICES+=("$i")
    fi
done

if [[ "${SEQUENTIAL}" -eq 0 && ${#RUN_PIDS[@]} -gt 0 ]]; then
    for pid_idx in "${!RUN_PIDS[@]}"; do
        wait "${RUN_PIDS[$pid_idx]}" 2>/dev/null || true
    done
fi

# Mark incomplete tests (killed or crashed without writing a result)
for i in "${!TEST_IDS[@]}"; do
    local tid="${TEST_IDS[$i]}"
    local result_file="${RUN_ARTIFACT_DIR}/${tid}/check_result.txt"
    if [[ ! -f "$result_file" ]]; then
        write_check_result "${RUN_ARTIFACT_DIR}/${tid}" "${tid}" "FAIL" "Did not complete"
    fi
done

# Phase 4: Collect results and write summary
echo ""
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo "${BOLD}                     Summary                              ${NC}"
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

total=0
passed=0
failed=0

for result_file in "${RUN_ARTIFACT_DIR}"/*/check_result.txt; do
    [[ -f "$result_file" ]] || continue
    local_name=""
    local_status=""
    local_detail=""
    IFS=$'\t' read -r local_name local_status local_detail < "$result_file"

    local case_dir
    case_dir="$(dirname "$result_file")"
    local duration_str="--"
    if [[ -f "${case_dir}/run_start_epoch.txt" && -f "${case_dir}/run_end_epoch.txt" ]]; then
        local t_start t_end elapsed
        t_start="$(< "${case_dir}/run_start_epoch.txt")"
        t_end="$(< "${case_dir}/run_end_epoch.txt")"
        elapsed=$((t_end - t_start))
        duration_str="$(format_duration $elapsed)"
    fi

    total=$((total + 1))
    if [[ "${local_status}" == "PASS" ]]; then
        printf "  ${GREEN}%-6s${NC}  %-28s %-40s  %s\n" "PASS" "${local_name}" "${local_detail}" "[${duration_str}]"
        passed=$((passed + 1))
    else
        printf "  ${RED}%-6s${NC}  %-28s %-40s  %s\n" "FAIL" "${local_name}" "${local_detail}" "[${duration_str}]"
        failed=$((failed + 1))
    fi
done

echo ""
echo "${BOLD}Total: ${total}   Passed: ${passed}   Failed: ${failed}${NC}"
echo "Artifacts: ${RUN_ARTIFACT_DIR}"
echo "Summary:   ${RUN_ARTIFACT_DIR}/summary.txt"
echo ""

if [[ ${failed} -eq 0 ]]; then
    echo "${GREEN}${BOLD}All tests passed.${NC}"
    if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
        rm -rf "${RUN_ARTIFACT_DIR}"
        echo "Removed success artifacts (use --keep-artifacts to retain logs)."
    fi
    exit 0
else
    echo "${RED}${BOLD}${failed} test(s) failed.${NC}"
    echo "Logs are in: ${RUN_ARTIFACT_DIR}"
    exit 1
fi
