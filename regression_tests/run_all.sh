#!/usr/bin/env bash
#
# STORM Regression Test Suite
#
# Modeled after RICH's regression_tests/run_all.sh.
# Builds STORM, runs each benchmark, and checks pass/fail.
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
TESTS_DIR="${ROOT_DIR}/regression_tests/tests"

if [[ ! -f "${CHECKS_SCRIPT}" ]]; then
    echo "Missing checks script: ${CHECKS_SCRIPT}" >&2
    exit 2
fi
if [[ ! -d "${TESTS_DIR}" ]]; then
    echo "Missing tests directory: ${TESTS_DIR}" >&2
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

ARTIFACT_ROOT="${ROOT_DIR}/regression_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"

# ==================== Result arrays ====================
declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_DETAILS=()

# ==================== Discover valid test IDs ====================
discover_test_ids() {
    local ids=""
    for f in "${TESTS_DIR}"/*.sh; do
        [[ -f "$f" ]] || continue
        ids+="$(basename "$f" .sh)|"
    done
    echo "${ids%|}"
}
VALID_TEST_IDS="$(discover_test_ids)"

# ==================== Usage ====================
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

STORM Regression Test Suite

Options:
  --with-mpi               Include MPI tests and compile with MPI support
  --no-ibv                 Disable IBV auto-detection (libibverbs)
  --mode <mode>            Run mode: serial, mpi, all (default: serial; --with-mpi sets to all)
  --mpi-np <N>             Override default MPI ranks for parallel tests
  --test <id>              Run only one test (${VALID_TEST_IDS//|/, })
  --build-type <type>      CMake build type: Release, Debug, RelWithDebInfo (default: Release)
  --nproc <N>              Override make -j parallelism (default: \$(nproc))
  --sequential             Run tests one at a time (default: parallel where possible)
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal
  --recheck                Re-run the CHECK step only for --test (no build/run)
  --clean-results          Delete regression_results and exit
  -h, --help               Show this help

Modes:
  serial           Run tests tagged "serial" only  (no MPI build needed)
  mpi              Run tests tagged "mpi" only     (requires --with-mpi)
  all              Run all tests                   (requires --with-mpi for MPI tests)

Examples:
  ./regression_tests/run_all.sh                          # serial tests only
  ./regression_tests/run_all.sh --with-mpi               # all tests (serial + MPI)
  ./regression_tests/run_all.sh --with-mpi --no-ibv      # all tests, force no IBV
  ./regression_tests/run_all.sh --test marshak_wave_1    # single test
  ./regression_tests/run_all.sh --recheck --test densmore2012
  ./regression_tests/run_all.sh --clean-results

Results are saved to: regression_results/<YYYYMMDD_HHMMSS>/
EOF
}

# ==================== Parse arguments ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-mpi)
            WITH_MPI=1
            shift
            ;;
        --no-ibv)
            NO_IBV=1
            shift
            ;;
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --mpi-np)
            MPI_NP_OVERRIDE="${2:-}"
            shift 2
            ;;
        --test)
            TEST_FILTER="${2:-}"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="${2:-}"
            shift 2
            ;;
        --nproc)
            NPROC="${2:-}"
            shift 2
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --recheck)
            RECHECK_MODE=1
            shift
            ;;
        --clean-results)
            CLEAN_RESULTS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
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

# ==================== Recheck mode ====================
if [[ "${RECHECK_MODE}" -eq 1 ]]; then
    if [[ -z "${TEST_FILTER}" ]]; then
        echo "--recheck requires --test <id>" >&2
        exit 2
    fi

    def_file="${TESTS_DIR}/${TEST_FILTER}.sh"
    if [[ ! -f "$def_file" ]]; then
        echo "Missing test definition: ${def_file}" >&2
        exit 2
    fi

    # Find newest artifact
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

    local_TEST_ID="" local_TAGS="" local_BUILD_TARGET="" local_RUN_DIR_REL=""
    local_RUN_COMMAND="" local_CHECK_FUNCTION="" local_TIMEOUT="3600"
    local_MPI_NP="4"
    source "$def_file"
    local_CHECK_FUNCTION="${CHECK_FUNCTION}"

    run_dir_abs="${ROOT_DIR}/${RUN_DIR_REL}"

    echo "${BOLD}Recheck analysis for '${TEST_FILTER}'${NC}"
    echo "  Artifact: ${latest_dir}"

    if "${local_CHECK_FUNCTION}" "${run_dir_abs}" "0" "${latest_dir}/run.stdout.log" "${latest_dir}/run.stderr.log"; then
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

record_result() {
    local name="$1"
    local status="$2"
    local detail="$3"
    RESULT_NAMES+=("$name")
    RESULT_STATUS+=("$status")
    RESULT_DETAILS+=("$detail")
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

    {
        echo "=== CMake configure ==="
        cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${cmake_args[@]}" 2>&1
        local cmake_rc=$?
        if [[ $cmake_rc -ne 0 ]]; then
            echo "CMake configure failed with exit code ${cmake_rc}"
            return $cmake_rc
        fi

        echo ""
        echo "=== CMake build ==="
        cmake --build "${BUILD_DIR}" -j "${NPROC}" 2>&1
        local build_rc=$?
        if [[ $build_rc -ne 0 ]]; then
            echo "Build failed with exit code ${build_rc}"
            return $build_rc
        fi
    } > "${build_log}" 2>&1

    local rc=$?

    if [[ $rc -ne 0 ]]; then
        print_status "BUILD" "STORM" "FAILED (see ${build_log})" "${RED}"
        echo ""
        echo "${RED}Build failed. Last 30 lines:${NC}"
        tail -30 "${build_log}"
        return 1
    fi

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

load_tests() {
    for def_file in "${TESTS_DIR}"/*.sh; do
        [[ -f "$def_file" ]] || continue

        TEST_ID=""
        TAGS=""
        BUILD_TARGET=""
        RUN_DIR_REL=""
        RUN_COMMAND=""
        CHECK_FUNCTION=""
        TIMEOUT="3600"
        MPI_NP="4"

        source "${def_file}"

        if [[ -z "${TEST_ID}" || -z "${BUILD_TARGET}" || -z "${RUN_DIR_REL}" || -z "${CHECK_FUNCTION}" ]]; then
            echo "${ORANGE}Skipping malformed test: ${def_file}${NC}" >&2
            continue
        fi

        if [[ -n "${TEST_FILTER}" && "${TEST_ID}" != "${TEST_FILTER}" ]]; then
            continue
        fi

        if ! test_matches_mode "${TAGS}"; then
            continue
        fi

        if [[ " ${TAGS} " == *" mpi "* ]] && [[ "${WITH_MPI}" -eq 0 ]]; then
            continue
        fi

        TEST_IDS+=("${TEST_ID}")
        TEST_TAGS+=("${TAGS}")
        TEST_BUILD_TARGETS+=("${BUILD_TARGET}")
        TEST_RUN_DIR_RELS+=("${RUN_DIR_REL}")
        TEST_RUN_COMMANDS+=("${RUN_COMMAND}")
        TEST_CHECK_FUNCTIONS+=("${CHECK_FUNCTION}")
        TEST_TIMEOUTS+=("${TIMEOUT}")
        TEST_MPI_NPS+=("${MPI_NP}")
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
        record_result "${test_id}" "FAIL" "Binary not found"
        return 1
    fi

    local MPI_NP="${mpi_np}"
    local run_cmd
    run_cmd="$(eval echo "${run_cmd_template}")"

    local stdout_log="${case_artifact_dir}/run.stdout.log"
    local stderr_log="${case_artifact_dir}/run.stderr.log"

    print_status "RUN" "${test_id}" "timeout=${timeout_sec}s binary=${STORM_BIN}"

    date +%s > "${case_artifact_dir}/run_start_epoch.txt"

    local run_rc=0
    if [[ "${VERBOSE}" -eq 1 ]]; then
        (cd "${run_dir_abs}" && timeout "${timeout_sec}" bash -c "${run_cmd}" 2>&1 | tee "${stdout_log}") 2>"${stderr_log}" || run_rc=$?
    else
        (cd "${run_dir_abs}" && timeout "${timeout_sec}" bash -c "${run_cmd}" > "${stdout_log}" 2>"${stderr_log}") || run_rc=$?
    fi

    date +%s > "${case_artifact_dir}/run_end_epoch.txt"
    echo "${run_rc}" > "${case_artifact_dir}/run_exit_code.txt"

    if [[ ${run_rc} -eq 124 ]]; then
        print_status "RUN" "${test_id}" "TIMEOUT after ${timeout_sec}s" "${RED}"
        record_result "${test_id}" "FAIL" "Timeout after ${timeout_sec}s"
        return 1
    fi

    if [[ ${run_rc} -ne 0 ]]; then
        print_status "RUN" "${test_id}" "Exit code ${run_rc}" "${RED}"
        record_result "${test_id}" "FAIL" "Run exit code ${run_rc}"
        return 1
    fi

    # Check phase
    local suite_start
    suite_start="$(< "${case_artifact_dir}/run_start_epoch.txt")"

    if "${check_fn}" "${run_dir_abs}" "${suite_start}" "${stdout_log}" "${stderr_log}"; then
        print_status "CHECK" "${test_id}" "${REGRESSION_CHECK_MSG}" "${GREEN}"
        record_result "${test_id}" "PASS" "${REGRESSION_CHECK_MSG}"
        return 0
    fi

    print_status "CHECK" "${test_id}" "${REGRESSION_CHECK_MSG}" "${RED}"
    record_result "${test_id}" "FAIL" "${REGRESSION_CHECK_MSG}"
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
echo "  Build type:   ${BUILD_TYPE}"
echo "  Tests:        ${#TEST_IDS[@]}"
echo "  Artifacts:    ${RUN_ARTIFACT_DIR}"
echo ""

# Phase 1: Build
if ! build_storm; then
    echo ""
    echo "${RED}${BOLD}BUILD FAILED — cannot run tests.${NC}"
    record_result "BUILD" "FAIL" "CMake build failed"

    echo ""
    echo "${BOLD}════════════════════════ Summary ════════════════════════${NC}"
    printf "  ${RED}%-28s %s${NC}\n" "BUILD" "FAIL"
    exit 1
fi
echo ""

# Phase 2: Run tests
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

# Phase 3: Summary
echo ""
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo "${BOLD}                     Summary                              ${NC}"
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

total=0
passed=0
failed=0
for i in "${!RESULT_NAMES[@]}"; do
    total=$((total + 1))
    local_name="${RESULT_NAMES[$i]}"
    local_status="${RESULT_STATUS[$i]}"
    local_detail="${RESULT_DETAILS[$i]}"
    if [[ "${local_status}" == "PASS" ]]; then
        printf "  ${GREEN}%-6s${NC}  %-28s %s\n" "PASS" "${local_name}" "${local_detail}"
        passed=$((passed + 1))
    else
        printf "  ${RED}%-6s${NC}  %-28s %s\n" "FAIL" "${local_name}" "${local_detail}"
        failed=$((failed + 1))
    fi
done

echo ""
echo "${BOLD}Total: ${total}   Passed: ${passed}   Failed: ${failed}${NC}"
echo "Artifacts: ${RUN_ARTIFACT_DIR}"
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
