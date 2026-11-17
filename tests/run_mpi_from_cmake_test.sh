#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 RUN_MPI_FROM_CMAKE" >&2
  exit 2
fi

launcher_wrapper="$1"
test_directory="$(mktemp -d)"
trap 'rm -rf -- "${test_directory}"' EXIT

build_directory="${test_directory}/build"
fake_launcher="${test_directory}/fake_mpiexec"
actual_arguments="${test_directory}/actual_arguments.txt"
expected_arguments="${test_directory}/expected_arguments.txt"
run_info="${test_directory}/run_info.txt"
mkdir -p "${build_directory}"

cat > "${fake_launcher}" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${FAKE_MPI_ARGUMENTS_FILE}"
LAUNCHER
chmod +x "${fake_launcher}"

cat > "${build_directory}/CMakeCache.txt" <<CACHE
MPIEXEC_EXECUTABLE:FILEPATH=${fake_launcher}
MPIEXEC_NUMPROC_FLAG:STRING=-n
MPIEXEC_PREFLAGS:STRING=--bootstrap;slurm
MPIEXEC_POSTFLAGS:STRING=--post;post value
CACHE

FAKE_MPI_ARGUMENTS_FILE="${actual_arguments}" THUNDER_RUN_INFO="${run_info}" \
  bash "${launcher_wrapper}" "${build_directory}" 16 /bin/true "application argument"

cat > "${expected_arguments}" <<'EXPECTED'
-n
16
--bootstrap
slurm
/bin/true
--post
post value
application argument
EXPECTED

diff -u "${expected_arguments}" "${actual_arguments}"
grep -Fx "MPI launcher: ${fake_launcher}" "${run_info}" >/dev/null
grep -Fx "MPI requested ranks: 16" "${run_info}" >/dev/null

if bash "${launcher_wrapper}" "${test_directory}/missing" 16 /bin/true \
    >/dev/null 2>&1; then
  echo "Launcher wrapper accepted a missing CMake cache" >&2
  exit 1
fi
