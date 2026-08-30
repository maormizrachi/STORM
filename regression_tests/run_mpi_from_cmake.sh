#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 BUILD_DIRECTORY RANKS BINARY [ARGUMENT ...]" >&2
  exit 2
fi

build_directory="$1"
requested_ranks="$2"
shift 2
binary="$1"
shift

if [[ ! "${requested_ranks}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid MPI rank count: ${requested_ranks}" >&2
  exit 2
fi

cache_file="${build_directory}/CMakeCache.txt"
if [[ ! -r "${cache_file}" ]]; then
  echo "CMake cache not found: ${cache_file}" >&2
  exit 2
fi

mpi_launcher=""
rank_flag=""
preflags_value=""
postflags_value=""

while IFS='=' read -r cache_key cache_value; do
  case "${cache_key}" in
    MPIEXEC_EXECUTABLE:*) mpi_launcher="${cache_value}" ;;
    MPIEXEC_NUMPROC_FLAG:*) rank_flag="${cache_value}" ;;
    MPIEXEC_PREFLAGS:*) preflags_value="${cache_value}" ;;
    MPIEXEC_POSTFLAGS:*) postflags_value="${cache_value}" ;;
  esac
done < "${cache_file}"

if [[ -z "${mpi_launcher}" || ! -x "${mpi_launcher}" ]]; then
  echo "Configured MPI launcher is not executable: ${mpi_launcher:-unset}" >&2
  exit 2
fi

if [[ -z "${rank_flag}" ]]; then
  echo "MPIEXEC_NUMPROC_FLAG is missing from ${cache_file}" >&2
  exit 2
fi

preflags=()
postflags=()
if [[ -n "${preflags_value}" ]]; then
  IFS=';' read -r -a preflags <<< "${preflags_value}"
fi
if [[ -n "${postflags_value}" ]]; then
  IFS=';' read -r -a postflags <<< "${postflags_value}"
fi

if [[ -n "${THUNDER_RUN_INFO:-}" ]]; then
  printf 'MPI launcher: %s\nMPI requested ranks: %s\n' \
    "${mpi_launcher}" "${requested_ranks}" >> "${THUNDER_RUN_INFO}"
fi

exec "${mpi_launcher}" "${rank_flag}" "${requested_ranks}" \
  "${preflags[@]}" "${binary}" "${postflags[@]}" "$@"
