#!/usr/bin/env bash

TEST_ID="hohlraum_parallel"
TAGS="mpi"
BUILD_TARGET="hohlraum_parallel"
RUN_DIR_REL="examples/hohlraum_parallel"
MPI_NP=4
RUN_COMMAND='mpirun --oversubscribe -np ${MPI_NP} "${STORM_BIN}"'
CHECK_FUNCTION="check_hohlraum_parallel_case"
TIMEOUT="7200"
