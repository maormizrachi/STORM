#!/usr/bin/env bash

TEST_ID="cartesian_parallel_check"
TAGS="mpi"
BUILD_TARGET="cartesian_parallel_check"
RUN_DIR_REL="examples/cartesian_parallel_check"
MPI_NP=4
RUN_COMMAND='mpirun --oversubscribe -np ${MPI_NP} "${STORM_BIN}"'
CHECK_FUNCTION="check_cartesian_parallel_case"
TIMEOUT="300"
