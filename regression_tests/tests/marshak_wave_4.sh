#!/usr/bin/env bash

TEST_ID="marshak_wave_4"
TAGS="mpi"
BUILD_TARGET="marshak_wave_4"
RUN_DIR_REL="examples/marshak_wave_4"
MPI_NP=4
RUN_COMMAND='mpirun --oversubscribe -np ${MPI_NP} "${STORM_BIN}"'
CHECK_FUNCTION="check_marshak_wave_case"
TIMEOUT="3600"
