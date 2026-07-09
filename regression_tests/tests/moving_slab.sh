#!/usr/bin/env bash

TEST_ID="moving_slab"
TAGS="mpi"
BUILD_TARGET="moving_slab"
RUN_DIR_REL="examples/moving_slab"
MPI_NP=4
RUN_COMMAND='mpirun --oversubscribe -np ${MPI_NP} "${STORM_BIN}"'
CHECK_FUNCTION="check_moving_slab_case"
TIMEOUT="7200"
