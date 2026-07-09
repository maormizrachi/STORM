#!/usr/bin/env bash

TEST_ID="serial_cartesian"
TAGS="serial"
BUILD_TARGET="serial_cartesian"
RUN_DIR_REL="examples/serial_cartesian"
RUN_COMMAND='"${STORM_BIN}"'
CHECK_FUNCTION="check_serial_cartesian_case"
TIMEOUT="300"
