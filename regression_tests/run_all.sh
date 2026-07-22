#!/usr/bin/env bash

set -euo pipefail

STORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THUNDER_ROOT="${STORM_ROOT}/regression_tests/THUNDER"
exec "${THUNDER_ROOT}/run_all.sh" \
  --config "${STORM_ROOT}/regression_tests/config.json" "$@"
