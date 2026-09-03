#!/usr/bin/env python3
"""Run the unchanged RICH moving-slab 32-group validation."""

from pathlib import Path
import runpy


CHECKER = (
    Path(__file__).resolve().parents[4]
    / "regression_tests"
    / "lib"
    / "check_moving_slab_mc_32.py"
)

if __name__ == "__main__":
    runpy.run_path(str(CHECKER), run_name="__main__")
