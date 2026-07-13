# Adding a New Regression Test

Tests are auto-discovered from `examples/*/REGRESSION_INFO` files. To add a new regression test, create a `REGRESSION_INFO` file in your example directory:

```bash
# examples/my_test/REGRESSION_INFO
TAGS="serial"
CHECK_FUNCTION="check_my_test"
TIMEOUT=1800
```

The test ID, build target, and run directory are derived from the directory name. Sensible defaults are applied based on test type (serial or mpi), so only non-default values need to be specified.

For MPI tests, set `MPI_NP` and optionally override `SBATCH_ARGS` and `RUN_COMMAND`:

```bash
# examples/my_mpi_test/REGRESSION_INFO
TAGS="mpi"
MPI_NP=64
CHECK_FUNCTION="check_my_test"
TIMEOUT=3600
```

## Available Fields

Defaults in parentheses.

| Field | Description |
|---|---|
| `TAGS` | **Required.** `"serial"` or `"mpi"` |
| `CHECK_FUNCTION` | **Required.** Validation function from `regression_checks.sh` |
| `BUILD_TARGET` | CMake target name (directory name) |
| `RUN_COMMAND` | Command to run (binary for serial, `mpirun` for mpi) |
| `TIMEOUT` | Max runtime in seconds (3600) |
| `MPI_NP` | Number of MPI ranks (4) |
| `SBATCH_ARGS` | sbatch resource flags (`-n 1` for serial, `-n ${MPI_NP}` for mpi) |
| `MEM_PER_CPU` | Memory per CPU; triggers auto `--ntasks-per-node` calculation |

All tests are submitted as individual SLURM batch jobs via `sbatch`.

## Check Functions

Each test must specify a `CHECK_FUNCTION` that validates results after the run completes. These functions are defined in `regression_tests/lib/regression_checks.sh`. The function receives the test's stdout/stderr log paths and should set `REGRESSION_CHECK_RESULT` to `0` (pass) or `1` (fail), and `REGRESSION_CHECK_MSG` to a human-readable summary.
