#!/bin/bash
# Densmore 2012 full-IMC CPU job (32 MPI ranks, 8 per node).
# Submit from this directory:  sbatch run_cpu.sh
#SBATCH --job-name=DensmoreCPU
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmoreCPU_%j.out
#SBATCH --error=DensmoreCPU_%j.err
# BEGIN = started running; END = finished; FAIL = failed after start.
# Cancelled jobs (including pending scancel) do not send FAIL.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"
if [[ ! -x ./densmore2012_cpu ]]; then
    echo "Missing executable ./densmore2012_cpu in $(pwd)" >&2
    exit 1
fi

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

# Default densmore2012 args: Nx=512, new_per_cell=16, boundary_per_cell=100.
# sbatch copies this script to a node-local spool path, so re-exec the shared
# filesystem copy: the spool path does not exist on the other nodes.
exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node \
    "$run_dir/densmore2012_cpu"
