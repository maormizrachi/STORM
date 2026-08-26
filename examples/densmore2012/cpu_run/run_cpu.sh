#!/bin/bash
# Densmore 2012 full-IMC CPU job (one MPI rank, one CPU core).
# Runs inside cpu_run/ so that densmore2012_profile.txt and the plot do not
# collide with the GPU run's artifacts in the parent directory.
# Submit from this directory:  sbatch run_cpu.sh
#SBATCH --job-name=DensmoreCPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --network=single_node_vni
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmoreCPU_%j.out
#SBATCH --error=DensmoreCPU_%j.err
# BEGIN = started running; END = finished; FAIL = failed after start.
# Cancelled jobs (including pending scancel) do not send FAIL.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

# Slurm executes a spool copy of this script, so BASH_SOURCE points into
# /var/spool on compute nodes.  Submit this script from cpu_run; Slurm records
# that persistent directory in SLURM_SUBMIT_DIR.
run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"
binary="$run_dir/densmore2012_cpu"
if [[ ! -x "$binary" ]]; then
    echo "Missing executable $binary" >&2
    exit 1
fi

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

# Default densmore2012 args: Nx=512, new_per_cell=16, boundary_per_cell=100.
# The single-node VNI above lets Open MPI/STORM initialize libfabric CXI.
exec mpirun -np "${SLURM_NTASKS:-1}" "$binary"
