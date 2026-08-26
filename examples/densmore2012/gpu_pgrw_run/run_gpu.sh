#!/bin/bash
# Densmore 2012 multigroup PGRW GPU validation.
#SBATCH --job-name=DensPGRW
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmorePGRW_%j.out
#SBATCH --error=DensmorePGRW_%j.err

set -euo pipefail

if [[ "${1:-}" == --bind ]]; then
    shift
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="${local_rank}"
    export OMP_NUM_THREADS=1
    exec "$@"
fi

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
binary="$(cd "$run_dir/.." && pwd)/densmore2012_gpu_pgrw"
cd "$run_dir"

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node \
    "$run_dir/run_gpu.sh" --bind "$binary"
