#!/bin/bash
# Densmore 2012 GPU run: one node, one GPU/GCD and one CPU core per MPI rank.
#
# Default: two rank/GPU/core triplets.  To change the GPU and core count, change
# only --ntasks below (each task always gets one GPU and one CPU core), or submit
# without editing, for example: sbatch --ntasks=8 run_gpu_single.sh
# Do not request more than eight tasks: a Frontier node has eight GPU GCDs.
# Submit from this directory with: sbatch run_gpu_single.sh
# The existing densmore2012_gpu executable is used; this script does not build.
#SBATCH --job-name=DensmoreGPUSingleNode
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --network=single_node_vni
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmoreGPUSingle_%j.out
#SBATCH --error=DensmoreGPUSingle_%j.err

set -euo pipefail

# Map each local Open MPI rank to one Frontier GPU/GCD.  ROCR renumbers the
# sole visible device to zero inside its rank.
cores_per_rank="${SLURM_CPUS_PER_TASK:-1}"
if [[ "${1:-}" == --bind ]]; then
    shift
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="${local_rank}"
    export OMP_NUM_THREADS="${cores_per_rank}"
    exec "$@"
fi

# Slurm executes a spool copy of this script, so BASH_SOURCE points into
# /var/spool on compute nodes.  Submit this script from the Densmore example
# directory; Slurm records that persistent directory in SLURM_SUBMIT_DIR.
run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"

if [[ ! -x ./densmore2012_gpu ]]; then
    echo "Missing executable ./densmore2012_gpu in $(pwd)" >&2
    exit 1
fi

# Frontier does not initialize the libfabric CXI provider for a single-node
# job unless the single-node VNI above is requested at submission time.
# Slurm grants each task one GPU/GCD.  Per-rank device masking
# is applied in the --bind branch above.
rank_count="${SLURM_NTASKS:-2}"
if (( rank_count < 1 || rank_count > 8 )); then
    echo "This single-node Frontier launch requires 1-8 ranks/GPU GCDs; got ${rank_count}." >&2
    exit 2
fi
export OMP_NUM_THREADS="${cores_per_rank}"

# Keep MPI registration caches bounded during packet-buffer resizing, matching
# the multi-rank GPU launch configuration.
export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export FI_CXI_RX_MATCH_MODE=hybrid

# Start one rank per GCD.  Re-exec the shared-filesystem script because Slurm
# submits a spool copy that is not visible to ranks launched by Open MPI.
exec mpirun -np "${rank_count}" --map-by "ppr:${rank_count}:node:PE=${cores_per_rank}" --bind-to core \
    "$run_dir/run_gpu_single.sh" --bind "$run_dir/densmore2012_gpu"
