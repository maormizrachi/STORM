#!/bin/bash
# Densmore 2012 full-IMC GPU job (1 MPI rank per GCD).
# Submit from this directory:  sbatch run_gpu.sh
#SBATCH --job-name=DensmoreGPU
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmoreGPU_%j.out
#SBATCH --error=DensmoreGPU_%j.err
# BEGIN = started running; END = finished; FAIL = failed after start.
# Cancelled jobs (including pending scancel) do not send FAIL.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

# When launched under mpirun, pin this rank to one GCD and exec the binary.
if [[ "${1:-}" == --bind ]]; then
    shift
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    # ROCR_VISIBLE_DEVICES already reduces this rank to a single GCD, which the
    # HIP runtime then renumbers to device 0. Setting a HIP/CUDA level mask on
    # top of it selects an index that no longer exists, so clear them.
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="${local_rank}"
    export OMP_NUM_THREADS=1
    exec "$@"
fi

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"
if [[ ! -x ./densmore2012_gpu ]]; then
    echo "Missing executable ./densmore2012_gpu in $(pwd)" >&2
    exit 1
fi

# Particle queues resize repeatedly. Do not let Open MPI/UCX retain the
# deregistered mappings: the default UCX allowance is 512 MiB per process,
# which can consume 8 GiB on a 16-rank node after several growth cycles.
export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

# 4 nodes × 8 GCDs. Kokkos initialize() uses HIP device 0 of the devices
# this rank can see; --bind hides every GCD but one.
# sbatch copies this script to a node-local spool path, so re-exec the shared
# filesystem copy: the spool path does not exist on the other nodes.
# Default densmore2012 args: Nx=512, new_per_cell=16, boundary_per_cell=100.
exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node \
    "$run_dir/run_gpu.sh" --bind "$run_dir/densmore2012_gpu"
