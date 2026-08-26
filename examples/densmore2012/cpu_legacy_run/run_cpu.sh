#!/bin/bash
# Densmore 2012 CPU baseline with the shared full-IMC kernel forced off, so the
# legacy CPU transport path runs. Used to bisect the shared-kernel regression.
#SBATCH --job-name=DensLegacy
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=DensmoreLegacy_%j.out
#SBATCH --error=DensmoreLegacy_%j.err

set -euo pipefail

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"
binary="$(cd "$run_dir/.." && pwd)/densmore2012_cpu_legacy"
if [[ ! -x "$binary" ]]; then
    echo "Missing executable $binary" >&2
    exit 1
fi

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node "$binary"
