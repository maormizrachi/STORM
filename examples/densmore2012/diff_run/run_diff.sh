#!/bin/bash
# Densmore run with the IMC differential harness enabled. Every transport step
# is executed twice on identical input, once through the shared AdvanceIMC
# kernel and once through the legacy CPU event code, and the two outcomes are
# compared. Divergences are printed to stderr as [IMC-DIFF] lines.
#
# Launch geometry mirrors the known-good CPU baseline script; only the binary
# and the (reduced) problem size differ.
#SBATCH --job-name=DensDiff
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=00:30:00
#SBATCH --output=DensmoreDiff_%j.out
#SBATCH --error=DensmoreDiff_%j.err

set -euo pipefail

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"

binary=/ccs/home/maormiz/RICH/source/monte/examples/densmore2012/densmore2012_diff
if [[ ! -x "$binary" ]]; then
    echo "Missing executable $binary" >&2
    exit 1
fi

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node "$binary" 256 4 10
