#!/bin/bash
# Same allocation, same binary, same deck: two-sided MPI vs STORM's native RDMA.
# Deck moves ~167M packets across ranks per cycle (Branson's own counter), so the
# transport layer is actually exercised here.
#SBATCH --partition=bigrun
#SBATCH --job-name=bh-p2p-vs-rdma
#SBATCH --nodes=30
#SBATCH --ntasks=480
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/maorm/RICH/source/monte/examples/branson_hohlraum/rdma_cmp_%j.out
#SBATCH --error=/home/maorm/RICH/source/monte/examples/branson_hohlraum/rdma_cmp_%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module restore
cd /home/maorm/RICH/source/monte/examples/branson_hohlraum
export OMP_NUM_THREADS=1
echo "nodelist=${SLURM_JOB_NODELIST}"
for mgr in p2p rdma p2p rdma; do
  echo "=================== manager=$mgr ==================="
  BH_MANAGER=$mgr mpirun -np "$SLURM_NTASKS" ./branson_hohlraum 15 5
done
echo "end=$(date)"
