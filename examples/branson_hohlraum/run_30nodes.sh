#!/bin/bash
# STORM running Branson's 3D hohlraum deck (wall at 0.83 keV) on the same 30 nodes,
# same packet budget and same partition weighting Branson uses, so the wall-clock
# difference is code, not problem.
#SBATCH --partition=bigrun
#SBATCH --job-name=bh-storm-30
#SBATCH --nodes=30
#SBATCH --ntasks=480
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/maorm/RICH/source/monte/examples/branson_hohlraum/storm30_%j.out
#SBATCH --error=/home/maorm/RICH/source/monte/examples/branson_hohlraum/storm30_%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module restore
cd /home/maorm/RICH/source/monte/examples/branson_hohlraum
export OMP_NUM_THREADS=1
echo "nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS weights=${BH_WEIGHTS:-branson} start=$(date)"
# nppc=15 -> cap 20*15=300 packets per wall cell (Branson gives 309), 51.8M total.
time mpirun -np "$SLURM_NTASKS" ./branson_hohlraum "${NPPC:-15}" "${STEPS:-5}"
echo "end=$(date)"
