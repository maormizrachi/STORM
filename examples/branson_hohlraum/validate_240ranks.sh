#!/bin/bash
#SBATCH --partition=core
#SBATCH --job-name=bh-validate240
#SBATCH --nodes=15
#SBATCH --ntasks=240
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/home/maorm/RICH/source/monte/examples/branson_hohlraum/validate_%j.out
#SBATCH --error=/home/maorm/RICH/source/monte/examples/branson_hohlraum/validate_%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module restore
cd /home/maorm/RICH/source/monte/examples/branson_hohlraum
# 240 ranks x 9.46 MB = 2.27 GB of border pairs under the old code -> past INT_MAX.
mpirun -np "$SLURM_NTASKS" ./branson_hohlraum 1 1
