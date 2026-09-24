#!/bin/bash
#SBATCH --partition=core
#SBATCH --job-name=bh-storm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/home/maorm/RICH/source/monte/examples/branson_hohlraum/smoke_%j.out
#SBATCH --error=/home/maorm/RICH/source/monte/examples/branson_hohlraum/smoke_%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module restore
cd /home/maorm/RICH/source/monte/examples/branson_hohlraum
mpirun -np "$SLURM_NTASKS" ./branson_hohlraum "${NPPC:-1}" "${STEPS:-2}"
