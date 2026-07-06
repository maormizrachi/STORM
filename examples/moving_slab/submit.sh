#!/bin/bash

#SBATCH --job-name="MovingSlab"
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=64
#SBATCH --partition=bigrun
#SBATCH --output=moving_slab_%j.out
#SBATCH --error=moving_slab_%j.err

srun systemctl start drop-caches
mpirun ./moving_slab
