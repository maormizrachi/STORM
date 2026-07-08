#!/bin/bash

#SBATCH --job-name="MovingSlab"
#SBATCH --ntasks-per-node=3
#SBATCH --ntasks=48
#SBATCH --exclusive
#SBATCH --partition=bigrun
#SBATCH --output=moving_slab_%j.out
#SBATCH --error=moving_slab_%j.err
#SBATCH --distribution=cyclic

srun systemctl start drop-caches
mpirun  ./moving_slab
