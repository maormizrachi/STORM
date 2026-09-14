#!/bin/bash
#SBATCH --job-name=olson-2d-2020
#SBATCH --partition=bigrun
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH --output=olson-2020-%j.out
#SBATCH --error=olson-2020-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OMPI_MCA_btl=self,tcp
# Keep a unique immutable image on the shared filesystem for this job.
# Rebuilding an executable mapped over NFS can otherwise invalidate pages.
mkdir -p .job_binaries
run_image="$PWD/.job_binaries/olson_2d_2020.${SLURM_JOB_ID}"
cp ./olson_2d_2020 "$run_image"
exec mpirun -np "${SLURM_NTASKS}" "$run_image" "${NX:-380}" "${NEW:-4}" "${SOURCE:-512}" "${DT:-1e-13}" "${OUTPUT:-output}" "${CENSUS:-80}"
