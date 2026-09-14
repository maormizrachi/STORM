#!/bin/bash
#SBATCH --job-name=su-olson
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=su-olson-%j.out
#SBATCH --error=su-olson-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OMPI_MCA_btl=self,tcp
# Keep a unique immutable image on the shared filesystem for this job.
# Rebuilding an executable mapped over NFS can otherwise invalidate pages.
mkdir -p .job_binaries
run_image="$PWD/.job_binaries/su_olson.${SLURM_JOB_ID}"
cp ./su_olson "$run_image"
exec mpirun -np "${SLURM_NTASKS}" "$run_image" "${NX:-512}" "${NEW:-8}" "${BOUNDARY:-10000}" "${DTAU:-0.05}" "${OUTPUT:-output}"
