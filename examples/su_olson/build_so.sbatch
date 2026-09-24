#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --job-name=so-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=so-build-%j.out
#SBATCH --error=so-build-%j.err
set -euo pipefail
if ! type -t module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module restore; hash -r
cd "$SLURM_SUBMIT_DIR"
BUILD_JOBS=8 ./build.sh
cp "${BUILD_DIR:-/home/maorm/RICH/build/olson_benchmarks}/examples/su_olson/su_olson" ./su_olson.ddmc 2>/dev/null || find /home/maorm/RICH/build/olson_benchmarks -name su_olson -type f -newer main.cpp -exec cp {} ./su_olson.ddmc \;
ls -la ./su_olson.ddmc && SO_DDMC=1 ./su_olson.ddmc --help 2>&1 | head -2 || true
echo BUILD_OK
