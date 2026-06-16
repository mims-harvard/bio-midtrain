#!/bin/bash
# Build flash-attn + evo2 in the bio venv. Submit from BioReason: sbatch job_install.sh
# Flash-attn compiles many CUDA files; allow the job enough time (1 day below is safe).

#SBATCH -J install
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 1-00:00
#SBATCH --gpus=1
#SBATCH -p seas_gpu
#SBATCH --mem=60GB
#SBATCH --account=barak_lab
#SBATCH -o logs/install/%j.out
#SBATCH -e logs/install/%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
: "${SCRATCH:?SCRATCH must be set (e.g. in ~/.bashrc on the cluster)}"

export MAX_JOBS="${MAX_JOBS:-8}"
bash install_evo2_stack.sh
