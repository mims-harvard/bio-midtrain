# Cluster environment for the bio-post launchers.
#
# The SLURM launchers (BioReason/*.sh, BioReason-Pro/scripts/*.sh) read these
# variables and fall back to the original Kempner-cluster defaults when they are
# unset, so the scripts run unchanged in their home environment. To reuse them
# elsewhere, copy this file, edit the values, and `source` it before launching:
#
#     cp cluster_env.example.sh cluster_env.sh   # then edit
#     source cluster_env.sh
#     bash BioReason/cpt_job.sh
#
# Each variable below shows its built-in default.

# Repository root (this checkout).
export PROJECT_ROOT="/n/home07/hanlinzhang/projects/evo_omics"

# Python interpreter used by the launchers (the project's conda/venv).
export PYTHON_BIN="/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python"

# Datasets root (KEGG / variant-effect / protein CSVs, GO embeddings, ...).
export BIOREASON_DATA_ROOT="/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason"

# Working/checkpoint root for training runs.
export BIOREASON_WORK_ROOT="/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm"

# Alternate working root for a few pre-existing variant-effect checkpoints.
export BIOREASON_WORK_ROOT_ALT="/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm"

# uv package cache.
export UV_CACHE_DIR="/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv"

# NOTE: SLURM `#SBATCH --account=` / `--partition=` directives cannot read shell
# variables (SLURM does not expand them), so edit those header lines directly in
# each script. Defaults used in this repo: account `kempner_sham_lab` (GPU) /
# `barak_lab` (CPU); partitions `kempner`, `kempner_h100`, `kempner_requeue`,
# `seas_compute`.
