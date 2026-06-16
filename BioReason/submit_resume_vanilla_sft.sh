#!/bin/bash
set -euo pipefail

# Submit ONLY resume jobs for unfinished vanilla BioReason SFT configs.
# Target configs:
#   c9  = Qwen3-4B, variant_effect_coding, max_epochs=5
#   c10 = Qwen3-4B, variant_effect_coding, max_epochs=10
#   c12 = Qwen3-4B, variant_effect_non_snv, max_epochs=10
#
# Usage:
#   bash submit_resume_vanilla_sft.sh
#   DRY_RUN=1 bash submit_resume_vanilla_sft.sh
#
# Optional env overrides:
#   PARTITION, ACCOUNT, CONSTRAINT, TIME_LIMIT, CPUS, MEM, GPUS, USER_NAME,
#   CACHE_DIR, CHECKPOINT_DIR, WORKING_DIR, WANDB_MODE, FORCE_BATCH_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${USER_NAME:-$(whoami)}"
PARTITION="${PARTITION:-seas_gpu}"
ACCOUNT="${ACCOUNT:-barak_lab}"
CONSTRAINT="${CONSTRAINT:-h200}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00}"
CPUS="${CPUS:-24}"
MEM="${MEM:-64G}"
GPUS="${GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

# Slurm feature syntax is usually "h200" rather than "gpu=h200".
# Accept either form and normalize.
if [[ "${CONSTRAINT}" == gpu=* ]]; then
  CONSTRAINT="${CONSTRAINT#gpu=}"
fi

CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

PYTHON="${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Please run on a Slurm login node." >&2
  exit 1
fi

mkdir -p "${WORKING_DIR}/logs/sweeps_resume"

# We observed two coding checkpoints from timed-out runs.
# Use the one with lower completed epoch for c9 (5ep target),
# and the one with higher completed epoch for c10 (10ep target).
declare -a RESUME_SPECS=(
  "c9|variant_effect_coding|5|2|${BIOREASON_WORK_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm}/BioReason/checkpoints/BioReason-variant_effect_coding-Qwen3-4B-20260402-015607/last.ckpt"
  "c10|variant_effect_coding|10|2|${BIOREASON_WORK_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm}/BioReason/checkpoints/BioReason-variant_effect_coding-Qwen3-4B-20260402-015602/last.ckpt"
  "c12|variant_effect_non_snv|10|2|${BIOREASON_WORK_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm}/BioReason/checkpoints/BioReason-variant_effect_non_snv-Qwen3-4B-20260402-015643/last.ckpt"
)

submit_resume_job() {
  local cfg_name="$1"
  local dataset="$2"
  local max_epochs="$3"
  local base_bs="$4"
  local ckpt_path="$5"
  local batch_size="${FORCE_BATCH_SIZE:-${base_bs}}"
  local job_name="bioreason-resume-${cfg_name}"
  local log_prefix="${WORKING_DIR}/logs/sweeps_resume/${cfg_name}"

  if [[ ! -e "${ckpt_path}" ]]; then
    echo "SKIP ${cfg_name}: checkpoint path not found: ${ckpt_path}"
    return 0
  fi

  local active_job_id
  active_job_id=$(squeue -h -u "${USER_NAME}" --states=PD,R,CF,S,RS --name "${job_name}" -o "%i" | head -n 1 || true)
  if [[ -n "${active_job_id}" ]]; then
    echo "SKIP ${cfg_name}: active job exists (${active_job_id})"
    return 0
  fi

  local cmd=(
    sbatch
    --job-name="${job_name}"
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --constraint="${CONSTRAINT}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --gpus="${GPUS}"
    --mem="${MEM}"
    --output="${log_prefix}_%j.out"
    --error="${log_prefix}_%j.err"
    --export=ALL,WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",CACHE_DIR="${CACHE_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR}",WANDB_MODE="${WANDB_MODE}",DATASET="${dataset}",MAX_EPOCHS="${max_epochs}",BATCH_SIZE="${batch_size}",CKPT_PATH="${ckpt_path}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${job_name} dataset=${dataset} max_epochs=${max_epochs} batch_size=${batch_size}"
    echo "[DRY_RUN] ckpt_path=${ckpt_path}"
    return 0
  fi

  local submit_out
  submit_out=$(
    "${cmd[@]}" <<'SBATCH'
#!/bin/bash
set -euo pipefail

module load cuda/12.4
module load gcc/12.2.0-fasrc01
module load cudnn/9.10.2.21_cuda12
if [[ -n "${CUDNN_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${WORKING_DIR}"

export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export HF_HOME="${CACHE_DIR}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TORCH_HOME="${CACHE_DIR}/torch"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export WANDB_DIR="${CACHE_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb_cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export MPLCONFIGDIR="${CACHE_DIR}/mpl"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
         "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${WANDB_DIR}" \
         "${WANDB_CACHE_DIR}" "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${TMPDIR}"

export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1
export TORCHDYNAMO_DISABLE=1
export NVTE_TORCH_COMPILE=0
export DEEPSPEED_NO_MPI=1
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"

stdbuf -oL -eL "${PYTHON}" train_dna_qwen.py \
  --cache_dir "${CACHE_DIR}" \
  --text_model_name Qwen/Qwen3-4B \
  --dna_model_name evo2_1b_base \
  --strategy deepspeed_stage_2 \
  --max_epochs "${MAX_EPOCHS}" \
  --num_gpus 1 \
  --batch_size "${BATCH_SIZE}" \
  --model_type dna-llm \
  --dataset_type "${DATASET}" \
  --max_length_dna 2048 \
  --truncate_dna_per_side 1024 \
  --dna_is_evo2 True \
  --dna_embedding_layer blocks.20.mlp.l3 \
  --return_answer_in_batch True \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --ckpt_path "${CKPT_PATH}"
SBATCH
  )
  echo "SUBMITTED ${cfg_name}: ${submit_out}"
}

echo "Submitting dedicated resume jobs for unfinished vanilla SFT configs..."
for spec in "${RESUME_SPECS[@]}"; do
  IFS='|' read -r cfg_name dataset max_epochs base_bs ckpt_path <<<"${spec}"
  submit_resume_job "${cfg_name}" "${dataset}" "${max_epochs}" "${base_bs}" "${ckpt_path}"
done

echo "Done."
