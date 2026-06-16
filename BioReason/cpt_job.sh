#!/bin/bash
set -euo pipefail

# Minimal CPT sweep launcher for FineFineWeb biology mid-training.
# Ensures benchmark eval happens:
# 1) during training: evaluation_strategy=steps + eval_steps
# 2) at end: final_eval_metrics.json (eval_loss + eval_perplexity)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
USER_NAME="${USER_NAME:-${USER}}"
cd "${WORKING_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH." >&2
  exit 1
fi

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_mzitnik_lab}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00}"
CPUS="${CPUS:-16}"
MEM="${MEM:-80G}"
GPUS="${GPUS:-1}"
CONSTRAINT="${CONSTRAINT:-}"

PYTHON="${PYTHON:-${SCRATCH:+${SCRATCH}/envs/bio/bin/python}}"
PYTHON="${PYTHON:-$(command -v python3)}"
[[ -x "${PYTHON}" ]] || { echo "No python at ${PYTHON}. Set PYTHON=..." >&2; exit 1; }
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRATCH:+${SCRATCH}/envs/bio/bin/activate}}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
SWEEP_ID="${SWEEP_ID:-cpt_ffw_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${CHECKPOINT_DIR}/drive/${SWEEP_ID}}"
LOG_ROOT="${LOG_ROOT:-${HOME}/projects/evo_omics/BioReason/logs/drive/cpt/${SWEEP_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-bio-cpt}"
REPORT_TO="${REPORT_TO:-wandb}"

MODELS=${MODELS:-"Qwen/Qwen3-1.7B Qwen/Qwen3-4B"}
LRS=${LRS:-"1e-5 3e-4"}
GAS=${GAS:-"64 128"}
MAX_LENS=${MAX_LENS:-"1024"}

TRAIN_SAMPLES="${TRAIN_SAMPLES:-200000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
SKIP_SAMPLES="${SKIP_SAMPLES:-5000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_PROC="${NUM_PROC:-4}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "Submitting CPT sweep: ${SWEEP_ID}"
echo "Output root: ${OUT_ROOT}"

jid_count=0
for model in ${MODELS}; do
  model_tag="$(echo "${model}" | tr '/:' '__')"
  for lr in ${LRS}; do
    for ga in ${GAS}; do
      for max_len in ${MAX_LENS}; do
        exp="m-${model_tag}_lr-${lr}_ga-${ga}_len-${max_len}"
        out_dir="${OUT_ROOT}/${exp}"
        mkdir -p "${out_dir}"

        wrap_cmd=$(cat <<EOF
set -euo pipefail
module load cuda/12.4 || true
module load gcc/12.2.0-fasrc01 || true
module load cudnn/9.10.2.21_cuda12 || true
if [[ -n "${VENV_ACTIVATE}" && -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
fi
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export HF_HOME="${CACHE_DIR}/hf_home"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TORCH_HOME="${CACHE_DIR}/torch"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export WANDB_DIR="${CACHE_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb_cache"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export MPLCONFIGDIR="${CACHE_DIR}/mpl"
export TMPDIR="\${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"
mkdir -p "\${HF_DATASETS_CACHE}" "\${HF_HOME}" "\${HUGGINGFACE_HUB_CACHE}" "\${TRANSFORMERS_CACHE}" "\${XDG_CACHE_HOME}" "\${TORCH_HOME}" "\${TRITON_CACHE_DIR}" "\${WANDB_DIR}" "\${WANDB_CACHE_DIR}" "\${PIP_CACHE_DIR}" "\${MPLCONFIGDIR}" "\${TMPDIR}"
cd "${WORKING_DIR}"
"${PYTHON}" train_finefineweb_midtrain.py \
  --model_name "${model}" \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${out_dir}" \
  --train_samples "${TRAIN_SAMPLES}" \
  --eval_samples "${EVAL_SAMPLES}" \
  --skip_samples "${SKIP_SAMPLES}" \
  --num_train_epochs "${EPOCHS}" \
  --max_length "${max_len}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${ga}" \
  --learning_rate "${lr}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --num_proc "${NUM_PROC}" \
  --report_to "${REPORT_TO}" \
  --bf16 ${RESUME:+--resume_from_checkpoint "${RESUME}"}
"${PYTHON}" - "${out_dir}" <<'PY'
import json, os, sys
out_dir = sys.argv[1]
state_f = os.path.join(out_dir, "trainer_state.json")
final_f = os.path.join(out_dir, "final_eval_metrics.json")
best = None
if os.path.exists(state_f):
    state = json.load(open(state_f))
    for row in state.get("log_history", []):
        if "eval_loss" in row:
            best = row["eval_loss"] if best is None else min(best, row["eval_loss"])
if os.path.exists(final_f):
    final = json.load(open(final_f))
    print(f"[benchmark] best_mid_eval_loss={best}")
    print(f"[benchmark] final_eval_loss={final.get('eval_loss')}")
    print(f"[benchmark] final_eval_perplexity={final.get('eval_perplexity')}")
else:
    print(f"[benchmark] missing {final_f}")
PY
EOF
)

        sbatch_args=(
          --job-name="cpt-${exp}"
          --partition="${PARTITION}"
          --account="${ACCOUNT}"
          --time="${TIME_LIMIT}"
          --nodes=1
          --ntasks=1
          --cpus-per-task="${CPUS}"
          --gpus="${GPUS}"
          --mem="${MEM}"
          --output="${LOG_ROOT}/${exp}_%j.out"
          --error="${LOG_ROOT}/${exp}_%j.err"
          --wrap="${wrap_cmd}"
        )
        if [[ -n "${CONSTRAINT}" ]]; then
          sbatch_args+=(--constraint="${CONSTRAINT}")
        fi

        submit_out="$(sbatch "${sbatch_args[@]}")"
        echo "SUBMITTED ${exp}: ${submit_out}"
        jid_count=$((jid_count + 1))
      done
    done
  done
done

echo "Total submitted jobs: ${jid_count}"
