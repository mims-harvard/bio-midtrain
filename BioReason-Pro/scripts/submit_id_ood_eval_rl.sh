#!/bin/bash
# Submit two short SLURM jobs that run --eval_only on a given Protein-LLM
# GRPO RL ckpt:
#   - --eval_split id   (id-test.csv)
#   - --eval_split ood  (ood-test.csv)
# Each job runs greedy decoding on the chosen split, computes propagated GO
# F1/P/R via reward_from_text, writes eval_<split>_metrics.json under the
# CKPT_DIR/_eval_<split>/.
#
# Usage:
#   CKPT_DIR=/path/to/protein-grpo-output_dir bash submit_id_ood_eval_rl.sh
#
# Required env: CKPT_DIR (the GRPO output_dir; eval auto-picks final/ if
#               present, else the highest checkpoint-N).
# Optional env: TEXT_MODEL (default Qwen/Qwen3-4B-Thinking-2507),
#               PARTITION (kempner_h100), ACCOUNT (kempner_sham_lab),
#               TIME (04:00:00), MEM (128GB), AFTEROK_JID,
#               GEN_MAX_NEW_TOKENS (512), GEN_TEMPERATURE (default 0.7).
#
# IMPORTANT: greedy decoding (T=0.0) on the Qwen3-4B-Thinking GRPO ckpts produces
# F1=0 across the entire eval set — the deterministic path stays inside the
# `<think>` block, never emits a `<|GO_SUMMARY_*|>` summary, and the extractor
# returns ∅. Training succeeds because rollouts use T=1.0. Diagnostic at T=0.7
# (jid 9013087/9013088) recovered running_f1 ≈ 0.14-0.15 on the same ckpt.
# Default flipped to 0.7 so future evals don't silently hit the bug.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT:-/n/home07/hanlinzhang/projects/evo_omics}/BioReason-Pro}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR=/path/to/protein-grpo-output_dir}"
# CKPT_DIR may not exist yet at submission time when chained via afterok on a
# training job that hasn't run; the eval job picks up the dir at runtime.
mkdir -p "${CKPT_DIR}"

REASONING_SFT_DATASET="${REASONING_SFT_DATASET:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein}"
for f in "${REASONING_SFT_DATASET}/id-test.csv" "${REASONING_SFT_DATASET}/ood-test.csv"; do
  [[ -f "${f}" ]] || { echo "Missing CSV: ${f}" >&2; exit 1; }
done

CONDA_ENV="${CONDA_ENV:-${SCRATCH}/envs/bio/bin/activate}"
[[ -e "${CONDA_ENV}" ]] || { echo "No conda env at ${CONDA_ENV}" >&2; exit 1; }

TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
GO_OBO_PATH="${GO_OBO_PATH:-${ROOT_DIR}/bioreason2/dataset/go-basic.obo}"

PARTITION="${PARTITION:-kempner_requeue}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
TIME="${TIME:-7-00:00}"
MEM="${MEM:-128GB}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
REQUEUE_FLAG="${REQUEUE_FLAG:-#SBATCH --requeue}"

GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"

WANDB_PROJECT="${WANDB_PROJECT:-esm3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/slurm/eval_id_ood_rl}"
mkdir -p "${LOG_DIR}"

CKPT_TAG="${RUN_TAG:-$(basename "${CKPT_DIR}")}"

submit_eval () {
  local split="$1"
  local run_name="protein-rl-eval-${CKPT_TAG}-${split}"
  local dep_arg=""
  if [[ -n "${AFTEROK_JID:-}" ]]; then
    dep_arg="#SBATCH --dependency=afterok:${AFTEROK_JID}"
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH -J ${run_name}
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t ${TIME}
#SBATCH -p ${PARTITION}
#SBATCH --mem=${MEM}
${CONSTRAINT:+#SBATCH --constraint=${CONSTRAINT}}
#SBATCH --account=${ACCOUNT}
${REQUEUE_FLAG}
#SBATCH -o ${LOG_DIR}/${run_name}_%j.out
#SBATCH -e ${LOG_DIR}/${run_name}_%j.err
${dep_arg}

source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
export HF_DATASETS_CACHE=${REASONING_SFT_DATASET}/hf_datasets_cache
mkdir -p \$WANDB_DIR \$TORCH_HOME \$TRITON_CACHE_DIR \$HF_DATASETS_CACHE
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_NAME=${run_name}
cd ${ROOT_DIR}

# Pick the actual ckpt dir at runtime. Default behaviour (BEST_CKPT_BY_REWARD=true)
# scans train_metrics.jsonl and selects the saved checkpoint whose centred
# rolling-mean training reward is highest — necessary because some warm-start
# GRPO runs collapse late (KL_BETA=1e-4 is weak past the first epoch), so
# final/ may have reward 0 while an earlier checkpoint had reward 0.7+.
# Set BEST_CKPT_BY_REWARD=false to keep the old "final/ → latest checkpoint-N"
# behaviour.
if [ "${BEST_CKPT_BY_REWARD:-true}" = "true" ] && [ -f "${CKPT_DIR}/train_metrics.jsonl" ]; then
  EVAL_CKPT=\$(python ${ROOT_DIR}/scripts/pick_best_ckpt.py "${CKPT_DIR}" 2>&1 | tail -1)
elif [ -d "${CKPT_DIR}/final" ]; then
  EVAL_CKPT="${CKPT_DIR}/final"
else
  EVAL_CKPT=\$(ls -1d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "\${EVAL_CKPT:-}" ]; then
  echo "[eval_rl] No ckpt under ${CKPT_DIR}" >&2
  exit 1
fi
echo "[eval_rl] using \$EVAL_CKPT"

python train_protein_grpo.py \\
    --eval_only True --eval_split ${split} \\
    --eval_max_new_tokens ${GEN_MAX_NEW_TOKENS} \\
    --eval_temperature ${GEN_TEMPERATURE} \\
    --dataset_name ${REASONING_SFT_DATASET} \\
    --model_name_or_path \$EVAL_CKPT \\
    --output_dir ${CKPT_DIR}/_eval_${split} \\
    --go_obo_path ${GO_OBO_PATH} \\
    --interpro_in_prompt True \\
    --ppi_in_prompt False \\
    --ask_all_go_aspects True \\
    --append_uniprot_suffix False \\
    --go_summary_tags_in_prompt True \\
    --reward_extraction sft_aligned \\
    --batch_size 1 --group_size 1 --epochs 1 \\
    --max_new_tokens ${GEN_MAX_NEW_TOKENS} \\
    --temperature 1.0 --top_p 1.0 \\
    --epsilon_low 0.2 --epsilon_high 0.28 --kl_beta 1e-4 \\
    --save_every_steps 100000 \\
    --seed 23
EOF
  echo "Submitted: ${run_name}"
}

submit_eval id
submit_eval ood
echo ""
echo "CKPT_DIR=${CKPT_DIR}"
echo "Logs: ${LOG_DIR}"
