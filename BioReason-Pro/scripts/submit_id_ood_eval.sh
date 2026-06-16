#!/bin/bash
# Submit two short SLURM jobs that run --eval_only --gen_eval on a given
# Protein-LLM SFT ckpt:
#   - one with --eval_split id   (id-test.csv)
#   - one with --eval_split ood  (ood-test.csv)
# Each job runs trainer.test, which invokes the new generation-based
# on_test_epoch_end (model.generate -> extract GO IDs -> propagate -> P/R/F1).
# Wandb run_name is suffixed with -eval-id / -eval-ood so metrics from ID and
# OOD land in separate runs.
#
# Usage:
#   CKPT=/path/to/some-protein-sft.ckpt LORA_RANK=128 \
#     bash BioReason-Pro/scripts/submit_id_ood_eval.sh
#
# Required env: CKPT
# Optional env: LORA_RANK (default 128), LORA_ALPHA (default 2*rank),
#               TEXT_MODEL (default Qwen/Qwen3-4B-Thinking-2507),
#               PROTEIN_MODEL (default esm3_sm_open_v1),
#               PARTITION (default kempner_h100), ACCOUNT (default kempner_sham_lab),
#               TIME (default 04:00:00), MEM (default 128GB),
#               GEN_MAX_NEW_TOKENS (default 512), GEN_TEMPERATURE (default 0.0).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT:-/n/home07/hanlinzhang/projects/evo_omics}/BioReason-Pro}"
CKPT="${CKPT:?Set CKPT=/path/to/protein-sft.ckpt}"
[[ -e "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }

REASONING_SFT_DATASET="${REASONING_SFT_DATASET:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein}"
for f in "${REASONING_SFT_DATASET}/train.csv" "${REASONING_SFT_DATASET}/id-test.csv" "${REASONING_SFT_DATASET}/ood-test.csv"; do
  [[ -f "${f}" ]] || { echo "Missing protein CSV: ${f}" >&2; exit 1; }
done

CONDA_ENV="${CONDA_ENV:-${SCRATCH}/envs/bio/bin/activate}"
[[ -e "${CONDA_ENV}" ]] || { echo "No conda env at ${CONDA_ENV}" >&2; exit 1; }

LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
PROTEIN_MODEL="${PROTEIN_MODEL:-esm3_sm_open_v1}"

PARTITION="${PARTITION:-kempner_requeue}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
TIME="${TIME:-7-00:00}"
MEM="${MEM:-128GB}"
# Default h100|h200 (kempner_h100 / kempner_requeue). Set CONSTRAINT="" or
# CONSTRAINT="a100" to use the kempner partition (A100-40GB nodes).
CONSTRAINT="${CONSTRAINT:-h100|h200}"
REQUEUE_FLAG="${REQUEUE_FLAG:-#SBATCH --requeue}"

GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"

WANDB_PROJECT="${WANDB_PROJECT:-esm3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/slurm/eval_id_ood}"
mkdir -p "${LOG_DIR}"

CKPT_BASE="$(basename "${CKPT}" .ckpt)"
RUN_TAG="${RUN_TAG:-${CKPT_BASE}}"

submit_eval () {
  local split="$1"
  local run_name="protein-eval-${RUN_TAG}-${split}"

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

source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
mkdir -p \$WANDB_DIR \$TORCH_HOME \$TRITON_CACHE_DIR
cd ${ROOT_DIR}

python train_protein_llm.py \\
    --eval_only True --eval_split ${split} --gen_eval True \\
    --gen_eval_max_new_tokens ${GEN_MAX_NEW_TOKENS} \\
    --gen_eval_temperature ${GEN_TEMPERATURE} \\
    --ckpt_path ${CKPT} \\
    --dataset_type hf_reasoning \\
    --reasoning_sft_dataset ${REASONING_SFT_DATASET} \\
    --text_model_name ${TEXT_MODEL} \\
    --protein_model_name ${PROTEIN_MODEL} \\
    --attn_implementation flash_attention_2 \\
    --use_unsloth False \\
    --model_type protein-llm \\
    --lora_rank ${LORA_RANK} \\
    --lora_alpha ${LORA_ALPHA} \\
    --lora_dropout 0 \\
    --batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --num_workers 0 \\
    --num_gpus 1 \\
    --num_nodes 1 \\
    --return_answer_in_batch True \\
    --max_epochs 1 \\
    --val_split_ratio 0.1 \\
    --val_check_interval 1.0 \\
    --num_sanity_val_steps 0 \\
    --protein_embedding_layer 37 \\
    --protein_model_finetune False \\
    --go_model_finetune False \\
    --unified_go_encoder False \\
    --wandb_project ${WANDB_PROJECT} \\
    --run_name ${run_name} \\
    --debug False
EOF
  echo "Submitted: ${run_name}"
}

SPLITS="${SPLITS:-id ood}"
for sp in ${SPLITS}; do submit_eval "${sp}"; done

echo ""
echo "CKPT=${CKPT}"
echo "Logs: ${LOG_DIR}"
