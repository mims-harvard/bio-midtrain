#!/bin/bash
# Submit two short SLURM jobs that run --eval_only on a given DNA-LLM RL ckpt:
#   - --eval_split id   (val_dataloader = id_test_network_split.csv)
#   - --eval_split ood  (val_dataloader = ood_test_network_split.csv)
# Each job runs trainer.evaluate which scores reward_funcs (correctness etc.)
# on the chosen split. Wandb run name is suffixed by split.
#
# Usage:
#   CKPT_DIR=/path/to/rl/output_dir TEXT_MODEL=Qwen/Qwen3-1.7B \
#     bash submit_id_ood_eval_rl.sh
#
# Required env: CKPT_DIR (path to the RL output_dir; the launcher auto-picks
#               the latest checkpoint-N inside, or final/ if present).
# Optional env: TEXT_MODEL (default Qwen/Qwen3-1.7B), DNA_MODEL (evo2_1b_base),
#               LORA_R (32), LORA_ALPHA (64),
#               PARTITION (kempner_h100), ACCOUNT (kempner_sham_lab),
#               TIME (02:00:00), MEM (80G), AFTEROK_JID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR=/path/to/rl/output_dir}"
# May not exist yet when chained via afterok; the eval job picks the actual
# checkpoint subdir at runtime.
mkdir -p "${CKPT_DIR}"

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
DRIVE_ROOT="${DRIVE_ROOT:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}}"
GENOMICS_DIR="${GENOMICS_DIR:-${DRIVE_ROOT}/genomics}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${WORKING_DIR}/logs/lora_sweeps/eval_id_ood_rl}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "No python at ${PYTHON}" >&2; exit 1; }

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen3-1.7B}"
DNA_MODEL="${DNA_MODEL:-evo2_1b_base}"

PARTITION="${PARTITION:-kempner_requeue}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
TIME="${TIME:-7-00:00}"
CPUS="${CPUS:-16}"
MEM="${MEM:-80G}"
REQUEUE_FLAG="${REQUEUE_FLAG:---requeue}"

for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing drive CSV: ${f}" >&2; exit 1; }
done

mkdir -p "${EVAL_OUT_ROOT}"

# At job runtime we pick the actual ckpt path (latest checkpoint-N or final/).
PICK_CKPT='if [ -d "$CKPT_DIR/final" ]; then EVAL_CKPT="$CKPT_DIR/final"; else EVAL_CKPT=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true); fi; if [ -z "${EVAL_CKPT:-}" ]; then echo "No HF checkpoint under $CKPT_DIR" >&2; exit 1; fi; echo "[eval_rl] using $EVAL_CKPT";'

ENV_PRELUDE='set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12;
if [ -n "${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi;
cd "$WORKING_DIR";
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"; export HF_HOME="$CACHE_DIR/hf_home"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub";
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"; export XDG_CACHE_HOME="$CACHE_DIR/xdg"; export TORCH_HOME="$CACHE_DIR/torch";
export TRITON_CACHE_DIR="$CACHE_DIR/triton"; export WANDB_DIR="$CACHE_DIR/wandb"; export WANDB_CACHE_DIR="$CACHE_DIR/wandb_cache";
export PIP_CACHE_DIR="$CACHE_DIR/pip"; export MPLCONFIGDIR="$CACHE_DIR/mpl"; export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}";
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR";
export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1;
export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))";'

submit_eval () {
  local split="$1"
  local jname="rl-eval-${split}-r${LORA_R}"
  local dep_arg=""
  if [[ -n "${AFTEROK_JID:-}" ]]; then
    dep_arg="--dependency=afterok:${AFTEROK_JID}"
  fi
  sbatch --parsable ${dep_arg} ${REQUEUE_FLAG} \
    --job-name="${jname}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --gpus=1 --mem="${MEM}" \
    --output="${EVAL_OUT_ROOT}/${jname}_%j.out" \
    --error="${EVAL_OUT_ROOT}/${jname}_%j.err" \
    --export=ALL,WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",CACHE_DIR="${CACHE_DIR}",GENOMICS_DIR="${GENOMICS_DIR}",CKPT_DIR="${CKPT_DIR}",SPLIT="${split}",LORA_R="${LORA_R}",LORA_ALPHA="${LORA_ALPHA}",TEXT_MODEL="${TEXT_MODEL}",DNA_MODEL="${DNA_MODEL}" \
    --wrap="${ENV_PRELUDE} ${PICK_CKPT} stdbuf -oL -eL \"\$PYTHON\" train_grpo.py \
      --eval_only true --eval_split \"\$SPLIT\" --eval_ckpt \"\$EVAL_CKPT\" \
      --text_model_name \"\$TEXT_MODEL\" --dna_model_name \"\$DNA_MODEL\" \
      --cache_dir \"\$CACHE_DIR\" --peft_ckpt False \
      --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 \
      --kegg_data_dir_local \"\$GENOMICS_DIR\" \
      --lora_r \"\$LORA_R\" --lora_alpha \"\$LORA_ALPHA\" --lora_dropout 0.05 \
      --gradient_accumulation_steps 1 --gradient_checkpointing True \
      --max_steps 0 --num_train_epochs 0 \
      --max_completion_length 800 --num_generations 8 \
      --generation_batch_size 8 \
      --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
      --beta 0.0 --learning_rate 1e-5 --logging_steps 1 \
      --temperature 1 --top_p 0.95 --top_k 20 \
      --output_dir \"\$CKPT_DIR/_eval_\$SPLIT\" --save_strategy no \
      --eval_strategy no \
      --use_vllm False --bf16 True \
      --resume_from_checkpoint False --report_to wandb \
      --run_name \"rl-eval-\$(basename \$CKPT_DIR)-\$SPLIT\""
}

ID_JID=$(submit_eval id)
OOD_JID=$(submit_eval ood)

echo "CKPT_DIR=${CKPT_DIR}"
echo "  ID  jid=${ID_JID}"
echo "  OOD jid=${OOD_JID}"
echo "Logs: ${EVAL_OUT_ROOT}"
