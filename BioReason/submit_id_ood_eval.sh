#!/bin/bash
# Submit two short SLURM jobs that run --eval_only on a given DNA-LLM SFT ckpt:
#   - one with --eval_split id   (val_dataloader = id_test_network_split.csv)
#   - one with --eval_split ood  (val_dataloader = ood_test_network_split.csv)
# Each job runs trainer.test, which invokes the existing on_test_epoch_end pipeline
# (generation + accuracy). Wandb run_name is suffixed with -eval-id / -eval-ood so
# metrics from ID and OOD land in separate runs.
#
# Usage:
#   CKPT=/path/to/some-sft.ckpt LORA_RANK=16 TEXT_MODEL=Qwen/Qwen3-1.7B \
#     bash submit_id_ood_eval.sh
#
# Required env: CKPT
# Optional env: LORA_RANK (default 16), LORA_ALPHA (default 2*rank),
#               TEXT_MODEL (default Qwen/Qwen3-1.7B), DNA_MODEL (default evo2_1b_base),
#               PARTITION (default kempner_h100), ACCOUNT (default kempner_sham_lab),
#               TIME (default 02:00:00), MEM (default 80G).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

CKPT="${CKPT:?Set CKPT=/path/to/SFT.ckpt}"
# DeepSpeed stage-2 ckpts are directories ('last.ckpt/' with checkpoint/, latest, zero_to_fp32.py).
[[ -e "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }
if [[ -d "${CKPT}" ]]; then
  STRATEGY="${STRATEGY:-deepspeed_stage_2}"
else
  STRATEGY="${STRATEGY:-ddp}"
fi

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
DRIVE_ROOT="${DRIVE_ROOT:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}}"
GENOMICS_DIR="${GENOMICS_DIR:-${DRIVE_ROOT}/genomics}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${WORKING_DIR}/logs/lora_sweeps/eval_id_ood}"
EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints/_eval_$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "No python at ${PYTHON}" >&2; exit 1; }

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen3-1.7B}"
DNA_MODEL="${DNA_MODEL:-evo2_1b_base}"

PARTITION="${PARTITION:-kempner_requeue}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
TIME="${TIME:-7-00:00}"
CPUS="${CPUS:-16}"
MEM="${MEM:-80G}"
# IMPORTANT: vortex flash_attn binaries are NOT compiled for A100-MIG slices, so jobs
# landing on a-mig fail with "no kernel image is available for execution on the device"
# during evo2 forward. Force h100|h200 by default to avoid this. Override via CONSTRAINT.
CONSTRAINT="${CONSTRAINT:-h100|h200}"
REQUEUE_FLAG="${REQUEUE_FLAG:---requeue}"

for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing drive CSV: ${f}" >&2; exit 1; }
done

mkdir -p "${EVAL_OUT_ROOT}" "${EVAL_CKPT_ROOT}"

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
  local jname="eval-${split}-r${LORA_RANK}"
  local dep_arg=""
  if [[ -n "${AFTEROK_JID:-}" ]]; then
    dep_arg="--dependency=afterok:${AFTEROK_JID}"
  fi
  sbatch --parsable ${dep_arg} ${REQUEUE_FLAG} \
    --job-name="${jname}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --gpus=1 --mem="${MEM}" --constraint="${CONSTRAINT}" \
    --output="${EVAL_OUT_ROOT}/${jname}_%j.out" \
    --error="${EVAL_OUT_ROOT}/${jname}_%j.err" \
    --export=ALL,WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",CACHE_DIR="${CACHE_DIR}",GENOMICS_DIR="${GENOMICS_DIR}",CKPT="${CKPT}",CKPT_DIR="${EVAL_CKPT_ROOT}",SPLIT="${split}",LORA_RANK="${LORA_RANK}",LORA_ALPHA="${LORA_ALPHA}",TEXT_MODEL="${TEXT_MODEL}",DNA_MODEL="${DNA_MODEL}",STRATEGY="${STRATEGY}" \
    --wrap="${ENV_PRELUDE} stdbuf -oL -eL \"\$PYTHON\" train_dna_qwen.py \
      --eval_only True --eval_split \"\$SPLIT\" --ckpt_path \"\$CKPT\" \
      --cache_dir \"\$CACHE_DIR\" --text_model_name \"\$TEXT_MODEL\" \
      --dna_model_name \"\$DNA_MODEL\" --strategy \"\$STRATEGY\" \
      --max_epochs 1 --num_gpus 1 --batch_size 1 \
      --model_type dna-llm --dataset_type kegg \
      --kegg_data_dir_local \"\$GENOMICS_DIR\" \
      --max_length_dna 2048 --truncate_dna_per_side 1024 \
      --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 \
      --return_answer_in_batch True \
      --lora_rank \"\$LORA_RANK\" --lora_alpha \"\$LORA_ALPHA\" \
      --checkpoint_dir \"\$CKPT_DIR\" \
      --wandb_project BioReason-kegg-lora-eval"
}

SPLITS="${SPLITS:-id ood}"
echo "CKPT=${CKPT}"
for sp in ${SPLITS}; do
  jid=$(submit_eval "${sp}")
  echo "  ${sp}  jid=${jid}"
done
echo "Logs: ${EVAL_OUT_ROOT}"
