#!/bin/bash
# DNA variant-effect-coding SFT epoch sweep — analogue of the KEGG SFT scaling
# in figures/results_id_ood.md ("DNA SFT (CPT epoch sweep, accuracy)") on a
# different DNA task.
#
# Sweeps EPOCHS={1,2,4,8,16} × 1 model (cpt_qwen3_1p7b by default).
# Each SFT job runs trainer.test() at end → prints test accuracy to .out;
# harvest the "Accuracy: 0.NNNN" line from the latest log per epoch.
#
# Usage:
#   bash BioReason/submit_variant_sft_sweep.sh                # 5 jobs (1.7B)
#   EPOCHS_OVERRIDE="2 8" MODEL_TAG=cpt_qwen3_4b TEXT_MODEL=/path/to/4b/final \
#     bash submit_variant_sft_sweep.sh                        # subset
#
# Env: TEXT_MODEL (default: best CPT 1.7B per dna_cpt_midtrain.md),
#      MODEL_TAG (default: cpt_qwen3_1p7b),
#      DATASET_TYPE (default: variant_effect_coding;
#                    alt: variant_effect_non_snv),
#      PARTITION (default: kempner_h100), CONSTRAINT (default: h100|h200),
#      SFT_TIME (default: 1-12:00), SFT_MEM (default: 80G).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"

# Best 1.7B CPT per results/dna_cpt_midtrain.md.
TEXT_MODEL="${TEXT_MODEL:-${CHECKPOINT_DIR}/drive/cpt_ffw_20260424_165721/m-Qwen_Qwen3-1.7B_lr-3e-4_ga-128_len-1024/final}"
MODEL_TAG="${MODEL_TAG:-cpt_qwen3_1p7b}"
DATASET_TYPE="${DATASET_TYPE:-variant_effect_coding}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "Missing PYTHON: ${PYTHON}" >&2; exit 1; }
[[ -d "${TEXT_MODEL}" ]] || { echo "Missing TEXT_MODEL dir: ${TEXT_MODEL}" >&2; exit 1; }

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
SFT_TIME="${SFT_TIME:-1-12:00}"
SFT_MEM="${SFT_MEM:-80G}"
SFT_CPUS="${SFT_CPUS:-16}"

EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-1 2 4 8 16}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ID="variant_sft_${STAMP}"
SWEEP_ROOT="${CHECKPOINT_DIR}/${SWEEP_ID}/${MODEL_TAG}"
LOG_ROOT="${WORKING_DIR}/logs/${SWEEP_ID}/${MODEL_TAG}"
mkdir -p "${SWEEP_ROOT}" "${LOG_ROOT}"

ENV_PRELUDE="set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12;
[ -n \"\${CUDNN_HOME:-}\" ] && export LD_LIBRARY_PATH=\"\${CUDNN_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\";
cd \"${WORKING_DIR}\";
export HF_DATASETS_CACHE=\"${CACHE_DIR}/datasets\"; export HF_HOME=\"${CACHE_DIR}/hf_home\"; export HUGGINGFACE_HUB_CACHE=\"\${HF_HOME}/hub\";
export TRANSFORMERS_CACHE=\"${CACHE_DIR}/transformers\"; export XDG_CACHE_HOME=\"${CACHE_DIR}/xdg\"; export TORCH_HOME=\"${CACHE_DIR}/torch\";
export TRITON_CACHE_DIR=\"${CACHE_DIR}/triton\"; export WANDB_DIR=\"${CACHE_DIR}/wandb\"; export WANDB_CACHE_DIR=\"${CACHE_DIR}/wandb_cache\";
export PIP_CACHE_DIR=\"${CACHE_DIR}/pip\"; export MPLCONFIGDIR=\"${CACHE_DIR}/mpl\"; export TMPDIR=\"\${SLURM_TMPDIR:-/tmp/\${USER}/bioreason_tmp}\";
mkdir -p \"\$HF_DATASETS_CACHE\" \"\$HF_HOME\" \"\$HUGGINGFACE_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$XDG_CACHE_HOME\" \"\$TORCH_HOME\" \"\$TRITON_CACHE_DIR\" \"\$WANDB_DIR\" \"\$WANDB_CACHE_DIR\" \"\$PIP_CACHE_DIR\" \"\$MPLCONFIGDIR\" \"\$TMPDIR\";
export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1;
export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT=\"\$((10000 + (SLURM_JOB_ID % 50000)))\";"

submit_one() {
  local epochs="$1"
  local run_name="variant-sft-${MODEL_TAG}-e${epochs}"
  local ckpt_root="${SWEEP_ROOT}/e${epochs}"
  mkdir -p "${ckpt_root}"

  local wrap_cmd="${ENV_PRELUDE} \
RESUME_CKPT=\$(ls -1td \"${ckpt_root}\"/*/last.ckpt 2>/dev/null | head -1 || true); \
RESUME_OPT=\"\"; if [ -n \"\$RESUME_CKPT\" ] && [ -e \"\$RESUME_CKPT\" ]; then RESUME_OPT=\"--ckpt_path \$RESUME_CKPT\"; fi; \
stdbuf -oL -eL \"${PYTHON}\" train_dna_qwen.py \
  --cache_dir \"${CACHE_DIR}\" --text_model_name \"${TEXT_MODEL}\" \
  --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 \
  --max_epochs ${epochs} --num_gpus 1 --batch_size 1 \
  --model_type dna-llm --dataset_type ${DATASET_TYPE} \
  --max_length_dna 2048 --truncate_dna_per_side 1024 \
  --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 \
  --merge_val_test_set True --return_answer_in_batch True \
  --checkpoint_dir \"${ckpt_root}\" \
  --wandb_project BioReason-variant-sft \$RESUME_OPT"

  jid=$(sbatch --parsable --requeue \
    --job-name="${run_name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" --time="${SFT_TIME}" --nodes=1 --ntasks=1 \
    --cpus-per-task="${SFT_CPUS}" --gpus=1 --mem="${SFT_MEM}" \
    --output="${LOG_ROOT}/${run_name}_%j.out" \
    --error="${LOG_ROOT}/${run_name}_%j.err" \
    --wrap="${wrap_cmd}")
  echo "  ${run_name}  jid=${jid}  (epochs=${epochs})"
}

echo "=== variant SFT scaling sweep ==="
echo "  sweep:    ${SWEEP_ID}"
echo "  model:    ${MODEL_TAG}  ←  ${TEXT_MODEL}"
echo "  dataset:  ${DATASET_TYPE}"
echo "  epochs:   ${EPOCHS_OVERRIDE}"
echo ""
for ep in ${EPOCHS_OVERRIDE}; do
  submit_one "${ep}"
done
echo ""
echo "Logs: ${LOG_ROOT}"
echo "Sweep root (rm -rf to cleanup): ${SWEEP_ROOT}"
