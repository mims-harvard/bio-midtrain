#!/bin/bash
# §4.6 / Figure-8 DNA epoch-split ablation: 8 epochs total, vary (sft_e, rl_e).
#
# For each pair (sft_e, rl_e) where sft_e + rl_e = 8:
#   - if sft_e > 0: SFT job for sft_e epochs from CPT init; saves to .../sft/e{sft_e}
#   - if rl_e  > 0: RL  job for rl_e  epochs, --sft_checkpoint = matching SFT
#                   (or CPT-final dir if sft_e == 0); --dependency=afterok on SFT
#
# Cells: (0,8) (1,7) (2,6) (3,5) (4,4) (5,3) (6,2) (7,1) (8,0)
# Skipped: (8,0) — reuses the existing cpt_qwen3_1p7b @ e8 row in
#                figures/results_id_ood.md § "DNA SFT (CPT epoch sweep)".
#
# Backbone: Qwen3-1.7B from CPT_FINAL_DIR (best 1.7B per results/dna_cpt_midtrain.md).
# All artefacts under CHECKPOINT_DIR/section46_dna_<stamp>/ for clean teardown.
#
# Usage:
#   bash BioReason/submit_dna_sft_rl_split.sh                  # fire all 8 cells
#   PAIRS="1:7 2:6"  bash submit_dna_sft_rl_split.sh           # subset
#   DRY_RUN=true     bash submit_dna_sft_rl_split.sh           # validate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
SCRATCH_DATA="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH_DATA}/genomics}"

# CPT 1.7B winner per results/dna_cpt_midtrain.md (lr=3e-4 ga=128, ppl 10.80).
CPT_FINAL_DIR="${CPT_FINAL_DIR:-${CHECKPOINT_DIR}/drive/cpt_ffw_20260424_165721/m-Qwen_Qwen3-1.7B_lr-3e-4_ga-128_len-1024/final}"
TEXT_MODEL="${TEXT_MODEL:-${CPT_FINAL_DIR}}"
MODEL_TAG="${MODEL_TAG:-cpt_qwen3_1p7b}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "Missing PYTHON: ${PYTHON}" >&2; exit 1; }
[[ -d "${CPT_FINAL_DIR}" ]] || { echo "Missing CPT_FINAL_DIR: ${CPT_FINAL_DIR}" >&2; exit 1; }
for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing data CSV: ${f}" >&2; exit 1; }
done

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
SFT_TIME="${SFT_TIME:-2-00:00}"
RL_TIME="${RL_TIME:-2-00:00}"
SFT_MEM="${SFT_MEM:-80G}"
RL_MEM="${RL_MEM:-80G}"
SFT_CPUS="${SFT_CPUS:-16}"
RL_CPUS="${RL_CPUS:-16}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-false}"

PAIRS="${PAIRS:-0:8 1:7 2:6 3:5 4:4 5:3 6:2 7:1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ID="section46_dna_${STAMP}"
SWEEP_ROOT="${CHECKPOINT_DIR}/${SWEEP_ID}"
LOG_ROOT="${WORKING_DIR}/logs/${SWEEP_ID}"
mkdir -p "${SWEEP_ROOT}" "${LOG_ROOT}"

# Pre-substitute wrapper-time vars (WORKING_DIR, CACHE_DIR) at script-build time;
# defer SLURM-runtime vars (CUDNN_HOME, SLURM_JOB_ID, USER) via \$.
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

submit_sft() {
  local sft_e="$1"
  local pair_tag="$2"
  local run_name="section46-dna-sft-${pair_tag}-e${sft_e}"
  local ckpt_root="${SWEEP_ROOT}/sft/${pair_tag}"
  mkdir -p "${ckpt_root}"

  local wrap_cmd
  wrap_cmd="${ENV_PRELUDE} \
    RESUME_CKPT=\$(ls -1td \"${ckpt_root}\"/*/last.ckpt 2>/dev/null | head -1 || true); \
    RESUME_OPT=\"\"; if [ -n \"\$RESUME_CKPT\" ] && [ -e \"\$RESUME_CKPT\" ]; then RESUME_OPT=\"--ckpt_path \$RESUME_CKPT\"; fi; \
    stdbuf -oL -eL \"${PYTHON}\" train_dna_qwen.py \
      --cache_dir \"${CACHE_DIR}\" --text_model_name \"${TEXT_MODEL}\" \
      --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 \
      --max_epochs ${sft_e} --num_gpus 1 --batch_size 1 \
      --model_type dna-llm --dataset_type kegg \
      --kegg_data_dir_local \"${GENOMICS_DIR}\" \
      --max_length_dna 2048 --truncate_dna_per_side 1024 \
      --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 \
      --merge_val_test_set True --return_answer_in_batch True \
      --checkpoint_dir \"${ckpt_root}\" \
      --wandb_project BioReason-section46-dna-sft \$RESUME_OPT"

  if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY] SFT ${run_name}  sft_e=${sft_e}"
    echo "DRYJID"
    return 0
  fi
  local jid
  jid=$(sbatch --parsable --requeue \
    --job-name="${run_name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" \
    --time="${SFT_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${SFT_CPUS}" \
    --gpus=1 --mem="${SFT_MEM}" \
    --output="${LOG_ROOT}/${run_name}_%j.out" \
    --error="${LOG_ROOT}/${run_name}_%j.err" \
    --wrap="${wrap_cmd}")
  echo "  SFT ${run_name}  jid=${jid}" >&2
  echo "${jid}"
}

submit_rl() {
  local rl_e="$1"
  local pair_tag="$2"
  local sft_jid="$3"           # may be empty for sft_e=0
  local sft_ckpt_root="$4"     # SFT checkpoint_dir; "" for sft_e=0
  local run_name="section46-dna-grpo-${pair_tag}-e${rl_e}"
  local rl_out="${SWEEP_ROOT}/rl/${pair_tag}"
  mkdir -p "${rl_out}"

  local dep_arg=""
  [ -n "${sft_jid}" ] && dep_arg="--dependency=afterok:${sft_jid}"

  # If sft_e > 0, RL uses the latest .ckpt under sft_ckpt_root.
  # If sft_e == 0, RL uses CPT_FINAL_DIR directly as --sft_checkpoint
  # (train_grpo accepts a HF model dir as sft_checkpoint when peft_ckpt False).
  local sft_ckpt_select
  if [ -n "${sft_ckpt_root}" ]; then
    sft_ckpt_select='BEST_SFT_CKPT=$(ls -1td "'"${sft_ckpt_root}"'"/*/last.ckpt 2>/dev/null | head -1 || ls -1td "'"${sft_ckpt_root}"'"/*/*.ckpt 2>/dev/null | head -1)'
  else
    sft_ckpt_select='BEST_SFT_CKPT="'"${CPT_FINAL_DIR}"'"'
  fi

  local wrap_cmd
  wrap_cmd="${ENV_PRELUDE} \
    ${sft_ckpt_select}; echo \"[section46-rl] sft_ckpt=\$BEST_SFT_CKPT\"; \
    stdbuf -oL -eL \"${PYTHON}\" train_grpo.py \
      --text_model_name \"${TEXT_MODEL}\" --dna_model_name evo2_1b_base \
      --cache_dir \"${CACHE_DIR}\" --sft_checkpoint \"\$BEST_SFT_CKPT\" \
      --peft_ckpt False --dna_is_evo2 True \
      --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 \
      --kegg_data_dir_local \"${GENOMICS_DIR}\" \
      --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
      --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
      --gradient_accumulation_steps 8 --gradient_checkpointing True \
      --max_steps -1 --num_train_epochs ${rl_e} \
      --max_completion_length 800 --num_generations 8 \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
      --beta 1e-4 --run_name \"section46-dna-rl-${pair_tag}-e${rl_e}\" \
      --learning_rate 1e-5 --logging_steps 1 \
      --temperature 1 --top_p 0.95 --top_k 20 \
      --output_dir \"${rl_out}\" --save_strategy epoch --save_total_limit 1 \
      --lr_scheduler_type cosine --warmup_ratio 0.03 \
      --log_completions True --use_vllm False --bf16 True \
      --resume_from_checkpoint true --report_to wandb"

  if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY] RL  ${run_name}  rl_e=${rl_e}  dep=${sft_jid:-none}  sft_ckpt_root=${sft_ckpt_root:-CPT_FINAL_DIR}"
    return 0
  fi
  local jid
  jid=$(sbatch --parsable --requeue ${dep_arg} \
    --job-name="${run_name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" \
    --time="${RL_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${RL_CPUS}" \
    --gpus=1 --mem="${RL_MEM}" \
    --output="${LOG_ROOT}/${run_name}_%j.out" \
    --error="${LOG_ROOT}/${run_name}_%j.err" \
    --wrap="${wrap_cmd}")
  echo "  RL  ${run_name}  jid=${jid}  dep=${sft_jid:-none}"
}

echo "=== §4.6 DNA epoch-split ablation ==="
echo "  sweep:        ${SWEEP_ID}"
echo "  ckpt root:    ${SWEEP_ROOT}"
echo "  log dir:      ${LOG_ROOT}"
echo "  CPT init:     ${CPT_FINAL_DIR}"
echo "  pairs:        ${PAIRS}"
echo "  partition:    ${PARTITION}  constraint=${CONSTRAINT}"
echo ""

for pair in ${PAIRS}; do
  sft_e="${pair%:*}"
  rl_e="${pair##*:}"
  sum=$(( sft_e + rl_e ))
  if [ "${sum}" -ne 8 ]; then
    echo "  [skip] pair ${pair} sums to ${sum}, expected 8"
    continue
  fi
  pair_tag="sft${sft_e}-rl${rl_e}"

  sft_jid=""
  sft_ckpt_root=""
  if [ "${sft_e}" -gt 0 ]; then
    sft_jid=$(submit_sft "${sft_e}" "${pair_tag}")
    sft_ckpt_root="${SWEEP_ROOT}/sft/${pair_tag}"
  fi
  if [ "${rl_e}" -gt 0 ]; then
    submit_rl "${rl_e}" "${pair_tag}" "${sft_jid}" "${sft_ckpt_root}"
  fi
done

echo ""
echo "Submitted. Monitor: squeue -u \$USER --name 'section46-dna-*'"
echo "Sweep root (rm -rf to cleanup): ${SWEEP_ROOT}"
