#!/bin/bash
# §4.6 DNA RL eval submitter. For each (RL output_dir, training_jid) pair:
#   - submit id eval on val_dataloader (id_test_network_split.csv)
#   - submit ood eval on test_dataloader (ood_test_network_split.csv)
# Each eval auto-picks the latest checkpoint-N under the RL output_dir at runtime
# and runs train_grpo.py --eval_only.
#
# Usage:
#   bash BioReason/submit_dna_rl_eval.sh                     # fire all 8 §4.6 cells
#   PAIRS="sft7-rl1:10006023" bash submit_dna_rl_eval.sh     # subset
#
# Env: SWEEP_ROOT (default: section46_dna_20260504_162156),
#      TEXT_MODEL (default: CPT 1.7B final), DNA_MODEL (default: evo2_1b_base),
#      PARTITION (default: kempner_h100), TIME (default: 06:00:00),
#      MEM (default: 80G), CPUS (default: 16).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
SCRATCH_DATA="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH_DATA}/genomics}"

CPT_FINAL_DIR="${CPT_FINAL_DIR:-${CHECKPOINT_DIR}/drive/cpt_ffw_20260424_165721/m-Qwen_Qwen3-1.7B_lr-3e-4_ga-128_len-1024/final}"
TEXT_MODEL="${TEXT_MODEL:-${CPT_FINAL_DIR}}"
DNA_MODEL="${DNA_MODEL:-evo2_1b_base}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "Missing PYTHON: ${PYTHON}" >&2; exit 1; }
for f in "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing CSV: ${f}" >&2; exit 1; }
done

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
TIME="${TIME:-06:00:00}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-16}"

SWEEP_ID="${SWEEP_ID:-section46_dna_20260504_162156}"
SWEEP_ROOT="${SWEEP_ROOT:-${CHECKPOINT_DIR}/${SWEEP_ID}}"
LOG_ROOT="${LOG_ROOT:-${WORKING_DIR}/logs/${SWEEP_ID}/eval}"
mkdir -p "${LOG_ROOT}"

# pair_tag:training_jid (jid empty = no afterany dep).
# 8 DNA §4.6 cells; sft6-rl2 (10006022) and sft7-rl1 (10006023) already COMPLETED.
PAIRS="${PAIRS:-sft0-rl8:10006015 sft1-rl7:10006016 sft2-rl6:10006017 sft3-rl5:10006018 sft4-rl4:10006020 sft5-rl3:10006021 sft6-rl2: sft7-rl1:}"

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

submit_eval() {
  local pair_tag="$1" split="$2" train_jid="$3"
  local rl_dir="${SWEEP_ROOT}/rl/${pair_tag}"
  local run_name="section46-dna-rl-eval-${pair_tag}-${split}"
  local dep=""
  [ -n "${train_jid}" ] && dep="--dependency=afterany:${train_jid}"

  # Resolve latest checkpoint at runtime (job may run after dep clears).
  local resolve='EVAL_CKPT=$(ls -1dt "'"${rl_dir}"'"/checkpoint-* 2>/dev/null | head -1); echo "[dna-rl-eval] eval_ckpt=$EVAL_CKPT"'
  local wrap="${ENV_PRELUDE} ${resolve}; \
stdbuf -oL -eL \"${PYTHON}\" train_grpo.py --text_model_name \"${TEXT_MODEL}\" --dna_model_name \"${DNA_MODEL}\" --cache_dir \"${CACHE_DIR}\" --sft_checkpoint \"${CPT_FINAL_DIR}\" --peft_ckpt False --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 --kegg_data_dir_local \"${GENOMICS_DIR}\" --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --gradient_accumulation_steps 8 --gradient_checkpointing True --max_steps 0 --num_train_epochs 1 --max_completion_length 800 --num_generations 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --beta 1e-4 --run_name \"${run_name}\" --learning_rate 1e-5 --logging_steps 1 --temperature 1 --top_p 0.95 --top_k 20 --output_dir \"${rl_dir}/_eval_${split}\" --eval_only True --eval_split \"${split}\" --eval_ckpt \"\$EVAL_CKPT\" --use_vllm False --bf16 True --report_to wandb"

  jid=$(sbatch --parsable --requeue ${dep} \
    --job-name="${run_name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" --time="${TIME}" --nodes=1 --ntasks=1 \
    --cpus-per-task="${CPUS}" --gpus=1 --mem="${MEM}" \
    --output="${LOG_ROOT}/${run_name}_%j.out" \
    --error="${LOG_ROOT}/${run_name}_%j.err" \
    --wrap="${wrap}")
  echo "  ${run_name}  jid=${jid}  dep=${train_jid:-none}"
}

echo "=== §4.6 DNA RL eval submission ==="
echo "  sweep:    ${SWEEP_ID}"
echo "  log:      ${LOG_ROOT}"
echo "  pairs:    ${PAIRS}"
echo ""

for entry in ${PAIRS}; do
  pair_tag="${entry%%:*}"
  train_jid="${entry##*:}"
  submit_eval "${pair_tag}" id  "${train_jid}"
  submit_eval "${pair_tag}" ood "${train_jid}"
done
