#!/bin/bash
# DNA LoRA RL eval — eval_only on each (sft_rank, rl_rank, backbone) cell of
# the dna_lora_20260425_030646 sweep. Picks latest checkpoint-N at runtime.
#
# Usage:
#   bash BioReason/submit_dna_lora_rl_eval.sh                  # all done cells
#   CELLS="qwen3_1p7b/sft64/r4 qwen3_4b/sft16/r4" bash submit... # subset
#
# Uses train_grpo.py --eval_only with the same num_generations=2 / batch=2
# fix as submit_dna_rl_eval.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
SCRATCH_DATA="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH_DATA}/genomics}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "Missing PYTHON: ${PYTHON}" >&2; exit 1; }

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-16}"

SWEEP_ID="${SWEEP_ID:-dna_lora_20260425_030646}"
SWEEP_ROOT="${SWEEP_ROOT:-${CHECKPOINT_DIR}/${SWEEP_ID}}"
LOG_ROOT="${LOG_ROOT:-${WORKING_DIR}/logs/${SWEEP_ID}/rl_eval}"
mkdir -p "${LOG_ROOT}"

# Default: walk all RL ckpt dirs that have a checkpoint-* (training done).
DEFAULT_CELLS=""
for d in "${SWEEP_ROOT}/rl"/*/sft*/r*/; do
  [ -d "$d" ] || continue
  if ls "${d}"checkpoint-* >/dev/null 2>&1; then
    rel="${d#${SWEEP_ROOT}/rl/}"; rel="${rel%/}"
    DEFAULT_CELLS+="$rel "
  fi
done
CELLS="${CELLS:-${DEFAULT_CELLS}}"

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
  local cell="$1" split="$2"
  # cell = qwen3_1p7b/sft64/r4 → tag, sft_rank, rl_rank
  local tag="${cell%%/*}" rest="${cell#*/}"
  local sft_dir="${rest%%/*}" rl_dir="${rest##*/}"
  local sft_rank="${sft_dir#sft}" rl_rank="${rl_dir#r}"
  local rl_alpha=$(( rl_rank * 2 ))
  local rl_root="${SWEEP_ROOT}/rl/${cell}"
  local sft_root="${SWEEP_ROOT}/sft/${tag}/r${sft_rank}"
  local text_model
  if [ "$tag" = "qwen3_1p7b" ]; then text_model="Qwen/Qwen3-1.7B"; else text_model="Qwen/Qwen3-4B"; fi
  local run_name="dna-lora-rl-eval-${tag}-sft${sft_rank}-r${rl_rank}-${split}"

  local sel='BEST_SFT_CKPT=$(ls -1td "'"${sft_root}"'"/*/last.ckpt 2>/dev/null | head -1 || ls -1td "'"${sft_root}"'"/*/*.ckpt 2>/dev/null | head -1); EVAL_CKPT=$(ls -1dt "'"${rl_root}"'"/checkpoint-* 2>/dev/null | head -1)'
  local wrap="${ENV_PRELUDE} ${sel}; echo \"[lora-rl-eval] sft=\$BEST_SFT_CKPT rl=\$EVAL_CKPT\"; \
stdbuf -oL -eL \"${PYTHON}\" train_grpo.py --text_model_name \"${text_model}\" --dna_model_name evo2_1b_base --cache_dir \"${CACHE_DIR}\" --sft_checkpoint \"\$BEST_SFT_CKPT\" --peft_ckpt False --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 --kegg_data_dir_local \"${GENOMICS_DIR}\" --lora_r ${rl_rank} --lora_alpha ${rl_alpha} --lora_dropout 0.05 --gradient_accumulation_steps 8 --gradient_checkpointing True --max_steps 0 --num_train_epochs 1 --max_completion_length 800 --num_generations 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --beta 1e-4 --run_name \"${run_name}\" --learning_rate 1e-5 --logging_steps 1 --temperature 1 --top_p 0.95 --top_k 20 --output_dir \"${rl_root}/_eval_${split}\" --eval_only True --eval_split \"${split}\" --eval_ckpt \"\$EVAL_CKPT\" --use_vllm False --bf16 True --report_to wandb"

  jid=$(sbatch --parsable --requeue \
    --job-name="${run_name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" --time="${TIME}" --nodes=1 --ntasks=1 \
    --cpus-per-task="${CPUS}" --gpus=1 --mem="${MEM}" \
    --output="${LOG_ROOT}/${run_name}_%j.out" \
    --error="${LOG_ROOT}/${run_name}_%j.err" \
    --wrap="${wrap}")
  echo "  ${run_name}  jid=${jid}"
}

echo "=== DNA LoRA RL eval submission ==="
echo "  sweep:    ${SWEEP_ID}"
echo "  log:      ${LOG_ROOT}"
echo "  cells:    ${CELLS}"
echo ""
for cell in ${CELLS}; do
  submit_eval "${cell}" id
  submit_eval "${cell}" ood
done
