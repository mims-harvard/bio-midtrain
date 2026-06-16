#!/bin/bash
# Smoke test for the two changes in submit_dna_lora_sweeps.sh + train_grpo.py:
#   A) SFT 1-epoch on drive CSVs via --kegg_data_dir_local
#   B) GRPO RL with rank mismatch (sft_r=16 -> rl_r=4) using an existing SFT ckpt,
#      bounded by --max_steps to keep wall time <= ~15 min.
# Submits both as small slurm jobs and prints their job ids.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
DRIVE_ROOT="${DRIVE_ROOT:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}}"
GENOMICS_DIR="${GENOMICS_DIR:-${DRIVE_ROOT}/genomics}"
SMOKE_OUT_ROOT="${SMOKE_OUT_ROOT:-${WORKING_DIR}/logs/lora_sweeps/smoke}"
SMOKE_CKPT_ROOT="${SMOKE_CKPT_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints/_smoke_$(date +%Y%m%d_%H%M%S)}"

# Existing SFT r=16 deepspeed ckpt (from prior dna_lora_20260424_123013 sweep).
SFT_R16_CKPT="${SFT_R16_CKPT:-${BIOREASON_WORK_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm}/BioReason/checkpoints/dna_lora_20260424_123013/sft/qwen3_1p7b/r16/BioReason-kegg-lora-sft-kegg-Qwen3-1.7B-20260424-123118/BioReason-kegg-lora-sft-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4651.ckpt}"

PYTHON="${PYTHON:-${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}}"
[[ -x "${PYTHON}" ]] || { echo "No python at ${PYTHON}" >&2; exit 1; }

PARTITION="${PARTITION:-kempner}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
CPUS="${CPUS:-16}"

for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing drive CSV: ${f}" >&2; exit 1; }
done
[[ -e "${SFT_R16_CKPT}" ]] || { echo "Missing SFT r16 ckpt: ${SFT_R16_CKPT}" >&2; exit 1; }

mkdir -p "${SMOKE_OUT_ROOT}" "${SMOKE_CKPT_ROOT}/sft" "${SMOKE_CKPT_ROOT}/rl"

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

echo "Submitting smoke A (SFT 1-epoch on drive CSV, Qwen3-1.7B r16) ..."
SFT_JID=$(sbatch --parsable \
  --job-name="smoke-sft-drive-r16" --partition="${PARTITION}" --account="${ACCOUNT}" \
  --time="00:45:00" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
  --gpus=1 --mem=60G \
  --output="${SMOKE_OUT_ROOT}/sft_drive_r16_%j.out" \
  --error="${SMOKE_OUT_ROOT}/sft_drive_r16_%j.err" \
  --export=ALL,WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",CACHE_DIR="${CACHE_DIR}",GENOMICS_DIR="${GENOMICS_DIR}",CKPT="${SMOKE_CKPT_ROOT}/sft" \
  --wrap="${ENV_PRELUDE} stdbuf -oL -eL \"\$PYTHON\" train_dna_qwen.py \
    --cache_dir \"\$CACHE_DIR\" --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 \
    --max_epochs 1 --num_gpus 1 --batch_size 1 \
    --model_type dna-llm --dataset_type kegg \
    --kegg_data_dir_local \"\$GENOMICS_DIR\" \
    --max_length_dna 2048 --truncate_dna_per_side 1024 \
    --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 \
    --merge_val_test_set True --return_answer_in_batch True \
    --lora_rank 16 --lora_alpha 32 \
    --checkpoint_dir \"\$CKPT\" \
    --wandb_project BioReason-kegg-lora-smoke")
echo "  smoke A SFT  jid=${SFT_JID}"

echo "Submitting smoke B (RL rank-mismatch sft_r=16 -> rl_r=4, max_steps=5) ..."
RL_JID=$(sbatch --parsable \
  --job-name="smoke-rl-r4-on-sft16" --partition="${PARTITION}" --account="${ACCOUNT}" \
  --time="00:45:00" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
  --gpus=1 --mem=80G \
  --output="${SMOKE_OUT_ROOT}/rl_r4_on_sft16_%j.out" \
  --error="${SMOKE_OUT_ROOT}/rl_r4_on_sft16_%j.err" \
  --export=ALL,WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",CACHE_DIR="${CACHE_DIR}",GENOMICS_DIR="${GENOMICS_DIR}",SFT_CKPT="${SFT_R16_CKPT}",RL_OUT="${SMOKE_CKPT_ROOT}/rl" \
  --wrap="${ENV_PRELUDE} stdbuf -oL -eL \"\$PYTHON\" train_grpo.py \
    --text_model_name Qwen/Qwen3-1.7B --dna_model_name evo2_1b_base \
    --cache_dir \"\$CACHE_DIR\" --sft_checkpoint \"\$SFT_CKPT\" --peft_ckpt False \
    --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 \
    --kegg_data_dir_local \"\$GENOMICS_DIR\" \
    --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
    --lora_r 4 --lora_alpha 8 --lora_dropout 0.05 \
    --gradient_accumulation_steps 1 --gradient_checkpointing True \
    --max_steps 5 --num_train_epochs 1 \
    --max_completion_length 256 --num_generations 4 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --beta 0.0 --learning_rate 1e-5 --logging_steps 1 \
    --temperature 1 --top_p 0.95 --top_k 20 \
    --output_dir \"\$RL_OUT\" --save_strategy no \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --log_completions True --use_vllm False --bf16 True \
    --resume_from_checkpoint False --report_to none \
    --run_name smoke-rl-r4-on-sft16")
echo "  smoke B RL   jid=${RL_JID}"

echo "Logs in: ${SMOKE_OUT_ROOT}"
echo "SFT_JID=${SFT_JID} RL_JID=${RL_JID}"
