#!/bin/bash
set -euo pipefail

# DNA (KEGG) LoRA-rank sweeps @ 4 epochs.
#   SFT: lora_rank in {16, 64, 256} x {Qwen3-1.7B, Qwen3-4B}   -> 6 jobs
#   RL : lora_r   in {4, 16, 64}   over each SFT checkpoint   -> 18 jobs
# RL jobs depend on their matching SFT job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SFT_RANKS=(16 64 256)
RL_RANKS=(4 16 64)
MODELS=("Qwen/Qwen3-1.7B:qwen3_1p7b" "Qwen/Qwen3-4B:qwen3_4b")
SFT_EPOCHS="${SFT_EPOCHS:-4}"
RL_EPOCHS="${RL_EPOCHS:-4}"

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
DRIVE_ROOT="${DRIVE_ROOT:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}}"
GENOMICS_DIR="${GENOMICS_DIR:-${DRIVE_ROOT}/genomics}"
PYTHON="${PYTHON:-${SCRATCH:+${SCRATCH}/envs/bio/bin/python}}"
PYTHON="${PYTHON:-$(command -v python3)}"
[[ -x "${PYTHON}" ]] || { echo "No python at ${PYTHON}. Set PYTHON=..." >&2; exit 1; }
command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 1; }
for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  [[ -f "${f}" ]] || { echo "Missing drive CSV: ${f}" >&2; exit 1; }
done

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
SFT_TIME="${SFT_TIME:-1-12:00}"
RL_TIME="${RL_TIME:-1-12:00}"
CPUS="${CPUS:-24}"
SFT_MEM="${SFT_MEM:-60G}"
RL_MEM="${RL_MEM:-80G}"
# vortex flash_attn binaries are not compiled for A100-MIG slices; force
# h100|h200 by default to avoid 'no kernel image' fwd errors on that GPU class.
CONSTRAINT="${CONSTRAINT:-h100|h200}"
WANDB_MODE="${WANDB_MODE:-online}"

SWEEP_ID="${SWEEP_ID:-dna_lora_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${WORKING_DIR}/logs/lora_sweeps"

# ---- shared env prelude for the srun wrap ----
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

submit_sft() {
  local text_model="$1" model_tag="$2" rank="$3"
  local alpha=$(( rank * 2 ))
  local name="dna-sft-${model_tag}-r${rank}"
  local out="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}/r${rank}"
  local log="${WORKING_DIR}/logs/lora_sweeps/sft/${model_tag}"
  mkdir -p "${out}" "${log}"

  sbatch --parsable \
    --job-name="${name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${SFT_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --gpus=1 --mem="${SFT_MEM}" --constraint="${CONSTRAINT}" \
    --output="${log}/r${rank}_%j.out" --error="${log}/r${rank}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",SFT_EPOCHS="${SFT_EPOCHS}",LORA_RANK="${rank}",LORA_ALPHA="${alpha}",CACHE_DIR="${CACHE_DIR}",CHECKPOINT_ROOT="${out}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",WANDB_MODE="${WANDB_MODE}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap="${ENV_PRELUDE} stdbuf -oL -eL \"\$PYTHON\" train_dna_qwen.py \
      --cache_dir \"\$CACHE_DIR\" --text_model_name \"\$TEXT_MODEL\" \
      --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 \
      --max_epochs \"\$SFT_EPOCHS\" --num_gpus 1 --batch_size 1 \
      --model_type dna-llm --dataset_type kegg \
      --kegg_data_dir_local \"\$GENOMICS_DIR\" \
      --max_length_dna 2048 --truncate_dna_per_side 1024 \
      --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 \
      --merge_val_test_set True --return_answer_in_batch True \
      --lora_rank \"\$LORA_RANK\" --lora_alpha \"\$LORA_ALPHA\" \
      --checkpoint_dir \"\$CHECKPOINT_ROOT\" \
      --wandb_project BioReason-kegg-lora-sft"
}

submit_rl() {
  local text_model="$1" model_tag="$2" sft_rank="$3" rl_rank="$4" dep="$5"
  local rl_alpha=$(( rl_rank * 2 ))
  local name="dna-rl-${model_tag}-sft${sft_rank}-r${rl_rank}"
  local sft_root="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}/r${sft_rank}"
  local out="${CHECKPOINT_DIR}/${SWEEP_ID}/rl/${model_tag}/sft${sft_rank}/r${rl_rank}"
  local log="${WORKING_DIR}/logs/lora_sweeps/rl/${model_tag}"
  mkdir -p "${out}" "${log}"

  sbatch --parsable \
    --job-name="${name}" --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${RL_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --gpus=1 --mem="${RL_MEM}" --constraint="${CONSTRAINT}" \
    ${dep:+--dependency=afterok:${dep}} \
    --output="${log}/sft${sft_rank}_r${rl_rank}_%j.out" \
    --error="${log}/sft${sft_rank}_r${rl_rank}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",RL_EPOCHS="${RL_EPOCHS}",LORA_R="${rl_rank}",LORA_ALPHA="${rl_alpha}",CACHE_DIR="${CACHE_DIR}",SFT_ROOT="${sft_root}",RL_OUT="${out}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",WANDB_MODE="${WANDB_MODE}",NAME="${name}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap="${ENV_PRELUDE} BEST_SFT=\$(\"\$PYTHON\" - \"\$SFT_ROOT\" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pat = re.compile(r'val_loss_epoch=([0-9]+(?:\.[0-9]+)?)')
best=None; best_loss=None
for c in root.rglob('*.ckpt'):
    if c.name=='last.ckpt': continue
    m=pat.search(c.name)
    if not m: continue
    v=float(m.group(1))
    if best_loss is None or v<best_loss: best_loss=v; best=c
if best is None:
    last=sorted(root.rglob('last.ckpt'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not last: raise SystemExit(f'no ckpt under {root}')
    best=last[0]
print(best)
PY
); echo \"[rl] using SFT ckpt: \$BEST_SFT\"; \
    stdbuf -oL -eL \"\$PYTHON\" train_grpo.py \
      --text_model_name \"\$TEXT_MODEL\" --dna_model_name evo2_1b_base \
      --cache_dir \"\$CACHE_DIR\" --sft_checkpoint \"\$BEST_SFT\" --peft_ckpt False \
      --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 \
      --kegg_data_dir_local \"\$GENOMICS_DIR\" \
      --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
      --lora_r \"\$LORA_R\" --lora_alpha \"\$LORA_ALPHA\" --lora_dropout 0.05 \
      --gradient_accumulation_steps 8 --gradient_checkpointing True \
      --max_steps -1 --num_train_epochs \"\$RL_EPOCHS\" \
      --max_completion_length 800 --num_generations 8 \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
      --beta 1e-4 --learning_rate 1e-5 --logging_steps 1 \
      --temperature 1 --top_p 0.95 --top_k 20 \
      --output_dir \"\$RL_OUT\" --save_strategy epoch --save_total_limit 2 \
      --lr_scheduler_type cosine --warmup_ratio 0.03 \
      --log_completions True --use_vllm False --bf16 True \
      --resume_from_checkpoint False --report_to wandb \
      --run_name \"\$NAME\""
}

echo "SWEEP_ID=${SWEEP_ID}  (SFT epochs=${SFT_EPOCHS}, RL epochs=${RL_EPOCHS})"
if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  echo "=== SFT: 2 models x 3 ranks = 6 jobs ==="
  for spec in "${MODELS[@]}"; do
    tm="${spec%%:*}"; tag="${spec##*:}"
    for r in "${SFT_RANKS[@]}"; do
      jid=$(submit_sft "${tm}" "${tag}" "${r}")
      echo "  SFT ${tag} r=${r} -> ${jid}"
      eval "JID_${tag//[^a-zA-Z0-9]/_}_${r}=${jid}"
    done
  done
else
  echo "=== SFT skipped (SKIP_SFT=1); RL jobs will run with no dependency ==="
fi

echo "=== RL: 2 models x 3 SFT-ranks x 3 RL-ranks = 18 jobs ==="
for spec in "${MODELS[@]}"; do
  tm="${spec%%:*}"; tag="${spec##*:}"
  safe="${tag//[^a-zA-Z0-9]/_}"
  for sr in "${SFT_RANKS[@]}"; do
    dep_var="JID_${safe}_${sr}"; dep="${!dep_var:-}"
    for rr in "${RL_RANKS[@]}"; do
      jid=$(submit_rl "${tm}" "${tag}" "${sr}" "${rr}" "${dep}")
      echo "  RL  ${tag} sft_r=${sr} rl_r=${rr} -> ${jid} (after ${dep})"
    done
  done
done

echo "Done. SFT root: ${CHECKPOINT_DIR}/${SWEEP_ID}/sft   RL root: ${CHECKPOINT_DIR}/${SWEEP_ID}/rl"
