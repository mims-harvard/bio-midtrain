#!/bin/bash
set -euo pipefail

# Submit DNA data-fraction sweeps on local drive genomics CSV data.
#
# SFT sweep:
#   data_pct = {20,40,60,80,100} x {Qwen3-1.7B, Qwen3-4B} @ SFT_EPOCHS (default 4) => 10 jobs
# RL sweep:
#   data_pct = {20,40,60,80,100} x {Qwen3-1.7B, Qwen3-4B} @ RL_EPOCHS (default 2) => 10 jobs
#   each RL job depends on the matching-fraction SFT job (same model, same pct)
#
# All jobs use --kegg_data_dir_local "$GENOMICS_DIR" pointing at
# ${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/genomics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -ra DATA_PCTS <<<"${DATA_PCTS_OVERRIDE:-20 40 60 80 100}"
SFT_EPOCHS="${SFT_EPOCHS:-4}"
RL_EPOCHS="${RL_EPOCHS:-2}"
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -ra MODELS <<<"${MODELS_OVERRIDE}"
else
  MODELS=("Qwen/Qwen3-1.7B:qwen3_1p7b" "Qwen/Qwen3-4B:qwen3_4b")
fi

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
SCRATCH_DATA="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH_DATA}/genomics}"

PYTHON="${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  exit 1
fi
for f in "${GENOMICS_DIR}/train_network_split.csv" "${GENOMICS_DIR}/id_test_network_split.csv" "${GENOMICS_DIR}/ood_test_network_split.csv"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required data file: ${f}" >&2
    exit 1
  fi
done
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Please run this on a Slurm login node." >&2
  exit 1
fi

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_mzitnik_lab}"
SFT_TIME="${SFT_TIME:-3-00:00}"
RL_TIME="${RL_TIME:-3-00:00}"
SFT_CPUS="${SFT_CPUS:-24}"
RL_CPUS="${RL_CPUS:-24}"
SFT_MEM="${SFT_MEM:-60G}"
RL_MEM="${RL_MEM:-80G}"
SFT_GPUS="${SFT_GPUS:-1}"
RL_GPUS="${RL_GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
SFT_SELECT_METRIC="${SFT_SELECT_METRIC:-accuracy}"

SWEEP_ID="${SWEEP_ID:-drive_genomics_data_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${WORKING_DIR}/logs/drive/data_sweeps"

submit_sft_job() {
  local text_model="$1" model_tag="$2" pct="$3"
  local frac
  frac=$(awk -v p="$pct" 'BEGIN{ printf "%.4f", p/100.0 }')
  local sft_root="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}/d${pct}"
  local log_root="${WORKING_DIR}/logs/drive/data_sweeps/sft/${model_tag}"
  mkdir -p "${sft_root}" "${log_root}"

  local active_job_id
  active_job_id=$(squeue -h -u "${USER_NAME}" --states=PD,R,CF,S,RS --name "drive-sft-data-${model_tag}-d${pct}" -o "%i" | head -n 1 || true)
  if [[ -n "${active_job_id}" && "${active_job_id}" =~ ^[0-9]+$ ]]; then
    echo "REUSE:${active_job_id}"
    return 0
  fi

  local latest_log
  latest_log=$(ls -1t "${log_root}/d${pct}_"*.out 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest_log}" ]] && grep -qE '^Accuracy:\s*[0-9]+(\.[0-9]+)?\s*$' "${latest_log}"; then
    echo "SKIP"
    return 0
  fi

  local sbatch_out
  sbatch_out=$(sbatch --parsable \
    --job-name="drive-sft-data-${model_tag}-d${pct}" \
    --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${SFT_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${SFT_CPUS}" \
    --gpus="${SFT_GPUS}" --mem="${SFT_MEM}" \
    --output="${log_root}/d${pct}_%j.out" --error="${log_root}/d${pct}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",SFT_EPOCHS="${SFT_EPOCHS}",DATA_FRAC="${frac}",DATA_PCT="${pct}",CACHE_DIR="${CACHE_DIR}",CHECKPOINT_ROOT="${sft_root}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",WANDB_MODE="${WANDB_MODE}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap='set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12; if [ -n "${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi; cd "$WORKING_DIR"; export HF_DATASETS_CACHE="$CACHE_DIR/datasets"; export HF_HOME="$CACHE_DIR/hf_home"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"; export XDG_CACHE_HOME="$CACHE_DIR/xdg"; export TORCH_HOME="$CACHE_DIR/torch"; export TRITON_CACHE_DIR="$CACHE_DIR/triton"; export WANDB_DIR="$CACHE_DIR/wandb"; export WANDB_CACHE_DIR="$CACHE_DIR/wandb_cache"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"; export PIP_CACHE_DIR="$CACHE_DIR/pip"; export MPLCONFIGDIR="$CACHE_DIR/mpl"; export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"; mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"; export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1; export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"; stdbuf -oL -eL "$PYTHON" train_dna_qwen.py --cache_dir "$CACHE_DIR" --text_model_name "$TEXT_MODEL" --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 --max_epochs "$SFT_EPOCHS" --num_gpus 1 --batch_size 1 --model_type dna-llm --dataset_type kegg --kegg_data_dir_local "$GENOMICS_DIR" --train_data_fraction "$DATA_FRAC" --max_length_dna 2048 --truncate_dna_per_side 1024 --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --merge_val_test_set True --return_answer_in_batch True --checkpoint_dir "$CHECKPOINT_ROOT" --wandb_project "BioReason-drive-genomics-data-sft-'"${model_tag}"'-d'"${pct}"'"')

  local job_id="${sbatch_out}"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse SFT job id from sbatch output: ${sbatch_out}" >&2
    exit 1
  fi
  echo "${job_id}"
}

submit_rl_job() {
  local text_model="$1" model_tag="$2" pct="$3" dependency_id="$4"
  local frac
  frac=$(awk -v p="$pct" 'BEGIN{ printf "%.4f", p/100.0 }')
  local sft_scan_root="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}/d${pct}"
  local sft_log_root="${WORKING_DIR}/logs/drive/data_sweeps/sft/${model_tag}"
  local rl_out="${CHECKPOINT_DIR}/${SWEEP_ID}/rl/${model_tag}/d${pct}"
  local log_root="${WORKING_DIR}/logs/drive/data_sweeps/rl/${model_tag}"
  mkdir -p "${rl_out}" "${log_root}"

  local dep_opt=""
  if [[ -n "${dependency_id}" && "${dependency_id}" =~ ^[0-9]+$ ]]; then
    dep_opt="--dependency=afterany:${dependency_id}"
  fi

  local sbatch_out
  sbatch_out=$(sbatch --parsable \
    --job-name="drive-rl-data-${model_tag}-d${pct}" \
    --partition="${PARTITION}" --account="${ACCOUNT}" \
    --time="${RL_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${RL_CPUS}" \
    --gpus="${RL_GPUS}" --mem="${RL_MEM}" \
    ${dep_opt:+${dep_opt}} \
    --output="${log_root}/d${pct}_%j.out" --error="${log_root}/d${pct}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",MODEL_TAG="${model_tag}",RL_EPOCHS="${RL_EPOCHS}",DATA_FRAC="${frac}",DATA_PCT="${pct}",CACHE_DIR="${CACHE_DIR}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",SFT_SCAN_ROOT="${sft_scan_root}",SFT_LOG_ROOT="${sft_log_root}",SFT_SELECT_METRIC="${SFT_SELECT_METRIC}",RL_OUT="${rl_out}",WANDB_MODE="${WANDB_MODE}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap='set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12; if [ -n "${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi; cd "$WORKING_DIR"; export HF_DATASETS_CACHE="$CACHE_DIR/datasets"; export HF_HOME="$CACHE_DIR/hf_home"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"; export XDG_CACHE_HOME="$CACHE_DIR/xdg"; export TORCH_HOME="$CACHE_DIR/torch"; export TRITON_CACHE_DIR="$CACHE_DIR/triton"; export WANDB_DIR="$CACHE_DIR/wandb"; export WANDB_CACHE_DIR="$CACHE_DIR/wandb_cache"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"; export PIP_CACHE_DIR="$CACHE_DIR/pip"; export MPLCONFIGDIR="$CACHE_DIR/mpl"; export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"; mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"; export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1; export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"; BEST_SFT_CKPT=$("$PYTHON" - "$SFT_SCAN_ROOT" "$SFT_LOG_ROOT" "$SFT_SELECT_METRIC" <<"PY"
import pathlib, re, sys, json
scan_root, log_root, metric = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
ckpts = sorted(scan_root.glob("last.ckpt"))
if not ckpts:
    ckpts = sorted(scan_root.glob("**/last.ckpt"))
if not ckpts:
    ckpts = sorted(scan_root.glob("**/*.ckpt"))
if not ckpts:
    raise SystemExit(f"No checkpoint found under {scan_root}")
print(str(ckpts[-1]))
PY
); echo "[drive-data-rl] Selected best SFT checkpoint: $BEST_SFT_CKPT"; stdbuf -oL -eL "$PYTHON" train_grpo.py --text_model_name "$TEXT_MODEL" --dna_model_name evo2_1b_base --cache_dir "$CACHE_DIR" --sft_checkpoint "$BEST_SFT_CKPT" --peft_ckpt False --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 --kegg_data_dir_local "$GENOMICS_DIR" --train_data_fraction "$DATA_FRAC" --deepspeed grpo_trainer_lora_model/ds_config_stage2.json --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --gradient_accumulation_steps 4 --gradient_checkpointing True --max_steps -1 --num_train_epochs "$RL_EPOCHS" --max_completion_length 800 --num_generations 8 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --beta 1e-4 --run_name "drive-genomics-rl-data-'"${model_tag}"'-d'"${pct}"'" --learning_rate 1e-5 --logging_steps 1 --temperature 1 --top_p 0.95 --top_k 20 --output_dir "$RL_OUT" --save_strategy epoch --save_total_limit 2 --lr_scheduler_type cosine --warmup_ratio 0.03 --log_completions True --use_vllm False --bf16 True --resume_from_checkpoint False --report_to wandb')

  local job_id="${sbatch_out}"
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse RL job id from sbatch output: ${sbatch_out}" >&2
    exit 1
  fi
  echo "${job_id}"
}

echo "Submitting SFT data sweep (${#DATA_PCTS[@]} fractions x ${#MODELS[@]} models @ ${SFT_EPOCHS} epochs)"
declare -A SFT_JIDS
for entry in "${MODELS[@]}"; do
  text_model="${entry%%:*}"
  model_tag="${entry##*:}"
  for pct in "${DATA_PCTS[@]}"; do
    jid=$(submit_sft_job "${text_model}" "${model_tag}" "${pct}")
    case "${jid}" in
      SKIP)
        echo "  SFT skipped:   model=${model_tag} pct=${pct} (already completed)"
        SFT_JIDS["${model_tag}|${pct}"]=""
        ;;
      REUSE:*)
        local_running_id="${jid#REUSE:}"
        echo "  SFT reused:    model=${model_tag} pct=${pct} running_job_id=${local_running_id}"
        SFT_JIDS["${model_tag}|${pct}"]="${local_running_id}"
        ;;
      *)
        echo "  SFT submitted: model=${model_tag} pct=${pct} job_id=${jid}"
        SFT_JIDS["${model_tag}|${pct}"]="${jid}"
        ;;
    esac
  done
done

echo ""
echo "Submitting RL data sweep (${#DATA_PCTS[@]} fractions x ${#MODELS[@]} models @ ${RL_EPOCHS} epochs), each depends on its matching-fraction SFT job"
for entry in "${MODELS[@]}"; do
  text_model="${entry%%:*}"
  model_tag="${entry##*:}"
  for pct in "${DATA_PCTS[@]}"; do
    dep="${SFT_JIDS["${model_tag}|${pct}"]:-}"
    jid=$(submit_rl_job "${text_model}" "${model_tag}" "${pct}" "${dep}")
    if [[ -n "${dep}" ]]; then
      echo "  RL submitted:  model=${model_tag} pct=${pct} job_id=${jid} depends_on=${dep}"
    else
      echo "  RL submitted:  model=${model_tag} pct=${pct} job_id=${jid} depends_on=<none>"
    fi
  done
done

echo ""
echo "SWEEP_ID=${SWEEP_ID}"
echo "Monitor with: squeue -u \$USER --name='drive-sft-data-*,drive-rl-data-*'"
