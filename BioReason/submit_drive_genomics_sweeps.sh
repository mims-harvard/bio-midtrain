#!/bin/bash
set -euo pipefail

# Submit full drive sweeps for BioReason on local genomics CSV data.
#
# SFT sweep:
#   epochs = {1,2,4,8,16,32} x {Qwen3-1.7B, Qwen3-4B} => 12 jobs
# RL sweep:
#   epochs = {1,2,4,8,16,32} x {Qwen3-1.7B, Qwen3-4B} => 12 jobs
#   each RL job depends on all SFT jobs of the same text model and auto-selects
#   the best SFT checkpoint from this sweep (by SFT_SELECT_METRIC; default accuracy).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -ra EPOCHS <<<"${EPOCHS_OVERRIDE:-1 2 4 8 16 32}"
# RL epochs hard-capped at 8 (override is honored but values >8 are dropped),
# matching BioReason-Pro/scripts/sweep_protein_grpo.sh.
if [[ -n "${RL_EPOCHS_OVERRIDE:-}" ]]; then
  read -ra _RL_RAW <<<"${RL_EPOCHS_OVERRIDE}"
else
  _RL_RAW=("${EPOCHS[@]}")
fi
RL_EPOCHS_LIST=()
for _e in "${_RL_RAW[@]}"; do (( _e <= 8 )) && RL_EPOCHS_LIST+=("${_e}"); done
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -ra MODELS <<<"${MODELS_OVERRIDE}"
else
  MODELS=("Qwen/Qwen3-1.7B:qwen3_1p7b" "Qwen/Qwen3-4B:qwen3_4b")
fi

USER_NAME="${USER_NAME:-$(whoami)}"
CACHE_DIR="${CACHE_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER_NAME}/evo_tfm/BioReason/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
SCRATCH="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH}/genomics}"

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

PARTITION="${PARTITION:-kempner}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
SFT_TIME="${SFT_TIME:-7-00:00}"
RL_TIME="${RL_TIME:-7-00:00}"
REQUEUE_FLAG="${REQUEUE_FLAG:---requeue}"
SFT_CPUS="${SFT_CPUS:-24}"
RL_CPUS="${RL_CPUS:-24}"
SFT_MEM="${SFT_MEM:-60G}"
RL_MEM="${RL_MEM:-80G}"
SFT_GPUS="${SFT_GPUS:-1}"
RL_GPUS="${RL_GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
SFT_SELECT_METRIC="${SFT_SELECT_METRIC:-accuracy}"

# Keep one sweep-id across all submitted jobs, so RL only scans this run.
SWEEP_ID="${SWEEP_ID:-drive_genomics_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${WORKING_DIR}/logs/drive/sweeps"

submit_sft_job() {
  local text_model="$1"
  local model_tag="$2"
  local sft_epochs="$3"

  local sft_root="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}/e${sft_epochs}"
  local log_root="${WORKING_DIR}/logs/drive/sweeps/sft/${model_tag}"
  mkdir -p "${sft_root}" "${log_root}"

  # Reuse active job with the same name to avoid duplicate submission.
  local active_job_id
  active_job_id=$(squeue -h -u "${USER_NAME}" --states=PD,R,CF,S,RS --name "drive-sft-${model_tag}-e${sft_epochs}" -o "%i" | head -n 1 || true)
  if [[ -n "${active_job_id}" && "${active_job_id}" =~ ^[0-9]+$ ]]; then
    echo "REUSE:${active_job_id}"
    return 0
  fi

  # Skip SFT if this epoch already finished for current SWEEP_ID.
  local latest_log
  latest_log=$(ls -1t "${log_root}/e${sft_epochs}_"*.out 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest_log}" ]] && grep -qE '^Accuracy:\s*[0-9]+(\.[0-9]+)?\s*$' "${latest_log}"; then
    echo "SKIP"
    return 0
  fi

  local sbatch_out
  sbatch_out=$(sbatch \
    --job-name="drive-sft-${model_tag}-e${sft_epochs}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${SFT_TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SFT_CPUS}" \
    --gpus="${SFT_GPUS}" \
    --mem="${SFT_MEM}" \
    ${REQUEUE_FLAG} \
    --output="${log_root}/e${sft_epochs}_%j.out" \
    --error="${log_root}/e${sft_epochs}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",SFT_EPOCHS="${sft_epochs}",CACHE_DIR="${CACHE_DIR}",CHECKPOINT_ROOT="${sft_root}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",WANDB_MODE="${WANDB_MODE}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap='set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12; if [ -n "${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi; cd "$WORKING_DIR"; export HF_DATASETS_CACHE="$CACHE_DIR/datasets"; export HF_HOME="$CACHE_DIR/hf_home"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"; export XDG_CACHE_HOME="$CACHE_DIR/xdg"; export TORCH_HOME="$CACHE_DIR/torch"; export TRITON_CACHE_DIR="$CACHE_DIR/triton"; export WANDB_DIR="$CACHE_DIR/wandb"; export WANDB_CACHE_DIR="$CACHE_DIR/wandb_cache"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"; export PIP_CACHE_DIR="$CACHE_DIR/pip"; export MPLCONFIGDIR="$CACHE_DIR/mpl"; export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"; mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"; export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1; export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"; RESUME_CKPT=$(ls -1t "$CHECKPOINT_ROOT"/*/last.ckpt 2>/dev/null | head -1 || true); RESUME_OPT=""; if [ -n "$RESUME_CKPT" ] && [ -e "$RESUME_CKPT" ]; then RESUME_OPT="--ckpt_path $RESUME_CKPT"; echo "[sft-resume] $RESUME_CKPT"; fi; stdbuf -oL -eL "$PYTHON" train_dna_qwen.py --cache_dir "$CACHE_DIR" --text_model_name "$TEXT_MODEL" --dna_model_name evo2_1b_base --strategy deepspeed_stage_2 --max_epochs "$SFT_EPOCHS" --num_gpus 1 --batch_size 1 --model_type dna-llm --dataset_type kegg --kegg_data_dir_local "$GENOMICS_DIR" --max_length_dna 2048 --truncate_dna_per_side 1024 --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --merge_val_test_set True --return_answer_in_batch True --checkpoint_dir "$CHECKPOINT_ROOT" --wandb_project "BioReason-drive-genomics-sft" $RESUME_OPT')

  local job_id
  job_id=$(awk '{print $NF}' <<<"${sbatch_out}")
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse SFT job id from sbatch output: ${sbatch_out}" >&2
    exit 1
  fi

  echo "${job_id}"
}

submit_rl_job() {
  local text_model="$1"
  local model_tag="$2"
  local rl_epochs="$3"
  local dependency_ids="$4"

  local sft_scan_root="${CHECKPOINT_DIR}/${SWEEP_ID}/sft/${model_tag}"
  local sft_log_root="${WORKING_DIR}/logs/drive/sweeps/sft/${model_tag}"
  local rl_out="${CHECKPOINT_DIR}/${SWEEP_ID}/rl/${model_tag}/e${rl_epochs}"
  local log_root="${WORKING_DIR}/logs/drive/sweeps/rl/${model_tag}"
  mkdir -p "${rl_out}" "${log_root}"

  local dependency_opt=""
  if [[ -n "${dependency_ids}" ]]; then
    dependency_opt="--dependency=afterany:${dependency_ids}"
  fi

  local sbatch_out
  sbatch_out=$(sbatch \
    --job-name="drive-rl-${model_tag}-e${rl_epochs}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${RL_TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${RL_CPUS}" \
    --gpus="${RL_GPUS}" \
    --mem="${RL_MEM}" \
    ${REQUEUE_FLAG} \
    ${dependency_opt:+${dependency_opt}} \
    --output="${log_root}/e${rl_epochs}_%j.out" \
    --error="${log_root}/e${rl_epochs}_%j.err" \
    --export=ALL,TEXT_MODEL="${text_model}",MODEL_TAG="${model_tag}",RL_EPOCHS="${rl_epochs}",CACHE_DIR="${CACHE_DIR}",WORKING_DIR="${WORKING_DIR}",PYTHON="${PYTHON}",SFT_SCAN_ROOT="${sft_scan_root}",SFT_LOG_ROOT="${sft_log_root}",SFT_SELECT_METRIC="${SFT_SELECT_METRIC}",RL_OUT="${rl_out}",WANDB_MODE="${WANDB_MODE}",GENOMICS_DIR="${GENOMICS_DIR}" \
    --wrap='set -eu; module load cuda/12.4; module load gcc/12.2.0-fasrc01; module load cudnn/9.10.2.21_cuda12; if [ -n "${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi; cd "$WORKING_DIR"; export HF_DATASETS_CACHE="$CACHE_DIR/datasets"; export HF_HOME="$CACHE_DIR/hf_home"; export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"; export XDG_CACHE_HOME="$CACHE_DIR/xdg"; export TORCH_HOME="$CACHE_DIR/torch"; export TRITON_CACHE_DIR="$CACHE_DIR/triton"; export WANDB_DIR="$CACHE_DIR/wandb"; export WANDB_CACHE_DIR="$CACHE_DIR/wandb_cache"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"; export PIP_CACHE_DIR="$CACHE_DIR/pip"; export MPLCONFIGDIR="$CACHE_DIR/mpl"; export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/bioreason_tmp}"; mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"; export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1; export TORCHDYNAMO_DISABLE=1; export NVTE_TORCH_COMPILE=0; export DEEPSPEED_NO_MPI=1; export RANK=0; export WORLD_SIZE=1; export LOCAL_RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"; BEST_SFT_CKPT=$("$PYTHON" - "$SFT_SCAN_ROOT" "$SFT_LOG_ROOT" "$SFT_SELECT_METRIC" <<"PY"
import pathlib
import re
import sys

sft_root = pathlib.Path(sys.argv[1])
sft_log_root = pathlib.Path(sys.argv[2])
select_metric = sys.argv[3].strip().lower()

if not sft_root.exists():
    raise SystemExit(f"SFT root does not exist: {sft_root}")

scan_root = sft_root

if select_metric in {"accuracy", "acc", "test_accuracy"} and sft_log_root.exists():
    accuracy_re = re.compile(r"^Accuracy:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    exp_re = re.compile(r"^(e[0-9]+)_\d+\.out$")
    best_exp = None
    best_acc = None

    for out_path in sorted(sft_log_root.glob("e*_*.out")):
        m_exp = exp_re.match(out_path.name)
        if not m_exp:
            continue
        exp_name = m_exp.group(1)
        final_acc = None
        with out_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_acc = accuracy_re.match(line.strip())
                if m_acc:
                    final_acc = float(m_acc.group(1))
        if final_acc is None:
            continue
        if best_acc is None or final_acc > best_acc:
            best_acc = final_acc
            best_exp = exp_name

    if best_exp is not None:
        candidate_root = sft_root / best_exp
        if candidate_root.exists():
            scan_root = candidate_root
            print(
                f"[drive-rl] SFT selection metric=accuracy best_exp={best_exp} best_acc={best_acc:.4f}",
                file=sys.stderr,
            )

val_loss_re = re.compile(r"val_loss_epoch=([0-9]+(?:\.[0-9]+)?)")
best_path = None
best_loss = None

for ckpt in scan_root.rglob("*.ckpt"):
    if ckpt.name == "last.ckpt":
        continue
    m = val_loss_re.search(ckpt.name)
    if not m:
        continue
    loss = float(m.group(1))
    if best_loss is None or loss < best_loss:
        best_loss = loss
        best_path = ckpt

if best_path is None:
    last_ckpts = sorted(scan_root.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not last_ckpts and scan_root != sft_root:
        last_ckpts = sorted(sft_root.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if last_ckpts:
            scan_root = sft_root
    if last_ckpts:
        best_path = last_ckpts[0]
    else:
        raise SystemExit(f"No checkpoint found under {scan_root}")

print(str(best_path))
PY
); echo "[drive-rl] Selected best SFT checkpoint: $BEST_SFT_CKPT"; stdbuf -oL -eL "$PYTHON" train_grpo.py --text_model_name "$TEXT_MODEL" --dna_model_name evo2_1b_base --cache_dir "$CACHE_DIR" --sft_checkpoint "$BEST_SFT_CKPT" --peft_ckpt False --dna_is_evo2 True --dna_embedding_layer blocks.20.mlp.l3 --truncate_dna_per_side 1024 --kegg_data_dir_local "$GENOMICS_DIR" --deepspeed grpo_trainer_lora_model/ds_config_stage2.json --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --gradient_accumulation_steps 8 --gradient_checkpointing True --max_steps -1 --num_train_epochs "$RL_EPOCHS" --max_completion_length 800 --num_generations 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --beta 1e-4 --run_name "drive-genomics-rl-${MODEL_TAG}-e${RL_EPOCHS}" --learning_rate 1e-5 --logging_steps 1 --temperature 1 --top_p 0.95 --top_k 20 --output_dir "$RL_OUT" --save_strategy epoch --save_total_limit 2 --lr_scheduler_type cosine --warmup_ratio 0.03 --log_completions True --use_vllm False --bf16 True --resume_from_checkpoint true --report_to wandb')

  local job_id
  job_id=$(awk '{print $NF}' <<<"${sbatch_out}")
  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Failed to parse RL job id from sbatch output: ${sbatch_out}" >&2
    exit 1
  fi

  echo "${job_id}"
}

declare -A SFT_DEP_BY_MODEL

echo "Submitting SFT sweep (12 jobs) with SWEEP_ID=${SWEEP_ID}"
for model_spec in "${MODELS[@]}"; do
  text_model="${model_spec%%:*}"
  model_tag="${model_spec##*:}"
  dep_ids=""

  for epoch in "${EPOCHS[@]}"; do
    jid=$(submit_sft_job "${text_model}" "${model_tag}" "${epoch}")
    if [[ "${jid}" =~ ^[0-9]+$ ]]; then
      if [[ -z "${dep_ids}" ]]; then
        dep_ids="${jid}"
      else
        dep_ids+="${dep_ids:+:}${jid}"
      fi
      echo "  SFT submitted: model=${model_tag} epochs=${epoch} job_id=${jid}"
    elif [[ "${jid}" =~ ^REUSE:([0-9]+)$ ]]; then
      local_running_id="${BASH_REMATCH[1]}"
      if [[ -z "${dep_ids}" ]]; then
        dep_ids="${local_running_id}"
      else
        dep_ids+="${dep_ids:+:}${local_running_id}"
      fi
      echo "  SFT reused:    model=${model_tag} epochs=${epoch} running_job_id=${local_running_id}"
    elif [[ "${jid}" == "SKIP" ]]; then
      echo "  SFT skipped:   model=${model_tag} epochs=${epoch} (already completed)"
    else
      echo "Unexpected submit_sft_job result: ${jid}" >&2
      exit 1
    fi
  done

  SFT_DEP_BY_MODEL["${model_tag}"]="${dep_ids}"
done

echo "Submitting RL sweep (${#RL_EPOCHS_LIST[@]} epochs x ${#MODELS[@]} models), each depends on same-model SFT jobs"
for model_spec in "${MODELS[@]}"; do
  text_model="${model_spec%%:*}"
  model_tag="${model_spec##*:}"
  deps="${SFT_DEP_BY_MODEL[${model_tag}]}"

  for epoch in "${RL_EPOCHS_LIST[@]}"; do
    jid=$(submit_rl_job "${text_model}" "${model_tag}" "${epoch}" "${deps}")
    if [[ -n "${deps}" ]]; then
      echo "  RL submitted:  model=${model_tag} epochs=${epoch} job_id=${jid} depends_on=${deps}"
    else
      echo "  RL submitted:  model=${model_tag} epochs=${epoch} job_id=${jid} depends_on=<none>"
    fi
  done
done

echo "All jobs submitted successfully."
echo "Sweep ID: ${SWEEP_ID}"
echo "SFT checkpoints root: ${CHECKPOINT_DIR}/${SWEEP_ID}/sft"
echo "RL checkpoints root:  ${CHECKPOINT_DIR}/${SWEEP_ID}/rl"
