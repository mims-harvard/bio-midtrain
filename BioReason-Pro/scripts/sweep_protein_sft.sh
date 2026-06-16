#!/bin/bash
# Sweep script for protein-only SFT (no GO graph embeddings).
# Launches separate SLURM jobs for:
#   - Epoch sweep: 20% data x {1, 2, 4, 8, 16, 32} epochs
#   - Data fraction sweep: 1 epoch x {20, 40, 60, 80, 100}% data

set -euo pipefail

# ============================================================
# USER CONFIGURATION — adjust these paths
# ============================================================
CONDA_ENV="$SCRATCH/envs/bio/bin/activate"
ROOT_DIR="${PROJECT_ROOT:-/n/home07/hanlinzhang/projects/evo_omics}/BioReason-Pro"

REASONING_SFT_DATASET="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein"
CACHE_DIR="$REASONING_SFT_DATASET/cache"               # e.g. /n/holylfs06/.../cache
CHECKPOINT_BASE_DIR="$REASONING_SFT_DATASET/checkpoints"     # e.g. /n/holylfs06/.../checkpoints
WANDB_PROJECT="esm3"

TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3_sm_open_v1"

MAX_LENGTH_TEXT=10000
MAX_LENGTH_PROTEIN=2000
LORA_RANK=128
LORA_ALPHA=256
LEARNING_RATE=1e-4
SEED=23
NUM_WORKERS=0
VAL_CHECK_INTERVAL=1.0
EPOCH_SWEEP_LIST="${EPOCH_SWEEP_LIST:-1 2 4 8 16 32}"
SKIP_COMPLETED_TASKS="${SKIP_COMPLETED_TASKS:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-true}"
DRY_RUN="${DRY_RUN:-false}"
# ============================================================

latest_err_log() {
    local run_name="$1"
    local log_dir="${2:-outputs/slurm}"
    ls -1t ${log_dir}/${run_name}_*.err 2>/dev/null | head -n 1 || true
}

is_task_completed() {
    local run_name="$1"
    local max_epochs="$2"
    local log_dir="${3:-outputs/slurm}"
    local err_log
    local checkpoint_dir="${CHECKPOINT_BASE_DIR:+${CHECKPOINT_BASE_DIR}/${run_name}}"
    local target_epoch_ckpt
    target_epoch_ckpt=$(printf "%s/%s-epoch=%02d.ckpt" "${checkpoint_dir}" "${run_name}" "$((max_epochs - 1))")

    # Primary signal: final epoch checkpoint exists.
    if [ -n "${checkpoint_dir}" ] && [ -f "${target_epoch_ckpt}" ]; then
        return 0
    fi

    err_log=$(latest_err_log "$run_name" "$log_dir")
    if [ -z "$err_log" ]; then
        return 1
    fi

    # Fallback signal: latest log explicitly reports reaching configured max epochs.
    tr -d '\000' < "$err_log" | grep -q "max_epochs=${max_epochs}.*reached"
}

submit_job() {
    local run_name="$1"
    local max_epochs="$2"
    local train_fraction="$3"
    local log_dir="${4:-outputs/slurm}"
    local checkpoint_dir="${CHECKPOINT_BASE_DIR:+${CHECKPOINT_BASE_DIR}/${run_name}}"
    local ckpt_arg=""

    if [ "${SKIP_COMPLETED_TASKS}" = "true" ] && is_task_completed "$run_name" "$max_epochs" "$log_dir"; then
        echo "Skipping completed task: ${run_name} (max_epochs=${max_epochs})"
        return 0
    fi

    if [ "${RESUME_FROM_CHECKPOINT}" = "true" ] && [ -n "${checkpoint_dir}" ]; then
        if [ -f "${checkpoint_dir}/last.ckpt" ]; then
            ckpt_arg="--ckpt_path ${checkpoint_dir}/last.ckpt"
            echo "Resuming from last checkpoint: ${checkpoint_dir}/last.ckpt"
        else
            local latest_resume_ckpt
            latest_resume_ckpt=$(ls -1t "${checkpoint_dir}/${run_name}"-recent-*.ckpt 2>/dev/null | head -n 1 || true)
            if [ -z "$latest_resume_ckpt" ]; then
                latest_resume_ckpt=$(ls -1t "${checkpoint_dir}/${run_name}"-epoch=*.ckpt 2>/dev/null | head -n 1 || true)
            fi
            if [ -n "$latest_resume_ckpt" ]; then
                ckpt_arg="--ckpt_path ${latest_resume_ckpt}"
                echo "Resuming from latest checkpoint: ${latest_resume_ckpt}"
            else
                echo "No checkpoint found in ${checkpoint_dir}; starting fresh."
            fi
        fi
    else
        echo "Starting from scratch for ${run_name} (RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT})."
    fi

    if [ "${DRY_RUN}" = "true" ]; then
        echo "DRY_RUN: would submit ${run_name} (epochs=${max_epochs}, data_fraction=${train_fraction}, ckpt_arg='${ckpt_arg}')"
        return 0
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH -J ${run_name}
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3-00:00
#SBATCH -p kempner_h100
#SBATCH --mem=128GB
#SBATCH --constraint=h100
#SBATCH --account=kempner_sham_lab
#SBATCH -o ${log_dir}/${run_name}_%j.out
#SBATCH -e ${log_dir}/${run_name}_%j.err

source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
mkdir -p \$WANDB_DIR \$TORCH_HOME \$TRITON_CACHE_DIR
cd ${ROOT_DIR}

mkdir -p ${log_dir}
${checkpoint_dir:+mkdir -p ${checkpoint_dir}}

python train_protein_llm.py \\
    --dataset_type hf_reasoning \\
    --reasoning_sft_dataset ${REASONING_SFT_DATASET} \\
    --text_model_name ${TEXT_MODEL_NAME} \\
    --protein_model_name ${PROTEIN_MODEL_NAME} \\
    --attn_implementation flash_attention_2 \\
    --use_unsloth False \\
    --model_type protein-llm \\
    --max_length_text ${MAX_LENGTH_TEXT} \\
    --max_length_protein ${MAX_LENGTH_PROTEIN} \\
    --lora_rank ${LORA_RANK} \\
    --lora_alpha ${LORA_ALPHA} \\
    --lora_dropout 0 \\
    --learning_rate ${LEARNING_RATE} \\
    --warmup_ratio 0.05 \\
    --max_epochs ${max_epochs} \\
    --train_data_fraction ${train_fraction} \\
    --batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --num_workers ${NUM_WORKERS} \\
    --num_gpus 1 \\
    --num_nodes 1 \\
    --weight_decay 0.01 \\
    --seed ${SEED} \\
    --val_split_ratio 0.1 \\
    --val_check_interval ${VAL_CHECK_INTERVAL} \\
    --log_every_n_steps 10 \\
    --num_sanity_val_steps 0 \\
    --save_top_k 1 \\
    --checkpoint_start_epoch 0 \\
    --protein_embedding_layer 37 \\
    --protein_model_finetune False \\
    --go_model_finetune False \\
    --unified_go_encoder False \\
    --wandb_project ${WANDB_PROJECT} \\
    --run_name ${run_name} \\
    ${checkpoint_dir:+--checkpoint_dir ${checkpoint_dir}} \\
    ${ckpt_arg:+${ckpt_arg} \\}
    --debug False
EOF
    echo "Submitted: ${run_name}  (epochs=${max_epochs}, data_fraction=${train_fraction})"
}

# ============================================================
# Epoch sweep — 20% data, vary epochs
# ============================================================
# echo "=== Epoch sweep (20% data) ==="
# for EPOCHS in ${EPOCH_SWEEP_LIST}; do
#     RUN_NAME="protein-sft-epochs${EPOCHS}-data20pct"
#     submit_job "$RUN_NAME" "$EPOCHS" "0.20"
# done

# ============================================================
# Data fraction sweep — 1 epoch, vary data %
# Skip 20% since it's already covered above (epoch=1 case)
# ============================================================
echo "=== Data fraction sweep (1 epoch) ==="
for PCT in 20 40 60 80 100; do
    FRACTION=$(echo "scale=2; ${PCT}/100" | bc)
    RUN_NAME="protein-sft-epochs1-data${PCT}pct"
    submit_job "$RUN_NAME" "1" "$FRACTION" "outputs/slurm/data"
done

echo ""
if [ "${DRY_RUN}" = "true" ]; then
    echo "Dry run complete. No jobs were submitted."
else
    echo "All jobs submitted. Monitor with: squeue -u \$USER"
fi
