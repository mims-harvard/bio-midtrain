#!/bin/bash
# Sweep script for protein-only GRPO (RL on standalone causal LM, optional LoRA).
# Mirrors sweep_protein_sft.sh. Launches separate SLURM jobs for:
#   - Epoch sweep: 20% data x {1, 2, 4, 8, 16, 32} epochs
#   - Data fraction sweep: 1 epoch x {20, 40, 60, 80, 100}% data

set -euo pipefail

# ============================================================
# USER CONFIGURATION — adjust these paths
# ============================================================
CONDA_ENV="$SCRATCH/envs/bio/bin/activate"
ROOT_DIR="${PROJECT_ROOT:-/n/home07/hanlinzhang/projects/evo_omics}/BioReason-Pro"

REASONING_SFT_DATASET="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein"
# IMPORTANT: keep CACHE_DIR OUTSIDE REASONING_SFT_DATASET. If the cache lives
# inside the data dir, load_dataset() recursively scans cached arrow files +
# the source csvs and fails with "Couldn't infer the same data file format
# for all splits" (train cached as arrow, test still csv).
CACHE_DIR="${SCRATCH:-/n/netscratch/kempner_sham_lab/Lab/$(whoami)}/bioreason_protein_cache"
CHECKPOINT_BASE_DIR="$REASONING_SFT_DATASET/checkpoints"     # e.g. /n/holylfs06/.../checkpoints
WANDB_PROJECT="esm3"

DATASET_NAME="${REASONING_SFT_DATASET}"  # local folder; load_dataset auto-detects csv/parquet files
DATASET_SPLIT="train"

MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
SFT_ADAPTER_PATH=""                                                # explicit adapter dir; overrides per-pct lookup
# When non-empty, each data-pct training picks
# ${SFT_ADAPTER_BASE}/protein-sft-epochs1-data${PCT}pct/sft_lora_adapter
# as its --sft_adapter_path. The epoch sweep (data20% fixed) always uses the
# data20 adapter. Set SFT_ADAPTER_BASE="" to disable warm-start (the original
# behaviour, which mode-collapses on data%≥40).
# Adapter dirs are produced by `BioReason-Pro/scripts/extract_sft_lora_adapter.py`.
SFT_ADAPTER_BASE="${SFT_ADAPTER_BASE:-}"
GO_OBO_PATH="${ROOT_DIR}/bioreason2/dataset/go-basic.obo"
IA_WEIGHTS_PATH=""                                                  # precomputed IA weights file; "" to skip

# Used to convert percentages -> --limit_examples (the GRPO trainer takes an
# absolute cap, not a fraction). Set this to the size of the gold-filtered
# train split. Override here when the dataset changes.
TOTAL_TRAIN_EXAMPLES=20000

LEARNING_RATE=3e-5
BATCH_SIZE=1   # was 2; full-FT Qwen3-4B + group=8 OOMs on H100 80GB at batch=2
GROUP_SIZE=2   # was 4; group=4 still OOMs at step 8 on H200 141GB (134GB allocated). g=2 verified stable.
MAX_NEW_TOKENS=256   # was 384; with policy gradient_checkpointing now enabled,
                      # 256 keeps activation+KV-cache peak well under 140 GB even
                      # if mode collapse temporarily spikes completion length.
TEMPERATURE=1.0
TOP_P=1.0
EPSILON_LOW=0.2
EPSILON_HIGH=0.28
KL_BETA="${KL_BETA:-1e-4}"   # overridable; β=1e-3 ablation is the §4.2 KL-strength axis
SAVE_EVERY_STEPS=200
SEED=23
# ============================================================

already_submitted() {
    local run_name="$1"
    # True if a job with this --job-name is currently RUNNING or PENDING in our queue.
    squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -Fxq "${run_name}"
}

submit_job() {
    local run_name="$1"
    local epochs="$2"
    local limit_examples="$3"
    local pct="$4"   # data percentage used to pick the matching SFT adapter
    local output_dir="${CHECKPOINT_BASE_DIR}/${run_name}"

    local adapter_path="${SFT_ADAPTER_PATH}"
    if [ -z "${adapter_path}" ] && [ -n "${SFT_ADAPTER_BASE}" ] && [ -n "${pct}" ]; then
        adapter_path="${SFT_ADAPTER_BASE}/protein-sft-epochs1-data${pct}pct/sft_lora_adapter"
        if [ ! -d "${adapter_path}" ]; then
            echo "[warn] SFT_ADAPTER_BASE set but adapter not found at ${adapter_path}; falling back to no warm-start."
            adapter_path=""
        fi
    fi

    if [ "${SKIP_RUNNING:-true}" = "true" ] && already_submitted "${run_name}"; then
        echo "Skipping ${run_name}: already in queue (RUNNING/PENDING)."
        return 0
    fi

    # Default: kempner_requeue (7-day, h200 141GB, preemptible -> auto-requeue).
    # Override env: PARTITION, TIME, CONSTRAINT (e.g. PARTITION=kempner_h100
    # TIME=1-12:00 CONSTRAINT=h100|h200 for shorter non-preemptible runs).
    local jid
    local _partition="${PARTITION:-kempner_requeue}"
    local _time="${TIME:-7-00:00}"
    local _constraint="${CONSTRAINT:-h200}"
    jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J ${run_name}
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t ${_time}
#SBATCH -p ${_partition}
#SBATCH --mem=128GB
#SBATCH --constraint=${_constraint}
#SBATCH --account=kempner_sham_lab
#SBATCH --requeue
#SBATCH -o outputs/slurm/${run_name}_%j.out
#SBATCH -e outputs/slurm/${run_name}_%j.err

module load cuda/12.4 || true
module load gcc/12.2.0-fasrc01 || true
module load cudnn/9.10.2.21_cuda12 || true
if [ -n "\${CUDNN_HOME:-}" ]; then export LD_LIBRARY_PATH="\${CUDNN_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"; fi
export CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda}
source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Without this, the training script's mean_reward / loss / kl print lines go
# to a 4KB stdout buffer and only flush hours later — so a healthy warm-start
# vs collapse cannot be distinguished mid-run. Cheap to set.
export PYTHONUNBUFFERED=1
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
export HF_DATASETS_CACHE=${CACHE_DIR}
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_NAME=${run_name}
mkdir -p \$WANDB_DIR \$TORCH_HOME \$TRITON_CACHE_DIR \$HF_DATASETS_CACHE
cd ${ROOT_DIR}

mkdir -p outputs/slurm
mkdir -p ${output_dir}

python train_protein_grpo.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_split ${DATASET_SPLIT} \
    --limit_examples ${limit_examples} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    ${adapter_path:+--sft_adapter_path ${adapter_path}} \
    --output_dir ${output_dir} \
    --go_obo_path ${GO_OBO_PATH} \
    ${IA_WEIGHTS_PATH:+--ia_weights_path ${IA_WEIGHTS_PATH}} \
    --interpro_in_prompt True \
    --ppi_in_prompt False \
    --ask_all_go_aspects True \
    --append_uniprot_suffix False \
    --go_summary_tags_in_prompt True \
    --reward_extraction sft_aligned \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.01 \
    --batch_size ${BATCH_SIZE} \
    --group_size ${GROUP_SIZE} \
    --epochs ${epochs} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --epsilon_low ${EPSILON_LOW} \
    --epsilon_high ${EPSILON_HIGH} \
    --kl_beta ${KL_BETA} \
    --save_every_steps ${SAVE_EVERY_STEPS} \
    --seed ${SEED}
EOF
)
    echo "Submitted: ${run_name}  (epochs=${epochs}, limit_examples=${limit_examples})  jid=${jid}"

    # Chain ID + OOD eval afterok on this training jid. submit_id_ood_eval_rl.sh
    # auto-picks the latest checkpoint at runtime, so partial training (e.g.
    # epoch sweep that doesn't reach 32 epochs in 7 days) is still evaluable.
    if [ "${SKIP_EVAL_CHAIN:-false}" != "true" ] && [ -n "${jid}" ]; then
        AFTEROK_JID="${jid}" \
        CKPT_DIR="${output_dir}" \
        RUN_TAG="${run_name}" \
        PARTITION="${EVAL_PARTITION:-kempner_requeue}" \
        ACCOUNT="${EVAL_ACCOUNT:-kempner_sham_lab}" \
        TIME="${EVAL_TIME:-7-00:00}" \
            bash "${ROOT_DIR}/scripts/submit_id_ood_eval_rl.sh" || \
            echo "[chain] submit_id_ood_eval_rl.sh failed for ${run_name}; you can submit it manually."
    fi
}

# ============================================================
# Epoch sweep — 20% data, vary epochs
# ============================================================
DATA20_LIMIT=$(( TOTAL_TRAIN_EXAMPLES * 20 / 100 ))
# RL epochs capped at 8 (anything larger is dropped, even with explicit override),
# matching submit_drive_genomics_sweeps.sh.
EPOCH_LIST_RAW="${EPOCH_LIST:-1 2 4 8}"
EPOCH_LIST=""
for _e in ${EPOCH_LIST_RAW}; do (( _e <= 8 )) && EPOCH_LIST+="${_e} "; done
EPOCH_LIST="${EPOCH_LIST% }"
# RUN_NAME_PREFIX lets a warm-started or otherwise re-targeted sweep coexist
# with the existing run names (e.g. RUN_NAME_PREFIX=warm- → warm-protein-grpo-...).
# Empty by default; output_dir is CHECKPOINT_BASE_DIR/${RUN_NAME_PREFIX}${RUN_NAME}.
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

echo "=== Epoch sweep (20% data = ${DATA20_LIMIT} examples; epochs=${EPOCH_LIST}) ==="
for EPOCHS in ${EPOCH_LIST}; do
    RUN_NAME="${RUN_NAME_PREFIX}protein-grpo-epochs${EPOCHS}-data20pct"
    submit_job "$RUN_NAME" "$EPOCHS" "$DATA20_LIMIT" "20"
done

# ============================================================
# Data fraction sweep — 1 epoch, vary data %
# Includes 20% even though it overlaps the 1-epoch case above, to match
# sweep_protein_sft.sh.
# ============================================================
PCT_LIST="${PCT_LIST:-20 40 60 80 100}"
echo "=== Data fraction sweep (1 epoch; pcts=${PCT_LIST}) ==="
for PCT in ${PCT_LIST}; do
    LIMIT=$(( TOTAL_TRAIN_EXAMPLES * PCT / 100 ))
    RUN_NAME="${RUN_NAME_PREFIX}protein-grpo-epochs1-data${PCT}pct"
    submit_job "$RUN_NAME" "1" "$LIMIT" "$PCT"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
