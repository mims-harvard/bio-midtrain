#!/bin/bash
# §4.6 / Figure-8 protein data-split ablation: 20K total, 1 epoch each stage,
# vary (n_sft, n_rl) where n_sft + n_rl = 20000.
#
# For each pair:
#   - if n_sft > 0: submit SFT (data_fraction = n_sft/20000)
#   - if n_rl  > 0: submit RL  (limit_examples = n_rl), with --dependency=afterok
#                   on the SFT job. The first GRPO step extracts the SFT
#                   adapter; if SFT_ADAPTER_BASE is set we point at the new dir.
#
# Skipped pair (already covered by existing rows in figures/results_id_ood.md):
#   - (20000, 0) ≡ protein-sft-epochs1-data100pct
#
# Defaults: kempner_h100, h100|h200 constraint, 2-day TIME, KL_BETA=1e-3 (klstrong).
# All artefacts land under CHECKPOINT_BASE_DIR/section46_<stamp>/ so cleanup is
# `rm -rf CHECKPOINT_BASE_DIR/section46_*`.
#
# Usage:
#   bash BioReason-Pro/scripts/submit_protein_data_split.sh                # fire all 4 cells
#   PAIRS="5000:15000 10000:10000" bash ... submit_protein_data_split.sh   # subset
#   DRY_RUN=true bash ... submit_protein_data_split.sh                     # validate

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT:-/n/home07/hanlinzhang/projects/evo_omics}/BioReason-Pro}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-/n/netscratch/kempner_sham_lab/Lab/$(whoami)/envs/bio/bin/activate}"

REASONING_SFT_DATASET="${REASONING_SFT_DATASET:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REASONING_SFT_DATASET}/checkpoints}"
GRPO_CACHE_DIR="${GRPO_CACHE_DIR:-${REASONING_SFT_DATASET}/hf_datasets_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-esm3}"

TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3_sm_open_v1"
DATASET_NAME="${REASONING_SFT_DATASET}"
DATASET_SPLIT="train"
GO_OBO_PATH="${ROOT_DIR}/bioreason2/dataset/go-basic.obo"

LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
SFT_LR="${SFT_LR:-1e-4}"
RL_LR="${RL_LR:-3e-5}"
SEED="${SEED:-23}"
KL_BETA="${KL_BETA:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_sham_lab}"
TIME_SFT="${TIME_SFT:-1-12:00}"
TIME_RL="${TIME_RL:-2-00:00}"
CONSTRAINT="${CONSTRAINT:-h100|h200}"
DRY_RUN="${DRY_RUN:-false}"

# (n_sft : n_rl) pairs — sum must be 20000. (20000:0) skipped (existing row).
PAIRS="${PAIRS:-0:20000 5000:15000 10000:10000 15000:5000}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_TAG="section46_${STAMP}"
CHECKPOINT_ROOT="${CHECKPOINT_BASE_DIR}/${SWEEP_TAG}"
LOG_DIR="${ROOT_DIR}/outputs/slurm/${SWEEP_TAG}"

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_DIR}"

submit_sft() {
    local n_sft="$1"
    local pair_tag="$2"
    local fraction
    fraction=$(awk -v n="${n_sft}" 'BEGIN { printf "%.4f", n/20000 }')
    local run_name="section46-protein-sft-${pair_tag}-n${n_sft}"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${run_name}"
    mkdir -p "${checkpoint_dir}"

    # Build python command as a single line. Bash's unquoted heredoc consumes
    # every \\<newline> as line continuation (this is the cause of the original
    # "unrecognized arguments:" failure in jids 9890840/2/5/60), so we keep
    # all python args on one logical line in the rendered script.
    local sft_args
    sft_args="--dataset_type hf_reasoning --reasoning_sft_dataset ${REASONING_SFT_DATASET}"
    sft_args+=" --text_model_name ${TEXT_MODEL_NAME} --protein_model_name ${PROTEIN_MODEL_NAME}"
    sft_args+=" --attn_implementation flash_attention_2 --use_unsloth False --model_type protein-llm"
    sft_args+=" --max_length_text 10000 --max_length_protein 2000"
    sft_args+=" --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} --lora_dropout 0"
    sft_args+=" --learning_rate ${SFT_LR} --warmup_ratio 0.05 --max_epochs 1"
    sft_args+=" --train_data_fraction ${fraction}"
    sft_args+=" --batch_size 1 --gradient_accumulation_steps 16"
    sft_args+=" --num_workers 0 --num_gpus 1 --num_nodes 1 --weight_decay 0.01 --seed ${SEED}"
    sft_args+=" --val_split_ratio 0.1 --val_check_interval 1.0 --log_every_n_steps 10"
    sft_args+=" --num_sanity_val_steps 0 --save_top_k 1 --checkpoint_start_epoch 0"
    sft_args+=" --protein_embedding_layer 37"
    sft_args+=" --protein_model_finetune False --go_model_finetune False --unified_go_encoder False"
    sft_args+=" --wandb_project ${WANDB_PROJECT} --run_name ${run_name}"
    sft_args+=" --checkpoint_dir ${checkpoint_dir}"
    sft_args+=" --debug False"
    # Resume support: if last.ckpt exists at submission time, include it.
    [ -f "${checkpoint_dir}/last.ckpt" ] && sft_args+=" --ckpt_path ${checkpoint_dir}/last.ckpt"

    local sbatch_script
    sbatch_script=$(cat <<EOF
#!/bin/bash
#SBATCH -J ${run_name}
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t ${TIME_SFT}
#SBATCH -p ${PARTITION}
#SBATCH --mem=128GB
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
#SBATCH -o ${LOG_DIR}/${run_name}_%j.out
#SBATCH -e ${LOG_DIR}/${run_name}_%j.err

set -euo pipefail
module load cuda/12.4 || true
module load gcc/12.2.0-fasrc01 || true
module load cudnn/9.10.2.21_cuda12 || true
[ -n "\${CUDNN_HOME:-}" ] && export LD_LIBRARY_PATH="\${CUDNN_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
export BIOREASON_CACHE_ROOT=${REASONING_SFT_DATASET}/bioreason_cache
mkdir -p "\$WANDB_DIR" "\$TORCH_HOME" "\$TRITON_CACHE_DIR" "\$BIOREASON_CACHE_ROOT"
cd ${ROOT_DIR}

python train_protein_llm.py ${sft_args}

# Extract LoRA adapter so the chained GRPO can warm-start from it.
BEST_CKPT=\$(ls -1t ${checkpoint_dir}/${run_name}-best-epoch=*.ckpt 2>/dev/null | head -1 || true)
[ -z "\$BEST_CKPT" ] && BEST_CKPT=\$(ls -1t ${checkpoint_dir}/${run_name}-epoch=*.ckpt 2>/dev/null | head -1 || true)
if [ -n "\$BEST_CKPT" ]; then
    python ${SCRIPT_DIR}/extract_sft_lora_adapter.py --ckpt "\$BEST_CKPT" --out ${checkpoint_dir}/sft_lora_adapter
fi
EOF
)

    if [ "${DRY_RUN}" = "true" ]; then
        echo "[DRY] SFT ${run_name} (data_fraction=${fraction})"
        echo "${LOG_DIR}/${run_name}_DRY.jid"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable <<<"${sbatch_script}")
    echo "  SFT ${run_name}  jid=${jid}  (data_fraction=${fraction})"
    echo "${jid}"
}

submit_rl() {
    local n_rl="$1"
    local pair_tag="$2"
    local sft_jid="$3"     # may be empty for n_sft=0 (pure RL)
    local sft_ckpt_dir="$4"  # may be empty
    local run_name="section46-protein-grpo-${pair_tag}-n${n_rl}"
    local output_dir="${CHECKPOINT_ROOT}/${run_name}"
    mkdir -p "${output_dir}"

    local dep_arg=""
    [ -n "${sft_jid}" ] && dep_arg="--dependency=afterok:${sft_jid}"
    local adapter_arg=""
    [ -n "${sft_ckpt_dir}" ] && adapter_arg="--sft_adapter_path ${sft_ckpt_dir}/sft_lora_adapter"

    # Single-line python invocation — see submit_sft() comment re: heredoc bug.
    local rl_args
    rl_args="--dataset_name ${DATASET_NAME} --dataset_split ${DATASET_SPLIT}"
    rl_args+=" --limit_examples ${n_rl} --model_name_or_path ${TEXT_MODEL_NAME}"
    [ -n "${adapter_arg}" ] && rl_args+=" ${adapter_arg}"
    rl_args+=" --output_dir ${output_dir} --go_obo_path ${GO_OBO_PATH}"
    rl_args+=" --interpro_in_prompt True --ppi_in_prompt False --ask_all_go_aspects True"
    rl_args+=" --append_uniprot_suffix False --go_summary_tags_in_prompt True"
    rl_args+=" --reward_extraction sft_aligned --learning_rate ${RL_LR} --weight_decay 0.01"
    rl_args+=" --batch_size ${BATCH_SIZE} --group_size ${GROUP_SIZE} --epochs 1"
    rl_args+=" --max_new_tokens ${MAX_NEW_TOKENS} --temperature 1.0 --top_p 1.0"
    rl_args+=" --epsilon_low 0.2 --epsilon_high 0.28 --kl_beta ${KL_BETA}"
    rl_args+=" --save_every_steps ${SAVE_EVERY_STEPS} --seed ${SEED}"

    local sbatch_script
    sbatch_script=$(cat <<EOF
#!/bin/bash
#SBATCH -J ${run_name}
#SBATCH -c 24
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t ${TIME_RL}
#SBATCH -p ${PARTITION}
#SBATCH --mem=128GB
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
#SBATCH -o ${LOG_DIR}/${run_name}_%j.out
#SBATCH -e ${LOG_DIR}/${run_name}_%j.err

set -euo pipefail
module load cuda/12.4 || true
module load gcc/12.2.0-fasrc01 || true
module load cudnn/9.10.2.21_cuda12 || true
[ -n "\${CUDNN_HOME:-}" ] && export LD_LIBRARY_PATH="\${CUDNN_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
source ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_DIR=${REASONING_SFT_DATASET}/wandb
export TORCH_HOME=${REASONING_SFT_DATASET}/torch_cache
export TRITON_CACHE_DIR=${REASONING_SFT_DATASET}/triton_cache
export HF_DATASETS_CACHE=${GRPO_CACHE_DIR}
export BIOREASON_CACHE_ROOT=${REASONING_SFT_DATASET}/bioreason_cache
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_NAME=${run_name}
mkdir -p "\$WANDB_DIR" "\$TORCH_HOME" "\$TRITON_CACHE_DIR" "\$HF_DATASETS_CACHE"
cd ${ROOT_DIR}

python train_protein_grpo.py ${rl_args}
EOF
)

    if [ "${DRY_RUN}" = "true" ]; then
        echo "[DRY] RL  ${run_name} (limit=${n_rl}; dep=${sft_jid:-none}; adapter=${adapter_arg:-none})"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable ${dep_arg} <<<"${sbatch_script}")
    echo "  RL  ${run_name}  jid=${jid}  (limit=${n_rl}; dep=${sft_jid:-none})"
}

echo "=== §4.6 protein data-split ablation ==="
echo "  sweep tag: ${SWEEP_TAG}"
echo "  ckpt root: ${CHECKPOINT_ROOT}"
echo "  log dir:   ${LOG_DIR}"
echo "  pairs:     ${PAIRS}"
echo "  KL_BETA=${KL_BETA}  partition=${PARTITION}  constraint=${CONSTRAINT}"
echo ""

for pair in ${PAIRS}; do
    n_sft="${pair%:*}"
    n_rl="${pair##*:}"
    sum=$(( n_sft + n_rl ))
    if [ "${sum}" -ne 20000 ]; then
        echo "  [skip] pair ${pair} sums to ${sum}, expected 20000"
        continue
    fi
    pair_tag="sft${n_sft}-rl${n_rl}"

    sft_jid=""
    sft_ckpt_dir=""
    if [ "${n_sft}" -gt 0 ]; then
        sft_jid=$(submit_sft "${n_sft}" "${pair_tag}" | tail -1)
        sft_ckpt_dir="${CHECKPOINT_ROOT}/section46-protein-sft-${pair_tag}-n${n_sft}"
    fi
    if [ "${n_rl}" -gt 0 ]; then
        submit_rl "${n_rl}" "${pair_tag}" "${sft_jid}" "${sft_ckpt_dir}"
    fi
done

echo ""
echo "Submitted. Monitor: squeue -u \$USER --name section46*"
echo "Sweep root (rm -rf to cleanup): ${CHECKPOINT_ROOT}"
