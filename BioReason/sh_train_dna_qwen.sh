#!/bin/bash
#SBATCH --job-name=train_dna_qwen    # Name of the job
#SBATCH --time=12:00:00    # Time limit
#SBATCH --partition=gpu_batch    # Partition
#SBATCH --gpus=1    # Number of GPUs
#SBATCH --ntasks=1    # Number of tasks
#SBATCH --cpus-per-task=8    # Number of cores
#SBATCH --mem=128gb    # Memory limit
#SBATCH --output=train_dna_qwen_%j_%x.out    # Output file
#SBATCH --error=train_dna_qwen_%j_%x.err    # Error file

## Environment Setup
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "which python: $(which python)"

## Configuration Variables
CACHE_DIR=${BIOREASON_WORK_ROOT_ALT:-/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm}/BioReason/models
WORKING_DIR=${BIOREASON_WORK_ROOT_ALT:-/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm}/BioReason

## Run with srun only under sbatch; when run with bash directly (e.g. salloc), use python only
if [ -n "${SLURM_JOB_ID:-}" ]; then RUN_CMD="srun"; else RUN_CMD=""; fi

## Setup Environment
cd "$WORKING_DIR"
export WANDB_MODE=disabled
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export HF_HOME="${CACHE_DIR}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TORCH_HOME="${CACHE_DIR}/torch"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export WANDB_DIR="${CACHE_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb_cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/.cache/uv}"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export MPLCONFIGDIR="${CACHE_DIR}/mpl"
export TMPDIR="${CACHE_DIR}/tmp"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
nvidia-smi                             # Check GPU status


## =============================================================================
## KEGG Dataset Training
## =============================================================================

# NT-500M + Qwen3-1.7B on KEGG
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --merge_val_test_set True \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-1.7B on KEGG
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --merge_val_test_set True \
    --return_answer_in_batch True

# Qwen3-4B on KEGG (LLM-only)
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type llm \
    --dataset_type kegg \
    --max_length_dna 4 \
    --max_length_text 8192 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --return_answer_in_batch True

## =============================================================================
## Variant Effect Prediction (VEP) Training
## =============================================================================

# NT-500M + Qwen3-4B on VEP
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-1.7B on VEP
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True

# Qwen3-4B on VEP (LLM-only) - Testing max length text
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 4 \
    --max_length_text 4096 \
    --truncate_dna_per_side 1024 \
    --return_answer_in_batch True

## =============================================================================
## Variant Effect Prediction Non-SNV Training
## =============================================================================

# NT-500M + Qwen3-4B on VEP Non-SNV
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 1 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-4B on VEP Non-SNV
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True

# Qwen3-4B on VEP Non-SNV (LLM-only) - Testing max length text
stdbuf -oL -eL $RUN_CMD python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 1 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 4 \
    --max_length_text 4096 \
    --truncate_dna_per_side 1024 \
    --return_answer_in_batch True
