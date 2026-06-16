#!/bin/bash
#SBATCH --job-name=train_dna    # Name of the job
#SBATCH --time=8:00:00    # Time limit
#SBATCH --partition=gpu_batch    # Partition
#SBATCH --gpus=1    # Number of GPUs
#SBATCH --ntasks=1    # Number of tasks
#SBATCH --cpus-per-task=6    # Number of cores
#SBATCH --mem=128gb    # Memory limit
#SBATCH --output=train_dna_%j_%x.out    # Output file
#SBATCH --error=train_dna_%j_%x.err    # Error file

## Environment Setup
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "which python: $(which python)"

## Configuration Variables
# Change these to match your setup
CONDA_ENV=CONDA_ENV                     # Change to your conda environment name
CACHE_DIR=CACHE_DIR                     # Change to your HuggingFace cache directory
WANDB_PROJECT=WANDB_PROJECT             # Change to your W&B project name

## Setup Environment
conda activate $CONDA_ENV               # Change to your conda environment
cd .../BioReason/    # Change to the directory containing the script
nvidia-smi                             # Check GPU status


## =============================================================================
## KEGG Dataset Training (DNA-only models)
## =============================================================================

# NT-500M on KEGG
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type kegg \
    --merge_val_test_set True

# EVO2-1B on KEGG
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name evo2_1b_base \
    --strategy ddp \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --learning_rate 3e-4 \
    --dataset_type kegg \
    --merge_val_test_set True

## =============================================================================
## Variant Effect Prediction (VEP) Training
## =============================================================================

# NT-500M on VEP
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_coding

# EVO2-1B on VEP
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name evo2_1b_base \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_coding

## =============================================================================
## Variant Effect Prediction Non-SNV Training
## =============================================================================

# NT-500M on VEP Non-SNV
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_non_snv

# EVO2-1B on VEP Non-SNV
stdbuf -oL -eL srun python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --dna_model_name evo2_1b_base \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_non_snv