#!/bin/bash
# Local test run for protein-only SFT sweep config.
# Uses debug=True (50 samples), 1 epoch, small context, sdpa attention (no unsloth needed).

set -euo pipefail

source $SCRATCH/envs/bio/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")"

python train_protein_llm.py \
    --dataset_type hf_reasoning \
    --reasoning_sft_dataset ${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}/protein \
    --text_model_name Qwen/Qwen3-4B-Thinking-2507 \
    --protein_model_name esm3_sm_open_v1 \
    --attn_implementation sdpa \
    --use_unsloth False \
    --model_type protein-llm \
    --max_length_text 1024 \
    --max_length_protein 384 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_epochs 1 \
    --train_data_fraction 0.20 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_gpus 1 \
    --num_nodes 1 \
    --weight_decay 0.01 \
    --seed 23 \
    --val_split_ratio 0.1 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1 \
    --num_sanity_val_steps 1 \
    --save_top_k 0 \
    --protein_embedding_layer 37 \
    --protein_model_finetune False \
    --go_model_finetune False \
    --unified_go_encoder False \
    --wandb_project bioreason-protein-sft-test \
    --run_name protein-sft-test-local \
    --debug True
