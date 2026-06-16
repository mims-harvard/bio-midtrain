# BioReason Training Configurations Overview

This document lists the 12 models to train: **Qwen3-1.7B** and **Qwen3-4B**, each with the **EVO2-1B DNA encoder**, on three tasks (KEGG, VEP SNV, VEP Non-SNV) with **5** and **10** SFT epochs per task.

**Common setup:** Run from the BioReason directory. Use `train_dna_qwen.py` with `--model_type dna-llm`, EVO2-1B encoder (`--dna_model_name evo2_1b_base`, `--dna_is_evo2 True`, etc.). SLURM script: `sh_train_dna_qwen.sh` (uncomment or extract the command you need).

---

## Summary Table

| # | LLM           | DNA Encoder | Task            | Epochs | Dataset type                |
|---|---------------|-------------|-----------------|--------|-----------------------------|
| 1 | Qwen3-1.7B    | EVO2-1B     | KEGG            | 5      | `kegg`                      |
| 2 | Qwen3-1.7B    | EVO2-1B     | KEGG            | 10     | `kegg`                      |
| 3 | Qwen3-1.7B    | EVO2-1B     | VEP SNV         | 5      | `variant_effect_coding`     |
| 4 | Qwen3-1.7B    | EVO2-1B     | VEP SNV         | 10     | `variant_effect_coding`     |
| 5 | Qwen3-1.7B    | EVO2-1B     | VEP Non-SNV     | 5      | `variant_effect_non_snv`    |
| 6 | Qwen3-1.7B    | EVO2-1B     | VEP Non-SNV     | 10     | `variant_effect_non_snv`    |
| 7 | Qwen3-4B      | EVO2-1B     | KEGG            | 5      | `kegg`                      |
| 8 | Qwen3-4B      | EVO2-1B     | KEGG            | 10     | `kegg`                      |
| 9 | Qwen3-4B      | EVO2-1B     | VEP SNV         | 5      | `variant_effect_coding`     |
|10 | Qwen3-4B      | EVO2-1B     | VEP SNV         | 10     | `variant_effect_coding`     |
|11 | Qwen3-4B      | EVO2-1B     | VEP Non-SNV     | 5      | `variant_effect_non_snv`    |
|12 | Qwen3-4B      | EVO2-1B     | VEP Non-SNV     | 10     | `variant_effect_non_snv`    |

---

## Environment variables (set before running)

```bash
export CACHE_DIR=/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm/BioReason/models
export WORKING_DIR=/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm/BioReason
cd "$WORKING_DIR"
export WANDB_MODE=online
export HF_DATASETS_CACHE="$CACHE_DIR"
```

Under SLURM, use `srun`; otherwise run `python` directly. Example wrapper: `RUN_CMD="srun"` when `SLURM_JOB_ID` is set, else `RUN_CMD=""`, then prefix the python command with `stdbuf -oL -eL $RUN_CMD`.

---

## Commands to run

### 1. Qwen3-1.7B + EVO2-1B — KEGG (5 epochs)

```bash
python train_dna_qwen.py \
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
```

### 2. Qwen3-1.7B + EVO2-1B — KEGG (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
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
```

### 3. Qwen3-1.7B + EVO2-1B — VEP SNV (5 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 4. Qwen3-1.7B + EVO2-1B — VEP SNV (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 5. Qwen3-1.7B + EVO2-1B — VEP Non-SNV (5 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 6. Qwen3-1.7B + EVO2-1B — VEP Non-SNV (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 7. Qwen3-4B + EVO2-1B — KEGG (5 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
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
```

### 8. Qwen3-4B + EVO2-1B — KEGG (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
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
```

### 9. Qwen3-4B + EVO2-1B — VEP SNV (5 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 10. Qwen3-4B + EVO2-1B — VEP SNV (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 11. Qwen3-4B + EVO2-1B — VEP Non-SNV (5 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

### 12. Qwen3-4B + EVO2-1B — VEP Non-SNV (10 epochs)

```bash
python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True
```

---

## SLURM

Submit with the script that defines the same env and runs one of the commands above, e.g.:

```bash
sbatch sh_train_dna_qwen.sh
```

To run a single config, edit `sh_train_dna_qwen.sh` so only the desired block is uncommented (or copy one command block into a minimal `sbatch` script). For 12 separate jobs, use 12 scripts or one script that takes job index / config name as an argument and runs the corresponding command.

---

## Differences at a glance

| Setting        | KEGG              | VEP SNV / Non-SNV   |
|----------------|-------------------|----------------------|
| `--dataset_type` | `kegg`            | `variant_effect_coding` / `variant_effect_non_snv` |
| `--batch_size` | 1                 | 2                    |
| `--merge_val_test_set` | `True`   | not used             |

All 12 runs use the same EVO2-1B DNA encoder options: `evo2_1b_base`, `max_length_dna 2048`, `truncate_dna_per_side 1024`, `dna_is_evo2 True`, `dna_embedding_layer blocks.20.mlp.l3`.
