#!/usr/bin/env python3
"""
Pre-download BioReason datasets so training can use cached data.
Uses the same cache directory as sh_train_dna_qwen.sh (CACHE_DIR).
Run from the BioReason directory with your conda env activated.
"""
import os
from datasets import load_dataset

CACHE_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/lfesser/evo_tfm/BioReason/models"

# Force cache into CACHE_DIR (env var alone can be ignored in some setups)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

DATASETS = [
    "wanglab/kegg",
    "wanglab/variant_effect_coding",
    "wanglab/variant_effect_non_snv",
]

if __name__ == "__main__":
    print(f"Downloading datasets to cache: {CACHE_DIR}")
    for name in DATASETS:
        print(f"  Loading {name} ...")
        load_dataset(name, cache_dir=CACHE_DIR)
        print(f"  Done: {name}")
    print("All datasets downloaded.")
