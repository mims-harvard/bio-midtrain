<h1 align="center">
🧬 bio-post<br>How Post-Training Shapes Biological Reasoning Models
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Preprint%202026-FF6B6B?style=for-the-badge" alt="Preprint">
  <img src="https://img.shields.io/badge/License-MIT-4A90E2?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.11%2B-00B89E?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
</p>

<p align="center">
  <i>Code for the paper</i> <b>"How Post-Training Shapes Biological Reasoning Models"</b> (Preprint, 2026).
</p>

<br>

## 🔍 Overview

Biological reasoning models couple a domain encoder (for DNA, protein, or
single-cell data) with a language model and are adapted through a multi-stage
post-training pipeline: **continued pre-training (CPT / mid-training) →
supervised fine-tuning (SFT) → reinforcement learning (GRPO)**. Each stage
introduces choices — how much data, how many epochs, whether RL warm-starts
from SFT, how strong the KL anchor is — yet their *interaction* and their
effect on in-distribution (ID) vs. out-of-distribution (OOD) generalization
are poorly understood.

This repository contains the code and harvested results for a controlled study
of those post-training dynamics across **two omics modalities**:

- **DNA** — KEGG disease-pathway reasoning, built on an Evo2 DNA encoder + Qwen3
  language backbone, mid-trained on the biology subset of FineFineWeb.
- **Protein** — CAFA-style Gene Ontology (GO) term prediction, built on an ESM-3
  protein encoder + Qwen3-4B-Thinking backbone with GO-GPT context.

We sweep the post-training axes that the paper studies and report matched ID/OOD
metrics so the stages can be compared on equal footing.

<br>

## ✨ Highlights

- **One pipeline, two modalities.** A controlled study of the full post-training
  pipeline — continued pre-training → SFT → RL (GRPO) — for both DNA (KEGG
  pathway reasoning) and protein (GO term prediction) reasoning models.
- **Data vs. epoch scaling, disentangled.** Separate sweeps over *how much* data
  and *how many* epochs at both the SFT and RL stages.
- **RL design choices.** Quantifies RL warm-starting from SFT vs. cold-starting
  from base, and the effect of the KL-anchor strength β.
- **Budget allocation.** How a fixed compute budget is best split between SFT
  and RL.
- **Matched ID/OOD reporting.** Every sweep cell is harvested into a single
  in-distribution vs. out-of-distribution table for apples-to-apples comparison.

<br>

### What we study

| Axis | DNA (KEGG) | Protein (GO) |
|------|------------|--------------|
| **CPT / mid-training** | LR × grad-accum sweep on FineFineWeb-bio (Qwen3-1.7B / 4B) | — |
| **SFT data scaling** | data-fraction sweep | data-fraction sweep (20–100%) |
| **SFT epoch scaling** | epoch sweep {1,2,4,8,16,32} | — |
| **RL (GRPO) data scaling** | — | data-fraction sweep |
| **RL (GRPO) epoch scaling** | epoch sweep {1,2,4,8} | epoch sweep {1,2,4,8} |
| **RL warm-start vs. base** | merge-SFT-then-RL | SFT-LoRA warm-start vs. cold base |
| **KL strength (β) ablation** | — | β ∈ {1e-4, 1e-3} |
| **Fixed SFT/RL budget split** | epoch budget = 8 | data budget = 20K |

Full experimental configuration (backbones, tokenization, optimizer, LoRA,
sweep grids, compute) is documented in the paper's experimental-setup appendix.

<br>

## 📂 Repository structure

```
bio-post/
├── BioReason/          # DNA reasoning codebase + our DNA post-training scripts
├── BioReason-Pro/      # Protein reasoning codebase + our protein SFT/RL sweeps
├── c2s_original/       # Cell2Sentence single-cell library (vendored reference)
├── cluster_env.example.sh  # Editable env-var defaults for the SLURM launchers
├── requirements.txt    # Consolidated pip dependencies (both components)
└── harvest_results.py  # Aggregates ID/OOD eval logs into result tables
```

`BioReason/`, `BioReason-Pro/`, and `c2s_original/` are vendored from their
upstream projects (see the Acknowledgements section below) and carry their
own `LICENSE` and `README`. The post-training-dynamics study lives in the
training/sweep/eval launchers inside `BioReason/` and `BioReason-Pro/` and in
the top-level harvesting and analysis scripts.

### Key entry points

| Stage | DNA (`BioReason/`) | Protein (`BioReason-Pro/`) |
|-------|--------------------|-----------------------------|
| CPT / mid-train | `train_finefineweb_midtrain.py`, `cpt_job.sh` | — |
| SFT | `train_dna_qwen.py` | `train_protein_llm.py` |
| RL (GRPO) | `train_grpo.py` | `train_protein_grpo.py` |
| Sweep launchers | `submit_drive_genomics_sweeps.sh`, `submit_drive_genomics_data_sweeps.sh` | `scripts/sweep_protein_sft.sh`, `scripts/sweep_protein_grpo.sh` |
| ID/OOD eval | `submit_id_ood_eval.sh`, `eval_kegg_dna_vllm.py` | `eval.py`, `scripts/submit_id_ood_eval{,_rl}.sh` |
| Inference | `reason.py` | `predict.py` |

<br>

## ⚙️ Installation

Each modality is a self-contained package with its own dependencies; we
recommend a separate environment per component. Python 3.11+ and a CUDA GPU are
required for training and evaluation.

```bash
git clone https://github.com/mims-harvard/bio-post.git
cd bio-post

# --- DNA (BioReason) ---
cd BioReason
pip install -e .                 # see BioReason/requirements.txt and install_evo2_stack.sh for the Evo2 stack
cd ..

# --- Protein (BioReason-Pro) ---
cd BioReason-Pro
pip install esm --no-deps        # ESM-3 (pin avoids a transformers/vllm conflict)
pip install -e .                 # pulls torch, vllm, transformers, ...
pip install flash-attn --no-build-isolation --no-cache-dir
cd ..
```

A consolidated [`requirements.txt`](requirements.txt) lists the pip-installable
dependencies for both components (the `esm`, `flash-attn`, and Evo2 stacks need
the special build steps noted above and in that file). See each subdirectory's
`README.md` for component-specific details (the Evo2 encoder stack for DNA,
ESM-3 + GO-GPT for protein).

> **Note on cluster configuration.** The sweep launchers (`*.sh`) were written
> for the Kempner SLURM cluster. Their absolute paths (repo root, Python
> interpreter, data, checkpoints, caches) are now read from environment
> variables that fall back to the original defaults, so the scripts run
> unchanged in their home environment. To reuse them elsewhere, copy and edit
> [`cluster_env.example.sh`](cluster_env.example.sh) and `source` it before
> launching. SLURM `#SBATCH --account=` / `--partition=` directives cannot read
> shell variables, so edit those header lines directly.

<br>

## 🚀 Reproducing the experiments

The launchers submit the sweeps described in the paper. A typical flow:

```bash
# DNA: CPT → post-CPT SFT/RL epoch sweep → eval
cd BioReason
bash cpt_job.sh                          # mid-train Qwen3 backbones on FineFineWeb-bio
bash submit_drive_genomics_sweeps.sh     # SFT/RL epoch sweep on KEGG
bash submit_id_ood_eval.sh               # ID/OOD evaluation

# Protein: SFT data sweep → GRPO sweep → eval
cd ../BioReason-Pro
bash scripts/sweep_protein_sft.sh        # SFT data-fraction sweep
bash scripts/sweep_protein_grpo.sh       # GRPO data/epoch sweep (warm-start + β ablation)
bash scripts/submit_id_ood_eval_rl.sh    # ID/OOD evaluation
```

Then harvest every sweep's logs into matched ID/OOD tables:

```bash
cd ..
python harvest_results.py --root .       # writes figures/results_id_ood.{md,json}
```

<br>

## 📊 Results

Full quantitative results — matched ID/OOD metrics for every SFT/RL sweep cell,
the DNA continued-pre-training perplexity sweep, the LoRA-rank ablation, and the
fixed SFT/RL budget-split ablations — are reported in the paper.

To regenerate the result tables from your own runs, run `harvest_results.py`
(see *Reproducing the experiments* above); it writes
`figures/results_id_ood.{md,json}`.

<br>

## 📑 Citation

If you find this work useful, please cite:

```bibtex
@article{fesser2026posttraining,
  title   = {How Post-Training Shapes Biological Reasoning Models},
  author  = {Fesser, Lukas and Zhang, Hanlin and Li, Michelle M. and Wang, Eric and Perozzi, Bryan and Azizi, Shekoofeh and Kakade, Sham M. and Zitnik, Marinka},
  journal = {Preprint},
  year    = {2026}
}
```

<sub>\* Lukas Fesser and Hanlin Zhang contributed equally.</sub>

<br>

## 🙏 Acknowledgements

This repository builds on, and vendors code from, several open-source projects:

- **[BioReason](https://github.com/bowang-lab/BioReason)** — the DNA-LLM reasoning
  architecture and KEGG benchmark (`BioReason/`).
- **[BioReason-Pro](https://github.com/bowang-lab/BioReason-Pro)** — the protein
  function-prediction reasoning model and GO-GPT (`BioReason-Pro/`).
- **[Cell2Sentence](https://github.com/vandijklab/cell2sentence)** — the
  single-cell language-model library (`c2s_original/`).

The vendored subdirectories retain their original licenses. We thank the
authors of these projects, as well as Qwen3, Evo2, and ESM-3, whose models we
build on. Experiments were run on the Kempner Institute cluster at Harvard.

<br>

## 📬 Contact

Questions and issues are welcome — please open a
[GitHub issue](https://github.com/mims-harvard/bio-post/issues). For other
inquiries, contact:

- Hanlin Zhang — `hanlinzhang@g.harvard.edu`
- Marinka Zitnik — `marinka@hms.harvard.edu`

<br>

## 📄 License

The top-level study code in this repository is released under the
[MIT License](LICENSE). The vendored subdirectories (`BioReason/`,
`c2s_original/`, and `BioReason-Pro/`) are covered by the licenses included
within each of them.
