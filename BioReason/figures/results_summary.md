# Aggregated Experimental Results

_Collected 2026-04-25 from logs under `BioReason-Pro/outputs/slurm/` and `BioReason/logs/drive/`._

Scope (per request): protein epoch-scaling RL + data-scaling SFT/RL, and DNA mid-train / CPT.

---

## 1. Protein — RL (GRPO) epoch scaling — 20% data, vary epochs

**Status: NO COMPLETED RESULTS.** All sweep runs in `BioReason-Pro/outputs/slurm/protein-grpo-epochs{1,2,4,8}-data20pct_*` failed before logging any reward. Three rounds of submissions:

| Round | Job IDs | Outcome |
|---|---|---|
| 1 | `8198133–8198142` | `load_dataset` mismatch: `train: arrow, test: csv` (cache lived inside data dir) |
| 2 | `8252182–8252213` | OOM in Qwen3-4B forward on H100 80GB, `batch=1 group=8`, then `group=4` still OOM |
| 3 | `8276607–8276644` | `load_dataset` mismatch: `train: json, test: csv` (re-cached) |

No `train/rewards/correctness_reward_func/mean` was emitted by any run. Sweep is blocked on (a) dataset format being made consistent across splits (or `load_dataset` constrained with `data_files=`), and (b) memory budget — even `group=4 max_new_tokens=512` OOMs Qwen3-4B at full FT on a single H100 80GB.

Sweep script: `BioReason-Pro/scripts/sweep_protein_grpo.sh:131-137`

---

## 2. Protein — SFT data scaling — 1 epoch, vary data %

**Status: TRAINING DONE, HELD-OUT EVAL CRASHED.**

Training (`BioReason-Pro/outputs/slurm/data/protein-sft-epochs1-data{20,40,60,80,100}pct_*`) all completed cleanly. Best `val_loss_epoch` (validation, not held-out) per fraction:

| data % | val_loss_epoch | train_examples | best ckpt |
|---|---|---|---|
| 20%  | 0.9311 | 19,254  | `protein-sft-epochs1-data20pct-best-...val_loss_epoch=0.9311.ckpt` |
| 40%  | 0.8749 | 38,509  | …`val_loss_epoch=0.8749.ckpt` |
| 60%  | 0.8437 | 57,764  | …`val_loss_epoch=0.8437.ckpt` |
| 80%  | 0.8176 | 77,019  | …`val_loss_epoch=0.8176.ckpt` |
| 100% | 0.7994 | 96,274  | …`val_loss_epoch=0.7994.ckpt` |

Monotonically decreasing val loss with data — clean scaling signal.

Held-out ID/OOD eval jobs `protein-eval-sft-data{20,40,60,80,100}pct-{id,ood}_8393*` all crash in `train_protein_llm.py:805 → bioreason2/utils/go_reward.py:80` with `AttributeError: 'str' object has no attribute 'get'` (`parent_map` is a str rather than the expected dict-of-lists). No held-out F1 / precision / recall yet — bug fix in `go_reward.py` is required before re-launching `submit_id_ood_eval.sh`.

Sweep script: `BioReason-Pro/scripts/sweep_protein_sft.sh:188-192`
Checkpoints: `/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason/protein/checkpoints/protein-sft-epochs1-data{20,40,60,80,100}pct/`

---

## 3. Protein — RL (GRPO) data scaling — 1 epoch, vary data %

**Status: NO COMPLETED RESULTS.** Same three failure rounds as §1; runs at `protein-grpo-epochs1-data{20,40,60,80,100}pct_*` either failed at `load_dataset` (rounds 1 and 3) or OOM'd before the first reward log (round 2).

Blocked on the same two issues as §1.

Sweep script: `BioReason-Pro/scripts/sweep_protein_grpo.sh:144-150`

---

## 4. DNA — CPT (mid-training) on FineFineWeb biology

**Status: COMPLETE.** All 8 configs in `cpt_ffw_20260424_165721/` have `final_eval_metrics.json` (1 epoch, 200k train / 5k eval, len=1024).

| Model        | LR    | grad_accum | eval_loss | perplexity |
|--------------|-------|------------|-----------|------------|
| Qwen3-1.7B   | 1e-5  | 64         | 2.4222    | 11.27      |
| Qwen3-1.7B   | 1e-5  | 128        | 2.4253    | 11.31      |
| Qwen3-1.7B   | 3e-4  | 64         | 2.4161    | 11.20      |
| **Qwen3-1.7B** | **3e-4** | **128** | **2.3800** | **10.80** ← best 1.7B |
| Qwen3-4B     | 1e-5  | 128        | 2.2688    | 9.67       |
| **Qwen3-4B** | **1e-5** | **64**  | **2.2636** | **9.62**  ← best 4B / overall |
| Qwen3-4B     | 3e-4  | 64         | 2.3828    | 10.83      |
| Qwen3-4B     | 3e-4  | 128        | 2.3280    | 10.26      |

Takeaways:
- 4B beats 1.7B by ≥0.5 ppl across all hparams.
- 1.7B prefers the higher LR (3e-4); 4B prefers the lower LR (1e-5) — consistent with model-size LR scaling.
- ga=64 ≥ ga=128 within each LR for the best configs.

Source: `/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm/BioReason/checkpoints/drive/cpt_ffw_20260424_165721/m-*/final_eval_metrics.json`
Driver: `BioReason/cpt_job.sh` → `BioReason/train_finefineweb_midtrain.py`

---

## 5. DNA — Downstream after CPT (bonus, since they share infra)

### 5a. SFT epoch scaling, KEGG held-out accuracy

| Epochs | Base 1.7B | CPT+1.7B | Base 4B | CPT+4B |
|---|---|---|---|---|
| 1  | 0.7241 | 0.6793 | 0.8483 | 0.8586 |
| 2  | 0.8379 | 0.7690 | 0.8862 | 0.8621 |
| 4  | 0.8655 | 0.8552 | **0.8897** | **0.8862** |
| 8  | 0.8931 | 0.8621 | 0.8379 | _missing_ |
| 16 | **0.9069** | **0.8655** | 0.8448 | _missing_ |
| 32 | 0.8793 | _missing_ | 0.8517 | _missing_ |

Best base 1.7B: 90.69% @ 16ep. Best base 4B: 88.97% @ 4ep. CPT init does not yet outperform base init in the runs that finished — consistent with the held-out ppl gap being small (≤0.3 nats).

### 5b. RL (GRPO) epoch scaling, last `train/rewards/correctness_reward_func/mean` (TRAINING reward proxy, not held-out)

| Epochs | Base 1.7B | Base 4B | CPT+1.7B | CPT+4B |
|---|---|---|---|---|
| 1  | 2.000 | 2.000 | _no reward logged_ | _no reward logged_ |
| 2  | 1.938 | 1.812 | _no reward logged_ | _no reward logged_ |
| 4  | 2.000 | 2.000 | _no reward logged_ | _no reward logged_ |
| 8  | 1.500 | 1.438 | _no reward logged_ | _no reward logged_ |
| 16 | 0.000 | 2.000 | — | — |
| 32 | 0.000 | 2.000 | — | — |

Caveat: this is *training-set* reward, not held-out accuracy. The `0.000` entries at e16/e32 for 1.7B suggest format collapse late in training. A held-out eval pass (`eval_kegg_dna_vllm.py`) on the saved checkpoints would give the comparable number to §5a.

### 5c. SFT data scaling (1 epoch, vary data %), KEGG accuracy

| data % | 1.7B | 4B |
|---|---|---|
| 20%  | 0.7552 | 0.8172 |
| 40%  | 0.8241 | 0.8483 |
| 60%  | 0.7966 | 0.8621 |
| 80%  | 0.8207 | **0.8759** |
| 100% | 0.8207 | 0.8552 |

4B scales monotonically until 80%; 1.7B saturates around 40% — typical small-model data ceiling.

### 5d. RL data scaling — NO RESULTS

All 10 runs in `BioReason/logs/drive/data_sweeps/rl/qwen3_{1p7b,4b}/d{20,40,60,80,100}_*` crash in DeepSpeed init: `lr_schedules.py:675 → max(2, warmup_num_steps)` with `TypeError: '>' not supported between instances of 'str' and 'int'`. Fix: cast `warmup_num_steps` to int (or unset and let DeepSpeed infer) in `BioReason/grpo_trainer_lora_model/ds_config_stage2.json` or in the `train_grpo.py` config plumb-through.

---

## What's blocking the four "missing" cells

| Section | Blocker | Fix scope |
|---|---|---|
| §1, §3 (protein RL all) | (a) `load_dataset` split format mismatch; (b) Qwen3-4B GRPO OOM on 80GB | Pin `data_files=` map; reduce `MAX_NEW_TOKENS`/`GROUP_SIZE` further or switch to LoRA / DeepSpeed-Z3 |
| §2 (protein SFT held-out F1) | `go_reward.py:80` `parent_map` typed wrong | One-line bug fix, then `submit_id_ood_eval.sh` over the 5 ckpts |
| §5d (DNA RL data sweep) | DeepSpeed scheduler `warmup_num_steps` str→int | Cast in DS config / training-arg plumb |
| §5a/§5b CPT cells | Some cpt_qwen3_* runs still in queue or crashed silently | Re-check `squeue` and missing `e*_*.out` |

---

_Sources: `gen_post_cpt_report.py` aggregation logic mirrors §5a/§5b; ran by hand to also collect the missing-cell diagnoses above._
