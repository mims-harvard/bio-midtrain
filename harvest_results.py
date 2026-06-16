#!/usr/bin/env python3
"""Harvest ID/OOD eval results from the four sweeps and write paper-ready tables.

Walks the SLURM log + ckpt trees seeded in this session and pulls out:
  - DNA SFT eval:   wandb run_name -eval-id / -eval-ood, accuracy from on_test_epoch_end logs
  - DNA RL eval:    train_grpo --eval_only metrics (eval_id/correctness, eval_ood/correctness)
  - Protein SFT eval: train_protein_llm --eval_only --gen_eval (id_f1, ood_f1)
  - Protein RL eval:  train_protein_grpo --eval_only (eval_<split>_metrics.json)

Run:  python harvest_results.py [--root /path/to/repo]

Outputs:
  - figures/results_id_ood.md   (Markdown tables, per-domain)
  - figures/results_id_ood.json (raw harvested rows for downstream scripting)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Default to the directory this script lives in; override with --root.
REPO_DEFAULT = str(Path(__file__).resolve().parent)
# Protein checkpoint root used during the reported runs; override with --protein_ckpt_root
# or the PROTEIN_CKPT_ROOT env var to point at your own checkpoints.
PROTEIN_CKPT_ROOT = os.environ.get(
    "PROTEIN_CKPT_ROOT",
    "/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason/protein/checkpoints",
)


# ---------- generic log helpers ------------------------------------------------

def tail_text(path: Path, max_bytes: int = 4_000_000) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def latest_log(dir_path: Path, pattern: str) -> Optional[Path]:
    if not dir_path.is_dir():
        return None
    matches = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


# ---------- DNA SFT eval -------------------------------------------------------
# Logs at: BioReason/logs/lora_sweeps/eval_id_ood/eval-{id,ood}-r32_<jid>.{out,err}
# wandb run_name encodes the model + epoch via the underlying CKPT path.
# train_dna_qwen on_test_epoch_end prints lines like:
#   "test_acc: 0.8XXX" or "Accuracy: 0.8XXX"
# We capture the last accuracy-like float in the log.

DNA_SFT_LOG_DIR = "BioReason/logs/lora_sweeps/eval_id_ood"
DNA_SFT_ACC_RE = re.compile(
    r"(?:test[_/](?:acc(?:uracy)?|accuracy_epoch)|Accuracy|accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.I,
)
DNA_SFT_CKPT_RE = re.compile(r"cpt_qwen3_(1p7b|4b)/e(\d+)/")


def harvest_dna_sft(root: Path) -> list[dict]:
    log_dir = root / DNA_SFT_LOG_DIR
    out: list[dict] = []
    for log in sorted(log_dir.glob("eval-*-r32_*.out")):
        text = tail_text(log) + "\n" + tail_text(log.with_suffix(".err"))
        ckpt_match = DNA_SFT_CKPT_RE.search(text)
        model_tag = f"cpt_qwen3_{ckpt_match.group(1)}" if ckpt_match else "?"
        epoch = int(ckpt_match.group(2)) if ckpt_match else -1
        # Split inferred from the job name: eval-id-... or eval-ood-...
        m = re.search(r"eval-(id|ood)-r32_", log.name)
        split = m.group(1) if m else "?"
        accs = DNA_SFT_ACC_RE.findall(text)
        # Skip if log is empty / job hasn't run yet
        finished = "Trainer.test" in text or "test_loss_epoch" in text or accs
        if not finished:
            continue
        out.append({
            "domain": "dna_sft",
            "model": model_tag,
            "epoch": epoch,
            "split": split,
            "accuracy": float(accs[-1]) if accs else None,
            "log": str(log),
        })
    return out


# ---------- DNA RL eval --------------------------------------------------------
# Logs at: BioReason/logs/lora_sweeps/eval_id_ood_rl/rl-eval-{id,ood}-r32_<jid>.{out,err}
# train_grpo eval_only prints "[eval_only] metrics: {...}".

DNA_RL_LOG_DIR = "BioReason/logs/lora_sweeps/eval_id_ood_rl"
DNA_RL_METRICS_RE = re.compile(r"\[eval_only\] metrics:\s*(\{.*?\})\s*$", re.M | re.S)
DNA_RL_CKPT_RE = re.compile(r"rl/cpt_qwen3_(1p7b|4b)/e(\d+)")


def harvest_dna_rl(root: Path) -> list[dict]:
    log_dir = root / DNA_RL_LOG_DIR
    out: list[dict] = []
    for log in sorted(log_dir.glob("rl-eval-*-r32_*.out")):
        text = tail_text(log) + "\n" + tail_text(log.with_suffix(".err"))
        ckpt_match = DNA_RL_CKPT_RE.search(text)
        model_tag = f"cpt_qwen3_{ckpt_match.group(1)}" if ckpt_match else "?"
        epoch = int(ckpt_match.group(2)) if ckpt_match else -1
        m = re.search(r"rl-eval-(id|ood)-r32_", log.name)
        split = m.group(1) if m else "?"

        metrics_match = DNA_RL_METRICS_RE.search(text)
        if not metrics_match:
            continue
        try:
            # The dict printed by python isn't strict JSON — eval cautiously.
            blob = metrics_match.group(1).replace("'", '"')
            metrics = json.loads(blob)
        except Exception:
            continue
        out.append({
            "domain": "dna_rl",
            "model": model_tag,
            "epoch": epoch,
            "split": split,
            "metrics": metrics,
            "log": str(log),
        })
    return out


# ---------- protein SFT eval ---------------------------------------------------
# Logs at: BioReason-Pro/outputs/slurm/eval_id_ood/
#   old:      protein-eval-sft-data{20..100}pct-{id,ood}_<jid>.out
#   resubmit: protein-eval-protein-sft-epochs1-data{N}pct-<tag>-{id,ood}_<jid>.out
# train_protein_llm on_test_epoch_end prints:
#   "[gen_eval] FINAL split=<id|ood> N=<n> P=<x> R=<y> F1=<z>"
# NOTE: only epoch=1 data-scaling logs belong here; epochs{2,4,8,...}-data20pct
# are the SFT *epoch* sweep and must be excluded.

PROTEIN_SFT_LOG_DIR = "BioReason-Pro/outputs/slurm/eval_id_ood"
PROTEIN_SFT_FINAL_RE = re.compile(
    r"\[gen_eval\]\s*FINAL\s+split=(\S+)\s+N=(\d+)\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)"
)
PROTEIN_SFT_RUNNING_RE = re.compile(
    r"\[gen_eval\]\s*split=\S+\s+batch=\d+\s+n=(\d+)\s+running_f1=([\d.]+)"
)
PROTEIN_SFT_TAG_RE = re.compile(
    r"protein-eval-(?:protein-)?sft-(?:epochs(\d+)-)?data(\d+)pct.*?-(id|ood)_\d+\.out$"
)


def harvest_protein_sft(root: Path) -> list[dict]:
    """One row per (data_pct, split). Prefer the most-advanced FINAL across all
    logs; fall back to the most-advanced running_f1 (n≥50, partial=True)."""
    log_dir = root / PROTEIN_SFT_LOG_DIR
    # Group logs by (pct, split); pick the one with the largest progress.
    best: dict[tuple[int, str], dict] = {}
    for log in sorted(log_dir.glob("protein-eval-*sft*data*pct*_*.out")):
        m = PROTEIN_SFT_TAG_RE.search(log.name)
        if not m:
            continue
        epoch_tok = m.group(1)
        if epoch_tok is not None and epoch_tok != "1":
            continue  # SFT epoch-sweep log, not the data-scaling table
        pct = int(m.group(2))
        split = m.group(3)
        text = tail_text(log) + "\n" + tail_text(log.with_suffix(".err"))
        final = PROTEIN_SFT_FINAL_RE.search(text)
        if final:
            row = {
                "domain": "protein_sft",
                "data_pct": pct,
                "split": split,
                "n": int(final.group(2)),
                "precision": float(final.group(3)),
                "recall": float(final.group(4)),
                "f1": float(final.group(5)),
                "log": str(log),
                "partial": False,
            }
        else:
            running = PROTEIN_SFT_RUNNING_RE.findall(text)
            if not running:
                continue
            n_str, f1_str = running[-1]
            n_val = int(n_str)
            if n_val < 50:
                continue
            row = {
                "domain": "protein_sft",
                "data_pct": pct,
                "split": split,
                "n": n_val,
                "precision": None,
                "recall": None,
                "f1": float(f1_str),
                "log": str(log),
                "partial": True,
            }
        prev = best.get((pct, split))
        if prev is None or (
            (not row["partial"]) and prev["partial"]
        ) or (
            row["partial"] == prev["partial"] and row["n"] > prev["n"]
        ):
            best[(pct, split)] = row
    return list(best.values())


# ---------- protein RL eval ----------------------------------------------------
# train_protein_grpo --eval_only writes <CKPT_DIR>/_eval_<split>/eval_<split>_metrics.json
# with keys eval_<split>/{precision,recall,f1,n_examples}.

PROTEIN_RL_RUN_RE = re.compile(r"^(?P<prefix>(?:warm-|klstrong-)?)protein-grpo-epochs(?P<epoch>\d+)-data(?P<pct>\d+)pct$")

# Latest running_f1 per (run, split) lives in the eval log:
#   BioReason-Pro/outputs/slurm/eval_id_ood_rl/protein-rl-eval-<run_tag>-<split>_<jid>.out
# train_protein_grpo prints "[eval_only] N/M running_f1=X" every 25 examples.
PROTEIN_RL_LOG_DIR = "BioReason-Pro/outputs/slurm/eval_id_ood_rl"
PROTEIN_RL_RUNNING_RE = re.compile(r"\[eval_only\]\s+(\d+)/\d+\s+running_f1=([0-9.]+)")
PROTEIN_RL_LOG_TAG_RE = re.compile(
    r"protein-rl-eval-(?P<tag>(?:warm-|klstrong-)?protein-grpo-epochs\d+-data\d+pct(?:-T\d+-DEBUG)?)-(?P<split>id|ood)_\d+\.out$"
)


def _latest_running_f1_per_runsplit(repo_root: Path) -> dict[tuple[str, str], dict]:
    """Scan eval_id_ood_rl/*.out and return the most-advanced running_f1 per
    (run_tag, split). 'Most advanced' = largest n; ties broken by latest mtime.
    Returns {(run_tag, split): {"running_f1": ..., "n": ..., "log": ...}}."""
    log_dir = repo_root / PROTEIN_RL_LOG_DIR
    out: dict[tuple[str, str], dict] = {}
    if not log_dir.is_dir():
        return out
    for log in sorted(log_dir.glob("protein-rl-eval-*.out")):
        m = PROTEIN_RL_LOG_TAG_RE.search(log.name)
        if not m:
            continue
        tag, split = m.group("tag"), m.group("split")
        text = tail_text(log)
        matches = PROTEIN_RL_RUNNING_RE.findall(text)
        if not matches:
            continue
        # The last matched (n, running_f1) is the most-recent line in this log.
        n_str, f1_str = matches[-1]
        n_here = int(n_str)
        prev = out.get((tag, split))
        if prev is None or n_here > prev["n"]:
            out[(tag, split)] = {"running_f1": float(f1_str), "n": n_here, "log": str(log)}
    return out


def _train_peak_reward(run_dir: Path, window: int = 50) -> Optional[dict]:
    """Read <run_dir>/train_metrics.jsonl and return the peak `window`-step
    rolling mean of mean_reward, plus the step at which it occurred and the
    most recent step. Returns None if the file is absent or empty."""
    p = run_dir / "train_metrics.jsonl"
    if not p.is_file():
        return None
    try:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    except Exception:
        return None
    rewards = [float(r.get("mean_reward", 0.0)) for r in rows]
    steps = [int(r.get("step", 0)) for r in rows]
    if len(rewards) < window:
        return None
    peak = -float("inf")
    peak_idx = 0
    cum = sum(rewards[:window])
    for i in range(window - 1, len(rewards)):
        if i >= window:
            cum += rewards[i] - rewards[i - window]
        avg = cum / window
        if avg > peak:
            peak = avg
            peak_idx = i
    last_avg = sum(rewards[-window:]) / window
    return {
        "peak_reward": peak,
        "peak_step": steps[peak_idx],
        "last_reward": last_avg,
        "last_step": steps[-1],
        "n_steps": len(rewards),
    }


def harvest_protein_rl(repo_root: Path | None = None) -> list[dict]:
    root = Path(PROTEIN_CKPT_ROOT)
    out: list[dict] = []
    if not root.is_dir():
        return out
    # Pre-scan eval logs once so we can attach a running_f1 fallback per (run, split).
    running = _latest_running_f1_per_runsplit(repo_root or Path(REPO_DEFAULT))
    # Glob base ("protein-grpo-..."), warm-started ("warm-protein-grpo-..."),
    # and KL-strength ablation ("klstrong-protein-grpo-...") variants.
    candidates = (
        list(root.glob("protein-grpo-epochs*-data*pct"))
        + list(root.glob("warm-protein-grpo-epochs*-data*pct"))
        + list(root.glob("klstrong-protein-grpo-epochs*-data*pct"))
    )
    for run_dir in sorted(set(candidates)):
        m = PROTEIN_RL_RUN_RE.match(run_dir.name)
        if not m:
            continue
        epoch, pct = int(m.group("epoch")), int(m.group("pct"))
        prefix = m.group("prefix")
        init = (
            "klstrong" if prefix == "klstrong-"
            else "warm" if prefix == "warm-"
            else "base"
        )
        train_peak = _train_peak_reward(run_dir)
        for split in ("id", "ood"):
            metrics_path = run_dir / f"_eval_{split}" / f"eval_{split}_metrics.json"
            f1 = pr = rc = n = None
            partial = False
            if metrics_path.is_file():
                try:
                    data = json.loads(metrics_path.read_text())
                except Exception:
                    data = {}
                f1 = data.get(f"eval_{split}/f1")
                pr = data.get(f"eval_{split}/precision")
                rc = data.get(f"eval_{split}/recall")
                n = data.get(f"eval_{split}/n_examples")
            if f1 is None:
                # Fall back to the most-advanced running_f1 from the eval logs,
                # so partial in-progress evals show up in the table.
                rfi = running.get((run_dir.name, split))
                if rfi is not None and rfi["n"] >= 50:
                    f1 = rfi["running_f1"]
                    n = rfi["n"]
                    partial = True
            if f1 is None:
                continue
            out.append({
                "domain": "protein_rl",
                "init": init,
                "epoch": epoch,
                "data_pct": pct,
                "split": split,
                "precision": pr,
                "recall": rc,
                "f1": f1,
                "n": n,
                "metrics_path": str(metrics_path),
                "train_peak": train_peak,
                "partial": partial,
            })
        # Even when no eval metrics exist, surface the train trajectory so the
        # user can see warm runs are healthy in advance of FINAL F1 landing.
        # Only emit a "training-only" row if we got nothing for either split.
        had_eval = any(
            r["init"] == init and r["epoch"] == epoch and r["data_pct"] == pct
            for r in out
        )
        if not had_eval and train_peak is not None:
            out.append({
                "domain": "protein_rl",
                "init": init,
                "epoch": epoch,
                "data_pct": pct,
                "split": "train_only",
                "f1": None,
                "n": None,
                "train_peak": train_peak,
                "metrics_path": str(run_dir / "train_metrics.jsonl"),
            })
    return out


# ---------- markdown rendering -------------------------------------------------

def render_dna_sft_table(rows: list[dict]) -> str:
    by_key: dict[tuple[str, int], dict] = defaultdict(dict)
    for r in rows:
        by_key[(r["model"], r["epoch"])][r["split"]] = r["accuracy"]
    if not by_key:
        return "_no DNA SFT eval results yet_\n"
    lines = ["| model | epoch | acc_id | acc_ood | gap |", "|---|---|---|---|---|"]
    for (model, epoch) in sorted(by_key.keys(), key=lambda t: (t[0], t[1])):
        row = by_key[(model, epoch)]
        id_ = row.get("id"); ood_ = row.get("ood")
        gap = (id_ - ood_) if (id_ is not None and ood_ is not None) else None
        lines.append(f"| {model} | {epoch} | {fmt(id_)} | {fmt(ood_)} | {fmt(gap)} |")
    return "\n".join(lines) + "\n"


def render_dna_rl_table(rows: list[dict]) -> str:
    by_key: dict[tuple[str, int], dict] = defaultdict(dict)
    for r in rows:
        # Pull eval_<split>/correctness (or rewards/correctness/mean) out of the metrics dict.
        m = r["metrics"]
        candidates = [
            f"eval_{r['split']}/rewards/correctness/mean",
            f"eval_{r['split']}/correctness",
            f"eval_{r['split']}_correctness",
            f"eval_{r['split']}/reward",
        ]
        score = next((m[k] for k in candidates if k in m), None)
        by_key[(r["model"], r["epoch"])][r["split"]] = score
    if not by_key:
        return "_no DNA RL eval results yet_\n"
    lines = ["| model | epoch | correctness_id | correctness_ood | gap |", "|---|---|---|---|---|"]
    for (model, epoch) in sorted(by_key.keys(), key=lambda t: (t[0], t[1])):
        row = by_key[(model, epoch)]
        id_ = row.get("id"); ood_ = row.get("ood")
        gap = (id_ - ood_) if (id_ is not None and ood_ is not None) else None
        lines.append(f"| {model} | {epoch} | {fmt(id_)} | {fmt(ood_)} | {fmt(gap)} |")
    return "\n".join(lines) + "\n"


def render_protein_sft_table(rows: list[dict]) -> str:
    by_pct: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pct[r["data_pct"]][r["split"]] = r
    if not by_pct:
        return "_no protein SFT eval results yet_\n"
    lines = ["_`*` marks partial running_f1 from in-flight evals (n shown)._\n",
             "| data % | id F1 | ood F1 | gap (F1) |",
             "|---|---|---|---|"]
    for pct in sorted(by_pct.keys()):
        row = by_pct[pct]
        idr = row.get("id"); odr = row.get("ood")
        id_ = idr["f1"] if idr else None; ood_ = odr["f1"] if odr else None
        gap = (id_ - ood_) if (id_ is not None and ood_ is not None) else None
        lines.append(f"| {pct} | {_f1_cell(idr)} | {_f1_cell(odr)} | {fmt(gap)} |")
    return "\n".join(lines) + "\n"


def _peak_reward_cell(r: Optional[dict]) -> str:
    """Render the train_peak rolling-mean as a cell, e.g. `0.85@350`."""
    if not r or "train_peak" not in r or r["train_peak"] is None:
        return "—"
    tp = r["train_peak"]
    return f"{tp['peak_reward']:.2f}@{tp['peak_step']}"


def _f1_cell(r: Optional[dict]) -> str:
    """Render an F1 cell, suffixing `*` and the running n for partial (mid-eval) values."""
    if not r:
        return "—"
    f1 = r.get("f1")
    if f1 is None:
        return "—"
    if r.get("partial"):
        return f"{f1:.3f}* (n={r.get('n')})"
    return f"{f1:.3f}"


def render_protein_rl_tables(rows: list[dict]) -> str:
    parts: list[str] = []
    # Render base- and warm-init variants in parallel sections so a degenerate
    # base sweep doesn't visually swamp the SFT-warm-started one.
    inits = sorted({r.get("init", "base") for r in rows})
    for init in inits:
        sub = [r for r in rows if r.get("init", "base") == init]
        epoch_rows = [r for r in sub if r["data_pct"] == 20]
        data_rows = [r for r in sub if r["epoch"] == 1]
        legend = {
            "base": "bare 4B-Thinking, KL_BETA=1e-4",
            "warm": "SFT-LoRA-warm-started, KL_BETA=1e-4",
            "klstrong": "SFT-LoRA-warm-started, KL_BETA=1e-3 (KL-strength ablation)",
        }.get(init, init)
        header = f"### Init = `{init}` ({legend})\n"
        local: list[str] = [header]
        if epoch_rows:
            by_e: dict[int, dict] = defaultdict(dict)
            for r in epoch_rows:
                by_e[r["epoch"]][r["split"]] = r
            local.append("**Epoch sweep (data=20%)** — `*` marks partial running_f1 from in-flight evals (n shown).\n")
            local.append("| epoch | id F1 | ood F1 | gap | peak50@step |")
            local.append("|---|---|---|---|---|")
            for e in sorted(by_e.keys()):
                idr = by_e[e].get("id"); odr = by_e[e].get("ood")
                id_ = idr["f1"] if idr else None; ood_ = odr["f1"] if odr else None
                gap = (id_ - ood_) if (id_ is not None and ood_ is not None) else None
                # train_peak is the same for both splits of a run; pick whichever exists
                peak_src = idr or odr or by_e[e].get("train_only")
                local.append(f"| {e} | {_f1_cell(idr)} | {_f1_cell(odr)} | {fmt(gap)} | {_peak_reward_cell(peak_src)} |")
            local.append("")
        if data_rows:
            by_p: dict[int, dict] = defaultdict(dict)
            for r in data_rows:
                by_p[r["data_pct"]][r["split"]] = r
            local.append("**Data sweep (epoch=1)** — `*` marks partial running_f1 from in-flight evals (n shown).\n")
            local.append("| data % | id F1 | ood F1 | gap | peak50@step |")
            local.append("|---|---|---|---|---|")
            for p in sorted(by_p.keys()):
                idr = by_p[p].get("id"); odr = by_p[p].get("ood")
                id_ = idr["f1"] if idr else None; ood_ = odr["f1"] if odr else None
                gap = (id_ - ood_) if (id_ is not None and ood_ is not None) else None
                peak_src = idr or odr or by_p[p].get("train_only")
                local.append(f"| {p} | {_f1_cell(idr)} | {_f1_cell(odr)} | {fmt(gap)} | {_peak_reward_cell(peak_src)} |")
            local.append("")
        if len(local) > 1:
            parts.extend(local)
    if not parts:
        return "_no protein RL eval results yet_\n"
    return "\n".join(parts) + "\n"


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


# ---------- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=REPO_DEFAULT)
    p.add_argument("--out_md", default=None)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    root = Path(args.root)
    out_md = Path(args.out_md or root / "figures" / "results_id_ood.md")
    out_json = Path(args.out_json or root / "figures" / "results_id_ood.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    dna_sft = harvest_dna_sft(root)
    dna_rl = harvest_dna_rl(root)
    prot_sft = harvest_protein_sft(root)
    prot_rl = harvest_protein_rl()

    md = ["# ID / OOD evaluation results", "",
          f"_harvested {os.popen('date -u').read().strip()}_", ""]
    md += ["## Protein SFT (data scaling, F1)", "", render_protein_sft_table(prot_sft)]
    md += ["## Protein RL / GRPO", "", render_protein_rl_tables(prot_rl)]
    md += ["## DNA SFT (CPT epoch sweep, accuracy)", "", render_dna_sft_table(dna_sft)]
    md += ["## DNA RL (CPT epoch sweep, correctness reward)", "", render_dna_rl_table(dna_rl)]

    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps({
        "dna_sft": dna_sft, "dna_rl": dna_rl,
        "protein_sft": prot_sft, "protein_rl": prot_rl,
    }, indent=2, default=str))

    print(f"[harvest] wrote {out_md}")
    print(f"[harvest] wrote {out_json}")
    print(f"[harvest] counts: dna_sft={len(dna_sft)} dna_rl={len(dna_rl)} "
          f"protein_sft={len(prot_sft)} protein_rl={len(prot_rl)}")


if __name__ == "__main__":
    sys.exit(main())
