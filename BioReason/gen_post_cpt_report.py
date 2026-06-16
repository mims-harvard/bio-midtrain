#!/usr/bin/env python3
"""Compose a final Base / SFT / RL / CPT+SFT / CPT+RL report.

SFT rows: parsed from `Accuracy: <float>` lines in e*_*.out (held-out test set).
RL rows: parsed from the last `correctness_reward_func/mean` value in e*_*.out.
         This is a TRAINING-SET reward proxy, NOT held-out test accuracy.
CPT row: eval_loss / perplexity from final_eval_metrics.json.

Run after the post_cpt sweep RL jobs complete.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


ACC_RE = re.compile(r"^Accuracy:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
RL_REWARD_RE = re.compile(r"'train/rewards/correctness_reward_func/mean':\s*([0-9.]+)")


def best_sft_acc(log_dir: str) -> tuple[str | None, float | None]:
    """Return (best_epoch_label, best_accuracy) by scanning e*_*.out logs."""
    best_ep, best_acc = None, None
    for f in sorted(glob.glob(os.path.join(log_dir, "e*_*.out"))):
        ep = re.match(r"^(e\d+)_\d+\.out$", os.path.basename(f))
        if not ep:
            continue
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        accs = ACC_RE.findall(txt)
        if not accs:
            continue
        last = float(accs[-1])
        if best_acc is None or last > best_acc:
            best_acc = last
            best_ep = ep.group(1)
    return best_ep, best_acc


def last_rl_reward(log_dir: str) -> dict:
    """Per-epoch last correctness_reward_func/mean. {ep_label: reward}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(log_dir, "e*_*.out"))):
        ep = re.match(r"^(e\d+)_\d+\.out$", os.path.basename(f))
        if not ep:
            continue
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        m = RL_REWARD_RE.findall(txt)
        if m:
            # take latest (most recent log if multiple files for same ep)
            key = ep.group(1)
            out[key] = max(out.get(key, -1), float(m[-1]))
    return out


def best_rl(log_dir: str) -> tuple[str | None, float | None]:
    rewards = last_rl_reward(log_dir)
    if not rewards:
        return None, None
    best_ep = max(rewards, key=rewards.get)
    return best_ep, rewards[best_ep]


def cpt_metrics(cpt_root: str) -> dict:
    """Aggregate per-config final_eval_loss/perplexity for the CPT sweep."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cpt_root, "*", "final_eval_metrics.json"))):
        cfg = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
            out[cfg] = {
                "eval_loss": d.get("eval_loss"),
                "ppl": d.get("eval_perplexity"),
            }
        except Exception:
            continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_root", default=os.path.expanduser("~/projects/evo_omics/BioReason/logs/drive/sweeps"))
    p.add_argument("--cpt_root", default="/n/holylfs06/LABS/mzitnik_lab/Lab/hanlinzhang/evo_tfm/BioReason/checkpoints/drive/cpt_ffw_20260424_165721")
    p.add_argument("--base_tag_1p7b", default="qwen3_1p7b")
    p.add_argument("--base_tag_4b", default="qwen3_4b")
    p.add_argument("--cpt_tag_1p7b", default="cpt_qwen3_1p7b")
    p.add_argument("--cpt_tag_4b", default="cpt_qwen3_4b")
    p.add_argument("--out", required=True, help="output markdown path")
    args = p.parse_args()

    rows = {}  # (variant, model) -> (ep_label, value, kind)
    for variant_label, kind, base_tag, cpt_tag in [
        ("SFT", "sft", args.base_tag_1p7b, None),
        ("RL", "rl", args.base_tag_1p7b, None),
        ("CPT+SFT", "sft", None, args.cpt_tag_1p7b),
        ("CPT+RL", "rl", None, args.cpt_tag_1p7b),
    ]:
        tag = base_tag or cpt_tag
        log_dir = os.path.join(args.logs_root, kind, tag)
        if kind == "sft":
            ep, val = best_sft_acc(log_dir)
        else:
            ep, val = best_rl(log_dir)
        rows[(variant_label, "1.7B")] = (ep, val, kind)
    for variant_label, kind, base_tag, cpt_tag in [
        ("SFT", "sft", args.base_tag_4b, None),
        ("RL", "rl", args.base_tag_4b, None),
        ("CPT+SFT", "sft", None, args.cpt_tag_4b),
        ("CPT+RL", "rl", None, args.cpt_tag_4b),
    ]:
        tag = base_tag or cpt_tag
        log_dir = os.path.join(args.logs_root, kind, tag)
        if kind == "sft":
            ep, val = best_sft_acc(log_dir)
        else:
            ep, val = best_rl(log_dir)
        rows[(variant_label, "4B")] = (ep, val, kind)

    cpt_m = cpt_metrics(args.cpt_root)

    md_lines = []
    md_lines.append(f"# Final BioReason Comparison Report")
    md_lines.append("")
    md_lines.append(f"_Generated: {os.popen('date -Iseconds').read().strip()}_")
    md_lines.append("")
    md_lines.append("## Headline table")
    md_lines.append("")
    md_lines.append("| Variant | Qwen3-1.7B | Qwen3-4B | Metric |")
    md_lines.append("|---|---|---|---|")
    md_lines.append("| Base (zero-shot) | TBD | TBD | held-out accuracy (need vllm eval) |")
    for v in ("SFT", "RL", "CPT+SFT", "CPT+RL"):
        cells = []
        for size in ("1.7B", "4B"):
            ep, val, kind = rows[(v, size)]
            if val is None:
                cells.append("(no data)")
            elif kind == "sft":
                cells.append(f"{val:.4f} @{ep}")
            else:
                cells.append(f"reward={val:.3f} @{ep}")
        metric = "held-out accuracy" if v.endswith("SFT") else "training reward proxy (NOT held-out)"
        md_lines.append(f"| {v} | {cells[0]} | {cells[1]} | {metric} |")
    md_lines.append("")
    md_lines.append("> Note: Base zero-shot and held-out RL accuracy require running `eval_kegg_dna_vllm.py` "
                    "on the relevant checkpoints. RL rows above use last-step training reward as a rough proxy only.")
    md_lines.append("")
    md_lines.append("## CPT eval (mid-training perplexity on FineFineWeb biology)")
    md_lines.append("")
    md_lines.append("| Config | eval_loss | perplexity |")
    md_lines.append("|---|---|---|")
    for cfg, m in sorted(cpt_m.items()):
        el = m.get("eval_loss")
        pp = m.get("ppl")
        md_lines.append(f"| `{cfg}` | {el:.4f} | {pp:.4f} |" if el is not None and pp is not None else f"| `{cfg}` | — | — |")
    md_lines.append("")
    Path(args.out).write_text("\n".join(md_lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
