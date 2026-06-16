#!/usr/bin/env python3
"""Pick the GRPO checkpoint with the highest rolling-avg train reward.

Reads <CKPT_DIR>/train_metrics.jsonl and chooses the checkpoint-N directory
whose step is closest to the global argmax of a 50-step rolling-mean of
mean_reward. Falls back to <CKPT_DIR>/final, then to the latest
checkpoint-N, then to <CKPT_DIR> itself if none of those exist.

Usage in a SLURM eval script:
    EVAL_CKPT=$(python BioReason-Pro/scripts/pick_best_ckpt.py "$CKPT_DIR")

Writes the absolute path to stdout. On any error (missing dir, no ckpts),
exits 1 with a diagnostic on stderr.

Why this exists: with the SFT-warm-started GRPO sweep, some configs (e.g.
warm-protein-grpo-epochs1-data80pct) reach reward ≈0.7 mid-training and
then drift back to 0.0 by `final/` because KL_BETA=1e-4 is too weak for
multi-epoch runs. Evaluating final/ throws away the useful trajectory.
This picker recovers the best ckpt without retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WINDOW = 50  # rolling-mean window for choosing the peak


def parse_step_from_ckpt(p: Path) -> int | None:
    name = p.name
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


def pick(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[pick_best_ckpt] CKPT_DIR not found: {ckpt_dir}")

    # Available checkpoint-N dirs.
    available = []
    for p in ckpt_dir.iterdir():
        s = parse_step_from_ckpt(p)
        if s is not None and p.is_dir():
            available.append((s, p))
    available.sort()

    # Score every saved checkpoint by the rolling-mean reward at its step
    # (window centred on that step, clipped at the boundaries). The picker
    # then chooses the saved checkpoint with the highest such rolling mean,
    # not the saved checkpoint nearest to a separately-derived peak — the
    # latter can land you on the *post-peak* side of a quick collapse.
    metrics_path = ckpt_dir / "train_metrics.jsonl"
    if metrics_path.is_file() and available:
        rows: list[dict] = []
        with metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            half = WINDOW // 2
            # Map step → list-index for fast lookup.
            step_to_idx = {int(r.get("step", -1)): i for i, r in enumerate(rows)}
            best_score = -float("inf")
            best_path: Path | None = None
            best_step: int | None = None
            for step, path in available:
                # Find the metrics row whose step is closest to this checkpoint step.
                idx = step_to_idx.get(step)
                if idx is None:
                    nearest = min(range(len(rows)),
                                  key=lambda i: abs(int(rows[i].get("step", 0)) - step))
                    idx = nearest
                lo = max(0, idx - half)
                hi = min(len(rows), idx + half + 1)
                window = rows[lo:hi]
                if not window:
                    continue
                avg = sum(float(r.get("mean_reward", 0.0)) for r in window) / len(window)
                if avg > best_score:
                    best_score = avg
                    best_path = path
                    best_step = step
            if best_path is not None:
                print(f"[pick_best_ckpt] best ckpt step={best_step} avg_reward={best_score:.4f} → {best_path.name}",
                      file=sys.stderr)
                return best_path

    # Fallbacks (no metrics file or no checkpoints): final/ → latest checkpoint-N → ckpt_dir.
    final = ckpt_dir / "final"
    if final.is_dir():
        print(f"[pick_best_ckpt] no metrics; using final/", file=sys.stderr)
        return final
    if available:
        print(f"[pick_best_ckpt] no metrics; using latest {available[-1][1].name}", file=sys.stderr)
        return available[-1][1]
    raise SystemExit(f"[pick_best_ckpt] No checkpoints under {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--no_metrics_ok", action="store_true",
                    help="If train_metrics.jsonl is absent, silently fall back to final/latest.")
    args = ap.parse_args()
    picked = pick(Path(args.ckpt_dir))
    print(picked.resolve())


if __name__ == "__main__":
    main()
