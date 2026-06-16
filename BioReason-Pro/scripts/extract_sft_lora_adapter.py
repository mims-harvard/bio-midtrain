#!/usr/bin/env python3
"""Extract the text-model LoRA adapter from a Lightning SFT checkpoint and save it
as a HF PEFT-compatible directory (adapter_config.json + adapter_model.safetensors).

Why this exists: train_protein_grpo.py warm-starts via
    PeftModel.from_pretrained(base, args.sft_adapter_path)
which expects a PEFT-format adapter directory, not a Lightning .ckpt. This script
bridges the two. Use it once per SFT data fraction:

    python BioReason-Pro/scripts/extract_sft_lora_adapter.py \
        --ckpt   .../protein-sft-epochs1-data20pct/protein-sft-epochs1-data20pct-best-epoch=epoch=00-val=val_loss_epoch=0.9311.ckpt \
        --out    .../protein-sft-epochs1-data20pct/sft_lora_adapter

Then in sweep_protein_grpo.sh set:
    SFT_ADAPTER_PATH=".../protein-sft-epochs1-data20pct/sft_lora_adapter"

Lightning state_dict keys look like:
    model.text_model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
PEFT adapter_model.safetensors keys look like:
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

This script does a key-rename and writes a minimal adapter_config.json hard-coded
to the SFT hyperparameters used in BioReason-Pro
(r=128, alpha=256, dropout=0.05, all 7 attn+MLP target modules).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


LIGHTNING_PREFIX = "model.text_model."
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lightning_to_peft_key(k: str) -> str | None:
    """Map a Lightning state_dict key to a PEFT adapter key, or None if not a LoRA tensor."""
    if not k.startswith(LIGHTNING_PREFIX):
        return None
    if ".lora_" not in k:
        return None
    stripped = k[len(LIGHTNING_PREFIX):]
    return stripped.replace(".default.weight", ".weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file from SFT")
    ap.add_argument("--out", required=True, help="Output directory for PEFT adapter")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Thinking-2507",
                    help="Base model name to record in adapter_config.json")
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output dir {out_dir} is non-empty; refusing to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Loading Lightning ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt.get("state_dict", ckpt)

    adapter_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = lightning_to_peft_key(k)
        if nk is None:
            continue
        # detach + clone to drop any mmap binding before save_file copies bytes
        adapter_sd[nk] = v.detach().clone().contiguous()

    if not adapter_sd:
        raise SystemExit("No LoRA keys found under expected prefix; check ckpt structure.")
    print(f"[extract] Collected {len(adapter_sd)} LoRA tensors.")

    # PEFT adapter_config.json (fields match peft.LoraConfig defaults +
    # the values used in train_protein_llm.py SFT)
    adapter_cfg = {
        "auto_mapping": None,
        "base_model_name_or_path": args.base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": args.lora_rank,
        "revision": None,
        "target_modules": DEFAULT_TARGET_MODULES,
        "task_type": "CAUSAL_LM",
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f, indent=2)

    save_file(adapter_sd, str(out_dir / "adapter_model.safetensors"))
    print(f"[extract] Wrote {out_dir/'adapter_model.safetensors'} "
          f"({sum(v.numel() for v in adapter_sd.values())/1e6:.1f}M params)")
    print(f"[extract] Wrote {out_dir/'adapter_config.json'}")
    print(f"\nUse with GRPO via:  SFT_ADAPTER_PATH={out_dir.resolve()}")


if __name__ == "__main__":
    main()
