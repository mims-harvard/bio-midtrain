"""
Convert checkpoints produced by ``train_protein_grpo.py`` (HF causal LM / PEFT adapters)
into weights compatible with ``ProteinLLMModel.text_model``.

Lightning / full ``ProteinLLMModel`` checkpoints (``pytorch_model.bin`` with ``text_model.*`` keys)
are handled separately in ``save_grpo_ckpt.py`` via direct ``load_state_dict``.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


class GrpoCheckpointKind(str, Enum):
    PROTEIN_LLM_PYTORCH_BIN = "protein_llm_pytorch_bin"
    HF_PEFT_ADAPTER = "hf_peft_adapter"
    HF_MERGED_CAUSAL_LM = "hf_merged_causal_lm"


def detect_grpo_checkpoint_kind(checkpoint_path: Union[str, Path]) -> GrpoCheckpointKind:
    """
    Distinguish:
    - Full ``ProteinLLMModel`` state dict (``text_model.*`` keys) in ``pytorch_model.bin``
    - ``train_protein_grpo`` PEFT output (``adapter_config.json``)
    - Merged HF causal LM directory (``config.json`` + weight shards, no adapter_config)
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {path}")

    if (path / "adapter_config.json").is_file():
        return GrpoCheckpointKind.HF_PEFT_ADAPTER

    bin_path = path / "pytorch_model.bin"
    config_path = path / "config.json"
    safetensors = list(path.glob("*.safetensors"))

    if bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            return GrpoCheckpointKind.HF_MERGED_CAUSAL_LM
        key0 = next(iter(state))
        if isinstance(key0, str) and key0.startswith("text_model."):
            return GrpoCheckpointKind.PROTEIN_LLM_PYTORCH_BIN
        return GrpoCheckpointKind.HF_MERGED_CAUSAL_LM

    if config_path.is_file() and safetensors:
        return GrpoCheckpointKind.HF_MERGED_CAUSAL_LM

    raise ValueError(
        f"Unrecognized checkpoint layout under {path}: "
        "expected adapter_config.json, or pytorch_model.bin, or config.json + *.safetensors."
    )


def load_merged_causal_lm(
    checkpoint_path: Union[str, Path],
    kind: GrpoCheckpointKind,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load a single causal LM (merged adapter or native HF save)."""
    cp = str(Path(checkpoint_path).resolve())
    if kind == GrpoCheckpointKind.HF_PEFT_ADAPTER:
        cfg_path = Path(cp) / "adapter_config.json"
        with cfg_path.open(encoding="utf-8") as f:
            acfg = json.load(f)
        base = acfg.get("base_model_name_or_path")
        if not base:
            raise KeyError("adapter_config.json missing base_model_name_or_path")
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, cp)
        return peft_model.merge_and_unload()

    if kind == GrpoCheckpointKind.HF_MERGED_CAUSAL_LM:
        return AutoModelForCausalLM.from_pretrained(
            cp,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    raise ValueError(f"load_merged_causal_lm does not accept kind={kind!r}")


def inject_merged_causal_into_text_model(
    dst_text_model: torch.nn.Module,
    src_causal: AutoModelForCausalLM,
) -> None:
    """
    Copy weights from a standalone causal LM into ``ProteinLLMModel.text_model``.

    When the destination tokenizer has extra special tokens (BioReason-Pro), embedding and
    lm_head rows are copied only for the shared prefix ``[0 : min(vocab_sizes))``.
    All other parameters are copied when shapes match exactly.
    """
    src_sd = src_causal.state_dict()
    dst_sd = dst_text_model.state_dict()

    with torch.no_grad():
        for name, src_t in src_sd.items():
            if name not in dst_sd:
                continue
            dst_t = dst_sd[name]
            if not torch.is_tensor(dst_t) or not torch.is_tensor(src_t):
                continue
            if dst_t.shape == src_t.shape:
                dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
            elif dst_t.dim() == 2 and src_t.dim() == 2 and dst_t.shape[1] == src_t.shape[1]:
                n = min(dst_t.shape[0], src_t.shape[0])
                dst_t[:n].copy_(src_t[:n].to(device=dst_t.device, dtype=dst_t.dtype))
            elif dst_t.dim() == 1 and src_t.dim() == 1:
                n = min(dst_t.shape[0], src_t.shape[0])
                dst_t[:n].copy_(src_t[:n].to(device=dst_t.device, dtype=dst_t.dtype))
