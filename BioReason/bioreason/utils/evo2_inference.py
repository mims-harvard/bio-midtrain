"""Run Evo2 (Vortex + TransformerEngine) forward outside Lightning bf16 autocast."""

from typing import Any, Dict, List

import torch


def run_evo2_embeddings(
    dna_model: Any,
    input_ids: torch.Tensor,
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """TorchDynamo must be off globally (set TORCHDYNAMO_DISABLE=1 before `import torch`)."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        _, embeddings = dna_model(
            input_ids,
            return_embeddings=True,
            layer_names=layer_names,
        )
    return embeddings
