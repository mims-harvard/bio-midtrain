"""
Force vortex to use its PyTorch Linear fallback (no Transformer Engine).

When BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1, any import of ``transformer_engine``
raises ImportError so ``vortex.model.layers`` sets HAS_TE=False. That avoids TE
``cublaslt_gemm`` failures under Lightning + DeepSpeed.

Call ``install_import_hook()`` before ``import torch`` and before importing evo2/vortex.
"""

from __future__ import annotations

import builtins
import os

_HOOK_INSTALLED = False


def is_enabled() -> bool:
    return os.environ.get("BIOREASON_USE_VORTEX_PYTORCH_LINEAR", "").lower() in (
        "1",
        "true",
        "yes",
    )


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED or not is_enabled():
        return
    _orig = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only block absolute imports of the external transformer_engine package.
        # Relative imports like ``from .transformer_engine import ...`` inside
        # accelerate should still work.
        if level == 0 and (name == "transformer_engine" or name.startswith("transformer_engine.")):
            raise ImportError(
                "transformer_engine blocked (BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1); "
                "vortex uses PyTorch Linear instead of Transformer Engine."
            )
        return _orig(name, globals, locals, fromlist, level)

    builtins.__import__ = _import
    _HOOK_INSTALLED = True
