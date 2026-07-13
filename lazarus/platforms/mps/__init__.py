"""Lazarus native Metal (PyTorch-MPS) platform for vLLM.

Unlike vllm-metal (which swaps in an MLX runner and masquerades as CPU), this
platform runs vLLM's **native** torch models on the Apple GPU via PyTorch-MPS.
CustomOp.forward_native covers norm/rope/activation for free; attention is
served by a QuixiCore-backed Metal backend. See docs/metal-native-backend.md.

Registered out-of-tree via the ``vllm.platform_plugins`` entry point.
"""

from __future__ import annotations


def register() -> str | None:
    """vLLM platform-plugin entry point.

    Returns the fully-qualified platform class name when a usable Apple GPU is
    present, else None (so vLLM falls back to its builtin platform selection).
    """
    try:
        import torch
    except Exception:
        return None

    if not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return None

    # NOTE: keep register() lightweight. It runs during vLLM's platform
    # resolution (mid-import), and any exception here is silently swallowed
    # (platforms/__init__.py), which would drop us to the CPU platform. Heavy
    # imports / monkeypatches happen later, in MpsPlatform.check_and_update_config.
    return "lazarus.platforms.mps.platform.MpsPlatform"


__all__ = ["register"]
