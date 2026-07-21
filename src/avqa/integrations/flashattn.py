"""FlashAttention interop for AVQA (spec §3.16).

Wraps ``flash_attn.flash_attn_func`` when available; falls back to
AVQA's reference backend otherwise. :func:`is_flash_attention_available`
reports the runtime state.
"""

from __future__ import annotations

import importlib.util

import torch

from avqa.backend import TorchBackend


def is_flash_attention_available() -> bool:
    """Return True iff ``flash_attn`` is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def flash_attention_interop(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Drop-in wrapper around ``flash_attn_func`` when available (spec §3.16).

    Falls back to AVQA's :class:`TorchBackend` when flash-attn is missing
    or when CUDA is unavailable.

    Args:
        query: ``[B, T, H, D]`` (note: HF layout — heads before head dim).
        key: ``[B, T, H, D]``.
        value: ``[B, T, H, D]``.

    Returns:
        Attention output with the same layout.
    """
    if not is_flash_attention_available() or not torch.cuda.is_available():
        # Convert HF layout [B, T, H, D] -> [B, H, T, D] for the backend.
        def to_avqa(t: torch.Tensor) -> torch.Tensor:
            """Permute HF-style ``[B, T, H, D]`` to AVQA ``[B, H, T, D]``."""
            return t.transpose(1, 2).contiguous()

        def from_avqa(t: torch.Tensor) -> torch.Tensor:
            """Permute AVQA ``[B, H, T, D]`` back to HF-style ``[B, T, H, D]``."""
            return t.transpose(1, 2).contiguous()

        backend = TorchBackend()
        out = backend.naive_attention(to_avqa(query), to_avqa(key), to_avqa(value))
        return from_avqa(out)

    # flash_attn_func expects [B, T, H, D] directly.
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
    except ImportError as exc:
        msg = 'flash_attn is not installed'
        raise ImportError(msg) from exc

    flash_out: torch.Tensor = _flash_attn_func(query, key, value, causal=False)
    return flash_out


__all__ = ["flash_attention_interop", "is_flash_attention_available"]
