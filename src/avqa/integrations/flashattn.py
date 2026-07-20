"""FlashAttention interop for AVQA (spec §3.16).

Wraps ``flash_attn.flash_attn_func`` when available; falls back to
AVQA's reference backend otherwise. :func:`is_flash_attention_available`
reports the runtime state.
"""

from __future__ import annotations

import torch


def is_flash_attention_available() -> bool:
    """Return True iff ``flash_attn`` is importable."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


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
        from avqa.backend import TorchBackend

        # Convert HF layout [B, T, H, D] -> [B, H, T, D] for the backend.
        def to_avqa(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous()

        def from_avqa(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous()

        backend = TorchBackend()
        out = backend.naive_attention(to_avqa(query), to_avqa(key), to_avqa(value))
        return from_avqa(out)

    # flash_attn_func expects [B, T, H, D] directly.
    import flash_attn

    out = flash_attn.flash_attn_func(query, key, value, causal=False)
    return out  # type: ignore[no-any-return]


__all__ = ["flash_attention_interop", "is_flash_attention_available"]
