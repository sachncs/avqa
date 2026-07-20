"""xFormers interop for AVQA (spec §3.16 alternative).

Wraps ``xformers.ops.memory_efficient_attention`` when available.
Falls back to AVQA's :class:`TorchBackend` otherwise.
"""

from __future__ import annotations

import torch

from avqa.backend import TorchBackend


def is_xformers_available() -> bool:
    """Return True iff ``xformers`` is importable."""
    try:
        import xformers  # noqa: F401
    except ImportError:
        return False
    return True


def xformers_interop(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Drop-in wrapper around ``xformers.ops.memory_efficient_attention`` (spec §3.16).

    Falls back to AVQA's :class:`TorchBackend` when xformers is missing.

    Args:
        query: ``[B, H, T, D]`` (AVQA layout).
        key: ``[B, H, T, D]``.
        value: ``[B, H, T, D]``.

    Returns:
        Attention output with the same layout.
    """
    if not is_xformers_available() or not torch.cuda.is_available():
        return TorchBackend().naive_attention(query, key, value)

    import xformers.ops as xops  # type: ignore[import-not-found]

    return xops.memory_efficient_attention(query, key, value)  # type: ignore[no-any-return]


__all__ = ["is_xformers_available", "xformers_interop"]
