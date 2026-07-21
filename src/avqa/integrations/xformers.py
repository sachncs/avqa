"""xFormers interop for AVQA (spec §3.16 alternative).

Wraps ``xformers.ops.memory_efficient_attention`` when available.
Falls back to AVQA's :class:`TorchBackend` otherwise.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import torch

from avqa.backend import TorchBackend


def is_xformers_available() -> bool:
    """Return True iff ``xformers`` is importable."""
    return importlib.util.find_spec("xformers") is not None


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

    if TYPE_CHECKING:
        try:
            from xformers import ops as xops
        except ImportError as exc:
            msg = 'xformers is not installed'
            raise ImportError(msg) from exc
    else:
        try:
            import xformers.ops as xops
        except ImportError as exc:
            msg = 'xformers is not installed'
            raise ImportError(msg) from exc

    result: torch.Tensor = xops.memory_efficient_attention(query, key, value)
    return result


__all__ = ["is_xformers_available", "xformers_interop"]
