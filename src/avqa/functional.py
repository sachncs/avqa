"""Functional stateless API for AVQA (spec §3.5, §5.7).

The functional API is a thin wrapper around :class:`AVQAttention` that
constructs an internal module per call. It is intended for quick
experimentation; for repeated use, instantiate an AVQAttention module
directly to amortize parameter initialization.

ponytail: the functional API is a single function. Spec §3.5 requires
"no internal state"; we construct a fresh module on each call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from avqa.attention_module import AVQAttention
from avqa.config import AVQConfig

if TYPE_CHECKING:
    from avqa.cache import KVCache


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: AVQConfig,
    mask: torch.Tensor | None = None,
    kv_cache: "KVCache | None" = None,
) -> torch.Tensor:
    """Stateless attention (spec §3.5, §5.7).

    Note:
        The functional API is not strictly stateless when ``kv_cache``
        is supplied — the cache is mutated in place. This is the only
        way to drive incremental decoding through the functional entry
        point; for fully stateless use, instantiate
        :class:`AVQAttention` directly.

    Args:
        query: ``[B, T_q, E]``.
        key: ``[B, T_k, E]``.
        value: ``[B, T_k, E]``.
        config: :class:`AVQConfig`.
        mask: Optional boolean mask (``[T_q, T_k]`` or ``[B, H, T_q, T_k]``).
        kv_cache: Optional KV-cache to extend in place.

    Returns:
        ``[B, T_q, E]`` attention output.

    Example:
        >>> import torch
        >>> from avqa import AVQConfig
        >>> from avqa.functional import attention
        >>> config = AVQConfig()
        >>> q = torch.randn(2, 8, 64)
        >>> k = torch.randn(2, 16, 64)
        >>> v = torch.randn(2, 16, 64)
        >>> out = attention(q, k, v, config)
        >>> out.shape
        torch.Size([2, 8, 64])
    """
    module = AVQAttention(config, in_proj=False, out_proj=False)
    out: torch.Tensor = module(query, key, value, mask=mask, kv_cache=kv_cache)
    return out


__all__ = ["attention"]
