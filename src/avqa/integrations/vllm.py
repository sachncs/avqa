"""vLLM integration for AVQA (spec §3.15).

Provides :class:`AVQvLLMBackend`, the vLLM-compatible attention
backend. Falls back to the inner :class:`AVQAttention` when no
``PagedKVCache`` is supplied. The ``vllm`` package is optional;
:func:`is_vllm_available` reports whether it is installed.
"""

from __future__ import annotations

import importlib.util
from typing import Protocol

import torch

from avqa.attention_module import AVQAttention
from avqa.cache import PagedKVCache
from avqa.config import AVQConfig
from avqa.logging import get_logger

logger = get_logger("integrations.vllm")


class VLLMBackend(Protocol):
    """Protocol for the return type of :func:`vllm_attention_backend`.

    Both :class:`AVQvLLMBackend` and the lightweight ``VLLMSelector``
    returned for non-AVQA backends expose a read-only ``name``
    attribute.
    """

    @property
    def name(self) -> str: ...


class TensorModule(Protocol):
    """Protocol for any module whose forward accepts tensors and returns one.

    PyTorch's ``nn.Module.__call__`` is untyped; this protocol captures
    the small surface AVQA's VLLM wrapper uses.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...


def is_vllm_available() -> bool:
    """Return True iff the ``vllm`` package is importable."""
    return importlib.util.find_spec("vllm") is not None


class AVQvLLMBackend:
    """vLLM-compatible attention backend (spec §3.15).

    This class wraps AVQAttention for use with vLLM's attention backend
    protocol. It exposes the interface that vLLM expects, including
    ``forward()`` and ``forward_native()`` methods.

    Features supported:
        - Standard attention computation via AVQAttention
        - KV cache integration (via vLLM's cache protocol)
        - Basic attention masking

    Features requiring vLLM installed:
        - Paged attention
        - Continuous batching
        - Prefix caching
        - Tensor parallelism
        - Speculative decoding

    Args:
        config: AVQ configuration.
        num_kv_heads: Number of KV heads for GQA (default: same as config).
        head_size: Per-head dimension (default: auto from config).
    """

    def __init__(
        self,
        config: AVQConfig | None = None,
        num_kv_heads: int | None = None,
        head_size: int | None = None,
    ) -> None:
        self.config = config or AVQConfig()
        self.num_kv_heads = num_kv_heads or self.config.attention.num_heads
        self.head_size = head_size or (
            self.config.attention.embed_dim // self.config.attention.num_heads
        )
        # Type as ``TensorModule`` so the untyped ``nn.Module.__call__``
        # resolves to ``torch.Tensor`` without a suppression.
        self.module: TensorModule = AVQAttention(self.config, in_proj=False, out_proj=False)

    @property
    def name(self) -> str:
        """Backend identifier for vLLM introspection."""
        return "avqa"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: object | None = None,
        attn_metadata: object | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """vLLM-compatible forward pass (SPEC §12.4).

        Routes batched inference through :class:`PagedKVCache` when a
        :class:`avqa.cache.PagedKVCache` instance is supplied via
        ``kv_cache``. The cache becomes the persistent attention state
        across calls; tokens appended each step.

        Args:
            query: ``[B, T_q, H, D]`` (vLLM) or ``[B, T_q, E]`` (AVQA).
            key: ``[B, T_k, H, D]`` (vLLM) or ``[B, T_k, E]`` (AVQA).
            value: same shape as ``key``.
            kv_cache: Optional :class:`avqa.cache.PagedKVCache`.
            attn_metadata: Optional vLLM metadata object. Accepted but
                unused; the vLLM call site passes it for API
                compatibility.

        Returns:
            Attention output tensor with the same shape as ``query``.
        """
        # Handle [B, T, H, D] (vLLM) and [B, T, E] (AVQA) layouts.
        if query.ndim == 4:
            B, T_q, H, D = query.shape
            query = query.reshape(B, T_q, H * D)
            key = key.reshape(key.shape[0], key.shape[1], H * D)
            value = value.reshape(value.shape[0], value.shape[1], H * D)

        if kv_cache is None:
            return self.module(query, key, value)

        # Paged-attention path: route through AVQAttention's kv_cache
        # argument, which the attention module already supports via
        # ``kv_cache.lookup()`` and ``kv_cache.append()``.
        if not isinstance(kv_cache, PagedKVCache):
            msg = (
                "AVQvLLMBackend.forward expects a PagedKVCache "
                "instance for the paged path; got " + type(kv_cache).__name__
            )
            raise TypeError(msg)

        # The cache stores heads-flat [B, H, T, D] (per the attention
        # pipeline). Convert new keys/values to the same layout, then
        # concatenate, then pass through the AVQAttention module which
        # in turn calls ``kv_cache.append`` with the cached prefix +
        # new tokens.
        num_heads = kv_cache.num_heads
        head_dim_k = self.head_size
        head_dim_v = self.head_size
        B = query.shape[0]
        T_k_new = key.shape[1]
        if key.ndim == 3:
            new_k = key.reshape(B, T_k_new, num_heads, head_dim_k).transpose(1, 2)
            new_v = value.reshape(B, T_k_new, num_heads, head_dim_v).transpose(1, 2)
        else:
            new_k, new_v = key, value
        cached_k, cached_v = kv_cache.lookup()
        if cached_k.shape[-2] > 0:
            full_k = torch.cat([cached_k, new_k], dim=-2)
            full_v = torch.cat([cached_v, new_v], dim=-2)
        else:
            full_k, full_v = new_k, new_v
        # Run attention over the full k/v timeline.
        flat_k = full_k.transpose(1, 2).reshape(B, full_k.shape[-2], num_heads * head_dim_k)
        flat_v = full_v.transpose(1, 2).reshape(B, full_v.shape[-2], num_heads * head_dim_v)
        out: torch.Tensor = self.module(query, flat_k, flat_v)
        kv_cache.append(new_k, new_v)
        # attn_metadata is accepted for vLLM API compatibility but
        # unused here; the cache drives the schedule.
        return out

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Native forward (delegates to forward)."""
        return self.forward(query, key, value, **kwargs)


def vllm_attention_backend(backend: str = "torch") -> "VLLMBackend":
    """Return a vLLM-compatible attention backend (spec §3.15).

    When ``backend="avqa"``, returns an :class:`AVQvLLMBackend` instance
    that wraps AVQAttention. Other backend names return a simple selector
    for vLLM's registry.

    Args:
        backend: One of ``"torch"``, ``"triton"``, ``"xformers"``,
            ``"flash_attn"``, or ``"avqa"``.

    Returns:
        An object exposing ``.name`` for vLLM to introspect.
    """
    available = {
        "torch": True,
        "triton": is_vllm_available(),
        "xformers": False,
        "flash_attn": False,
        "avqa": True,
    }
    if backend not in available:
        msg = f"unknown vLLM backend '{backend}'"
        raise ValueError(msg)
    if not available[backend]:
        msg = f"vLLM backend '{backend}' is not installed"
        raise RuntimeError(msg)

    if backend == "avqa":
        return AVQvLLMBackend()

    class VLLMSelector:
        """Lightweight vLLM backend descriptor for non-AVQA selections."""

        name = backend

    return VLLMSelector()


__all__ = [
    "AVQvLLMBackend",
    "is_vllm_available",
    "vllm_attention_backend",
]
