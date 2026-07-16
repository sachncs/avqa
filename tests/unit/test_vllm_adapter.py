"""vLLM paged-attention routing tests (TASK-12.002)."""

from __future__ import annotations

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.cache import PagedKVCache
from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig
from avqa.integrations import AVQvLLMBackend


def _make_module(embed_dim: int, num_heads: int) -> AVQAttention:
    """Build an AVQAttention module sized to (embed_dim, num_heads)."""
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=embed_dim, num_heads=num_heads, head_dim=embed_dim // num_heads
        ),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )
    mod = AVQAttention(config, in_proj=False, out_proj=False)
    mod.eval()
    return mod


@pytest.mark.parametrize(
    ("embed_dim", "num_heads"),
    [(64, 4), (128, 2)],
)
def test_vllm_adapter_routes_through_paged_cache(embed_dim: int, num_heads: int) -> None:
    """Paged cache concatenates the cached prefix and appends new tokens."""
    page_size = 16
    cache = PagedKVCache(
        page_size=page_size,
        num_heads=num_heads,
        head_dim_k=embed_dim // num_heads,
        head_dim_v=embed_dim // num_heads,
    )
    backend = AVQvLLMBackend(num_kv_heads=num_heads, head_size=embed_dim // num_heads)
    backend.module = _make_module(embed_dim, num_heads)
    # Match num_kv_heads and head_size used by the page cache.
    backend.num_kv_heads = num_heads
    backend.head_size = embed_dim // num_heads

    q1 = torch.randn(1, 1, embed_dim)
    k1 = torch.randn(1, 8, embed_dim)
    v1 = torch.randn(1, 8, embed_dim)
    backend.forward(q1, k1, v1, kv_cache=cache)
    assert cache.size == 8

    q2 = torch.randn(1, 1, embed_dim)
    k2 = torch.randn(1, 9, embed_dim)
    v2 = torch.randn(1, 9, embed_dim)
    out2 = backend.forward(q2, k2, v2, kv_cache=cache)
    assert out2.shape == q2.shape
    assert cache.size == 17
    assert cache.num_pages >= 2


def test_vllm_adapter_rejects_non_paged_cache() -> None:
    """Passing anything other than PagedKVCache raises a typed error."""
    cache = object()
    backend = AVQvLLMBackend(num_kv_heads=1, head_size=8)
    backend.module = _make_module(embed_dim=8, num_heads=1)
    backend.num_kv_heads = 1
    backend.head_size = 8
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 1, 8)
    v = torch.randn(1, 1, 8)
    with pytest.raises(TypeError, match="PagedKVCache"):
        backend.forward(q, k, v, kv_cache=cache)


def test_vllm_adapter_without_cache_runs_inline() -> None:
    """When kv_cache is None the adapter uses the inner module."""
    backend = AVQvLLMBackend(num_kv_heads=2, head_size=8)
    backend.module = _make_module(embed_dim=16, num_heads=2)
    backend.num_kv_heads = 2
    backend.head_size = 8
    q = torch.randn(1, 3, 16)
    k = torch.randn(1, 5, 16)
    v = torch.randn(1, 5, 16)
    out = backend.forward(q, k, v)
    assert out.shape == q.shape
