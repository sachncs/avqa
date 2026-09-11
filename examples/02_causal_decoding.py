"""Causal-masked decoding example using InMemoryKVCache.

Usage::

    python examples/02_causal_decoding.py
"""

from __future__ import annotations

import torch

from avqa import AVQAttention, AVQConfig, InMemoryKVCache
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)


def main() -> None:
    """Autoregressive decoding with KV cache."""
    torch.manual_seed(0)
    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=128, num_heads=4, head_dim=32),
        codebook=CodebookConfig(num_codewords=32, children_per_codeword=4),
        routing=RoutingConfig(refinement_budget=4),
        causal=True,
    )
    attention = AVQAttention(config, in_proj=False, out_proj=False).eval()
    cache = InMemoryKVCache(num_heads=4, head_dim_k=32, head_dim_v=32, max_size=64)

    B, T, E = 1, 8, 128
    full_q = torch.randn(B, T, E)
    full_k = torch.randn(B, T, E)
    full_v = torch.randn(B, T, E)

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(T):
            step_q = full_q[:, step : step + 1, :]
            step_k = full_k[:, step : step + 1, :]
            step_v = full_v[:, step : step + 1, :]
            out = attention(step_q, step_k, step_v, kv_cache=cache)
            outputs.append(out)

    final = torch.cat(outputs, dim=1)
    print(f"Decoded shape: {tuple(final.shape)}")
    print(f"Cache stats: {cache.cache_stats()}")


if __name__ == "__main__":
    main()