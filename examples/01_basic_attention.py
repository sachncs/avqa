"""AVQA reproducibly performs attention on random tensors.

Usage::

    python examples/01_basic_attention.py
"""

from __future__ import annotations

import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)


def main() -> None:
    """Run a single AVQAttention forward pass."""
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=128,
            num_heads=4,
            head_dim=32,
        ),
        codebook=CodebookConfig(
            num_codewords=32,
            children_per_codeword=4,
        ),
        routing=RoutingConfig(
            refinement_budget=8,
        ),
    )

    attention = AVQAttention(config)

    batch_size = 2
    seq_len = 16
    embed_dim = 128
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    output = attention(query, key, value)
    print(f"Input  shape: {tuple(query.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Sample output norm: {output.norm(dim=-1).mean().item():.4f}")


if __name__ == "__main__":
    main()
