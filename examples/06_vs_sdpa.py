"""Compare AVQA against torch.nn.functional.scaled_dot_product_attention.

Usage::

    python examples/06_vs_sdpa.py
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
    """Run AVQA and SDPA on the same inputs and compare output."""
    torch.manual_seed(0)
    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=128, num_heads=4, head_dim=32),
        codebook=CodebookConfig(num_codewords=32, children_per_codeword=4),
        routing=RoutingConfig(refinement_budget=8),
    )
    avqa = AVQAttention(config, in_proj=False, out_proj=False).eval()

    B, T_q, T_k, E = 1, 16, 16, 128
    q = torch.randn(B, T_q, E)
    k = torch.randn(B, T_k, E)
    v = torch.randn(B, T_k, E)

    with torch.no_grad():
        out_avqa = avqa(q, k, v)
        q_bh = q.view(B, T_q, 4, 32).transpose(1, 2)
        k_bh = k.view(B, T_k, 4, 32).transpose(1, 2)
        v_bh = v.view(B, T_k, 4, 32).transpose(1, 2)
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_bh, k_bh, v_bh)
        out_sdpa = out_sdpa.transpose(1, 2).contiguous().view(B, T_q, E)

    print(f"AVQA output shape: {tuple(out_avqa.shape)}")
    print(f"SDPA output shape: {tuple(out_sdpa.shape)}")
    diff = (out_avqa - out_sdpa).abs()
    print(f"max |AVQA - SDPA|: {diff.max().item():.4f}")
    print(f"mean |AVQA - SDPA|: {diff.mean().item():.4f}")


if __name__ == "__main__":
    main()