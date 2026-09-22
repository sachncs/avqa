"""Smoke-test an installed AVQA environment and its public attention API."""

from __future__ import annotations

import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig


def main() -> int:
    """Validate importability and execute a minimal public forward pass."""
    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=16, num_heads=2, head_dim=8),
        codebook=CodebookConfig(num_codewords=4, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )
    module = AVQAttention(config, in_proj=False, out_proj=False)
    query = torch.randn(1, 2, 16)
    output = module(query, query, query)
    if output.shape != query.shape:
        raise SystemExit(f"environment smoke failed: output shape {output.shape}")
    if not torch.isfinite(output).all():
        raise SystemExit("environment smoke failed: output contains non-finite values")
    print("AVQA environment smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
