"""End-to-end AVQA integration test (TASK-12.005).

Verifies a full forward pass on a tiny BERT-shaped attention layer
constructed directly via :class:`AVQAttention`. Confirms that the
AVQA module is deterministic across two consecutive forward passes
with identical inputs.

The original HF/vLLM replacement variants were removed when the
optional-dependency integrations package was dropped (see CHANGELOG
"[0.1.0] Removed"). They can be re-introduced by users as their
own adapters; this file covers the still-shipped native path.
"""

from __future__ import annotations

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)

pytestmark = pytest.mark.integration


def test_avqa_native_round_trip_cpu() -> None:
    """A direct AVQAttention call works on CPU and is deterministic."""
    embed_dim = 32
    num_heads = 2
    head_dim = embed_dim // num_heads
    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )
    mod = AVQAttention(config, in_proj=False, out_proj=False)
    mod.eval()

    torch.manual_seed(0)
    q = torch.randn(2, 6, embed_dim)
    k = torch.randn(2, 6, embed_dim)
    v = torch.randn(2, 6, embed_dim)
    out1 = mod(q, k, v)
    out2 = mod(q, k, v)
    assert torch.allclose(out1, out2)
