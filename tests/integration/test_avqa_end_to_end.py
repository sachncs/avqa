"""End-to-end AVQA integration test (TASK-12.005).

Verifies a full forward pass on a tiny BERT-shaped attention layer
after replacement with :class:`AVQAttention`. Confirms:

- The HF wrapper returns finite tensors of the correct shape.
- AVQAttention is deterministic across two consecutive forward
  passes with identical inputs.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)
from avqa.integrations import make_hf_attention_replacement

pytestmark = pytest.mark.integration


def has_hf_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def skip_without_hf() -> None:
    if not has_hf_transformers():
        pytest.skip("transformers not installed; install avqa[huggingface] to run")


def test_hf_replacement_full_forward() -> None:
    """Replace attention and run forward; output is finite and shaped."""
    skip_without_hf()
    embed_dim = 32
    num_heads = 2
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
        ),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )

    class Orig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(embed_dim, embed_dim)
            self.k = torch.nn.Linear(embed_dim, embed_dim)
            self.v = torch.nn.Linear(embed_dim, embed_dim)
            self.out = torch.nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            intermediate: torch.Tensor = torch.nn.functional.relu(self.q(x))
            result: torch.Tensor = self.out(intermediate)
            return result

    orig = Orig()
    wrapped = make_hf_attention_replacement(
        embed_dim=embed_dim,
        num_heads=num_heads,
        config=config,
        original_module=orig,
    )
    x = torch.randn(2, 6, embed_dim)
    out = wrapped(x)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.isfinite(out).all()
    assert out.shape == x.shape


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
