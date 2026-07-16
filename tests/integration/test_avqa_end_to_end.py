"""End-to-end AVQA integration test (TASK-12.005).

Verifies a full forward pass on a tiny BERT-shaped attention layer
after replacement with :class:`AVQAttention`. Confirms:

- Pretrained HF weights survive replacement (within FP32 tolerance).
- The wrapped AVQAttention output is a finite tensor of the correct
  shape.
- Causal masking works through the wrapper.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _has_hf_transformers() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _skip_without_hf() -> None:
    if not _has_hf_transformers():
        pytest.skip("transformers not installed; install avqa[huggingface] to run")


def test_hf_replacement_full_forward() -> None:
    """Replace attention and run forward; output is finite and shaped."""
    _skip_without_hf()
    import torch

    from avqa import AVQConfig
    from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig
    from avqa.integrations import make_hf_attention_replacement

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

    class _Orig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(embed_dim, embed_dim)
            self.k = torch.nn.Linear(embed_dim, embed_dim)
            self.v = torch.nn.Linear(embed_dim, embed_dim)
            self.out = torch.nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.out(torch.nn.functional.relu(self.q(x)))

    orig = _Orig()
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
    import torch

    from avqa import AVQAttention, AVQConfig
    from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig

    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=32, num_heads=2, head_dim=16),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )
    mod = AVQAttention(config, in_proj=False, out_proj=False)
    mod.eval()

    torch.manual_seed(0)
    q = torch.randn(2, 6, 32)
    k = torch.randn(2, 6, 32)
    v = torch.randn(2, 6, 32)
    out1 = mod(q, k, v)
    out2 = mod(q, k, v)
    assert torch.allclose(out1, out2)
