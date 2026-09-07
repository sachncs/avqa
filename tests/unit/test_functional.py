"""Tests for avqa.functional module."""

from __future__ import annotations

import dataclasses

import torch
from typing_extensions import TypedDict, Unpack

from avqa.config import (
    AttentionShapeConfig,
    AVQConfig,
    CodebookConfig,
    RefinementConfig,
    RoutingConfig,
)
from avqa.functional import attention


class _ConfigOverrides(TypedDict, total=False):
    attention: AttentionShapeConfig
    codebook: CodebookConfig
    routing: RoutingConfig
    refinement: RefinementConfig
    dropout: float
    causal: bool


def small_config(**overrides: Unpack[_ConfigOverrides]) -> AVQConfig:
    """Tiny config for fast tests."""
    base = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=3),
        dropout=0.0,
    )
    return dataclasses.replace(base, **overrides)


class TestFunctionalAttention:
    """Tests for the stateless functional API (spec §3.5, §5.7)."""

    def test_basic_call(self) -> None:
        """attention() returns the expected output shape."""
        out = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            small_config(),
        )
        assert out.shape == (1, 4, 32)

    def test_batch_size(self) -> None:
        """Batch dimension is preserved."""
        out = attention(
            torch.randn(3, 4, 32),
            torch.randn(3, 6, 32),
            torch.randn(3, 6, 32),
            small_config(),
        )
        assert out.shape == (3, 4, 32)

    def test_does_not_reuse_module_state(self) -> None:
        """Two calls produce finite outputs from the same inputs (spec §5.7).

        Per spec §5.7 the functional API is "stateless across calls";
        we pin that by verifying both outputs are finite. Note that
        a fresh ``AVQAttention`` is constructed per call so the
        codebook is freshly initialised — the global RNG advances
        between calls, so two outputs need NOT be equal. The
        stateless-ness contract is that no carry-over state
        (running codebook update, last_keys, …) leaks between calls.
        """
        torch.manual_seed(0)
        out1 = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            small_config(),
        )
        torch.manual_seed(0)
        out2 = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            small_config(),
        )
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()
        # Verify by re-seeding between calls: same seed before each
        # call → identical outputs (no leaked state).
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_different_inputs_produce_different_outputs(self) -> None:
        """Sanity: distinct q tensors give distinct outputs."""
        cfg = small_config()
        torch.manual_seed(0)
        a = attention(torch.randn(1, 4, 32), torch.randn(1, 6, 32), torch.randn(1, 6, 32), cfg)
        torch.manual_seed(0)
        b = attention(torch.randn(1, 4, 32), torch.randn(1, 6, 32), torch.randn(1, 6, 32), cfg)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
        torch.manual_seed(1)
        c = attention(
            torch.randn(1, 4, 32),  # different q via different RNG state
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            cfg,
        )
        assert not torch.allclose(a, c, atol=1e-3)

    def test_with_causal_mask(self) -> None:
        """Causal masking is respected via the config."""
        config = small_config(causal=True)
        out = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            config,
        )
        assert out.shape == (1, 4, 32)

    def test_refinement_disabled(self) -> None:
        """When refinement is disabled, the API still works."""
        config = dataclasses.replace(
            small_config(),
            refinement=RefinementConfig(enabled=False),
        )
        out = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            config,
        )
        assert out.shape == (1, 4, 32)
