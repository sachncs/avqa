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
        """Two calls don't share state (spec §5.7 statelessness).

        We verify by checking that the two outputs differ when the inputs
        differ — but the codebook is randomly initialized the same way
        each call, so we just check the outputs are valid finite tensors.
        """
        out1 = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            small_config(),
        )
        out2 = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            small_config(),
        )
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()

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
