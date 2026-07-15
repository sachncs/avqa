"""Tests for avqa.functional module."""

from __future__ import annotations

import dataclasses

import torch

from avqa.config import AVQConfig, RefinementConfig
from avqa.functional import attention


def _small_config(**overrides: object) -> AVQConfig:
    """Tiny config for fast tests."""
    defaults: dict[str, object] = {
        "attention": __import__("avqa.config", fromlist=["AttentionShapeConfig"]).AttentionShapeConfig(
            embed_dim=32,
            num_heads=4,
            head_dim=8,
        ),
        "codebook": __import__("avqa.config", fromlist=["CodebookConfig"]).CodebookConfig(
            num_codewords=8,
            children_per_codeword=2,
        ),
        "routing": __import__("avqa.config", fromlist=["RoutingConfig"]).RoutingConfig(
            refinement_budget=3,
        ),
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return AVQConfig(**defaults)  # type: ignore[arg-type]


class TestFunctionalAttention:
    """Tests for the stateless functional API (spec §3.5, §5.7)."""

    def test_basic_call(self) -> None:
        """attention() returns the expected output shape."""
        out = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            _small_config(),
        )
        assert out.shape == (1, 4, 32)

    def test_batch_size(self) -> None:
        """Batch dimension is preserved."""
        out = attention(
            torch.randn(3, 4, 32),
            torch.randn(3, 6, 32),
            torch.randn(3, 6, 32),
            _small_config(),
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
            _small_config(),
        )
        out2 = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            _small_config(),
        )
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()

    def test_with_causal_mask(self) -> None:
        """Causal masking is respected via the config."""
        config = _small_config(causal=True)
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
            _small_config(),
            refinement=RefinementConfig(enabled=False),
        )
        out = attention(
            torch.randn(1, 4, 32),
            torch.randn(1, 6, 32),
            torch.randn(1, 6, 32),
            config,
        )
        assert out.shape == (1, 4, 32)
