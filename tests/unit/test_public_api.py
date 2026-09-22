"""Tests for the stable top-level import contract."""

from __future__ import annotations

import torch

from avqa import AVQConfig, attention as public_attention
from avqa.functional import attention as functional_attention


def test_functional_attention_is_exported_and_matches_module_contract() -> None:
    """The documented functional entry point is available from both paths."""
    assert public_attention is functional_attention

    config = AVQConfig()
    query = torch.randn(1, 2, config.attention.embed_dim)
    key = torch.randn(1, 3, config.attention.embed_dim)
    value = torch.randn(1, 3, config.attention.embed_dim)

    output = public_attention(query, key, value, config)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
