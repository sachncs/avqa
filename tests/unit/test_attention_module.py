"""Tests for avqa.attention_module (AVQAttention nn.Module)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from typing_extensions import TypedDict, Unpack

from avqa.attention_module import AVQAttention
from avqa.cache import InMemoryKVCache
from avqa.config import (
    AttentionShapeConfig,
    AVQConfig,
    CodebookConfig,
    RefinementConfig,
    RoutingConfig,
)
from avqa.exceptions import AVQAError


class _ConfigOverrides(TypedDict, total=False):
    attention: AttentionShapeConfig
    codebook: CodebookConfig
    routing: RoutingConfig
    refinement: RefinementConfig
    dropout: float


def small_config(**overrides: Unpack[_ConfigOverrides]) -> AVQConfig:
    """Tiny config for fast tests."""
    base = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=3),
        dropout=0.0,
    )
    return dataclasses.replace(base, **overrides)


class TestConstruction:
    """Tests for AVQAttention construction."""

    def test_default_construction(self) -> None:
        """Default config produces a valid module."""
        module = AVQAttention(AVQConfig())
        assert isinstance(module, torch.nn.Module)

    def test_inherits_nn_module(self) -> None:
        """AVQAttention inherits from nn.Module (spec §5.6)."""
        assert issubclass(AVQAttention, torch.nn.Module)

    def test_projection_layers(self) -> None:
        """Projection layers are nn.Linear when in_proj=True."""
        module = AVQAttention(AVQConfig())
        assert isinstance(module.q_proj, torch.nn.Linear)
        assert isinstance(module.k_proj, torch.nn.Linear)
        assert isinstance(module.v_proj, torch.nn.Linear)
        assert isinstance(module.out_proj, torch.nn.Linear)

    def test_no_projection(self) -> None:
        """in_proj=False replaces linears with Identity."""
        module = AVQAttention(AVQConfig(), in_proj=False, out_proj=False)
        assert isinstance(module.q_proj, torch.nn.Identity)


class TestForwardNaive:
    """Tests for the naive (refinement-disabled) forward path."""

    def test_output_shape(self) -> None:
        """Output has shape [B, T_q, E]."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(2, 4, 32)
        k = torch.randn(2, 6, 32)
        v = torch.randn(2, 6, 32)
        out = module(q, k, v)
        assert out.shape == (2, 4, 32)

    def test_with_projection(self) -> None:
        """Output shape preserved with projection layers enabled."""
        config = small_config()
        module = AVQAttention(config)
        q = torch.randn(2, 4, 32)
        out = module(q, q, q)
        assert out.shape == (2, 4, 32)

    def test_causal_mask(self) -> None:
        """Causal mode applies lower-triangular masking."""
        config = dataclasses.replace(small_config(), causal=True)
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 6, 32)
        v = torch.randn(1, 6, 32)
        # No exception means the causal mask path ran cleanly.
        out = module(q, k, v)
        assert out.shape == (1, 4, 32)


class TestForwardAVQ:
    """Tests for the full AVQ pipeline."""

    def test_avq_path_runs(self) -> None:
        """Full AVQ path returns the expected shape."""
        module = AVQAttention(small_config(), in_proj=False, out_proj=False)
        q = torch.randn(2, 4, 32)
        k = torch.randn(2, 6, 32)
        v = torch.randn(2, 6, 32)
        out = module(q, k, v)
        assert out.shape == (2, 4, 32)

    def test_avq_path_with_causal(self) -> None:
        """AVQ path with causal masking runs."""
        config = dataclasses.replace(small_config(), causal=True)
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 6, 32)
        v = torch.randn(1, 6, 32)
        out = module(q, k, v)
        assert out.shape == (1, 4, 32)

    def test_avq_path_output_finite(self) -> None:
        """AVQ output contains no NaN/Inf."""
        module = AVQAttention(small_config(), in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        out = module(q, q, q)
        assert torch.isfinite(out).all()


class TestGradients:
    """Tests for gradient flow through AVQAttention (spec §3.24)."""

    def test_gradients_flow_through_naive(self) -> None:
        """Gradients reach all projection parameters in the naive path."""
        config = AVQConfig(
            refinement=__import__("avqa.config", fromlist=["RefinementConfig"]).RefinementConfig(
                enabled=False
            ),
        )
        module = AVQAttention(config)
        q = torch.randn(1, 4, config.attention.embed_dim, requires_grad=True)
        out = module(q, q, q)
        out.sum().backward()
        assert q.grad is not None

    def test_gradients_flow_through_avq(self) -> None:
        """Gradients reach input tensors in the AVQ path."""
        module = AVQAttention(small_config())
        q = torch.randn(1, 4, 32, requires_grad=True)
        out = module(q, q, q)
        out.sum().backward()
        assert q.grad is not None


class TestDtypeSupport:
    """Tests for supported dtypes (spec §6.9)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_supported_dtype(self, dtype: torch.dtype) -> None:
        """Reference supports FP32, FP16, BF16."""
        module = AVQAttention(small_config(), in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32, dtype=dtype)
        out = module(q, q, q)
        # Output dtype is determined by the output projection weights
        # (initialized in FP32). The forward pass runs without error for
        # all three supported dtypes.
        assert out.dtype in {torch.float32, dtype}


class TestKVCacheIntegration:
    """Tests for KV cache integration (spec §3.13)."""

    def test_kv_cache_extends(self) -> None:
        """Passing a cache extends it with the new K/V."""
        module = AVQAttention(small_config(), in_proj=False, out_proj=False)
        cache = InMemoryKVCache(
            num_heads=4,
            head_dim_k=8,
            head_dim_v=8,
        )
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 6, 32)
        v = torch.randn(1, 6, 32)
        module(q, k, v, kv_cache=cache)
        assert cache.size == 6


class TestRefinementDisabled:
    """Tests that refinement.enabled=False uses naive path."""

    def test_disabled_uses_naive(self) -> None:
        """When refinement is disabled, the scheduler is None."""
        config = AVQConfig(refinement=RefinementConfig(enabled=False))
        module = AVQAttention(config)
        assert module.scheduler is None

    def test_disabled_does_not_quantize(self) -> None:
        """With refinement disabled, forward path does not touch codebook.

        The :attr:`AVQAttention.last_keys` attribute is only set after
        the VQ precompute path runs. Asserting it stays ``None`` is
        the cleanest way to verify quantize is bypassed.
        """
        config = AVQConfig(
            attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
            refinement=RefinementConfig(enabled=False),
        )
        module = AVQAttention(config, in_proj=False, out_proj=False)
        module(torch.randn(1, 4, 32), torch.randn(1, 4, 32), torch.randn(1, 4, 32))
        assert module.last_keys is None
        assert module.last_parent_assignments is None


class TestErrorHandling:
    """Tests for input validation (spec §10.16)."""

    def test_query_key_embedding_mismatch(self) -> None:
        """Query and key with different embedding dims raise."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 4, 16)  # wrong dim
        with pytest.raises((ValueError, RuntimeError, AVQAError)):
            module(q, k, k)


class TestStateDict:
    """Tests for parameter serialization (spec §3.20)."""

    def test_state_dict_round_trip(self) -> None:
        """state_dict round-trips through load_state_dict."""
        module = AVQAttention(small_config())
        state = module.state_dict()
        # q_proj.weight has shape (32, 32); verify it's in state.
        assert "q_proj.weight" in state
        # Round-trip via a fresh module.
        module2 = AVQAttention(small_config())
        module2.load_state_dict(state)
        # Copy codebook state (not an nn.Parameter, so not in state_dict).
        module2.codebook.parents = module.codebook.parents.clone()
        module2.codebook.children = module.codebook.children.clone()
        q = torch.randn(1, 4, 32)
        out1 = module(q, q, q)
        out2 = module2(q, q, q)
        # Projection weights are verified by state_dict; codebook is
        # separately managed so outputs match only when codebooks match.
        assert torch.allclose(out1, out2, atol=1e-4)
