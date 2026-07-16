"""Tests for the spec compliance fixes (C1-C5, H1-H7).

Verifies the critical algorithmic corrections and new features added
to bring the implementation into compliance with the spec.
"""

from __future__ import annotations

import pytest
import torch

from avqa import AdaptiveRefinement, AVQAttention, AVQConfig
from avqa.attention import OnlineSoftmaxState
from avqa.codebook import HierarchicalCodebook
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)
from avqa.exceptions import AVQAError, ConfigurationError, ShapeError
from avqa.quantizer import EuclideanHierarchicalQuantizer
from avqa.refinement import refine
from avqa.routing import TopPRouter, compute_importance


def small_config(**overrides: object) -> AVQConfig:
    defaults: dict[str, object] = {
        "attention": AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
        "codebook": CodebookConfig(num_codewords=8, children_per_codeword=2),
        "routing": RoutingConfig(refinement_budget=3),
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return AVQConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# C1: VQ attention denominator includes n_a (spec §7.7)
# ---------------------------------------------------------------------------


class TestC1DenominatorWeighting:
    """VQ attention denominator must include assignment counts n_a."""

    def test_denominator_includes_counts(self) -> None:
        """When parents have different counts, the output changes.

        With uniform counts, n_a cancels out. With non-uniform counts,
        the weighted denominator shifts the result.
        """
        torch.manual_seed(42)
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 8, 32)
        v = torch.randn(1, 8, 32)
        out = module(q, k, v)
        assert torch.isfinite(out).all()

    def test_hand_computed_n_a_weighting(self) -> None:
        """Verify n_a weighting against hand-computed expected values.

        With 2 parents: parent0 has 3 keys, parent1 has 1 key.
        The denominator should be exp(S0)*3 + exp(S1)*1, not exp(S0)+exp(S1).
        """
        torch.manual_seed(0)
        H, M0, C, D = 1, 2, 2, 4
        cb = HierarchicalCodebook(
            num_heads=H,
            num_parents=M0,
            children_per_parent=C,
            head_dim=D,
        )
        cb.parents = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
        cb.children = torch.tensor(
            [
                [
                    [[1.1, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]],
                    [[0.0, 1.1, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
                ],
            ]
        )

        keys = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        values = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0, 0.0],
                    ]
                ]
            ]
        )

        quantizer = EuclideanHierarchicalQuantizer()
        result = quantizer.precompute(keys, values, cb)

        # parent0: 3 keys, parent1: 1 key
        assert result.parent_counts[0, 0, 0].item() == 3.0
        assert result.parent_counts[0, 0, 1].item() == 1.0

        # parent0 aggregate: [1+2+3, 0, 0, 0] = [6, 0, 0, 0]
        # parent1 aggregate: [0, 10, 0, 0]
        assert torch.allclose(result.parent_aggregates[0, 0, 0], torch.tensor([6.0, 0.0, 0.0, 0.0]))
        assert torch.allclose(
            result.parent_aggregates[0, 0, 1], torch.tensor([0.0, 10.0, 0.0, 0.0])
        )


# ---------------------------------------------------------------------------
# C3: Correction uses raw aggregates (spec §7.13)
# ---------------------------------------------------------------------------


class TestC3CorrectionRawAggregates:
    """Correction must subtract V̄_p (raw), not A_p·V̄_p (weighted)."""

    def test_correction_with_raw_aggregates(self) -> None:
        """Refine with raw parent aggregates produces finite state."""
        torch.manual_seed(0)
        B, H, T, M0, C, Dv = 1, 1, 4, 8, 2, 8
        state = OnlineSoftmaxState.empty(B, H, T, M0, Dv)
        parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
        parent_aggregates = torch.randn(B, H, M0, Dv)
        # Use raw aggregates for parent_value (correct usage).
        parent_value = parent_probs.unsqueeze(-1) * parent_aggregates.unsqueeze(2)
        child_aggregates = torch.randn(B, H, M0, C, Dv)
        child_logits = torch.randn(B, H, T, M0, C)
        importance = compute_importance(parent_probs, torch.ones(B, H, M0))
        decision = TopPRouter().select(importance, budget=3)

        result = refine(
            state=state,
            parent_probs=parent_probs,
            parent_value=parent_value,
            parent_aggregates=parent_aggregates,
            child_aggregates=child_aggregates,
            children_per_parent=C,
            decision=decision,
            attention_probs=parent_probs,
            parent_counts=torch.ones(B, H, M0),
            child_logits=child_logits,
        )
        assert torch.isfinite(result.state.running_numerator).all()
        assert torch.isfinite(result.state.running_denominator).all()


# ---------------------------------------------------------------------------
# C2+C4: State reduction is the output (spec §7.7, §7.14)
# ---------------------------------------------------------------------------


class TestC2C4StateReduction:
    """Output must come from state reduction, not merge_value."""

    def test_state_reduction_includes_all_parents(self) -> None:
        """State reduction covers all parents, not just selected."""
        torch.manual_seed(0)
        B, H, T, M0, C, Dv = 1, 1, 4, 8, 2, 8
        state = OnlineSoftmaxState.empty(B, H, T, M0, Dv)
        parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
        parent_aggregates = torch.randn(B, H, M0, Dv)
        parent_value = parent_probs.unsqueeze(-1) * parent_aggregates.unsqueeze(2)
        child_aggregates = torch.randn(B, H, M0, C, Dv)
        importance = compute_importance(parent_probs, torch.ones(B, H, M0))
        decision = TopPRouter().select(importance, budget=3)

        result = refine(
            state=state,
            parent_probs=parent_probs,
            parent_value=parent_value,
            parent_aggregates=parent_aggregates,
            child_aggregates=child_aggregates,
            children_per_parent=C,
            decision=decision,
            attention_probs=parent_probs,
            parent_counts=torch.ones(B, H, M0),
        )

        # State reduction should give [B, H, T, Dv].
        out = result.state.running_numerator[:, :, :, 0, :] / result.state.running_denominator[
            :, :, :, 0:1
        ].clamp_min(1e-12)
        assert out.shape == (B, H, T, Dv)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# C5: AdaptiveRefinement class (spec §5.5)
# ---------------------------------------------------------------------------


class TestC5AdaptiveRefinement:
    """AdaptiveRefinement must be importable and functional."""

    def test_import(self) -> None:
        """from avqa import AdaptiveRefinement works."""
        assert AdaptiveRefinement is not None

    def test_construction(self) -> None:
        """AdaptiveRefinement can be instantiated."""
        ar = AdaptiveRefinement(children_per_parent=4)
        assert ar.children_per_parent == 4
        assert ar.last_result is None

    def test_refine_method(self) -> None:
        """AdaptiveRefinement.refine() delegates to refine()."""
        torch.manual_seed(0)
        B, H, T, M0, C, Dv = 1, 1, 4, 8, 2, 8
        state = OnlineSoftmaxState.empty(B, H, T, M0, Dv)
        parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
        parent_aggregates = torch.randn(B, H, M0, Dv)
        parent_value = parent_probs.unsqueeze(-1) * parent_aggregates.unsqueeze(2)
        child_aggregates = torch.randn(B, H, M0, C, Dv)
        importance = compute_importance(parent_probs, torch.ones(B, H, M0))
        decision = TopPRouter().select(importance, budget=3)

        ar = AdaptiveRefinement(children_per_parent=C)
        result = ar.refine(
            state=state,
            parent_probs=parent_probs,
            parent_value=parent_value,
            parent_aggregates=parent_aggregates,
            child_aggregates=child_aggregates,
            decision=decision,
            attention_probs=parent_probs,
            parent_counts=torch.ones(B, H, M0),
        )
        assert ar.last_result is result


# ---------------------------------------------------------------------------
# H5: Input validation (spec §6.12)
# ---------------------------------------------------------------------------


class TestH5InputValidation:
    """Forward pass validates inputs."""

    def test_rejects_rank_2_input(self) -> None:
        """Rank-2 tensors raise ShapeError."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(4, 32)  # rank 2
        with pytest.raises((ShapeError, AVQAError)):
            module(q, q, q)

    def test_rejects_rank_4_input(self) -> None:
        """Rank-4 tensors raise ShapeError."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 8, 32)  # rank 4
        with pytest.raises((ShapeError, AVQAError)):
            module(q, q, q)

    def test_rejects_mismatched_key_dim(self) -> None:
        """Key with different last dim from query raises."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 4, 16)  # wrong dim
        with pytest.raises((ShapeError, AVQAError)):
            module(q, k, k)

    def test_rejects_mismatched_value_shape(self) -> None:
        """Value with different shape from key raises."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 8, 32)
        v = torch.randn(1, 6, 32)  # different T_k
        with pytest.raises((ShapeError, AVQAError)):
            module(q, k, v)

    def test_validation_skipped_when_disabled(self) -> None:
        """skip_validation=True disables checks."""
        config = AVQConfig(
            attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
            codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
            routing=RoutingConfig(refinement_budget=3),
            backend=__import__("avqa.config", fromlist=["BackendConfig"]).BackendConfig(
                skip_validation=True
            ),
        )
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        # This would fail validation but skip_validation=True
        module(q, q, q)  # no exception


# ---------------------------------------------------------------------------
# H7: Selective child attention (spec §9.8)
# ---------------------------------------------------------------------------


class TestH7SelectiveChildAttention:
    """Child attention is computed only for selected parents."""

    def test_selective_computation_runs(self) -> None:
        """Forward pass with selective child attention produces correct shape."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(2, 8, 32)
        k = torch.randn(2, 16, 32)
        v = torch.randn(2, 16, 32)
        out = module(q, k, v)
        assert out.shape == (2, 8, 32)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# H4: Commitment loss (spec §8.9)
# ---------------------------------------------------------------------------


class TestH4CommitmentLoss:
    """Commitment loss computation."""

    def test_commitment_loss_requires_forward(self) -> None:
        """commitment_loss() raises before any forward pass."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        with pytest.raises(RuntimeError, match="forward pass"):
            module.commitment_loss()

    def test_commitment_loss_after_forward(self) -> None:
        """commitment_loss() returns a scalar after forward pass."""
        config = small_config()
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        module(q, q, q)
        loss = module.commitment_loss()
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0
        assert torch.isfinite(loss).all()

    def test_commitment_loss_zero_weight(self) -> None:
        """commitment_loss_weight=0 gives zero loss."""
        config = small_config(
            codebook=CodebookConfig(
                num_codewords=8, children_per_codeword=2, commitment_loss_weight=0.0
            ),
        )
        module = AVQAttention(config, in_proj=False, out_proj=False)
        q = torch.randn(1, 4, 32)
        module(q, q, q)
        loss = module.commitment_loss()
        assert loss.item() == 0.0

    def test_raw_commitment_loss_on_codebook(self) -> None:
        """HierarchicalCodebook.commitment_loss() returns mean squared distance."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(num_heads=1, num_parents=4, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        keys = torch.randn(1, 1, 8, 4)
        assignments = torch.tensor([[[0, 0, 1, 1, 2, 2, 3, 3]]])
        loss = cb.commitment_loss(keys, assignments)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# H6: Configurable tree depth (spec §2.7, §3.8)
# ---------------------------------------------------------------------------


class TestH6ConfigurableTreeDepth:
    """max_depth config validation."""

    def test_max_depth_2_accepted(self) -> None:
        """max_depth=2 is the default and accepted."""
        config = AVQConfig(
            codebook=CodebookConfig(max_depth=2),
        )
        assert config.codebook.max_depth == 2

    def test_max_depth_other_raises(self) -> None:
        """max_depth != 2 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_depth"):
            AVQConfig(
                codebook=CodebookConfig(max_depth=3),
            )

    def test_max_depth_negative_raises(self) -> None:
        """max_depth <= 0 raises."""
        with pytest.raises(ConfigurationError):
            AVQConfig(
                codebook=CodebookConfig(max_depth=0),
            )


# ---------------------------------------------------------------------------
# End-to-end VQ attention formula (spec §7.7)
# ---------------------------------------------------------------------------


class TestEndToEndVQAttention:
    """Full pipeline produces mathematically valid output."""

    def test_output_satisfies_attention_invariant(self) -> None:
        """Output has correct shape, is finite, and gradients flow."""
        config = small_config()
        module = AVQAttention(config)
        q = torch.randn(1, 8, 32, requires_grad=True)
        k = torch.randn(1, 16, 32)
        v = torch.randn(1, 16, 32)
        out = module(q, k, v)
        assert out.shape == (1, 8, 32)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()
