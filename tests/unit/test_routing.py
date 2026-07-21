"""Tests for avqa.routing module."""

from __future__ import annotations

import pytest
import torch

from avqa.exceptions import RoutingError
from avqa.routing import (
    BudgetRouter,
    Router,
    ThresholdRouter,
    TopPRouter,
    compute_importance,
)


class TestComputeImportance:
    """Tests for compute_importance (spec §7.10)."""

    def test_shape(self) -> None:
        """Output has shape [B, H, M_0]."""
        A = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
        n = torch.rand(2, 4, 16) + 0.1
        w = compute_importance(A, n)
        assert w.shape == (2, 4, 16)

    def test_formula(self) -> None:
        """Importance equals counts * sum_i (A_ij / Z_i)."""
        torch.manual_seed(0)
        B, H, N, M = 1, 1, 8, 4
        A_logits = torch.randn(B, H, N, M)
        A = torch.softmax(A_logits, dim=-1)
        n = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        w = compute_importance(A, n)
        # Manual reference
        Z = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w_ref = n * (A / Z).sum(dim=-2)
        assert torch.allclose(w, w_ref, atol=1e-5)

    def test_zero_counts_zero_importance(self) -> None:
        """Zero-count codewords have zero importance (no key contributes)."""
        A = torch.softmax(torch.randn(1, 1, 4, 3), dim=-1)
        n = torch.tensor([[[0.0, 1.0, 2.0]]])
        w = compute_importance(A, n)
        assert w[0, 0, 0].item() == 0.0

    def test_uniform_attention_recovers_counts(self) -> None:
        """Uniform attention over codewords gives importance proportional to counts."""
        B, H, N, M = 1, 1, 8, 4
        A = torch.ones(B, H, N, M) / M
        n = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        w = compute_importance(A, n)
        # With uniform attention and normalized probs, importance = (N/M) * n_j.
        expected = (N / M) * n
        assert torch.allclose(w, expected, atol=1e-5)


class TestComputeImportanceErrors:
    """Tests for input validation."""

    def test_wrong_rank_attention(self) -> None:
        """attention_probs must be rank 4."""
        A = torch.randn(1, 4, 3)
        n = torch.rand(1, 4, 3)
        with pytest.raises(RoutingError, match="rank 4"):
            compute_importance(A, n)

    def test_wrong_rank_counts(self) -> None:
        """counts must be rank 3."""
        A = torch.softmax(torch.randn(1, 1, 4, 3), dim=-1)
        n = torch.rand(1, 3)
        with pytest.raises(RoutingError, match="rank 3"):
            compute_importance(A, n)

    def test_mismatched_batch(self) -> None:
        """Batch shapes must match."""
        A = torch.softmax(torch.randn(2, 1, 4, 3), dim=-1)
        n = torch.rand(1, 1, 3)
        with pytest.raises(RoutingError, match="incompatible batch"):
            compute_importance(A, n)


class TestTopPRouter:
    """Tests for TopPRouter (spec §9.6)."""

    def test_basic_selection(self) -> None:
        """Top-P picks the P highest-scoring parents."""
        importance = torch.tensor([[[0.4, 0.9, 0.1, 0.7]]])
        decision = TopPRouter().select(importance, budget=2)
        assert decision.selected_indices.tolist() == [[[1, 3]]]

    def test_deterministic_tie_break(self) -> None:
        """Ties resolve to lower indices (spec gap G14)."""
        importance = torch.tensor([[[0.5, 0.5, 0.5]]])
        decision = TopPRouter().select(importance, budget=2)
        # All tied at 0.5; deterministic sort picks lowest index first.
        assert decision.selected_indices.tolist() == [[[0, 1]]]

    def test_budget_equals_codewords(self) -> None:
        """budget == num_codewords returns all (in score-descending order)."""
        importance = torch.tensor([[[0.1, 0.2, 0.3]]])
        decision = TopPRouter().select(importance, budget=3)
        # Stable sort descending: highest first, then by original index.
        assert decision.selected_indices.tolist() == [[[2, 1, 0]]]

    def test_invalid_budget_zero(self) -> None:
        """budget=0 raises."""
        with pytest.raises(RoutingError, match="budget"):
            TopPRouter().select(torch.zeros(1, 1, 4), budget=0)

    def test_invalid_budget_too_large(self) -> None:
        """budget > num_codewords raises."""
        with pytest.raises(RoutingError, match="exceeds"):
            TopPRouter().select(torch.zeros(1, 1, 4), budget=10)

    def test_preserves_importance(self) -> None:
        """Decision.importance echoes the input."""
        importance = torch.tensor([[[0.4, 0.9, 0.1, 0.7]]])
        decision = TopPRouter().select(importance, budget=2)
        assert torch.equal(decision.importance, importance)


class TestThresholdRouter:
    """Tests for ThresholdRouter."""

    def test_threshold_filters(self) -> None:
        """Only codewords above threshold are selected."""
        importance = torch.tensor([[[0.4, 0.9, 0.6, 0.3]]])
        # budget=2 means we take at most 2 entries; only indices 1 and 2 qualify.
        decision = ThresholdRouter(threshold=0.5).select(importance, budget=2)
        assert decision.selected_indices.tolist() == [[[1, 2]]]

    def test_threshold_below_all(self) -> None:
        """Low threshold selects all."""
        importance = torch.tensor([[[0.4, 0.5, 0.6, 0.7]]])
        decision = ThresholdRouter(threshold=0.0).select(importance, budget=4)
        assert decision.selected_indices.tolist() == [[[3, 2, 1, 0]]]

    def test_negative_threshold_rejected(self) -> None:
        """Negative thresholds are rejected at construction."""
        with pytest.raises(RoutingError):
            ThresholdRouter(threshold=-0.1)

    def test_invalid_budget(self) -> None:
        """budget=0 raises."""
        with pytest.raises(RoutingError):
            ThresholdRouter(threshold=0.0).select(torch.zeros(1, 1, 4), budget=0)


class TestRoutingDecision:
    """Tests for the RoutingDecision dataclass."""

    def test_num_selected(self) -> None:
        """num_selected reflects the budget."""
        importance = torch.zeros(1, 2, 8)
        decision = TopPRouter().select(importance, budget=4)
        assert decision.num_selected == 4


class TestThresholdRouterValidation:
    """ThresholdRouter refuses when fewer than budget entries meet threshold."""

    def test_below_threshold_raises(self) -> None:
        """When all entries are below threshold, raise RoutingError."""
        importance = torch.zeros(1, 1, 8)
        router = ThresholdRouter(threshold=1.0)
        with pytest.raises(RoutingError, match="threshold"):
            router.select(importance, budget=2)

    def test_partial_under_threshold_raises(self) -> None:
        """When some (b, h) positions have fewer hits than budget, raise."""
        importance = torch.tensor(
            [[[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )  # [B=1, H=1, M_0=8]
        router = ThresholdRouter(threshold=0.5)
        with pytest.raises(RoutingError, match="threshold"):
            router.select(importance, budget=3)


class TestBudgetRouter:
    """BudgetRouter returns exactly ``budget`` indices per (b, h)."""

    def test_returns_exact_budget(self) -> None:
        importance = torch.tensor([[[0.1, 0.9, 0.3, 0.5]]])
        decision = BudgetRouter().select(importance, budget=2)
        assert decision.num_selected == 2
        assert decision.selected_indices.shape == (1, 1, 2)
        # Highest two by score: 1, 3.
        assert decision.selected_indices[0, 0].tolist() == [1, 3]

    def test_ties_break_by_lower_index(self) -> None:
        importance = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        decision = BudgetRouter().select(importance, budget=2)
        # Tie-break: lowest indices win.
        assert decision.selected_indices[0, 0].tolist() == [0, 1]

    def test_zero_budget_raises(self) -> None:
        with pytest.raises(RoutingError, match="budget"):
            BudgetRouter().select(torch.zeros(1, 1, 4), budget=0)

    def test_over_budget_raises(self) -> None:
        with pytest.raises(RoutingError, match="budget"):
            BudgetRouter().select(torch.zeros(1, 1, 4), budget=8)


class TestAbstractInterface:
    """Tests for the Router abstract base."""

    def test_cannot_instantiate(self) -> None:
        """Router cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Router.__new__(Router)

    def test_subclass_relationship(self) -> None:
        """All concrete routers inherit from Router."""
        assert issubclass(TopPRouter, Router)
        assert issubclass(ThresholdRouter, Router)

    def test_create_factory(self) -> None:
        """``Router.create`` maps strategy names to concrete classes."""
        assert isinstance(Router.create("topp"), TopPRouter)
        assert isinstance(Router.create("threshold"), ThresholdRouter)
        assert isinstance(Router.create("budget"), BudgetRouter)
        with pytest.raises(RoutingError, match="unknown"):
            Router.create("nothing")
