"""Tests for avqa.refinement module."""

from __future__ import annotations

import pytest
import torch

from avqa.attention import OnlineSoftmaxState
from avqa.exceptions import RoutingError
from avqa.refinement import RefinementResult, refine
from avqa.routing import RoutingDecision, TopPRouter, compute_importance


def setup(
    B: int = 1,
    H: int = 1,
    T: int = 4,
    M0: int = 8,
    C: int = 4,
    Dv: int = 16,
    seed: int = 0,
) -> tuple[
    OnlineSoftmaxState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the inputs needed by refine()."""
    torch.manual_seed(seed)
    state = OnlineSoftmaxState.empty(B, H, T, M0, Dv)
    parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
    parent_value = torch.randn(B, H, T, M0, Dv)
    parent_aggregates = torch.randn(B, H, M0, Dv)
    child_aggregates = torch.randn(B, H, M0, C, Dv)
    # Real child logits: Q · C_c^T / sqrt(D).
    child_logits = torch.randn(B, H, T, M0, C)
    attention_probs = parent_probs.clone()
    parent_counts = torch.rand(B, H, M0) + 0.1
    return (
        state,
        parent_probs,
        parent_value,
        parent_aggregates,
        child_aggregates,
        attention_probs,
        parent_counts,
        child_logits,
    )


def make_decision(
    attention_probs: torch.Tensor, parent_counts: torch.Tensor, budget: int
) -> RoutingDecision:
    """Build a RoutingDecision for testing."""
    importance = compute_importance(attention_probs, parent_counts)
    return TopPRouter().select(importance, budget)


class TestRefine:
    """Tests for the refine() orchestrator (spec §9.7)."""

    def test_returns_refinement_result(self) -> None:
        """Result is a RefinementResult with the right types."""
        state, pp, pv, pa, ca, ap, pc, cl = setup()
        decision = make_decision(ap, pc, budget=4)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert isinstance(result, RefinementResult)
        assert isinstance(result.state, OnlineSoftmaxState)

    def test_selected_parents_count(self) -> None:
        """Number of selected parents equals budget."""
        state, pp, pv, pa, ca, ap, pc, cl = setup(M0=8)
        decision = make_decision(ap, pc, budget=3)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert result.selected_parents.shape == (1, 1, 3)

    def test_selected_in_range(self) -> None:
        """Selected parent indices are valid (in [0, M_0))."""
        state, pp, pv, pa, ca, ap, pc, cl = setup(M0=8)
        decision = make_decision(ap, pc, budget=5)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert result.selected_parents.min().item() >= 0
        assert result.selected_parents.max().item() < 8

    def test_merge_value_shape(self) -> None:
        """merge_value has shape [B, H, T, D_v]."""
        state, pp, pv, pa, ca, ap, pc, cl = setup(Dv=16)
        decision = make_decision(ap, pc, budget=4)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert result.merge_value.shape == (1, 1, 4, 16)

    def test_state_finite(self) -> None:
        """Running state remains finite after refinement."""
        state, pp, pv, pa, ca, ap, pc, cl = setup()
        decision = make_decision(ap, pc, budget=4)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert torch.isfinite(result.state.running_max).all()
        assert torch.isfinite(result.state.running_denominator).all()
        assert torch.isfinite(result.state.running_numerator).all()

    def test_invalid_budget_zero(self) -> None:
        """budget=0 raises RoutingError."""
        state, pp, pv, pa, ca, ap, pc, cl = setup()
        decision = RoutingDecision(
            selected_indices=torch.zeros(1, 1, 0, dtype=torch.long),
            importance=torch.ones(1, 1, 8),
        )
        with pytest.raises(RoutingError, match="budget"):
            refine(
                state,
                pp,
                pv,
                pa,
                ca,
                children_per_parent=4,
                decision=decision,
                attention_probs=ap,
                parent_counts=pc,
                child_logits=cl,
            )

    def test_invalid_budget_too_large(self) -> None:
        """budget > num_parents raises RoutingError."""
        state, pp, pv, pa, ca, ap, pc, cl = setup(M0=4)
        decision = RoutingDecision(
            selected_indices=torch.zeros(1, 1, 10, dtype=torch.long),
            importance=torch.ones(1, 1, 4),
        )
        with pytest.raises(RoutingError, match="exceeds"):
            refine(
                state,
                pp,
                pv,
                pa,
                ca,
                children_per_parent=4,
                decision=decision,
                attention_probs=ap,
                parent_counts=pc,
                child_logits=cl,
            )

    def test_budget_equals_parents(self) -> None:
        """budget == num_parents refines everything."""
        state, pp, pv, pa, ca, ap, pc, cl = setup(M0=6)
        decision = make_decision(ap, pc, budget=6)
        result = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        assert result.selected_parents.shape == (1, 1, 6)
        assert sorted(result.selected_parents.flatten().tolist()) == list(range(6))

    def test_correction_changes_state(self) -> None:
        """Refinement with real child logits changes the running state."""
        state, pp, pv, pa, ca, ap, pc, cl = setup()
        decision = make_decision(ap, pc, budget=4)
        # Run with child_logits=None (approximation).
        result_approx = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
        )
        # Run with real child_logits.
        result_real = refine(
            state,
            pp,
            pv,
            pa,
            ca,
            children_per_parent=4,
            decision=decision,
            attention_probs=ap,
            parent_counts=pc,
            child_logits=cl,
        )
        # The two should differ (real logits ≠ approximation).
        assert not torch.allclose(
            result_approx.state.running_numerator,
            result_real.state.running_numerator,
            atol=1e-6,
        )
