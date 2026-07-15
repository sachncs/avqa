"""Tests for avqa.refinement module."""

from __future__ import annotations

import pytest
import torch

from avqa.attention import OnlineSoftmaxState
from avqa.refinement import RefinementResult, refine


def _setup(
    B: int = 1,
    H: int = 1,
    T: int = 4,
    M0: int = 8,
    C: int = 4,
    Dv: int = 16,
    seed: int = 0,
):
    """Build the inputs needed by refine()."""
    torch.manual_seed(seed)
    state = OnlineSoftmaxState.empty(B, H, T, M0, Dv)
    parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
    parent_value = torch.randn(B, H, T, M0, Dv)
    parent_aggregates = torch.randn(B, H, M0, Dv)
    child_aggregates = torch.randn(B, H, M0, C, Dv)
    attention_probs = parent_probs.clone()
    parent_counts = torch.rand(B, H, M0) + 0.1
    return state, parent_probs, parent_value, parent_aggregates, child_aggregates, attention_probs, parent_counts


class TestRefine:
    """Tests for the refine() orchestrator (spec §9.7)."""

    def test_returns_refinement_result(self) -> None:
        """Result is a RefinementResult with the right types."""
        state, pp, pv, pa, ca, ap, pc = _setup()
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=4, attention_probs=ap, parent_counts=pc,
        )
        assert isinstance(result, RefinementResult)
        assert isinstance(result.state, OnlineSoftmaxState)

    def test_selected_parents_count(self) -> None:
        """Number of selected parents equals budget."""
        state, pp, pv, pa, ca, ap, pc = _setup(M0=8)
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=3, attention_probs=ap, parent_counts=pc,
        )
        assert result.selected_parents.shape == (1, 1, 3)

    def test_selected_in_range(self) -> None:
        """Selected parent indices are valid (in [0, M_0))."""
        state, pp, pv, pa, ca, ap, pc = _setup(M0=8)
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=5, attention_probs=ap, parent_counts=pc,
        )
        assert result.selected_parents.min().item() >= 0
        assert result.selected_parents.max().item() < 8

    def test_merge_value_shape(self) -> None:
        """merge_value has shape [B, H, T, D_v]."""
        state, pp, pv, pa, ca, ap, pc = _setup(Dv=16)
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=4, attention_probs=ap, parent_counts=pc,
        )
        assert result.merge_value.shape == (1, 1, 4, 16)

    def test_state_finite(self) -> None:
        """Running state remains finite after refinement."""
        state, pp, pv, pa, ca, ap, pc = _setup()
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=4, attention_probs=ap, parent_counts=pc,
        )
        assert torch.isfinite(result.state.running_max).all()
        assert torch.isfinite(result.state.running_denominator).all()
        assert torch.isfinite(result.state.running_numerator).all()

    def test_invalid_budget_zero(self) -> None:
        """budget=0 raises."""
        state, pp, pv, pa, ca, ap, pc = _setup()
        with pytest.raises(ValueError, match="budget"):
            refine(
                state, pp, pv, pa, ca,
                children_per_parent=4, budget=0, attention_probs=ap, parent_counts=pc,
            )

    def test_invalid_budget_too_large(self) -> None:
        """budget > num_parents raises."""
        state, pp, pv, pa, ca, ap, pc = _setup(M0=4)
        with pytest.raises(ValueError, match="exceeds"):
            refine(
                state, pp, pv, pa, ca,
                children_per_parent=4, budget=10, attention_probs=ap, parent_counts=pc,
            )

    def test_budget_equals_parents(self) -> None:
        """budget == num_parents refines everything."""
        state, pp, pv, pa, ca, ap, pc = _setup(M0=6)
        result = refine(
            state, pp, pv, pa, ca,
            children_per_parent=4, budget=6, attention_probs=ap, parent_counts=pc,
        )
        assert result.selected_parents.shape == (1, 1, 6)
        assert sorted(result.selected_parents.flatten().tolist()) == list(range(6))
