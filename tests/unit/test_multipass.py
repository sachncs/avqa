"""Tests for the MultiPassRefiner (SPEC \u00a715, ACMPR / OPT-0004)."""

from __future__ import annotations

import pytest
import torch

from avqa.attention import OnlineSoftmaxState
from avqa.multipass import MultiPassRefiner, compute_pass_budgets
from avqa.routing import RoutingDecision


class TestComputePassBudgets:
    """``compute_pass_budgets`` matches the geometric formula in SPEC \u00a715.2."""

    def test_decay_05_geometric(self) -> None:
        assert compute_pass_budgets(8, 4, 0.5) == [8, 4, 2, 1]

    def test_decay_1_constant(self) -> None:
        """\u03c1 = 1 keeps the budget constant across passes (paper special case)."""
        assert compute_pass_budgets(8, 4, 1.0) == [8, 8, 8, 8]

    def test_decay_03_shrinks_fast(self) -> None:
        assert compute_pass_budgets(8, 5, 0.3) == [8, 2, 1, 1, 1]

    def test_floor_at_1(self) -> None:
        """``max(1, ...)`` keeps the budget at 1 for very small decays."""
        assert compute_pass_budgets(1, 4, 0.1) == [1, 1, 1, 1]

    def test_rejects_invalid_decay(self) -> None:
        with pytest.raises(ValueError, match="decay must be in"):
            compute_pass_budgets(8, 4, 0.0)
        with pytest.raises(ValueError, match="decay must be in"):
            compute_pass_budgets(8, 4, 1.5)

    def test_rejects_invalid_passes(self) -> None:
        with pytest.raises(ValueError, match="passes must be positive"):
            compute_pass_budgets(8, 0, 0.5)

    def test_rejects_invalid_base(self) -> None:
        with pytest.raises(ValueError, match="base must be positive"):
            compute_pass_budgets(0, 4, 0.5)


class TestMultiPassRefiner:
    """``MultiPassRefiner`` wrapper around ``refine``."""

    def test_passes_1_emits_residual_zero(self) -> None:
        """``passes=1`` is the paper-equivalent path; residual is [0.0]."""
        torch.manual_seed(0)
        m = MultiPassRefiner(passes=1, decay=1.0)
        final_state, residuals = _run_refiner(
            m, B=1, H=1, T=4, P=4, M0=8, C=2, D=8
        )
        assert len(residuals) == 1
        assert residuals[0] == 0.0
        assert final_state.running_numerator.shape == (1, 1, 4, 1, 8)

    def test_passes_4_returns_four_residuals(self) -> None:
        """``passes=4`` returns a residual norm per pass (k-1 entries)."""
        torch.manual_seed(0)
        m = MultiPassRefiner(passes=4, decay=0.5)
        _, residuals = _run_refiner(m, B=1, H=1, T=4, P=4, M0=8, C=2, D=8)
        assert len(residuals) == 4
        # Residual norms are non-negative scalars.
        for r in residuals:
            assert r >= 0.0

    def test_passes_4_decay_halves_budget(self) -> None:
        """At decay 0.5 the per-pass budget sequence is ``[P, P/2, P/4, P/8]``."""
        m = MultiPassRefiner(passes=4, decay=0.5)
        assert m.pass_budgets(8) == [8, 4, 2, 1]

    def test_passes_4_decay_1_holds_budget(self) -> None:
        """At decay 1.0 the per-pass budget is held constant (paper special case)."""
        m = MultiPassRefiner(passes=4, decay=1.0)
        assert m.pass_budgets(8) == [8, 8, 8, 8]

    def test_rejects_invalid_passes(self) -> None:
        with pytest.raises(ValueError, match="passes must be positive"):
            MultiPassRefiner(passes=0, decay=0.5)

    def test_rejects_invalid_decay(self) -> None:
        with pytest.raises(ValueError, match="decay must be in"):
            MultiPassRefiner(passes=2, decay=0.0)


def _make_dummy_inputs(B: int, H: int, T: int, M0: int, C: int, D: int) -> tuple:
    """Build the argument bundle ``refine`` consumes with deterministic data."""
    state = OnlineSoftmaxState.empty(B, H, T, 1, D)
    parent_probs = torch.softmax(torch.randn(B, H, T, M0), dim=-1)
    parent_value = parent_probs.unsqueeze(-1) * torch.randn(B, H, T, M0, D)
    parent_aggregates = torch.randn(B, H, M0, D)
    child_aggregates = torch.randn(B, H, M0, C, D)
    attention_probs = parent_probs
    parent_counts = torch.full((B, H, M0), float(C), dtype=torch.float32)
    child_counts = torch.full((B, H, M0, C), 1.0, dtype=torch.float32)
    decision = RoutingDecision(
        selected_indices=torch.zeros(B, H, C, dtype=torch.long),
        importance=torch.randn(B, H, M0),
    )
    child_logits = torch.randn(B, H, T, C, 2, dtype=torch.float32)
    return (
        state,
        parent_probs,
        parent_value,
        parent_aggregates,
        child_aggregates,
        attention_probs,
        parent_counts,
        child_counts,
        decision,
        child_logits,
    )


def _run_refiner(
    m: MultiPassRefiner, B: int, H: int, T: int, P: int, M0: int, C: int, D: int
) -> tuple[OnlineSoftmaxState, list[float]]:
    """Drive ``m`` through ``P`` passes on a synthetic state."""
    (
        state,
        parent_probs,
        parent_value,
        parent_aggregates,
        child_aggregates,
        attention_probs,
        parent_counts,
        child_counts,
        decision,
        child_logits,
    ) = _make_dummy_inputs(B, H, T, M0, C, D)
    return m.refine(
        state=state,
        parent_probs=parent_probs,
        parent_value=parent_value,
        parent_aggregates=parent_aggregates,
        child_aggregates=child_aggregates,
        children_per_parent=C,
        decision=decision,
        attention_probs=attention_probs,
        parent_counts=parent_counts,
        child_logits=child_logits,
        child_counts=child_counts,
    )
