"""Tests for avqa.attention module (online softmax + correction)."""

from __future__ import annotations

import pytest
import torch

from avqa.attention import OnlineSoftmaxState, recover_parent_logits
from avqa.exceptions import RoutingError
from avqa.utils.numerics import online_softmax_step


def make_state(
    B: int = 1, H: int = 1, T: int = 4, Dk: int = 8, Dv: int = 16
) -> OnlineSoftmaxState:
    """Allocate an empty OnlineSoftmaxState."""
    return OnlineSoftmaxState.empty(B, H, T, Dk, Dv)


class TestRecoverParentLogits:
    """Tests for recover_parent_logits (spec §7.12)."""

    def test_perfect_mean(self) -> None:
        """If children are exactly the parent mean, recovered parent matches."""
        parent = torch.tensor([[[[3.0]]]])
        children = torch.tensor([[[[1.0, 3.0, 5.0]]]])
        recovered = recover_parent_logits(children, num_children=3)
        assert torch.allclose(recovered, parent, atol=1e-6)

    def test_zero_num_children_raises(self) -> None:
        """num_children <= 0 raises RoutingError."""
        with pytest.raises(RoutingError, match="num_children"):
            recover_parent_logits(torch.zeros(1, 1, 1, 3), num_children=0)

    def test_negative_num_children_raises(self) -> None:
        with pytest.raises(RoutingError, match="num_children"):
            recover_parent_logits(torch.zeros(1, 1, 1, 3), num_children=-1)

    def test_shape(self) -> None:
        """Output has trailing singleton dim."""
        out = recover_parent_logits(torch.zeros(2, 4, 8, 5), num_children=5)
        assert out.shape == (2, 4, 8, 1)


class TestOnlineSoftmaxState:
    """Tests for OnlineSoftmaxState (spec §7.14)."""

    def test_empty_state(self) -> None:
        """Empty state has -inf max and 0 denominator/numerator."""
        state = make_state()
        assert torch.isinf(state.running_max).all()
        assert state.running_max.lt(0).all()
        assert torch.equal(
            state.running_denominator,
            torch.zeros_like(state.running_denominator),
        )
        assert torch.equal(
            state.running_numerator,
            torch.zeros_like(state.running_numerator),
        )

    def test_merge_matches_numerics_helper(self) -> None:
        """State.merge agrees with avqa.utils.numerics.online_softmax_step."""
        torch.manual_seed(0)
        B, H, T, Dk, Dv = 1, 1, 4, 8, 16
        state = make_state(B, H, T, Dk, Dv)
        tile_max = torch.randn(B, H, T, Dk)
        tile_denom = torch.rand(B, H, T, Dk) + 0.1
        tile_num = torch.randn(B, H, T, Dk, Dv)
        merged = state.merge(tile_max, tile_denom, tile_num)
        expected_max, expected_denom, expected_num = online_softmax_step(
            state.running_max,
            state.running_denominator,
            state.running_numerator,
            tile_max,
            tile_denom,
            tile_num,
        )
        assert torch.allclose(merged.running_max, expected_max)
        assert torch.allclose(merged.running_denominator, expected_denom)
        assert torch.allclose(merged.running_numerator, expected_num)

    def test_empty_tile_no_op(self) -> None:
        """Merging an empty tile preserves state."""
        state = make_state()
        torch.manual_seed(0)
        tile_max = torch.randn(1, 1, 4, 8)
        tile_denom = torch.rand(1, 1, 4, 8) + 0.1
        tile_num = torch.randn(1, 1, 4, 8, 16)
        state = state.merge(tile_max, tile_denom, tile_num)
        empty_max = torch.full((1, 1, 4, 8), float("-inf"))
        empty_denom = torch.zeros(1, 1, 4, 8)
        empty_num = torch.zeros(1, 1, 4, 8, 16)
        merged = state.merge(empty_max, empty_denom, empty_num)
        assert torch.allclose(merged.running_max, state.running_max)
        assert torch.allclose(merged.running_denominator, state.running_denominator)
        assert torch.allclose(merged.running_numerator, state.running_numerator)


class TestOnlineSoftmaxStateReplace:
    """Direct tests for OnlineSoftmaxState.replace() (the central correcting
    attention primitive). The pipeline only exercises this through
    :func:`avqa.refinement.vectorized_correction` and :class:`MultiPassRefiner`
    on real data; here we validate the algebraic identity, the m_anchor
    path, and the no-op cases."""

    def test_no_op_when_removed_added_cancel(self) -> None:
        """replace(removed=R, added=R) adds then removes R exactly, leaving state rescaled by ``exp(state.max - new_max)``.

        The state.running_max becomes ``max(state.max, R.max)`` (the
        global scale); the state contribution then equals the original
        state contribution rescaled by ``exp(state.max - new_max)`` —
        amount of which is the standard online-softmax merge scaling.
        """
        torch.manual_seed(0)
        B, H, T, Dk, Dv = 1, 2, 4, 4, 8
        state = make_state(B, H, T, Dk, Dv)
        tile_max = torch.randn(B, H, T, Dk)
        tile_denom = torch.rand(B, H, T, Dk) + 0.1
        tile_num = torch.randn(B, H, T, Dk, Dv)
        state = state.merge(tile_max, tile_denom, tile_num)

        max_r = tile_max  # equal to state.running_max exactly
        denom_r = torch.rand(B, H, T, Dk) + 0.1
        num_r = torch.randn(B, H, T, Dk, Dv)

        new = state.replace(
            removed_max=max_r,
            removed_denominator=denom_r,
            removed_numerator=num_r,
            added_max=max_r,
            added_denominator=denom_r,
            added_numerator=num_r,
        )
        # new_max = max(state.running_max, max_r, max_r) = state.running_max,
        # scale_old = exp(state.max - new_max) = 1, removed and added cancel.
        assert torch.allclose(new.running_max, state.running_max)
        assert torch.allclose(new.running_denominator, state.running_denominator)
        assert torch.allclose(new.running_numerator, state.running_numerator)

    def test_no_state_when_empty(self) -> None:
        """replace against an empty (-inf max) state works without NaN."""
        torch.manual_seed(0)
        B, H, T, Dk, Dv = 1, 1, 3, 4, 6
        state = make_state(B, H, T, Dk, Dv)
        max_r = torch.zeros(B, H, T, Dk)
        denom_r = torch.ones(B, H, T, Dk)
        num_r = torch.ones(B, H, T, Dk, Dv)
        new = state.replace(
            removed_max=max_r,
            removed_denominator=denom_r,
            removed_numerator=num_r,
            added_max=max_r,
            added_denominator=denom_r,
            added_numerator=num_r,
        )
        # After empty -inf + a tile with max=0, the new max should be 0
        # (no contribution cancellation since added==removed).
        assert torch.isfinite(new.running_max).all()
        assert torch.equal(new.running_max, torch.zeros_like(new.running_max))
        assert torch.equal(
            new.running_denominator, torch.zeros_like(new.running_denominator)
        )
        assert torch.equal(
            new.running_numerator, torch.zeros_like(new.running_numerator)
        )

    def test_m_anchor_matches_caller_scale(self) -> None:
        """When m_anchor is passed, replace uses it (not the raw removed_max)
        as the reference for removed_denominator / added_denominator."""
        torch.manual_seed(0)
        B, H, T, Dk, Dv = 1, 1, 3, 4, 6
        # Empty state (running_max = -inf).
        state = make_state(B, H, T, Dk, Dv)
        # Caller pre-computed exp(x - 1.0) for both tiles.
        m_anchor = torch.full((B, H, T, Dk), 1.0)
        removed_max = torch.full((B, H, T, Dk), 5.0)  # raw logit max (5 > m_anchor).
        denom_r = torch.ones(B, H, T, Dk)
        num_r = torch.ones(B, H, T, Dk, Dv)
        new = state.replace(
            removed_max=removed_max,
            removed_denominator=denom_r,
            removed_numerator=num_r,
            added_max=removed_max,
            added_denominator=denom_r,
            added_numerator=num_r,
            m_anchor=m_anchor,
        )
        # added == removed → no-op → all zeros.
        assert torch.isfinite(new.running_max).all()
        assert torch.equal(
            new.running_denominator, torch.zeros_like(new.running_denominator)
        )
        assert torch.equal(
            new.running_numerator, torch.zeros_like(new.running_numerator)
        )

    def test_m_anchor_differs_from_default(self) -> None:
        """Without m_anchor, replace uses max(state, removed, added).

        With m_anchor, the running_max becomes max(m_anchor, state) —
        different from the default when m_anchor < removed/added.
        """
        torch.manual_seed(0)
        B, H, T, Dk, Dv = 1, 1, 2, 3, 4
        state = make_state(B, H, T, Dk, Dv)
        # Tile stats with raw max = 10.
        max_r = torch.full((B, H, T, Dk), 10.0)
        denom_r = torch.ones(B, H, T, Dk)
        num_r = torch.ones(B, H, T, Dk, Dv)
        # Empty state (-inf running_max) — default uses max(-inf, 10, 10) = 10.
        default = state.replace(
            removed_max=max_r,
            removed_denominator=denom_r,
            removed_numerator=num_r,
            added_max=max_r,
            added_denominator=denom_r,
            added_numerator=num_r,
        )
        # With m_anchor = 0, new_max = max(0, -inf) = 0.
        with_anchor = state.replace(
            removed_max=max_r,
            removed_denominator=denom_r,
            removed_numerator=num_r,
            added_max=max_r,
            added_denominator=denom_r,
            added_numerator=num_r,
            m_anchor=torch.zeros_like(max_r),
        )
        # Different new_max → different running_max.
        assert not torch.allclose(default.running_max, with_anchor.running_max)
        assert torch.equal(default.running_max, torch.full_like(default.running_max, 10.0))
        assert torch.equal(with_anchor.running_max, torch.zeros_like(with_anchor.running_max))
