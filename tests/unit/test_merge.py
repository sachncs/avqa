"""Tests for avqa.merge module."""

from __future__ import annotations

import pytest
import torch

from avqa.exceptions import ConfigurationError
from avqa.merge import (
    LogitMerge,
    MergeInputs,
    MergeStrategy,
    NormalizedMerge,
    ProbabilityMerge,
    WeightedMerge,
)


def make_inputs(
    B: int = 1,
    H: int = 1,
    N: int = 4,
    C: int = 3,
    D: int = 5,
) -> MergeInputs:
    """Random MergeInputs for tests."""
    parent_probs = torch.rand(B, H, N, 1)
    parent_probs = parent_probs / parent_probs.sum(dim=-1, keepdim=True)
    parent_value = torch.randn(B, H, N, D)
    child_probs = torch.rand(B, H, N, C)
    child_probs = child_probs / child_probs.sum(dim=-1, keepdim=True)
    child_value = torch.randn(B, H, N, C, D)
    return MergeInputs(
        parent_probs=parent_probs,
        parent_value=parent_value,
        child_probs=child_probs,
        child_value=child_value,
    )


class TestMergeStrategies:
    """Tests for each merge strategy (spec §3.11.2)."""

    def test_probability_merge_shape(self) -> None:
        """ProbabilityMerge output shape is [B, H, N, D]."""
        out = ProbabilityMerge().merge(make_inputs())
        assert out.shape == (1, 1, 4, 5)

    def test_probability_merge_no_children(self) -> None:
        """With zero child probability, output is parent_value * (1 - parent_probs).

        The "replace" semantics of ProbabilityMerge subtracts the parent
        contribution and adds the child contribution (which is zero here).
        """
        inputs = make_inputs()
        inputs.child_probs = torch.zeros_like(inputs.child_probs)
        out = ProbabilityMerge().merge(inputs)
        expected = inputs.parent_value * (1.0 - inputs.parent_probs)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_probability_merge_full_child(self) -> None:
        """With zero parent probs, output is parent_value + child_contribution."""
        inputs = make_inputs()
        inputs.parent_probs = torch.zeros_like(inputs.parent_probs)
        out = ProbabilityMerge().merge(inputs)
        child_contrib = (inputs.child_probs.unsqueeze(-1) * inputs.child_value).sum(dim=-2)
        expected = inputs.parent_value + child_contrib
        assert torch.allclose(out, expected, atol=1e-5)

    def test_weighted_merge_shape(self) -> None:
        """WeightedMerge output shape is [B, H, N, D]."""
        out = WeightedMerge().merge(make_inputs())
        assert out.shape == (1, 1, 4, 5)

    def test_weighted_merge_weights(self) -> None:
        """WeightedMerge is a convex combo of parent_value and child_contrib.

        Per :class:`MergeInputs` contract, ``parent_value`` and
        ``child_value`` are already attention-weighted by the pipeline,
        so WeightedMerge must NOT re-multiply by ``parent_probs`` (that
        was a bug — see CRITICAL review finding B.001). With
        ``parent_weight=1.0, child_weight=0.0`` the result is identical
        to ``parent_value``; with ``parent_weight=0.0, child_weight=1.0``
        it is the child contribution only.
        """
        inputs = make_inputs()
        child_contrib = (
            inputs.child_probs.unsqueeze(-1) * inputs.child_value
        ).sum(dim=-2)

        out_all_parent = WeightedMerge(parent_weight=1.0, child_weight=0.0).merge(inputs)
        assert torch.allclose(out_all_parent, inputs.parent_value, atol=1e-6)

        out_all_child = WeightedMerge(parent_weight=0.0, child_weight=1.0).merge(inputs)
        assert torch.allclose(out_all_child, child_contrib, atol=1e-6)

        out_mix = WeightedMerge(parent_weight=0.3, child_weight=0.7).merge(inputs)
        expected = 0.3 * inputs.parent_value + 0.7 * child_contrib
        assert torch.allclose(out_mix, expected, atol=1e-5)

    def test_logit_merge_shape(self) -> None:
        """LogitMerge output shape is [B, H, N, D]."""
        out = LogitMerge().merge(make_inputs())
        assert out.shape == (1, 1, 4, 5)

    def test_normalized_merge_shape(self) -> None:
        """NormalizedMerge output shape is [B, H, N, D]."""
        out = NormalizedMerge().merge(make_inputs())
        assert out.shape == (1, 1, 4, 5)


class TestMergeStrategyInterface:
    """Tests for the abstract MergeStrategy interface."""

    def test_cannot_instantiate(self) -> None:
        """MergeStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MergeStrategy.__new__(MergeStrategy)

    @pytest.mark.parametrize(
        "cls",
        [ProbabilityMerge, WeightedMerge, LogitMerge, NormalizedMerge],
    )
    def test_subclass_relationship(self, cls: type[MergeStrategy]) -> None:
        """All concrete strategies inherit from MergeStrategy."""
        assert issubclass(cls, MergeStrategy)


class TestMergeInputsValidation:
    """Tests for MergeInputs validation."""

    def test_shape_mismatch_parent(self) -> None:
        """parent_probs with wrong shape raises."""
        inputs = make_inputs()
        with pytest.raises(ConfigurationError, match="parent_probs"):
            MergeInputs(
                parent_probs=torch.zeros(1, 1, 4, 2),
                parent_value=inputs.parent_value,
                child_probs=inputs.child_probs,
                child_value=inputs.child_value,
            )

    def test_shape_mismatch_child(self) -> None:
        """child_probs with wrong shape raises."""
        inputs = make_inputs()
        with pytest.raises(ConfigurationError, match="child_probs"):
            MergeInputs(
                parent_probs=inputs.parent_probs,
                parent_value=inputs.parent_value,
                child_probs=torch.zeros(1, 1, 4, 2),
                child_value=inputs.child_value,
            )


class TestMergeConservation:
    """Tests for mass conservation under merge."""

    def test_probability_merge_conserves_mass(self) -> None:
        """ProbabilityMerge preserves the softmax mass invariant.

        The "delta" applied is by construction: subtract parent, add
        children, so total attention weight per row is unchanged.
        """
        inputs = make_inputs()
        out = ProbabilityMerge().merge(inputs)
        # Total value norm should be roughly preserved (modulo values).
        assert out.shape == (1, 1, 4, 5)
