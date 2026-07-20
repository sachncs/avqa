"""Hand-computed oracle tests.

These tests verify AVQA's algebraic core against closed-form answers
derived directly from the algorithm specifications rather than from
the codebase. They are the closest thing the project has to a
mathematical specification test suite.

ISSUE-0025: spec §3.25 acceptance criterion ``functional acceptance``.
If any of these break, the algorithm has drifted from the spec, not
just from an internal implementation.
"""

from __future__ import annotations

import pytest
import torch

from avqa.codebook import HierarchicalCodebook
from avqa.hopfield import (
    hopfield_logits,
    paper_beta,
    per_query_beta,
)
from avqa.quantizer import EuclideanHierarchicalQuantizer


class TestHandComputedReference:
    """Verify quantizer against a hand-derived expected output."""

    def test_known_assignment(self) -> None:
        """Keys assigned to the nearest parent; aggregates match."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(
            num_heads=1,
            num_parents=2,
            children_per_parent=2,
            head_dim=2,
        )
        # Set parents to known positions.
        cb.parents = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
        # Set children near parents (mean constraint: parent = mean(children)).
        cb.children = torch.tensor(
            [
                [[[0.1, 0.1], [-0.1, -0.1]], [[10.1, 10.1], [9.9, 9.9]]],
            ]
        )

        # Keys: two near parent 0, two near parent 1.
        keys = torch.tensor([[[[0.05, 0.05], [-0.05, -0.05], [10.05, 10.05], [9.95, 9.95]]]])
        # Values: distinct so we can verify aggregation.
        values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]])

        result = EuclideanHierarchicalQuantizer().precompute(keys, values, cb)

        # Expected parent assignments: [0, 0, 1, 1].
        expected_parent = torch.tensor([[[0, 0, 1, 1]]])
        assert torch.equal(result.parent_assignments, expected_parent)

        # Parent 0 aggregate = [1,2] + [3,4] = [4, 6].
        # Parent 1 aggregate = [5,6] + [7,8] = [12, 14].
        expected_agg = torch.tensor([[[[4.0, 6.0], [12.0, 14.0]]]])
        assert torch.allclose(result.parent_aggregates, expected_agg, atol=1e-5)

        # Parent counts: [2, 2].
        expected_counts = torch.tensor([[[2.0, 2.0]]])
        assert torch.equal(result.parent_counts, expected_counts)

    def test_single_key_exact_match(self) -> None:
        """A key that exactly equals a parent gets assigned there."""
        cb = HierarchicalCodebook(
            num_heads=1,
            num_parents=3,
            children_per_parent=2,
            head_dim=4,
        )
        cb.parents = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )
        cb.children = torch.tensor(
            [
                [
                    [[1.1, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]],
                    [[0.0, 1.1, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
                    [[0.0, 0.0, 1.1, 0.0], [0.0, 0.0, 0.9, 0.0]],
                ],
            ]
        )

        # Key exactly matches parent 1.
        key = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
        value = torch.tensor([[[[10.0, 20.0, 30.0, 40.0]]]])

        result = EuclideanHierarchicalQuantizer().precompute(key, value, cb)
        assert result.parent_assignments.item() == 1
        # Parent 1 aggregate should equal the key's value.
        assert torch.allclose(
            result.parent_aggregates[0, 0, 1],
            value[0, 0],
            atol=1e-5,
        )

    def test_conservation_hand_computed(self) -> None:
        """Conservation invariant with known values."""
        cb = HierarchicalCodebook(
            num_heads=1,
            num_parents=2,
            children_per_parent=2,
            head_dim=2,
        )
        cb.parents = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])
        cb.children = torch.tensor(
            [
                [[[0.1, 0.1], [-0.1, -0.1]], [[5.1, 5.1], [4.9, 4.9]]],
            ]
        )

        keys = torch.tensor([[[[0.0, 0.0], [5.0, 5.0]]]])
        values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

        result = EuclideanHierarchicalQuantizer().precompute(keys, values, cb)

        # Sum of aggregates = [1,2] + [3,4] = [4, 6] = sum of values.
        agg_sum = result.parent_aggregates.sum(dim=2)[0, 0]
        val_sum = values.sum(dim=2)[0, 0]
        assert torch.allclose(agg_sum, val_sum, atol=1e-5)

    def test_one_hot_peaked_distribution_doubles_beta(self) -> None:
        """`per_query_beta`: peaked distribution (entropy = 0) doubles beta_q.

        Hand-derived from ``hopfield.per_query_beta``: H_top = 0 →
        schedule = ``1 + 1 / (1 + 0) = 2``; beta_q = beta_init · 2.
        """
        p_one_hot = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        result = per_query_beta(p_one_hot, beta_init=1.0, adaptive="entropy")
        # beta_q = 1.0 · (1 + 1/(1+0)) = 2.0
        assert result.item() == 2.0

    def test_linear_peaked_unchanged(self) -> None:
        """`per_query_beta`: linear schedule at H_top = 0 gives beta_q = beta_init.

        From ``hopfield.per_query_beta``: schedule = ``1 + alpha * H_top``.
        At ``H_top = 0`` this is 1; at ``beta_init = 1.5, alpha = 0.7`` it is 1.5.
        """
        p_one_hot = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        result = per_query_beta(p_one_hot, beta_init=1.5, adaptive="linear", alpha=0.7)
        assert result.item() == 1.5

    def test_paper_beta_d_value(self) -> None:
        """``paper_beta(d) = 1 / sqrt(d)`` for the canonical d values."""
        assert paper_beta(64) == pytest.approx(1.0 / 8.0)
        assert paper_beta(128) == pytest.approx(128**-0.5)

    def test_hopfield_logits_constant_beta_passthrough(self) -> None:
        """``hopfield_logits`` with beta_q = 1 returns the raw logits."""
        base = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        beta_q = torch.ones(1, 1, 1)
        out = hopfield_logits(base, beta_q)
        torch.testing.assert_close(out, base)

    def test_hopfield_logits_double_beta_doubles(self) -> None:
        """Doubling beta_q doubles the logits (scalar multiplication)."""
        base = torch.tensor([[[[1.0, -2.0, 3.0]]]])
        beta_q = torch.full((1, 1, 1), 2.0)
        out = hopfield_logits(base, beta_q)
        torch.testing.assert_close(out, base * 2.0)
