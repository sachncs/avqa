"""Tests for avqa.quantizer module."""

from __future__ import annotations

import pytest
import torch

from avqa.codebook import HierarchicalCodebook
from avqa.exceptions import ShapeError
from avqa.quantizer import (
    EuclideanHierarchicalQuantizer,
    QuantizationResult,
    VectorQuantizer,
)


def make_setup(
    num_heads: int = 2,
    num_parents: int = 8,
    children_per_parent: int = 4,
    head_dim: int = 16,
    num_keys: int = 32,
    batch_size: int = 1,
    seed: int = 0,
) -> tuple[
    HierarchicalCodebook,
    torch.Tensor,
    torch.Tensor,
    EuclideanHierarchicalQuantizer,
]:
    """Build a codebook + random keys/values for tests."""
    torch.manual_seed(seed)
    cb = HierarchicalCodebook(
        num_heads=num_heads,
        num_parents=num_parents,
        children_per_parent=children_per_parent,
        head_dim=head_dim,
    )
    cb.initialize_parents_random()
    keys = torch.randn(batch_size, num_heads, num_keys, head_dim)
    values = torch.randn(batch_size, num_heads, num_keys, head_dim)
    return cb, keys, values, EuclideanHierarchicalQuantizer()


class TestAssignmentShapes:
    """Tests for the documented tensor shapes (spec §8.8)."""

    def test_assignments_shape(self) -> None:
        """parent/child assignments have shape [B, H, N]."""
        cb, keys, values, q = make_setup()
        result = q.precompute(keys, values, cb)
        B, H, T, _ = keys.shape
        N = T
        assert result.parent_assignments.shape == (B, H, N)
        assert result.child_assignments.shape == (B, H, N)

    def test_aggregates_shapes(self) -> None:
        """Aggregates have shape [B, H, M_0, D] and [B, H, M_0, C, D]."""
        cb, keys, values, q = make_setup(num_parents=8, children_per_parent=4, head_dim=16)
        result = q.precompute(keys, values, cb)
        B, H, _, D = keys.shape
        assert result.parent_aggregates.shape == (B, H, 8, D)
        assert result.child_aggregates.shape == (B, H, 8, 4, D)

    def test_counts_shapes(self) -> None:
        """Counts have shape [B, H, M_0] and [B, H, M_0, C]."""
        cb, keys, values, q = make_setup()
        result = q.precompute(keys, values, cb)
        B, H, _, _ = keys.shape
        assert result.parent_counts.shape == (B, H, 8)
        assert result.child_counts.shape == (B, H, 8, 4)


class TestMathematicalInvariants:
    """Tests for the conservation invariants (spec §7.17, §8.6)."""

    def test_count_invariant(self) -> None:
        """Sum of parent counts equals N (spec §7.17)."""
        cb, keys, values, q = make_setup(num_keys=32)
        result = q.precompute(keys, values, cb)
        assert torch.equal(
            result.parent_counts.sum(dim=-1),
            torch.full_like(result.parent_counts.sum(dim=-1), 32.0),
        )

    def test_count_invariant_sum_of_child(self) -> None:
        """Sum of child counts equals N."""
        cb, keys, values, q = make_setup(num_keys=32)
        result = q.precompute(keys, values, cb)
        total = result.child_counts.sum(dim=(-2, -1))
        assert torch.equal(
            total,
            torch.full_like(total, 32.0),
        )

    def test_conservation_invariant(self) -> None:
        """Sum of parent aggregates equals sum of values (spec §7.17)."""
        cb, keys, values, q = make_setup(num_keys=16)
        result = q.precompute(keys, values, cb)
        assert torch.allclose(
            result.parent_aggregates.sum(dim=-2),
            values.sum(dim=-2),
            atol=1e-5,
        )

    def test_child_aggregates_subset_of_parent(self) -> None:
        """Sum of child aggregates for a parent equals that parent's aggregate."""
        cb, keys, values, q = make_setup()
        result = q.precompute(keys, values, cb)
        child_sum = result.child_aggregates.sum(dim=-2)
        assert torch.allclose(child_sum, result.parent_aggregates, atol=1e-5)


class TestNearestAssignment:
    """Tests that the quantizer really picks the nearest codeword (spec §7.5)."""

    def test_key_assigned_to_closest_parent(self) -> None:
        """A key is assigned to the parent with minimum L2 distance."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(num_heads=1, num_parents=4, children_per_parent=2, head_dim=8)
        cb.initialize_parents_random()
        # Build a key that exactly equals parent 0 (so distance is 0 there).
        key = cb.parents[0, 0].clone().reshape(1, 1, 1, -1)
        values = torch.ones_like(key)
        result = EuclideanHierarchicalQuantizer().precompute(key, values, cb)
        assert result.parent_assignments.item() == 0

    def test_child_within_selected_parent(self) -> None:
        """Child index is bounded by children_per_parent."""
        cb, keys, values, q = make_setup(children_per_parent=4)
        result = q.precompute(keys, values, cb)
        assert result.child_assignments.min().item() >= 0
        assert result.child_assignments.max().item() < 4


class TestShapeErrors:
    """Tests for shape-validation guards (spec §6.12)."""

    def test_keys_values_shape_mismatch(self) -> None:
        """keys/values with different shapes raise ShapeError."""
        cb, _, _, q = make_setup()
        keys = torch.randn(1, 2, 16, 16)
        values = torch.randn(1, 2, 16, 8)
        with pytest.raises(ShapeError, match="identical shapes"):
            q.precompute(keys, values, cb)

    def test_keys_rank_mismatch(self) -> None:
        """keys with wrong rank raise ShapeError."""
        cb, _, _, q = make_setup()
        keys = torch.randn(1, 2, 16, 16, 16)
        values = torch.randn(1, 2, 16, 16, 16)
        with pytest.raises(ShapeError, match="rank 4"):
            q.precompute(keys, values, cb)

    def test_keys_head_dim_mismatch(self) -> None:
        """keys with wrong head_dim raise ShapeError."""
        cb, _, _, q = make_setup(head_dim=16)
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        with pytest.raises(ShapeError, match="head_dim"):
            q.precompute(keys, values, cb)


class TestDeterministicMode:
    """Tests for deterministic assignment (spec §8.13)."""

    def test_same_seed_same_assignments(self) -> None:
        """Two runs with the same codebook+keys yield identical assignments."""
        cb, keys, values, q = make_setup()
        result1 = q.precompute(keys, values, cb)
        result2 = q.precompute(keys, values, cb)
        assert torch.equal(result1.parent_assignments, result2.parent_assignments)
        assert torch.equal(result1.child_assignments, result2.child_assignments)


class TestQuantizationResultValidation:
    """Tests for QuantizationResult.validate_shapes."""

    def test_validate_passes(self) -> None:
        """Valid shapes pass validation."""
        cb, keys, values, q = make_setup(
            num_heads=2, num_parents=4, children_per_parent=3, head_dim=8
        )
        result = q.precompute(keys, values, cb)
        # Should not raise
        result.validate_shapes(num_heads=2, num_parents=4, children_per_parent=3, head_dim=8)

    def test_validate_rejects_wrong_shape(self) -> None:
        """Wrong-shape result raises ShapeError."""
        result = QuantizationResult(
            parent_assignments=torch.zeros(1, 1, 8, dtype=torch.long),
            child_assignments=torch.zeros(1, 1, 8, dtype=torch.long),
            parent_aggregates=torch.zeros(1, 1, 2, 4),
            child_aggregates=torch.zeros(1, 1, 2, 2, 4),
            parent_counts=torch.zeros(1, 1, 2),
            child_counts=torch.zeros(1, 1, 2, 2),
        )
        with pytest.raises(ShapeError, match="parent_aggregates"):
            result.validate_shapes(num_heads=1, num_parents=99, children_per_parent=2, head_dim=4)


class TestAbstractInterface:
    """Tests for the abstract VectorQuantizer interface."""

    def test_base_class_is_abstract(self) -> None:
        """VectorQuantizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorQuantizer()  # type: ignore[abstract]

    def test_euclidean_is_subclass(self) -> None:
        """EuclideanHierarchicalQuantizer subclasses VectorQuantizer."""
        assert issubclass(EuclideanHierarchicalQuantizer, VectorQuantizer)
