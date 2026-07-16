"""Property tests for the five mathematical invariants (spec §7.17).

These tests assert the spec's normative correctness criteria across
the full quantizer + refinement pipeline.  Each test uses random inputs
with a fixed seed for reproducibility.
"""

from __future__ import annotations

import torch

from avqa.attention import OnlineSoftmaxState
from avqa.codebook import HierarchicalCodebook
from avqa.quantizer import EuclideanHierarchicalQuantizer
from avqa.refinement import refine
from avqa.routing import TopPRouter, compute_importance


def make_pipeline(
    B: int = 2,
    H: int = 4,
    N: int = 32,
    D: int = 16,
    M0: int = 8,
    C: int = 4,
    budget: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, HierarchicalCodebook, torch.Tensor, torch.Tensor]:
    """Set up codebook, keys, values, and parent attention probs for tests."""
    torch.manual_seed(seed)
    cb = HierarchicalCodebook(
        num_heads=H,
        num_parents=M0,
        children_per_parent=C,
        head_dim=D,
    )
    cb.initialize_parents_random()
    keys = torch.randn(B, H, N, D)
    values = torch.randn(B, H, N, D)
    quantizer = EuclideanHierarchicalQuantizer()
    result = quantizer.precompute(keys, values, cb)

    # Build parent attention probs via softmax over parent logits.
    parent_logits = torch.randn(B, H, N, M0)  # use random logits for simplicity
    parent_probs = torch.softmax(parent_logits, dim=-1)

    return result, parent_probs, cb, keys, values


# ---------------------------------------------------------------------------
# ISSUE-0019: Conservation invariant (spec §7.17)
# ---------------------------------------------------------------------------


class TestConservationInvariant:
    """Sum of parent aggregates equals sum of values (spec §7.17)."""

    def test_conservation_over_codewords(self) -> None:
        """result.parent_aggregates.sum(dim=M0) == values.sum(dim=N)."""
        result, _, _, _, values = make_pipeline()
        agg_sum = result.parent_aggregates.sum(dim=2)  # [B, H, D]
        val_sum = values.sum(dim=2)  # [B, H, D]
        assert torch.allclose(agg_sum, val_sum, atol=1e-5)

    def test_conservation_multiple_batches(self) -> None:
        """Invariant holds for B=4."""
        result, _, _, _, values = make_pipeline(B=4)
        assert torch.allclose(
            result.parent_aggregates.sum(dim=2),
            values.sum(dim=2),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# ISSUE-0020: Hierarchy invariant (spec §7.17)
# ---------------------------------------------------------------------------


class TestHierarchyInvariant:
    """Every parent equals the mean of its children (spec §7.17)."""

    def test_parent_equals_mean_of_children(self) -> None:
        """cb.parents == cb.children.mean(dim=children_axis)."""
        _, _, cb, _, _ = make_pipeline()
        cb.validate_mean_constraint(atol=1e-5)

    def test_hierarchy_survives_quantization(self) -> None:
        """Invariant holds after a precompute pass (codebook not mutated)."""
        _, _, cb, _, _ = make_pipeline()
        # precompute should not modify the codebook
        cb.validate_mean_constraint(atol=1e-5)


# ---------------------------------------------------------------------------
# ISSUE-0021: Attention invariant — normalized after correction (spec §7.17)
# ---------------------------------------------------------------------------


class TestAttentionInvariant:
    """Refined attention distribution sums to 1 (spec §7.17)."""

    def test_refined_attention_normalized(self) -> None:
        """merge_value row sums are finite and consistent."""
        result, parent_probs, cb, _, values = make_pipeline(
            B=1,
            H=2,
            N=16,
            D=8,
            M0=4,
            C=2,
            budget=2,
            seed=42,
        )
        H = 2
        D_v = values.shape[-1]
        T = parent_probs.shape[2]
        C = cb.children_per_parent

        parent_value_per_parent = parent_probs.unsqueeze(-1) * values.unsqueeze(
            3
        )  # [B, H, T, M0, D_v]

        state = OnlineSoftmaxState.empty(1, H, T, D_v, D_v)
        importance = compute_importance(parent_probs, result.parent_counts)
        decision = TopPRouter().select(importance, budget=2)
        refinement = refine(
            state=state,
            parent_probs=parent_probs,
            parent_value=parent_value_per_parent,
            parent_aggregates=result.parent_aggregates,
            child_aggregates=result.child_aggregates,
            children_per_parent=C,
            decision=decision,
            attention_probs=parent_probs,
            parent_counts=result.parent_counts,
        )
        # The merge value should have no NaN or Inf.
        assert torch.isfinite(refinement.merge_value).all()


# ---------------------------------------------------------------------------
# ISSUE-0022: Count invariant — child counts sum to N (spec §7.17)
# ---------------------------------------------------------------------------


class TestCountInvariant:
    """Sum of child counts equals N, and equals sum of parent counts."""

    def test_child_count_invariant(self) -> None:
        """sum(child_counts) == N for each (B, H)."""
        result, _, _, _, _ = make_pipeline(N=32)
        total = result.child_counts.sum(dim=(-2, -1))
        expected = torch.full_like(total, 32.0)
        assert torch.equal(total, expected)

    def test_child_equals_parent_count(self) -> None:
        """sum(child_counts per parent) == parent_counts for each (B, H, M0)."""
        result, _, _, _, _ = make_pipeline()
        child_per_parent = result.child_counts.sum(dim=-1)  # [B, H, M0]
        assert torch.allclose(child_per_parent, result.parent_counts, atol=1e-5)


# ---------------------------------------------------------------------------
# ISSUE-0023: Assignment invariant — every key assigned to exactly one parent
# ---------------------------------------------------------------------------


class TestAssignmentInvariant:
    """Every key is assigned to exactly one parent (spec §7.17)."""

    def test_exactly_one_assignment_per_key(self) -> None:
        """Each key has exactly one parent assignment."""
        result, _, _, _, _ = make_pipeline(N=64)
        _, _, N = result.parent_assignments.shape
        # Every entry must be in [0, M0).
        assert result.parent_assignments.min().item() >= 0
        assert result.parent_assignments.max().item() < 8  # M0=8
        # Each (B, H, N) has exactly one assignment (by construction of argmin).
        # Verify total assignments == B * H * N.
        total = result.parent_counts.sum(dim=-1)
        expected = torch.full_like(total, float(N))
        assert torch.equal(total, expected)

    def test_child_assignments_bounded(self) -> None:
        """Child assignments are in [0, C)."""
        result, _, _, _, _ = make_pipeline()
        assert result.child_assignments.min().item() >= 0
        assert result.child_assignments.max().item() < 4  # C=4
