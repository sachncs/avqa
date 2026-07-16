"""Tests for avqa.codebook module."""

from __future__ import annotations

import pytest
import torch

from avqa.codebook import CodebookStats, HierarchicalCodebook
from avqa.exceptions import CodebookError


class TestHierarchicalCodebookConstruction:
    """Tests for codebook construction and shape invariants."""

    def test_default_construction(self) -> None:
        """Defaults produce the documented shapes (spec §8.8)."""
        cb = HierarchicalCodebook()
        assert cb.parents.shape == (1, 64, 64)
        assert cb.children.shape == (1, 64, 4, 64)

    def test_custom_construction(self) -> None:
        """Custom dimensions are honored."""
        cb = HierarchicalCodebook(
            num_heads=8,
            num_parents=32,
            children_per_parent=8,
            head_dim=64,
        )
        assert cb.parents.shape == (8, 32, 64)
        assert cb.children.shape == (8, 32, 8, 64)

    def test_rejects_invalid_arguments(self) -> None:
        """Invalid arguments raise CodebookError."""
        with pytest.raises(CodebookError):
            HierarchicalCodebook(num_heads=0)
        with pytest.raises(CodebookError):
            HierarchicalCodebook(num_parents=0)
        with pytest.raises(CodebookError):
            HierarchicalCodebook(children_per_parent=0)
        with pytest.raises(CodebookError):
            HierarchicalCodebook(head_dim=0)
        with pytest.raises(CodebookError):
            HierarchicalCodebook(perturbation_scale=-1.0)

    def test_dtype_passthrough(self) -> None:
        """Dtype is applied to both tensors."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=4, head_dim=8, dtype=torch.float16)
        assert cb.parents.dtype == torch.float16
        assert cb.children.dtype == torch.float16


class TestMeanConstraint:
    """Tests for the parent-child mean invariant (spec §7.9, §7.17)."""

    def test_initial_children_satisfy_constraint(self) -> None:
        """After construction + child init, parent == mean(children)."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        cb.initialize_parents_random()
        cb.validate_mean_constraint()

    def test_initialize_children_satisfies_constraint(self) -> None:
        """initialize_children_around_parents preserves the constraint."""
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        cb.initialize_parents_random()
        cb.initialize_children_around_parents()
        cb.validate_mean_constraint()

    def test_validate_raises_on_violation(self) -> None:
        """validate_mean_constraint raises when children are perturbed."""
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        cb.initialize_parents_random()
        cb.children = cb.children + 1.0  # break the constraint
        with pytest.raises(CodebookError, match="mean constraint"):
            cb.validate_mean_constraint()

    def test_reproject_restores_constraint(self) -> None:
        """reproject_parents restores the constraint after perturbation."""
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        cb.initialize_parents_random()
        cb.children = cb.children + 1.0
        cb.reproject_parents()
        cb.validate_mean_constraint()

    def test_reproject_sets_parents_to_mean(self) -> None:
        """After reproject, parents equal the children mean exactly."""
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        cb.initialize_parents_random()
        cb.initialize_children_around_parents()
        cb.reproject_parents()
        assert torch.allclose(cb.parents, cb.children.mean(dim=2))


class TestChildInitialization:
    """Tests for child initialization (spec §8.10)."""

    def test_perturbation_scale_applied(self) -> None:
        """Children are within the configured perturbation scale of parents."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(
            num_heads=2,
            num_parents=8,
            children_per_parent=4,
            head_dim=16,
            perturbation_scale=0.1,
        )
        cb.initialize_parents_random()
        cb.initialize_children_around_parents()
        diff = (cb.children - cb.parents.unsqueeze(2)).abs().max().item()
        # Children are at parent + scale * noise; max diff < 4 * scale with
        # high probability for Gaussian noise.
        assert diff < 4 * cb.perturbation_scale

    def test_empirical_perturbation_scale(self) -> None:
        """Empirical std of perturbation matches configured scale (spec §8.10)."""
        torch.manual_seed(42)
        scale = 0.2
        cb = HierarchicalCodebook(
            num_heads=2, num_parents=32, children_per_parent=16, head_dim=32,
            perturbation_scale=scale,
        )
        cb.initialize_parents_random()
        cb.initialize_children_around_parents()
        noise = (cb.children - cb.parents.unsqueeze(2)) / scale
        # Empirical std should be close to 1.0 (std of N(0,I)).
        empirical_std = noise.std().item()
        assert abs(empirical_std - 1.0) < 0.15

    def test_custom_perturbation_shape_validated(self) -> None:
        """Wrong-shape perturbation raises CodebookError."""
        cb = HierarchicalCodebook(num_heads=2, num_parents=8, children_per_parent=4, head_dim=16)
        bad_perturbation = torch.zeros(1, 1, 1, 1)
        with pytest.raises(CodebookError, match="shape"):
            cb.initialize_children_around_parents(perturbation=bad_perturbation)

    def test_deterministic_with_generator(self) -> None:
        """Same generator seed produces same codebook."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        cb1 = HierarchicalCodebook(num_heads=1, num_parents=4, children_per_parent=2, head_dim=8)
        cb2 = HierarchicalCodebook(num_heads=1, num_parents=4, children_per_parent=2, head_dim=8)
        cb1.initialize_parents_random(generator=gen1)
        cb2.initialize_parents_random(generator=gen2)
        cb1.initialize_children_around_parents(generator=gen1)
        cb2.initialize_children_around_parents(generator=gen2)
        assert torch.equal(cb1.parents, cb2.parents)
        assert torch.equal(cb1.children, cb2.children)


class TestEMAUpdate:
    """Tests for the EMA training step (spec §8.9)."""

    def test_ema_decay_applied_to_children(self) -> None:
        """EMA smoothly interpolates children between old and new."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        old_children = cb.children.clone()
        new_children = torch.zeros_like(cb.children) + 1.0
        cb.ema_update(new_parents=cb.parents.clone(), new_children=new_children, decay=0.9)
        # children <- 0.9 * old + 0.1 * new
        assert torch.allclose(cb.children, 0.9 * old_children + 0.1 * new_children)

    def test_ema_parents_equal_mean_of_children(self) -> None:
        """After EMA, parents = mean(children) per spec §8.9."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        new_children = torch.randn_like(cb.children)
        cb.ema_update(new_parents=cb.parents.clone(), new_children=new_children, decay=0.9)
        assert torch.allclose(cb.parents, cb.children.mean(dim=2))

    def test_ema_preserves_constraint(self) -> None:
        """EMA on codewords that satisfy the constraint keeps it."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        cb.validate_mean_constraint()
        cb.ema_update(
            new_parents=cb.parents.clone(),
            new_children=cb.children.clone(),
            decay=0.99,
        )
        cb.validate_mean_constraint()

    def test_ema_shape_validation(self) -> None:
        """Shape mismatches raise CodebookError."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        bad = torch.zeros(1, 1, 1)
        with pytest.raises(CodebookError, match="shape"):
            cb.ema_update(new_parents=bad, new_children=cb.children, decay=0.9)

    def test_ema_decay_range(self) -> None:
        """Decay outside [0, 1] raises."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        with pytest.raises(CodebookError):
            cb.ema_update(
                new_parents=cb.parents,
                new_children=cb.children,
                decay=1.5,
            )
        with pytest.raises(CodebookError):
            cb.ema_update(
                new_parents=cb.parents,
                new_children=cb.children,
                decay=-0.1,
            )


class TestCodebookSerialization:
    """Tests for codebook state_dict round-trip (spec §3.20)."""

    def test_round_trip(self) -> None:
        """state_dict round-trips faithfully."""
        torch.manual_seed(0)
        cb = HierarchicalCodebook(num_heads=2, num_parents=4, children_per_parent=2, head_dim=8)
        cb.initialize_parents_random()
        state = cb.state_dict()
        cb2 = HierarchicalCodebook(num_heads=2, num_parents=4, children_per_parent=2, head_dim=8)
        cb2.load_state_dict(state)
        assert torch.equal(cb.parents, cb2.parents)
        assert torch.equal(cb.children, cb2.children)

    def test_load_validates_mean_constraint(self) -> None:
        """load_state_dict enforces the constraint on restored tensors."""
        cb = HierarchicalCodebook(num_heads=1, num_parents=2, children_per_parent=2, head_dim=4)
        cb.initialize_parents_random()
        state = cb.state_dict()
        # Corrupt the state
        state["children"] = state["children"] + 1.0
        with pytest.raises(CodebookError, match="mean constraint"):
            cb.load_state_dict(state)

    def test_load_rejects_missing_keys(self) -> None:
        """load_state_dict requires both parents and children."""
        cb = HierarchicalCodebook()
        with pytest.raises(CodebookError, match="parents"):
            cb.load_state_dict({"children": torch.zeros(1, 64, 4, 64)})


class TestCodebookStats:
    """Tests for the CodebookStats dataclass (spec §3.8, §8.13)."""

    def test_utilization_in_range(self) -> None:
        """Utilization is in [0, 1]."""
        parent_counts = torch.tensor([[0.0, 5.0, 0.0, 3.0]])
        child_counts = torch.zeros(1, 4, 2)
        stats = CodebookStats(parent_counts=parent_counts, child_counts=child_counts)
        assert 0.0 <= stats.parent_utilization.item() <= 1.0
        assert torch.all(stats.child_utilization >= 0.0)
        assert torch.all(stats.child_utilization <= 1.0)

    def test_dead_count(self) -> None:
        """Dead counts reflect zero-count codewords."""
        parent_counts = torch.tensor([[0.0, 0.0, 5.0]])
        child_counts = torch.zeros(1, 3, 2)
        stats = CodebookStats(parent_counts=parent_counts, child_counts=child_counts)
        assert stats.dead_parent_count.item() == 2


class TestCodebookRepr:
    """Tests for codebook repr."""

    def test_repr_includes_key_fields(self) -> None:
        """repr shows heads/parents/children/dim."""
        cb = HierarchicalCodebook(num_heads=4, num_parents=16, children_per_parent=8, head_dim=32)
        r = repr(cb)
        assert "H=4" in r
        assert "M0=16" in r
        assert "C=8" in r
        assert "D=32" in r
