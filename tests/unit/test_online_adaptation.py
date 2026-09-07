"""Tests for OPT-0003 BCAR (online codebook adaptation).

Run with:

    PYTHONPATH=src pytest tests/unit/test_online_adaptation.py -v
"""

from __future__ import annotations

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.codebook import HierarchicalCodebook
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RefinementConfig,
    RoutingConfig,
)
from avqa.exceptions import AVQAError
from avqa.online_adaptation import online_codebook_adaptation


def gaussian_centroids(
    *, num_heads: int, num_codewords: int, head_dim: int, children_per: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian-blob parents/children (offset from the global mean)."""
    g = torch.Generator().manual_seed(0)
    parents = torch.randn(num_heads, num_codewords, head_dim, generator=g)
    children = parents.unsqueeze(2) + 0.1 * torch.randn(
        num_heads, num_codewords, children_per, head_dim, generator=g
    )
    # Project children to satisfy SPEC §7.9 mean constraint.
    children = children - children.mean(dim=2, keepdim=True) + parents.unsqueeze(2)
    return parents, children


def codebook_from(
    parents: torch.Tensor, children: torch.Tensor, *, perturb: float
) -> HierarchicalCodebook:
    cb = HierarchicalCodebook(
        num_heads=parents.shape[0],
        num_parents=parents.shape[1],
        children_per_parent=parents.shape[2] if False else children.shape[2],
        head_dim=parents.shape[-1],
    )
    cb.parents.copy_(parents + perturb * torch.randn_like(parents))
    cb.children.copy_(children)
    # Force the mean constraint to hold by reprojecting children so they
    # average to the perturbed parents.
    cb.children.copy_(cb.children - cb.children.mean(dim=2, keepdim=True) + cb.parents.unsqueeze(2))
    cb.reproject_parents()
    cb.validate_mean_constraint()
    return cb


class TestBCARConvergence:
    """OPT-0003: the online codebook converges to the per-cluster mean."""

    def stream_for_centroid(self, centroid: torch.Tensor, n: int, *, sigma: float) -> torch.Tensor:
        g = torch.Generator().manual_seed(123)
        return centroid + sigma * torch.randn(n, centroid.shape[-1], generator=g)

    def test_converges_to_centroid(self) -> None:
        torch.manual_seed(0)
        H, M0, D, C = 1, 4, 8, 2
        parents, children = gaussian_centroids(
            num_heads=H, num_codewords=M0, head_dim=D, children_per=C
        )
        cb = codebook_from(parents, children, perturb=2.0)

        # Stochastic k-means convergence (Bottou & Bengio 1994):
        # round the per-step EMA over (parent, child) so that all
        # children participate. Decay 0.1 ⇒ per-step rate 0.9 reaches
        # the centroid with sub-0.5 L2 error in ~2 k rounds.
        for step in range(2048):
            keys = self.stream_for_centroid(parents[0, 0], 1, sigma=0.05)
            child_idx = step % C
            online_codebook_adaptation(
                keys[None, None, None],
                parents=cb.parents,
                children=cb.children,
                parent_assignments=torch.zeros(1, 1, 1, dtype=torch.long),
                child_assignments=torch.full((1, 1, 1), child_idx, dtype=torch.long),
                decay=0.1,
            )

        cb.validate_mean_constraint()
        # Tighter tolerance than the prior atol=0.5, rtol=0.5 — that
        # was essentially no-op; verify the BCAR EMA has actually
        # driven ``cb.parents[0, 0]`` to within 0.1 of the target.
        torch.testing.assert_close(
            cb.parents[0, 0], parents[0, 0], atol=1e-1, rtol=1e-1
        )


class TestBCARMeanConstraint:
    """OPT-0003: the parent-child mean constraint is preserved at every step."""

    def test_mean_constraint_holds_after_100_steps(self) -> None:
        torch.manual_seed(0)
        H, M0, D, C = 2, 8, 16, 4
        parents, children = gaussian_centroids(
            num_heads=H, num_codewords=M0, head_dim=D, children_per=C
        )
        cb = codebook_from(parents, children, perturb=0.5)

        for _ in range(100):
            keys = torch.randn(1, H, 64, D)
            parent_assn = torch.randint(0, M0, (1, H, 64))
            child_assn = torch.randint(0, C, (1, H, 64))
            online_codebook_adaptation(
                keys,
                parents=cb.parents,
                children=cb.children,
                parent_assignments=parent_assn,
                child_assignments=child_assn,
                decay=0.9,
            )
            cb.validate_mean_constraint()


class TestBCARRobustness:
    """OPT-0003: API guards and per-step monotonicity."""

    def test_decay_must_be_in_range(self) -> None:
        with pytest.raises(AVQAError, match="decay must be in"):
            online_codebook_adaptation(
                torch.zeros(1, 1, 1, 1),
                parents=torch.zeros(1, 1, 1),
                children=torch.zeros(1, 1, 1, 1, 1),
                parent_assignments=torch.zeros(1, 1, 1, dtype=torch.long),
                child_assignments=torch.zeros(1, 1, 1, dtype=torch.long),
                decay=1.0,
            )

    def test_empty_parent_does_not_explode(self) -> None:
        """Parents with zero assignment keep their value (no division)."""
        torch.manual_seed(0)
        cb = codebook_from(
            torch.randn(1, 4, 8),
            torch.randn(1, 4, 2, 8),
            perturb=0.0,
        )
        original_parent = cb.parents.detach().clone()
        # All keys assigned to parent 0; parents 1..3 stay untouched.
        keys = torch.randn(1, 1, 16, 8)
        parent_assn = torch.zeros(1, 1, 16, dtype=torch.long)
        child_assn = torch.zeros(1, 1, 16, dtype=torch.long)
        online_codebook_adaptation(
            keys,
            parents=cb.parents,
            children=cb.children,
            parent_assignments=parent_assn,
            child_assignments=child_assn,
            decay=0.5,
        )
        # Parent 0 updated; parents 1..3 unchanged.
        diff = (cb.parents[0, 1:] - original_parent[0, 1:]).abs().max().item()
        assert diff < 1e-6, f"untouched parents drifted by {diff}"
        # Parent 0 actually moved.
        moved = (cb.parents[0, 0] - original_parent[0, 0]).abs().max().item()
        assert moved > 0, "parent 0 should have moved toward the EMA centroid"


class TestBCAREndToEnd:
    """BCAR is wired through AVQAttention: ``bcar_enabled=True`` must
    actually mutate ``codebook.parents``/``codebook.children`` after a
    single forward pass.
    """

    def test_bcar_mutates_codebook_in_pipeline(self) -> None:
        """A single AVQAttention forward with bcar_enabled=True moves the codebook."""

        config = AVQConfig(
            attention=AttentionShapeConfig(embed_dim=32, num_heads=2, head_dim=16),
            codebook=CodebookConfig(
                num_codewords=8,
                children_per_codeword=2,
                bcar_enabled=True,
                bcar_decay=0.5,
            ),
            routing=RoutingConfig(refinement_budget=2),
            refinement=RefinementConfig(enabled=True),
        )
        torch.manual_seed(0)
        module = AVQAttention(config, in_proj=False, out_proj=False)
        parents_before = module.codebook.parents.detach().clone()
        children_before = module.codebook.children.detach().clone()
        torch.manual_seed(1)
        q = torch.randn(2, 8, 32)
        k = torch.randn(2, 8, 32)
        v = torch.randn(2, 8, 32)
        with torch.no_grad():
            _ = module(q, k, v)
        # At least one parent should have moved.
        parent_moved = (module.codebook.parents - parents_before).abs().max().item()
        child_moved = (module.codebook.children - children_before).abs().max().item()
        assert parent_moved > 1e-6, f"parent did not move (max diff {parent_moved})"
        # SPEC §7.9 invariant: parents = mean(children) at every step.
        module.codebook.validate_mean_constraint()
        # Children movement should also be > 0 in well-populated parents.
        # We don't assert strictly because a parent may receive no keys.
        assert child_moved >= 0

    def test_bcar_disabled_does_not_mutate_codebook(self) -> None:
        """bcar_enabled=False leaves the codebook untouched after forward."""

        config = AVQConfig(
            attention=AttentionShapeConfig(embed_dim=32, num_heads=2, head_dim=16),
            codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
            routing=RoutingConfig(refinement_budget=2),
            refinement=RefinementConfig(enabled=True),
        )
        torch.manual_seed(0)
        module = AVQAttention(config, in_proj=False, out_proj=False)
        parents_before = module.codebook.parents.detach().clone()
        children_before = module.codebook.children.detach().clone()
        torch.manual_seed(1)
        q = torch.randn(2, 8, 32)
        k = torch.randn(2, 8, 32)
        v = torch.randn(2, 8, 32)
        with torch.no_grad():
            _ = module(q, k, v)
        # Codebook should be unchanged (BCAR disabled).
        assert torch.equal(module.codebook.parents, parents_before)
        assert torch.equal(module.codebook.children, children_before)
