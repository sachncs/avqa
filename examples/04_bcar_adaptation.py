"""BCAR online codebook adaptation walkthrough.

Usage::

    python examples/04_bcar_adaptation.py
"""

from __future__ import annotations

import torch

from avqa.codebook import HierarchicalCodebook
from avqa.online_adaptation import online_codebook_adaptation
from avqa.quantizer import EuclideanHierarchicalQuantizer


def main() -> None:
    """Run BCAR on a synthetic stream and check the mean constraint."""
    torch.manual_seed(0)
    H, M0, C, D = 2, 8, 4, 16
    cb = HierarchicalCodebook(
        num_heads=H,
        num_parents=M0,
        children_per_parent=C,
        head_dim=D,
    )
    cb.initialize_parents_random()
    quantizer = EuclideanHierarchicalQuantizer()

    B, N = 2, 32
    keys = torch.randn(B, H, N, D)
    values = torch.randn(B, H, N, D)
    result = quantizer.precompute(keys, values, cb)

    parent_before = cb.parents.clone()
    child_before = cb.children.clone()
    online_codebook_adaptation(
        keys,
        parents=cb.parents,
        children=cb.children,
        parent_assignments=result.parent_assignments,
        child_assignments=result.child_assignments,
        decay=0.9,
    )
    print(f"parents updated: {not torch.equal(cb.parents, parent_before)}")
    print(f"children updated: {not torch.equal(cb.children, child_before)}")
    diff = cb.parents - cb.children.mean(dim=2)
    print(f"mean-constraint residual (should be ~0): {diff.abs().max().item():.6f}")


if __name__ == "__main__":
    main()