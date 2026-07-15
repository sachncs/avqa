"""Vector quantization engine for AVQA (spec §7.5, §8.3 to 8.7).

Implements the two-stage hierarchical vector quantization required by
AVQ-Attention:

1. Each key is assigned to its nearest parent codeword (Euclidean).
2. Within that parent, the key is assigned to its nearest child.
3. During the same pass, values are aggregated per codeword and
   assignment counts are accumulated.

The fused preprocessing outputs are the direct inputs to the attention
kernel (spec §8.6.2): parent aggregates, child aggregates, parent
counts, child counts, parent assignments, child assignments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from avqa.exceptions import ShapeError
from avqa.logging import get_logger
from avqa.registry import QUANTIZER_REGISTRY

if TYPE_CHECKING:
    from avqa.codebook import HierarchicalCodebook

_logger = get_logger("quantizer")


@dataclass
class QuantizationResult:
    """Result of a vector-quantization precompute pass.

    Attributes:
        parent_assignments: Per-key parent index. Shape ``[B, H, N]``.
        child_assignments: Per-key child index within parent. Same shape.
        parent_aggregates: Sum of values per parent. Shape ``[B, H, M_0, D]``.
        child_aggregates: Sum of values per child. Shape ``[B, H, M_0, C, D]``.
        parent_counts: Number of keys assigned per parent. Shape ``[B, H, M_0]``.
        child_counts: Number of keys assigned per child. Shape ``[B, H, M_0, C]``.
    """

    parent_assignments: torch.Tensor
    child_assignments: torch.Tensor
    parent_aggregates: torch.Tensor
    child_aggregates: torch.Tensor
    parent_counts: torch.Tensor
    child_counts: torch.Tensor

    def validate_shapes(
        self,
        num_heads: int,
        num_parents: int,
        children_per_parent: int,
        head_dim: int,
    ) -> None:
        """Validate every tensor matches the documented contract."""
        B, H, N = self.parent_assignments.shape
        if self.child_assignments.shape != (B, H, N):
            raise ShapeError(
                "child_assignments shape mismatch",
                expected=(B, H, N),
                actual=tuple(self.child_assignments.shape),
            )
        if self.parent_aggregates.shape != (B, H, num_parents, head_dim):
            raise ShapeError(
                "parent_aggregates shape mismatch",
                expected=(B, H, num_parents, head_dim),
                actual=tuple(self.parent_aggregates.shape),
            )
        if self.child_aggregates.shape != (
            B,
            H,
            num_parents,
            children_per_parent,
            head_dim,
        ):
            raise ShapeError(
                "child_aggregates shape mismatch",
                expected=(B, H, num_parents, children_per_parent, head_dim),
                actual=tuple(self.child_aggregates.shape),
            )
        if self.parent_counts.shape != (B, H, num_parents):
            raise ShapeError(
                "parent_counts shape mismatch",
                expected=(B, H, num_parents),
                actual=tuple(self.parent_counts.shape),
            )
        if self.child_counts.shape != (B, H, num_parents, children_per_parent):
            raise ShapeError(
                "child_counts shape mismatch",
                expected=(B, H, num_parents, children_per_parent),
                actual=tuple(self.child_counts.shape),
            )


class VectorQuantizer(ABC):
    """Abstract base for AVQA vector quantizers (spec §4.7)."""

    @abstractmethod
    def precompute(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        codebook: HierarchicalCodebook,
    ) -> QuantizationResult:
        """Assign keys to codewords and aggregate values in one pass (spec §8.7).

        Args:
            keys: ``[B, H, N, D]``.
            values: ``[B, H, N, D]``.
            codebook: Hierarchical codebook (parents/children).

        Returns:
            :class:`QuantizationResult` populated with all six outputs.
        """


class EuclideanHierarchicalQuantizer(VectorQuantizer):
    """Hierarchical Euclidean VQ with fused value aggregation (spec §7.5, §8.5).

    Per-key cost is O(M_0 + C) distance evaluations instead of O(M_0 * C),
    achieved by the two-stage assignment described in §8.5.

    Args:
        deterministic: When ``True``, ties in distance are broken by
            selecting the smaller index. Default: ``False`` (relies on
            torch.argmin which returns the first occurrence, equivalent
            to deterministic behaviour for ties).

    Example:
        >>> cb = HierarchicalCodebook(num_heads=2, num_parents=4, children_per_parent=2, head_dim=8)
        >>> torch.manual_seed(0)
        >>> cb.initialize_parents_random()
        >>> q = EuclideanHierarchicalQuantizer()
        >>> keys = torch.randn(1, 2, 16, 8)
        >>> values = torch.randn(1, 2, 16, 8)
        >>> result = q.precompute(keys, values, cb)
        >>> result.parent_assignments.shape
        torch.Size([1, 2, 16])
    """

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic

    def precompute(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
codebook: HierarchicalCodebook,
    ) -> QuantizationResult:
        """Fused two-stage VQ with aggregation (spec §8.5 to 8.7)."""
        if keys.shape != values.shape:
            raise ShapeError(
                "keys and values must have identical shapes",
                expected=tuple(keys.shape),
                actual=tuple(values.shape),
            )
        if keys.ndim != 4:
            raise ShapeError(
                "keys must have rank 4 [B, H, N, D]",
                expected="[B, H, N, D]",
                actual=tuple(keys.shape),
            )

        B, H, N, D = keys.shape
        M0 = codebook.num_parents
        C = codebook.children_per_parent
        if codebook.head_dim != D:
            raise ShapeError(
                "key head_dim mismatch",
                expected=codebook.head_dim,
                actual=D,
            )

        # Stage 1: parent assignment via pairwise Euclidean distance
        # ``[B, H, N, M_0]`` where M_0 corresponds to parents[H, M_0, D].
        keys_flat = keys.reshape(B * H, N, D)
        parents_flat = codebook.parents.reshape(H, M0, D)
        # Compute squared distances via ||k||^2 - 2 k.p^T + ||p||^2.
        # argmin of squared == argmin of L2, so we skip the sqrt.
        k_sq = (keys_flat * keys_flat).sum(dim=-1, keepdim=True)            # [B*H, N, 1]
        p_sq = (parents_flat * parents_flat).sum(dim=-1).unsqueeze(1)        # [H, 1, M_0]
        cross = torch.einsum("bnd,hmd->bnm", keys_flat, parents_flat)        # [B*H, N, M_0]
        dist_sq = k_sq + p_sq - 2.0 * cross                                  # [B*H, N, M_0]
        parent_assign = dist_sq.argmin(dim=-1)                               # [B*H, N]
        parent_assign = parent_assign.reshape(B, H, N)

        # Stage 2: child assignment restricted to each key's parent.
        # For each key, gather that key's parent's children: [B*H, N, C, D].
        children = codebook.children                                          # [H, M_0, C, D]
        child_assign = torch.empty(B, H, N, dtype=parent_assign.dtype, device=keys.device)
        # Expand children to per-batch: [B*H, M_0, C, D]
        children_flat = children.unsqueeze(0).expand(B, H, M0, C, D).reshape(B * H, M0, C, D)
        # Index per-key: parent_assign.reshape(B*H, N) -> gather over M_0.
        idx = parent_assign.reshape(B * H, N).unsqueeze(-1).unsqueeze(-1).expand(B * H, N, C, D)
        gathered = torch.gather(children_flat, 1, idx)                        # [B*H, N, C, D]
        k_sq_c = (keys_flat * keys_flat).sum(dim=-1, keepdim=True)            # [B*H, N, 1]
        c_sq = (gathered * gathered).sum(dim=-1)                             # [B*H, N, C]
        cross_c = (keys_flat.unsqueeze(-2) * gathered).sum(dim=-1)            # [B*H, N, C]
        dist_sq_c = k_sq_c + c_sq - 2.0 * cross_c
        child_assign = dist_sq_c.argmin(dim=-1).reshape(B, H, N)

        # Aggregate values and counts in a fused pass.
        parent_aggregates = torch.zeros(B, H, M0, D, device=values.device, dtype=values.dtype)
        child_aggregates = torch.zeros(B, H, M0, C, D, device=values.device, dtype=values.dtype)
        parent_counts = torch.zeros(B, H, M0, device=values.device, dtype=values.dtype)
        child_counts = torch.zeros(B, H, M0, C, device=values.device, dtype=values.dtype)

        # Use index_add_ which fuses the scatter-add.
        parent_assign_flat = parent_assign.reshape(B * H, N)
        child_assign_flat = child_assign.reshape(B * H, N)
        values_flat = values.reshape(B * H, N, D)
        for bh in range(B * H):
            parent_aggregates.view(B * H, M0, D)[bh].index_add_(
                0, parent_assign_flat[bh], values_flat[bh]
            )
            parent_counts.view(B * H, M0)[bh].index_add_(
                0, parent_assign_flat[bh], torch.ones_like(parent_assign_flat[bh], dtype=values.dtype)
            )
            # Combine parent+child into a flat M_0*C index for child scatter.
            flat_child_idx = parent_assign_flat[bh] * C + child_assign_flat[bh]
            child_aggregates.view(B * H, M0 * C, D)[bh].index_add_(
                0, flat_child_idx, values_flat[bh]
            )
            child_counts.view(B * H, M0 * C)[bh].index_add_(
                0,
                flat_child_idx,
                torch.ones_like(flat_child_idx, dtype=values.dtype),
            )

        return QuantizationResult(
            parent_assignments=parent_assign,
            child_assignments=child_assign,
            parent_aggregates=parent_aggregates,
            child_aggregates=child_aggregates,
            parent_counts=parent_counts,
            child_counts=child_counts,
        )


# Register the default quantizer in the spec-mandated registry.
QUANTIZER_REGISTRY.register("euclidean_hierarchical")(EuclideanHierarchicalQuantizer)  # type: ignore[arg-type]


__all__ = [
    "EuclideanHierarchicalQuantizer",
    "QuantizationResult",
    "VectorQuantizer",
]
