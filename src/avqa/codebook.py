"""Hierarchical codebook for AVQA (spec §7.8, §7.9, §8.3).

Implements the two-level hierarchical codebook required by AVQ-Attention:

- ``parents`` of shape ``[H, M_0, D]`` — parent codewords (one per head).
- ``children`` of shape ``[H, M_0, C, D]`` — child codewords per parent.

The parent-child mean constraint (§7.9) requires:

    parent_p == mean(children_p, dim=0)

This is enforced after every training update via
:meth:`HierarchicalCodebook.reproject_parents`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from avqa.exceptions import CodebookError


def require_positive(value: int, field_name: str) -> None:
    """Raise ``CodebookError`` if ``value`` is not positive."""
    if value <= 0:
        msg = f"{field_name} must be > 0, got {value}"
        raise CodebookError(msg, {field_name: value})


@dataclass
class CodebookStats:
    """Per-codeword assignment counts and utilization (spec §3.8, §8.13).

    Attributes:
        parent_counts: Per-parent assignment count. Shape ``[H, M_0]``.
        child_counts: Per-child assignment count. Shape ``[H, M_0, C]``.
        utilization: Fraction of codewords that received at least one
            assignment. Shape ``[H]`` (parents) and ``[H, M_0]`` (children).
    """

    parent_counts: torch.Tensor
    child_counts: torch.Tensor

    @property
    def parent_utilization(self) -> torch.Tensor:
        """Fraction of parents with count > 0."""
        return (self.parent_counts > 0).float().mean(dim=-1)

    @property
    def child_utilization(self) -> torch.Tensor:
        """Fraction of children with count > 0 per parent."""
        return (self.child_counts > 0).float().mean(dim=-1)

    @property
    def dead_parent_count(self) -> torch.Tensor:
        """Number of unused parents per head."""
        return (self.parent_counts == 0).sum(dim=-1)

    @property
    def dead_child_count(self) -> torch.Tensor:
        """Number of unused children per (head, parent)."""
        return (self.child_counts == 0).sum(dim=-1)


class HierarchicalCodebook:
    """Hierarchical parent-child codebook (spec §7.8, §8.3).

    Each head owns its own codebook. The codebook has ``num_parents`` parents
    and ``children_per_parent`` children per parent. The parent-child mean
    constraint is enforced via :meth:`reproject_parents`.

    Args:
        num_heads: Number of attention heads (H).
        num_parents: Number of parent codewords per head (M_0).
        children_per_parent: Children per parent (C).
        head_dim: Per-head dimension (D).
        perturbation_scale: Std of the Gaussian noise used to initialize
            children near their parent (spec §8.10).
        device: Device for the codebook tensors.
        dtype: Dtype for the codebook tensors.

    Raises:
        CodebookError: If any constructor argument is invalid.

    Example:
        >>> cb = HierarchicalCodebook(
        ...     num_heads=2, num_parents=8, children_per_parent=4, head_dim=16
        ... )
        >>> cb.parents.shape
        torch.Size([2, 8, 16])
        >>> cb.children.shape
        torch.Size([2, 8, 4, 16])
        >>> cb.validate_mean_constraint()
    """

    def __init__(
        self,
        num_heads: int = 1,
        num_parents: int = 64,
        children_per_parent: int = 4,
        head_dim: int = 64,
        perturbation_scale: float = 0.1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        require_positive(num_heads, "num_heads")
        require_positive(num_parents, "num_parents")
        require_positive(children_per_parent, "children_per_parent")
        require_positive(head_dim, "head_dim")
        if perturbation_scale <= 0:
            msg = f"perturbation_scale must be > 0, got {perturbation_scale}"
            raise CodebookError(msg)

        self.num_heads = num_heads
        self.num_parents = num_parents
        self.children_per_parent = children_per_parent
        self.head_dim = head_dim
        self.perturbation_scale = perturbation_scale

        # Parents start at zero; children are initialized to satisfy the
        # mean constraint by construction (parent = mean(child + noise)).
        self.parents = torch.zeros(num_heads, num_parents, head_dim, device=device, dtype=dtype)
        self.children = torch.zeros(
            num_heads,
            num_parents,
            children_per_parent,
            head_dim,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Initialization (spec §8.10)
    # ------------------------------------------------------------------

    def initialize_children_around_parents(
        self,
        perturbation: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize children near their parent (spec §8.10).

        The mean of (parent + perturbation) over children equals the parent
        exactly only when the perturbation has zero mean. We achieve this
        by sampling one perturbation per child and subtracting its mean
        across the children axis.

        Args:
            perturbation: Optional pre-sampled noise of shape
                ``[H, M_0, C, D]``. If ``None``, samples Gaussian noise.
            generator: Optional RNG generator for reproducibility.
        """
        if perturbation is None:
            perturbation = torch.randn(
                self.num_heads,
                self.num_parents,
                self.children_per_parent,
                self.head_dim,
                device=self.children.device,
                dtype=self.children.dtype,
                generator=generator,
            )
        else:
            expected_shape = (
                self.num_heads,
                self.num_parents,
                self.children_per_parent,
                self.head_dim,
            )
            if tuple(perturbation.shape) != expected_shape:
                msg = f"perturbation shape {tuple(perturbation.shape)} != expected {expected_shape}"
                raise CodebookError(msg)
        # Subtract mean to keep parent == mean(children) after construction.
        # ponytail: spec §8.10 says C_{p,c} = C_p - 0.1*epsilon; sign
        # doesn't matter since epsilon ~ N(0,I) is symmetric.
        perturbation = perturbation - perturbation.mean(dim=2, keepdim=True)
        self.children = self.parents.unsqueeze(2) + self.perturbation_scale * perturbation
        # Enforce the constraint exactly (float rounding can drift).
        self.reproject_parents()

    def initialize_parents_random(
        self,
        generator: torch.Generator | None = None,
        scale: float = 1.0,
    ) -> None:
        """Randomize parents from N(0, scale^2).

        Args:
            generator: Optional RNG.
            scale: Standard deviation of the initialization distribution.
        """
        self.parents = (
            torch.randn(
                self.num_heads,
                self.num_parents,
                self.head_dim,
                device=self.parents.device,
                dtype=self.parents.dtype,
                generator=generator,
            )
            * scale
        )
        self.children = torch.zeros_like(self.children)
        self.initialize_children_around_parents(generator=generator)

    # ------------------------------------------------------------------
    # Mean constraint (spec §7.9)
    # ------------------------------------------------------------------

    def reproject_parents(self) -> None:
        """Set each parent to the mean of its children (spec §7.9).

        Maintains the parent-child mean constraint after every operation
        that perturbs ``children``.
        """
        self.parents = self.children.mean(dim=2)

    def validate_mean_constraint(self, atol: float = 1e-5) -> None:
        """Raise :class:`CodebookError` if the mean constraint is violated.

        Args:
            atol: Absolute tolerance.

        Raises:
            CodebookError: If ``||parent - mean(children)|| > atol``.
        """
        diff = (self.parents - self.children.mean(dim=2)).abs().max().item()
        if diff > atol:
            msg = f"mean constraint violated: max |parent - mean(child)| = {diff}"
            raise CodebookError(msg, {"max_diff": diff})

    # ------------------------------------------------------------------
    # EMA training (spec §8.9)
    # ------------------------------------------------------------------

    def ema_update(
        self,
        new_parents: torch.Tensor,
        new_children: torch.Tensor,
        decay: float = 0.99,
    ) -> None:
        """Apply EMA update to codebook entries.

        Args:
            new_parents: New parent estimates ``[H, M_0, D]``.
            new_children: New child estimates ``[H, M_0, C, D]``.
            decay: EMA decay factor. Higher = slower update.

        Raises:
            CodebookError: On shape mismatch or invalid decay.
        """
        if not 0.0 <= decay <= 1.0:
            msg = f"EMA decay must be in [0, 1], got {decay}"
            raise CodebookError(msg)
        expected_parent_shape = (self.num_heads, self.num_parents, self.head_dim)
        expected_child_shape = (
            self.num_heads,
            self.num_parents,
            self.children_per_parent,
            self.head_dim,
        )
        if tuple(new_parents.shape) != expected_parent_shape:
            msg = f"new_parents shape {tuple(new_parents.shape)} != {expected_parent_shape}"
            raise CodebookError(msg)
        if tuple(new_children.shape) != expected_child_shape:
            msg = f"new_children shape {tuple(new_children.shape)} != {expected_child_shape}"
            raise CodebookError(msg)
        self.parents = decay * self.parents + (1.0 - decay) * new_parents
        self.children = decay * self.children + (1.0 - decay) * new_children
        self.reproject_parents()

    # ------------------------------------------------------------------
    # Commitment loss (spec §8.9)
    # ------------------------------------------------------------------

    def commitment_loss(
        self,
        keys: torch.Tensor,
        parent_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Compute commitment (encoding) loss (spec §8.9).

        For each key, measures the squared distance to its assigned
        parent codeword. This encourages keys to lie close to their
        assigned code, improving quantization quality during training.

        Args:
            keys: Encoder outputs ``[B, H, N, D]``.
            parent_assignments: Per-key parent index ``[B, H, N]``.

        Returns:
            Scalar mean commitment loss.
        """
        B, H, N, D = keys.shape
        # Gather assigned codewords: [B, H, N, D].
        assigned = self.parents.unsqueeze(0).expand(B, H, self.num_parents, D)
        idx = parent_assignments.unsqueeze(-1).expand(B, H, N, D)
        codewords = torch.gather(assigned, 2, idx)
        # Squared L2 distance.
        return ((keys - codewords) ** 2).mean()

    # ------------------------------------------------------------------
    # Serialization (spec §3.20)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable snapshot of the codebook."""
        return {
            "parents": self.parents.detach().clone(),
            "children": self.children.detach().clone(),
            "schema_version": torch.tensor(1, dtype=torch.int32),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the codebook from :meth:`state_dict` output."""
        if "parents" not in state or "children" not in state:
            msg = "state_dict must contain 'parents' and 'children'"
            raise CodebookError(msg)
        self.parents = state["parents"].to(self.parents.device, self.parents.dtype)
        self.children = state["children"].to(self.children.device, self.children.dtype)
        self.validate_mean_constraint()

    def __repr__(self) -> str:
        return (
            f"HierarchicalCodebook(H={self.num_heads}, M0={self.num_parents}, "
            f"C={self.children_per_parent}, D={self.head_dim}, "
            f"device={self.parents.device}, dtype={self.parents.dtype})"
        )


__all__ = ["CodebookStats", "HierarchicalCodebook", "require_positive"]
