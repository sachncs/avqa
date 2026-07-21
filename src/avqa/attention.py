"""Online softmax state and correction operators (spec §7.14, §7.13).

Spec §7.14 mandates FlashAttention-style online softmax with running
max, numerator, and denominator. Spec §7.13 mandates correcting
attention that replaces (not augments) parent contributions with child
contributions.

ponytail: inlines the running-state and correction logic in one module.
The :class:`OnlineSoftmaxState` is a tiny data class; the correction
operator is one function (:func:`recover_parent_logits`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from avqa.exceptions import RoutingError
from avqa.utils.numerics import online_softmax_step


@dataclass
class OnlineSoftmaxState:
    """Running online-softmax accumulators (spec §7.14).

    Attributes:
        running_max: Per-row running maximum ``[B, H, T, D_k]``.
        running_denominator: Per-row running denominator ``[B, H, T, D_k]``.
        running_numerator: Per-row running numerator ``[B, H, T, D_k, D_v]``.
    """

    running_max: torch.Tensor
    running_denominator: torch.Tensor
    running_numerator: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim_k: int,
        head_dim_v: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> OnlineSoftmaxState:
        """Allocate an empty state with running_max = -inf."""
        running_max = torch.full(
            (batch_size, num_heads, seq_len, head_dim_k),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        running_denominator = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim_k),
            device=device,
            dtype=dtype,
        )
        running_numerator = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim_k, head_dim_v),
            device=device,
            dtype=dtype,
        )
        return cls(
            running_max=running_max,
            running_denominator=running_denominator,
            running_numerator=running_numerator,
        )

    def merge(
        self,
        tile_max: torch.Tensor,
        tile_denominator: torch.Tensor,
        tile_numerator: torch.Tensor,
    ) -> OnlineSoftmaxState:
        """Merge a new tile into the running state (FlashAttention §3.2).

        Args:
            tile_max: ``[B, H, T, D_k]`` per-row max of the new tile.
            tile_denominator: ``[B, H, T, D_k]`` per-row denominator.
            tile_numerator: ``[B, H, T, D_k, D_v]`` per-row numerator.

        Returns:
            New :class:`OnlineSoftmaxState` covering both old and tile.
        """
        new_max, new_denom, new_num = online_softmax_step(
            self.running_max,
            self.running_denominator,
            self.running_numerator,
            tile_max,
            tile_denominator,
            tile_numerator,
        )
        return OnlineSoftmaxState(
            running_max=new_max,
            running_denominator=new_denom,
            running_numerator=new_num,
        )

    def replace(
        self,
        removed_max: torch.Tensor,
        removed_denominator: torch.Tensor,
        removed_numerator: torch.Tensor,
        added_max: torch.Tensor,
        added_denominator: torch.Tensor,
        added_numerator: torch.Tensor,
        m_anchor: torch.Tensor | None = None,
    ) -> OnlineSoftmaxState:
        """Replace a tile: ``state_new = state - removed + added``.

        Used by correcting attention (spec §7.13). All three tiles
        are broadcast-compatible with the state shape ``[B, H, T, D_k]``.

        When the caller has already chosen a common scale for the
        removed/added tiles (their ``removed_denominator`` /
        ``added_denominator`` were computed against ``m_anchor``), pass
        that scale as ``m_anchor`` so the rescales inside ``replace``
        line up. Without ``m_anchor`` the function falls back to
        ``max(state.running_max, removed_max, added_max)`` — backwards
        compatible with single-shot callers (e.g. one-tile corrections
        that don't pre-compute a common scale).

        Args:
            removed_max: ``[B, H, T, D_k]`` per-row max of the removed tile.
            removed_denominator: ``[B, H, T, D_k]`` per-row denominator to subtract.
            removed_numerator: ``[B, H, T, D_k, D_v]`` per-row numerator to subtract.
            added_max: ``[B, H, T, D_k]`` per-row max of the added tile.
            added_denominator: ``[B, H, T, D_k]`` per-row denominator to add.
            added_numerator: ``[B, H, T, D_k, D_v]`` per-row numerator to add.
            m_anchor: Optional ``[B, H, T, D_k]`` common scale already used
                to compute ``removed_denominator`` / ``added_denominator``.
                ``new_max`` then becomes ``max(m_anchor, state.running_max)``.

        Returns:
            New :class:`OnlineSoftmaxState` with the replacement applied.
        """
        if m_anchor is None:
            new_max = torch.maximum(
                torch.maximum(self.running_max, removed_max), added_max
            )
            scale_removed = torch.exp(removed_max - new_max)
            scale_added = torch.exp(added_max - new_max)
        else:
            new_max = torch.maximum(m_anchor, self.running_max)
            scale_removed = torch.exp(m_anchor - new_max)
            scale_added = torch.exp(m_anchor - new_max)
        scale_old = torch.exp(self.running_max - new_max)
        new_denom = (
            self.running_denominator * scale_old
            - removed_denominator * scale_removed
            + added_denominator * scale_added
        )
        new_num = (
            self.running_numerator * scale_old.unsqueeze(-1)
            - removed_numerator * scale_removed.unsqueeze(-1)
            + added_numerator * scale_added.unsqueeze(-1)
        )
        return OnlineSoftmaxState(
            running_max=new_max,
            running_denominator=new_denom,
            running_numerator=new_num,
        )


def recover_parent_logits(
    child_logits: torch.Tensor,
    num_children: int,
) -> torch.Tensor:
    """Reconstruct parent logits from children (spec §7.12).

    Under the parent-child mean constraint, parent logits satisfy::

        S_p = (1 / C) * sum_c S_c

    This avoids recomputing parent-query matrix multiplications.

    Args:
        child_logits: Per-(B, H, T, C) child logits.
        num_children: Number of children per parent (C).

    Returns:
        Parent logits of shape ``[B, H, T, 1]`` (one parent per child group).

    Raises:
        RoutingError: If ``num_children`` is not positive.
    """
    if num_children <= 0:
        raise RoutingError(f"num_children must be > 0, got {num_children}")
    return child_logits.sum(dim=-1, keepdim=True) / num_children


__all__ = [
    "OnlineSoftmaxState",
    "recover_parent_logits",
]
