"""Online softmax state and correction operators (spec §7.14, §7.13).

Spec §7.14 mandates FlashAttention-style online softmax with running
max, numerator, and denominator. Spec §7.13 mandates correcting
attention that replaces (not augments) parent contributions with child
contributions.

ponytail: inlines the running-state and correction logic in one module.
The :class:`OnlineSoftmaxState` is a tiny data class; the correction
operator is a single function.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avqa.utils.numerics import online_softmax_step as _online_softmax_step


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
        new_max, new_denom, new_num = _online_softmax_step(
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
    ) -> OnlineSoftmaxState:
        """Replace a tile: state_new = state - removed + added.

        Used by correcting attention (spec §7.13). All three tiles
        are broadcast-compatible with the state shape ``[B, H, T, D_k]``.
        Uses a numerically stable three-way max.

        Args:
            removed_max: ``[B, H, T, D_k]`` per-row max of the removed tile.
            removed_denominator: ``[B, H, T, D_k]`` per-row denominator to subtract.
            removed_numerator: ``[B, H, T, D_k, D_v]`` per-row numerator to subtract.
            added_max: ``[B, H, T, D_k]`` per-row max of the added tile.
            added_denominator: ``[B, H, T, D_k]`` per-row denominator to add.
            added_numerator: ``[B, H, T, D_k, D_v]`` per-row numerator to add.

        Returns:
            New :class:`OnlineSoftmaxState` with the replacement applied.
        """
        new_max = torch.maximum(torch.maximum(self.running_max, removed_max), added_max)
        scale_old = torch.exp(self.running_max - new_max)
        scale_removed = torch.exp(removed_max - new_max)
        scale_added = torch.exp(added_max - new_max)
        new_denom = (
            self.running_denominator * scale_old
            - removed_denominator * scale_removed
            + added_denominator * scale_added
        )
        # Numerator: [B, H, T, D_k, D_v]. scale broadcasts across D_v.
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

    Under the parent-child mean constraint, parent logits satisfy:

        S_p = (1 / C) * sum_c S_c

    This avoids recomputing parent-query matrix multiplications.

    Args:
        child_logits: Per-(B, H, T, C) child logits.
        num_children: Number of children per parent (C).

    Returns:
        Parent logits of shape ``[B, H, T, 1]`` (one parent per child group).
    """
    if num_children <= 0:
        msg = f"num_children must be > 0, got {num_children}"
        raise ValueError(msg)
    return child_logits.sum(dim=-1, keepdim=True) / num_children


def correct_parent_contribution(
    state: OnlineSoftmaxState,
    parent_logits: torch.Tensor,
    child_logits: torch.Tensor,
    parent_value: torch.Tensor,
    child_value: torch.Tensor,
    num_children: int,
) -> OnlineSoftmaxState:
    """Apply correcting attention to replace parent with children (spec §7.13).

    Uses :meth:`OnlineSoftmaxState.replace` to subtract the parent
    contribution and add the child contribution in a numerically stable
    way (common scale = max over all three tiles).

    Steps:

    1. Reconstruct parent logits from children (avoiding matmul).
    2. Compute the tile stats for the parent (to remove) and children (to add).
    3. Update the running state via ``state - parent + children``.

    Args:
        state: Current :class:`OnlineSoftmaxState`.
        parent_logits: ``[B, H, T, 1]`` parent logit per query
            (used for parent tile max; if ``None``, recovered from children).
        child_logits: ``[B, H, T, C]`` child logits (same group as parent).
        parent_value: ``[B, H, T, 1, D_v]`` parent raw value aggregate.
        child_value: ``[B, H, T, C, D_v]`` child raw value aggregates.
        num_children: Number of children per parent (C).

    Returns:
        Updated :class:`OnlineSoftmaxState`.
    """
    if parent_logits is None:
        parent_logits = recover_parent_logits(child_logits, num_children)
    # The state carries a D_k dimension. Use the running max as the common
    # scale; replace -inf with the new tile max to avoid overflow. Tile
    # outputs are broadcast to [B, H, T, D_k] so they multiply cleanly
    # with the state (the D_k dim is degenerate — same value everywhere).
    m_raw = state.running_max[..., 0:1]  # [B, H, T, 1]
    new_max_1d = torch.maximum(
        parent_logits.amax(dim=-1, keepdim=True), child_logits.amax(dim=-1, keepdim=True)
    )  # [B, H, T, 1]
    m_1d = torch.where(torch.isinf(m_raw) & (m_raw < 0), new_max_1d, m_raw)  # [B, H, T, 1]
    # Expand scalar m to full state tile: [B, H, T, 1] → broadcast against D_k.
    parent_exp_1d = torch.exp(parent_logits - m_1d)  # [B, H, T, 1]
    child_exp_1d = torch.exp(child_logits - m_1d)  # [B, H, T, C]
    parent_tile_denom_1d = parent_exp_1d.squeeze(-1)  # [B, H, T]
    parent_tile_num_5d = parent_exp_1d.unsqueeze(-1) * parent_value  # [B, H, T, 1, D_v]
    child_tile_denom_1d = child_exp_1d.sum(dim=-1)  # [B, H, T]
    child_tile_num_5d = (child_exp_1d.unsqueeze(-1) * child_value).sum(dim=-2)  # [B, H, T, D_v]
    # Add the D_k singleton dim so shapes match state.
    return state.replace(
        removed_max=parent_logits.squeeze(-1).unsqueeze(-1),  # [B, H, T, 1]
        removed_denominator=parent_tile_denom_1d.unsqueeze(-1),  # [B, H, T, 1]
        removed_numerator=parent_tile_num_5d,  # [B, H, T, 1, D_v]
        added_max=child_logits.amax(dim=-1).unsqueeze(-1),  # [B, H, T, 1]
        added_denominator=child_tile_denom_1d.unsqueeze(-1),  # [B, H, T, 1]
        added_numerator=child_tile_num_5d.unsqueeze(-2),  # [B, H, T, 1, D_v]
    )


__all__ = [
    "OnlineSoftmaxState",
    "correct_parent_contribution",
    "recover_parent_logits",
]
