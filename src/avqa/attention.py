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

    Steps:

    1. Reconstruct parent logits from children (avoiding matmul).
    2. Compute the delta between child and parent logit-normalized values.
    3. Update the running state by merging the delta tile.

    Args:
        state: Current :class:`OnlineSoftmaxState`.
        parent_logits: ``[B, H, T, 1]`` parent logit per query.
            (kept for documentation symmetry with the algorithm; the
            correction uses recovered parent logits instead.)
        child_logits: ``[B, H, T, C]`` child logits (same group as parent).
        parent_value: ``[B, H, T, 1, D_v]`` parent value contribution.
        child_value: ``[B, H, T, C, D_v]`` child value contributions.
        num_children: Number of children per parent (C).

    Returns:
        Updated :class:`OnlineSoftmaxState`.
    """
    del parent_logits  # correction uses recovered parent logits
    # Recovered parent logits (used to remove the parent's contribution).
    recovered_parent = recover_parent_logits(child_logits, num_children)
    delta_logits = child_logits - recovered_parent
    # Numerator deltas: each child contributes (exp(delta_logit) * v_c),
    # the parent removes (exp(0) * v_p) = v_p.
    parent_contribution = parent_value                                # [B, H, T, 1, D_v]
    child_contribution = child_value                                   # [B, H, T, C, D_v]
    # Tile max for the delta: max over the C children.
    delta_max = delta_logits.amax(dim=-1, keepdim=True)               # [B, H, T, 1]
    delta_denom = torch.exp(delta_logits - delta_max).sum(
        dim=-1, keepdim=True
    )                                                                  # [B, H, T, 1]
    # Numerator delta: sum_c exp(delta_logit - delta_max) * v_c - v_p.
    delta_num = (
        torch.exp(delta_logits.unsqueeze(-1) - delta_max.unsqueeze(-1))
        * child_contribution
    ).sum(dim=-2, keepdim=True) - parent_contribution                   # [B, H, T, 1, D_v]
    return state.merge(delta_max, delta_denom, delta_num)


__all__ = [
    "OnlineSoftmaxState",
    "correct_parent_contribution",
    "recover_parent_logits",
]
