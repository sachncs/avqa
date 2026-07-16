"""Adaptive refinement orchestrator (spec §9.3, §9.6 to 9.8).

The refinement stage is the conductor: it queries the router for the
top-P parents, gathers their children, and hands them to the merge
+ correction operators.

ponytail: collapsed the planned refinement package (5 sub-modules) into
one src/avqa/refinement.py. The orchestrator is a single function.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avqa.attention import OnlineSoftmaxState
from avqa.merge import MergeInputs, ProbabilityMerge
from avqa.routing import RoutingDecision


@dataclass
class RefinementResult:
    """Output of one refinement step (spec §9.7).

    Attributes:
        state: Updated :class:`OnlineSoftmaxState` after correction.
        selected_parents: Indices of refined parents per (B, H). Shape
            ``[B, H, P]``.
        merge_value: Per-query refined value contribution. Shape
            ``[B, H, T, D_v]``.
    """

    state: OnlineSoftmaxState
    selected_parents: torch.Tensor
    merge_value: torch.Tensor


def _vectorized_correction(
    state: OnlineSoftmaxState,
    parent_logit: torch.Tensor,
    child_logits: torch.Tensor,
    parent_value: torch.Tensor,
    child_value: torch.Tensor,
    num_children: int,
) -> OnlineSoftmaxState:
    """Apply correcting attention vectorized over all (B, H, P) groups.

    Args:
        state: Running state.
        parent_logit: ``[B, H, T, P]`` parent logits.
        child_logits: ``[B, H, T, P, C]`` child logits (real Q · C_c^T).
        parent_value: ``[B, H, T, P, D_v]`` parent weighted values.
        child_value: ``[B, H, P, C, D_v]`` child aggregates (per-parent, not per-query).
        num_children: C.

    Returns:
        Updated state.
    """
    B, H, T, P = parent_logit.shape
    C = num_children
    D_v = child_value.shape[-1]

    # Recovered parent logits from children (spec §7.12).
    recovered_parent = child_logits.sum(dim=-1, keepdim=True) / C      # [B, H, T, P, 1]
    delta_logits = child_logits - recovered_parent                      # [B, H, T, P, C]

    # Parent contribution to remove: exp(0) * v_p = v_p.
    # child_value is [B, H, P, C, D_v]; expand to [B, H, T, P, C, D_v].
    cv = child_value.unsqueeze(2).expand(B, H, T, P, C, D_v)
    # Parent value to remove: [B, H, T, P, 1, D_v].
    pv = parent_value.unsqueeze(-2)                                     # [B, H, T, P, 1, D_v]

    # Tile max/denom/numerator for the delta.
    delta_max = delta_logits.amax(dim=-1, keepdim=True)                # [B, H, T, P, 1]
    delta_exp = torch.exp(delta_logits - delta_max)                    # [B, H, T, P, C]
    delta_denom = delta_exp.sum(dim=-1, keepdim=True)                  # [B, H, T, P, 1]

    # Numerator: sum_c exp(delta_logit - delta_max) * v_c - v_p.
    # Sum over C (dim=-2 of the 6-dim product), then unsqueeze to match pv.
    child_weighted = (delta_exp.unsqueeze(-1) * cv).sum(dim=-2)         # [B, H, T, P, D_v]
    delta_num = child_weighted.unsqueeze(-2) - pv                       # [B, H, T, P, 1, D_v]

    # Merge into state: reshape to [B*H*T, P, ...] and iterate over P.
    # ponytail: P is small (budget, typically 4-16), so iterating over
    # P is acceptable. The expensive B*H*T dimension is vectorized.
    delta_max_flat = delta_max.reshape(B * H * T, P, 1)
    delta_denom_flat = delta_denom.reshape(B * H * T, P, 1)
    delta_num_flat = delta_num.reshape(B * H * T, P, 1, D_v)

    new_state = state
    for p_idx in range(P):
        tile_max = delta_max_flat[:, p_idx, :]                         # [B*H*T, 1]
        tile_denom = delta_denom_flat[:, p_idx, :]                     # [B*H*T, 1]
        tile_num = delta_num_flat[:, p_idx, :, :]                      # [B*H*T, 1, D_v]
        # Reshape back to [B, H, T, ...] for state.merge.
        tile_max = tile_max.reshape(B, H, T, 1)
        tile_denom = tile_denom.reshape(B, H, T, 1)
        tile_num = tile_num.reshape(B, H, T, 1, D_v)
        new_state = new_state.merge(tile_max, tile_denom, tile_num)

    return new_state


def refine(
    state: OnlineSoftmaxState,
    parent_probs: torch.Tensor,
    parent_value: torch.Tensor,
    parent_aggregates: torch.Tensor,  # noqa: ARG001  (documented contract)
    child_aggregates: torch.Tensor,
    children_per_parent: int,
    decision: RoutingDecision,
    attention_probs: torch.Tensor,  # noqa: ARG001  (documented contract)
    parent_counts: torch.Tensor,  # noqa: ARG001  (documented contract)
    child_logits: torch.Tensor | None = None,
) -> RefinementResult:
    """Run one refinement step (spec §9.7, §9.11).

    Pipeline:

    1. Gather child aggregates for selected parents.
    2. Apply the correction operator to the running state.
    3. Apply a merge strategy to produce the refined value.

    Args:
        state: Running :class:`OnlineSoftmaxState` from the parent pass.
        parent_probs: ``[B, H, T, M_0]`` parent attention probabilities.
        parent_value: ``[B, H, T, M_0, D_v]`` parent-weighted value.
        parent_aggregates: ``[B, H, M_0, D_v]`` parent value aggregates.
        child_aggregates: ``[B, H, M_0, C, D_v]`` child value aggregates.
        children_per_parent: Number of children per parent (C).
        decision: Pre-computed :class:`RoutingDecision` from the orchestrator.
        attention_probs: ``[B, H, T, M_0]`` attention probabilities.
        parent_counts: ``[B, H, M_0]`` per-parent assignment counts.
        child_logits: Optional ``[B, H, T, M_0, C]`` real child attention
            logits (Q · C_c^T / sqrt(D)). When ``None``, falls back to the
            approximation child_logits = parent_logit / C (spec §7.12).

    Returns:
        :class:`RefinementResult` with the updated state, selected parent
        indices, and the merge value tensor.
    """
    selected = decision.selected_indices                              # [B, H, P]
    budget = selected.shape[-1]
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    if budget > parent_probs.shape[-1]:
        raise ValueError(
            f"budget ({budget}) exceeds number of parents ({parent_probs.shape[-1]})",
        )

    B, H, T, _, D_v = parent_value.shape
    C = children_per_parent
    selected = decision.selected_indices                              # [B, H, P]
    P = selected.shape[-1]

    # Gather child aggregates for selected parents.
    parent_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D_v)
    children = torch.gather(child_aggregates, 2, parent_idx)          # [B, H, P, C, D_v]

    # Gather parent logits and value for selected parents.
    parent_logit_gathered = parent_probs.gather(-1, selected.unsqueeze(-2).expand(B, H, T, P))
    parent_value_gathered = parent_value.gather(
        -2,
        selected.unsqueeze(-2).unsqueeze(-1).expand(B, H, T, P, D_v),
    )

    # Child logits: use real Q · C_c^T when provided, else approximate.
    if child_logits is not None:
        # Gather along dim=-2 (M0 parent dimension) to get [B, H, T, P, C].
        child_idx = selected.unsqueeze(2).unsqueeze(-1).expand(B, H, T, P, C)
        child_logits_gathered = child_logits.gather(-2, child_idx)
    else:
        # Approximation: child_logits = parent_logit / C (spec §7.12).
        child_logits_gathered = parent_logit_gathered.unsqueeze(-1).expand(B, H, T, P, C) / C

    # Correction: replace each selected parent's contribution with children.
    new_state = _vectorized_correction(
        state,
        parent_logit_gathered,
        child_logits_gathered,
        parent_value_gathered,
        children,
        num_children=C,
    )

    # Merge strategy: combine parent and child aggregates.
    parent_probs_for_merge = parent_logit_gathered.unsqueeze(-1)        # [B, H, T, P, 1]
    parent_value_for_merge = parent_value_gathered                     # [B, H, T, P, D_v]
    # Child probs from child logits via softmax.
    child_probs_for_merge = child_logits_gathered.softmax(dim=-1)      # [B, H, T, P, C]
    child_value_for_merge = children.unsqueeze(2).expand(B, H, T, P, C, D_v)
    merge_value = ProbabilityMerge().merge(
        MergeInputs(
            parent_probs=parent_probs_for_merge,
            parent_value=parent_value_for_merge,
            child_probs=child_probs_for_merge,
            child_value=child_value_for_merge,
        )
    )
    merge_value = merge_value.sum(dim=-2)                              # [B, H, T, D_v]

    return RefinementResult(
        state=new_state,
        selected_parents=selected,
        merge_value=merge_value,
    )


__all__ = ["RefinementResult", "refine"]
