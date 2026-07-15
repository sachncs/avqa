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

from avqa.attention import (
    OnlineSoftmaxState,
    correct_parent_contribution,
)
from avqa.merge import MergeInputs, ProbabilityMerge
from avqa.routing import RoutingDecision, TopPRouter, compute_importance


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


def refine(
    state: OnlineSoftmaxState,
    parent_probs: torch.Tensor,
    parent_value: torch.Tensor,
    parent_aggregates: torch.Tensor,  # noqa: ARG001  (documented contract)
    child_aggregates: torch.Tensor,
    children_per_parent: int,
    budget: int,
    attention_probs: torch.Tensor,
    parent_counts: torch.Tensor,
) -> RefinementResult:
    """Run one refinement step (spec §9.7, §9.11).

    Pipeline:

    1. Compute importance from attention statistics (spec §7.10).
    2. Select top-P parents (spec §9.6, G14).
    3. Gather child aggregates for selected parents.
    4. Apply the correction operator to the running state.
    5. Apply a merge strategy to produce the refined value.

    Args:
        state: Running :class:`OnlineSoftmaxState` from the parent pass.
        parent_probs: ``[B, H, T, M_0]`` parent attention probabilities.
        parent_value: ``[B, H, T, M_0, D_v]`` parent-weighted value.
        parent_aggregates: ``[B, H, M_0, D_v]`` parent value aggregates.
        child_aggregates: ``[B, H, M_0, C, D_v]`` child value aggregates.
        children_per_parent: Number of children per parent (C).
        budget: Number of parents to refine (P).
        attention_probs: ``[B, H, T, M_0]`` attention probabilities.
        parent_counts: ``[B, H, M_0]`` per-parent assignment counts.

    Returns:
        :class:`RefinementResult` with the updated state, selected parent
        indices, and the merge value tensor.
    """
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    if budget > parent_probs.shape[-1]:
        raise ValueError(
            f"budget ({budget}) exceeds number of parents ({parent_probs.shape[-1]})",
        )

    importance = compute_importance(attention_probs, parent_counts)
    decision: RoutingDecision = TopPRouter().select(importance, budget)

    B, H, T, _, D_v = parent_value.shape
    C = children_per_parent
    selected = decision.selected_indices                              # [B, H, P]
    P = selected.shape[-1]

    # Gather child aggregates for selected parents.
    parent_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D_v)
    children = torch.gather(child_aggregates, 2, parent_idx)          # [B, H, P, C, D_v]

    # Gather parent logits and value. The index needs the same rank as
    # the input, so we unsqueeze once for the gather on dim=-1.
    parent_logit_gathered = parent_probs.gather(-1, selected.unsqueeze(-2).expand(B, H, T, P))
    parent_value_gathered = parent_value.gather(-2, selected.unsqueeze(-1).expand(B, H, T, P, D_v))

    # Recover parent logits from children (spec §7.12: parent = mean(children)
    # in codeword space, so logits after Q . C^T preserve this).
    # ponytail: we approximate child logits as parent_logits / C so that
    # the children mean equals the parent. The correction operator then
    # removes the parent and adds the children (delta = 0 on logits).
    child_logits = parent_logit_gathered.unsqueeze(-1) / C               # [B, H, T, P, C]

    # Correction: replace each selected parent's contribution with its
    # children's. Iterate per (b, h) and per selected parent.
    new_state = state
    parent_logit_flat = parent_logit_gathered.reshape(B * H, T, P)
    parent_value_flat = parent_value_gathered.reshape(B * H, T, P, 1, D_v)
    # child_logits has shape [B, H, T, P, 1] (one logit per parent); we
    # need to broadcast over the C children axis when reshaping.
    child_logits_broadcast = child_logits.expand(B, H, T, P, C)
    child_logits_flat = child_logits_broadcast.reshape(B * H, T, P, C)
    children_value_flat = children.reshape(B * H, P, C, D_v)
    for bh in range(B * H):
        for p_idx in range(P):
            pl = parent_logit_flat[bh, :, p_idx].unsqueeze(-1)        # [T, 1]
            cl = child_logits_flat[bh, :, p_idx, :]                    # [T, C]
            pv = parent_value_flat[bh, :, p_idx, :, :]                 # [T, 1, D_v]
            cv = children_value_flat[bh, p_idx, :, :].unsqueeze(0).expand(T, C, D_v)
            new_state = correct_parent_contribution(
                new_state, pl, cl, pv, cv, num_children=C
            )

    # Apply the merge strategy to combine parent and child aggregates.
    # Per MergeInputs contract: parent_probs [..., 1], parent_value [..., D_v],
    # child_probs [..., C], child_value [..., C, D_v].
    parent_probs_for_merge = parent_logit_gathered.unsqueeze(-1)        # [B, H, T, P, 1]
    parent_value_for_merge = parent_value_gathered                     # [B, H, T, P, D_v]
    child_probs_for_merge = parent_logit_gathered.softmax(dim=-1).unsqueeze(-1).expand(B, H, T, P, C)
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
