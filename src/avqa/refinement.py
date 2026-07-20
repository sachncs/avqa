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


class AdaptiveRefinement:
    """Refinement orchestrator (spec §5.5, §9.3, §9.6 to §9.8).

    Wraps the :func:`refine` function in a stateful interface that
    caches the most recent :class:`RefinementResult`.

    Args:
        children_per_parent: Number of children per parent (C).
    """

    def __init__(self, children_per_parent: int = 4) -> None:
        self.children_per_parent = children_per_parent
        self.last_result: RefinementResult | None = None

    def refine(
        self,
        state: OnlineSoftmaxState,
        parent_probs: torch.Tensor,
        parent_value: torch.Tensor,
        parent_aggregates: torch.Tensor,
        child_aggregates: torch.Tensor,
        decision: RoutingDecision,
        attention_probs: torch.Tensor,
        parent_counts: torch.Tensor,
        child_logits: torch.Tensor | None = None,
    ) -> RefinementResult:
        """Run one refinement step (delegates to :func:`refine`)."""
        result = refine(
            state=state,
            parent_probs=parent_probs,
            parent_value=parent_value,
            parent_aggregates=parent_aggregates,
            child_aggregates=child_aggregates,
            children_per_parent=self.children_per_parent,
            decision=decision,
            attention_probs=attention_probs,
            parent_counts=parent_counts,
            child_logits=child_logits,
        )
        self.last_result = result
        return result


def vectorized_correction(
    state: OnlineSoftmaxState,
    parent_logit: torch.Tensor,
    child_logits: torch.Tensor,
    parent_value: torch.Tensor,
    child_value: torch.Tensor,
    num_children: int,
    parent_counts: torch.Tensor | None = None,
    child_counts: torch.Tensor | None = None,
) -> OnlineSoftmaxState:
    """Apply correcting attention vectorized over all (B, H, P) groups.

    Spec §7.13: replace parent's contribution with children's. Uses
    :meth:`OnlineSoftmaxState.replace` for numerically stable subtraction
    and addition at a common scale.

    Args:
        state: Running state.
        parent_logit: ``[B, H, T, P]`` parent logits.
        child_logits: ``[B, H, T, P, C]`` child logits (real Q · C_c^T).
        parent_value: ``[B, H, T, P, D_v]`` raw parent aggregates (V̄_p).
        child_value: ``[B, H, P, C, D_v]`` child aggregates (per-parent, not per-query).
        num_children: C.
        parent_counts: Optional ``[B, H, P]`` parent assignment counts. When
            provided, scales the parent contribution by n_p (spec §9.12: empty
            codewords contribute zero). Defaults to ones (every parent counts).
        child_counts: Optional ``[B, H, P, C]`` child assignment counts. When
            provided, scales each child contribution by n_c.

    Returns:
        Updated state.
    """
    B, H, T, P = parent_logit.shape
    C = num_children
    D_v = child_value.shape[-1]

    # Scale factors (spec §9.12): empty codewords contribute zero.
    if parent_counts is None:
        parent_scale = torch.ones(B, H, T, P, device=parent_logit.device, dtype=parent_logit.dtype)
    else:
        parent_scale = parent_counts.unsqueeze(2).expand(B, H, T, P).to(parent_logit.dtype)
    if child_counts is None:
        child_scale = torch.ones(B, H, P, C, device=parent_logit.device, dtype=parent_logit.dtype)
    else:
        child_scale = child_counts.to(parent_logit.dtype)

    # The state carries a D_k dimension. Use the running max as the common
    # scale; replace -inf with the new tile max to avoid overflow. Tile
    # outputs are kept at [B, H, T, 1] so they broadcast against D_k.
    m_raw = state.running_max[..., 0:1]  # [B, H, T, 1]
    new_max_1d = torch.maximum(
        parent_logit.amax(dim=-1, keepdim=True),
        child_logits.amax(dim=(-1, -2), keepdim=True).squeeze(-1),  # [B, H, T, 1]
    )
    m_scalar_safe = torch.where(torch.isinf(m_raw) & (m_raw < 0), new_max_1d, m_raw)
    parent_exp = torch.exp(parent_logit - m_scalar_safe) * parent_scale  # [B, H, T, P]
    parent_contrib_denom = parent_exp.sum(dim=-1, keepdim=True)  # [B, H, T, 1]
    parent_contrib_num = (parent_exp.unsqueeze(-1) * parent_value).sum(
        dim=-2, keepdim=True
    )  # [B, H, T, 1, D_v]

    child_exp = torch.exp(child_logits - m_scalar_safe.unsqueeze(-1)) * child_scale.unsqueeze(
        2
    )  # [B, H, T, P, C]
    child_contrib_denom = child_exp.sum(dim=(-1, -2), keepdim=True)  # [B, H, T, 1, 1]
    child_contrib_denom = child_contrib_denom.squeeze(-1)  # [B, H, T, 1]
    cv = child_value.unsqueeze(2).expand(B, H, T, P, C, D_v)
    child_contrib_num = (child_exp.unsqueeze(-1) * cv).sum(
        dim=(-2, -3), keepdim=True
    )  # [B, H, T, 1, 1, D_v]
    child_contrib_num = child_contrib_num.squeeze(-2)  # [B, H, T, 1, D_v]

    parent_tile_max = parent_logit.amax(dim=-1, keepdim=True)  # [B, H, T, 1]
    child_tile_max = child_logits.amax(dim=(-1, -2), keepdim=True)  # [B, H, T, 1, 1]
    child_tile_max = child_tile_max.squeeze(-1)  # [B, H, T, 1]

    return state.replace(
        removed_max=parent_tile_max,
        removed_denominator=parent_contrib_denom,
        removed_numerator=parent_contrib_num,
        added_max=child_tile_max,
        added_denominator=child_contrib_denom,
        added_numerator=child_contrib_num,
    )


def refine(
    state: OnlineSoftmaxState,
    parent_probs: torch.Tensor,
    parent_value: torch.Tensor,
    parent_aggregates: torch.Tensor,
    child_aggregates: torch.Tensor,
    children_per_parent: int,
    decision: RoutingDecision,
    attention_probs: torch.Tensor,  # noqa: ARG001  (documented contract)
    parent_counts: torch.Tensor,
    child_logits: torch.Tensor | None = None,
    child_counts: torch.Tensor | None = None,
    merge_strategy: str = "probability",
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
        child_counts: Optional ``[B, H, M_0, C]`` per-child assignment counts.
        merge_strategy: One of ``"probability"``, ``"weighted"``, ``"logit"``,
            ``"normalized"`` (spec §3.11). Defaults to ``"probability"``.

    Returns:
        :class:`RefinementResult` with the updated state, selected parent
        indices, and the merge value tensor.
    """
    selected = decision.selected_indices  # [B, H, P]
    budget = selected.shape[-1]
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    if budget > parent_probs.shape[-1]:
        raise ValueError(
            f"budget ({budget}) exceeds number of parents ({parent_probs.shape[-1]})",
        )

    B, H, T, _, D_v = parent_value.shape
    C = children_per_parent
    selected = decision.selected_indices  # [B, H, P]
    P = selected.shape[-1]

    # Gather child aggregates for selected parents.
    parent_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D_v)
    children = torch.gather(child_aggregates, 2, parent_idx)  # [B, H, P, C, D_v]

    # Gather parent logits and weighted values for selected parents.
    parent_logit_gathered = parent_probs.gather(-1, selected.unsqueeze(-2).expand(B, H, T, P))
    parent_value_gathered = parent_value.gather(
        -2,
        selected.unsqueeze(-2).unsqueeze(-1).expand(B, H, T, P, D_v),
    )

    # Gather RAW parent aggregates (V̄_p) for the correction step.
    # The correction removes V̄_p (not A_p·V̄_p) — see spec §7.13.
    parent_aggregates_gathered = parent_aggregates.gather(
        2,
        selected.unsqueeze(-1).expand(B, H, P, D_v),
    )  # [B, H, P, D_v]
    # Expand to [B, H, T, P, D_v] (same aggregate for all queries).
    parent_agg_for_correction = parent_aggregates_gathered.unsqueeze(2).expand(B, H, T, P, D_v)

    # Gather parent counts and child counts (spec §9.12: empty codewords = 0).
    parent_counts_gathered = parent_counts.gather(-1, selected)  # [B, H, P]
    if child_counts is not None:
        child_idx = selected.unsqueeze(-1).expand(B, H, P, C)
        child_counts_gathered = child_counts.gather(2, child_idx)  # [B, H, P, C]
    else:
        child_counts_gathered = None

    # Child logits: use real Q · C_c^T when provided, else approximate.
    if child_logits is not None:
        # If child_logits is already [B, H, T, P, C] (pre-gathered by caller),
        # skip the internal gather. Otherwise gather from [B, H, T, M0, C].
        if child_logits.shape[-2] == P:
            child_logits_gathered = child_logits
        else:
            child_idx = selected.unsqueeze(2).unsqueeze(-1).expand(B, H, T, P, C)
            child_logits_gathered = child_logits.gather(-2, child_idx)
    else:
        # Approximation: child_logits = parent_logit / C (spec §7.12).
        child_logits_gathered = parent_logit_gathered.unsqueeze(-1).expand(B, H, T, P, C) / C

    # Correction: replace each selected parent's contribution with children.
    # Uses OnlineSoftmaxState.replace() (subtract parent + add children).
    new_state = vectorized_correction(
        state,
        parent_logit_gathered,
        child_logits_gathered,
        parent_agg_for_correction,  # raw V̄_p, not A_p·V̄_p
        children,
        num_children=C,
        parent_counts=parent_counts_gathered,
        child_counts=child_counts_gathered,
    )

    # Merge strategy: combine parent and child aggregates.
    parent_probs_for_merge = parent_logit_gathered.unsqueeze(-1)  # [B, H, T, P, 1]
    parent_value_for_merge = parent_value_gathered  # [B, H, T, P, D_v]
    # Child probs from child logits via softmax.
    child_probs_for_merge = child_logits_gathered.softmax(dim=-1)  # [B, H, T, P, C]
    child_value_for_merge = children.unsqueeze(2).expand(B, H, T, P, C, D_v)
    # M6: Select merge strategy from configured value.
    if merge_strategy == "probability":
        merger = ProbabilityMerge()
    elif merge_strategy == "weighted":
        from avqa.merge import WeightedMerge

        merger = WeightedMerge()
    elif merge_strategy == "logit":
        from avqa.merge import LogitMerge

        merger = LogitMerge()
    elif merge_strategy == "normalized":
        from avqa.merge import NormalizedMerge

        merger = NormalizedMerge()
    else:
        msg = f"unknown merge strategy: {merge_strategy!r}"
        raise ValueError(msg)
    merge_value = merger.merge(
        MergeInputs(
            parent_probs=parent_probs_for_merge,
            parent_value=parent_value_for_merge,
            child_probs=child_probs_for_merge,
            child_value=child_value_for_merge,
        )
    )
    merge_value = merge_value.sum(dim=-2)  # [B, H, T, D_v]

    return RefinementResult(
        state=new_state,
        selected_parents=selected,
        merge_value=merge_value,
    )


__all__ = ["AdaptiveRefinement", "RefinementResult", "refine", "vectorized_correction"]
