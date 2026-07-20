"""Multi-pass refinement (SPEC \u00a715, ACMPR / OPT-0004).

The ``passes=1`` path is the paper-equivalent contract.  The
``passes>1`` path implements **disjoint-set multi-pass correction**:
each pass corrects a *different* subset of parents.  After pass *k*,
already-refined parents are masked out, the router re-selects the
top-P parents from the remaining set with budget decay, and fresh
child logits are computed.  This guarantees converging residual norms
(each pass corrects less-important parents) rather than the
divergent ``state_0 + k*(child - parent)`` of the naive approach.

Residual-norms: ``||cur_attn - prev_attn||`` is recorded per pass.
With ``passes=1`` this is ``[0.0]`` by construction; with ``passes>1``
the norms should decrease monotonically.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from avqa.attention import OnlineSoftmaxState
from avqa.logging import get_logger
from avqa.refinement import refine
from avqa.routing import RoutingDecision, TopPRouter, compute_importance

_logger = get_logger("multipass")


@dataclass(frozen=True)
class PassBudget:
    """Per-pass budget record (SPEC \u00a715.2)."""

    pass_index: int
    budget: int
    geometric_decay: float


def compute_pass_budgets(base: int, passes: int, decay: float) -> list[int]:
    """Return ``passes`` budgets via the geometric decay formula.

    Args:
        base: Base budget ``P``. Budgets are ``max(1, \u2308P \u00b7 \u03c1^i\u230b)``
            for ``i = 0, 1, ..., passes - 1``.
        passes: Number of passes to run.
        decay: Geometric decay rate ``\u03c1 \u2208 (0, 1]``. ``1.0`` is the
            constant-budget special case (paper-exact; the residual
            bound becomes ``\u03b1^k \u00b7 ||\u0394_0||`` at constant ``P``).

    Returns:
        List of integer per-pass budgets, length ``passes``.

    Raises:
        ValueError: If ``decay`` is outside ``(0, 1]`` or ``passes``
            is non-positive.
    """
    if decay <= 0.0 or decay > 1.0:
        msg = f"decay must be in (0, 1], got {decay}"
        raise ValueError(msg)
    if passes <= 0:
        msg = f"passes must be positive, got {passes}"
        raise ValueError(msg)
    if base <= 0:
        msg = f"base must be positive, got {base}"
        raise ValueError(msg)
    return [max(1, round(base * (decay**i))) for i in range(passes)]


class MultiPassRefiner:
    """Multi-pass correction wrapper (SPEC \u00a715.3).

    Args:
        passes: Number of correction passes (default ``1`` = paper).
        decay: Geometric per-pass budget decay ``\u03c1`` (default
            ``1.0`` = constant budget; set to ``0.5`` to halve the
            budget each pass).

    The public API is a single ``refine(...)`` method that mimics
    ``avqa.refinement.refine``'s signature, returning the post-pass
    :class:`OnlineSoftmaxState` and the per-pass residual norm.
    """

    def __init__(self, passes: int = 1, decay: float = 1.0) -> None:
        if passes <= 0:
            msg = f"passes must be positive, got {passes}"
            raise ValueError(msg)
        if decay <= 0.0 or decay > 1.0:
            msg = f"decay must be in (0, 1], got {decay}"
            raise ValueError(msg)
        self.passes = passes
        self.decay = decay
        _logger.debug(
            "MultiPassRefiner initialised: passes=%d decay=%.3f",
            passes,
            decay,
        )

    def pass_budgets(self, base: int) -> list[int]:
        """Return the per-pass budgets for a given base budget."""
        return compute_pass_budgets(base, self.passes, self.decay)

    def refine(
        self,
        state: OnlineSoftmaxState,
        parent_probs: torch.Tensor,
        parent_value: torch.Tensor,
        parent_aggregates: torch.Tensor,
        child_aggregates: torch.Tensor,
        children_per_parent: int,
        decision: RoutingDecision,
        attention_probs: torch.Tensor,
        parent_counts: torch.Tensor,
        child_logits: torch.Tensor | None = None,
        child_counts: torch.Tensor | None = None,
        merge_strategy: str = "probability",
        query: torch.Tensor | None = None,
        child_keys: torch.Tensor | None = None,
    ) -> tuple[OnlineSoftmaxState, list[float]]:
        """Run ``self.passes`` correction passes and return the final state.

        Args:
            state: Running online-softmax state from the parent pass.
            parent_probs, parent_value, parent_aggregates: As in
                :func:`avqa.refinement.refine`.
            child_aggregates, children_per_parent: As in
                :func:`avqa.refinement.refine`.
            decision: Routing decision for the FIRST pass.
            attention_probs, parent_counts, child_logits, child_counts:
                As in :func:`avqa.refinement.refine`.
            merge_strategy: As in :func:`avqa.refinement.refine`.
            query: Optional ``[B, H, T, D]`` query tensor. When provided
                together with ``child_keys``, enables disjoint-set
                re-routing on passes > 1 (each pass corrects a different
                subset of parents).  When ``None``, multi-pass falls back
                to single-pass with a warning.
            child_keys: Optional ``[H, M_0, C, D]`` children codebook
                tensor.  Enables child-logit recomputation for re-routing.

        Returns:
            Tuple ``(final_state, residual_norms)`` where
            ``residual_norms[i]`` is the L2 norm of the residual
            between pass ``i`` and pass ``i+1``'s state.
        """
        # Single-pass: paper-equivalent behaviour, residual is zero by
        # construction.
        if self.passes == 1:
            result = refine(
                state=state,
                parent_probs=parent_probs,
                parent_value=parent_value,
                parent_aggregates=parent_aggregates,
                child_aggregates=child_aggregates,
                children_per_parent=children_per_parent,
                decision=decision,
                attention_probs=attention_probs,
                parent_counts=parent_counts,
                child_logits=child_logits,
                child_counts=child_counts,
                merge_strategy=merge_strategy,
            )
            return result.state, [0.0]

        # Multi-pass with disjoint-set re-routing: each pass corrects a
        # different subset of parents.  Requires query + child_keys for
        # child-logit recomputation; falls back to single-pass otherwise.
        if query is None or child_keys is None:
            _logger.warning(
                "passes=%d but query/child_keys not provided; "
                "re-routing disabled, falling back to single-pass",
                self.passes,
            )
            result = refine(
                state=state,
                parent_probs=parent_probs,
                parent_value=parent_value,
                parent_aggregates=parent_aggregates,
                child_aggregates=child_aggregates,
                children_per_parent=children_per_parent,
                decision=decision,
                attention_probs=attention_probs,
                parent_counts=parent_counts,
                child_logits=child_logits,
                child_counts=child_counts,
                merge_strategy=merge_strategy,
            )
            return result.state, [0.0]

        budgets = self.pass_budgets(decision.num_selected)
        B, H, _T, D = query.shape
        M0 = parent_probs.shape[-1]
        C = children_per_parent
        device = state.running_max.device

        # Pre-expand child_keys for efficient einsum: [B, H, M0, C, D].
        child_keys_exp = child_keys.unsqueeze(0).expand(B, H, M0, C, D)

        # Track which parents have been refined to exclude them from
        # subsequent passes (disjoint-set property).
        refined_mask = torch.zeros(B, H, M0, dtype=torch.bool, device=device)
        router = TopPRouter()

        residual_norms: list[float] = []
        current_state = state
        prev_state = state

        for pass_budget in budgets:
            if pass_budget <= 0:
                break

            # Re-route on passes > 0: mask out already-refined parents.
            current_budget = pass_budget
            if refined_mask.any():
                importance = compute_importance(attention_probs, parent_counts)
                importance = importance.masked_fill(refined_mask, float("-inf"))
                num_remaining = int((~refined_mask).sum(dim=-1).min().item())
                current_budget = min(pass_budget, num_remaining)
                if current_budget <= 0:
                    break
                current_decision = router.select(importance, current_budget)
            else:
                current_decision = decision

            # Recompute child_logits: Q . C_c^T / sqrt(D).
            current_child_logits = (
                torch.einsum("bhtd,bhmcd->bhtmc", query, child_keys_exp) / (D**0.5)
            )
            if child_counts is not None:
                current_child_logits = current_child_logits.masked_fill(
                    ~child_counts.unsqueeze(2).bool(), float("-inf")
                )

            result = refine(
                state=current_state,
                parent_probs=parent_probs,
                parent_value=parent_value,
                parent_aggregates=parent_aggregates,
                child_aggregates=child_aggregates,
                children_per_parent=C,
                decision=current_decision,
                attention_probs=attention_probs,
                parent_counts=parent_counts,
                child_logits=current_child_logits,
                child_counts=child_counts,
                merge_strategy=merge_strategy,
            )
            current_state = result.state

            # Mark newly refined parents.
            refined_mask.scatter_(
                2,
                current_decision.selected_indices,
                torch.ones_like(current_decision.selected_indices, dtype=torch.bool),
            )

            # Residual norm against the previous pass.
            denom = current_state.running_denominator.clamp_min(1e-12)
            cur_attn = current_state.running_numerator / denom.unsqueeze(-1)
            prev_denom = prev_state.running_denominator.clamp_min(1e-12)
            prev_attn = prev_state.running_numerator / prev_denom.unsqueeze(-1)
            residual_norms.append(float((cur_attn - prev_attn).norm().item()))
            prev_state = current_state

        return current_state, residual_norms


__all__ = ["MultiPassRefiner", "PassBudget", "compute_pass_budgets"]
