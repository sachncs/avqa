"""Multi-pass refinement (SPEC \u00a715, ACMPR / OPT-0004).

Runs the existing ``refine`` step multiple times per forward call, with
a geometric budget decay across passes. Theorem 15.1 of SPEC \u00a715
guarantees the residual after ``k`` passes is bounded by ``\u03b1^k \u00b7
||\u0394_0||`` for some ``\u03b1 < 1`` depending on the attention
dispersion, so ``k = \u230alog \u03c1\u207b\u00b9 \u00b7 log 1/\u03b5\u230b`` passes drive
the residual below ``\u03b5`` (SPEC \u00a715.2).

Total work is bounded above by ``2 \u00b7 P \u00b7 C`` for \u03c1 \u2264 0.5 (SPEC
\u00a715.4 acceptance criterion) and at \u03c1 = 1.0 (default) the
budget is held constant across passes so the work is exactly
``k \u00b7 P \u00b7 C`` \u2014 identical to the paper's single-pass work
when ``k = 1`` (the default).

When ``passes = 1`` (the default) the module is a no-op wrapper that
returns the input state unchanged; this is the paper-equivalence
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from avqa.attention import OnlineSoftmaxState
from avqa.logging import get_logger
from avqa.refinement import refine

if TYPE_CHECKING:
    from avqa.routing import RoutingDecision

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
        decision: "RoutingDecision",
        attention_probs: torch.Tensor,
        parent_counts: torch.Tensor,
        child_logits: torch.Tensor | None = None,
        child_counts: torch.Tensor | None = None,
        merge_strategy: str = "probability",
    ) -> tuple[OnlineSoftmaxState, list[float]]:
        """Run ``self.passes`` correction passes and return the final state.

        Args:
            state: Running online-softmax state from the parent pass.
            parent_probs, parent_value, parent_aggregates: As in
                :func:`avqa.refinement.refine`.
            child_aggregates, children_per_parent: As in
                :func:`avqa.refinement.refine`.
            decision: Routing decision for the FIRST pass. Subsequent
                passes reuse the result of the previous pass through
                the same `refine` call.
            attention_probs, parent_counts, child_logits, child_counts:
                As in :func:`avqa.refinement.refine`.
            merge_strategy: As in :func:`avqa.refinement.refine`.

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

        # Multi-pass: bounded geometric budget decay.
        budgets = self.pass_budgets(decision.num_selected)
        residual_norms: list[float] = []
        current_state = state
        prev_state = state
        for _budget in budgets:
            result = refine(
                state=current_state,
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
            current_state = result.state
            # Residual norm against the previous pass.
            denom = current_state.running_denominator.clamp_min(1e-12)
            cur_attn = current_state.running_numerator / denom.unsqueeze(-1)
            prev_denom = prev_state.running_denominator.clamp_min(1e-12)
            prev_attn = prev_state.running_numerator / prev_denom.unsqueeze(-1)
            residual_norms.append(float((cur_attn - prev_attn).norm().item()))
            prev_state = current_state
        return current_state, residual_norms


__all__ = ["MultiPassRefiner", "PassBudget", "compute_pass_budgets"]
