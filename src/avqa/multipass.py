"""Multi-pass refinement (SPEC \u00a715, ACMPR / OPT-0004).

**Status (2026-07-16).** The ``passes=1`` path is the paper-equivalent
contract and ships today. The ``passes>1`` path is **disabled by
default and mathematically non-trivial**: the existing
``refine`` operator re-applies ``state - parent + child`` each pass, so
calling it twice yields ``state_0 + 2*(child - parent)`` which
**diverges from the all-children oracle rather than converging**
(validated by ``tests/unit/test_multipass.py::test_passes_4_returns_four_residuals``
which only checks that the residual is non-negative, not that it
decreases). A correct multi-pass refinement would require either a
second-order residual update on the running state or a re-derived
child_logits with a fresh budget each pass; that is a separate
algorithmic contribution (tracked as a future work item in
``RESEARCH.md``).

This module therefore ships a **strict no-op wrapper when
``passes=1``** and the budget-decay helper for downstream adoption.
The integration in ``AVQAttention.forward`` only invokes this class
when ``refinement.passes > 1`` and the future-work item is closed.

Residual-norms: when the ``refine`` operator is invoked, we record
``||cur_attn - prev_attn||`` as a diagnostic. With ``passes=1`` this is
``[0.0]`` by construction; with ``passes>1`` it is **not** the
paper-valid residual bound and the integration is gated off.
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
