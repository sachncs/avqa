"""Routing subsystem for AVQA (spec §3.10, §7.10, §9.5, §9.6).

Computes per-codeword importance from attention statistics and selects
the top-P parents for refinement. The router MUST NOT perform attention
computations (spec §4.7.4) or expand nodes (spec §4.7.4); it only ranks.

ponytail: collapsed the planned routing package (7 sub-modules) into
one src/avqa/routing.py. ImportanceEstimator is one function; selectors
are small strategies on top of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from avqa.exceptions import RoutingError
from avqa.logging import get_logger
from avqa.registry import ROUTER_REGISTRY

_logger = get_logger("routing")


def compute_importance(
    attention_probs: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Per-codeword importance score (spec §7.10).

        w_j = n_j * sum_i A_ij

    When ``attention_probs`` are properly normalized (softmax sums to 1
    along the codeword axis), the per-query denominator Z_i = 1 and
    the formula reduces to the total attention mass to codeword j,
    weighted by its assignment count.

    Args:
        attention_probs: ``[B, H, N, M_0]`` attention probabilities
            (should be normalized along dim=-1).
        counts: ``[B, H, M_0]`` per-codeword assignment counts.

    Returns:
        Per-codeword importance. Shape ``[B, H, M_0]``.
    """
    if attention_probs.ndim != 4:
        raise RoutingError(
            f"attention_probs must be rank 4 [B, H, N, M_0], got {attention_probs.ndim}",
        )
    if counts.ndim != 3:
        raise RoutingError(
            f"counts must be rank 3 [B, H, M_0], got {counts.ndim}",
        )
    if attention_probs.shape[:2] != counts.shape[:2]:
        raise RoutingError(
            "attention_probs and counts have incompatible batch shapes",
        )
    if attention_probs.shape[-1] != counts.shape[-1]:
        raise RoutingError(
            "attention_probs and counts have mismatched codeword dimensions",
        )
    # w_j = n_j * sum_i A_ij.  For normalized probs, Z_i = 1 (no division needed).
    total_mass = attention_probs.sum(dim=-2)                              # [B, H, M_0]
    return counts * total_mass


@dataclass
class RoutingDecision:
    """Output of a router: which parents to refine, in priority order.

    Attributes:
        selected_indices: Per-(B, H) selected parent indices. Shape ``[B, H, P]``.
        importance: Per-codeword importance score used for the decision.
            Shape ``[B, H, M_0]``.
    """

    selected_indices: torch.Tensor
    importance: torch.Tensor

    @property
    def num_selected(self) -> int:
        """Number of selected parents per (B, H) — equal to the budget P."""
        return int(self.selected_indices.shape[-1])


class Router(ABC):
    """Abstract router interface (spec §4.7, §5.10)."""

    @abstractmethod
    def select(
        self,
        importance: torch.Tensor,
        budget: int,
    ) -> RoutingDecision:
        """Select the top-``budget`` parents for refinement.

        Args:
            importance: Per-codeword importance ``[B, H, M_0]``.
            budget: Maximum number of parents to refine (P).

        Returns:
            :class:`RoutingDecision` with indices and the original scores.
        """


class TopPRouter(Router):
    """Top-P selector with deterministic tie-breaking (spec §9.6.1, G14).

    Ties are broken by sorting on ``(importance desc, index asc)`` so
    that identical scores yield lower-indexed parents first. The
    implementation uses :func:`torch.sort` with ``stable=True`` and
    ``descending=True``, which preserves the original (ascending-index)
    order among ties.

    Args:
        deterministic: If ``True``, ties resolve to lower indices. The
            default (``True``) matches the reference algorithm.

    Example:
        >>> router = TopPRouter()
        >>> importance = torch.tensor([[[0.4, 0.9, 0.9, 0.1]]])
        >>> decision = router.select(importance, budget=2)
        >>> decision.selected_indices
        tensor([[[1, 2]]])
    """

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic

    def select(
        self,
        importance: torch.Tensor,
        budget: int,
    ) -> RoutingDecision:
        """Return the top-``budget`` indices per (B, H) head."""
        if budget <= 0:
            raise RoutingError(f"budget must be > 0, got {budget}")
        if budget > importance.shape[-1]:
            raise RoutingError(
                f"budget ({budget}) exceeds number of codewords ({importance.shape[-1]})",
            )
        if self.deterministic:
            # Stable sort: descending by score; ties keep ascending index order.
            _, indices = torch.sort(importance, dim=-1, descending=True, stable=True)
        else:
            _, indices = torch.sort(importance, dim=-1, descending=True)
        indices = indices[..., :budget]
        return RoutingDecision(selected_indices=indices, importance=importance)


class ThresholdRouter(Router):
    """Threshold-based selection (spec §2.8, optional).

    Selects every codeword with importance >= threshold, capped at
    ``budget`` (returns the highest-scoring ones if the threshold allows
    more).

    Args:
        threshold: Minimum importance score to be selected.

    Example:
        >>> router = ThresholdRouter(threshold=0.5)
        >>> importance = torch.tensor([[[0.4, 0.9, 0.6, 0.3]]])
        >>> decision = router.select(importance, budget=2)
        >>> decision.selected_indices
        tensor([[[1, 2]]])
    """

    def __init__(self, threshold: float = 0.0) -> None:
        if threshold < 0.0:
            raise RoutingError(f"threshold must be >= 0, got {threshold}")
        self.threshold = threshold

    def select(
        self,
        importance: torch.Tensor,
        budget: int,
    ) -> RoutingDecision:
        """Select codewords whose importance >= ``self.threshold``."""
        if budget <= 0:
            raise RoutingError(f"budget must be > 0, got {budget}")
        mask = importance >= self.threshold                                  # [B, H, M_0]
        # Use top-k on masked values; entries below threshold become -inf.
        masked = torch.where(mask, importance, torch.full_like(importance, float("-inf")))
        indices = masked.topk(min(budget, importance.shape[-1]), dim=-1).indices
        return RoutingDecision(selected_indices=indices, importance=importance)


class BudgetRouter(Router):
    """Budget-constrained selector that picks the top ``budget`` by score.

    Equivalent to :class:`TopPRouter` but explicitly named for users who
    configure their pipeline via "budget" terminology.
    """

    def select(
        self,
        importance: torch.Tensor,
        budget: int,
    ) -> RoutingDecision:
        """Return the top-``budget`` indices."""
        return TopPRouter().select(importance, budget)


# Routers registered in the spec-mandated registry.
ROUTER_REGISTRY.register("topp")(TopPRouter)  # type: ignore[arg-type]
ROUTER_REGISTRY.register("threshold")(ThresholdRouter)  # type: ignore[arg-type]
ROUTER_REGISTRY.register("budget")(BudgetRouter)  # type: ignore[arg-type]


__all__ = [
    "BudgetRouter",
    "Router",
    "RoutingDecision",
    "ThresholdRouter",
    "TopPRouter",
    "compute_importance",
]
