"""Scheduler subsystem (spec §2.8, §4.7).

Schedulers allocate the refinement budget per forward pass. The reference
implementation uses a fixed budget; an entropy-driven adaptive variant
is provided as an extension.

ponytail: collapsed the planned scheduler package (5 sub-modules) into
one src/avqa/scheduler.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Scheduler(ABC):
    """Abstract scheduler interface (spec §4.7)."""

    @classmethod
    def create(cls, strategy: str = "default", *, budget: int = 8) -> Scheduler:
        """Factory: resolve ``strategy`` to a concrete :class:`Scheduler`.

        Args:
            strategy: ``"default"`` (constant budget) or ``"adaptive"``
                (entropy-driven).
            budget: Constant budget for the default scheduler; used as
                ``max_budget`` for the adaptive variant.

        Returns:
            A fresh :class:`Scheduler` instance.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        if strategy == "default":
            return DefaultScheduler(budget=budget)
        if strategy == "adaptive":
            return AdaptiveScheduler(min_budget=max(1, budget // 2), max_budget=budget)
        msg = f"unknown scheduler strategy: {strategy!r}"
        raise ValueError(msg)

    @abstractmethod
    def budget_for(self, importance: torch.Tensor) -> int:
        """Return the refinement budget for the given importance scores.

        Args:
            importance: Per-codeword importance ``[B, H, M_0]``.

        Returns:
            Number of parents to refine (P).
        """


class DefaultScheduler(Scheduler):
    """Fixed-budget scheduler (spec §2.8 default).

    Args:
        budget: Constant number of parents to refine.

    Example:
        >>> s = DefaultScheduler(budget=8)
        >>> s.budget_for(torch.zeros(1, 1, 64))
        8
    """

    def __init__(self, budget: int = 8) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        self.budget = budget

    def budget_for(self, importance: torch.Tensor) -> int:
        """Return the constant budget."""
        return self.budget


class AdaptiveScheduler(Scheduler):
    """Entropy-driven adaptive budget (spec §2.8, optional extension).

    Increases the budget when the importance distribution is
    concentrated (low entropy) and shrinks it when mass is spread
    across many codewords (high entropy).

    Args:
        min_budget: Minimum refinement budget.
        max_budget: Maximum refinement budget.
        entropy_threshold: Importance-entropy below which budget is
            increased (focused attention).
    """

    def __init__(
        self,
        min_budget: int = 4,
        max_budget: int = 32,
        entropy_threshold: float = 0.5,
    ) -> None:
        if min_budget <= 0:
            raise ValueError(f"min_budget must be > 0, got {min_budget}")
        if max_budget < min_budget:
            raise ValueError(
                f"max_budget ({max_budget}) < min_budget ({min_budget})",
            )
        if not 0.0 < entropy_threshold <= 1.0:
            raise ValueError(
                f"entropy_threshold must be in (0, 1], got {entropy_threshold}",
            )
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.entropy_threshold = entropy_threshold

    def budget_for(self, importance: torch.Tensor) -> int:
        """Adapt budget based on average entropy across (B, H)."""
        # Average importance over (B, H) → [M_0].
        flat = importance.mean(dim=tuple(range(importance.ndim - 1)))
        flat = flat / flat.sum().clamp_min(1e-12)
        entropy = -(flat * flat.clamp_min(1e-12).log()).sum().item()
        max_entropy = float(torch.log(torch.tensor(flat.numel())).item())
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        # Low entropy → focused → expand budget; high entropy → spread → shrink.
        if norm_entropy < self.entropy_threshold:
            return self.max_budget
        return self.min_budget


__all__ = ["AdaptiveScheduler", "DefaultScheduler", "Scheduler"]
