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

from avqa.exceptions import ConfigurationError, RoutingError


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
            RoutingError: If ``strategy`` is unknown.
        """
        if strategy == "default":
            return DefaultScheduler(budget=budget)
        if strategy == "adaptive":
            return AdaptiveScheduler(min_budget=max(1, budget // 2), max_budget=budget)
        msg = f"unknown scheduler strategy: {strategy!r}"
        raise RoutingError(msg)

    @abstractmethod
    def budget_for(self, importance: torch.Tensor) -> int | torch.Tensor:
        """Return the refinement budget for the given importance scores.

        Args:
            importance: Per-codeword importance ``[B, H, M_0]``.

        Returns:
            Number of parents to refine (P). May be a scalar (``int``)
            or a per-(B, H) tensor of shape ``[B, H]``.
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
            raise RoutingError(f"budget must be > 0, got {budget}")
        self.budget = budget

    def budget_for(self, importance: torch.Tensor) -> int:
        """Return the constant budget."""
        del importance  # constant budget does not depend on importance
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
            raise ConfigurationError(f"min_budget must be > 0, got {min_budget}")
        if max_budget < min_budget:
            raise ConfigurationError(
                f"max_budget ({max_budget}) < min_budget ({min_budget})",
            )
        if not 0.0 < entropy_threshold <= 1.0:
            raise ConfigurationError(
                f"entropy_threshold must be in (0, 1], got {entropy_threshold}",
            )
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.entropy_threshold = entropy_threshold

    def budget_for(self, importance: torch.Tensor) -> int | torch.Tensor:
        """Adapt budget per-(B, H) based on importance entropy.

        Returns a tensor of shape ``[B, H]`` when ``importance`` is
        ``[B, H, M_0]`` so each (B, H) position receives its own
        budget. Returns a scalar when ``importance`` has rank 1.
        """
        if importance.ndim < 2:
            # Scalar fallback for low-rank inputs.
            flat = importance / importance.sum().clamp_min(1e-12)
            entropy = -(flat * flat.clamp_min(1e-12).log()).sum().item()
            max_entropy = float(torch.log(torch.tensor(flat.numel())).item())
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            return self.max_budget if norm_entropy < self.entropy_threshold else self.min_budget
        # Per-(B, H) entropy: [B, H, M_0] -> [B, H]
        p = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy = float(torch.log(torch.tensor(p.shape[-1])).item())
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        budget = torch.where(
            norm_entropy < self.entropy_threshold,
            torch.full_like(norm_entropy, float(self.max_budget)),
            torch.full_like(norm_entropy, float(self.min_budget)),
        )
        return budget.to(torch.int64)


__all__ = ["AdaptiveScheduler", "DefaultScheduler", "Scheduler"]
