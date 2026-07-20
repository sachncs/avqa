"""Merge strategies for AVQA (spec §3.11, §9.9).

Combines coarse parent attention with refined child attention. The
strategies differ in how they weight and normalize the two contributions.

ponytail: collapsed the planned merge package (8 sub-modules) into one
src/avqa/merge.py. Four strategies on a single input contract; each is a
small function.
"""
from __future__ import annotations



from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from avqa.exceptions import ConfigurationError
from avqa.registry import MERGE_REGISTRY


@dataclass
class MergeInputs:
    """Inputs to a merge strategy (spec §9.9).

    Attributes:
        parent_probs: Per-(B, H, N, 1) parent attention probability.
        parent_value: Per-(B, H, N, D) parent-weighted value.
        child_probs: Per-(B, H, N, C) refined child attention probabilities.
        child_value: Per-(B, H, N, C, D) refined child-weighted values.
    """

    parent_probs: torch.Tensor
    parent_value: torch.Tensor
    child_probs: torch.Tensor
    child_value: torch.Tensor

    def __post_init__(self) -> None:
        expected_parent = (*self.parent_value.shape[:-1], 1)
        if tuple(self.parent_probs.shape) != expected_parent:
            raise ConfigurationError(
                "parent_probs shape must match parent_value[..., -1:]",
                {
                    "parent_probs": tuple(self.parent_probs.shape),
                    "parent_value": tuple(self.parent_value.shape),
                },
            )
        if tuple(self.child_probs.shape) != self.child_value.shape[:-1]:
            raise ConfigurationError(
                "child_probs shape must match child_value[..., -1]",
                {
                    "child_probs": tuple(self.child_probs.shape),
                    "child_value": tuple(self.child_value.shape),
                },
            )


class MergeStrategy(ABC):
    """Abstract merge strategy (spec §4.7.6, §5.10)."""

    @abstractmethod
    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        """Combine parent + child contributions into a single per-(N, D) tensor.

        Returns:
            Tensor of shape ``[B, H, N, D]``.
        """


class ProbabilityMerge(MergeStrategy):
    """Probability-space merge: child probs replace parent probs (spec §3.11.2).

    The parent's mass is subtracted from the denominator and the child's
    mass is added in its place. This preserves the denominator invariant
    of the softmax (attention sums to 1) under correction.
    """

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        """Subtract parent, add child (already weighted)."""
        delta = (inputs.child_probs.unsqueeze(-1) * inputs.child_value).sum(
            dim=-2
        ) - inputs.parent_probs * inputs.parent_value
        return inputs.parent_value + delta


class WeightedMerge(MergeStrategy):
    """Weighted merge: parents weighted by ``parent_weight``, children by ``child_weight``.

    Args:
        parent_weight: Multiplier on parent contribution. Default ``0.5``.
        child_weight: Multiplier on child contribution. Default ``0.5``.
    """

    def __init__(self, parent_weight: float = 0.5, child_weight: float = 0.5) -> None:
        self.parent_weight = parent_weight
        self.child_weight = child_weight

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        return (
            self.parent_weight * inputs.parent_probs * inputs.parent_value
            + self.child_weight
            * (inputs.child_probs.unsqueeze(-1) * inputs.child_value).sum(dim=-2)
        )


class LogitMerge(MergeStrategy):
    """Logit-space merge: combine parent and child logits before re-softmax (spec §3.11.2).

    Concatenates the parent logit with the child logits, applies
    log-softmax to get normalized weights, then computes the weighted
    sum of parent and child values.  This differs from
    :class:`ProbabilityMerge` in that the normalization is joint over
    parent + children rather than a subtract-parent / add-children delta.
    """

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        # Log-probabilities: parent [B,H,T,P,1] and children [B,H,T,P,C].
        parent_log = inputs.parent_probs.clamp_min(1e-12).log()
        child_log = inputs.child_probs.clamp_min(1e-12).log()
        # Concatenate and softmax: [B,H,T,P, 1+C].
        combined = torch.cat([parent_log, child_log], dim=-1)
        weights = combined.softmax(dim=-1)
        parent_weight = weights[..., :1]  # [B,H,T,P,1]
        child_weight = weights[..., 1:]  # [B,H,T,P,C]
        parent_contrib = parent_weight * inputs.parent_value
        child_contrib = (child_weight.unsqueeze(-1) * inputs.child_value).sum(dim=-2)
        return parent_contrib + child_contrib


class NormalizedMerge(MergeStrategy):
    """Normalized merge: re-normalize so the attention distribution sums to 1.

    Computes the unnormalized merge (ProbabilityMerge) and then divides
    by the total probability mass per row.
    """

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        base = ProbabilityMerge().merge(inputs)
        # Reconstruction of probs: parent_probs (mass 1) - parent_probs + sum(child_probs)
        total_mass = inputs.child_probs.sum(dim=-1, keepdim=True)
        # Normalize by mass so that, in the linear regime, attention sums to 1.
        return base / total_mass.clamp_min(1e-12)


# Register each merge strategy under its canonical name (spec §3.11.2).
MERGE_REGISTRY.register("probability")(ProbabilityMerge)  # type: ignore[arg-type]
MERGE_REGISTRY.register("weighted")(WeightedMerge)  # type: ignore[arg-type]
MERGE_REGISTRY.register("logit")(LogitMerge)  # type: ignore[arg-type]
MERGE_REGISTRY.register("normalized")(NormalizedMerge)  # type: ignore[arg-type]


__all__ = [
    "LogitMerge",
    "MergeInputs",
    "MergeStrategy",
    "NormalizedMerge",
    "ProbabilityMerge",
    "WeightedMerge",
]
