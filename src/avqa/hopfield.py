"""Hopfield-VQ-Attention (HVAQ) primitive (SPEC \u00a716, OPT-0005).

HVAQ generalises the paper's fixed-temperature softmax attention
``softmax(q \u00b7 k^T / \u221ad) \u00b7 v`` with per-parent and per-query
inverse temperature. The mathematical core (SPEC \u00a716.2):

    parent_logits = \u03b2_q \u00b7 \u03b2_p \u00b7 (q \u00b7 k^T)        (paper uses \u03b2_p = \u03b2_q = 1/\u221ad)
    parent_probs   = softmax(parent_logits)

The online softmax machinery of SPEC \u00a77.14 absorbs any positive
scaling, so the existing reference path is reused unchanged: the
``\u03b2_q \u00b7 \u03b2_p`` factor enters via the dot product.

Per-query temperature: from the router's top-`P` selection we compute
the attention-mass entropy and feed it to one of two schedules:

- ``\"entropy\"`` (HVAQ-ENT): ``\u03b2_q = \u03b2_0 \u00b7 (1 + 1 / (1 + H_top))``
  (peaked distribution \u21d2 doubled temperature, uniform \u21d2 paper).
- ``\"linear\"`` (HVAQ-LIN):  ``\u03b2_q = \u03b2_0 \u00b7 (1 + \u03b1 \u00b7 H_top)``
  (linear in the entropy).
- ``\"none\"``: ``\u03b2_q = \u03b2_0`` (paper-exact).

Paper-equivalence: with ``\u03b2_init = 1 / \u221ad`` and
``adaptive = \"none\"`` HVAQ matches the paper to FP32.
"""
from __future__ import annotations

import math
from typing import Literal

import torch

from avqa.exceptions import ConfigurationError

AdaptiveSchedule = Literal["none", "entropy", "linear"]


def paper_beta(head_dim: int) -> float:
    """Return the paper's default temperature ``1 / \u221ad``."""
    if head_dim <= 0:
        msg = f"head_dim must be > 0, got {head_dim}"
        raise ValueError(msg)
    return 1.0 / math.sqrt(head_dim)


def validate_adaptive(value: str) -> str:
    """Accept only the documented schedule identifiers."""
    allowed = ("none", "entropy", "linear")
    if value not in allowed:
        msg = f"adaptive must be one of {allowed}, got {value!r}"
        raise ConfigurationError(msg, {"adaptive": value})
    return value


def per_query_beta(
    parent_probs: torch.Tensor,
    *,
    beta_init: float | torch.Tensor,
    adaptive: str,
    alpha: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute the per-query temperature ``\u03b2_q`` (SPEC \u00a716.2).

    Args:
        parent_probs: ``[B, H, N, M_0]`` parent attention probabilities
            (``M_0`` are the top-`P` parents selected by the router).
        beta_init: Base temperature; for the paper this is
            ``1 / \u221ad``. Positive scalar.
        adaptive: ``\"none\"`` \u2192 constant \u03b2_0; ``\"entropy\"`` \u2192
            ``\u03b2_0 \u00b7 (1 + 1 / (1 + H_top))``; ``\"linear\"`` \u2192
            ``\u03b2_0 \u00b7 (1 + \u03b1 \u00b7 H_top)``.
        alpha: Slope of the linear schedule.

    Returns:
        ``[B, H, N]`` per-query temperature; always positive.

    Raises:
        ValueError: If ``beta_init`` is non-positive.
        ConfigurationError: If ``adaptive`` is unknown.
    """
    if beta_init <= 0.0:
        msg = f"beta_init must be > 0, got {beta_init}"
        raise ValueError(msg)
    validate_adaptive(adaptive)
    if adaptive == "none":
        return torch.full(
            parent_probs.shape[:-1],
            float(beta_init),
            dtype=parent_probs.dtype,
            device=parent_probs.device,
        )
    # Compute H_top = -sum(p log p) for p > 0; the top-P distribution
    # is already softmax-normalised, so this is the attention-mass
    # entropy in nats.
    log_p = torch.where(
        parent_probs > 0,
        parent_probs.log(),
        torch.zeros((), dtype=parent_probs.dtype, device=parent_probs.device),
    )
    h_top = -(parent_probs * log_p).sum(dim=-1)  # [B, H, N]
    schedule = 1.0 + 1.0 / (1.0 + h_top) if adaptive == "entropy" else 1.0 + alpha * h_top
    return beta_init * schedule


def hopfield_logits(
    base_logits: torch.Tensor,
    per_query_beta: torch.Tensor,
    *,
    parent_beta: torch.Tensor = 1.0,
) -> torch.Tensor:
    """Apply the HVAQ temperature schedule to a base logit tensor.

    Args:
        base_logits: ``[B, H, N, M_0]`` raw dot products ``q \u00b7 p^T`` (no
            ``1 / \u221ad`` scaling). The HVAQ temperatures absorb that
            scaling.
        per_query_beta: ``[B, H, N]`` per-query \u03b2_q (the output of
            :func:`per_query_beta`).
        parent_beta: Either a scalar or ``[H, M_0]`` per-parent
            scalar. Default ``1.0`` leaves the parent attention mass
            unchanged up to the overall ``\u03b2_q \u00b7 \u03b2_0`` scaling.

    Returns:
        ``[B, H, N, M_0]`` temperature-scaled logits ready for softmax.
    """
    if base_logits.dim() != 4:
        msg = f"base_logits must be rank 4 [B, H, N, M_0], got {base_logits.dim()}"
        raise ValueError(msg)
    if per_query_beta.shape != base_logits.shape[:-1]:
        msg = (
            f"per_query_beta shape {tuple(per_query_beta.shape)} must match "
            f"base_logits.shape[:-1]={tuple(base_logits.shape[:-1])}"
        )
        raise ValueError(msg)
    # ``[B, H, N, 1] * [B, H, N, M_0]`` broadcasts over the M_0 axis.
    return per_query_beta.unsqueeze(-1) * base_logits * parent_beta


__all__ = [
    "AdaptiveSchedule",
    "hopfield_logits",
    "paper_beta",
    "per_query_beta",
    "validate_adaptive",
]
