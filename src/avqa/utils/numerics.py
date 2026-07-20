"""Numerical helpers for AVQA.

Spec §7.14 mandates FlashAttention's online-softmax algorithm. Stdlib
``torch.softmax`` is already numerically stable via max-subtraction, so
we do not re-implement ``stable_softmax``. The one thing stdlib does NOT
provide is the online-softmax merge step that FlashAttention uses to
combine partial tiles, which is implemented here.
"""
from __future__ import annotations



import torch

# ponytail: stable_softmax omitted — torch.softmax already subtracts the
# row max and is numerically stable. No need to re-wrap.


def online_softmax_step(
    m_old: torch.Tensor,
    l_old: torch.Tensor,
    acc_old: torch.Tensor,
    m_new: torch.Tensor,
    l_new: torch.Tensor,
    acc_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two online-softmax accumulators (FlashAttention-2 algorithm).

    Given the running state ``(m_old, l_old, acc_old)`` for one tile and
    the freshly computed ``(m_new, l_new, acc_new)`` for the next tile,
    return the merged ``(m, l, acc)`` covering both tiles. See spec
    §7.14 and FlashAttention-2 §3.2.

    Args:
        m_old: Per-row running max from prior tiles. Shape ``[..., D_k]``.
        l_old: Per-row running denominator (sum of exp(x_i - m_old)). Same
            shape as ``m_old``.
        acc_old: Per-row running numerator (sum of exp(x_i - m_old) * v_i).
            Shape ``[..., D_k, D_v]``.
        m_new: Max of the new tile. Same shape as ``m_old``.
        l_new: Denominator for the new tile (sum of exp(x_j - m_new)).
            Same shape as ``m_old``.
        acc_new: Numerator for the new tile (sum of exp(x_j - m_new) * v_j).
            Same shape as ``acc_old``.

    Returns:
        Tuple ``(m, l, acc)`` covering both tiles, all with the same
        shapes as their ``_old`` counterparts.

    Example:
        >>> import torch
        >>> m_old = torch.tensor([1.0, 2.0])
        >>> l_old = torch.tensor([1.0, 2.0])
        >>> acc_old = torch.tensor([[1.0], [2.0]])
        >>> m_new = torch.tensor([2.0, 1.0])
        >>> l_new = torch.tensor([0.5, 0.25])
        >>> acc_new = torch.tensor([[0.5], [0.25]])
        >>> m, l, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        >>> m.shape, l.shape, acc.shape
        (torch.Size([2]), torch.Size([2]), torch.Size([2, 1]))
    """
    m = torch.maximum(m_old, m_new)
    alpha = torch.exp(m_old - m)
    beta = torch.exp(m_new - m)
    denom = alpha * l_old + beta * l_new
    acc = alpha.unsqueeze(-1) * acc_old + beta.unsqueeze(-1) * acc_new
    return m, denom, acc


__all__ = ["online_softmax_step"]
