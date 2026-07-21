"""Online codebook adaptation (BCAR, OPT-0003).

Bias-Corrected Online Codebook Adaptation extends AVQ-Attention with
an inference-time EMA update of the child codebook; parents are then
re-derived as the mean of their children so the parent-child mean
constraint (SPEC §7.9) is preserved exactly at every step.

Concretely, per child cell ``(p, c)``:

    m_{p,c}   = sum_{j : (a(j), a_c(j)) = (p, c)} k_j / max(1, n_{p,c})
    C'_{p,c}  = decay * C_{p,c} + (1 - decay) * m_{p,c}

and afterwards::

    C'_p = mean_c C'_{p,c}

so the parent-child constraint holds. Note that the EMA input is the
keys tensor — child codewords live in K=Q space (the pipeline dot
products the children with the queries in :func:`avqa.pipeline.child_logits`).

References
----------

- AVQ-Attention paper (§8.9): offline EMA training of the codebook.
- Bottou & Bengio (1994): stochastic K-means convergence rate.
"""
from __future__ import annotations

import torch

from avqa.exceptions import CodebookError, ConfigurationError
from avqa.logging import get_logger

logger = get_logger("bcar")


def online_codebook_adaptation(
    keys: torch.Tensor,
    *,
    parents: torch.Tensor,
    children: torch.Tensor,
    parent_assignments: torch.Tensor,
    child_assignments: torch.Tensor,
    decay: float = 0.99,
) -> None:
    """Apply one EMA update of the children codebook in place.

    Args:
        keys: ``[B, H, N, D]`` keys that produced the assignments.
        parents: ``[H, M_0, D]`` parent codebook; RE-DERIVED in place
            from the updated children via ``parents = mean(children)``.
        children: ``[H, M_0, C, D]`` child codebook; modified in place.
        parent_assignments: ``[B, H, N]`` int64 parent indices.
        child_assignments: ``[B, H, N]`` int64 child indices.
        decay: EMA decay in ``[0, 1)``. ``(1 - decay)`` is the
            contribution of the new mean.

    Raises:
        ConfigurationError: If ``decay`` is outside ``[0, 1)``.
        CodebookError: If the codebook shapes are wrong or assignments
            are missing.
    """
    if decay < 0.0 or decay >= 1.0:
        msg = f"decay must be in [0, 1), got {decay}"
        raise ConfigurationError(msg, {"decay": decay})
    if parents.dim() != 3 or children.dim() != 4:
        msg = (
            f"unexpected shapes: parents {tuple(parents.shape)}, "
            f"children {tuple(children.shape)}"
        )
        raise CodebookError(msg, {"parents": tuple(parents.shape), "children": tuple(children.shape)})

    if parent_assignments is None:
        if child_assignments is None:
            msg = "online_codebook_adaptation needs at least parent assignments"
        else:
            msg = "either both parent and child assignments or just parent must be supplied"
        raise CodebookError(msg)

    H = parents.shape[0]
    M0 = parents.shape[1]
    D = parents.shape[2]
    C = children.shape[2]
    B, _, N = parent_assignments.shape

    bh = B * H
    parent_index = parent_assignments.reshape(bh, N).long()
    if child_assignments is not None:
        child_index = child_assignments.reshape(bh, N).long()
    else:
        child_index = torch.zeros_like(parent_index)

    keys_flat = keys.reshape(bh, N, D).to(parents.dtype)

    # Tile ``children`` over the batch so we can scatter-add in
    # [bh, M_0, C, D].
    children_bh = children.repeat_interleave(B, dim=0)  # [bh, M_0, C, D]

    # Per-(parent, child) scatter-mean: m_{p,c} = sum k_j / n_{p,c}.
    flat_pc_index = parent_index * C + child_index  # [bh, N]
    one_hot_pc = torch.nn.functional.one_hot(flat_pc_index, num_classes=M0 * C).to(children.dtype)
    sum_keys_per_pc = torch.einsum("bnc,bnd->bcd", one_hot_pc, keys_flat)  # [bh, M_0 * C, D]
    count_per_pc = one_hot_pc.sum(dim=1)  # [bh, M_0 * C]
    inv_pc_count = torch.where(
        count_per_pc > 0,
        count_per_pc.reciprocal(),
        torch.zeros_like(count_per_pc),
    )
    mean_per_pc = sum_keys_per_pc * inv_pc_count.unsqueeze(-1)
    mean_per_pc_view = mean_per_pc.view(bh, M0, C, D)

    # Skip the EMA when a cell received no keys (otherwise the EMA
    # would shrink the child toward zero).
    has_pc_bh = (count_per_pc > 0).view(bh, M0, C)
    update_pc_weight = ((1.0 - decay) * has_pc_bh).to(children_bh.dtype).unsqueeze(-1)
    new_children_bh = children_bh + update_pc_weight * (mean_per_pc_view - children_bh)
    new_children_per_bh = new_children_bh.view(B, H, M0, C, D).mean(dim=0)
    children.copy_(
        torch.where(
            has_pc_bh.view(B, H, M0, C).any(dim=0).unsqueeze(-1),
            new_children_per_bh,
            children,
        )
    )

    # SPEC §7.9 invariant: parents = mean(children) at every step.
    parents.copy_(children.mean(dim=2))


__all__ = ["online_codebook_adaptation"]
