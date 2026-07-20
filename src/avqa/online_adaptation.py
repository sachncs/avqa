"""Online codebook adaptation (BCAR, OPT-0003).

Bias-Corrected Online Codebook Adaptation extends AVQ-Attention with
an inference-time EMA update of the hierarchical codebook. The
update is mathematically the same per-codeword mean estimator that
the paper uses offline (SPEC §8.9), but is applied to the
deployment stream rather than the training corpus.

Per parent codeword ``p``:

    m_p   = sum_{j : a(j) = p} k_j / max(1, n_p)
    C_p'  = decay * C_p + (1 - decay) * m_p

Per child codeword ``{p, c}``:

    m_{p,c} = sum_{j : (a(j), a_c(j)) = (p, c)} v_j / max(1, n_{p,c})
    C_{p,c}' = decay * C_{p,c} + (1 - decay) * m_{p,c}

After the child EMA we reproject the parent

    C_p <- mean_c C_{p,c}'

so the parent-child mean constraint (SPEC §7.9) is preserved at
every step.

The EMA is summed across the batch and sequence dimensions in one
fused scatter-add so that every key contributes exactly once.

References
----------

- AVQ-Attention paper (§8.9): offline EMA training of the codebook.
- Bottou & Bengio (1994): stochastic K-means convergence rate.
"""
from __future__ import annotations



import torch

from avqa.logging import get_logger

_logger = get_logger("bcar")


def online_codebook_adaptation(
    keys: torch.Tensor,
    *,
    parents: torch.Tensor,
    children: torch.Tensor,
    parent_assignments: torch.Tensor,
    child_assignments: torch.Tensor,
    decay: float = 0.99,
) -> None:
    """Apply one EMA update of the parents/children codebook in place.

    Args:
        keys: ``[B, H, N, D]`` keys that produced the assignments.
        parents: ``[H, M_0, D]`` parent codebook; modified in place.
        children: ``[H, M_0, C, D]`` child codebook; modified in place.
        parent_assignments: ``[B, H, N]`` int64 parent indices.
        child_assignments: ``[B, H, N]`` int64 child indices.
        decay: EMA decay in ``[0, 1)``. ``(1 - decay)`` is the
            contribution of the new mean.

    Returns:
        None. ``parents`` and ``children`` are mutated in place.

    Note:
        The keys argument is the ``keys`` tensor passed to the
        attention pipeline; we use one tensor for both parent and
        child assignment updates to keep the implementation concise.
    """
    if decay < 0.0 or decay >= 1.0:
        msg = f"decay must be in [0, 1), got {decay}"
        raise ValueError(msg)
    if parents.dim() != 3 or children.dim() != 4:
        msg = f"unexpected shapes: parents {tuple(parents.shape)}, children {tuple(children.shape)}"
        raise ValueError(msg)

    if parent_assignments is None:
        if child_assignments is None:
            msg = "online_codebook_adaptation needs at least parent assignments"
            raise ValueError(msg)
        msg = "either both parent and child assignments or just parent must be supplied"
        raise ValueError(msg)

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

    # Tile ``parents`` and ``children`` over the batch dimension so we can
    # operate in [bh, M_0, D] / [bh, M_0, C, D] without squeezing.
    parents_bh = parents.repeat_interleave(B, dim=0)  # [bh, M_0, D]
    children_bh = children.repeat_interleave(B, dim=0)  # [bh, M_0, C, D]  # fmt: skip

    # Per-parent scatter-mean: m_p = sum_{j : a(j) = p} k_j / n_p.
    one_hot_parent = torch.nn.functional.one_hot(parent_index, num_classes=M0).to(parents.dtype)
    sum_keys_per_parent = torch.einsum("bnm,bnd->bmd", one_hot_parent, keys_flat)
    count_per_parent = one_hot_parent.sum(dim=1)  # [bh, M_0]

    # Avoid division-by-zero; empty parents get no update (weight 0).
    inv_count = torch.where(
        count_per_parent > 0,
        count_per_parent.reciprocal(),
        torch.zeros_like(count_per_parent),
    )
    mean_per_parent = sum_keys_per_parent * inv_count.unsqueeze(-1)  # [bh, M_0, D]

    # EMA parents (per (b, h)) only when there is at least one
    # assignment; empty parents keep their existing value.
    has_per_bh = (count_per_parent > 0).view(B, H, M0).any(dim=0)  # [H, M_0]
    update_weight_bh = ((1.0 - decay) * (count_per_parent > 0)).to(parents_bh.dtype).unsqueeze(-1)
    new_parents_bh = parents_bh + update_weight_bh * (mean_per_parent - parents_bh)
    agg_parents = new_parents_bh.view(B, H, M0, D).mean(dim=0)
    parents.copy_(torch.where(has_per_bh.unsqueeze(-1), agg_parents, parents))

    # Online EMA on CHILDREN only; parents are derived at the end so
    # the mean constraint SPEC §7.9 is preserved exactly.
    # Per-(parent, child) update.
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
    # Skip the EMA when the (parent, child, batch) cell received no
    # keys (otherwise ``mean_per_pc_view = 0`` and the EMA shrinks the
    # child toward zero — the "untouched" parents in the empty-cell
    # test catch this drift).
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

    # Reproject: parents = mean(children) — SPEC §7.9 invariant.
    parents.copy_(children.mean(dim=2))


__all__ = ["online_codebook_adaptation"]
