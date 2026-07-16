"""Fused VQ precompute Triton kernel (SPEC §11.4).

This module is imported lazily by :mod:`avqa.triton` only when the
runtime has both Triton and CUDA available. The kernel signature is
documented in SPEC §11.4.

Algorithmic summary:

1. Stream keys over the sequence dimension in tiles of ``BLOCK_N``.
2. For each tile, compute pairwise squared L2 distance to all parents.
3. ``argmin`` produces parent assignment.
4. Gather that parent's children, recompute pairwise distance.
5. ``argmin`` produces child assignment.
6. Scatter-add the value vector into both parent / child aggregates
   using ``tl.atomic_add`` (read-modify-write is fine because each
   (parent, child) cell is written exactly once per scatter iteration
   in the streaming order when ``BLOCK_N == 1``; we use atomic adds
   to be safe under larger ``BLOCK_N``).
7. Increment the count buffers with the same atomic-add machinery.
"""

from __future__ import annotations

import torch

from avqa.logging import get_logger

_logger = get_logger("triton.vq")


@torch.no_grad()
def vq_precompute(
    keys: torch.Tensor,
    values: torch.Tensor,
    parents: torch.Tensor,
    children: torch.Tensor,
    *,
    block_n: int = 128,
    block_m: int = 64,
) -> dict[str, torch.Tensor]:
    """Fused VQ precompute.

    Args:
        keys: ``[B, H, N, D]``.
        values: ``[B, H, N, D]``.
        parents: ``[H, M_0, D]``.
        children: ``[H, M_0, C, D]``.
        block_n: Per-step streaming tile size.
        block_m: Per-codebook tile size (must be a power of 2).

    Returns:
        Dictionary containing ``parent_assignments``, ``child_assignments``,
        ``parent_aggregates``, ``child_aggregates``, ``parent_counts``,
        ``child_counts``.
    """
    import triton
    import triton.language as tl

    B, H, N, D = keys.shape
    M0, C = parents.shape[1], children.shape[2]
    device = keys.device

    parent_assign = torch.empty(B, H, N, dtype=torch.int32, device=device)
    child_assign = torch.empty(B, H, N, dtype=torch.int32, device=device)
    parent_agg = torch.zeros(B, H, M0, D, device=device, dtype=values.dtype)
    child_agg = torch.zeros(B, H, M0, C, D, device=device, dtype=values.dtype)
    parent_counts = torch.zeros(B, H, M0, device=device, dtype=values.dtype)
    child_counts = torch.zeros(B, H, M0, C, device=device, dtype=values.dtype)

    # Broadcast parents / children to per-batch.
    parents_b = parents.unsqueeze(0).expand(B, H, M0, D).contiguous()
    children_b = children.unsqueeze(0).expand(B, H, M0, C, D).contiguous()

    @triton.jit  # type: ignore[misc]
    def vq_kernel(
        keys_ptr,
        values_ptr,
        parents_ptr,
        children_ptr,
        parent_assign_ptr,
        child_assign_ptr,
        parent_agg_ptr,
        child_agg_ptr,
        parent_counts_ptr,
        child_counts_ptr,
        N,
        M0,
        C,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ) -> None:
        bh = tl.program_id(0)
        n_start = tl.program_id(1)
        off = n_start + tl.arange(0, BLOCK_N)
        n_mask = off < N

        # Stage 1: distance to parents.
        m_offsets = tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M0
        # keys [BLOCK_N, D] @ parents^T [D, BLOCK_M] -> [BLOCK_N, BLOCK_M].
        k = tl.load(
            keys_ptr + bh * N * D + off[:, None] * D + tl.arange(0, D),
            mask=n_mask[:, None],
            other=0.0,
        )
        p = tl.load(
            parents_ptr + bh * M0 * D + m_offsets[None, :] * D + tl.arange(0, D),
            mask=m_mask[None, :],
            other=0.0,
        )
        k_sq = tl.sum(k * k, axis=1)  # [BLOCK_N]
        p_sq = tl.sum(p * p, axis=1)  # [BLOCK_M]
        dist_sq_p = k_sq[:, None] + p_sq[None, :] - 2.0 * tl.dot(k, p.trans(1, 0), allow_tf32=False)
        # Mask invalid parent slots so argmin ignores them.
        dist_sq_p = tl.where(m_mask[None, :], dist_sq_p, float("inf"))
        parent_idx = tl.argmin(dist_sq_p, axis=1)  # [BLOCK_N]

        # Store parent assignment.
        tl.store(parent_assign_ptr + bh * N + off, parent_idx, mask=n_mask)

        # Stage 2: gather children of assigned parents.
        # children is [M0, C, D]; we need each key's chosen parent.
        chosen_children = tl.load(
            children_ptr
            + bh * M0 * C * D
            + parent_idx[:, None, None] * (C * D)
            + tl.arange(0, C)[None, :, None] * D
            + tl.arange(0, D)[None, None, :],
            mask=n_mask[:, None, None],
            other=0.0,
        )  # [BLOCK_N, C, D]
        cc_sq = tl.sum(chosen_children * chosen_children, axis=2)  # [BLOCK_N, C]
        kk = k[:, None, :]  # [BLOCK_N, 1, D]
        cross_c = tl.sum(kk * chosen_children, axis=2)  # [BLOCK_N, C]
        dist_sq_c = k_sq[:, None] + cc_sq - 2.0 * cross_c
        # If C is not known statically fall back to runtime for the mask.
        dist_sq_c = tl.where(tl.arange(0, C)[None, :] < C, dist_sq_c, float("inf"))
        child_idx = tl.argmin(dist_sq_c, axis=1)  # [BLOCK_N]

        # Store child assignment.
        tl.store(child_assign_ptr + bh * N + off, child_idx, mask=n_mask)

        # Aggregate values to parents (atomic add).
        v = tl.load(
            values_ptr + bh * N * D + off[:, None] * D + tl.arange(0, D),
            mask=n_mask[:, None],
            other=0.0,
        )
        tl.atomic_add(
            parent_agg_ptr + bh * M0 * D + parent_idx[:, None] * D + tl.arange(0, D),
            v,
            mask=n_mask[:, None],
        )
        tl.atomic_add(
            parent_counts_ptr + bh * M0 + parent_idx,
            tl.where(n_mask, 1.0, 0.0),
            mask=n_mask,
        )
        flat_child = parent_idx * C + child_idx
        tl.atomic_add(
            child_agg_ptr + bh * M0 * C * D + flat_child[:, None] * D + tl.arange(0, D),
            v,
            mask=n_mask[:, None],
        )
        tl.atomic_add(
            child_counts_ptr + bh * M0 * C + flat_child,
            tl.where(n_mask, 1.0, 0.0),
            mask=n_mask,
        )

    grid = (B * H, triton.cdiv(N, block_n))
    vq_kernel[grid](
        keys,
        values,
        parents_b,
        children_b,
        parent_assign,
        child_assign,
        parent_agg,
        child_agg,
        parent_counts,
        child_counts,
        N,
        M0,
        C,
        D=D,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
    )

    return {
        "parent_assignments": parent_assign,
        "child_assignments": child_assign,
        "parent_aggregates": parent_agg,
        "child_aggregates": child_agg,
        "parent_counts": parent_counts,
        "child_counts": child_counts,
    }


__all__ = ["vq_precompute"]
