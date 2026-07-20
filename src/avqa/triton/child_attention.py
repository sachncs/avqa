"""Triton child attention kernel (SPEC §11.6)."""
from __future__ import annotations

import torch

from avqa.logging import get_logger

_logger = get_logger("triton.child_attention")


@torch.no_grad()
def child_attention(
    query: torch.Tensor,
    children: torch.Tensor,
    selected_indices: torch.Tensor,
    child_aggregates: torch.Tensor,
    child_counts: torch.Tensor,
    *,
    block_t: int = 64,
) -> dict[str, torch.Tensor]:
    """Recompute attention only for selected parents' children.

    Args:
        query: ``[B, H, T_q, D]``.
        children: ``[H, M_0, C, D]``.
        selected_indices: ``[B, H, P]`` (int32/int64).
        child_aggregates: ``[B, H, M_0, C, D_v]``.
        child_counts: ``[B, H, M_0, C]``.
        block_t: Query-row tile size.

    Returns:
        Dictionary with ``child_logits`` (``[B, H, T_q, P, C]``) and
        ``child_running_state``.
    """

    import triton
    import triton.language as tl

    B, H, T_q, D = query.shape
    P = selected_indices.shape[-1]
    C = children.shape[2]
    D_v = child_aggregates.shape[-1]
    device = query.device

    child_logits = torch.empty(B, H, T_q, P, C, device=device, dtype=query.dtype)
    running_max = torch.full((B, H, T_q), float("-inf"), device=device, dtype=query.dtype)

    children_b = children.unsqueeze(0).expand(B, H, *children.shape[1:]).contiguous()
    child_aggregates_b = child_aggregates.contiguous()

    @triton.jit  # type: ignore[misc]
    def child_kernel(
        query_ptr,
        children_ptr,
        selected_ptr,
        cagg_ptr,
        cc_ptr,
        out_logits_ptr,
        out_state_ptr,
        P,
        T,
        D: tl.constexpr,
        C: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_T: tl.constexpr,
        scale: tl.constexpr,
    ) -> None:
        bh = tl.program_id(0)
        p = tl.program_id(1)
        t_off = tl.arange(0, BLOCK_T)
        t_mask = t_off < T
        sel = tl.load(selected_ptr + bh * P + p).to(tl.int32)

        # Load query tile [BLOCK_T, D].
        q = tl.load(
            query_ptr + bh * T * D + t_off[:, None] * D + tl.arange(0, D),
            mask=t_mask[:, None],
            other=0.0,
        )
        # Load selected parent's children [C, D].
        c = tl.load(
            children_ptr
            + bh * (P * C * D)
            + sel * (C * D)
            + tl.arange(0, C)[:, None] * D
            + tl.arange(0, D)[None, :],
        )
        logits = tl.dot(q, c.trans(1, 0), allow_tf32=False) * scale
        # Mask empty children.
        cc = tl.load(cc_ptr + bh * (P * C) + sel * C + tl.arange(0, C))
        logits = tl.where(tl.arange(0, C)[None, :] < C, logits, float("-inf"))
        logits = tl.where((cc > 0.0)[None, :], logits, float("-inf"))

        # Store logits [BLOCK_T, C] at output offset.
        for i in range(C):
            tl.store(
                out_logits_ptr + bh * (T * P * C) + t_off * (P * C) + p * C + i,
                logits[:, i],
                mask=t_mask,
            )

    scale: float = 1.0 / (D**0.5)
    grid = (B * H, P)
    child_kernel[grid](
        query,
        children_b,
        selected_indices.to(torch.int32),
        child_aggregates_b,
        child_counts,
        child_logits,
        running_max,
        P,
        T_q,
        D=D,
        C=C,
        DV=D_v,
        BLOCK_T=block_t,
        scale=scale,
    )

    return {
        "child_logits": child_logits,
        "running_state": running_max,
    }


__all__ = ["child_attention"]
