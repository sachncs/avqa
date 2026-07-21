"""Online-softmax parent attention Triton kernel (SPEC §11.5).

This kernel materialises one query tile of size ``BLOCK_T`` and
computes attention against the parent codebook in shared memory,
maintaining FlashAttention-2 running accumulators (max, denom, num).

Signature and contract: SPEC §11.5.
"""
from __future__ import annotations

import torch

from avqa.logging import get_logger

logger = get_logger("triton.parent_attention")


@torch.no_grad()
def parent_attention(
    query: torch.Tensor,
    parents: torch.Tensor,
    parent_values: torch.Tensor,
    parent_counts: torch.Tensor,
    *,
    block_t: int = 64,
    block_m: int = 64,
) -> dict[str, torch.Tensor]:
    """Online-softmax tiled parent attention.

    Args:
        query: ``[B, H, T_q, D]``.
        parents: ``[H, M_0, D]``.
        parent_values: ``[B, H, M_0, D_v]``.
        parent_counts: ``[B, H, M_0]`` (zero-where-empty).
        block_t: Query-row tile size.
        block_m: Codebook tile size.

    Returns:
        Dictionary with ``parent_attention_probs``, ``running_state``.
    """
    try:
        import triton as _triton_module
        import triton.language as _tl
    except ImportError as exc:
        msg = 'triton is not installed'
        raise RuntimeError(msg) from exc

    B, H, T_q, D = query.shape
    M0 = parents.shape[1]
    D_v = parent_values.shape[-1]

    probs = torch.empty(B, H, T_q, M0, device=query.device, dtype=query.dtype)
    state_max = torch.full((B, H, T_q), float("-inf"), device=query.device, dtype=query.dtype)
    state_denom = torch.zeros(B, H, T_q, device=query.device, dtype=query.dtype)
    state_num = torch.zeros(B, H, T_q, D_v, device=query.device, dtype=query.dtype)

    parents_b = parents.unsqueeze(0).expand(B, H, M0, D).contiguous()
    parent_values_b = parent_values.contiguous()

    @_triton_module.jit
    def parent_kernel(
        query_ptr,
        parents_ptr,
        pv_ptr,
        pc_ptr,
        probs_ptr,
        state_max_ptr,
        state_denom_ptr,
        state_num_ptr,
        T,
        M0,
        D: _tl.constexpr,
        DV: _tl.constexpr,
        BLOCK_T: _tl.constexpr,
        BLOCK_M: _tl.constexpr,
        scale: _tl.constexpr,
    ) -> None:
        """Parent-attention kernel: Q·C_p for the parent codebook (SPEC §11.5)."""
        bh = _tl.program_id(0)
        t_start = _tl.program_id(1)
        t_off = t_start + _tl.arange(0, BLOCK_T)
        t_mask = t_off < T
        m_off = _tl.arange(0, BLOCK_M)
        m_mask = m_off < M0

        q = _tl.load(
            query_ptr + bh * T * D + t_off[:, None] * D + _tl.arange(0, D),
            mask=t_mask[:, None],
            other=0.0,
        )  # [BLOCK_T, D]
        p = _tl.load(
            parents_ptr + bh * M0 * D + m_off[None, :] * D + _tl.arange(0, D),
            mask=m_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, D]
        logits = _tl.dot(q, p.trans(1, 0), allow_tf32=False) * scale
        # Mask empty parents to -inf.
        pc = _tl.load(pc_ptr + bh * M0 + m_off, mask=m_mask, other=0.0)
        logits = _tl.where(m_mask[None, :] & (pc > 0.0), logits, float("-inf"))
        # Numeric stable softmax.
        m_tile = _tl.max(logits, axis=1)
        m_old = _tl.load(state_max_ptr + bh * T + t_off, mask=t_mask, other=float("-inf"))
        m_new = _tl.maximum(m_old, m_tile)
        alpha = _tl.exp(m_old - m_new)
        exp_logits = _tl.exp(logits - m_new[:, None])
        exp_logits = _tl.where(m_mask[None, :], exp_logits, 0.0)
        # Update denom using counts scaling (SPEC §7.7 VQ-denominator).
        pc_b = pc[None, :].to(_tl.float32)
        denom_tile = _tl.sum(exp_logits * pc_b, axis=1)
        d_old = _tl.load(state_denom_ptr + bh * T + t_off, mask=t_mask, other=0.0)
        d_new = alpha * d_old + denom_tile
        _tl.store(state_max_ptr + bh * T + t_off, m_new, mask=t_mask)
        _tl.store(state_denom_ptr + bh * T + t_off, d_new, mask=t_mask)

        # Numerator: contract exp_logits over M_0 with parent values.
        pv = _tl.load(
            pv_ptr + bh * M0 * DV + m_off[:, None] * DV + _tl.arange(0, DV),
            mask=m_mask[:, None],
            other=0.0,
        )  # [BLOCK_M, DV]
        # Build diag-weighted update via outer product.
        contrib = _tl.dot(exp_logits.to(pv.dtype), pv, allow_tf32=False)  # [BLOCK_T, DV]
        n_old = _tl.load(
            state_num_ptr + bh * T * DV + t_off[:, None] * DV + _tl.arange(0, DV),
            mask=t_mask[:, None],
            other=0.0,
        )
        n_new = alpha[:, None] * n_old + contrib
        _tl.store(
            state_num_ptr + bh * T * DV + t_off[:, None] * DV + _tl.arange(0, DV),
            n_new,
            mask=t_mask[:, None],
        )

        # Store attention probabilities for the slicing read by the
        # routing module (computed in fp32 for tie stability then cast).
        probs_chunk = exp_logits / _tl.sum(exp_logits, axis=1, keep_dims=True).clamp_min(1e-12)
        # Write only the valid M_0 range.
        for i in range(BLOCK_M):
            _tl.store(
                probs_ptr + bh * T * M0 + t_off * M0 + m_off[i],
                probs_chunk[:, i],
                mask=t_mask & (m_off[i] < M0),
            )

    scale: float = 1.0 / (D**0.5)
    grid = (B * H, _triton_module.cdiv(T_q, block_t))
    parent_kernel[grid](
        query,
        parents_b,
        parent_values_b,
        parent_counts,
        probs,
        state_max,
        state_denom,
        state_num,
        T_q,
        M0,
        D=D,
        DV=D_v,
        BLOCK_T=block_t,
        BLOCK_M=block_m,
        scale=scale,
    )

    return {
        "parent_attention_probs": probs,
        "state_max": state_max,
        "state_denom": state_denom,
        "state_num": state_num,
    }


__all__ = ["parent_attention"]
