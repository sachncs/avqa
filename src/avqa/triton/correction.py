"""Triton correction kernel (SPEC §11.7).

Implements the FlashAttention-2 tile merge that combines the running
state with the corrected parent contribution, all without materialising
the full attention matrix.

Signature and contract: SPEC §11.7.
"""
from __future__ import annotations

import torch

from avqa.logging import get_logger

logger = get_logger("triton.correction")


@torch.no_grad()
def correction(
    state_max: torch.Tensor,
    state_denom: torch.Tensor,
    state_num: torch.Tensor,
    parent_max: torch.Tensor,
    parent_denom: torch.Tensor,
    parent_num: torch.Tensor,
    child_max: torch.Tensor,
    child_denom: torch.Tensor,
    child_num: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fused three-way tile merge (state - parent + child).

    All inputs share shape ``[B, H, T, 1]`` for the scalar accumulators
    and ``[B, H, T, 1, D_v]`` for the numerator tensors.

    Returns:
        Dictionary with the updated ``state_max``, ``state_denom``,
        ``state_num``.
    """
    try:
        import triton as _triton_module
        import triton.language as _tl
    except ImportError:
        return None  # type: ignore[return-value]

    B, H, T, _ = state_max.shape
    D_v = state_num.shape[-1]

    out_max = torch.empty_like(state_max)
    out_denom = torch.empty_like(state_denom)
    out_num = torch.empty_like(state_num)

    @_triton_module.jit
    def correction_kernel(
        sm_ptr,
        sd_ptr,
        sn_ptr,
        pm_ptr,
        pd_ptr,
        pn_ptr,
        cm_ptr,
        cd_ptr,
        cn_ptr,
        om_ptr,
        od_ptr,
        on_ptr,
        T,
        DV: _tl.constexpr,
    ) -> None:
        """Online-softmax tile-correction kernel (SPEC §11.7)."""
        bht = _tl.program_id(0)
        new_max = _tl.maximum(
            _tl.maximum(
                _tl.load(sm_ptr + bht),
                _tl.load(pm_ptr + bht),
            ),
            _tl.load(cm_ptr + bht),
        )
        s_old = _tl.exp(_tl.load(sm_ptr + bht) - new_max)
        s_rem = _tl.exp(_tl.load(pm_ptr + bht) - new_max)
        s_add = _tl.exp(_tl.load(cm_ptr + bht) - new_max)
        _tl.store(om_ptr + bht, new_max)

        denom = (
            _tl.load(sd_ptr + bht) * s_old
            - _tl.load(pd_ptr + bht) * s_rem
            + _tl.load(cd_ptr + bht) * s_add
        )
        _tl.store(od_ptr + bht, denom)

        d_off = _tl.arange(0, DV)
        n_old = _tl.load(sn_ptr + bht * DV + d_off)
        n_rem = _tl.load(pn_ptr + bht * DV + d_off)
        n_add = _tl.load(cn_ptr + bht * DV + d_off)
        n_new = n_old * s_old - n_rem * s_rem + n_add * s_add
        _tl.store(on_ptr + bht * DV + d_off, n_new)

    grid = (B * H * T,)
    correction_kernel[grid](
        state_max,
        state_denom,
        state_num,
        parent_max,
        parent_denom,
        parent_num,
        child_max,
        child_denom,
        child_num,
        out_max,
        out_denom,
        out_num,
        T,
        DV=D_v,
    )

    return {
        "state_max": out_max,
        "state_denom": out_denom,
        "state_num": out_num,
    }


__all__ = ["correction"]
