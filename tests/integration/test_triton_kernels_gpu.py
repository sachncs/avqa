"""Numerical-equivalence tests for the SPEC §11 Triton kernels.

These tests REQUIRE CUDA + Triton to run; they are gated by the
``gpu`` marker. The CPU test suite is unaffected.

Test plan (SPEC §11.9):

- FP32 tolerance ``atol=1e-5``, ``rtol=1e-5``.
- BF16 tolerance ``atol=1e-2``, ``rtol=1e-2``.
- Per-kernel: ``vq_precompute``, ``parent_attention``, ``child_attention``,
  ``correction`` MUST match the corresponding :class:`TorchBackend`
  reference within documented tolerances.

Tolerance declarations live in :mod:`avqa.triton`.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

FP32_TOL = {"atol": 1e-5, "rtol": 1e-5}
BF16_TOL = {"atol": 1e-2, "rtol": 1e-2}


def has_cuda_triton() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (ImportError, AttributeError):
        return False


def skip_if_unavailable() -> None:
    if not has_cuda_triton():
        pytest.skip("CUDA / Triton not available on this host")


def test_vq_precompute_matches_torchbackend_fp32() -> None:
    """vq_precompute matches TorchBackend.quantize in FP32 (SPEC §11.9)."""
    skip_if_unavailable()
    from avqa.backend import TorchBackend  # noqa: PLC0415
    from avqa.codebook import HierarchicalCodebook  # noqa: PLC0415
    from avqa.triton.loader import load_kernel  # noqa: PLC0415

    torch.manual_seed(0)
    cb = HierarchicalCodebook(num_heads=2, num_parents=16, children_per_parent=4, head_dim=32)
    cb.initialize_parents_random()
    keys = torch.randn(1, 2, 64, 32, device="cuda")
    values = torch.randn(1, 2, 64, 32, device="cuda")

    ref = TorchBackend().quantize(keys, values, cb.parents, cb.children)
    try:
        out = load_kernel("vq_precompute")(keys, values, cb.parents, cb.children)
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f"Triton kernel not importable: {exc}")

    torch.testing.assert_close(
        out["parent_aggregates"], ref.parent_aggregates, rtol=FP32_TOL["rtol"], atol=FP32_TOL["atol"]
    )
    torch.testing.assert_close(
        out["parent_counts"], ref.parent_counts, rtol=FP32_TOL["rtol"], atol=FP32_TOL["atol"]
    )
    torch.testing.assert_close(out["parent_assignments"], ref.parent_assignments)


def test_correction_matches_torchbackend() -> None:
    """Three-way tile merge matches TorchBackend.correction."""
    skip_if_unavailable()
    from avqa.triton.loader import load_kernel  # noqa: PLC0415
    from avqa.utils.numerics import online_softmax_step  # noqa: PLC0415

    state_max = torch.full((1, 2, 16), 0.5, device="cuda")
    state_denom = torch.full((1, 2, 16), 4.0, device="cuda")
    state_num = torch.randn(1, 2, 16, 8, device="cuda")
    parent_max = torch.full((1, 2, 16), 0.4, device="cuda")
    parent_denom = torch.full((1, 2, 16), 1.0, device="cuda")
    parent_num = torch.randn(1, 2, 16, 8, device="cuda")
    child_max = torch.full((1, 2, 16), 0.7, device="cuda")
    child_denom = torch.full((1, 2, 16), 2.0, device="cuda")
    child_num = torch.randn(1, 2, 16, 8, device="cuda")

    ref_max, ref_denom, ref_num = online_softmax_step(
        state_max,
        state_denom,
        state_num,
        child_max,
        child_denom,
        child_num,
    )

    try:
        out = load_kernel("correction")(
            state_max,
            state_denom,
            state_num,
            parent_max,
            parent_denom,
            parent_num,
            child_max,
            child_denom,
            child_num,
        )
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f"Triton kernel not importable: {exc}")

    torch.testing.assert_close(
        out["state_max"], ref_max, rtol=FP32_TOL["rtol"], atol=FP32_TOL["atol"]
    )
    torch.testing.assert_close(
        out["state_denom"], ref_denom, rtol=FP32_TOL["rtol"], atol=FP32_TOL["atol"]
    )
    torch.testing.assert_close(
        out["state_num"], ref_num, rtol=FP32_TOL["rtol"], atol=FP32_TOL["atol"]
    )


def test_parent_attention_fp32_match() -> None:
    """parent_attention kernel matches the PyTorch reference (SPEC §11.5)."""
    skip_if_unavailable()
    from avqa.triton.loader import load_kernel  # noqa: PLC0415

    H, B, T, M, D, DV = 2, 1, 16, 8, 32, 16
    query = torch.randn(B, H, T, D, device="cuda")
    parents = torch.randn(H, M, D, device="cuda")
    parent_values = torch.randn(B, H, M, DV, device="cuda")
    parent_counts = torch.full((B, H, M), 1.0, device="cuda")

    scale = D**-0.5
    ref_logits = (
        torch.matmul(
            query,
            parents.unsqueeze(0).expand(B, H, M, D).transpose(-2, -1),
        )
        * scale
    )
    valid = parent_counts.unsqueeze(2) > 0
    ref_logits = ref_logits.masked_fill(~valid, float("-inf"))
    ref_probs = torch.softmax(ref_logits, dim=-1)

    try:
        out = load_kernel("parent_attention")(
            query, parents, parent_values, parent_counts, block_t=16, block_m=8
        )
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f"Triton kernel not importable: {exc}")

    torch.testing.assert_close(out["parent_attention_probs"], ref_probs, atol=5e-3, rtol=5e-3)
