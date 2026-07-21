"""FlashAttention / xFormers interop numerical-equivalence tests.

Each test invokes :func:`avqa.integrations.flash_attention_interop` /
:func:`avqa.integrations.xformers_interop` and compares against a
reference PyTorch softmax(QK^T/sqrt(d)) V computation. Tests skip when
the optional dependency is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from avqa.integrations import flash_attention_interop, xformers_interop

pytestmark = pytest.mark.gpu


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: PLC0415

        return True
    except ImportError:
        return False


def has_xformers() -> bool:
    try:
        import xformers  # noqa: PLC0415

        return True
    except ImportError:
        return False


def skip_unless_flash_attn() -> None:
    if not has_flash_attn():
        pytest.skip("flash_attn not installed; install avqa[flash-attn] to run")


def skip_unless_xformers() -> None:
    if not has_xformers():
        pytest.skip("xformers not installed; install avqa[xformers] to run")


def test_flash_attention_interop_matches_reference() -> None:
    """FlashAttention wrapper matches the AVQA reference."""
    skip_unless_flash_attn()
    torch.manual_seed(0)
    q = torch.randn(2, 16, 4, 32, device="cuda")
    k = torch.randn(2, 16, 4, 32, device="cuda")
    v = torch.randn(2, 16, 4, 32, device="cuda")

    out = flash_attention_interop(q, k, v)
    scale = 32**-0.5
    ref_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    ref = torch.matmul(torch.softmax(ref_logits, dim=-1), v)

    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_xformers_interop_matches_reference() -> None:
    """xFormers wrapper matches the AVQA reference."""
    skip_unless_xformers()
    torch.manual_seed(0)
    q = torch.randn(2, 16, 4, 32, device="cuda")
    k = torch.randn(2, 16, 4, 32, device="cuda")
    v = torch.randn(2, 16, 4, 32, device="cuda")

    out = xformers_interop(q, k, v)
    scale = 32**-0.5
    ref_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    ref = torch.matmul(torch.softmax(ref_logits, dim=-1), v)

    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
