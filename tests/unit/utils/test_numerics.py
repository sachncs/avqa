"""Tests for avqa.utils.numerics module."""

from __future__ import annotations

import torch

from avqa.utils.numerics import online_softmax_step


def naive_merge(
    m_old: torch.Tensor,
    l_old: torch.Tensor,
    acc_old: torch.Tensor,
    m_new: torch.Tensor,
    l_new: torch.Tensor,
    acc_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive reference implementation used to verify online_softmax_step."""
    m = torch.maximum(m_old, m_new)
    alpha = torch.exp(m_old - m)
    beta = torch.exp(m_new - m)
    denom = alpha * l_old + beta * l_new
    acc = alpha.unsqueeze(-1) * acc_old + beta.unsqueeze(-1) * acc_new
    return m, denom, acc


class TestOnlineSoftmaxStep:
    """Tests for online_softmax_step merge operation."""

    def test_shapes_preserved(self) -> None:
        """Output shapes match input shapes."""
        m_old = torch.zeros(4)
        l_old = torch.zeros(4)
        acc_old = torch.zeros(4, 8)
        m_new = torch.zeros(4)
        l_new = torch.zeros(4)
        acc_new = torch.zeros(4, 8)
        m, denom, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        assert m.shape == m_old.shape
        assert denom.shape == l_old.shape
        assert acc.shape == acc_old.shape

    def test_identical_to_naive_reference(self) -> None:
        """Online merge equals the naive two-tile softmax."""
        torch.manual_seed(0)
        m_old = torch.randn(8) * 2
        l_old = torch.rand(8) + 0.1
        acc_old = torch.randn(8, 16)
        m_new = torch.randn(8) * 2
        l_new = torch.rand(8) + 0.1
        acc_new = torch.randn(8, 16)
        m, denom, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        m_ref, l_ref, acc_ref = naive_merge(m_old, l_old, acc_old, m_new, l_new, acc_new)
        assert torch.allclose(m, m_ref)
        assert torch.allclose(denom, l_ref)
        assert torch.allclose(acc, acc_ref)

    def test_max_is_max_of_both(self) -> None:
        """Output max equals max(m_old, m_new)."""
        m_old = torch.tensor([1.0, 5.0])
        l_old = torch.tensor([0.5, 0.5])
        acc_old = torch.tensor([[0.5], [0.5]])
        m_new = torch.tensor([3.0, 2.0])
        l_new = torch.tensor([0.5, 0.5])
        acc_new = torch.tensor([[0.5], [0.5]])
        m, _, _ = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        assert torch.equal(m, torch.tensor([3.0, 5.0]))

    def test_renormalization_invariant(self) -> None:
        """When acc = l * output, the merged l equals sum-of-exp across tiles."""
        torch.manual_seed(1)
        # Tile 1: scores x1, value v1
        x1 = torch.tensor([[1.0, 2.0, 3.0]])
        v1 = torch.tensor([[10.0, 20.0, 30.0]])
        m1 = x1.max(dim=-1, keepdim=True).values
        l1 = torch.exp(x1 - m1).sum(dim=-1)
        acc1 = (torch.exp(x1 - m1).unsqueeze(-1) * v1.unsqueeze(-2)).sum(dim=-2)

        # Tile 2: scores x2, value v2 — same total count as tile 1 by padding
        x2 = torch.tensor([[0.5, 4.0, 0.0]])
        v2 = torch.tensor([[5.0, 40.0, 0.0]])
        m2 = x2.max(dim=-1, keepdim=True).values
        l2 = torch.exp(x2 - m2).sum(dim=-1)
        acc2 = (torch.exp(x2 - m2).unsqueeze(-1) * v2.unsqueeze(-2)).sum(dim=-2)

        _m, denom, acc = online_softmax_step(m1.squeeze(-1), l1, acc1, m2.squeeze(-1), l2, acc2)
        # Reference: compute the global softmax output and check acc/l matches
        x_full = torch.cat([x1, x2], dim=-1)
        m_full = x_full.max(dim=-1, keepdim=True).values
        l_ref = torch.exp(x1 - m_full).sum(dim=-1) + torch.exp(x2 - m_full).sum(dim=-1)
        assert torch.allclose(denom, l_ref, atol=1e-5)
        acc_ref = (torch.exp(x1 - m_full).unsqueeze(-1) * v1.unsqueeze(-2)).sum(dim=-2) + (
            torch.exp(x2 - m_full).unsqueeze(-1) * v2.unsqueeze(-2)
        ).sum(dim=-2)
        assert torch.allclose(acc, acc_ref, atol=1e-5)

    def test_empty_tile_zero_contribution(self) -> None:
        """Empty tile (l=0, acc=0) does not perturb the running state."""
        m_old = torch.tensor([2.0])
        l_old = torch.tensor([1.5])
        acc_old = torch.tensor([[3.0]])
        m_new = torch.tensor([float("-inf")])
        l_new = torch.tensor([0.0])
        acc_new = torch.tensor([[0.0]])
        m, denom, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        # With m_new = -inf, exp(m_new - m) = 0, so output == old state
        assert torch.allclose(m, m_old)
        assert torch.allclose(denom, l_old)
        assert torch.allclose(acc, acc_old)

    def test_batched_shapes(self) -> None:
        """Works for batched (B, H, T) shapes."""
        m_old = torch.zeros(2, 4, 16)
        l_old = torch.ones(2, 4, 16)
        acc_old = torch.ones(2, 4, 16, 32)
        m_new = torch.zeros(2, 4, 16)
        l_new = torch.ones(2, 4, 16)
        acc_new = torch.ones(2, 4, 16, 32)
        m, denom, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        assert m.shape == (2, 4, 16)
        assert denom.shape == (2, 4, 16)
        assert acc.shape == (2, 4, 16, 32)

    def test_extreme_values_no_overflow(self) -> None:
        """Large differences between old and new max do not produce NaN/Inf."""
        m_old = torch.tensor([1000.0])
        l_old = torch.tensor([1.0])
        acc_old = torch.tensor([[1.0]])
        m_new = torch.tensor([-1000.0])
        l_new = torch.tensor([1.0])
        acc_new = torch.tensor([[1.0]])
        m, denom, acc = online_softmax_step(m_old, l_old, acc_old, m_new, l_new, acc_new)
        assert torch.isfinite(m).all()
        assert torch.isfinite(denom).all()
        assert torch.isfinite(acc).all()
