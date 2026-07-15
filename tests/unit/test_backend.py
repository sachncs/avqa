"""Tests for avqa.backend module."""

from __future__ import annotations

import pytest
import torch

from avqa.backend import Backend, TorchBackend, TritonBackend, create_backend
from avqa.merge import MergeInputs, ProbabilityMerge


class TestNaiveAttention:
    """Tests for the naive O(N^2) attention path."""

    def test_shape(self) -> None:
        """Output has shape [B, H, T, D_v]."""
        torch.manual_seed(0)
        Q = torch.randn(1, 2, 4, 8)
        K = torch.randn(1, 2, 6, 8)
        V = torch.randn(1, 2, 6, 16)
        out = TorchBackend().naive_attention(Q, K, V)
        assert out.shape == (1, 2, 4, 16)

    def test_matches_manual(self) -> None:
        """Naive attention equals manual softmax(QK^T / sqrt(d)) V."""
        torch.manual_seed(0)
        Q = torch.randn(1, 1, 3, 4)
        K = torch.randn(1, 1, 5, 4)
        V = torch.randn(1, 1, 5, 8)
        scale = 4 ** -0.5
        expected = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) @ V
        out = TorchBackend().naive_attention(Q, K, V)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_mask_applied(self) -> None:
        """Mask positions are excluded from attention."""
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 3, 4)
        V = torch.randn(1, 1, 3, 4)
        mask = torch.tensor([[[1, 1, 0]]])                  # [B, H, N]
        out = TorchBackend().naive_attention(Q, K, V, mask=mask)
        # Last key/value is masked out.
        K_masked = K[:, :, :2, :]
        V_masked = V[:, :, :2, :]
        expected = TorchBackend().naive_attention(Q, K_masked, V_masked)
        assert torch.allclose(out, expected, atol=1e-5)


class TestOnlineSoftmaxAttention:
    """Tests for the tiled online-softmax attention path."""

    def test_matches_naive(self) -> None:
        """Online-softmax attention equals naive attention within tolerance."""
        torch.manual_seed(0)
        Q = torch.randn(1, 1, 4, 8)
        K = torch.randn(1, 1, 8, 8)
        V = torch.randn(1, 1, 8, 16)
        backend = TorchBackend()
        out_naive = backend.naive_attention(Q, K, V)
        out_online = backend.online_softmax_attention(Q, K, V, block_size=4)
        assert torch.allclose(out_online, out_naive, atol=1e-4)

    def test_block_size_one(self) -> None:
        """block_size=1 (most granular tile) still agrees with naive."""
        torch.manual_seed(0)
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 3, 4)
        V = torch.randn(1, 1, 3, 4)
        backend = TorchBackend()
        out_naive = backend.naive_attention(Q, K, V)
        out_online = backend.online_softmax_attention(Q, K, V, block_size=1)
        assert torch.allclose(out_online, out_naive, atol=1e-4)

    def test_block_size_larger_than_seq(self) -> None:
        """Block size larger than sequence still works."""
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 3, 4)
        V = torch.randn(1, 1, 3, 4)
        backend = TorchBackend()
        out_online = backend.online_softmax_attention(Q, K, V, block_size=10)
        out_naive = backend.naive_attention(Q, K, V)
        assert torch.allclose(out_online, out_naive, atol=1e-5)

    def test_with_mask(self) -> None:
        """Masked online-softmax attention matches naive masked attention."""
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 4, 4)
        V = torch.randn(1, 1, 4, 4)
        mask = torch.tensor([[[1, 1, 0, 1]]])
        backend = TorchBackend()
        out_naive = backend.naive_attention(Q, K, V, mask=mask)
        out_online = backend.online_softmax_attention(Q, K, V, block_size=2, mask=mask)
        assert torch.allclose(out_online, out_naive, atol=1e-4)


class TestQuantize:
    """Tests for the quantize() backend method."""

    def test_returns_quantization_result(self) -> None:
        """quantize() delegates to EuclideanHierarchicalQuantizer."""
        from avqa.quantizer import QuantizationResult  # noqa: PLC0415

        torch.manual_seed(0)
        keys = torch.randn(1, 1, 8, 4)
        values = torch.randn(1, 1, 8, 4)
        parents = torch.randn(1, 2, 4)
        children = torch.randn(1, 2, 4, 4)
        result = TorchBackend().quantize(keys, values, parents, children)
        assert isinstance(result, QuantizationResult)
        assert result.parent_assignments.shape == (1, 1, 8)


class TestMerge:
    """Tests for the merge() backend method."""

    def test_default_merge_is_probability(self) -> None:
        """merge() delegates to ProbabilityMerge."""
        torch.manual_seed(0)
        parent_probs = torch.softmax(torch.randn(1, 1, 4, 1), dim=-2)
        parent_value = torch.randn(1, 1, 4, 8)
        child_probs = torch.softmax(torch.randn(1, 1, 4, 3), dim=-1)
        child_value = torch.randn(1, 1, 4, 3, 8)
        inputs = MergeInputs(
            parent_probs=parent_probs,
            parent_value=parent_value,
            child_probs=child_probs,
            child_value=child_value,
        )
        backend_out = TorchBackend().merge(inputs)
        ref_out = ProbabilityMerge().merge(inputs)
        assert torch.allclose(backend_out, ref_out)


class TestCorrection:
    """Tests for the correction() backend method."""

    def test_matches_numerics_helper(self) -> None:
        """backend.correction agrees with avqa.utils.numerics.online_softmax_step."""
        from avqa.utils.numerics import online_softmax_step  # noqa: PLC0415

        m_old = torch.zeros(2)
        l_old = torch.ones(2)
        acc_old = torch.randn(2, 4)
        m_new = torch.ones(2)
        l_new = torch.ones(2)
        acc_new = torch.randn(2, 4)
        m, denom, acc = TorchBackend().correction(
            m_old, l_old, acc_old, m_new, l_new, acc_new
        )
        m_ref, l_ref, acc_ref = online_softmax_step(
            m_old, l_old, acc_old, m_new, l_new, acc_new
        )
        assert torch.allclose(m, m_ref)
        assert torch.allclose(denom, l_ref)
        assert torch.allclose(acc, acc_ref)


class TestReduction:
    """Tests for the reduction() backend method."""

    def test_output_shape(self) -> None:
        """Output has shape matching num."""
        num = torch.randn(2, 4, 8)
        denom = torch.rand(2, 4) + 0.1
        out = TorchBackend().reduction(num, denom)
        assert out.shape == num.shape

    def test_division(self) -> None:
        """Output is num / denom (with denom broadcast over the last dim)."""
        num = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])           # [1, 2, 2]
        denom = torch.tensor([[2.0, 3.0]])                          # [1, 2]
        out = TorchBackend().reduction(num, denom)
        # Broadcasting denom[1,2] -> [1,2,1] across num[1,2,2].
        expected = torch.tensor([[[1.0, 2.0], [2.0, 8.0 / 3.0]]])
        assert torch.allclose(out, expected, atol=1e-5)


class TestTritonBackend:
    """Tests for the Triton backend (gated by availability)."""

    def test_is_available_returns_bool(self) -> None:
        """is_available returns a boolean."""
        assert isinstance(TritonBackend.is_available(), bool)

    def test_naive_falls_back_to_torch(self) -> None:
        """naive_attention delegates to TorchBackend."""
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 3, 4)
        V = torch.randn(1, 1, 3, 4)
        out_triton = TritonBackend().naive_attention(Q, K, V)
        out_torch = TorchBackend().naive_attention(Q, K, V)
        assert torch.allclose(out_triton, out_torch)

    def test_online_falls_back_to_torch(self) -> None:
        """online_softmax_attention delegates to TorchBackend."""
        Q = torch.randn(1, 1, 2, 4)
        K = torch.randn(1, 1, 3, 4)
        V = torch.randn(1, 1, 3, 4)
        out_triton = TritonBackend().online_softmax_attention(Q, K, V)
        out_torch = TorchBackend().online_softmax_attention(Q, K, V)
        assert torch.allclose(out_triton, out_torch, atol=1e-5)


class TestBackendFactory:
    """Tests for create_backend factory (spec §5.11)."""

    def test_create_torch(self) -> None:
        """create_backend('torch') returns a TorchBackend."""
        backend = create_backend("torch")
        assert isinstance(backend, TorchBackend)

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend name raises ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            create_backend("nonexistent_backend")

    def test_default_is_torch(self) -> None:
        """create_backend() defaults to 'torch'."""
        backend = create_backend()
        assert isinstance(backend, TorchBackend)


class TestAbstractInterface:
    """Tests for the abstract Backend interface."""

    def test_cannot_instantiate(self) -> None:
        """Backend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()  # type: ignore[abstract]

    def test_subclass_relationship(self) -> None:
        """Concrete backends inherit from Backend."""
        assert issubclass(TorchBackend, Backend)
        assert issubclass(TritonBackend, Backend)
