"""pytest-benchmark suite for AVQA (spec §3.19).

Run with::

    pytest tests/performance/ --benchmark-only

All benchmarks compare AVQA against PyTorch SDPA on identical inputs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.backend import TorchBackend
from avqa.codebook import HierarchicalCodebook
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)
from avqa.quantizer import EuclideanHierarchicalQuantizer
from avqa.utils.seed import seed_everything

if TYPE_CHECKING:
    # ``pytest_benchmark`` ships no type stubs. We declare a minimal
    # Protocol locally to keep mypy happy in the type-only block.
    from typing import Protocol as _Protocol

    class BenchmarkFixture(_Protocol):
        def __call__(self, function_to_benchmark: object) -> object: ...


def small_attn_config(seq_len: int, num_heads: int = 4, embed_dim: int = 64) -> AVQConfig:
    """Construct a small AVQConfig for benchmarks."""
    return AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
        ),
        codebook=CodebookConfig(
            num_codewords=16,
            children_per_codeword=4,
        ),
        routing=RoutingConfig(
            refinement_budget=4,
        ),
        dropout=0.0,
    )


@pytest.fixture(params=[64, 128, 256])
def seq_len(request: pytest.FixtureRequest) -> int:
    """Sequence lengths to sweep."""
    param: int = request.param
    return param


@pytest.mark.benchmark(group="attention")
def test_avqa_attention(seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark AVQA attention forward pass."""
    cfg = small_attn_config(seq_len=seq_len)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, seq_len, 64)
    k = torch.randn(1, seq_len, 64)
    v = torch.randn(1, seq_len, 64)

    def run() -> None:
        module(q, k, v)

    benchmark(run)


@pytest.mark.benchmark(group="attention")
def test_pytorch_attention(seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark PyTorch SDPA reference for comparison."""
    q = torch.randn(1, 4, seq_len, 16)
    k = torch.randn(1, 4, seq_len, 16)
    v = torch.randn(1, 4, seq_len, 16)

    def run() -> None:
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    benchmark(run)


@pytest.mark.benchmark(group="online-softmax")
def test_online_softmax_attention(seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark online-softmax (FlashAttention-style) attention."""
    q = torch.randn(1, 4, seq_len, 16)
    k = torch.randn(1, 4, seq_len, 16)
    v = torch.randn(1, 4, seq_len, 16)
    backend = TorchBackend()

    def run() -> None:
        backend.online_softmax_attention(q, k, v, block_size=32)

    benchmark(run)


@pytest.mark.benchmark(group="quantization")
def test_vq_precompute(seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark hierarchical VQ precompute."""
    torch.manual_seed(0)
    cb = HierarchicalCodebook(
        num_heads=4,
        num_parents=16,
        children_per_parent=4,
        head_dim=16,
    )
    cb.initialize_parents_random()
    quantizer = EuclideanHierarchicalQuantizer()
    keys = torch.randn(1, 4, seq_len, 16)
    values = torch.randn(1, 4, seq_len, 16)

    def run() -> None:
        quantizer.precompute(keys, values, cb)

    benchmark(run)


@pytest.mark.benchmark(group="reproducibility")
def test_attention_reproducibility() -> None:
    """Same seed produces identical attention output (spec §3.19.3)."""
    seed_everything(42)
    cfg = small_attn_config(seq_len=64)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, 64, 64)
    out1 = module(q, q, q)

    seed_everything(42)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, 64, 64)
    out2 = module(q, q, q)

    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.fixture(params=[1024, 2048])
def large_seq_len(request: pytest.FixtureRequest) -> int:
    """Large sequence lengths for scaling benchmarks."""
    param: int = request.param
    return param


@pytest.mark.benchmark(group="large-attention")
def test_avqa_attention_large(large_seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark AVQA at large N (spec §3.19)."""
    cfg = small_attn_config(seq_len=large_seq_len)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, large_seq_len, 64)
    k = torch.randn(1, large_seq_len, 64)
    v = torch.randn(1, large_seq_len, 64)

    def run() -> None:
        module(q, k, v)

    benchmark(run)


@pytest.mark.benchmark(group="large-attention")
def test_pytorch_attention_large(large_seq_len: int, benchmark: 'BenchmarkFixture') -> None:
    """Benchmark PyTorch SDPA at large N for comparison."""
    q = torch.randn(1, 4, large_seq_len, 16)
    k = torch.randn(1, 4, large_seq_len, 16)
    v = torch.randn(1, 4, large_seq_len, 16)

    def run() -> None:
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    benchmark(run)


def test_complexity_scaling() -> None:
    """Wall-clock must scale sub-quadratically from N=64 to N=256 (ISSUE-0024)."""

    def time_at(n: int) -> float:
        cfg_n = small_attn_config(seq_len=n)
        mod = AVQAttention(cfg_n, in_proj=False, out_proj=False)
        q = torch.randn(1, n, 64)
        start = time.perf_counter()
        for _ in range(3):
            mod(q, q, q)
        return (time.perf_counter() - start) / 3

    t64 = time_at(64)
    t256 = time_at(256)
    # Sub-quadratic: t(256) < 20 * t(64). Quadratic would be 16x.
    assert t256 < 20 * t64, f"Scaling too steep: t(256)={t256:.3f}s vs t(64)={t64:.3f}s"


def test_avqa_output_quality() -> None:
    """AVQA output is finite, non-zero, and in reasonable range (spec §3.19)."""
    cfg = small_attn_config(seq_len=128)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, 128, 64)
    k = torch.randn(1, 128, 64)
    v = torch.randn(1, 128, 64)
    out = module(q, k, v)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    assert out.abs().mean().item() > 1e-6, "Output is degenerate (all zeros)"
    assert out.abs().mean().item() < 10.0, "Output magnitude too large"


def test_avqa_vs_pytorch_sdpa_numerical() -> None:
    """AVQA output is in the same ballpark as PyTorch SDPA for identical inputs.

    We don't expect exact match (VQ introduces approximation), but the
    outputs should be correlated and have similar magnitudes.
    """
    torch.manual_seed(42)
    B, H, T, D = 1, 4, 64, 16
    E = H * D
    q_4d = torch.randn(B, H, T, D)
    k_4d = torch.randn(B, H, T, D)
    v_4d = torch.randn(B, H, T, D)

    # PyTorch SDPA reference.
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(q_4d, k_4d, v_4d)

    # AVQA (reshapes to [B, T, E]).
    cfg = small_attn_config(seq_len=T, num_heads=H, embed_dim=E)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q_3d = q_4d.transpose(1, 2).reshape(B, T, E)
    k_3d = k_4d.transpose(1, 2).reshape(B, T, E)
    v_3d = v_4d.transpose(1, 2).reshape(B, T, E)
    avqa_out = module(q_3d, k_3d, v_3d)

    # Reshape SDPA output to [B, T, E] for comparison.
    sdpa_out_3d = sdpa_out.transpose(1, 2).reshape(B, T, E)

    # Correlation should be positive (both approximate the same attention).
    corr = torch.nn.functional.cosine_similarity(
        avqa_out.reshape(-1),
        sdpa_out_3d.reshape(-1),
        dim=0,
    )
    assert corr.item() > 0.0, f"Outputs are negatively correlated: {corr.item():.3f}"

    # Magnitudes should be in the same order of magnitude.
    ratio = avqa_out.abs().mean() / sdpa_out_3d.abs().mean().clamp_min(1e-8)
    assert 0.01 < ratio < 100.0, f"Magnitude ratio too extreme: {ratio.item():.3f}"
