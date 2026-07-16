"""pytest-benchmark suite for AVQA (spec §3.19).

Run with::

    pytest tests/performance/ --benchmark-only

All benchmarks compare AVQA against PyTorch SDPA on identical inputs.
"""

from __future__ import annotations

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


def _small_attn_config(seq_len: int, num_heads: int = 4, embed_dim: int = 64) -> AVQConfig:
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
    return request.param


@pytest.mark.benchmark(group="attention")
def test_avqa_attention(seq_len: int, benchmark: object) -> None:
    """Benchmark AVQA attention forward pass."""
    cfg = _small_attn_config(seq_len=seq_len)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, seq_len, 64)
    k = torch.randn(1, seq_len, 64)
    v = torch.randn(1, seq_len, 64)

    def run() -> None:
        module(q, k, v)

    benchmark(run)


@pytest.mark.benchmark(group="attention")
def test_pytorch_attention(seq_len: int, benchmark: object) -> None:
    """Benchmark PyTorch SDPA reference for comparison."""
    q = torch.randn(1, 4, seq_len, 16)
    k = torch.randn(1, 4, seq_len, 16)
    v = torch.randn(1, 4, seq_len, 16)

    def run() -> None:
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    benchmark(run)


@pytest.mark.benchmark(group="online-softmax")
def test_online_softmax_attention(seq_len: int, benchmark: object) -> None:
    """Benchmark online-softmax (FlashAttention-style) attention."""
    q = torch.randn(1, 4, seq_len, 16)
    k = torch.randn(1, 4, seq_len, 16)
    v = torch.randn(1, 4, seq_len, 16)
    backend = TorchBackend()

    def run() -> None:
        backend.online_softmax_attention(q, k, v, block_size=32)

    benchmark(run)


@pytest.mark.benchmark(group="quantization")
def test_vq_precompute(seq_len: int, benchmark: object) -> None:
    """Benchmark hierarchical VQ precompute."""
    torch.manual_seed(0)
    cb = HierarchicalCodebook(
        num_heads=4, num_parents=16, children_per_parent=4, head_dim=16,
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
    cfg = _small_attn_config(seq_len=64)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, 64, 64)
    out1 = module(q, q, q)

    seed_everything(42)
    module = AVQAttention(cfg, in_proj=False, out_proj=False)
    q = torch.randn(1, 64, 64)
    out2 = module(q, q, q)

    assert torch.allclose(out1, out2, atol=1e-5)
