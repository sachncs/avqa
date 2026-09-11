"""Profiler + Visualizer usage example.

Usage::

    python examples/05_profiler_visualizer.py
"""

from __future__ import annotations

import torch

from avqa import AVQAttention, AVQConfig, Profiler
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)


def main() -> None:
    """Run a profiled forward pass and dump the report."""
    torch.manual_seed(0)
    config = AVQConfig(
        attention=AttentionShapeConfig(embed_dim=128, num_heads=4, head_dim=32),
        codebook=CodebookConfig(num_codewords=32, children_per_codeword=4),
        routing=RoutingConfig(refinement_budget=4),
    )
    attention = AVQAttention(config, in_proj=False, out_proj=False).eval()
    profiler = Profiler()

    q = torch.randn(2, 16, 128)
    k = torch.randn(2, 16, 128)
    v = torch.randn(2, 16, 128)
    with profiler.stage("forward"), torch.no_grad():
        out = attention(q, k, v)

    report = profiler.report
    print(f"output shape: {tuple(out.shape)}")
    print(f"stages recorded: {[s.name for s in report.stage_timers]}")


if __name__ == "__main__":
    main()
