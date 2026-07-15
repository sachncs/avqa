"""Adaptive Vector Quantized Attention (AVQA).

AVQA is a production-grade Python library implementing Adaptive Vector
Quantized Attention (AVQ-Attention) as described in:

    https://arxiv.org/html/2607.12789v1

This is an independent, community-driven implementation. See the README
for the disclaimer and citation information.

Public API surface (see avqa/__init__.py and avqa/functional.py):

- AVQAttention: primary attention module
- AVQConfig: immutable configuration object
- VectorQuantizer: hierarchical vector quantization
- HierarchicalCodebook: parent-child codebook with mean constraint
- Router: routing strategy interface
- AdaptiveRefinement: refinement orchestrator
- Scheduler: refinement budget scheduler
- KVCache: autoregressive cache
- Backend: execution backend abstraction
- Profiler: runtime profiler
- Visualizer: visualization utilities

Example:
    >>> import torch
    >>> from avqa import AVQAttention, AVQConfig
    >>> config = AVQConfig(
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     num_codewords=64,
    ...     children_per_codeword=4,
    ...     refinement_budget=8,
    ... )
    >>> attention = AVQAttention(config)
    >>> query = torch.randn(2, 8, 128, 64)
    >>> key = torch.randn(2, 8, 128, 64)
    >>> value = torch.randn(2, 8, 128, 64)
    >>> output = attention(query, key, value)
"""

from avqa._version import __version__

__all__ = [
    "__version__",
    "AVQAttention",
    "AVQConfig",
    "VectorQuantizer",
    "HierarchicalCodebook",
    "Router",
    "AdaptiveRefinement",
    "Scheduler",
    "KVCache",
    "Backend",
    "Profiler",
]

__version_info__ = tuple(int(part) for part in __version__.split(".") if part.isdigit())


def __getattr__(name: str) -> object:
    """Lazy import of public API.

    Public symbols are imported on first access to keep ``import avqa`` cheap
    and avoid forcing every optional dependency at module load time.
    """
    if name == "AVQAttention":
        from avqa.attention.module import AVQAttention

        return AVQAttention
    if name == "AVQConfig":
        from avqa.config.avq import AVQConfig

        return AVQConfig
    if name == "VectorQuantizer":
        from avqa.quantizer.base import VectorQuantizer

        return VectorQuantizer
    if name == "HierarchicalCodebook":
        from avqa.codebook.hierarchical import HierarchicalCodebook

        return HierarchicalCodebook
    if name == "Router":
        from avqa.routing.base import Router

        return Router
    if name == "AdaptiveRefinement":
        from avqa.refinement.adaptive import AdaptiveRefinement

        return AdaptiveRefinement
    if name == "Scheduler":
        from avqa.scheduler.base import Scheduler

        return Scheduler
    if name == "KVCache":
        from avqa.cache.base import KVCache

        return KVCache
    if name == "Backend":
        from avqa.backend.base import Backend

        return Backend
    if name == "Profiler":
        from avqa.profiling.base import Profiler

        return Profiler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
