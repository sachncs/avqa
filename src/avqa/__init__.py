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
    >>> from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig
    >>> config = AVQConfig(
    ...     attention=AttentionShapeConfig(embed_dim=64, num_heads=4, head_dim=16),
    ...     codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
    ...     routing=RoutingConfig(refinement_budget=3),
    ... )
    >>> attention = AVQAttention(config, in_proj=False, out_proj=False)
    >>> query = torch.randn(2, 8, 64)
    >>> key = torch.randn(2, 16, 64)
    >>> value = torch.randn(2, 16, 64)
    >>> output = attention(query, key, value)
"""

from avqa._version import __version__

__all__ = [
    "AVQAttention",
    "AVQConfig",
    "AdaptiveRefinement",
    "Backend",
    "Codebook",
    "HierarchicalCodebook",
    "KVCache",
    "Profiler",
    "Router",
    "Scheduler",
    "VectorQuantizer",
    "__version__",
]

__version_info__ = tuple(int(part) for part in __version__.split(".") if part.isdigit())


def __getattr__(name: str) -> object:
    """Lazy import of public API.

    Public symbols are imported on first access to keep ``import avqa`` cheap
    and avoid forcing every optional dependency at module load time.
    """
    if name == "AVQAttention":
        from avqa.attention_module import AVQAttention as _AVQAttention

        return _AVQAttention
    if name == "AVQConfig":
        from avqa.config import AVQConfig as _AVQConfig

        return _AVQConfig
    if name == "VectorQuantizer":
        from avqa.quantizer import VectorQuantizer as _VectorQuantizer

        return _VectorQuantizer
    if name == "HierarchicalCodebook":
        from avqa.codebook import HierarchicalCodebook as _HierarchicalCodebook

        return _HierarchicalCodebook
    if name == "Codebook":
        from avqa.codebook import HierarchicalCodebook as _Codebook

        return _Codebook
    if name == "Router":
        from avqa.routing import Router as _Router

        return _Router
    if name == "AdaptiveRefinement":
        from avqa.refinement import AdaptiveRefinement as _AdaptiveRefinement

        return _AdaptiveRefinement
    if name == "Scheduler":
        from avqa.scheduler import Scheduler as _Scheduler

        return _Scheduler
    if name == "KVCache":
        from avqa.cache import KVCache as _KVCache

        return _KVCache
    if name == "Backend":
        from avqa.backend import Backend as _Backend

        return _Backend
    if name == "Profiler":
        from avqa.profiling import Profiler as _Profiler

        return _Profiler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
