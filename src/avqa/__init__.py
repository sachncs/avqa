"""Adaptive Vector Quantized Attention (AVQA).

AVQA is a production-grade Python library implementing Adaptive Vector
Quantized Attention (AVQ-Attention) as described in:

    https://arxiv.org/html/2607.12789v1

This is an independent, community-driven implementation. See the README
for the disclaimer and citation information.

Public API (see avqa/functional.py for the functional entry point):

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

from __future__ import annotations

import warnings

from avqa._version import __version__
from avqa.attention_module import AVQAttention
from avqa.backend import Backend, TorchBackend
from avqa.cache import KVCache
from avqa.codebook import HierarchicalCodebook
from avqa.config import AVQConfig
from avqa.exceptions import (
    AVQAError,
    BackendError,
    CodebookError,
    ConfigurationError,
    DeviceError,
    DtypeError,
    RoutingError,
    ShapeError,
)
from avqa.profiling import Profiler
from avqa.refinement import AdaptiveRefinement
from avqa.routing import Router
from avqa.scheduler import Scheduler
from avqa.utils.validation import (
    validate_contiguous,
    validate_device,
    validate_device_match,
    validate_dtype,
    validate_embed_dim,
    validate_finite,
    validate_rank,
    validate_shape,
)

__all__ = [
    "AVQAError",
    "AVQAttention",
    "AVQConfig",
    "AdaptiveRefinement",
    "Backend",
    "BackendError",
    "Codebook",
    "CodebookError",
    "ConfigurationError",
    "DeviceError",
    "DtypeError",
    "HierarchicalCodebook",
    "KVCache",
    "Profiler",
    "Router",
    "RoutingError",
    "Scheduler",
    "ShapeError",
    "TorchBackend",
    "VectorQuantizer",
    "Visualizer",
    "__version__",
    "validate_contiguous",
    "validate_device",
    "validate_device_match",
    "validate_dtype",
    "validate_embed_dim",
    "validate_finite",
    "validate_rank",
    "validate_shape",
]


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints.

    Falls back to (0,) for unparseable strings so downstream consumers
    can still do ``avqa.__version_info__[0]`` without erroring.
    """
    out: list[int] = []
    for part in v.split("."):
        digits = ""
        for c in part:
            if c.isdigit():
                digits += c
            elif digits:
                break
        if digits:
            out.append(int(digits))
    return tuple(out) or (0,)


__version_info__ = _parse_version(__version__)

# Aliases for the two most common user-facing names.
Codebook = HierarchicalCodebook

# :class:`Visualizer` is the abstract base; the concrete JSONVisualizer
# remains on avqa.visualization. We re-export via a lazy attribute to
# avoid importing matplotlib/graphviz at module-load time.
_BUILTIN_VISUALIZERS = {"json"}


def __getattr__(name: str) -> object:
    """Lazy attribute lookup for genuinely expensive imports only.

    Eager imports live at module top so ``from avqa import AVQAttention``
    is straightforward. The :class:`Visualizer` factory and the
    :class:`TritonBackend` need optional heavy dependencies, so they
    remain lazy.
    """
    if name == "Visualizer":
        from avqa.visualization import Visualizer as _Visualizer

        return _Visualizer
    if name == "TritonBackend":
        try:
            from avqa.backend import TritonBackend as _TritonBackend
        except ImportError as exc:
            warnings.warn(
                f"TritonBackend unavailable in this environment: {exc}",
                stacklevel=2,
            )
            raise
        return _TritonBackend
    raise AttributeError(name)
