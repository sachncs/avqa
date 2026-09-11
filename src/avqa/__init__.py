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

from avqa.attention_module import AVQAttention
from avqa.backend import Backend, TorchBackend
from avqa.cache import InMemoryKVCache, KVCache, PagedKVCache
from avqa.codebook import HierarchicalCodebook
from avqa.config import (
    DEFAULT_EMA_DECAY,
    SCHEMA_VERSION,
    AttentionShapeConfig,
    AVQConfig,
    BackendConfig,
    CacheConfig,
    CodebookConfig,
    ExecutionConfig,
    HopfieldConfig,
    MergeConfig,
    PrecisionConfig,
    RefinementConfig,
    RoutingConfig,
)
from avqa.data import (
    HEAD,
    HEAD_DIM,
    SHAPE_OUTPUT,
    SUPPORTED_DEVICES,
    SUPPORTED_DTYPES,
    TensorContract,
    is_supported_device,
    is_supported_dtype,
    make_default_contract,
)
from avqa.exceptions import (
    AVQAError,
    BackendError,
    CodebookError,
    ConfigurationError,
    DeviceError,
    DtypeError,
    MergeError,
    NotInitializedError,
    RoutingError,
    ShapeError,
)
from avqa.hopfield import (
    AdaptiveSchedule,
    hopfield_logits,
    paper_beta,
    per_query_beta,
    validate_adaptive,
)
from avqa.logging import (
    AVQA_LOGGER_NAME,
    configure_logger,
    get_logger,
    is_configured,
)
from avqa.merge import MergeStrategy
from avqa.multipass import MultiPassRefiner, PassBudget
from avqa.online_adaptation import online_codebook_adaptation
from avqa.profiling import Profiler, ProfilerReport, StageTimer
from avqa.quantizer import (
    EuclideanHierarchicalQuantizer,
    QuantizationResult,
    VectorQuantizer,
)
from avqa.refinement import AdaptiveRefinement, RefinementResult, refine
from avqa.routing import (
    BudgetRouter,
    Router,
    RoutingDecision,
    ThresholdRouter,
    TopPRouter,
    compute_importance,
)
from avqa.scheduler import AdaptiveScheduler, DefaultScheduler, Scheduler
from avqa.streaming_vq import StreamingVQBuffer, empty_realisation
from avqa.utils.seed import DEFAULT_SEED, seed_everything
from avqa.utils.validation import (
    NonFiniteTensorError,
    RankLike,
    ShapeLike,
    coerce_shape,
    shape_to_string,
    validate_contiguous,
    validate_device,
    validate_device_match,
    validate_dtype,
    validate_embed_dim,
    validate_finite,
    validate_rank,
    validate_shape,
)
from avqa.version import __version__
from avqa.visualization import (
    CodebookUtilizationDict,
    HeatmapData,
    JSONVisualizer,
    RefinementTreeDict,
    RoutingPathDict,
    TimelineDict,
    TimelineEvent,
    TreeNode,
    Visualizer,
)

__all__ = [
    "AVQA_LOGGER_NAME",
    "DEFAULT_EMA_DECAY",
    "DEFAULT_SEED",
    "HEAD",
    "HEAD_DIM",
    "SCHEMA_VERSION",
    "SHAPE_OUTPUT",
    "SUPPORTED_DEVICES",
    "SUPPORTED_DTYPES",
    "AVQAError",
    "AVQAttention",
    "AVQConfig",
    "AdaptiveRefinement",
    "AdaptiveSchedule",
    "AdaptiveScheduler",
    "AttentionShapeConfig",
    "Backend",
    "BackendConfig",
    "BackendError",
    "BudgetRouter",
    "CacheConfig",
    "Codebook",
    "CodebookConfig",
    "CodebookError",
    "CodebookUtilizationDict",
    "ConfigurationError",
    "DefaultScheduler",
    "DeviceError",
    "DtypeError",
    "EuclideanHierarchicalQuantizer",
    "ExecutionConfig",
    "HeatmapData",
    "HierarchicalCodebook",
    "HopfieldConfig",
    "InMemoryKVCache",
    "JSONVisualizer",
    "KVCache",
    "MergeConfig",
    "MergeError",
    "MergeStrategy",
    "MultiPassRefiner",
    "NonFiniteTensorError",
    "NotInitializedError",
    "PagedKVCache",
    "PassBudget",
    "PrecisionConfig",
    "Profiler",
    "ProfilerReport",
    "QuantizationResult",
    "RankLike",
    "RefinementConfig",
    "RefinementResult",
    "RefinementTreeDict",
    "Router",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingError",
    "RoutingPathDict",
    "Scheduler",
    "ShapeError",
    "ShapeLike",
    "StageTimer",
    "StreamingVQBuffer",
    "TensorContract",
    "ThresholdRouter",
    "TimelineDict",
    "TimelineEvent",
    "TopPRouter",
    "TorchBackend",
    "TreeNode",
    "VectorQuantizer",
    "Visualizer",
    "__version__",
    "coerce_shape",
    "compute_importance",
    "configure_logger",
    "empty_realisation",
    "get_logger",
    "hopfield_logits",
    "is_configured",
    "is_supported_device",
    "is_supported_dtype",
    "make_default_contract",
    "online_codebook_adaptation",
    "paper_beta",
    "per_query_beta",
    "refine",
    "seed_everything",
    "shape_to_string",
    "validate_adaptive",
    "validate_contiguous",
    "validate_device",
    "validate_device_match",
    "validate_dtype",
    "validate_embed_dim",
    "validate_finite",
    "validate_rank",
    "validate_shape",
]


def parse_version(v: str) -> tuple[int, ...]:
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


__version_info__ = parse_version(__version__)


# Aliases for the two most common user-facing names.
Codebook = HierarchicalCodebook
