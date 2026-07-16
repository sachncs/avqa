"""Configuration objects for AVQA.

Spec §5.8 mandates immutable dataclass configuration with validation,
serialization, equality, and versioning. Spec §3.6 lists the minimum
required fields: codebook size, branching factor, refinement budget,
routing strategy, merge strategy, backend, execution mode, precision,
cache configuration.

ponytail: collapsed nine planned sub-config files (avq, codebook,
routing, refinement, merge, backend, cache, precision, execution) into
one module. They are interdependent frozen dataclasses composing a
single :class:`AVQConfig` root; splitting them across files would just
create import ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from avqa.exceptions import ConfigurationError

# Spec §3.6 / §5.19 — version of the configuration schema.
SCHEMA_VERSION: str = "1"

# Spec §8.9 — default EMA decay for codebook training.
DEFAULT_EMA_DECAY: float = 0.99


def _require_positive(value: float, field_name: str) -> None:
    if value <= 0:
        msg = f"{field_name} must be > 0, got {value}"
        raise ConfigurationError(msg, {field_name: value})


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0, got {value}"
        raise ConfigurationError(msg, {field_name: value})


def _require_in_range(value: float, lo: float, hi: float, field_name: str) -> None:
    if not lo <= value <= hi:
        msg = f"{field_name} must be in [{lo}, {hi}], got {value}"
        raise ConfigurationError(msg, {field_name: value})


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CodebookConfig:
    """Hierarchical codebook parameters (spec §3.6, §8.3).

    Attributes:
        num_codewords: Number of parent codewords per head (M₀).
        children_per_codeword: Number of children per parent (C).
        perturbation_scale: Std of the Gaussian noise used to initialize
            children near their parent (spec §8.10).
        ema_decay: EMA decay for codebook training (spec §8.9).

    Example:
        >>> cb = CodebookConfig()
        >>> cb.num_codewords
        64
    """

    num_codewords: int = 64
    children_per_codeword: int = 4
    perturbation_scale: float = 0.1
    ema_decay: float = DEFAULT_EMA_DECAY

    def __post_init__(self) -> None:
        _require_positive(self.num_codewords, "num_codewords")
        _require_positive(self.children_per_codeword, "children_per_codeword")
        _require_positive(self.perturbation_scale, "perturbation_scale")
        _require_in_range(self.ema_decay, 0.0, 1.0, "ema_decay")


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Routing strategy parameters (spec §3.6, §3.10).

    Attributes:
        strategy: Routing strategy name (e.g., ``"topp"``).
        refinement_budget: Maximum number of parents to refine (P).
        importance_temperature: Optional temperature scaling for importance.

    Example:
        >>> RoutingConfig()
        RoutingConfig(strategy='topp', refinement_budget=8, ...)
    """

    strategy: str = "topp"
    refinement_budget: int = 8
    importance_temperature: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.refinement_budget, "refinement_budget")
        _require_positive(self.importance_temperature, "importance_temperature")


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Refinement orchestration parameters (spec §3.6, §9.6)."""

    enabled: bool = True
    threshold: float = 0.0
    adaptive_budget: bool = False

    def __post_init__(self) -> None:
        _require_in_range(self.threshold, 0.0, 1.0, "threshold")


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Merge-strategy selection (spec §3.11).

    Attributes:
        strategy: One of ``"probability"``, ``"weighted"``, ``"logit"``,
            ``"normalized"``.
    """

    strategy: str = "probability"

    def __post_init__(self) -> None:
        allowed = {"probability", "weighted", "logit", "normalized"}
        if self.strategy not in allowed:
            msg = f"merge.strategy must be one of {sorted(allowed)}, got {self.strategy!r}"
            raise ConfigurationError(msg, {"strategy": self.strategy})


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Backend selection (spec §3.12, §5.9).

    Attributes:
        name: Backend identifier (e.g., ``"torch"``, ``"triton"``).
        enable_autotune: Whether to autotune kernel parameters when supported.
        skip_validation: Disable runtime tensor validation (spec §6.12).
    """

    name: str = "torch"
    enable_autotune: bool = False
    skip_validation: bool = False

    def __post_init__(self) -> None:
        allowed = {"torch", "triton"}
        if self.name not in allowed:
            msg = f"backend.name must be one of {sorted(allowed)}, got {self.name!r}"
            raise ConfigurationError(msg, {"backend.name": self.name})


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """KV cache configuration (spec §3.13)."""

    enabled: bool = True
    max_size: int = 0  # 0 means unbounded

    def __post_init__(self) -> None:
        _require_non_negative(self.max_size, "cache.max_size")


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Mixed-precision configuration (spec §3.6, §6.9).

    Attributes:
        dtype: Computation dtype. Must be in :data:`avqa.data.SUPPORTED_DTYPES`.
        autocast: Whether to enable PyTorch autocast for the forward pass.
    """

    dtype: str = "float32"
    autocast: bool = False

    def __post_init__(self) -> None:
        allowed = {"float32", "float16", "bfloat16"}
        if self.dtype not in allowed:
            msg = f"precision.dtype must be one of {sorted(allowed)}, got {self.dtype!r}"
            raise ConfigurationError(msg, {"precision.dtype": self.dtype})


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Execution-mode selection (spec §4.13, §10.15).

    Attributes:
        mode: ``"reference"``, ``"optimized"``, or ``"research"``.
        deterministic: Enable strict deterministic algorithms.
        seed: Optional RNG seed applied at module init.
    """

    mode: str = "optimized"
    deterministic: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        allowed = {"reference", "optimized", "research"}
        if self.mode not in allowed:
            msg = f"execution.mode must be one of {sorted(allowed)}, got {self.mode!r}"
            raise ConfigurationError(msg, {"execution.mode": self.mode})
        _require_non_negative(self.seed, "execution.seed")


@dataclass(frozen=True, slots=True)
class AttentionShapeConfig:
    """Attention-shape parameters (spec §3.4).

    Attributes:
        embed_dim: Embedding dimension (E).
        num_heads: Number of attention heads (H). Must divide ``embed_dim``.
        head_dim: Per-head dimension (D). Defaults to ``embed_dim // num_heads``.
        max_sequence_length: Optional upper bound for sequence lengths.
            ``0`` means unbounded.
    """

    embed_dim: int = 512
    num_heads: int = 8
    head_dim: int = 0  # 0 -> auto-derive as embed_dim // num_heads
    max_sequence_length: int = 0

    def __post_init__(self) -> None:
        _require_positive(self.embed_dim, "embed_dim")
        _require_positive(self.num_heads, "num_heads")
        _require_non_negative(self.max_sequence_length, "max_sequence_length")
        if self.embed_dim % self.num_heads != 0:
            msg = (
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
            raise ConfigurationError(
                msg, {"embed_dim": self.embed_dim, "num_heads": self.num_heads}
            )


# ---------------------------------------------------------------------------
# Top-level AVQConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AVQConfig:
    """Top-level immutable configuration for AVQA (spec §5.6, §5.8).

    Composes every sub-configuration into a single object that downstream
    modules accept in their constructors. Frozen dataclass — mutation
    raises.

    Attributes:
        attention: Attention-shape parameters.
        codebook: Hierarchical codebook parameters.
        routing: Routing strategy parameters.
        refinement: Refinement orchestration parameters.
        merge: Merge-strategy selection.
        backend: Backend selection.
        cache: KV cache configuration.
        precision: Mixed-precision configuration.
        execution: Execution-mode selection.
        dropout: Dropout probability (spec §3.4.7).
        causal: Enable causal masking (spec §3.4.4).
        tolerance_atol: Absolute tolerance for numerical equivalence checks
            (spec gap G6).
        tolerance_rtol: Relative tolerance for numerical equivalence checks.

    Example:
        >>> cfg = AVQConfig()
        >>> cfg.attention.num_heads
        8
        >>> cfg.precision.dtype
        'float32'
    """

    attention: AttentionShapeConfig = field(default_factory=AttentionShapeConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    dropout: float = 0.0
    causal: bool = False
    tolerance_atol: float = 1e-5
    tolerance_rtol: float = 1e-5

    def __post_init__(self) -> None:
        _require_in_range(self.dropout, 0.0, 1.0, "dropout")
        _require_positive(self.tolerance_atol, "tolerance_atol")
        _require_positive(self.tolerance_rtol, "tolerance_rtol")
        # Auto-derive head_dim if user left it at 0.
        if self.attention.head_dim == 0:
            object.__setattr__(
                self, "attention", AttentionShapeConfig(
                    embed_dim=self.attention.embed_dim,
                    num_heads=self.attention.num_heads,
                    head_dim=self.attention.embed_dim // self.attention.num_heads,
                    max_sequence_length=self.attention.max_sequence_length,
                )
            )

    # ------------------------------------------------------------------
    # Serialization (spec §3.20, §5.12)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Recursively convert to a plain dictionary.

        The returned dict carries a ``"schema_version"`` key for forward
        compatibility (spec §3.20.2).
        """
        result: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif isinstance(value, tuple) and hasattr(type(value), "_fields"):
                result[f.name] = tuple(_to_primitive(v) for v in value)
            else:
                result[f.name] = _to_primitive(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AVQConfig:
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Unknown fields raise :class:`ConfigurationError` so silent
        schema drift is impossible.
        """
        if not isinstance(data, dict):
            msg = f"AVQConfig.from_dict expected dict, got {type(data).__name__}"
            raise ConfigurationError(msg)
        schema = data.get("schema_version")
        if schema is not None and schema != SCHEMA_VERSION:
            msg = (
                f"AVQConfig schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {schema!r}"
            )
            raise ConfigurationError(msg, {"schema_version": schema})
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            field_type = f.type
            # Try to use the declared type's from_dict if available.
            target = cls._resolve_field_type(field_type)
            if isinstance(value, dict) and hasattr(target, "from_dict"):
                kwargs[f.name] = target.from_dict(value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _resolve_field_type(type_string: object) -> type:
        """Resolve a dataclass field type annotation back to a class.

        ponytail: only used by ``from_dict`` to dispatch sub-config
        reconstruction. Falls back to ``object`` if the annotation cannot
        be statically resolved.
        """
        if isinstance(type_string, type):
            return type_string
        return object


def _to_primitive(value: object) -> object:
    """Convert a value to a JSON-serializable primitive."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


__all__ = [
    "DEFAULT_EMA_DECAY",
    "SCHEMA_VERSION",
    "AVQConfig",
    "AttentionShapeConfig",
    "BackendConfig",
    "CacheConfig",
    "CodebookConfig",
    "ExecutionConfig",
    "MergeConfig",
    "PrecisionConfig",
    "RefinementConfig",
    "RoutingConfig",
]
