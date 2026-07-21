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

import dataclasses
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from avqa.exceptions import ConfigurationError

# Spec §3.6 / §5.19 — version of the configuration schema.
SCHEMA_VERSION: str = "1"

# Spec §8.9 — default EMA decay for codebook training.
DEFAULT_EMA_DECAY: float = 0.99


def require_positive(value: float, field_name: str) -> None:
    """Raise ``ConfigurationError`` if ``value`` is not positive."""
    if value <= 0:
        msg = f"{field_name} must be > 0, got {value}"
        raise ConfigurationError(msg, {field_name: value})


def require_non_negative(value: float, field_name: str) -> None:
    """Raise ``ConfigurationError`` if ``value`` is negative."""
    if value < 0:
        msg = f"{field_name} must be >= 0, got {value}"
        raise ConfigurationError(msg, {field_name: value})


def require_in_range(value: float, lo: float, hi: float, field_name: str) -> None:
    """Raise ``ConfigurationError`` if ``value`` is outside ``[lo, hi]``."""
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
        commitment_loss_weight: Weight for the commitment (encoding) loss
            during training (spec §8.9). ``0.25`` is the paper default.
            Set to ``0.0`` to disable.
        max_depth: Maximum hierarchy depth. Currently only ``2`` (parent +
            child) is supported. Depths > 2 will raise at construction
            time. Arbitrary depth is planned for v0.2.0 (spec §2.7, §3.8).
        bcar_enabled: When ``True`` the AVQAttention forward pass applies
            an inference-time EMA update (BCAR, OPT-0003) to the
            hierarchical codebook. Defaults to ``False`` so the
            algorithm matches the paper exactly.
        bcar_decay: EMA decay for BCAR updates (default 0.99). Smaller
            values adapt faster but trade off stability against
            distribution shift.

    Example:
        >>> cb = CodebookConfig()
        >>> cb.num_codewords
        64
    """

    num_codewords: int = 64
    children_per_codeword: int = 4
    perturbation_scale: float = 0.1
    ema_decay: float = DEFAULT_EMA_DECAY
    commitment_loss_weight: float = 0.25
    max_depth: int = 2
    bcar_enabled: bool = False
    bcar_decay: float = 0.99

    def __post_init__(self) -> None:
        require_positive(self.num_codewords, "num_codewords")
        require_positive(self.children_per_codeword, "children_per_codeword")
        require_positive(self.perturbation_scale, "perturbation_scale")
        require_in_range(self.ema_decay, 0.0, 1.0, "ema_decay")
        require_non_negative(self.commitment_loss_weight, "commitment_loss_weight")
        require_in_range(self.bcar_decay, 0.0, 1.0, "bcar_decay")
        if self.max_depth != 2:
            msg = (
                f"max_depth={self.max_depth} is not yet supported; "
                f"only max_depth=2 (parent + child) is implemented. "
                f"Arbitrary tree depth is planned for v0.2.0."
            )
            raise ConfigurationError(msg, {"max_depth": self.max_depth})


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
        require_positive(self.refinement_budget, "refinement_budget")
        require_positive(self.importance_temperature, "importance_temperature")
        allowed = {"topp", "threshold", "budget"}
        if self.strategy not in allowed:
            raise ConfigurationError(
                f"routing.strategy must be one of {sorted(allowed)}, got {self.strategy!r}",
                {"strategy": self.strategy},
            )


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Refinement orchestration parameters (spec §3.6, §9.6)."""

    enabled: bool = True
    threshold: float = 0.0
    adaptive_budget: bool = False
    passes: int = 1
    pass_decay: float = 1.0

    def __post_init__(self) -> None:
        require_in_range(self.threshold, 0.0, 1.0, "threshold")


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
        name: ``"torch"`` (the only shipped backend).
        enable_autotune: Whether to autotune kernel parameters when supported.
        skip_validation: Disable runtime tensor validation (spec §6.12).
        hopfield: When ``True`` AVQAttention applies the HVAQ
            temperature schedule (SPEC §16) to the parent attention
            logits. Default ``False`` keeps the paper-exact softmax.
    """

    name: str = "torch"
    enable_autotune: bool = False
    skip_validation: bool = False
    hopfield: bool = False

    def __post_init__(self) -> None:
        allowed = {"torch"}
        if self.name not in allowed:
            msg = f"backend.name must be one of {sorted(allowed)}, got {self.name!r}"
            raise ConfigurationError(msg, {"backend.name": self.name})


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """KV cache configuration (spec §3.13).

    Attributes:
        enabled: Whether to enable KV caching.
        max_size: Maximum number of cached entries. ``0`` means unbounded.
    """

    enabled: bool = True
    max_size: int = 0  # 0 means unbounded

    def __post_init__(self) -> None:
        require_non_negative(self.max_size, "cache.max_size")


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
        compile_enabled: When ``True`` the AVQAttention forward is
            wrapped in ``torch.compile`` to reduce Python overhead on
            CPU (OPT-0002). Requires stable input shapes.
    """

    mode: str = "optimized"
    deterministic: bool = False
    seed: int = 0
    compile_enabled: bool = False
    causal_incremental: bool = False

    def __post_init__(self) -> None:
        allowed = {"reference", "optimized", "research"}
        if self.mode not in allowed:
            msg = f"execution.mode must be one of {sorted(allowed)}, got {self.mode!r}"
            raise ConfigurationError(msg, {"execution.mode": self.mode})
        require_non_negative(self.seed, "execution.seed")


@dataclass(frozen=True, slots=True)
class HopfieldConfig:
    """Hopfield-VQ-Attention (HVAQ) configuration (SPEC §16).

    Attributes:
        enabled: When ``True`` AVQAttention applies the HVAQ
            temperature schedule to the parent attention logits.
            ``BackendConfig.hopfield`` is the user-facing master
            switch; this attribute exists for forward compatibility.
        beta_init: Base temperature ``beta_0``. The paper uses
            ``1 / √d``; the HVAQ schedules scale around this
            base. ``0.0`` (default) auto-derives from the attention
            head dimension so the schedule is paper-exact when
            ``adaptive="none"``.
        adaptive: ``"none"`` (constant beta_0), ``"entropy"``
            (beta_0 · (1 + 1 / (1 + H_top))), or ``"linear"``
            (beta_0 · (1 + alpha * H_top)).
        alpha: Slope of the linear schedule; HVAQ-LIN only.
        learnable_parent_beta: When ``True``, adds a per-parent
            learnable inverse temperature ``beta_p`` as an
            ``nn.Parameter`` (initialized to 1.0). Gradient flows
            through :func:`hopfield_logits`.
        learnable_alpha: When ``True``, adds a per-head learnable
            ``alpha`` as an ``nn.Parameter`` (initialized from
            ``alpha``). Overrides the fixed ``alpha`` in entropy
            and linear schedules.
    """

    enabled: bool = False
    beta_init: float = 0.0
    adaptive: str = "none"
    alpha: float = 1.0
    learnable_parent_beta: bool = False
    learnable_alpha: bool = False

    def __post_init__(self) -> None:
        if self.beta_init < 0.0:
            msg = f"hopfield.beta_init must be >= 0, got {self.beta_init}"
            raise ConfigurationError(msg, {"hopfield.beta_init": self.beta_init})
        if self.alpha < 0.0:
            msg = f"hopfield.alpha must be >= 0, got {self.alpha}"
            raise ConfigurationError(msg, {"hopfield.alpha": self.alpha})


@dataclass(frozen=True, slots=True)
class AttentionShapeConfig:
    """Attention-shape parameters (spec §3.4).

    Attributes:
        embed_dim: Embedding dimension (E).
        num_heads: Number of attention heads (H). Must divide ``embed_dim``.
        head_dim: Per-head dimension (D). Defaults to ``embed_dim // num_heads``.
    """

    embed_dim: int = 512
    num_heads: int = 8
    head_dim: int = 0  # 0 -> auto-derive as embed_dim // num_heads

    def __post_init__(self) -> None:
        require_positive(self.embed_dim, "embed_dim")
        require_positive(self.num_heads, "num_heads")
        if self.embed_dim % self.num_heads != 0:
            msg = f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
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
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    dropout: float = 0.0
    causal: bool = False
    tolerance_atol: float = 1e-5
    tolerance_rtol: float = 1e-5

    def __post_init__(self) -> None:
        require_in_range(self.dropout, 0.0, 1.0, "dropout")
        require_positive(self.tolerance_atol, "tolerance_atol")
        require_positive(self.tolerance_rtol, "tolerance_rtol")
        # Auto-derive head_dim if user left it at 0.
        if self.attention.head_dim == 0:
            object.__setattr__(
                self,
                "attention",
                AttentionShapeConfig(
                    embed_dim=self.attention.embed_dim,
                    num_heads=self.attention.num_heads,
                    head_dim=self.attention.embed_dim // self.attention.num_heads,
                ),
            )

    # ------------------------------------------------------------------
    # Serialization (spec §3.20, §5.12)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Recursively convert to a plain dictionary.

        The returned dict carries a ``"schema_version"`` key for forward
        compatibility (spec §3.20.2). All nested sub-config dataclasses
        are recursively converted, so the result is JSON-serializable.
        """
        result: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif hasattr(value, "__dataclass_fields__"):
                # M6: sub-config dataclass → asdict for JSON safety.
                result[f.name] = to_primitive(asdict(value))
            elif isinstance(value, tuple) and hasattr(type(value), "_fields"):
                result[f.name] = tuple(to_primitive(v) for v in value)
            else:
                result[f.name] = to_primitive(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object] | object) -> AVQConfig:
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Unknown fields raise :class:`ConfigurationError` so silent
        schema drift is impossible.
        """
        if not isinstance(data, dict):
            msg = f"AVQConfig.from_dict expected dict, got {type(data).__name__}"
            raise ConfigurationError(msg)
        schema = data.get("schema_version")
        if schema is not None and schema != SCHEMA_VERSION:
            msg = f"AVQConfig schema_version mismatch: expected {SCHEMA_VERSION!r}, got {schema!r}"
            raise ConfigurationError(msg, {"schema_version": schema})
        known_fields = {f.name for f in fields(cls)}
        # M6: reject unknown fields to prevent silent schema drift.
        for k in data:
            if k != "schema_version" and k not in known_fields:
                msg = f"AVQConfig.from_dict got unknown field {k!r}; known fields are {sorted(known_fields)}"
                raise ConfigurationError(msg, {"unknown_field": k})
        # Use ``cls.__init__`` so the outer ``__post_init__`` runs
        # (validation, auto-derivation, etc.). Sub-configs are
        # reconstructed by their own ``__init__`` for the same reason.
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if isinstance(value, dict):
                target = cls.resolve_field_type(f.type)
                if target is not object and dataclasses.is_dataclass(target):
                    value = target(**value)
            kwargs[f.name] = value
        # mypy can't narrow dict[str, object] through **kwargs into the
        # dataclass's union-of-sub-configs; the runtime constructor
        # performs the actual validation via __post_init__.
        return cls(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def resolve_field_type(type_string: object) -> type:
        """Resolve a dataclass field type annotation back to a class.

        ponytail: only used by ``from_dict`` to dispatch sub-config
        reconstruction. Falls back to ``object`` if the annotation cannot
        be statically resolved.
        """
        if isinstance(type_string, type):
            return type_string
        # M6: resolve string annotations ("AttentionShapeConfig") back to
        # the actual class by name lookup in this module.
        if isinstance(type_string, str):
            cls_obj = globals().get(type_string)
            if isinstance(cls_obj, type):
                return cls_obj
        return object

    # ------------------------------------------------------------------
    # JSON file I/O (SPEC §3.20, §5.12)
    # ------------------------------------------------------------------

    def save_json(self, path: str | Path) -> Path:
        """Serialize this configuration to a JSON file.

        Args:
            path: Destination filesystem path. Parent directories are
                created if missing.

        Returns:
            The resolved path written to.

        Raises:
            ConfigurationError: If the file cannot be written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        # Spec §3.20.2: human-readable and diff-friendly JSON.
        try:
            target.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError as exc:
            msg = f"failed to write AVQConfig JSON to {target}"
            raise ConfigurationError(msg, {"path": str(target)}) from exc
        except TypeError as exc:
            # ``json.dumps`` raises TypeError when ``payload`` contains
            # a non-JSON-serializable value (set, dtype, unknown object).
            msg = (
                f"AVQConfig.to_dict() produced non-JSON-serializable value "
                f"for {target}"
            )
            raise ConfigurationError(msg, {"path": str(target)}) from exc
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> AVQConfig:
        """Load and reconstruct an :class:`AVQConfig` from a JSON file.

        Args:
            path: Source path produced by :meth:`save_json`.

        Returns:
            The reconstructed :class:`AVQConfig`.

        Raises:
            ConfigurationError: If the file is missing, unreadable, or
                carries an incompatible schema version.
        """
        source = Path(path)
        try:
            text = source.read_text()
        except OSError as exc:
            msg = f"failed to read AVQConfig JSON from {source}"
            raise ConfigurationError(msg, {"path": str(source)}) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON in AVQConfig file {source}: {exc.msg}"
            raise ConfigurationError(msg, {"path": str(source)}) from exc
        if not isinstance(data, dict):
            msg = f"AVQConfig JSON must be an object, got {type(data).__name__}"
            raise ConfigurationError(msg, {"path": str(source)})
        return cls.from_dict(data)


def to_primitive(value: object) -> object:
    """Convert a value to a JSON-serializable primitive."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [to_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_primitive(v) for k, v in value.items()}
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
    "require_in_range",
    "require_non_negative",
    "require_positive",
    "to_primitive",
]
