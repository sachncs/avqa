"""Tests for avqa.config module."""

from __future__ import annotations

import pytest

from avqa.config import (
    SCHEMA_VERSION,
    AttentionShapeConfig,
    AVQConfig,
    BackendConfig,
    CacheConfig,
    CodebookConfig,
    ExecutionConfig,
    MergeConfig,
    PrecisionConfig,
    RoutingConfig,
)
from avqa.exceptions import ConfigurationError


class TestCodebookConfig:
    """Tests for CodebookConfig."""

    def test_defaults(self) -> None:
        """Defaults match spec §8.9 and §8.10."""
        cb = CodebookConfig()
        assert cb.num_codewords == 64
        assert cb.children_per_codeword == 4
        assert cb.perturbation_scale == 0.1
        assert cb.ema_decay == 0.99

    def test_positive_num_codewords(self) -> None:
        """num_codewords must be > 0."""
        with pytest.raises(ConfigurationError):
            CodebookConfig(num_codewords=0)

    def test_positive_children(self) -> None:
        """children_per_codeword must be > 0."""
        with pytest.raises(ConfigurationError):
            CodebookConfig(children_per_codeword=0)

    def test_ema_decay_range(self) -> None:
        """ema_decay must be in [0, 1]."""
        with pytest.raises(ConfigurationError):
            CodebookConfig(ema_decay=1.5)
        with pytest.raises(ConfigurationError):
            CodebookConfig(ema_decay=-0.1)

    def test_immutable(self) -> None:
        """Frozen — assignment raises."""
        cb = CodebookConfig()
        with pytest.raises((AttributeError, Exception)):
            cb.num_codewords = 128  # type: ignore[misc]


class TestRoutingConfig:
    """Tests for RoutingConfig."""

    def test_defaults(self) -> None:
        """Default refinement budget is 8."""
        r = RoutingConfig()
        assert r.strategy == "topp"
        assert r.refinement_budget == 8

    def test_budget_positive(self) -> None:
        """refinement_budget must be > 0."""
        with pytest.raises(ConfigurationError):
            RoutingConfig(refinement_budget=0)


class TestMergeConfig:
    """Tests for MergeConfig."""

    @pytest.mark.parametrize("strategy", ["probability", "weighted", "logit", "normalized"])
    def test_valid_strategies(self, strategy: str) -> None:
        """All four documented strategies are accepted."""
        MergeConfig(strategy=strategy)

    def test_invalid_strategy_raises(self) -> None:
        """Unknown strategies are rejected."""
        with pytest.raises(ConfigurationError, match=r"merge\.strategy"):
            MergeConfig(strategy="not_a_strategy")


class TestBackendConfig:
    """Tests for BackendConfig."""

    @pytest.mark.parametrize("name", ["torch", "triton"])
    def test_valid_backends(self, name: str) -> None:
        """Both documented backends are accepted."""
        BackendConfig(name=name)

    def test_invalid_backend_raises(self) -> None:
        """Unknown backends are rejected."""
        with pytest.raises(ConfigurationError, match=r"backend\.name"):
            BackendConfig(name="not_a_backend")


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_max_size_is_unbounded(self) -> None:
        """max_size=0 means unbounded."""
        c = CacheConfig()
        assert c.max_size == 0

    def test_negative_max_size_rejected(self) -> None:
        """Negative max_size is rejected."""
        with pytest.raises(ConfigurationError):
            CacheConfig(max_size=-1)


class TestPrecisionConfig:
    """Tests for PrecisionConfig."""

    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_valid_dtypes(self, dtype: str) -> None:
        """The three reference dtypes are accepted."""
        PrecisionConfig(dtype=dtype)

    def test_invalid_dtype_rejected(self) -> None:
        """Unsupported dtypes are rejected."""
        with pytest.raises(ConfigurationError, match=r"precision\.dtype"):
            PrecisionConfig(dtype="float128")


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    @pytest.mark.parametrize("mode", ["reference", "optimized", "research"])
    def test_valid_modes(self, mode: str) -> None:
        """All three documented modes are accepted."""
        ExecutionConfig(mode=mode)

    def test_invalid_mode_rejected(self) -> None:
        """Unknown modes are rejected."""
        with pytest.raises(ConfigurationError, match=r"execution\.mode"):
            ExecutionConfig(mode="production")


class TestAttentionShapeConfig:
    """Tests for AttentionShapeConfig."""

    def test_head_dim_auto_derived(self) -> None:
        """head_dim is derived from embed_dim // num_heads in AVQConfig."""
        cfg = AVQConfig()
        assert cfg.attention.head_dim == cfg.attention.embed_dim // cfg.attention.num_heads

    def test_embed_dim_divisibility(self) -> None:
        """embed_dim must be divisible by num_heads."""
        with pytest.raises(ConfigurationError, match="divisible"):
            AVQConfig(attention=AttentionShapeConfig(embed_dim=100, num_heads=8))

    def test_max_seq_length_validation(self) -> None:
        """max_sequence_length must be non-negative."""
        with pytest.raises(ConfigurationError):
            AVQConfig(attention=AttentionShapeConfig(max_sequence_length=-1))


class TestAVQConfig:
    """Tests for the top-level AVQConfig."""

    def test_default_construction(self) -> None:
        """All sub-configs have defaults."""
        cfg = AVQConfig()
        assert isinstance(cfg.codebook, CodebookConfig)
        assert isinstance(cfg.routing, RoutingConfig)
        assert isinstance(cfg.merge, MergeConfig)
        assert isinstance(cfg.backend, BackendConfig)

    def test_immutable(self) -> None:
        """Top-level config is frozen."""
        cfg = AVQConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.dropout = 0.5  # type: ignore[misc]

    def test_dropout_range(self) -> None:
        """dropout must be in [0, 1]."""
        with pytest.raises(ConfigurationError):
            AVQConfig(dropout=1.5)

    def test_nested_subconfig_immutable(self) -> None:
        """Sub-configs cannot be replaced via assignment."""
        cfg = AVQConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.codebook = CodebookConfig(num_codewords=128)  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two configs with the same fields are equal."""
        a = AVQConfig()
        b = AVQConfig()
        assert a == b

    def test_inequality(self) -> None:
        """Different field values produce unequal configs."""
        a = AVQConfig()
        b = AVQConfig(dropout=0.1)
        assert a != b

    def test_tolerance_fields(self) -> None:
        """tolerance_atol and tolerance_rtol are accessible (spec §3.16, G6)."""
        cfg = AVQConfig(tolerance_atol=1e-3, tolerance_rtol=1e-4)
        assert cfg.tolerance_atol == 1e-3
        assert cfg.tolerance_rtol == 1e-4


class TestAVQConfigSerialization:
    """Tests for AVQConfig (de)serialization (spec §3.20, §5.12)."""

    def test_round_trip_default(self) -> None:
        """Default config round-trips through dict."""
        cfg = AVQConfig()
        data = cfg.to_dict()
        restored = AVQConfig.from_dict(data)
        assert restored == cfg

    def test_round_trip_custom(self) -> None:
        """Custom config round-trips."""
        cfg = AVQConfig(
            dropout=0.1,
            causal=True,
            codebook=CodebookConfig(num_codewords=128, children_per_codeword=8),
        )
        data = cfg.to_dict()
        restored = AVQConfig.from_dict(data)
        assert restored == cfg

    def test_to_dict_includes_schema_version(self) -> None:
        """Serialized dict carries schema_version for forward compatibility."""
        cfg = AVQConfig()
        data = cfg.to_dict()
        assert data["schema_version"] == SCHEMA_VERSION

    def test_schema_mismatch_rejected(self) -> None:
        """Unknown schema_version raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="schema_version"):
            AVQConfig.from_dict({"schema_version": "999"})

    def test_non_dict_rejected(self) -> None:
        """Non-dict input is rejected."""
        with pytest.raises(ConfigurationError):
            AVQConfig.from_dict("not a dict")  # type: ignore[arg-type]


class TestAVQConfigEquality:
    """Tests for config equality (spec §5.8.2)."""

    def test_dict_equality(self) -> None:
        """Configs compare equal when all fields match."""
        a = AVQConfig(causal=True)
        b = AVQConfig(causal=True)
        assert a == b

    def test_hashable(self) -> None:
        """Configs are hashable (frozen dataclass)."""
        cfg = AVQConfig()
        assert hash(cfg) == hash(AVQConfig())
