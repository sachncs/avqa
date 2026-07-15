"""Tests for avqa.registry module."""

from __future__ import annotations

import pytest

from avqa.registry import (
    BACKEND_REGISTRY,
    MERGE_REGISTRY,
    PROFILER_REGISTRY,
    QUANTIZER_REGISTRY,
    ROUTER_REGISTRY,
    SCHEDULER_REGISTRY,
    VISUALIZER_REGISTRY,
    Registry,
)


class TestRegistryBasics:
    """Tests for the Registry class."""

    def test_register_and_get(self) -> None:
        """Registered class is retrievable."""
        r: Registry[type] = Registry("test")

        class Foo:
            pass

        r.register("foo")(Foo)
        assert r.get("foo") is Foo

    def test_contains(self) -> None:
        """``in`` operator reflects registration."""
        r: Registry[type] = Registry("test")

        class Bar:
            pass

        r.register("bar")(Bar)
        assert "bar" in r
        assert "missing" not in r

    def test_contains_rejects_non_string(self) -> None:
        """Non-string keys are never contained (no TypeError)."""
        r: Registry[type] = Registry("test")
        assert 42 not in r
        assert None not in r

    def test_len(self) -> None:
        """Length reflects number of registered items."""
        r: Registry[type] = Registry("test")
        assert len(r) == 0
        r.register("a")(type("A", (), {}))
        assert len(r) == 1
        r.register("b")(type("B", (), {}))
        assert len(r) == 2

    def test_keys(self) -> None:
        """keys() returns all registered keys."""
        r: Registry[type] = Registry("test")
        r.register("a")(type("A", (), {}))
        r.register("b")(type("B", (), {}))
        assert set(r.keys()) == {"a", "b"}

    def test_double_register_raises(self) -> None:
        """Re-registering the same key raises ValueError."""
        r: Registry[type] = Registry("test")

        class Foo:
            pass

        r.register("foo")(Foo)
        with pytest.raises(ValueError, match="already registered"):
            r.register("foo")(Foo)

    def test_get_unknown_raises(self) -> None:
        """Unknown key raises KeyError listing available keys."""
        r: Registry[type] = Registry("test")
        r.register("a")(type("A", (), {}))
        with pytest.raises(KeyError, match="unknown key"):
            r.get("missing")
        # Error message lists available keys
        with pytest.raises(KeyError, match=r"\ba\b"):
            r.get("missing")

    def test_try_get_returns_none_for_missing(self) -> None:
        """try_get returns None instead of raising."""
        r: Registry[type] = Registry("test")
        assert r.try_get("nope") is None

    def test_try_get_returns_class_when_present(self) -> None:
        """try_get returns the class when registered."""

        class Foo:
            pass

        r: Registry[type] = Registry("test")
        r.register("foo")(Foo)
        assert r.try_get("foo") is Foo

    def test_repr_includes_name(self) -> None:
        """Repr includes the registry name."""
        r: Registry[type] = Registry("named")
        assert "named" in repr(r)


class TestSpecRegistries:
    """Tests for the spec-defined category registries (§5.10)."""

    @pytest.mark.parametrize(
        "registry",
        [
            QUANTIZER_REGISTRY,
            ROUTER_REGISTRY,
            MERGE_REGISTRY,
            SCHEDULER_REGISTRY,
            BACKEND_REGISTRY,
            PROFILER_REGISTRY,
            VISUALIZER_REGISTRY,
        ],
    )
    def test_registry_exists(self, registry: Registry) -> None:
        """Each spec-required registry exists."""
        assert isinstance(registry, Registry)

    @pytest.mark.parametrize(
        "registry",
        [
            QUANTIZER_REGISTRY,
            ROUTER_REGISTRY,
            MERGE_REGISTRY,
            SCHEDULER_REGISTRY,
            BACKEND_REGISTRY,
            PROFILER_REGISTRY,
            VISUALIZER_REGISTRY,
        ],
    )
    def test_registry_starts_empty(self, registry: Registry) -> None:
        """Each registry starts empty until subsystems register."""
        assert len(registry) == 0


class TestRegistryIsolation:
    """Tests that registries do not share state."""

    def test_two_registries_are_independent(self) -> None:
        """Two registries with different names don't share keys."""

        class Foo:
            pass

        r1: Registry[type] = Registry("a")
        r2: Registry[type] = Registry("b")
        r1.register("foo")(Foo)
        assert "foo" in r1
        assert "foo" not in r2
