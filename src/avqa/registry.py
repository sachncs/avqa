"""Registry mechanism for AVQA (spec §5.10).

Each registry is a category-specific mapping from string name to class.
New implementations register without modifying existing source files.

Usage::

    from avqa.registry import Registry

    BACKENDS = Registry("backend")


    @BACKENDS.register("torch")
    class TorchBackend: ...


    cls = BACKENDS.get("torch")
"""
from __future__ import annotations



from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple, type-hinted name → class registry.

    Args:
        name: Human-readable category (e.g., ``"quantizer"``, ``"router"``).
            Used in error messages and introspection.

    Example:
        >>> r: Registry[type] = Registry("demo")
        >>> class Foo: ...
        >>> r.register("foo")(Foo)
        >>> r.get("foo") is Foo
        True
        >>> "foo" in r
        True
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.items: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers a class under ``key``.

        Args:
            key: Lookup name. Must be unique within this registry.

        Returns:
            A decorator that returns the class unchanged after registering it.

        Raises:
            ValueError: If ``key`` is already registered.

        Example:
            >>> r = Registry("example")
            >>> @r.register("foo")
            ... class Foo:
            ...     pass
            >>> r.get("foo") is Foo
            True
        """

        def decorator(cls: type[T]) -> type[T]:
            if key in self.items:
                msg = f"{self.name}: '{key}' is already registered ({self.items[key].__name__})"
                raise ValueError(msg)
            self.items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Return the class registered under ``key``.

        Args:
            key: Lookup name.

        Returns:
            The registered class.

        Raises:
            KeyError: If ``key`` is not registered. The message lists
                available keys to help debugging.
        """
        if key not in self.items:
            available = ", ".join(sorted(self.items)) or "<empty>"
            msg = f"{self.name}: unknown key '{key}'. Available: {available}"
            raise KeyError(msg)
        return self.items[key]

    def try_get(self, key: str) -> type[T] | None:
        """Return the registered class or ``None`` if not found."""
        return self.items.get(key)

    def keys(self) -> tuple[str, ...]:
        """Return all registered keys."""
        return tuple(self.items)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.items

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, items={list(self.items)})"


# Categories defined by spec §5.10. They are empty here; subsystems fill
# them in via @REGISTRY.register(...) at import time.

QUANTIZER_REGISTRY: Registry[type] = Registry("quantizer")
ROUTER_REGISTRY: Registry[type] = Registry("router")
MERGE_REGISTRY: Registry[type] = Registry("merge")
SCHEDULER_REGISTRY: Registry[type] = Registry("scheduler")
BACKEND_REGISTRY: Registry[type] = Registry("backend")
PROFILER_REGISTRY: Registry[type] = Registry("profiler")
VISUALIZER_REGISTRY: Registry[type] = Registry("visualizer")


__all__ = [
    "BACKEND_REGISTRY",
    "MERGE_REGISTRY",
    "PROFILER_REGISTRY",
    "QUANTIZER_REGISTRY",
    "ROUTER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "VISUALIZER_REGISTRY",
    "Registry",
]
