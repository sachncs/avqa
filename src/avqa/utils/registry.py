"""Thread-safe factories for pluggable AVQA strategies."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar

from avqa.exceptions import AVQAError

if TYPE_CHECKING:
    from collections.abc import Callable

Strategy = TypeVar("Strategy")


class FactoryRegistry(Generic[Strategy]):
    """Resolve named strategy implementations without conditional dispatch."""

    def __init__(
        self,
        base_type: type[object],
        error_type: type[AVQAError],
        *,
        label: str,
    ) -> None:
        """Create a registry for implementations of ``base_type``."""
        self._base_type: type[object] = base_type
        self._error_type = error_type
        self._label = label
        self._factories: dict[str, Callable[..., Strategy]] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        factory: Callable[..., Strategy],
        *,
        replace: bool = False,
    ) -> None:
        """Register a named zero-argument strategy factory.

        Args:
            name: Stable, non-empty strategy identifier.
            factory: Callable returning an instance of the registered base type.
            replace: Permit replacing a previous registration.

        Raises:
            AVQAError: If the name, factory, or registration conflicts.

        """
        if not isinstance(name, str) or not name.strip():
            raise self._error_type(f"{self._label} name must be non-empty")
        if not callable(factory):
            raise self._error_type(f"{self._label} '{name}' factory must be callable")
        with self._lock:
            if name in self._factories and not replace:
                raise self._error_type(f"{self._label} '{name}' is already registered")
            self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> Strategy:
        """Construct a registered strategy and validate its type."""
        with self._lock:
            factory = self._factories.get(name)
        if factory is None:
            raise self._error_type(f"unknown {self._label}: {name!r}")
        try:
            instance = factory(**kwargs)
        except Exception as exc:
            raise self._error_type(f"{self._label} '{name}' failed during construction") from exc
        if not isinstance(instance, self._base_type):
            raise self._error_type(
                f"{self._label} '{name}' factory must return {self._base_type.__name__}"
            )
        return instance

    def is_registered(self, name: str) -> bool:
        """Return whether ``name`` currently has a registered factory."""
        with self._lock:
            return name in self._factories
