"""Tests for avqa.utils.seed module."""

from __future__ import annotations

import builtins
from collections.abc import Callable
import os
import random

import pytest
import torch

from avqa.exceptions import AVQAError, ConfigurationError
from avqa.utils.seed import deterministic_algorithms, seed_everything


class TestSeedEverything:
    """Tests for the seed_everything function."""

    def test_returns_seed(self) -> None:
        """Returns the seed that was applied."""
        assert seed_everything(42) == 42
        assert seed_everything(0) == 0

    def test_seeds_python_random(self) -> None:
        """Python random is seeded."""
        seed_everything(123)
        first = random.random()
        seed_everything(123)
        second = random.random()
        assert first == second

    def test_seeds_torch(self) -> None:
        """Torch CPU RNG is seeded."""
        seed_everything(7)
        first = torch.rand(3)
        seed_everything(7)
        second = torch.rand(3)
        assert torch.equal(first, second)

    def test_sets_pythonhashseed(self) -> None:
        """PYTHONHASHSEED environment variable is set."""
        seed_everything(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_seeds_numpy_when_available(self) -> None:
        """NumPy is seeded when importable."""
        np = pytest.importorskip("numpy")
        seed_everything(11)
        first = np.random.rand(3)
        seed_everything(11)
        second = np.random.rand(3)
        np.testing.assert_array_equal(first, second)

    def test_works_without_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Seeding works even when NumPy is not importable."""

        original_import: Callable[..., object] = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy" or name.startswith("numpy."):
                raise ImportError("numpy disabled for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = seed_everything(5)
        assert result == 5

    def test_negative_seed_raises(self) -> None:
        """Negative seeds raise ConfigurationError (AVQAError subclass)."""
        with pytest.raises(ConfigurationError, match="non-negative"):
            seed_everything(-1)
        with pytest.raises(AVQAError, match="non-negative"):
            seed_everything(-1)

    def test_default_seed(self) -> None:
        """Default seed is 0."""
        assert seed_everything() == 0

    def test_deterministic_mode_does_not_raise(self) -> None:
        """Deterministic mode can be enabled without immediate error."""
        result = seed_everything(0, deterministic=True)
        assert result == 0
        # Restore to non-deterministic to not affect other tests
        torch.use_deterministic_algorithms(False)


class TestDeterministicReproducibility:
    """Integration tests verifying reproducibility across seed cycles."""

    def test_full_reproducibility_cycle(self) -> None:
        """Complete seed cycle yields identical tensors."""
        seed_everything(2024)
        first = torch.randn(5, 5)
        seed_everything(2024)
        second = torch.randn(5, 5)
        assert torch.allclose(first, second)

    def test_different_seeds_produce_different_tensors(self) -> None:
        """Different seeds produce different tensors."""
        seed_everything(1)
        first = torch.randn(3, 3)
        seed_everything(2)
        second = torch.randn(3, 3)
        assert not torch.equal(first, second)

    def test_deterministic_context_restores_torch_state(self) -> None:
        """Scoped deterministic mode does not leak into the caller."""
        torch.use_deterministic_algorithms(False, warn_only=True)
        with deterministic_algorithms(True):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

    def test_deterministic_context_restores_state_after_exception(self) -> None:
        """Deterministic mode is restored when the block fails."""
        torch.use_deterministic_algorithms(False, warn_only=False)
        with pytest.raises(RuntimeError, match="failure"), deterministic_algorithms(True):
            raise RuntimeError("failure")
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
