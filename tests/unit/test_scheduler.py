"""Tests for avqa.scheduler module."""

from __future__ import annotations

import pytest
import torch

from avqa.exceptions import AVQAError, ConfigurationError, RoutingError
from avqa.scheduler import AdaptiveScheduler, DefaultScheduler, Scheduler


class TestDefaultScheduler:
    """Tests for the fixed-budget scheduler."""

    def test_default_budget(self) -> None:
        """Default budget is 8."""
        s = DefaultScheduler()
        assert s.budget == 8

    def test_returns_constant_budget(self) -> None:
        """budget_for returns the configured budget."""
        s = DefaultScheduler(budget=4)
        assert s.budget_for(torch.zeros(1, 1, 64)) == 4
        assert s.budget_for(torch.ones(2, 4, 32)) == 4

    def test_invalid_budget(self) -> None:
        """budget <= 0 raises."""
        with pytest.raises(RoutingError):
            DefaultScheduler(budget=0)


class TestAdaptiveScheduler:
    """Tests for the entropy-driven adaptive scheduler."""

    def test_low_entropy_returns_max(self) -> None:
        """Focused (low-entropy) importance → max_budget."""
        s = AdaptiveScheduler(min_budget=4, max_budget=32)
        importance = torch.zeros(1, 1, 64)
        importance[0, 0, 0] = 100.0
        budget = s.budget_for(importance)
        assert int(budget.max().item()) == 32

    def test_high_entropy_returns_min(self) -> None:
        """Spread (high-entropy) importance → min_budget."""
        s = AdaptiveScheduler(min_budget=4, max_budget=32)
        importance = torch.ones(1, 1, 64)
        budget = s.budget_for(importance)
        assert int(budget.max().item()) == 4

    def test_returns_per_bh_tensor(self) -> None:
        """budget_for returns a per-(B, H) tensor for [B, H, M_0] input."""
        s = AdaptiveScheduler(min_budget=2, max_budget=8)
        importance = torch.zeros(2, 4, 16)
        budget = s.budget_for(importance)
        assert isinstance(budget, torch.Tensor)
        assert budget.shape == (2, 4)
        assert budget.dtype == torch.int64

    def test_per_bh_budgets_respond_to_entropy(self) -> None:
        """Two heads with different entropy get different per-(B, H) budgets."""
        s = AdaptiveScheduler(min_budget=2, max_budget=8)
        importance = torch.zeros(2, 2, 16)
        importance[0, 0, 0] = 100.0  # focused head
        importance[1, 0, :] = 1.0  # spread head
        importance[0, 1, :] = 1.0  # spread head
        importance[1, 1, 0] = 100.0  # focused head
        budget = s.budget_for(importance)
        assert budget[0, 0].item() == 8
        assert budget[0, 1].item() == 2
        assert budget[1, 0].item() == 2
        assert budget[1, 1].item() == 8

    def test_invalid_min_budget(self) -> None:
        """min_budget <= 0 raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            AdaptiveScheduler(min_budget=0)

    def test_invalid_max_budget(self) -> None:
        """max_budget < min_budget raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            AdaptiveScheduler(min_budget=10, max_budget=4)

    def test_invalid_entropy_threshold(self) -> None:
        """entropy_threshold outside (0, 1] raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            AdaptiveScheduler(entropy_threshold=0.0)
        with pytest.raises(ConfigurationError):
            AdaptiveScheduler(entropy_threshold=2.0)

    def test_raises_are_avqaerrors(self) -> None:
        """All scheduler raises are AVQAError subclasses (catchable)."""
        with pytest.raises(AVQAError):
            DefaultScheduler(budget=0)
        with pytest.raises(AVQAError):
            AdaptiveScheduler(min_budget=0)


class TestAbstractInterface:
    """Tests for the abstract Scheduler base."""

    def test_cannot_instantiate(self) -> None:
        """Scheduler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Scheduler.__new__(Scheduler)

    def test_subclass_relationship(self) -> None:
        """Concrete schedulers inherit from Scheduler."""
        assert issubclass(DefaultScheduler, Scheduler)
        assert issubclass(AdaptiveScheduler, Scheduler)
