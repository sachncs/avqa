"""Tests for user-registered polymorphic strategies."""

from __future__ import annotations

import pytest
import torch

from avqa.config import AVQConfig, MergeConfig, RoutingConfig
from avqa.exceptions import RoutingError
from avqa.merge import MergeStrategy, ProbabilityMerge
from avqa.routing import Router, RoutingDecision, TopPRouter


class CustomRouter(TopPRouter):
    """Minimal user-defined router used to verify extension dispatch."""


class CustomMerge(ProbabilityMerge):
    """Minimal user-defined merge strategy for configuration tests."""


def test_registered_router_is_created_and_dispatched_polymorphically() -> None:
    """A custom router is selectable through the same public factory."""
    name = "test_custom_router"
    Router.register(name, CustomRouter)

    router = Router.create(name)

    assert isinstance(router, CustomRouter)
    assert isinstance(router, Router)
    scores = torch.tensor([[[0.1, 0.9, 0.2]]])
    decision = router.select(scores, budget=1)
    assert isinstance(decision, RoutingDecision)
    assert decision.selected_indices.tolist() == [[[1]]]


def test_registered_strategies_are_accepted_by_public_configuration() -> None:
    """Registered router and merge implementations can be selected by config."""
    router_name = "test_config_router"
    merge_name = "test_config_merge"
    Router.register(router_name, CustomRouter)
    MergeStrategy.register(merge_name, CustomMerge)

    config = AVQConfig(
        routing=RoutingConfig(strategy=router_name),
        merge=MergeConfig(strategy=merge_name),
    )

    assert Router.is_registered(config.routing.strategy)
    assert isinstance(Router.create(config.routing.strategy), CustomRouter)
    assert isinstance(MergeStrategy.create(config.merge.strategy), CustomMerge)


def test_registry_rejects_duplicate_names_without_replace() -> None:
    """Registrations are not silently overwritten."""
    name = "test_duplicate_router"
    Router.register(name, CustomRouter)

    with pytest.raises(RoutingError, match="already registered"):
        Router.register(name, CustomRouter)


def test_registry_rejects_factory_returning_wrong_type() -> None:
    """Factory output is checked against its declared strategy interface."""
    name = "test_invalid_router"

    def invalid_factory() -> Router:
        return object()  # type: ignore[return-value]

    Router.register(name, invalid_factory)

    with pytest.raises(RoutingError, match="must return Router"):
        Router.create(name)
