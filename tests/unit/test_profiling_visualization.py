"""Tests for avqa.profiling and avqa.visualization modules."""

from __future__ import annotations

import pytest
import torch

from avqa.profiling import Profiler
from avqa.routing import RoutingDecision
from avqa.visualization import (
    JSONVisualizer,
    TimelineEvent,
    TreeNode,
    Visualizer,
)


class TestProfilerSession:
    """Tests for the Profiler.session context manager."""

    def test_session_records_total_time(self) -> None:
        """Session end populates total_duration_ms."""
        profiler = Profiler()
        with profiler.session():
            pass
        assert profiler.report.total_duration_ms >= 0

    def test_stage_records_duration(self) -> None:
        """Stage timing records a non-negative duration."""
        profiler = Profiler()
        with profiler.session(), profiler.stage("test"):
            sum(range(100))  # body inside the stage context
        assert len(profiler.report.stage_timers) == 1
        assert profiler.report.stage_timers[0].name == "test"
        assert profiler.report.stage_timers[0].duration_ms >= 0

    def test_export_json(self) -> None:
        """JSON export produces a string with the expected keys."""
        profiler = Profiler()
        with profiler.session(), profiler.stage("s1"):
            pass
        data = profiler.report.to_dict()
        assert "stage_timers" in data
        assert "schema_version" in data
        assert data["schema_version"] == "avqa_profiler_v1"


class TestProfilerCounters:
    """Tests for the profiler's counter helpers."""

    def test_record_routing(self) -> None:
        """Routing decision is recorded with num_selected."""
        profiler = Profiler()
        decision = RoutingDecision(
            selected_indices=torch.tensor([[1, 3, 5]]),
            importance=torch.zeros(1, 1, 8),
        )
        profiler.record_routing(decision)
        assert len(profiler.report.routing_stats) == 1
        assert profiler.report.routing_stats[0]["num_selected"] == 3

    def test_record_refinement(self) -> None:
        """Refinement step is recorded with budget and num_refined."""
        profiler = Profiler()
        profiler.record_refinement(budget=4, num_refined=3)
        assert profiler.report.refinement_stats[-1] == {"budget": 4, "num_refined": 3}

    def test_record_cache_hit_miss(self) -> None:
        """Cache hits and misses are counted."""
        profiler = Profiler()
        profiler.record_cache_hit()
        profiler.record_cache_hit()
        profiler.record_cache_miss()
        assert profiler.report.cache_hits == 2
        assert profiler.report.cache_misses == 1

    def test_codebook_utilization(self) -> None:
        """Codebook utilization is stored per head."""
        profiler = Profiler()
        profiler.set_codebook_utilization({"head_0": 0.75, "head_1": 0.50})
        assert profiler.report.codebook_utilization["head_0"] == 0.75


class TestVisualizerInterface:
    """Tests for the abstract Visualizer interface."""

    def test_cannot_instantiate(self) -> None:
        """Visualizer is abstract; instantiation must raise TypeError."""
        # ``Abstract`` is the ABC-marker attribute set when a class has
        # unimplemented abstract methods. Asserting it directly avoids
        # the type checker's refusal to construct an abstract class.
        assert Visualizer.__abstractmethods__, (
            "Visualizer must declare unimplemented abstract methods"
        )
        with pytest.raises(TypeError):
            # Use ``object.__new__`` (which Python does allow even on
            # abstract classes) to verify the runtime error path.
            # mypy refuses ``object.__new__(Visualizer)`` so we
            # look the call up dynamically.
            Visualizer.__new__(Visualizer)

    def test_json_visualizer_is_subclass(self) -> None:
        """JSONVisualizer inherits from Visualizer."""
        assert issubclass(JSONVisualizer, Visualizer)


class TestJSONVisualizer:
    """Tests for the JSONVisualizer."""

    def test_refinement_tree(self) -> None:
        """Refinement tree renders to nested dict."""
        root = TreeNode(label="root", metadata={"count": 10})
        root.children = [TreeNode(label="child1"), TreeNode(label="child2")]
        viz = JSONVisualizer()
        out = viz.render_refinement_tree(root)
        assert out["label"] == "root"
        assert len(out["children"]) == 2

    def test_routing_path(self) -> None:
        """Routing path renders to dict with selected indices."""
        viz = JSONVisualizer()
        decision = RoutingDecision(
            selected_indices=torch.tensor([[[2, 4, 6]]]),
            importance=torch.zeros(1, 1, 8),
        )
        out = viz.render_routing_path(decision)
        assert out["num_selected"] == 3
        assert out["selected"] == [[[2, 4, 6]]]
        assert out["importance"] == [[[0.0] * 8]]

    def test_attention_heatmap(self) -> None:
        """Attention heatmap renders to a 2D matrix."""
        viz = JSONVisualizer()
        attn = torch.softmax(torch.randn(2, 4, 5), dim=-1)
        hm = viz.render_attention_heatmap(attn, title="test")
        assert hm.title == "test"
        assert len(hm.matrix) == 2
        assert len(hm.matrix[0]) == 4

    def test_codebook_utilization(self) -> None:
        """Codebook utilization renders parent + child fractions."""
        viz = JSONVisualizer()
        # [B=1, H=1, M_0=4]: 3 of 4 parents active => mean = 0.75
        parent = torch.tensor([[[1.0, 0.0, 2.0, 3.0]]])
        # [B=1, H=1, M_0=4, C=2]
        child = torch.zeros(1, 1, 4, 2)
        out = viz.render_codebook_utilization(parent, child)
        assert out["parent_utilization"] == [[0.75]]
        assert "child_utilization" in out

    def test_timeline(self) -> None:
        """Timeline renders events and total duration."""
        viz = JSONVisualizer()
        events = [
            TimelineEvent(name="step1", start_ms=0.0, duration_ms=5.0),
            TimelineEvent(name="step2", start_ms=5.0, duration_ms=3.0),
        ]
        out = viz.render_timeline(events)
        assert out["total_duration_ms"] == 8.0
        assert len(out["events"]) == 2
