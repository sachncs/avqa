"""Visualization subsystem for AVQA (spec §3.18, §5.16).

ponytail: visualization is rendered as JSON-able data structures.
Real matplotlib/graphviz rendering is optional and not loaded by
default; the data is what consumers (notebooks, dashboards) consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from avqa.registry import VISUALIZER_REGISTRY


@dataclass
class TreeNode:
    """A node in the refinement tree (spec §3.18)."""

    label: str
    children: list[TreeNode] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class HeatmapData:
    """Data for an attention heatmap (spec §3.18)."""

    matrix: list[list[float]]
    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    title: str = ""


@dataclass
class TimelineEvent:
    """One event in an execution timeline (spec §3.18)."""

    name: str
    start_ms: float
    duration_ms: float
    category: str = ""


class Visualizer(ABC):
    """Abstract visualizer (spec §3.18, §5.16)."""

    @abstractmethod
    def render_refinement_tree(self, root: TreeNode) -> dict[str, object]:
        """Render a refinement tree to a JSON-able dict."""

    @abstractmethod
    def render_routing_path(self, decision: object) -> dict[str, object]:
        """Render a routing decision to a JSON-able dict."""

    @abstractmethod
    def render_attention_heatmap(
        self,
        attention: torch.Tensor,
        title: str = "Attention",
    ) -> HeatmapData:
        """Render an attention heatmap (spec §3.18)."""

    @abstractmethod
    def render_codebook_utilization(
        self,
        parent_counts: torch.Tensor,
        child_counts: torch.Tensor,
    ) -> dict[str, object]:
        """Render codebook utilization (spec §3.18)."""

    @abstractmethod
    def render_timeline(self, events: list[TimelineEvent]) -> dict[str, object]:
        """Render an execution timeline (spec §3.18)."""


class JSONVisualizer(Visualizer):
    """Default JSON-only visualizer (no external deps)."""

    def render_refinement_tree(self, root: TreeNode) -> dict[str, object]:
        """Render a refinement tree as nested dicts."""
        return tree_to_dict(root)

    def render_routing_path(self, decision: object) -> dict[str, object]:
        """Render a routing decision as a JSON dict."""
        selected = getattr(decision, "selected_indices", None)
        importance = getattr(decision, "importance", None)
        return {
            "type": "routing",
            "num_selected": int(getattr(decision, "num_selected", 0)),
            "selected": selected.tolist() if selected is not None else [],
            "importance": importance.tolist() if importance is not None else [],
        }

    def render_attention_heatmap(
        self,
        attention: torch.Tensor,
        title: str = "Attention",
    ) -> HeatmapData:
        """Render an attention heatmap as a 2D matrix + labels."""
        return HeatmapData(
            matrix=attention.float().tolist(),
            x_labels=[str(i) for i in range(attention.shape[-1])],
            y_labels=[str(i) for i in range(attention.shape[-2])],
            title=title,
        )

    def render_codebook_utilization(
        self,
        parent_counts: torch.Tensor,
        child_counts: torch.Tensor,
    ) -> dict[str, object]:
        """Render codebook utilization fractions per head."""
        parent_util = (parent_counts > 0).float().mean(dim=-1).tolist()
        child_util = (child_counts > 0).float().mean(dim=(-2, -1)).tolist()
        return {
            "type": "codebook_utilization",
            "parent_utilization": parent_util,
            "child_utilization": child_util,
            "dead_parents": (parent_counts == 0).sum(dim=-1).tolist(),
            "dead_children": (child_counts == 0).sum(dim=-1).tolist(),
        }

    def render_timeline(self, events: list[TimelineEvent]) -> dict[str, object]:
        """Render a timeline as ordered events."""
        return {
            "type": "timeline",
            "events": [e.__dict__ for e in events],
            "total_duration_ms": max(
                (e.start_ms + e.duration_ms for e in events),
                default=0.0,
            ),
        }


def tree_to_dict(node: TreeNode) -> dict[str, object]:
    """Recursively convert a TreeNode to a JSON-able dict."""
    return {
        "label": node.label,
        "metadata": dict(node.metadata),
        "children": [tree_to_dict(c) for c in node.children],
    }


VISUALIZER_REGISTRY.register("json")(JSONVisualizer)  # type: ignore[arg-type]


__all__ = [
    "HeatmapData",
    "JSONVisualizer",
    "TimelineEvent",
    "TreeNode",
    "Visualizer",
    "tree_to_dict",
]
