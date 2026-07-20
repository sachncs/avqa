"""Profiling subsystem for AVQA (spec §3.17, §5.15).

ponytail: the profiler is one Profiler class that collects stage
timings, memory, and per-step statistics. Report export is JSON-only;
visual rendering lives in avqa.visualization.
"""
from __future__ import annotations



from collections.abc import Iterator  # noqa: TC003
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import IO

import torch

from avqa.logging import get_logger

_logger = get_logger("profiling")


@dataclass
class StageTimer:
    """Per-stage timing record."""

    name: str
    duration_ms: float
    memory_bytes: int = 0


@dataclass
class ProfilerReport:
    """Aggregated profiling data (spec §3.17).

    Attributes:
        stage_timers: One entry per executed stage.
        total_duration_ms: Sum of all stage durations.
        peak_memory_bytes: Peak resident memory observed.
        routing_stats: Per-step routing decisions (selected parents etc.).
        refinement_stats: Per-step refinement budget and active counts.
        codebook_utilization: Per-head codeword assignment fractions.
        cache_hits / cache_misses: Cache usage counters.
        total_flops: M7 — total FLOPs consumed by the pipeline (spec §3.17).
            Estimated from sequence lengths, codebook sizes, and refinement
            budget; not an exact hardware FLOP counter.
    """

    stage_timers: list[StageTimer] = field(default_factory=list)
    total_duration_ms: float = 0.0
    peak_memory_bytes: int = 0
    routing_stats: list[dict[str, object]] = field(default_factory=list)
    refinement_stats: list[dict[str, object]] = field(default_factory=list)
    codebook_utilization: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    total_flops: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict (JSON-compatible)."""
        return {
            "stage_timers": [t.__dict__ for t in self.stage_timers],
            "total_duration_ms": self.total_duration_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "routing_stats": self.routing_stats,
            "refinement_stats": self.refinement_stats,
            "codebook_utilization": self.codebook_utilization,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "schema_version": "avqa_profiler_v1",
        }


class Profiler:
    """Lightweight per-stage profiler (spec §3.17, §5.15).

    Usage::

        profiler = Profiler()
        with profiler.session():
            with profiler.stage("parent_attention"):
                ...
            with profiler.stage("refinement"):
                ...

    The profiler does NOT alter algorithmic behavior (spec §4.6.5); it
    only observes.
    """

    @classmethod
    def create(cls, name: str = "default") -> Profiler:
        """Factory: resolve ``name`` to a :class:`Profiler` instance.

        Args:
            name: ``"default"`` (the only profiler shipped).
        """
        if name == "default":
            return cls()
        msg = f"unknown profiler: {name!r}"
        raise ValueError(msg)

    def __init__(self) -> None:
        self.report = ProfilerReport()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Open a profiling session."""
        self.session_start = time.perf_counter()
        try:
            yield
        finally:
            self.report.total_duration_ms = (time.perf_counter() - self.session_start) * 1000.0
            _logger.debug(
                "Profiler session: %.2f ms, %d stages",
                self.report.total_duration_ms,
                len(self.report.stage_timers),
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage."""
        start = time.perf_counter()
        mem_before = peak_memory_bytes() if torch.cuda.is_available() else 0
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            mem_after = peak_memory_bytes() if torch.cuda.is_available() else 0
            self.report.stage_timers.append(
                StageTimer(
                    name=name,
                    duration_ms=duration_ms,
                    memory_bytes=max(mem_after - mem_before, 0),
                )
            )
            if torch.cuda.is_available():
                self.report.peak_memory_bytes = max(
                    self.report.peak_memory_bytes,
                    torch.cuda.max_memory_allocated(),
                )

    def record_routing(self, decision: object) -> None:
        """Record a routing decision summary."""
        selected = getattr(decision, "selected_indices", None)
        if selected is None:
            return
        self.report.routing_stats.append(
            {
                "num_selected": int(getattr(decision, "num_selected", 0)),
            }
        )

    def record_refinement(self, budget: int, num_refined: int) -> None:
        """Record a refinement step summary."""
        self.report.refinement_stats.append(
            {"budget": budget, "num_refined": num_refined},
        )

    def set_codebook_utilization(self, head_utilization: dict[str, float]) -> None:
        """Record per-head codebook utilization."""
        self.report.codebook_utilization.update(head_utilization)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.report.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.report.cache_misses += 1

    def export_json(self, path: str | Path | IO[str]) -> None:
        """Export the report to a JSON file (spec §3.17 reproducibility).

        Args:
            path: File path or open file handle.
        """
        data = json.dumps(self.report.to_dict(), indent=2)
        if isinstance(path, (str, Path)):
            with Path(path).open("w") as f:
                f.write(data)
        else:
            path.write(data)


def peak_memory_bytes() -> int:
    """Return current peak CUDA memory in bytes; 0 if CUDA unavailable."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


__all__ = ["Profiler", "ProfilerReport", "StageTimer", "peak_memory_bytes"] 
