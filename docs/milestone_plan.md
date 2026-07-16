# Milestone Plan

This document breaks the implementation into 16 milestones (M0–M15). Each
milestone has a defined deliverable, exit criteria, and reference issues.

## Milestone Table

| Milestone | Title | Scope | TODO Range | Exit Criteria |
|-----------|-------|-------|------------|---------------|
| M0 | Scaffolding | pyproject, lint/type/test config, CI stub, directory skeleton | TASK-0007..TASK-0017 | `pip install -e .` succeeds; `make lint type test` runs. |
| M1 | Foundations | Exceptions, logging, utils, data contracts, registry | TASK-0018..TASK-0035 | All foundation modules importable; all unit tests pass; coverage ≥ 90%. |
| M2 | Configuration | AVQConfig and sub-configs with validation, serialization, versioning | TASK-0036..TASK-0047 | Round-trip JSON serialization works for all configs; immutability enforced; equality tested. |
| M3 | Codebook | Hierarchical codebook, mean constraint, init, EMA, stats | TASK-0048..TASK-0055 | Parent-child mean invariant holds after every operation; serialization round-trip works. |
| M4 | Quantization | Two-stage hierarchical VQ, value aggregation, EMA, stats | TASK-0056..TASK-0065 | Counts conservation invariant; sum-of-values invariant; deterministic mode tested. |
| M5 | Routing | Importance estimator, TopP/Threshold/Budget selectors, stats | TASK-0066..TASK-0072 | Importance derivation matches §7.10 exactly; tie-breaking deterministic. |
| M6 | Merge + Correction | Merge strategies, online softmax, correction, parent logit recovery | TASK-0073..TASK-0084 | All merge strategies numerically tested; correction preserves normalization invariant; parent logit recovery identity holds. |
| M7 | Refinement | Adaptive refinement orchestrator | TASK-0085..TASK-0089 | Refinement bounded by budget; child expansion correct; correction applied to selected subset only. |
| M8 | Backend | Torch reference + online softmax + Triton + factory/selection | TASK-0090..TASK-0100 | Naive and online-softmax agree within tolerance; backend selection respects config. |
| M9 | Scheduler + Cache | Default + adaptive scheduler, in-memory + paged cache | TASK-0101..TASK-0108 | Cache supports incremental updates, reset, serialization; scheduler respects budget. |
| M10 | Attention Module | Pipeline orchestration + nn.Module wrapper | TASK-0109..TASK-0123 | End-to-end forward passes for FP32/FP16/BF16; gradients flow; causal mask works. |
| M11 | Functional API | Stateless entry point | TASK-0124..TASK-0126 | Functional API stateless; batched inputs supported. |
| M12 | Profiling + Visualization | Profiler, metrics, reporters; tree, heatmap, timeline, utilization | TASK-0127..TASK-0135 | Profiler reports in JSON and human-readable format; visualizers produce valid outputs. |
| M13 | Integrations | HF, vLLM, FlashAttention, xFormers | TASK-0136..TASK-0148 | HF replacement e2e test passes on tiny model; vLLM paged/batching/prefix tests pass; FA and xformers interop tests pass. |
| M14 | Benchmarks + Examples | pytest-benchmark, sweeps, examples | TASK-0149..TASK-0157 | All benchmarks reproducible from fixed seed; examples run end-to-end. |
| M15 | Release + Compliance | Changelog, release notes, spec_compliance.md, packaging | TASK-0158..TASK-0170 | Every SHALL statement traceable; `python -m build` produces wheel+sdist; pre-commit hooks pass. |

## Milestone Dependencies

```
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11
                          ↑                       │
                          │                       ▼
                          └── M12 (profiling/visualization in parallel with M7-M10)
                                                  │
                                                  ▼
                                                 M13 → M14 → M15
```

## Acceptance Gate per Milestone

Before declaring a milestone complete, all of the following MUST hold:

1. All TODO entries in the milestone are marked `[x]` with commit SHAs.
2. `ruff check .` passes with zero warnings.
3. `ruff format --check .` passes.
4. `black --check .` passes.
5. `mypy src/avqa` passes with strict mode.
6. `pytest tests/unit -q --cov=avqa --cov-fail-under=90` passes.
7. `pytest tests/integration -q` passes (or is skipped with documented reason).
8. `docs/spec_compliance.md` is regenerated and includes the milestone's
   new requirements.
9. `CHANGELOG.md` includes an entry for the milestone.
10. README example(s) relevant to the milestone run successfully.

## Milestone Burn-Down Tracking

Each milestone tracks:

- Tasks planned: count of TODO entries in range.
- Tasks completed: count of `[x]` entries.
- Coverage delta: coverage before/after.
- Open spec gaps: count of partially implemented requirements affecting the milestone.

These metrics are reported in `docs/spec_compliance.md` after each milestone.

## Critical Path Length Estimate

From M0 through M15, the critical path comprises approximately **150
sequential atomic commits** (the minimum chain through the dependency graph).
With ~5 minutes per atomic commit on average for design + implement + test +
documentation + validation, the critical path is approximately 12–15 hours
of focused engineering work, though realistic calendar time is multiple
weeks given review, refactoring, and bug investigation overhead.

## Risk Mitigation per Milestone

| Milestone | Risk | Mitigation |
|-----------|------|------------|
| M0 | Wrong packaging layout | Adopt `src/` layout (PEP 517); validates install. |
| M1 | Inconsistent exception types | Single base `AVQAError` with documented subclasses. |
| M2 | Mutable config state | `@dataclass(frozen=True)`; mutability tests required. |
| M3 | Mean constraint violation | Invariant property test on every training step. |
| M4 | Non-deterministic assignment | Determinism test using seeded RNG. |
| M5 | Importance derivation error | Property test: importance = sum of A_ij * n_j / Z_i. |
| M6 | Normalization broken after correction | Property test: attention probs sum to 1 after each correction. |
| M7 | Refinement over-budget | Cost assertion in test. |
| M8 | Backend numerical drift | Tolerated-agreement test against naive reference. |
| M9 | Cache concurrency | Single-threaded API; thread-safety documented. |
| M10 | Gradient instability | Gradient flow tests + bf16 backward test. |
| M11 | Hidden state in functional API | Module-level fixtures verify no globals mutated. |
| M12 | Profiler overhead | Profiler is opt-in; overhead benchmark included. |
| M13 | Framework version drift | Optional extras; versioned in `pyproject.toml`. |
| M14 | Non-reproducible benchmarks | Seed helper; benchmark reproducibility test. |
| M15 | Missing spec trace | Every SHALL traced in `docs/spec_compliance.md`. |

## Post-Milestone Reviews

After each milestone:

1. Re-read the corresponding spec sections (per `docs/spec_compliance.md`).
2. Identify any partially implemented requirements.
3. Identify any undocumented behavior.
4. Create GitHub issues for newly discovered work.
5. Update `docs/spec_compliance.md` status column.
6. Tag the commit: `git tag -a m<N>-complete -m "Milestone M<N> complete"`.

## Stretch Goals (post-M15)

These are explicitly out of scope for v1.0 but may be added in minor
releases:

- Distributed tensor parallelism (spec §6.16).
- Multi-GPU execution.
- Speculative decoding integration with vLLM (spec §3.15).
- FP8 / INT8 dtypes (spec §6.9 optional).
- DEAD-code resampling (spec §8.11 optional).
- FAISS-accelerated assignment (spec §3.7 optional).
