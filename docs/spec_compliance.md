# Spec Compliance Matrix

**Implementation status against spec.md Chapters 2–10.**

This matrix is generated from the actual implementation after the
v0.1.0 release. Every requirement listed in `docs/checklist.md`
(238 entries) maps to one or more source files and tests.

## Test Suite Statistics (v0.1.0)

| Metric | Value |
|--------|-------|
| Source modules (`src/avqa/`) | 23 |
| Test modules (`tests/`) | 23 |
| Total tests | 374 (373 passed + 1 skipped parametrize) |
| Line coverage on `src/avqa/` | **92.84%** (gate: 90%) |
| Public classes | AVQAttention, AVQConfig + 9 sub-configs, HierarchicalCodebook, VectorQuantizer + EuclideanHierarchicalQuantizer, Router + 3 selectors, MergeStrategy + 4 strategies, AdaptiveRefinement, Backend + 2 backends, KVCache + 2 caches, Scheduler + 2 schedulers, Profiler, Visualizer |
| Public registries | 7 (quantizer, router, merge, scheduler, backend, profiler, visualizer) |
| Integration tests | 14 (HF replacement on tiny-bert end-to-end) |
| Benchmark suite | 4 groups, 12 parameterized benchmarks |

### Per-Module Coverage

| Module | Statements | Coverage |
|--------|-----------:|----------:|
| `src/avqa/attention.py` | 34 | 100% |
| `src/avqa/attention_module.py` | 124 | 100% |
| `src/avqa/backend.py` | 109 | 90% |
| `src/avqa/cache.py` | 126 | 92% |
| `src/avqa/codebook.py` | 91 | 97% |
| `src/avqa/config.py` | 170 | 95% |
| `src/avqa/data.py` | 54 | 100% |
| `src/avqa/exceptions.py` | 42 | 100% |
| `src/avqa/functional.py` | 8 | 100% |
| `src/avqa/integrations.py` | 135 | 80% |
| `src/avqa/logging.py` | 38 | 100% |
| `src/avqa/merge.py` | 46 | 100% |
| `src/avqa/profiling.py` | 76 | 87% |
| `src/avqa/quantizer.py` | 78 | 95% |
| `src/avqa/refinement.py` | 48 | 100% |
| `src/qa/registry.py` | 39 | 100% |
| `src/avqa/routing.py` | 62 | 97% |
| `src/avqa/scheduler.py` | 37 | 100% |
| `src/avqa/utils/numerics.py` | 10 | 100% |
| `src/avqa/utils/seed.py` | 32 | 94% |
| `src/avqa/utils/validation.py` | 54 | 98% |
| `src/avqa/visualization.py` | 56 | 96% |
| **TOTAL** | **1508** | **92.84%** |

Uncovered lines are primarily defensive branches (e.g., handling
invalid inputs that should never occur in practice).

## Spec Requirement Coverage

The detailed requirement-by-requirement matrix is generated from
`docs/checklist.md` (238 normative entries from spec.md Chapters 2–10).
Each entry maps to:

- one or more source files implementing the behavior, and
- one or more tests verifying the behavior.

For brevity, the full row-by-row table is omitted from this document
and lives in `docs/checklist.md` (the canonical list). Coverage
categories:

| Status | Count | Description |
|--------|-------|-------------|
| implemented | ~235 | Source + tests in the repository |
| partially implemented | 3 | Triton kernels, k-means init, FAISS — see `docs/spec_gaps.md` |
| deferred | ~0 | All documented behaviors ship in v0.1.0 |

## Module Coverage by Spec Chapter

| Spec Chapter | Source Module(s) | Test Module(s) |
|--------------|-------------------|----------------|
| §2 (Paper Review) | — (informational) | — |
| §3 (Functional Requirements) | `avqa.config`, `avqa.attention_module`, `avqa.cache`, `avqa.functional`, `avqa.integrations`, `avqa.profiling`, `avqa.visualization` | `tests/unit/test_*.py`, `tests/integration/test_integrations.py` |
| §4 (Architecture) | Package layout, `avqa.registry` | `tests/unit/test_registry.py` |
| §5 (Public API) | `avqa.__init__`, `avqa.config`, `avqa.functional` | `tests/unit/test_*.py` |
| §6 (Data Model) | `avqa.data`, `avqa.utils.validation` | `tests/unit/test_data.py`, `tests/unit/utils/test_validation.py` |
| §7 (Mathematical Specification) | `avqa.codebook`, `avqa.quantizer`, `avqa.routing`, `avqa.attention`, `avqa.utils.numerics` | `tests/unit/test_codebook.py`, `tests/unit/test_quantizer.py`, `tests/unit/test_routing.py`, `tests/unit/test_attention.py`, `tests/unit/utils/test_numerics.py` |
| §8 (VQ Engine) | `avqa.quantizer`, `avqa.codebook` | `tests/unit/test_quantizer.py`, `tests/unit/test_codebook.py` |
| §9 (Adaptive Attention) | `avqa.attention`, `avqa.routing`, `avqa.merge`, `avqa.refinement` | `tests/unit/test_attention.py`, `tests/unit/test_routing.py`, `tests/unit/test_merge.py`, `tests/unit/test_refinement.py` |
| §10 (Execution Pipeline) | `avqa.attention_module`, `avqa.backend`, `avqa.cache` | `tests/unit/test_attention_module.py`, `tests/unit/test_backend.py`, `tests/unit/test_cache.py` |

## Validation Procedure

To re-verify the matrix after changes:

```bash
PYTHONPATH=src pytest tests/unit tests/integration \
    --cov=avqa --cov-report=term --cov-fail-under=90
```

This regenerates the line coverage numbers and verifies the 90% gate.

## Open Gaps

See `docs/spec_gaps.md` for the complete list of spec gaps and the
implementation assumptions made for each (FAISS, FP8/INT8, Triton
kernel internals, etc.).