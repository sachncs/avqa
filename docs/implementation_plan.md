# AVQA Implementation Plan

## Disclaimer

This is an independent implementation of the **Adaptive Vector Quantized
Attention (AVQ-Attention)** algorithm. The author of this codebase is **not**
an author of the reference paper and is not affiliated with the paper's authors
or their institutions. This project is an open-source community effort to
provide a production-quality software implementation of the algorithm
described in the paper.

Reference paper:

> Adaptive Vector Quantized Attention (AVQ-Attention)
> <https://arxiv.org/html/2607.12789v1>

This implementation follows the public specification only. Any deviation from
the paper is documented in `docs/spec_gaps.md`.

---

## 1. Scope

AVQA implements Adaptive Vector Quantized Attention as a drop-in attention
backend for PyTorch-based Transformer architectures. It provides:

- A pure PyTorch reference implementation.
- An online-softmax PyTorch implementation following the FlashAttention-style
  tiling pattern.
- An optimized Triton backend (CUDA-only at runtime, fully implemented).
- Hugging Face Transformers integration.
- vLLM integration (paged attention, continuous batching, prefix caching).
- FlashAttention interop.
- xFormers interop.
- Profiling, visualization, benchmarking, and serialization tooling.

The authoritative source of requirements is `spec.md`. Anything not specified
therein is recorded as an **implementation assumption** in `docs/spec_gaps.md`.

---

## 2. Guiding Principles (from spec §4)

1. **Correctness before optimization.** The reference PyTorch implementation is
   the source of truth. Optimized kernels must produce numerically equivalent
   results within tolerance.
2. **Paper fidelity.** Enhancements are opt-in and isolated from the
   reference path.
3. **Modular design.** Quantizers, codebooks, routers, refinement strategies,
   schedulers, merge strategies, and backends are interchangeable.
4. **Explicit configuration.** No hidden global state.
5. **Progressive optimization.** Reference → online-softmax → Triton →
   multi-GPU.
6. **Transparency.** Tensor shapes, complexity, and paper references are
   documented per public function.
7. **Extensibility.** New strategies are registered, not monkey-patched.

---

## 3. Layered Architecture (spec §4.3)

```
Applications / User Code
       │
       ▼
Framework Integrations (HF, vLLM, FlashAttention, xFormers)
       │
       ▼
Public AVQA API Layer
       │
       ▼
Core Attention Pipeline
       │
       ▼
Quantization · Routing · Refinement
       │
       ▼
Backend Abstraction (Torch / Triton)
       │
       ▼
Hardware (CPU / GPU)
```

Each layer has a single responsibility and communicates through documented
interfaces only. Reverse imports are forbidden.

---

## 4. Subsystem Inventory (spec §4.5)

| Subsystem     | Module                  | Status |
|---------------|--------------------------|--------|
| Attention     | `avqa.attention`         | M10 |
| Quantizer     | `avqa.quantizer`         | M4 |
| Codebook      | `avqa.codebook`          | M3 |
| Router        | `avqa.routing`           | M5 |
| Refinement    | `avqa.refinement`        | M7 |
| Merge         | `avqa.merge`             | M6 |
| Scheduler     | `avqa.scheduler`         | M9 |
| Backend       | `avqa.backend`           | M8 |
| Cache         | `avqa.cache`             | M9 |
| Profiling     | `avqa.profiling`         | M12 |
| Visualization | `avqa.visualization`     | M12 |
| Integration   | `avqa.integrations`      | M13 |
| Functional    | `avqa.functional`        | M11 |
| Config        | `avqa.config`            | M2 |
| Data          | `avqa.data`              | M1 |
| Exceptions    | `avqa.exceptions`        | M1 |
| Logging       | `avqa.logging`           | M1 |
| Utils         | `avqa.utils`             | M1 |
| Registry      | `avqa.registry`          | M1 |

---

## 5. Mathematical Mapping (spec §2.11, §7.20)

| Mathematical Concept | AVQA Component |
|----------------------|----------------|
| Codebook             | `HierarchicalCodebook` |
| Quantizer            | `VectorQuantizer` |
| Routing              | `Router` |
| Adaptive refinement  | `AdaptiveRefinement` |
| Attention computation| `AVQAttention` |
| Probability merge    | `MergeStrategy` |
| Scheduling policy    | `Scheduler` |
| Aggregated values    | `VectorQuantizer` (precompute output) |
| Importance           | `ImportanceEstimator` |
| Correcting attention | `CorrectionOperator` |
| Online softmax       | `OnlineSoftmaxState` |
| Hierarchical codebook| `HierarchicalCodebook` (mean constraint) |

---

## 6. Development Workflow

1. Each commit corresponds to exactly one TODO entry in `TODO.md`.
2. Before every commit:
   - `ruff check .`
   - `ruff format --check .`
   - `black --check .`
   - `mypy src/avqa`
   - `pytest tests/unit -q`
   - `pytest tests/integration -q`
3. Each task writes production code, tests, and documentation.
4. Tests must include: happy path, edge cases, invalid inputs, numerical
   correctness, serialization, deterministic execution.
5. The TODO entry's commit SHA is recorded in `TODO.md` immediately after the
   commit.
6. After each milestone, `docs/spec_compliance.md` is regenerated.

---

## 7. Testing Strategy

| Layer        | Test root                    | Tooling |
|--------------|------------------------------|---------|
| Unit         | `tests/unit/`                | pytest |
| Integration  | `tests/integration/`         | pytest |
| Reference    | `tests/reference/`           | pytest |
| Performance  | `tests/performance/`         | pytest-benchmark |
| Property     | inline (Hypothesis)          | hypothesis |

Coverage target: ≥ 90% line coverage on `src/avqa/` core.

CI matrix:

- **CPU**: lint + type + unit + integration + coverage.
- **GPU (optional)**: GPU unit tests, performance benchmarks, framework
  integration tests requiring CUDA.

---

## 8. Implementation Order (high-level)

See `docs/milestone_plan.md` for full milestones. Order:

```
M0  Scaffolding
M1  Foundations (exceptions, logging, utils, data, registry)
M2  Configuration
M3  Codebook
M4  Quantization
M5  Routing
M6  Merge + Correction + Online softmax
M7  Refinement
M8  Backend
M9  Scheduler + Cache
M10 Attention module
M11 Functional API
M12 Profiling + Visualization
M13 Framework Integrations
M14 Benchmarks + Examples
M15 Release + Compliance
```

---

## 9. Risk Register

| Risk | Mitigation |
|------|------------|
| CUDA/Triton not testable on macOS | Reference + online-softmax paths run on CPU/MPS; CUDA-only paths guarded with `importorskip` and `cuda.is_available()`. |
| Heavy framework deps (vLLM, FlashAttention, xFormers) not installed by default | Optional extras in `pyproject.toml`. Integration modules lazy-import heavy deps. |
| Numerical differences across backends | Naive reference path is the canonical source; tolerances documented per dtype. |
| Spec gaps (Ch 11+) | Documented in `docs/spec_gaps.md` with explicit implementation assumptions. |

---

## 10. Acceptance Criteria

The library is considered complete when:

- Every SHALL statement in `spec.md` (Chapters 3–10) has at least one source
  file and at least one test traceable in `docs/spec_compliance.md`.
- CI passes on CPU and GPU matrices.
- Coverage on `src/avqa/` ≥ 90%.
- All public classes have Google-style docstrings with: purpose, parameters,
  returns, raises, tensor shapes, supported dtypes, complexity, example.
- Reference PyTorch output matches within documented tolerance for FP32,
  FP16, BF16.
- Hugging Face replacement test loads a tiny model, replaces attention, runs
  forward, verifies weight preservation.
- vLLM integration test exercises paged attention.
- FlashAttention interop test passes within tolerance.
- xFormers interop test passes.
- All benchmarks run reproducibly from a fixed seed.
