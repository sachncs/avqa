# TODO.md

> **Project:** AVQA – Adaptive Vector Quantized Attention
>
> **Purpose:** Atomic implementation backlog.
>
> This document is the authoritative engineering work queue for AVQA.
>
> Every source code modification MUST originate from a TODO item.
>
> Every TODO item MUST correspond to exactly one atomic Git commit.
>
> Every completed TODO MUST be fully implemented, tested, documented,
> verified, and benchmarked where applicable.
>
> **Ledger policy (v2):** tasks discovered in the new engineering pass
> use the `TASK-N.MMM` numbering convention. Historical M1–M15 work
> (commits `86dcaed`..`97608f7`) is summarized in the "Historical
> completeness ledger" below; that record is append-only and never
> rewritten.

---

## Status Legend

| Marker | Meaning                                 |
|--------|-----------------------------------------|
| `[ ]`  | Pending — work has not started.         |
| `[~]`  | In progress — implementation underway.  |
| `[x]`  | Completed — commit SHA recorded.        |
| `[!]`  | Blocked — note in commit field.         |

---

## Workflow (SPEC §Implementation Workflow)

```text id="g67q3s"
SPEC.md
      │
      ▼
Requirement
      │
      ▼
TODO.md
      │
      ▼
Implementation
      │
      ▼
Unit Tests
      │
      ▼
Integration Tests
      │
      ▼
Documentation
      │
      ▼
Benchmark (if applicable)
      │
      ▼
Verification
      │
      ▼
Atomic Commit
```

No code may bypass this workflow.

---

## Task Lifecycle

```text id="owrw2q"
Pending → Ready → In Progress → Implemented → Tested
→ Verified → Committed → Completed
```

A task SHALL NOT skip states.

---

## Atomic Commit Policy

One TODO item, one implementation, one documentation update, one test
update, one atomic commit. Never bundle. Never partially implement.

---

## Validation Gates

Every completed task MUST satisfy:

- Ruff passes
- Black passes
- mypy passes
- pytest passes
- Documentation updated
- Public API documented
- Google-style docstrings complete
- Examples added (if public API)
- Benchmark completed (if applicable)
- Atomic commit created with SHA recorded

---

## Active Tasks

### Foundation

- [x] TASK-1.001 Initialize package structure (commit `86dcaed`)
- [x] TASK-1.002 Configure pyproject.toml (commit `86dcaed`)
- [x] TASK-1.003 Configure Ruff (commit `b6b2439`)
- [x] TASK-1.004 Configure Black (commit `ce76d46`)
- [x] TASK-1.005 Configure mypy (commit `1a8052a`)
- [x] TASK-1.006 Configure pytest (commit `2a974ed`)
- [x] TASK-1.007 Configure CI (commits `30d81b1` + `b40875b`,
      reformulated to single workflow `d10e056` family)

### Configuration

- [x] TASK-5.001 AVQConfig dataclass (commit `ab35160`)
- [x] TASK-5.002 Configuration validation (commit `ab35160`)
- [x] TASK-5.003 Configuration serialization (commit `ab35160`)
- [x] TASK-5.004 Configuration file I/O (commit `442d1d0`)
- [x] TASK-5.005 Configuration unit tests (commit `ab35160`)

### Codebook

- [x] TASK-8.001 HierarchicalCodebook (commit `5e4bea4`)
- [x] TASK-8.002 Parent node representation (commit `5e4bea4`)
- [x] TASK-8.003 Child node representation (commit `5e4bea4`)
- [x] TASK-8.004 Parent-child projection (commit `5e4bea4`)
- [x] TASK-8.005 Serialization (commit `5e4bea4`)
- [x] TASK-8.006 EMA updates (commit `5e4bea4`)
- [x] TASK-8.007 Initialization (commit `5e4bea4`)
- [x] TASK-8.008 Unit tests (commit `5e4bea4`)

### Quantizer

- [x] TASK-8.101 VectorQuantizer interface (commit `2073498`)
- [x] TASK-8.102 Parent assignment (commit `2073498`,
      refactor `cd28499`)
- [x] TASK-8.103 Child assignment (commit `2073498`,
      fix `09ec75d`)
- [x] TASK-8.104 Aggregation (commit `2073498`)
- [x] TASK-8.105 Count accumulation (commit `2073498`)
- [x] TASK-8.106 Validation (commit `2073498`)
- [x] TASK-8.107 Unit tests (commit `2073498`,
      fix `03f0c70`, `ec347c1`)

### Adaptive Attention

- [x] TASK-9.001 Parent attention (commit `de51f00`,
      fix `d1a5b03`)
- [x] TASK-9.002 Importance estimation (commit `c203ee2`,
      fix `cd28499`)
- [x] TASK-9.003 Parent selection (commit `c203ee2`,
      refactor `d5cf197`)
- [x] TASK-9.004 Child refinement (commit `e37d70c`)
- [x] TASK-9.005 Correcting attention (commit `de51f00`,
      fix `d1a5b03`, `85d98c9`)
- [x] TASK-9.006 Numerical validation (commit `7d18646`,
      `95a5a38`)
- [x] TASK-9.007 Unit tests (commit `e37d70c`,
      polish `d9fc802`)

### Execution Engine

- [x] TASK-10.001 Execution engine (commit `bce845f`)
- [x] TASK-10.002 Pipeline orchestration (commit `bce845f`,
      wire autocast `85d98c9`)
- [x] TASK-10.003 Backend dispatch (commit `6eec666`)
- [x] TASK-10.004 Output reduction (commit `bce845f`,
      fix `d1a5b03`)
- [x] TASK-10.005 Integration tests (`tests/integration/test_*.py`;
      include vLLM, FlashAttention, xFormers, AVQA end-to-end)
      (commit `e7c818d`)

### Triton Backend

- [~] TASK-11.001 Triton VQ precompute kernel
      (skeleton staged in `bb660fd`; CPU-only stub delegates to
      TorchBackend in `6eec666`/`1fede67`; real GPU kernel blocked
      on OPT-0001 acceptance gate, see `PUBLICATION.md`
      "Outstanding Gaps Before Publication" #1)
- [~] TASK-11.002 Triton online-softmax attention kernel
      (CPU stub in `bb660fd`/`1fede67`; GPU kernel same OPT-0001
      blocker as 11.001)
- [~] TASK-11.003 Triton correcting-attention kernel
      (CPU stub in `bb660fd`/`1fede67`; GPU kernel same OPT-0001
      blocker as 11.001)
- [ ] TASK-11.004 Numerical verification + Triton versus Torch
      benchmarks (BENCHMARKS.md) — blocked on OPT-0001 GPU runner
      (same blocker as 11.001)

### Framework Integrations

> Historical note: the entries below describe integrations that existed in
> pre-alpha development history. They were removed before `v0.1.0` and are
> deferred from the current support contract. The authoritative current-state
> record is `SPEC_COMPLIANCE.md` and `docs/compatibility.md`.

- [x] TASK-12.001 Hugging Face adapter (commit `0e0d397`;
      weight-transfer hardening `155285a`)
- [x] TASK-12.002 vLLM paged-attention adapter (commit `2131523`;
      `attn_metadata` accepted by `AVQvLLMBackend.forward`, routed via
      `PagedKVCache`)
- [x] TASK-12.003 FlashAttention interop (commit `ac6a841`;
      GPU equivalence in `tests/integration/test_attention_interops_gpu.py`)
- [x] TASK-12.004 xFormers interop (commit `ac6a841`;
      GPU equivalence in `tests/integration/test_attention_interops_gpu.py`)
- [x] TASK-12.005 End-to-end integration tests
      (commit `e7c818d`; `tests/integration/test_avqa_end_to_end.py`,
      `tests/integration/test_integrations.py`)

---

## Live next-open task

All core items in the "Active Tasks" section above are now `[x]` except for
`TASK-11.001`–`TASK-11.004` (Triton GPU kernels and benchmarking), which
remain blocked on the CUDA-matrix CI runner. The framework-integration entries
above are historical and deferred from the public-alpha support contract; they
are not shipped capabilities. See
`PUBLICATION.md` "Outstanding Gaps Before Publication" and `OPTIMIZATIONS.md`
"Optimization Backlog" for the cross-referenced gates.

### Public-alpha reliability follow-ups

- [x] TASK-P.001 Execute every checked-in example in CI and compile examples in
      the test suite so public API walkthroughs cannot silently drift (commit
      `9e642e6`).
- [x] TASK-P.002 Keep CI dependencies aligned with the supported alpha surface;
      removed the stale framework-integration install (commit `2abc5c6`).
- [x] TASK-P.003 Reconcile historical integration entries with the current
      public-alpha support contract (commit `1868991`).
- [x] TASK-P.004 Validate paged-cache checkpoint sequence and batch invariants
      during restore (commit `664e53c`).
- [x] TASK-P.005 Reject non-integral paged-cache position metadata instead of
      silently coercing it (commit `365a9f1`).
- [x] TASK-P.006 Make in-memory cache append and restore failure-atomic across
      tensor conversion failures (commit `21ced43`).
- [x] TASK-P.007 Validate integer-only configuration fields before runtime use
      (commit `f7f0be0`).
- [x] TASK-P.008 Validate boolean configuration fields before runtime use
      (commit `f423a96`).
- [x] TASK-P.009 Check repository Markdown links in CI (commit `78ac108`).
- [x] TASK-P.010 Reconcile benchmark protocol and historical experiment
      boundaries with the public-alpha support contract (commit `14253e4`).
- [x] TASK-P.011 Include GitHub contribution templates in the Markdown link
      gate (commit `a6e91d2`).
- [x] TASK-P.012 Align package metadata and release copy with the tested Python
      3.10–3.12 support range (commit `62e0527`).
- [x] TASK-P.013 Make first-time logger configuration thread-safe so concurrent
      application startup cannot install duplicate handlers (commit `ef6b520`).
- [x] TASK-P.014 Make paged KV-cache appends transactional across conversion
      and concatenation failures (commit `e3509d7`).
- [x] TASK-P.015 Add a validated backend registration seam for external
      implementations without hard-coded core imports (commit `b4feca6`).
- [x] TASK-P.016 Normalize nested configuration deserialization failures to
      the public ConfigurationError contract (commit `b5782e2`).
- [x] TASK-P.017 Validate online-softmax tile sizes at the public backend
      boundary (commit `021d0ac`).
- [x] TASK-P.018 Declare pytest-benchmark in the development environment so
      the documented full test target can collect performance tests (commit `7927fe9`).
- [x] TASK-P.019 Mark the performance module as benchmark-only so standard
      correctness tests exclude exploratory timing/quality assertions (commit `7927fe9`).
- [x] TASK-P.020 Execute the documented repository test target in CI on Python
      3.12 to prevent environment and scope drift (commit `b78347d`).
- [x] TASK-P.021 Reconcile README test commands with the benchmark-only
      performance boundary (commit `1dff96a`).
- [x] TASK-P.022 Run the repository test target in tagged-release verification
      to keep release and CI support boundaries aligned (commit `52e76f2`).
- [x] TASK-P.023 Fail closed when distribution smoke tests see ambiguous wheel
      or sdist artifacts (commit `1585fe1`).
- [x] TASK-P.024 Validate public attention masks before pipeline execution,
      including KV-cache-resolved lengths (commit `291aeee`).
- [x] TASK-P.025 Validate user masks before mutating KV caches so rejected
      requests remain failure-atomic (commit `a82a41d`).
- [x] TASK-P.026 Keep KV-cache appends transactional across downstream forward
      failures (commit `9870aba`).
- [x] TASK-P.027 Validate and normalize resolved KV-cache tensors at the
      attention pipeline boundary (commit `38381d7`).
- [x] TASK-P.028 Inject the canonical Python release version into the frontend
      build and remove duplicated public version literals (commit `daac321`).
- [x] TASK-P.029 Preserve multi-batch first append for empty KV-cache sentinels
      (commit `ff9edea`).
- [x] TASK-P.030 Reject non-finite attention inputs before pipeline mutation
      (commit `e757ae3`).
- [x] TASK-P.031 Make codebook checkpoint restores failure-atomic
      (commit `258e26d`).
- [x] TASK-P.032 Reject non-finite codebook checkpoint tensors before restore
      (commit `45c1a6f`).
- [x] TASK-P.033 Make EMA codebook updates finite and failure-atomic
      (commit `5fa24c1`).
- [x] TASK-P.034 Reject non-finite tensors at the KV-cache storage boundary
      (commit `a109e29`).
- [x] TASK-P.035 Make BCAR online adaptation finite and failure-atomic
      (commit `ee0d2c0`).
- [x] TASK-P.036 Reject non-finite inputs in StreamingVQBuffer before mutation
      (commit `a7d3d99`).
- [x] TASK-P.037 Make shared profiler recording and export concurrency-safe
      (commit `7c334bb`).
- [x] TASK-P.038 Apply ExecutionConfig.seed during module initialization without
      mutating caller RNG state (commit `149f6d7`).
- [x] TASK-P.039 Make KV-cache state transitions and snapshots thread-safe
      (commit `a4083cb`).
- [x] TASK-P.040 Restore CI formatter compliance after initialization refactor
      (commit `fb1794b`).
- [x] TASK-P.041 Cancel superseded CI and Pages runs on the main branch
      (commit `071d5c4`).
- [x] TASK-P.042 Return isolated KV-cache lookup snapshots
      (commit `3416919`).
- [x] TASK-P.043 Remove stale current-suite test-count wording from readiness docs
      (commit `cf0ed99`).
- [x] TASK-P.044 Enforce the supported Python range in CI and release verification
      (commit `04ea7eb`).
- [x] TASK-P.045 Apply ExecutionConfig.deterministic during forward safely
      (commit `c641dde`).
- [x] TASK-P.046 Complete frontend navigation landmarks and icon labels
      (commit `f707f78`).
- [x] TASK-P.047 Make mobile navigation dismissible with Escape
      (commit `4b3e551`).
- [x] TASK-P.048 Add a docs navigation route contract gate
      (commit `f992b44`).
- [x] TASK-P.049 Verify deterministic execution wiring through AVQAttention
      (commit `a47078d`).
- [x] TASK-P.050 Execute a public-API forward pass in wheel and sdist smoke envs
      (commit `2bd9b5c`).
- [x] TASK-P.051 Audit locked frontend dependencies in CI
      (commit `d965c9f`).
- [x] TASK-P.052 Document a concrete confidential Code of Conduct reporting path
      (commit `9a0a53d`).
- [x] TASK-P.053 Label historical compliance coverage snapshots explicitly
      (commit `54f99c9`).
- [x] TASK-P.054 Reconcile Unreleased changelog with current readiness work
      (commit `854e12b`).
- [x] TASK-P.055 Gate PyPI publication behind explicit trusted-publishing
      approval (commit `d7df3a4`).
- [x] TASK-P.056 Add a compatibility-report issue template for the support
      matrix (commit `a93865a`).
- [x] TASK-P.057 Reconcile benchmark protocol paths with tracked evidence and
      test required experiment artifacts (commit `57828b4`).
- [x] TASK-P.058 Expand profiler contract coverage for export and failure
      paths (commit `a10a076`).
- [x] TASK-P.059 Verify the bootstrapped environment with an import and public
      API forward smoke test (commit `eff2f52`).

### Cleanup pass (2026-07-21)

The following catch-up items are owned by the sweep (verified by the
then-current test suite + ruff/mypy clean + 90.5% coverage). The suite
has since grown; run `make test` for current evidence. Each corresponds
to the deep-review findings reported in CHANGELOG "Unreleased Fixed".

- [x] TASK-A.001 Visualizer public-API restore
- [x] TASK-A.002 integration test import fix
- [x] TASK-B.001 WeightedMerge double-count
- [x] TASK-B.002 ProfilerReport.to_dict total_flops
- [x] TASK-B.003 NaN-safe masked-tile exp (backend + pipeline)
- [x] TASK-B.004 codeword-level mask plumbing
- [x] TASK-B.005 ThresholdRouter over-budget raise
- [x] TASK-B.006 OnlineSoftmaxState.replace m_anchor
- [x] TASK-B.007 Delete dead code (correct_parent_contribution, compute_routing)
- [x] TASK-B.008 HVAQ-ENT top-P entropy per SPEC §16.2
- [x] TASK-D.002 direct OnlineSoftmaxState.replace tests
- [x] TASK-C.001 exception-hierarchy sweep (~30 sites)
- [x] TASK-C.002 NotInitializedError raiseable
- [x] TASK-D.001 weak-test tightening
- [x] TASK-D.003 BCAR tolerance + e2e
- [x] TASK-D.004 disjoint-set assertion
- [x] TASK-F.001 BudgetRouter honest impl
- [x] TASK-F.002 Profiler memory + selected_indices
- [x] TASK-F.003 InMemoryKVCache eviction contiguous
- [x] TASK-F.006 BCAR drop dead per-batch EMA
- [x] TASK-F.007 BCAR docstring v_j -> k_j
- [x] TASK-F.008 functional.attention kv_cache
- [x] TASK-F.009 from_dict run __post_init__
- [x] TASK-F.010 save_json catch TypeError
- [x] TASK-E.001 docs coverage gate -> 90%
- [x] TASK-E.002 docs test-count reconciliation
- [x] TASK-E.003 SPEC_COMPLIANCE remove deleted files

---

## Recent Work (2026-07-21)

- [x] Forward refactor: broke `forward_impl` into 10+ pipeline stage helpers
- [x] Learnable HVAQ parameters: `_parent_beta` and `_alpha` nn.Parameters
- [x] Code quality: dead code removal, renamed public classes, specific exceptions, docstrings
- [x] CI/CD: build job, pip-audit, release workflow, pre-commit updates
- [x] Dead test removal: deleted `@parametrize([])` test
- [x] Multi-pass refinement: disjoint-set re-routing with converging residual norms
- [x] torch.compile numerical equivalence test on CPU
- [x] HF adapter: head_mask/past_key_value debug logging when dropped
- [x] Docs: CHANGELOG, README (project structure, features, test count), math.md (HVAQ, BCAR, multi-pass)

---

## Historical Completeness Ledger (append-only)

| SPEC Chapter | First SHA      | Last SHA       | Notes                                          |
|-------------:|---------------:|---------------:|------------------------------------------------|
| 1, 5.3, 5.5  | `86dcaed`      | `9f9afff`      | Package skeleton + public API exports          |
| 2, 7.20      | `d10e056`      | `d10e056`      | Spec / docs bootstrap                          |
| 3.6, 5.8     | `ab35160`      | `85d98c9`      | AVQConfig + sub-configs + serialization       |
| 3.7, 8.3     | `5e4bea4`      | `cd28499`      | Hierarchical codebook + mean constraint + EMA  |
| 3.7, 8.5-8.7 | `2073498`      | `cd28499`      | Vector quantization engine + aggregation       |
| 3.10, 9.5-9.6| `c203ee2`      | `d5cf197`      | Routing subsystem (TopP/Threshold/Budget)      |
| 3.11, 7.13   | `de51f00`      | `85d98c9`      | Merge strategies + online softmax + correction |
| 3.9, 9.3-9.8 | `e37d70c`      | `d5cf197`      | Adaptive refinement orchestrator               |
| 3.12, 4.10   | `6eec666`      | `6eec666`      | Backend abstraction (Torch + Triton stub)      |
| 3.13, 5.5    | `3047d60`      | `3047d60`      | KV cache (InMemory + Paged)                    |
| 4.7, 2.8     | `e8b0112`      | `e8b0112`      | Scheduler (Default + Adaptive)                 |
| 3.4, 5.6     | `bce845f`      | `d9fc802`      | AVQAttention nn.Module + pipeline              |
| 3.5, 5.7     | `9cb1f53`      | `9cb1f53`      | Functional API                                 |
| 3.17-3.18    | `6a5b9d3`      | `6a5b9d3`      | Profiling + Visualization                      |
| 3.14-3.16    | `0e0d397`      | `0e0d397`      | HF/vLLM/FA/xFormers adapters                   |
| 3.19         | `a297bd1`      | `a297bd1`      | Benchmark suite + examples                     |
| 3.3, 5.19    | `3d61b8b`      | `97608f7`      | Release + compliance + CHANGELOG               |

---

## Definition of Done

A task is considered complete only when:

- It satisfies every requirement in `SPEC.md`.
- All tests pass.
- Documentation is complete.
- Static analysis passes.
- Formatting passes.
- Integration verification passes.
- Benchmark verification passes (where applicable).
- The implementation has been committed in a single atomic commit.
- The commit SHA has been recorded in this document.

Any task failing one or more of these conditions SHALL remain open.
