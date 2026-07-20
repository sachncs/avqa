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

All items in the "Active Tasks" section above are now `[x]` except for
`TASK-11.001`–`TASK-11.004` (Triton GPU kernels and benchmarking), which
remain blocked on the CUDA-matrix CI runner. See
`PUBLICATION.md` "Outstanding Gaps Before Publication" and `OPTIMIZATIONS.md`
"Optimization Backlog" for the cross-referenced gates. The HVAQ
downstream-consumer invariant (per-parent probability) test is the next
incremental addition; tracked under HVAQ Risks in `PUBLICATION.md` L106.

---

## Recent Work (2026-07-17)

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
