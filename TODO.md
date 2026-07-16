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

- [ ] TASK-1.001 Initialize package structure (historical `86dcaed`)
- [ ] TASK-1.002 Configure pyproject.toml (historical `86dcaed`)
- [ ] TASK-1.003 Configure Ruff (historical `b6b2439`)
- [x] TASK-1.004 Configure Black (commit `ce76d46`)
- [ ] TASK-1.005 Configure mypy (historical `1a8052a`)
- [ ] TASK-1.006 Configure pytest (historical `2a974ed`)
- [ ] TASK-1.007 Configure CI (historical `30d81b1` + `b40875b`,
      reformulated to single workflow `d10e056` family)

### Configuration

- [ ] TASK-5.001 AVQConfig dataclass (historical `ab35160`)
- [ ] TASK-5.002 Configuration validation (historical `ab35160`)
- [ ] TASK-5.003 Configuration serialization (historical `ab35160`)
- [ ] TASK-5.004 Configuration file I/O
- [ ] TASK-5.005 Configuration unit tests (historical `ab35160`)

### Codebook

- [ ] TASK-8.001 HierarchicalCodebook (historical `5e4bea4`)
- [ ] TASK-8.002 Parent node representation (historical `5e4bea4`)
- [ ] TASK-8.003 Child node representation (historical `5e4bea4`)
- [ ] TASK-8.004 Parent-child projection (historical `5e4bea4`)
- [ ] TASK-8.005 Serialization (historical `5e4bea4`)
- [ ] TASK-8.006 EMA updates (historical `5e4bea4`)
- [ ] TASK-8.007 Initialization (historical `5e4bea4`)
- [ ] TASK-8.008 Unit tests (historical `5e4bea4`)

### Quantizer

- [ ] TASK-8.101 VectorQuantizer interface (historical `2073498`)
- [ ] TASK-8.102 Parent assignment (historical `2073498`,
      refactor `cd28499`)
- [ ] TASK-8.103 Child assignment (historical `2073498`,
      fix `09ec75d`)
- [ ] TASK-8.104 Aggregation (historical `2073498`)
- [ ] TASK-8.105 Count accumulation (historical `2073498`)
- [ ] TASK-8.106 Validation (historical `2073498`)
- [ ] TASK-8.107 Unit tests (historical `2073498`,
      fix `03f0c70`, `ec347c1`)

### Adaptive Attention

- [ ] TASK-9.001 Parent attention (historical `de51f00`,
      fix `d1a5b03`)
- [ ] TASK-9.002 Importance estimation (historical `c203ee2`,
      fix `cd28499`)
- [ ] TASK-9.003 Parent selection (historical `c203ee2`,
      refactor `d5cf197`)
- [ ] TASK-9.004 Child refinement (historical `e37d70c`)
- [ ] TASK-9.005 Correcting attention (historical `de51f00`,
      fix `d1a5b03`, `85d98c9`)
- [ ] TASK-9.006 Numerical validation (historical `7d18646`,
      `95a5a38`)
- [ ] TASK-9.007 Unit tests (historical `e37d70c`,
      polish `d9fc802`)

### Execution Engine

- [ ] TASK-10.001 Execution engine (historical `bce845f`)
- [ ] TASK-10.002 Pipeline orchestration (historical `bce845f`,
      wire autocast `85d98c9`)
- [ ] TASK-10.003 Backend dispatch (historical `6eec666`)
- [ ] TASK-10.004 Output reduction (historical `bce845f`,
      fix `d1a5b03`)
- [ ] TASK-10.005 Integration tests (next-open after framework
      adapters)

### Triton Backend

- [ ] TASK-11.001 Triton VQ precompute kernel
      (currently `src/avqa/backend.py:220` delegates to TorchBackend;
      deferred in `6eec666`)
- [ ] TASK-11.002 Triton online-softmax attention kernel
      (deferred in `6eec666`)
- [ ] TASK-11.003 Triton correcting-attention kernel
      (deferred in `6eec666`)
- [ ] TASK-11.004 Numerical verification + Triton versus Torch
      benchmarks (BENCHMARKS.md)

### Framework Integrations

- [ ] TASK-12.001 Hugging Face adapter
      (historical `0e0d397`, hardening required for weight transfer)
- [ ] TASK-12.002 vLLM paged-attention adapter
      (historical `0e0d397`, current ignores `kv_cache` /
      `attn_metadata` per `src/avqa/integrations.py:415`)
- [ ] TASK-12.003 FlashAttention interop
      (historical `0e0d397`, equivalence test required)
- [ ] TASK-12.004 xFormers interop
      (historical `0e0d397`, equivalence test required)
- [ ] TASK-12.005 End-to-end integration tests
      (historical `0e0d397`, to be hardened)

---

## Live next-open task

`TASK-5.004 Configuration file I/O` is the next item in numerical
order that is genuinely unimplemented: the current `AVQConfig`
exposes `to_dict`/`from_dict` in `src/avqa/config.py:329` but no
JSON-path helper. Per SPEC §3.20/§5.12, configurations SHALL support
serialization, and the file-path read/write helper closes the gap.

Subsequent live items after `TASK-5.004` are `TASK-1.001`–`TASK-1.003`
/ `1.005`–`1.007` / `TASK-5.001`–`TASK-5.003` / `5.005` / `8.*` /
`8.101`-`8.107` / `9.*` / `10.001`-`10.005` (most are already shipped
historically — see ledger below).

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
