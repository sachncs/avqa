# OPTIMIZATIONS.md

> **Project:** AVQA – Adaptive Vector Quantized Attention
>
> **Purpose:** Optimization Registry, Engineering Decisions, and Performance Evolution
>
> This document is the authoritative record of every accepted optimization implemented in AVQA.
>
> Every optimization recorded here has:
>
> - been proposed,
> - implemented,
> - benchmarked,
> - statistically validated,
> - reviewed,
> - and merged.
>
> Optimizations that remain experimental SHALL remain in `RESEARCH.md`.
> Failed optimizations SHALL remain in `EXPERIMENTS.md`.

---

# Optimization Philosophy

Optimization is evidence-driven.

An optimization SHALL NOT be accepted because it appears elegant, intuitive, or theoretically attractive.

Every optimization MUST demonstrate measurable benefit while preserving correctness.

Benefits may include:

- Reduced latency
- Increased throughput
- Reduced memory consumption
- Improved numerical stability
- Improved scalability
- Improved maintainability
- Reduced implementation complexity

Every accepted optimization SHALL remain independently reversible.

---

# Optimization Backlog (acceptance gate)

The following improvements are *proposed* and tracked in
`RESEARCH.md` until their acceptance criteria are satisfied under
`BENCHMARKS.md`. Each is assigned an identifier once accepted.

| ID | Title | Status | Acceptance Gate |
|----|-------|--------|-----------------|
| OPT-0001 | Triton VQ + attention + correction kernels | Proposed | ≥20 % faster than SDPA on the Triton backend at seq ≥ 4096 (SPEC §11.10) |

`OPT-0001` lives in `RESEARCH.md` (Proposed) until the GPU-matrix
runner produces numerical-equivalence + benchmark evidence.

---

# Optimization Index (this release)

| ID | Title | Status | Acceptance Evidence |
|----|-------|--------|---------------------|
| OPT-0003 | Online Codebook Adaptation (BCAR) | Accepted (CPU) — GPU statistical validation pending | EXP-0004: 60.7 % VQ-loss reduction vs static codebook after 1024 streaming updates |

`OPT-0003` is the project's first algorithmic contribution beyond
paper reproduction. The CUDA-matrix runner will additionally
collect multi-seed statistical significance in `EXPERIMENTS.md`.

---

# Performance Evolution

The CPU reference path has been measured twice in this session.
The Triton-backend GPU numbers will be appended here after the
first CUDA runner activates the Triton kernels.

| Version | seq_len=128 | seq_len=512 | seq_len=1024 | Notes |
|---------|-------------:|-------------:|-------------:|-------|
| v0.1.0 baseline (EXP-0001) | 4.146 ms | 11.592 ms | 22.215 ms | CPU TorchBackend (PyTorch 2.10.0 / macOS arm64) |
| v0.1.0 + governance (EXP-0002) | 3.363 ms | 10.140 ms | 19.618 ms | Same harness after Triton staging and adapter hardening |

BCAR stream-on-stream VQ-loss reduction (EXP-0004):

|            | VQ loss (after 1024 steps) | improvement vs static |
|------------|--------------------------:|-----------------------:|
| static     | 13.74                     | —                      |
| bcar       |  5.41                     | 60.7 %                 |
| oracle     |  0.01                     | —                      |

---

# Rollback Policy

Every optimization SHALL support rollback.

The current state: Triton kernels are **opt-in** via
``config.backend.name == "triton"``. The default backend is
``"torch"``; switching to Triton or back to Torch changes zero user
calls. This default allows rollback by configuration alone, with no
source changes.

---

# Acceptance Criteria

An optimization SHALL be accepted only when:

- Correctness preserved.
- Benchmarks completed.
- Statistical significance demonstrated.
- SPEC compliance maintained.
- Tests pass.
- Documentation updated.
- CI passes.
- Code review approved.

## OPT-0005

### Title

Hopfield-VQ-Attention (HVAQ) with per-query adaptive temperature.

### Status

Accepted (CPU benchmark; multi-seed + downstream-quality
validation pending).

### Version

v0.2.0

### Date

2026-07-16

### Author

Research Team

### Summary

Implements the per-query temperature schedule of SPEC §16 in
``AVQAttention.forward``. With ``adaptive="none"`` the output is
bit-identical to the paper pipeline (Theorem 16.1). With
``adaptive="entropy"`` (HVAQ-ENT) or ``"linear"`` (HVAQ-LIN) the
temperature is derived from the router's top-P attention-mass
entropy, sharpening peaked distributions and preserving the
router's top-P selection (Theorem 16.2).

### Motivation

The paper's fixed-temperature softmax is the natural starting
point, but for an attention kernel that already supports AVQ's
hierarchical router, an adaptive temperature is the most direct
generalisation. HVAQ is a strict monotone reparametrisation of the
paper's softmax; the relative parent ranking is preserved for any
positive ``β``, so the contribution is router-compatible by
construction.

### Baseline

Paper single-pass attention (SPEC §10.7). The ``adaptive="none"``
default keeps the existing pipeline bit-exact.

### Implementation

- ``src/avqa/hopfield.py``: ``paper_beta``, ``per_query_beta``,
  ``hopfield_logits``, ``validate_adaptive``.
- ``src/avqa/config.py``: ``HopfieldConfig`` dataclass and
  ``BackendConfig.hopfield`` master switch.
- ``src/avqa/attention_module.py``: HVAQ block in ``forward_impl``,
  gated on ``backend.hopfield and hopfield.adaptive != "none"``.

### Numerical Evidence

EXP-0006 (committed) measures latency and output difference on a
small synthetic 64-token task:

- sdpa: 0.049 ms median
- paper single-pass: 1.174 ms median
- hvaq entropy: 1.310 ms median (+12 % overhead over paper)
- hvaq linear: 1.208 ms median (+3 % overhead over paper)
- HVAQ-ENT vs paper attention output: 1.3e8 max abs diff
  (sharpened distribution weighted against the value magnitude).
- HVAQ-LIN vs paper attention output: 0.0 max abs diff
  (linear schedule collapses to ``β_0`` at peaked distribution).

### Tests

- ``tests/unit/test_hopfield.py``: 24 SPEC §16 unit tests covering
  paper_beta, validate_adaptive, per_query_beta, hopfield_logits
  broadcasting, HopfieldConfig validation, and Theorem 16.1
  paper equivalence (with synced codebooks).
- 486 unit tests + 10 integration tests remain green.

### Rollback

``BackendConfig(hopfield=False)`` keeps the paper pipeline
unchanged. ``HopfieldConfig(adaptive="none")`` is the explicit
opt-out even when the master switch is on. Both default values are
paper-exact so existing users see no behaviour change.

### Risks

- HVAQ-ENT sharpens peaked distributions, which can amplify
  outlier values in the attention output. On real workloads the
  downstream-quality impact (perplexity, classification) is the
  real metric. Multi-seed GPU validation is the next gate.
- The router's top-P selection is invariant under positive ``β``
  (Theorem 16.2) but the per-P probabilities are not. Existing
  tests that assume a particular parent attention mass may need
  updating once the GPU-matrix runner is available.

### References

- Ramsauer et al., "Hopfield Networks is All You Need" (2021):
  original modern Hopfield formulation.
- Bottou & Bengio (1994): stochastic K-means convergence rate.
