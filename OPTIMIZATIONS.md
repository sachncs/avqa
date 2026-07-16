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
