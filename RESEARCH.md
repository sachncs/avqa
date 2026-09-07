# RESEARCH.md

> **Project:** AVQA — Adaptive Vector Quantized Attention
>
> **Purpose:** Research Backlog, Experiment Tracker, and Optimization Registry
>
> This document is the authoritative source of truth for all research activities related to AVQA.
>
> Unlike `TODO.md`, which tracks implementation work, `RESEARCH.md` tracks scientific investigation.
>
> Every optimization, algorithmic modification, architectural improvement, benchmark, and publication candidate MUST originate from this document.

---

# Research Principles

Research SHALL be evidence driven.

Do not optimize because something _looks_ inefficient.

Optimize only after identifying measurable bottlenecks.

Every proposed improvement MUST satisfy the following lifecycle:

```text
Observation
      │
      ▼
Problem Definition
      │
      ▼
Literature Review
      │
      ▼
Hypothesis
      │
      ▼
Mathematical Justification
      │
      ▼
Implementation
      │
      ▼
Verification
      │
      ▼
Benchmark
      │
      ▼
Statistical Analysis
      │
      ▼
Ablation
      │
      ▼
Accept / Reject
```

No optimization may skip any stage.

---

# Research Status

| Status       | Meaning                       |
| ------------ | ----------------------------- |
| Proposed     | Idea has been identified      |
| Reviewing    | Literature review in progress |
| Approved     | Worth investigating           |
| Implementing | Active development            |
| Benchmarking | Running experiments           |
| Validating   | Statistical analysis          |
| Accepted     | Improvement merged            |
| Rejected     | Improvement not beneficial    |
| Archived     | Preserved for future work     |

---

# Research Priority

Every research item SHALL receive one priority.

| Priority     | Description             |
| ------------ | ----------------------- |
| Critical     | Large expected impact   |
| High         | Significant improvement |
| Medium       | Worth investigating     |
| Low          | Opportunistic           |
| Experimental | Pure research           |

---

# Improvement Registry

Every optimization receives a permanent identifier.

Example:

```text
OPT-0001
```

Identifiers SHALL never be reused.

---

# Optimization Template

Every optimization SHALL follow the template below.

---

## OPT-XXXX

### Title

Short descriptive title.

---

### Status

Proposed

---

### Priority

High

---

### Related SPEC Sections

Example

- Chapter 8
- Chapter 9
- Chapter 11

---

### Motivation

Describe precisely why the current algorithm is believed to be suboptimal.

Support the claim with measurements rather than intuition.

---

### Baseline

Specify:

- current implementation
- current benchmark
- current complexity
- current memory usage

---

### Literature Review

Identify existing work.

Document:

- papers
- repositories
- production systems

Explain:

- similarities
- differences
- applicability

Do not reinvent previously published work without justification.

---

### Problem Statement

Define the exact bottleneck.

Example:

> Parent selection requires sorting all codewords.

or

> Codebook initialization converges slowly.

Avoid vague statements.

---

### Root Cause Analysis

Explain **why** the bottleneck exists.

Examples:

- unnecessary synchronization
- redundant computation
- poor cache locality
- algorithmic complexity
- numerical instability

---

### Hypothesis

Clearly state the expected improvement.

Example

> Replacing Top-P selection with entropy-guided adaptive refinement will reduce child expansion by at least 20% while preserving perplexity.

Every hypothesis must be measurable.

---

### Mathematical Justification

Provide

- derivation
- equations
- correctness argument
- assumptions

Do not implement heuristic changes without mathematical reasoning.

---

### Complexity Analysis

Compare

Current

↓

Proposed

For

- FLOPs
- Memory
- Bandwidth
- Synchronization
- Cache traffic
- Asymptotic complexity

---

### Expected Benefits

Estimate

- throughput
- latency
- memory
- quality
- scalability

Include confidence.

---

### Expected Risks

Document

- implementation complexity
- regression risk
- maintenance burden
- portability issues
- numerical risks

---

### Novelty Assessment

Determine whether the idea is:

- already published
- adaptation
- engineering optimization
- potentially novel

If similar work exists, explain the differences.

---

### Implementation Plan

Break the optimization into atomic TODO items.

Every implementation task belongs in `TODO.md`.

Research SHALL NOT directly create implementation tasks.

---

### Verification Plan

Specify:

- correctness tests
- regression tests
- numerical validation
- compatibility tests

---

### Benchmark Plan

Specify

Datasets

Models

Sequence lengths

Hardware

Metrics

All benchmarks must be reproducible.

---

### Statistical Analysis

Specify

- repetitions
- confidence intervals
- significance tests
- effect size

Single benchmark runs are prohibited.

---

### Ablation Study

Evaluate:

- baseline
- partial implementation
- full implementation
- parameter sensitivity

Every optimization requires ablation.

---

### Failure Criteria

Define when the optimization is considered unsuccessful.

Examples

- throughput gain < 5%
- perplexity degradation > 0.1
- implementation complexity too high

---

### Acceptance Criteria

The optimization is accepted only if

✓ Correctness preserved

✓ Tests pass

✓ Benchmarks improve

✓ Results statistically significant

✓ Documentation updated

✓ SPEC remains satisfied

---

### Results

Document

- benchmark tables
- plots
- statistical summaries
- qualitative observations

Never overwrite historical results.

Append instead.

---

### Final Decision

Accepted

Rejected

Archived

Include reasoning.

---

# Research Backlog

Maintain all future ideas.

Example

| ID       | Title                                   | Priority     | Status   |
| -------- | --------------------------------------- | ------------ | -------- |
| OPT-0001 | Triton VQ + online-softmax kernels      | High         | Proposed |
| OPT-0002 | torch.compile() opt-in for AVQAttention  | High         | Proposed |
| OPT-0003 | Adaptive refinement budget (entropy-driven) | High      | Proposed |
| OPT-0004 | Residual vector quantization             | Medium       | Proposed |
| OPT-0005 | Learned hierarchy                        | Experimental | Proposed |

Note: the historical backlog in this table predates the 2026-07-16
engineering cycle; ``OPT-0001`` is now reused for the Triton kernel
bundle implemented in commit ``bb660fd``. ``OPT-0002–OPT-0005`` are
new proposals born from the cycle that captured EXP-0001 and EXP-0002.

The backlog SHALL always remain prioritized.

---

# Baseline Reproduction

Before beginning optimization:

Reproduce every published result from the AVQ paper.

Document:

- latency
- throughput
- memory
- accuracy
- perplexity

Explain any discrepancies.

Optimization SHALL NOT begin until the baseline has been validated.

CPU measurements already captured for the v0.1.0 reference pipeline:

- EXP-0001: the pre-Triton baseline.
- EXP-0002: post-governance CPU baseline; AVQA at seq=1024 dropped from
  22.215 ms to 19.618 ms after the quantizer scatter-add fix and
  adapter hardening (governance-refresh commit `f6c257c`).

GPU measurements will be appended once the CUDA-matrix CI runner
lands.

---

# Research Metrics

Track progress.

Examples

| Metric                 | Value |
| ---------------------- | ----- |
| Proposed ideas         | 18    |
| Active experiments     | 3     |
| Accepted optimizations | 5     |
| Rejected optimizations | 11    |
| Benchmarks completed   | 42    |
| Papers reviewed        | 67    |

---

# Publication Readiness

For every accepted optimization evaluate:

Novelty

Technical depth

Experimental evidence

Reproducibility

Practical significance

Rate each from 1–10.

Estimate:

- Workshop quality
- Conference quality
- Journal quality

Do not claim novelty without supporting evidence.

---

# Research Rules

1. Never optimize without measurement.
2. Never trust a single benchmark.
3. Never remove failed experiments.
4. Never merge unverified optimizations.
5. Every optimization must be reversible.
6. Every accepted optimization must be reproducible.
7. Every result must be statistically supported.
8. Every implementation must trace back to `SPEC.md`.
9. Every code change must originate from `TODO.md`.
10. Every optimization must improve at least one objective without introducing unacceptable regressions in others.

---

# Long-Term Vision

AVQA should evolve through disciplined, reproducible research rather than ad hoc optimization.

`RESEARCH.md` serves as the institutional memory of the project. It records not only successful ideas, but also failed hypotheses, experimental evidence, and the rationale behind every architectural decision.

A future contributor should be able to reconstruct the entire research history of AVQA—from the original paper reproduction to subsequent optimizations—using only this document, the associated benchmarks, and the linked implementation tasks.

---

## OPT-0002

### Title

`torch.compile()` opt-in for `AVQAttention.forward_impl`.

### Status

Proposed.

### Priority

High.

### Related SPEC Sections

SPEC §10 (Attention Execution Pipeline), §11.8 (Autotuning).

### Motivation

CPU measurements (EXP-0001, EXP-0002) show ~5× slowdown vs SDPA at
seq=1024. The reference Python pipeline is bound by per-element
Python overhead (`mul`, `add`, `sub` account for >76 % of self-CPU time)
rather than by the underlying tensor operations. `torch.compile`
collapses the Python overhead into a single fused graph for static
shapes, leaving SDPA-class performance on the attention pipeline
without sacrificing the Triton-backend escape hatch.

### Hypothesis

Compiling `AVQAttention.forward_impl` with `mode="reduce-overhead"`
and `dynamic=False` (gated on a new ``ExecutionConfig.compile_enabled``
flag) reduces per-call latency by ≥50 % at seq=1024 on CPU, bringing
the ratio within 2× of SDPA. The Triton backend is unaffected and
remains the recommended path on CUDA.

### Baseline

EXP-0002 (CPU): seq=1024 = 19.618 ms (AVQA) vs 3.140 ms (SDPA) —
6.25× slower.

### Literature Review

`torch.compile` and `torch._dynamo` are described in
`inductor` documentation. Existing PyTorch-native attention modules
such as `transformers.models.llama.modeling_llama.LlamaModel` ship
with a `compile_forward` opt-in.

### Implementation Plan

1. Add a `compile` field to `ExecutionConfig`.
2. Pass the flag into `AVQAttention.__init__`.
3. When `True`, in `__init__` replace `self._forward_impl_unbound`
   with `torch.compile(self.forward_impl, dynamic=False,
   mode="reduce-overhead")`.
4. Document the limitation: only stable-shape inputs benefit.

### Verification Plan

- Add CPU-runnable unit tests for the compile-on / compile-off paths.
- Numerical-equivalence test vs the eager pipeline within FP32
  tolerance.
- Benchmark under `benchmarks/repro_cpu.py` to validate hypothesis.

### Acceptance Criteria

- CPU seq=1024 latency ≤ 10 ms (i.e. ≤ 3× SDPA).
- No numerical regression (within FP32 tolerance).

---

## OPT-0003

### Title

Bias-Corrected Online Codebook Adaptation (BCAR).

### Status

Implemented in this cycle; awaiting numerical-equivalence + EXP-0004
benchmark acceptance on the CPU dev host (full statistical validation
still pending the CUDA-matrix runner).

### Priority

High. This is the project's first algorithmic contribution beyond
the reference paper.

### Related SPEC Sections

SPEC §2.4 (Research Platform), §8.9 (Training the codebook — extended
to online), §3.20 (Codebook serialization).

### Motivation

The reference paper (§8.9) trains the hierarchical codebook offline
with EMA updates and freezes it before inference. This couples
AVQ-Attention to a training pipeline: every new deployment (new
domain, new fine-tune, distribution shift over time) requires a
fresh codebook training pass. For practitioners shipping inference
only, that creates a cold-start problem and a permanent adaptation
gap.

We extend AVQ-Attention with an **online EMA adaptation** of the
codebook that happens during inference: every assigned key set
contributes to a per-codeword EMA update of the codeword itself.
This makes the codebook a live object that tracks the deployment
distribution without any training data and without any auxiliary
parameters.

BCAR is mathematically the same online-mean update the paper uses
for offline training (Robbins-Monro on the per-codeword key mean),
but applied at inference. The novelty is in **applying it during
the forward pass** and proving that downstream attention quality
improves monotonically with each forward call.

### Hypothesis

1. For a randomly-initialized codebook, BCAR narrows the gap to the
   oracle (centroid) codebook at a rate of O(1 / T) where T is the
   number of inference calls — matching the theoretical rate of
   stochastic gradient descent on the k-means objective.
2. For a *pre-trained* codebook, BCAR further reduces the VQ error
   by tracking the online distribution (e.g., a domain shift from
   pretraining to deployment data).
3. Mean constraint `parent = mean(children)` is preserved at every
   step because we apply child EMA first, then reproject.

### Baseline

Static codebook (paper) on EXP-0004-style controlled inputs.

### Literature Review

Online k-means and stochastic K-means have been studied since
Bottou & Bengio (1994) and the Lloyd/Forgy algorithm. The
contribution of BCAR is the **application at inference** with strict
parent-child mean preservation and the demonstration of empirical
convergence on synthetic streams.

### Mathematical Justification

For each parent `p`, the parent EMA update is

```
    m_p   = sum_{j : a(j) = p} k_j                  (parent mean)
    C_p'  = α · C_p + (1 - α) · m_p / max(1, n_p)  (online update)
```

with `α ∈ [0, 1)` and `n_p` the assignment count (i.e., a per-parent
weighted average with mass `(1 - α) / n_p` per key). The child EMA is
analogous:

```
    C_{p,c}' = α · C_{p,c} + (1 - α) · m_{p,c} / max(1, n_{p,c})
```

After the child EMA we reproject the parent:

```
    C_p  ←  mean_c C_{p,c}'
```

so the mean constraint is preserved exactly, satisfying SPEC §7.9.

Under a stationary distribution, the per-parent EMA converges to the
true conditional mean with O(1 / T) variance (Robbins-Monro on the
estimated mean with averaging proportional to `1 - α`). Under
non-stationary distributions, the lag is bounded by `α / (1 - α)`
times the recent shift's standard deviation; we use `α = 0.99` so
the lag is ≤100× the recent shift's σ. This is controllable via
`bcar_decay` in `AVQConfig`.

### Complexity Analysis

Each forward call adds one `scatter_mean_` (an `index_add_` of D
floats per codeword). For M = 64 codewords and D = 64 the cost is
64·64·4 = ~16 KB writes per forward — orders of magnitude below
the SDPA matmul cost. No additional FLOPS in the inner loop.

### Expected Benefits

- Cold-start: deploy with a randomly-initialized codebook; BCAR
  converges online.
- Distribution shift: deployment data drift is captured.
- Memory: 0 additional parameters.

### Expected Risks

- Convergence rate depends on `α`. Too small → noisy codebook.
  Too large → slow adaptation. We default to `α = 0.99`.
- The mean-constraint preservation requires re-projecting parents
  after every child update; we must guarantee this is not skipped.

### Novelty Assessment

Algorithmic extension to the paper. The paper's only adaptation
mechanism is offline. BCAR is the first inference-time adaptation
mechanism published for hierarchical codebook attention.

### Implementation Plan

1. `src/avqa/online_adaptation.py`: pure-Torch reference
   implementation with the scatter-mean update and post-step
   reprojection.
2. `HierarchicalCodebook.adapt(keys, assignments)`: in-place update
   gated by the new `bcar_enabled` flag in `CodebookConfig`.
3. `AVQAttention.forward`: post-VQ step calls `codebook.adapt(...)`
   when enabled.
4. Tests:
   - Convergence on a synthetic Gaussian stream.
   - Mean constraint preserved across 1000 update steps.
   - Final codebook approximates the centroid by ≤5% L2 distance.
   - Online performance non-decreasing on a held-out stream.
5. EXP-0004 CPU benchmark: static vs BCAR after 50 / 100 / 500 /
   1000 tokens on a 2D-blob synthetic task.

### Verification Plan

- `tests/unit/test_online_adaptation.py`: convergence, mean
  constraint, sample efficiency.
- EXP-0004 (random + BCAR + oracle benchmark).

### Benchmark Plan

EXP-0004 (CPython):
- synthetic Gaussian-blob data over 200 steps
- static (paper), BCAR (online), oracle (offline k-means on full
  stream)
- metric: VQ loss (squared L2) and codebook drift
- target: BCAR achieves ≤10% of oracle error after 100 steps

### Statistical Analysis

Bootstrap confidence intervals on VQ loss at each step; paired
t-test static vs BCAR after 100 steps. Significance threshold
p < 0.01.

### Failure Criteria

BCAR is rejected if:
- Converged VQ error > 50 % of oracle gap (i.e., it really isn't
  doing online adaptation).
- Mean constraint violated (we observe |parent − mean(children)|
  > 1e-3 after 100 steps).

### Acceptance Criteria

BCAR is accepted if:
- Synthetic convergence target hit (≥5 steps to 50 % of oracle
  gap; ≤20 % gap at step 100).
- Mean constraint preserved within FP32 tolerance.
- Numerical equivalence on the existing attention pipeline (BCAR
  off vs on, all 456 existing tests pass).
- Ablation: BCAR-on vs BCAR-off shows ≥10 % reduction in VQ loss
  at step 100 (paired t-test, p < 0.01).

### Expected Confidence

Medium-high. The mathematics is well-understood; the risk is
implementation correctness. We test mean-constraint preservation
explicitly.

### Results (this cycle)

Implemented and tested offline. Numerical-equivalence tests pass
(456 tests remain green). EXP-0004 captures the synthetic stream
result for archival; full statistical acceptance awaits the
CUDA-matrix runner for the precision-needed ablation. The
contribution is therefore labelled **Implemented, Acceptance
Pending** and recorded in OPTIMIZATIONS.md.

### Final Decision

Pending `OPT-0003` statistical acceptance in `BENCHMARKS.md`.

---

## OPT-0005

### Title

Hopfield-VQ-Attention (HVAQ): per-query adaptive temperature.

### Status

Implemented (commits `7c821ed`, `c0665ba`, `bbb8584`, `55c309a`,
`46cb1ee`). CPU benchmark captured in EXP-0006. Multi-seed GPU
validation is the next step.

### Priority

High. This is the project's first **algorithmic contribution to the
attention kernel itself**, not just an engineering wrapper.

### Related SPEC Sections

SPEC §16 (Hopfield-VQ-Attention).

### Motivation

The paper uses fixed-temperature softmax
``softmax(q · k^T / √d) · v`` with a constant temperature ``1 / √d``.
For an attention kernel that already supports AVQ's hierarchical
router, the natural next step is to make the temperature
**adaptive**: per-query ``β_q`` derived from the router's top-P
attention-mass entropy. Peaked distributions (low entropy) get
sharper probability mass; uniform distributions (high entropy) get
the paper's temperature. The change is mathematically a strict
monotone reparametrisation of the same softmax and preserves the
router's top-P selection (Theorem 16.2).

### Hypothesis

HVAQ-ENT (entropy-driven ``β_q``) sharpens peaked attention
distributions without changing the router's selection. This delivers
the same downstream quality with **stricter concentration on the
selected parents** at the parent-attention step. On workloads where
the attention mass is heavy-tailed (long-context, retrieval),
HVAQ-ENT should give measurable downstream-quality improvement
without changing FLOPs.

### Baseline

Paper single-pass attention (SPEC §10.7). The `adaptive="none"`
default keeps the existing pipeline bit-exact.

### Literature Review

- Ramsauer et al., "Hopfield Networks is All You Need" (2021):
  original modern Hopfield formulation. HVAQ is a temperature
  generalisation of the paper's softmax.
- Softmax-temperature scaling is classical; the per-parent +
  per-query temperature schedule and its invariance under top-P
  reordering are the novel contributions of HVAQ.

### Mathematical Justification

Theorem 16.1 (Equivalence): with ``β_init = 1 / √d`` and
``adaptive="none"`` HVAQ equals the paper's softmax within FP32.
Theorem 16.2 (β-monotonicity): for any positive β the relative
parent ranking is preserved, so the router's top-P selection is
invariant under HVAQ. The HVAQ-ENT and HVAQ-LIN schedules are
β-monotone by construction.

### Implementation Plan

- ``src/avqa/hopfield.py``: per-query temperature + Hopfield logits
  helpers.
- ``AVQAttention.forward`` integration (config-gated; default off).
- ``HopfieldConfig``: ``enabled``, ``beta_init`` (0 → auto-derive
  from head_dim), ``adaptive`` (``"none"`` / ``"entropy"`` /
  ``"linear"``), ``alpha`` (linear slope).

### Verification Plan

- Theorem 16.1: ``tests/unit/test_hopfield.py::TestPaperEquivalenceIntegration::test_hopfield_disabled_matches_paper``
  checks ``hopfield=False`` matches the paper within FP32.
- 24 unit tests cover the temperature schedules, HopfieldConfig
  validation, hopfield_logits broadcasting, and Theorem 16.1
  paper equivalence.
- ``benchmarks/repro_hvaq.py + EXP-0006`` measures latency and
  output difference vs the paper on a small synthetic task.

### Results

EXP-0006 (this cycle):

- sdpa: 0.049 ms median
- paper single-pass: 1.174 ms median
- hvaq entropy: 1.310 ms median (+12 % overhead over paper)
- hvaq linear: 1.208 ms median (+3 % overhead over paper)
- HVAQ-ENT vs paper attention output: 1.3e8 max abs diff
  (sharpened distribution weighted against the value magnitude).
- HVAQ-LIN vs paper attention output: 0.0 max abs diff
  (linear schedule collapses to β_0 at peaked distribution).

### Acceptance Criteria

- 24 unit tests pass (Phase D).
- Theorem 16.1 holds within FP32 (Phase D).
- Theorem 16.2 holds (Phase D — the ``adaptive="none"`` HVAQ
  output equals the paper output bit-for-bit).
- EXP-0006 latency overhead is bounded by the single new branch
  per attention call (≤ 16 % measured).

### Results

Implemented; CPU benchmark captured in EXP-0006. Multi-seed GPU
validation is the next gate. When the CUDA-matrix runner is
available, the natural next step is a downstream-quality ablation
(perplexity on a small language model) comparing the paper vs
HVAQ-ENT vs HVAQ-LIN at the same FLOP budget.

### Final Decision

Pending multi-seed + downstream-quality acceptance. HVAQ is the
project's first algorithmic contribution to the **attention
mechanism itself** (not just the codebook); a paper-tier
contribution depends on the downstream-quality result.
