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

| ID       | Title                        | Priority     | Status   |
| -------- | ---------------------------- | ------------ | -------- |
| OPT-0001 | Adaptive refinement budget   | High         | Proposed |
| OPT-0002 | Residual vector quantization | Medium       | Proposed |
| OPT-0003 | Learned hierarchy            | Experimental | Proposed |

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
