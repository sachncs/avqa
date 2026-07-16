# PUBLICATION.md

> **Project:** AVQA – Adaptive Vector Quantized Attention
>
> **Purpose:** Publication Readiness, Scientific Validation, and Novelty Assessment
>
> This document defines the publication standards for AVQA.
>
> Every algorithmic improvement, optimization, architectural modification, and experimental result SHALL be evaluated against this document before being presented publicly as a novel contribution.
>
> The objective is to ensure that all scientific claims are reproducible, statistically sound, ethically reported, and supported by sufficient evidence.

---

# Publication Dashboard

| Metric | Value |
|--------|------:|
| Publication Candidates | 1 (OPT-0003 / BCAR; candidate PUB-0001 staged) |
| Engineering Improvements | 14 |
| Algorithmic Contributions | 1 (OPT-0003 — Accepted on CPU evidence; GPU statistical validation pending) |
| Systems Contributions | 0 |
| Theoretical Contributions | 0 |
| Accepted Papers | 0 |
| Under Review | 1 (PUB-0001 candidate) |
| Freshly Landed Optimizations | 1 (OPT-0003) |
| Benchmarks Reproduced | 4 (EXP-0001, EXP-0002, EXP-0003 harness, EXP-0004) |
| GPU Benchmarks Pending CUDA Runner | 2 (OPT-0001 GPU and OPT-0003 multi-seed) |

The project has crossed from "production-grade implementation" into
the first algorithmic contribution beyond paper reproduction
(BCAR/OPT-0003). A publication candidate (PUB-0001) is staged
pending the multi-seed statistical validation on the GPU-matrix
runner.

---

# PUB-0001 (Candidate)

## Research Question

Can a previously-static hierarchical codebook be adapted to a
deployment distribution at inference time with the same
per-codeword mean estimator the paper uses offline, while preserving
the parent-child mean constraint of SPEC §7.9 at every step?

## Novelty Claim

BCAR (Bias-Corrected Online Codebook Adaptation) generalises the
paper's offline EMA training (§8.9) to inference time. The
contribution is the demonstration that the same per-codeword
estimator produces a self-adapting codebook on a stationary stream
without any auxiliary training pipeline, and that a clean mean
reprojection after every step maintains SPEC §7.9 exactly (no
approximation gap).

## Prior Art

- Online k-means (Bottou & Bengio 1994) — the convergence rate
  theory we rely on.
- Stochastic K-means — standard references for VQ codebook
  adaptation.
- The AVQ-Attention paper (§8.9): offline EMA training that we
  extend to inference time.

## Mathematical Contribution

The proof is in the algorithm itself: SPEC §13.4 derives the
O(1/N) per-codeword estimator variance under a stationary
distribution; SPEC §7.9 shows that ``parents = mean(children)``
preserves the hierarchical invariant exactly after the children
update.

## Algorithmic Contribution

`src/avqa/online_adaptation.py` implements the algorithm. Compared
to the paper:

- Default behaviour (bcar_enabled=False) is identical to the
  paper.
- Opt-in BCAR adds inference-time adaptation with zero auxiliary
  parameters and zero new dependencies.

## Experimental Evidence

EXP-0004 closes 60.7 % of the static-to-oracle VQ-loss gap in
1024 streaming updates on a synthetic 4-centroid task. Statistical
significance (multi-seed) is the next gate on the CUDA-matrix
runner.

## Threats to Validity

- Single synthetic distribution; real-data downstream perplexity
  ablation is the next step.
- CPU-only so far; GPU timing is folded into OPT-0001's run.

## Release Target

Workshop-quality (e.g., systems-for-ML venues) is the realistic
submission tier; full conference publication requires the multi-seed
validation and at least one downstream quality ablation.

## Readiness Score

| Criterion | Score |
|-----------|------:|
| Novelty           | 6 |
| Technical Depth   | 7 |
| Experimental Evidence | 5 (single-seed CPU) |
| Writing           | N/A (manuscript not drafted) |
| Reproducibility   | 9 |
| Overall           | N/A (manuscript not started) |

---

# Readiness Score (this engineering cycle)

| Criterion | Score (1–10) | Notes |
|-----------|--------------|-------|
| Novelty | 7 | BCAR contributes a genuine algorithmic extension (inference-time EMA); plus speculative compression level is unchanged. |
| Technical Depth | 7 | Triton kernel package is non-trivial; online-softmax accumulators; correcting-attention invariant; online adaptation with mean reprojection. |
| Experimental Evidence | 6 | Four experiments reproduced (EXP-0001 through EXP-0004); GPU statistical significance still pending. |
| Writing | N/A | No publication candidate manuscript yet. |
| Reproducibility | 9 | Scripts + raw artifacts committed; CI mandatory. |
| Overall | N/A | Until a candidate manuscript exists, "readiness" is undefined. |

---

# Readiness Score (this engineering cycle)

| Criterion | Score (1–10) | Notes |
|-----------|--------------|-------|
| Novelty | 6 | AVQ-Attention is the paper's contribution; AVQA reproduces it. Optimization work is engineering unless a novel algorithmic improvement lands. |
| Technical Depth | 7 | Triton kernel package is non-trivial; online-softmax accumulators; correcting-attention invariant. |
| Experimental Evidence | 5 | CPU baselines captured (EXP-0001, EXP-0002); GPU evidence pending CI runner. |
| Writing | N/A | No publication candidate yet. |
| Reproducibility | 9 | Scripts + raw artifacts committed; CI mandatory. |
| Overall | N/A | Until a candidate manuscript exists, "readiness" is undefined. |

---

# Outstanding Gaps Before Publication

1. **CUDA-matrix CI runner**: required to validate the Triton
   kernels against SPEC §11.10 (≥20 % faster than SDPA at
   seq ≥ 4096).
2. **Numerical-equivalence evidence** under SPEC §11.9 tolerances
   (FP32, BF16, FP16) per OPT-0001.
3. **Ablation study** separating the contribution of each Triton
   kernel (VQ, parent attention, child attention, correction) and
   the architectural changes (regenerated ``SPEC.md`` Chapters
   11 + 12).

# Novelty Assessment (current state)

| Question | Answer |
|----------|--------|
| Has this idea already been published? | The algorithm is the paper's contribution; AVQA is an independent implementation. |
| How does AVQA differ from the reference implementation? | A production-grade Python package with a Triton backend; published reference artifacts are paper-level pseudocode. |
| Is the contribution algorithmic, systems, or engineering? | Mostly engineering; one systems contribution (Triton kernels) currently Proposed. |
| Is the improvement incremental or fundamental? | Incremental engineering so far. |
| Would the contribution remain valuable without benchmark gains? | Yes — the package itself, the integration layer, the harness, and the SPEC repopulation are valuable artifacts beyond benchmark wins. |
