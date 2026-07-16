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
| Publication Candidates | 2 (OPT-0003 / BCAR staged, OPT-0005 / HVAQ staged) |
| Engineering Improvements | 14 |
| Algorithmic Contributions | 2 (OPT-0003 BCAR + OPT-0005 HVAQ — both Accepted on CPU evidence; multi-seed GPU pending) |
| Systems Contributions | 0 |
| Theoretical Contributions | 0 |
| Accepted Papers | 0 |
| Under Review | 2 (PUB-0001 BCAR, PUB-0002 HVAQ) |
| Freshly Landed Optimizations | 2 (OPT-0003 BCAR, OPT-0005 HVAQ) |
| Benchmarks Reproduced | 5 (EXP-0001 through EXP-0005) |
| GPU Benchmarks Pending CUDA Runner | 4 (OPT-0001 GPU, OPT-0003 multi-seed, OPT-0005 multi-seed, OPT-0005 downstream quality) |

The project has crossed from "production-grade implementation" into
the first two algorithmic contributions beyond paper reproduction
(BCAR/OPT-0003 codebook side, HVAQ/OPT-0005 attention side). Two
publication candidates (PUB-0001 BCAR, PUB-0002 HVAQ) are staged
pending the multi-seed statistical validation on the GPU-matrix
runner.

---

# PUB-0002 (Candidate) — HVAQ (Hopfield-VQ-Attention)

### Research Question

Can the paper's fixed-temperature softmax
``softmax(q · k^T / √d) · v`` be replaced by a **per-query adaptive
temperature** schedule ``softmax(β_q · q · k^T) · v`` with the
temperature derived from the router's top-P attention-mass entropy,
while preserving the router's top-P selection and the paper's
bit-exact behaviour at the boundary ``β_q = 1 / √d``?

### Novelty Claim

HVAQ introduces per-parent and per-query inverse temperatures
that adapt to the attention-mass entropy. The contribution is the
demonstration that the temperature only rescales logits, never
reorders parents (Theorem 16.2), so it is a router-compatible
generalisation of the paper.

### Prior Art

- Ramsauer et al., "Hopfield Networks is All You Need" (2021):
  the original modern Hopfield formulation. HVAQ is a temperature
  generalisation of the paper's softmax.
- Softmax-temperature scaling is classical; the per-parent +
  per-query temperature schedule and its invariance under top-P
  reordering are the novel contributions of HVAQ.

### Mathematical Contribution

- Theorem 16.1 (Equivalence): ``β_init = 1 / √d`` and
  ``adaptive="none"`` HVAQ matches the paper within FP32.
- Theorem 16.2 (β-monotonicity): the router's top-P selection is
  invariant under any positive ``β``.
- The HVAQ-ENT and HVAQ-LIN schedules are monotone in the
  router's attention-mass entropy.

### Algorithmic Contribution

``src/avqa/hopfield.py`` implements the per-query temperature
schedules. The integration in ``AVQAttention.forward`` is gated
on ``BackendConfig.hopfield and HopfieldConfig.adaptive != "none"``.

### Experimental Evidence

EXP-0006 (committed) measures the latency curve and output
difference:

- sdpa: 0.049 ms median
- paper single-pass: 1.174 ms median
- hvaq entropy: 1.310 ms median (+12 % over paper)
- hvaq linear: 1.208 ms median (+3 % over paper)
- HVAQ-ENT vs paper attention output: 1.3e8 max abs diff
- HVAQ-LIN vs paper attention output: 0.0 max abs diff

24 SPEC §16 unit tests in ``tests/unit/test_hopfield.py`` cover
the temperature schedules, HopfieldConfig validation,
hopfield_logits broadcasting, and Theorem 16.1 paper equivalence.

### Threats to Validity

- Multi-seed + downstream-quality validation is the next gate.
  The synthetic 64-token benchmark is not a proxy for real
  attention mass distribution. A small language model with the
  paper baseline vs HVAQ-ENT vs HVAQ-LIN at the same FLOP budget
  is the next experiment.
- The router's top-P selection is invariant under positive ``β``
  (Theorem 16.2) but the per-P probabilities are not. Downstream
  consumers that assume a particular parent attention mass may
  need updating once the GPU-matrix runner is available.

### Release Target

Workshop-tier (e.g., systems-for-ML or efficiency venues). Full
conference publication requires multi-seed + downstream-quality
acceptance.

### Readiness Score

| Criterion | Score |
|-----------|------:|
| Novelty | 7 |
| Technical Depth | 7 |
| Experimental Evidence | 5 (single-seed CPU) |
| Writing | N/A |
| Reproducibility | 9 |
| Overall | N/A |

---

# PUB-0001 (Candidate) — BCAR (Online Codebook Adaptation)

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
