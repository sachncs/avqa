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
| Publication Candidates | 0 |
| Engineering Improvements | 14 |
| Algorithmic Contributions | 1 (OPT-0001, Proposed) |
| Systems Contributions | 0 |
| Theoretical Contributions | 0 |
| Accepted Papers | 0 |
| Under Review | 0 |
| Freshly Landed Optimizations | 0 |
| Benchmarks Reproduced | 2 (EXP-0001, EXP-0002) |
| GPU Benchmarks Pending CUDA Runner | 1 |

The project is in the **production-grade implementation** stage; a
publication candidate cannot be claimed until at least one accepted
optimization clears the publication gates.

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
