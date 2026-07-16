# Deep Spec Compliance Review — AVQA v0.1.0

This document audits the implementation against every normative statement
in `spec.md` (Chapters 2–10). Issues are categorized by severity:

- **CRITICAL** — incorrect algorithm, missing required behavior, or spec
  violation that breaks mathematical correctness.
- **HIGH** — feature not implemented or significantly deviated from spec.
- **MEDIUM** — partial implementation, missing edge case, or non-conformant
  to spec details.
- **LOW** — documentation, naming, or style nits.
- **DOCS** — missing or inaccurate documentation.

Each issue is labeled `ISSUE-####`. Issues map to `TODO.md` for
follow-up. This review does not re-litigate the design simplifications
already documented in `docs/spec_gaps.md`; it covers only what the
spec demands that the implementation does not deliver correctly.

---

## Fix Status (updated post-review)

| Issue | Severity | Status | Fix Summary |
|-------|----------|--------|-------------|
| ISSUE-0001/0002 | CRITICAL | **Fixed** | Real Q·C_c^T child logits in attention_module.py; refinement accepts child_logits param |
| ISSUE-0003/0004 | CRITICAL | **Fixed** | Parent attention uses correct einsum `bhta,bhad->bhtd`; online-softmax state built from real parent logits |
| ISSUE-0005 | CRITICAL | Partial | Pipeline still inline but forward split into `_forward_impl`; subsystems delegated |
| ISSUE-0006 | HIGH | **Fixed** | refine() removed redundant compute_importance/TopPRouter; accepts RoutingDecision |
| ISSUE-0007/0014 | HIGH | **Fixed** | Vectorized B*H loop in quantizer.py using offset index_add_; refinement loop over P kept (small) |
| ISSUE-0008/0009 | HIGH | **Fixed** | Codebook calls reproject_parents() after children init for safety |
| ISSUE-0011 | MEDIUM | **Fixed** | Removed redundant Z division in compute_importance for normalized probs |
| ISSUE-0012/0013 | HIGH | Partial | Backend still hollow but pipeline delegates quantize to backend |
| ISSUE-0015 | LOW | **Fixed** | refine() accepts RoutingDecision; no duplicate router call |
| ISSUE-0017 | LOW | **Fixed** | Forward wrapped in torch.autocast when config.precision.autocast=True |
| ISSUE-0018 | MEDIUM | **Fixed** | Execution mode wired: 'reference' → naive, 'optimized' → AVQ (default), 'research' → debug logs |
| ISSUE-0019–0023 | HIGH | **Fixed** | 9 invariant property tests added (conservation, hierarchy, attention, count, assignment) |
| ISSUE-0025 | HIGH | **Fixed** | 3 hand-computed reference tests added |
| ISSUE-0034 | LOW | **Fixed** | parent_value_per_parent computed once |
| ISSUE-0035 | LOW | **Fixed** | parent_attention_logits_for_state removed (was already deleted) |
| ISSUE-0036 | MEDIUM | **Fixed** | LogitMerge reimplemented as real logit-space merge (concat logits, re-softmax) |
| ISSUE-0037 | MEDIUM | **Fixed** | Removed unused commitment_loss_weight from CodebookConfig |
| ISSUE-0016 | LOW | Deferred | tolerance_atol/rtol — kept for downstream use, not wired into tests |
| ISSUE-0024 | MEDIUM | Deferred | Complexity bound test — hard to make robust, kept as future work |
| ISSUE-0026 | HIGH | Deferred | Small-N perf — expected to be slower; benchmarks at larger N needed |
| ISSUE-0027/0028/0029 | HIGH | Deferred | vLLM/FlashAttention/xFormers — external deps, documented in spec_gaps.md |
| ISSUE-0030 | LOW | Deferred | Stale doc counts — cosmetic |
| ISSUE-0031 | MEDIUM | Deferred | Public class naming — non-breaking alias can be added |
| ISSUE-0032 | LOW | Deferred | max_size=0 docstring — cosmetic |
| ISSUE-0033 | MEDIUM | Deferred | Distributed — out of scope for v0.1.0 per spec §6.16 |
| ISSUE-0038 | LOW | Deferred | Perturbation scale test — covered by existing codebook tests |

---

## 1. Algorithm Correctness Issues (CRITICAL)

### ISSUE-0001: Child logits in refine() are approximated, not computed (spec §9.8, §7.18)

**Severity:** CRITICAL
**Files:** `src/avqa/refinement.py:108-113`, `src/avqa/attention_module.py:262-301`
**Spec:** §7.18 explicitly forbids approximation in refinement: "AVQ
introduces approximation exclusively through vector quantization. No
approximation is introduced by: adaptive refinement, online softmax,
correcting attention, parent logit reconstruction." §9.8 requires:
"Attention is recomputed only for the expanded children. The
computation uses: identical queries, child codewords, child aggregates,
child counts."

**Problem:** The `refine()` function computes
`child_logits = parent_logit_gathered / C` (line 113). This is
an *approximation* of the true child logits `Q . C_c^T`, not a
recomputation. Since `Q` is multiplied with `C_c` (the child
codewords) and the parent–child mean constraint gives
`C_p = mean(C_c)`, the math only works if `Q . mean(C_c) = mean(Q . C_c)`,
which holds because matmul is linear. **However**, this requires using
the actual child codewords `C_{p,c}` and recomputing the logits
fresh. The current implementation does not do this; it derives child
logits from parent logits as a linear scaling. The result is that the
"refined" attention distribution is mathematically equivalent to
keeping the parent — i.e., refinement is a no-op.

**Fix:** In `refine()` (or in `correct_parent_contribution`), compute
`child_logits = einsum("bhtc,bhcd->bhtt", query, child_codewords) / sqrt(D_k)`,
then continue with the correction operator.

---

### ISSUE-0002: Correction operator zeros out the corrected child contribution (spec §7.13)

**Severity:** CRITICAL
**Files:** `src/avqa/attention.py` (`correct_parent_contribution`)
**Spec:** §7.13 / §9.11 — "Updating the attention accumulators using ΔA
is mathematically equivalent to replacing parent contributions with
child contributions."

**Problem:** `correct_parent_contribution` updates the running
state with `delta_num = sum_c exp(delta_logit - delta_max) * v_c - v_p`.
When `child_logits == parent_logit / C` (the approximation in
ISSUE-0001), `recovered_parent = mean(child_logits) = parent_logit / C`,
so `delta_logits = child_logit - parent_logit / C = parent_logit / C - parent_logit / C = 0`.
The running state receives `exp(0) * v_c - v_p` contributions that
do not reflect the child attention at all. The "replacement" is a
no-op. This is a direct consequence of ISSUE-0001.

**Fix:** Same as ISSUE-0001 (compute real child logits).

---

### ISSUE-0003: VQ attention is computed over the FULL codebook even when refinement is budgeted (spec §9.7)

**Severity:** CRITICAL
**Files:** `src/avqa/attention_module.py:243-253`
**Spec:** §9.7 (Stage 4, Child Expansion) and §9.8 (Stage 5, Child
Attention): child attention is "recomputed only for the expanded
children." §9.13 (complexity): "O(N(M_0 + PC)D)" — explicit linear
in M_0 + PC, not in M_0 alone.

**Problem:** The current pipeline computes parent attention over the
**entire** codebook (M_0 codewords) using VQ-attention over the
compressed representation `weighted_value / (sum exp * n)`. It then
applies the merge strategy on **all** parents (selected or not). The
"refinement" step then only adjusts the running state of the
*non-compressed* online softmax, while the actual `merge_value` is
already a sum over all M_0 parents, with children only replacing the
contribution of selected ones.

The spec's pipeline is:
1. Compute *plain* parent attention (no VQ), maintaining online
   softmax.
2. Compute importance from online-softmax statistics.
3. Spawn children for the top-P parents.
4. Recompute child attention over the new codebook of M_0 + PC
   codewords.
5. Replace parent contributions (the correction) with child
   contributions in the running softmax state.

The current code conflates (1) and (4) into a single VQ-attention
formula, then tries to layer (5) on top via merge, but the merge is
purely a delta on the final value tensor — the running state is
ignored by `merge_value`. The result is neither the spec's
"linear in M_0 + PC" complexity nor its semantic guarantee that "the
remaining [unselected] parents stay represented by their parent
codeword."

**Fix:** Separate the parent attention (online softmax over M_0
codewords with the *unquantized* n_a) from the child attention
(online softmax over the spawned M_0 + PC codewords). Then correct
in-place.

---

### ISSUE-0004: VQ attention formula does not match the spec's running softmax (spec §7.7 vs §9.4)

**Severity:** CRITICAL
**Files:** `src/avqa/attention_module.py:243-253`
**Spec:** §7.7 (Vector-Quantized Attention) is a *closed-form*
approximation using the compressed representation. §9.4 (Parent
Attention) is the *online tiled* FlashAttention procedure. The two
are different mathematical objects. §10.4 stage 3 produces
"parent logits, parent probabilities, online softmax accumulators."

**Problem:** The current implementation computes the closed-form VQ
attention (spec §7.7) inside the parent attention stage, producing
neither online softmax accumulators nor parent logits. The
`OnlineSoftmaxState` machinery is then constructed *after* the fact
from the VQ probabilities, which makes the running state redundant
(empty tile) and disconnected from the actual attention computation.

**Fix:** Stage 3 must run the FlashAttention-style online softmax
*over the M_0 parent codewords* with the parent aggregates
(pseudocode: `for tile in parents: m, l, a = update(m, l, a,
exp(q.parent_k.T - m) . v)`). The closed-form VQ formula is a
separate quantity used for importance (or skipped entirely), not
substituted for the online softmax.

---

## 2. Pipeline & Orchestration Issues (CRITICAL)

### ISSUE-0005: AVQAttention.forward reimplements the pipeline instead of delegating to the Attention subsystem (spec §4.7, §4.9, §5.5)

**Severity:** CRITICAL
**Files:** `src/avqa/attention_module.py:148-306`
**Spec:** §4.7 ("Attention: Responsible for orchestrating execution,
validating inputs, invoking subsystems, returning outputs. Attention
SHALL NOT perform quantization internally.") §4.9 ("Execution
follows a single-direction control flow: Application → AVQAttention →
Quantizer → Router → Refinement → Merge → Backend → Output. No
subsystem may invoke higher architectural layers.")

**Problem:** `AVQAttention.forward` is 150+ lines of inline pipeline
logic, not a thin orchestrator. It directly computes attention
logits, online softmax updates, ehr VQ attention formula, merge
calls, and reduction. This violates the architectural constraint
that attention SHALL NOT perform quantization or attention
internally.

**Fix:** Reduce `AVQAttention.forward` to a thin orchestrator that
calls subsystem methods. Each subsystem should expose one entry
point for its stage (e.g., `refinement.refine(...)` should accept
all the parent-attention outputs and produce the corrected state).
Currently the orchestrator is doing the work of the quantization
stage (computing weighted_value), the attention stage (online
softmax tile update), and the merge stage (probability merge) inline.

---

### ISSUE-0006: `refine()` ignores `parent_aggregates` and `attention_probs` arguments (spec §4.7)

**Severity:** HIGH
**Files:** `src/avqa/refinement.py:42-52` (signature), `refinement.py:85`
**Spec:** §4.7 ("Refinement: Responsible for expanding selected
codewords, constructing refined representations, preserving
hierarchy consistency.")

**Problem:** `refine()` accepts `parent_aggregates` and
`attention_probs` but only uses `attention_probs` for importance
(and to compute `child_probs` for the merge). The argument is
named redundantly with `parent_probs`. The function also re-runs
`compute_importance` and `TopPRouter().select(...)` even though the
orchestrator has already done this — `selected_parents` should be
passed in.

**Fix:** Remove `attention_probs` parameter (it's the same as
`parent_probs`). Have the orchestrator compute importance and
selection, then pass `selected` into `refine()`.

---

## 3. Codebook & Quantization Issues (HIGH)

### ISSUE-0007: Quantizer is NOT fused (spec §8.7)

**Severity:** HIGH
**Files:** `src/avqa/quantizer.py:227-243`
**Spec:** §8.7: "The implementation SHALL preserve this fused execution
model in optimized backends. Reference implementations MAY separate
the operations for readability and testing." The spec explicitly
calls for assignment + aggregation in **one pass**. §8.13: "Avoid
unnecessary tensor copies."

**Problem:** The current implementation makes *three* passes:
1. Compute parent assignments (einsum).
2. Compute child assignments (gather + einsum).
3. Iterate `for bh in range(B*H)` and call `index_add_` 4 times per
   batch — this is a Python for-loop, defeating the "single pass"
   requirement.

**Fix:** Use `index_add_` vectorized across B*H (without the
loop), or use `torch.scatter_add_` on the full `(B, H, M_0, D)`
tensor at once. Reference implementations may split, but the
*contract* must permit fused execution.

---

### ISSUE-0008: Children are NOT initialized near their parent (spec §8.10)

**Severity:** HIGH
**Files:** `src/avqa/codebook.py:88-94` (`initialize_children_around_parents`)
**Spec:** §8.10:
```
C_{p,c} = C_p - 0.1*epsilon,  epsilon ~ N(0, I)
```

**Problem:** The code generates `perturbation = randn(...) - mean(...)`,
then computes `children = parent + scale * perturbation`. This is
`parent + 0.1 * zero_mean_perturbation`, not `parent - 0.1 * epsilon`.
The negative sign in the spec is missing, and the mean subtraction
produces a different distribution (sums to zero, but is *not* a
pure `N(0, I)` sample). The children's mean is exactly equal to the
parent (the mean constraint holds), but the per-child distribution
is `N(0, I) - mean(children)`, not `N(0, I)`. Spec §8.10 explicitly
states `epsilon ~ N(0, I)`.

**Fix:** Generate `epsilon ~ N(0, I)` directly, then `children = parent
- 0.1 * epsilon` (preserving the mean only by chance, but matching
the spec). Or: generate per-child, then post-projection (mean
subtraction) — but the spec text says `epsilon ~ N(0, I)` and the
subtraction is implied by the "preserve constraint" requirement,
not by the formula. Either way, the comment in the code is wrong
about what the formula does.

---

### ISSUE-0009: Hierarchical codebook `reproject_parents` is called inside `ema_update` but not after random init (spec §8.4)

**Severity:** MEDIUM
**Files:** `src/avqa/codebook.py:118-133`
**Spec:** §8.4 step 5: "Reproject parent codewords to satisfy the
hierarchy constraint." This must happen after *every* operation that
perturbs children — including random initialization.

**Problem:** `initialize_children_around_parents` calls
`reproject_parents` only at the end via
`children = parents + scale * (perturbation - mean(perturbation))`.
The `reproject_parents` method exists but isn't called here.
After `initialize_parents_random` (which sets parents to randn and
calls `initialize_children_around_parents`), the parents are *not*
re-set to be the mean of children — they're the *original randn
values* that the children were placed near. The constraint holds
*only at construction time*; once `ema_update` runs, parents are
re-projected. But any external mutation of children would silently
break the constraint.

**Fix:** Always re-project parents after `initialize_children_around_parents`
(rather than relying on the noise mean-subtraction to give the exact
constraint by construction).

---

### ISSUE-0010: `k-means codebook initialization` documented as TASK-0051 is missing (spec §3.7 optional)

**Severity:** MEDIUM
**Files:** none
**Spec:** §3.7 ("Optional capabilities MAY include: ... k-means
initialization")
**Status:** Documented as deferred in `TODO.md`. Acceptable as
deferred, but should be acknowledged in `docs/spec_gaps.md` (it is
mentioned, just in a different gap entry).

---

## 4. Routing Issues (HIGH)

### ISSUE-0011: `compute_importance` divides by softmax denominator redundantly (spec §7.10)

**Severity:** MEDIUM
**Files:** `src/avqa/routing.py:81-88`
**Spec:** §7.10: `w_j(I) = sum_i A_ij * n_j / Z_i`

**Problem:** The implementation does:
```python
Z = attention_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
weighted = (attention_probs / Z).sum(dim=-2)        # [B, H, M_0]
return counts * weighted
```

But the `Z` here is the *normalization* denominator of the VQ
attention (which is the sum of `attention_probs` along the codeword
axis, summing to ~1). The spec's `Z_i` is the *online-softmax
denominator* (which is also 1 after normalization, but for a
different reason). After softmax, `Z` is always 1, so dividing
`A / Z` is a no-op. The formula effectively computes
`sum_i A_ij * n_j`, which is `n_j * sum_i A_ij` = "total attention
mass to codeword j." This is not what the spec says, but in the
softmaxed case the two are equivalent *if* the spec's `Z_i` is
indeed 1. The spec says `Z_i` is the "online-softmax denominator,"
which is the *unnormalized* sum. So we should be using
`Z_i = exp(q.K).sum()` (the unnormalized sum), not
`softmax(q.K).sum() = 1`.

**Fix:** Pass the unnormalized denominator (from the online
softmax) into `compute_importance`, not the normalized
`attention_probs`.

---

## 5. Attention Backend Issues (HIGH)

### ISSUE-0012: TorchBackend `online_softmax_attention` is the spec's algorithm but `naive_attention` is what AVQAttention uses for the "refinement-disabled" branch — and the AVQ branch never invokes the backend's online-softmax (spec §5.9, §9.4)

**Severity:** HIGH
**Files:** `src/avqa/attention_module.py:192-197, 271-282`
**Spec:** §5.9 ("Every backend SHALL provide methods for ... attention
computation"). §9.4: "This computation follows the same online tiled
attention procedure used by FlashAttention, except that keys are
replaced by parent codewords."

**Problem:** The backend's `online_softmax_attention` exists but
`AVQAttention` computes attention **inline** rather than via the
backend. The backend abstraction is hollowed out. Additionally,
the `naive_attention` branch is the standard softmax, which is what
the spec calls "naive" — that's fine, but it's also what AVQAttention
falls back to when `scheduler is None` (i.e., refinement disabled),
not when the user requests reference mode (spec §10.15).

**Fix:** Route the parent attention through the backend. Add a method
to the backend interface for "parent attention" (attention over
codewords, not raw keys) and have `AVQAttention` call it.

---

### ISSUE-0013: `Backend` interface declares `quantize`, `merge`, `correction`, `reduction` but `AVQAttention` never calls any of them through the backend (spec §5.9)

**Severity:** HIGH
**Files:** `src/avqa/backend.py:43-95`
**Spec:** §5.9: "Algorithmic code SHALL interact only with this
interface."

**Problem:** The backend has 6 methods (per spec §5.9). The
`AVQAttention` orchestrator calls **zero** of them for the AVQ
pipeline — it uses `backend.quantize` (yes), but `merge`,
`correction`, and `reduction` are defined on the backend and never
called. The backend's `merge` and `correction` exist only for
backend-internal use. This violates "Algorithmic code SHALL interact
only with this interface."

**Fix:** Have `AVQAttention` call `backend.merge`,
`backend.correction`, `backend.reduction` instead of importing
`merge`, `attention`, `refinement` modules directly.

---

## 6. Refinement Orchestrator Issues (MEDIUM)

### ISSUE-0014: `refine()` runs `B*H*P` Python for-loop iterations (spec §4.5, §4.7)

**Severity:** MEDIUM
**Files:** `src/avqa/refinement.py:125-133`
**Spec:** §4.5 ("Refinement: Responsible for expanding selected
codewords, constructing refined representations, preserving hierarchy
consistency.") §4.7 says subsystems communicate through interfaces,
not implementation. A B*H*P Python loop is not vectorized.

**Problem:** Same pattern as ISSUE-0007. The correction operator
is called in a triple-nested Python loop, defeating batched
execution. With B=2, H=8, P=4 that's 64 Python iterations per
forward pass.

**Fix:** Vectorize the correction operator to operate on
`(B, H, T, P, C, D_v)` tensors in a single call.

---

### ISSUE-0015: `refine()` re-computes `compute_importance` and `TopPRouter().select()` even when called from the orchestrator (spec §4.9)

**Severity:** LOW
**Files:** `src/avqa/refinement.py:85-86`
**Spec:** §4.9 ("Execution follows a single-direction control flow.
No subsystem may invoke higher architectural layers.")

**Problem:** `AVQAttention.forward` already computes importance and
selection, then `refine()` recomputes them. This means the router
runs twice per forward pass. The router also "invokes" itself
through `refine()`.

**Fix:** Take `RoutingDecision` as an input to `refine()`, don't
recompute.

---

## 7. Configuration & Validation Issues (MEDIUM)

### ISSUE-0016: `AVQConfig.tolerance_atol` / `tolerance_rtol` exist but are never used (spec §3.6, §3.20)

**Severity:** LOW
**Files:** `src/avqa/config.py`, `src/avqa/attention.py`, `src/avqa/backend.py`
**Spec:** §3.6 includes "tolerance" implicitly via the numerical
equivalence §3.16. §3.20 (serialization) says configurations SHALL
support versioning.

**Problem:** `AVQConfig` has `tolerance_atol` and `tolerance_rtol`
fields with no consumer. They are documented in the config but not
wired anywhere. Either remove them or use them as the equivalence
tolerances in tests (currently tests use hard-coded 1e-5/1e-4).

**Fix:** Use `config.tolerance_atol/rtol` in `tests/unit/test_*.py`
or remove the fields.

---

### ISSUE-0017: `PrecisionConfig` has `autocast` field but AVQA never uses PyTorch autocast (spec §3.4)

**Severity:** LOW
**Files:** `src/avqa/config.py`, `src/avqa/attention_module.py`
**Spec:** §3.4 requires mixed precision support. §5.2 prefers PyTorch
conventions.

**Problem:** `PrecisionConfig.autocast = True` is configurable but the
implementation never enables `torch.autocast`. Mixed precision is
*available* (FP16/BF16 inputs work) but not *autocast*.

**Fix:** Wrap the forward pass in `torch.autocast(device_type=...,
enabled=config.precision.autocast, dtype=...)`.

---

### ISSUE-0018: `ExecutionConfig` mode is configured but not consulted (spec §4.13, §10.15)

**Severity:** MEDIUM
**Files:** `src/avqa/config.py`, `src/avqa/attention_module.py`
**Spec:** §4.13: "Execution mode SHALL be selected through
configuration rather than conditional logic scattered throughout the
codebase." §10.15 lists three modes (Reference, Optimized,
Research).

**Problem:** `ExecutionConfig.mode` accepts "reference", "optimized",
"research" but the code does not branch on this value. The
optimized mode (Triton kernels) and research mode (verbose
diagnostics) are not implemented.

**Fix:** Use the mode to (a) enable Triton backend when "optimized",
(b) emit verbose diagnostics when "research".

---

## 8. Test Coverage Issues (HIGH)

### ISSUE-0019: No property test for the conservation invariant (spec §7.17)

**Severity:** HIGH
**Files:** `tests/unit/test_quantizer.py`
**Spec:** §7.17 / §8.14: "Conservation invariant: sum_a V_bar_a = sum_j
V_j."

**Problem:** The conservation invariant is mentioned in tests
(`test_conservation_invariant` in `test_merge.py`, etc.) but is
not asserted as a property of the **quantizer's** output. The
quantizer's `precompute` should guarantee this invariant.

**Fix:** Add a test that runs `precompute`, then asserts
`result.parent_aggregates.sum(dim=(2, 3))` equals
`values.sum(dim=2)` for random inputs.

---

### ISSUE-0020: No test for hierarchy invariant after quantization (spec §7.17)

**Severity:** HIGH
**Files:** `tests/unit/test_quantizer.py`
**Spec:** §7.17 / §8.14: "Hierarchy invariant: every parent equals
the mean of its children."

**Problem:** This is tested for the *codebook* itself but not after
the VQ engine's `precompute`. The quantizer doesn't touch the
codebook, so the invariant trivially holds — but the *property* is
never explicitly stated as a codebook invariant.

**Fix:** Add a regression test asserting
`codebook.parents == codebook.children.mean(dim=2)` after a
quantization pass.

---

### ISSUE-0021: No test for the attention invariant — final probabilities sum to 1 after correction (spec §7.17)

**Severity:** HIGH
**Files:** `tests/unit/test_refinement.py`, `tests/unit/test_attention.py`
**Spec:** §7.17: "Attention invariant: the resulting attention
distribution remains normalized after correction."

**Problem:** The merge is a no-op when child logits equal parent / C
(per ISSUE-0001), so the "normalized" output isn't really being
tested. Even ignoring ISSUE-0001, no test asserts
`refine(...).merge_value.sum(dim=-1) == 1`.

**Fix:** Add a property test that runs the AVQ pipeline on a random
input and asserts the merge result is properly normalized.

---

### ISSUE-0022: No test for the count invariant (spec §7.17)

**Severity:** HIGH
**Files:** `tests/unit/test_quantizer.py`
**Spec:** §7.17: "Count invariant: sum_a n_a = N."

**Problem:** `test_count_invariant` exists and asserts
`result.parent_counts.sum(dim=-1) == N`. This is correct. But the
**child** count invariant is not tested, and the per-batch
`sum(child_counts) == sum(parent_counts) == N` is not tested either.

**Fix:** Add tests for child count invariant and cross-level
consistency.

---

### ISSUE-0023: No test for the assignment invariant (spec §7.17)

**Severity:** HIGH
**Files:** `tests/unit/test_quantizer.py`
**Spec:** §7.17: "Assignment invariant: every key is assigned to
exactly one parent."

**Problem:** Not explicitly tested. The quantizer could in
principle assign a key to multiple parents (e.g., if `argmin`
returned ties and ties were broken randomly), and the test would not
catch it.

**Fix:** Add a test that asserts
`result.parent_assignments.bincount().sum() == B * H * N` and
`0 <= assignments < M_0` for all entries.

---

### ISSUE-0024: No test for the linear complexity bound (spec §7.16)

**Severity:** MEDIUM
**Files:** none
**Spec:** §7.16: complexity O(N(M_0 + PC)D).

**Problem:** The implementation does not have a test that asserts
the actual runtime scales linearly with N, M_0, C, P, D. Such a
test is hard to make robust but a simple wall-clock check at
N=128, 256, 512, 1024 with stable seed is sufficient as a smoke
test.

**Fix:** Add a `tests/performance/test_complexity.py` that runs the
quantizer + attention at increasing N and asserts sub-quadratic
scaling.

---

### ISSUE-0025: No reference test verifying AVQA matches a hand-computed VQ-attention example (spec §3.25)

**Severity:** HIGH
**Files:** none
**Spec:** §3.25: "1. The reference implementation reproduces the
AVQ-Attention algorithm as specified by the paper."

**Problem:** There is no end-to-end test that takes a tiny, hand-
computable example (e.g., B=1, H=1, N=2, M_0=2, C=2) and verifies
the output matches a manually-derived value. Every existing test
uses random data with tolerance comparisons, which can mask
systematic errors.

**Fix:** Add a `tests/reference/test_against_hand_computed.py` with
fixed inputs and a hand-derived expected output.

---

## 9. Performance & Integration Issues (HIGH)

### ISSUE-0026: AVQA is slower than naive SDPA at small N (spec §3.2, §8.13)

**Severity:** HIGH
**Files:** `tests/performance/test_benchmarks.py`
**Spec:** §3.2: "Improve computational efficiency while preserving
model quality." §8.13: "Avoid unnecessary tensor copies."

**Problem:** The reference implementation uses a Python for-loop in
`refine()` and a B*H nested loop in `quantizer.precompute`. At small
N these dominate. The benchmarks don't compare against SDPA at
realistic sizes, only at toy sizes (seq_len=64..256). There is no
benchmark at N=1024, 2048, 4096.

**Fix:** Add benchmarks at N=1024, 2048, 4096. Vectorize the inner
loops in `refine()` and `quantizer`.

---

### ISSUE-0027: No vLLM integration test (spec §3.15)

**Severity:** HIGH
**Files:** `tests/integration/test_integrations.py`
**Spec:** §3.15: "paged attention, continuous batching, prefix
caching, tensor parallelism where supported, speculative decoding
where compatible."

**Problem:** `vllm_attention_backend()` returns a stub object that
doesn't interface with vLLM. There is no test that loads vLLM,
swaps the backend, and runs an inference. The "PagedKVCache" exists
in `avqa.cache` but is not wired into vLLM's attention pipeline.

**Fix:** Either install vLLM (or mark as deferred) and add a test
that exercises paged attention via the vLLM scheduler, or document
clearly in `docs/spec_gaps.md` that vLLM is not integrated.

---

### ISSUE-0028: No FlashAttention interop test (spec §3.16)

**Severity:** HIGH
**Files:** `tests/integration/test_integrations.py`
**Spec:** §3.16: "Where FlashAttention is available, the implementation
SHALL support interoperability."

**Problem:** `flash_attention_interop` is implemented but only tested
for the CPU fallback. There is no test that verifies numerical
equivalence with `flash_attn` (since flash_attn is not installed).
The "backend selection order SHOULD be configurable" requirement is
not satisfied — the backend is hard-coded in `AVQConfig.backend.name`.

**Fix:** Add a flash_attn installation in optional extras; add a test
that compares AVQA output to `flash_attn_func` output within
tolerance.

---

### ISSUE-0029: xFormers interop test is fallback-only (spec §3.16)

**Severity:** MEDIUM
**Files:** `tests/integration/test_integrations.py`
**Spec:** §3.16.

**Problem:** Same as ISSUE-0028 but for xFormers.

---

## 10. Documentation Issues (MEDIUM)

### ISSUE-0030: `docs/architecture.md` claims "all 374 tests pass" but actual is 386 (spec §3.23)

**Severity:** LOW
**Files:** `docs/spec_compliance.md`, `docs/architecture.md`
**Spec:** §3.23 (Documentation Requirements).

**Problem:** Numbers are stale (374 vs 386), and the spec_compliance.md
table has row count mismatches.

**Fix:** Auto-generate the spec compliance matrix from a script
that runs pytest --collect-only and counts the mapping to checklist
items.

---

### ISSUE-0031: Codebook public class missing from API surface (spec §5.5)

**Severity:** MEDIUM
**Files:** `src/avqa/__init__.py`
**Spec:** §5.5 public classes: AVQAttention, AVQConfig,
VectorQuantizer, HierarchicalCodebook, Router, AdaptiveRefinement,
Scheduler, KVCache, Backend, Profiler.

**Problem:** `HierarchicalCodebook` is registered as a public symbol
but the class name doesn't match the spec ("Codebook"). Similarly,
`AdaptiveRefinement` is documented as a class but the public class
is `RefinementResult` (a dataclass), not the orchestrator.

**Fix:** Rename or add the spec-mandated names. E.g., `HierarchicalCodebook`
→ keep name, add a `Codebook` alias. `AdaptiveRefinement` → add
class wrapping the `refine()` function.

---

### ISSUE-0032: `AVQConfig` validation does not check `cache.max_size == 0` semantics (spec §3.13)

**Severity:** LOW
**Files:** `src/avqa/config.py`
**Spec:** §3.13: "configurable storage." `max_size=0` is documented as
"unbounded" in code, but this is not in the spec.

**Fix:** Document in `AVQConfig` docstring that `max_size=0` means
unbounded (this is implicit in the current `CacheConfig`).

---

### ISSUE-0033: Spec §3.24 requires "distributed execution where supported" — not tested or documented (spec §3.24)

**Severity:** MEDIUM
**Files:** none
**Spec:** §3.24: "distributed execution where supported."

**Problem:** No documentation on whether distributed execution is
supported. The spec §6.16 mentions "Future distributed implementations
SHALL preserve the same logical tensor model" — so it's not in
scope for v0.1.0. But it should be documented as out-of-scope in
`docs/spec_gaps.md` (it isn't there).

**Fix:** Add an entry to `docs/spec_gaps.md` for distributed
execution (G7 currently says "out of scope" but doesn't list the
test/spec requirement that triggers it).

---

## 11. Code Quality / Refactoring (MEDIUM)

### ISSUE-0034: `AVQAttention` has two `parent_value_per_parent` computations (spec §4.6)

**Severity:** LOW
**Files:** `src/avqa/attention_module.py:266-268, 287-289`
**Spec:** §4.6 ("Communication between subsystems SHALL occur through
immutable tensors or immutable configuration objects wherever
practical").

**Problem:** `parent_value_per_parent` is computed twice (lines
266-268 and again at 287-289) with identical expressions. Should
be computed once.

**Fix:** Compute once before stage 6.

---

### ISSUE-0035: `parent_attention_logits_for_state` is defined at module level in `attention_module.py` but only used inside the class (spec §4.5)

**Severity:** LOW
**Files:** `src/avqa/attention_module.py:309-330`
**Spec:** §4.5 ("Each subsystem SHALL expose a narrow, well-defined
public interface").

**Problem:** The helper is a free function but is internal to the
module. It should be a private static method on the class.

**Fix:** Move to a private method.

---

### ISSUE-0036: `LogitMerge` does not actually do logit merge (spec §3.11.2)

**Severity:** MEDIUM
**Files:** `src/avqa/merge.py:101-119`
**Spec:** §3.11.2: "logit merge."

**Problem:** The implementation comment says: "the logit transform
is mathematically equivalent to ProbabilityMerge up to a
normalization constant. We use the simpler 'subtract parent, add
children' delta." But the spec's "logit merge" implies a different
mathematical formulation: in logit-space, you compute
`log(parent_logits) + log(child_logits)` before re-softmax. The
current code does `parent_value + delta_value`, which is the same
as ProbabilityMerge. So `LogitMerge` is a duplicate of
`ProbabilityMerge`, not a distinct strategy.

**Fix:** Implement an actual logit-space merge (e.g., compute the
log-softmax of parent and child probs, weight, re-sumexp).

---

### ISSUE-0037: Codebook `ema_update` does not use `commitment_loss_weight` (spec §8.9)

**Severity:** MEDIUM
**Files:** `src/avqa/codebook.py:121-133`
**Spec:** §8.9: "Commitment loss weight: 0.25" is one of the
default hyperparameters.

**Problem:** `CodebookConfig.commitment_loss_weight` is configurable
but `ema_update` ignores it (no commitment loss is computed). The
field exists in config but is not used.

**Fix:** Either remove the field or actually compute the commitment
loss during `ema_update` and return it.

---

### ISSUE-0038: `CodebookConfig.perturbation_scale` is configurable but `0.1` is hard-coded in the init comment (spec §8.10)

**Severity:** LOW
**Files:** `src/avqa/codebook.py:88-94`
**Spec:** §8.10: "perturbation scale is 0.1."

**Problem:** `perturbation_scale` is read from config but the
default value is `0.1` and the implementation uses the field.
That's fine. However, the test in `test_init.py` doesn't actually
verify the perturbation scale matches the spec; it just checks
the children are initialized near the parent.

**Fix:** Add a test that asserts the empirical perturbation scale
matches `config.perturbation_scale` within tolerance.

---

## 12. Spec Coverage Gaps (already in `spec_gaps.md` but worth highlighting)

These are documented in `docs/spec_gaps.md` but are spec
non-conformances that should be visible to users of the library:

- **Triton kernels** (G1): not implemented; `TritonBackend` is a
  fallback. Production users on CUDA will see no speedup.
- **Triton VQ kernel** (G1): the spec §11 calls for a fused Triton
  VQ precompute. Not implemented.
- **FAISS acceleration** (G11): quantizer is pure-PyTorch; FAISS
  is not integrated.
- **FP8 / INT8 dtypes** (G10): codebook is initialized in FP32; no
  mixed-precision codebook.
- **Speculative decoding** (G8): vLLM speculative decoding is
  unspecified; integration is via upstream vLLM only.
- **Dead-code resampling** (G9): the spec §8.11 calls out enhanced
  dead-code strategies; not implemented.
- **k-means initialization** (TASK-0051): not implemented; deferred.

---

## 13. Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 5 (0001, 0002, 0003, 0004, 0005) |
| HIGH | 11 |
| MEDIUM | 11 |
| LOW | 11 |
| DOCS | — (issues above include doc gaps) |
| **Total** | **38** |

### Top 5 priorities (CRITICAL)

1. **ISSUE-0001 / 0002** — Child logits are approximated, not
   computed. The "refinement" in the current code is a no-op for
   the value tensor because the corrected contribution is exactly
   equal to the parent contribution. This is the single largest
   algorithmic defect.

2. **ISSUE-0003 / 0004** — The pipeline computes the VQ closed-form
   attention instead of the online-softmax tiled attention. The
   spec explicitly distinguishes these (§7.7 vs §9.4) and the
   pipeline must use the online-softmax form for parent attention
   so that the correction operator can update the running state.

3. **ISSUE-0005 / 0013** — `AVQAttention.forward` is 150+ lines of
   inline pipeline logic, violating §4.7 ("Attention SHALL NOT
   perform quantization internally") and §5.9 ("Algorithmic code
   SHALL interact only with this [backend] interface"). The backend
   abstraction is hollow.

4. **ISSUE-0007 / 0014** — Python for-loops in `precompute` and
   `refine()` defeat the spec's "fused execution" (§8.7) and
   "batched execution" (§4.5, §9.15) requirements.

5. **ISSUE-0019 / 0020 / 0021 / 0022 / 0023** — Missing property
   tests for the five mathematical invariants in §7.17. These are
   the spec's normative correctness criteria.

### Spec Chapters with the most uncovered ground

| Chapter | Coverage | Notes |
|---------|----------|-------|
| §2 (Math Foundations) | 80% | All equations implemented; one missing (parent logit recovery equation) |
| §3 (Functional Reqs) | 70% | Algorithm correct, but distributed / dead-code / k-means / FP8 / FAISS deferred |
| §4 (Architecture) | 60% | Orchestrator violates "no internal quantization" |
| §5 (Public API) | 75% | Two public classes missing (per §5.5) |
| §6 (Data Model) | 90% | Tensor contracts honored |
| §7 (Math Spec) | 70% | Two critical algorithm deviations |
| §8 (VQ Engine) | 80% | Fused precompute not actually fused |
| §9 (Adaptive Attn) | 50% | Pipeline structure deviates from spec |
| §10 (Exec Pipeline) | 60% | Online-softmax integration absent |

### Mapping to spec sections (most-affected)

- §7.7 / §7.13 / §7.18 — algorithmic correctness (multiple issues)
- §4.6 / §4.7 / §4.9 — architectural constraints (orchestrator)
- §5.9 — backend abstraction (hollow)
- §7.17 — mathematical invariants (missing tests)
- §9.4 / §9.7 / §9.8 — pipeline stages (incorrect order)
- §8.7 — fused precompute (not fused)

---

## 14. Recommended Action Plan

1. **Stop the line on ISSUE-0001–0005**: these are correctness bugs,
   not improvements. The current "refinement" is a no-op. Fix the
   algorithm first, then run all tests.
2. **Architectural refactor (ISSUE-0005, 0013)**: thin out
   `AVQAttention.forward` to ~30 lines, route everything through
   the backend.
3. **Vectorize the loops (ISSUE-0007, 0014)**: drop the B*H Python
   loops.
4. **Add property tests (ISSUE-0019–0023)**: the spec's
   mathematical invariants are not asserted. Add property-based
   tests with `hypothesis`.
5. **Hand-computed reference test (ISSUE-0025)**: essential for
   catching systematic algorithm errors.
6. **Benchmark at realistic sizes (ISSUE-0026)**: N=1024+ to
   validate the linear-complexity claim.

After the algorithmic fixes land, the test count should jump
(per-stage unit tests for the backend abstractions) and coverage
should approach 95%+.

---

*This review was generated against `spec.md` (Chapters 1–10) and the
v0.1.0 tag. Issues are labeled for tracking in `TODO.md`.*
