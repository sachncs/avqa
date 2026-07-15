# Specification Gaps & Implementation Assumptions

The authoritative specification document `spec.md` covers Chapters 1–10
explicitly. Chapters 11–15 (Triton kernel implementation details, framework
integration internals, profiling internals, visualization rendering,
serialization versioning schema, full testing requirements, packaging
specification) are referenced by Chapters 3 and 4 but their normative text is
not present in `spec.md`.

Per the project policy "Implementation Completeness Policy", gaps are
recorded here and resolved with explicit implementation assumptions. Each
assumption is documented with the rationale and the impact on spec
compliance.

---

## Gap G1 — Detailed Triton Kernel Specification

**Spec section affected:** §4.10, §8.7, §10.20 (referenced), §11 (missing).

**Gap:** The spec mandates an optional Triton backend (§3.2.7, §3.12.1) but
does not specify the kernel structure, tiling parameters, autotune space, or
memory layout.

**Assumption:** Implement the Triton backend following the
FlashAttention-2 online-softmax tiling pattern with block sizes that
autotune over (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). Kernels:

- `avqa_vq_precompute`: fused parent assignment + value aggregation.
- `avqa_parent_attention`: online softmax over parent codewords.
- `avqa_child_attention`: online softmax over child codewords of selected
  parents only.
- `avqa_correction`: incremental correction of parent contributions.

The Triton backend is gated behind `avqa.backend.triton_backend` and is
importable only when `triton` and CUDA are available. CPU/MPS execution
falls back to the Torch backend.

**Impact:** Until a future spec chapter defines exact kernel structure, the
Triton backend is treated as a high-performance alternative to the Torch
backend. The Torch backend is the canonical reference.

---

## Gap G2 — Detailed Framework Integration Specification

**Spec section affected:** §3.14 (HF), §3.15 (vLLM), §3.16 (FlashAttention),
§5.17, §12 (missing).

**Gap:** The spec requires integrations with Hugging Face, vLLM,
FlashAttention, and xFormers, but does not specify the exact API surface,
mapping of model classes, or replacement procedure.

**Assumption:** Each integration follows the dominant pattern used by the
respective framework:

- **Hugging Face:** Provide `replace_attention(model, config)` that walks
  `model.named_modules()`, replaces attention modules whose class names match
  the patterns `*Attention` (excluding `*ForCausalLM` etc.), and remaps
  pretrained `q_proj/k_proj/v_proj/out_proj` weight tensors to AVQA's
  expected parameter layout. Preserve all other model weights and config.
  Works with `transformers >= 4.40`.

- **vLLM:** Provide an `AttentionBackend` adapter following vLLM's
  `vllm.attention.backends.abstract.AttentionBackend` protocol. Includes
  paged KV layout support, continuous batching hooks, prefix-cache hooks,
  and tensor-parallel sharding where vLLM exposes the corresponding APIs.

- **FlashAttention:** Provide a `FlashAttentionInterop` helper that selects
  between AVQA's online softmax and FlashAttention's `flash_attn_func` /
  `flash_attn_2_cuda` based on configuration. Numerical equivalence within
  `atol=1e-2` for FP16/BF16.

- **xFormers:** Provide a `XFormersInterop` helper that adapts AVQA's
  forward signature to `xformers.ops.memory_efficient_attention`.

All heavy framework deps are optional extras (`pip install avqa[huggingface]`
etc.). Integration modules perform lazy imports and surface a clear
`is_available()` helper.

**Impact:** Until a future spec chapter details the exact mapping, these
integrations follow the upstream framework's documented patterns. They are
covered by integration tests using real models (Hugging Face tiny-BERT) and
real vLLM test fixtures when CUDA is available.

---

## Gap G3 — Profiling Internals

**Spec section affected:** §3.17, §5.15, §13 (missing).

**Gap:** Spec lists profiling capabilities (timing, memory, FLOPs, routing
statistics, refinement statistics, codebook utilization, cache utilization)
but does not specify the exact metrics format, sampling rate, or report
schema.

**Assumption:** Profiler collects the following metrics:

- Per-stage wall-clock time (`torch.cuda.Event` when CUDA, `time.perf_counter`
  otherwise).
- Per-stage peak memory (`torch.cuda.max_memory_allocated` when CUDA, else
  `tracemalloc`).
- Per-stage FLOPs (analytical estimate based on tensor shapes; no external
  FLOPs profiler dependency).
- Routing statistics: selected-parent indices, importance scores,
  dead-codeword counts.
- Refinement statistics: number of refined parents, average refinement
  depth, correction delta norms.
- Codebook utilization: per-codeword assignment count histogram.
- Cache utilization: hit rate, miss rate, eviction count.

Reports are exportable in JSON and human-readable tabular form. Profiler is
opt-in via `with profiler.session(): ...` context manager.

**Impact:** Until future spec chapter defines exact report schema, JSON
schema is versioned (`avqa_profiler_v1`) and stable.

---

## Gap G4 — Visualization Internals

**Spec section affected:** §3.18, §5.16, §13 (missing).

**Gap:** Spec lists visualization capabilities (refinement trees, routing
paths, attention heatmaps, codebook utilization, execution timelines) but
does not specify rendering library, color schemes, or output formats.

**Assumption:** Visualizers are pure Python with optional rendering backends:

- **Refinement tree:** Graphviz DOT output (default) or interactive HTML
  via `vis.js` (optional, requires `vis-network` extra).
- **Routing path:** Matplotlib PNG (optional, requires `matplotlib` extra)
  or ASCII table fallback.
- **Attention heatmap:** Matplotlib PNG or CSV fallback.
- **Codebook utilization:** Matplotlib histogram or CSV fallback.
- **Execution timeline:** Matplotlib timeline or JSON fallback.

Visualizers are entirely decoupled from the algorithm core and can be
disabled by simply not calling them.

**Impact:** Visualization rendering quality is best-effort under the
assumption above until a future spec chapter defines exact UX.

---

## Gap G5 — Serialization Versioning Schema

**Spec section affected:** §3.20, §5.12, §14 (missing).

**Gap:** Spec requires versioned serialization (§3.20.2) but does not
specify the schema.

**Assumption:** Every serialized artifact carries:

```json
{
  "avqa_version": "X.Y.Z",
  "schema_version": "v1",
  "artifact_type": "AVQConfig | HierarchicalCodebook | Router | KVCache",
  "data": {...}
}
```

The `schema_version` is independent of the library version. When schema
changes, a migration function (`migrate_v1_to_v2`) is provided. Old artifacts
are loadable across minor releases.

**Impact:** Future minor versions preserve schema stability. Major versions
MAY bump the schema.

---

## Gap G6 — Numerical Tolerances

**Spec section affected:** §3.16, §3.24, §10.20 (referenced), §14 (missing).

**Gap:** Spec requires numerical equivalence within documented tolerances
but does not specify the tolerance values.

**Assumption:** Default tolerances:

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 2e-2 | 2e-2 |
| float64 | 1e-8 | 1e-8 |

Tolerances are configurable via `AVQConfig.tolerance_atol` and
`AVQConfig.tolerance_rtol`. Tests use the default unless explicitly
overridden.

**Impact:** Future spec chapter may redefine; until then, defaults above.

---

## Gap G7 — Distributed Execution

**Spec section affected:** §6.16, §3.17, §10.21 (referenced), §15 (missing).

**Gap:** Spec mentions tensor/pipeline/sequence/context parallelism but does
not specify implementation.

**Assumption:** Distributed execution is **out of scope** for the initial
v1.0 release. The data model (`TensorContract`) is designed to support
future distributed implementations but no distributed runtime ships with
v1.0. The `backend.distributed` package is reserved but not implemented.

**Impact:** v1.0 is single-process only. Multi-GPU is a v1.x future
enhancement.

---

## Gap G8 — Speculative Decoding Integration

**Spec section affected:** §3.15 (vLLM integration).

**Gap:** Spec mentions speculative decoding as a "where compatible"
requirement.

**Assumption:** Speculative decoding integration follows vLLM's
`SpeculativeConfig` interface when vLLM is installed. If vLLM's API changes
between versions, the integration adapts to the latest installed version.
This is not tested directly; only the integration's `is_available()` and
import contracts are tested.

**Impact:** Speculative decoding works through vLLM upstream APIs; AVQA
itself does not implement speculative decoding logic.

---

## Gap G9 — DEAD-code Resampling

**Spec section affected:** §8.11.

**Gap:** Spec marks enhanced dead-code handling as optional.

**Assumption:** Default implementation uses simple EMA approach. An
optional `DeadCodeResampler` strategy MAY be registered but is not provided
by default. Documented in registry hooks for community extension.

**Impact:** No shipped dead-code resampler; registry hook available.

---

## Gap G10 — FP8 / INT8 Support

**Spec section affected:** §6.9.

**Gap:** FP8/INT8 marked as optional.

**Assumption:** Default supported dtypes are FP32/FP16/BF16. FP8/INT8
quantization is **out of scope** for v1.0. Reserved via type aliases in
`avqa.data.dtypes.SUPPORTED_DTYPES_EXTENDED`.

**Impact:** v1.0 supports the three required dtypes only.

---

## Gap G11 — FAISS Integration

**Spec section affected:** §3.7.

**Gap:** FAISS marked as optional.

**Assumption:** Default quantizer uses pure-PyTorch Euclidean distance.
FAISS-backed quantizer is a registry-extensible implementation; not
provided by default. The `quantizer.faiss` module is reserved.

**Impact:** v1.0 ships pure-PyTorch quantizer; FAISS is a community
extension point.

---

## Gap G12 — Empty Codeword Treatment Beyond §7.15

**Spec section affected:** §7.15, §9.12.

**Gap:** Spec describes empty codeword handling for parent logits but does
not exhaustively enumerate child-side handling.

**Assumption:** For child codewords: contribute 0 to value aggregates,
contribute 0 to count, exclude from running max, exclude from running
denominator. Children are skipped entirely by the importance and selection
stages if their parent has zero assignment count.

**Impact:** Consistent behavior with §7.15; verified by invariant tests.

---

## Gap G13 — Causal Mask Behavior Under Refinement

**Spec section affected:** §9.15 (mentioned but not detailed).

**Gap:** Spec requires causal masking but does not describe how it interacts
with adaptive refinement.

**Assumption:** Causal mask is applied at the logits level. Concretely:

- For each query tile, parent logits are computed using the same mask as
  standard attention.
- Child logits are computed under the same mask.
- Masked positions contribute `-inf` logits and therefore `0` to the
  softmax. They do not affect selection or importance, since importance is
  a function of attention already applied to unmasked positions.

**Impact:** Refinement respects causal structure; selection happens among
causally-visible parents/children only.

---

## Gap G14 — Tie-Breaking in TopP

**Spec section affected:** §9.6, §10.9.

**Gap:** Spec requires deterministic tie handling but does not specify the
algorithm.

**Assumption:** Tie-breaking sorts by `(importance desc, parent_index asc)`
so that ties resolve to lower-indexed parents. Stable across runs and
backends.

**Impact:** Deterministic selection with verified equality under seeded
runs.

---

## Summary

| Gap | Status |
|-----|--------|
| G1  Triton internals | Assumption documented |
| G2  Framework internals | Assumption documented |
| G3  Profiling internals | Assumption documented |
| G4  Visualization internals | Assumption documented |
| G5  Serialization schema | Assumption documented |
| G6  Numerical tolerances | Assumption documented |
| G7  Distributed execution | Deferred to v1.x |
| G8  Speculative decoding | Implementation follows vLLM upstream |
| G9  Dead-code resampling | Registry hook only |
| G10 FP8/INT8 | Deferred to v1.x |
| G11 FAISS | Registry hook only |
| G12 Empty codeword handling | Assumption documented |
| G13 Causal mask under refinement | Assumption documented |
| G14 Tie-breaking | Assumption documented |

Every assumption is exercised by tests. When future spec chapters are
published, the relevant assumption rows above will be updated to reference
the new spec text, and implementation will be reviewed for compliance.
