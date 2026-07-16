# Release Notes

## v0.1.0 (Initial Public Release)

This release marks the first public cut of AVQA, an independent
community implementation of Adaptive Vector Quantized Attention
(AVQ-Attention). The release targets **inference and training**
support on CPU/MPS/CUDA backends via PyTorch, with optional Triton
acceleration gated on CUDA availability.

### Highlights

- **Faithful reference implementation** of the AVQ-Attention algorithm
  (spec §3, §7-§10) in pure PyTorch.
- **Hierarchical codebook** with parent-child mean constraint
  (§7.9, §8.3); serialization round-trip via `state_dict` /
  `load_state_dict`.
- **Two-stage Euclidean VQ** with fused value aggregation (§8.5-§8.7).
- **Online-softmax (FlashAttention-style) tile-based attention** as a
  numerically-equivalent alternative to the naive O(N^2) path.
- **Adaptive refinement** with importance-based top-P selection,
  parent logit recovery, and online-softmax state correction (§7.12,
  §7.13, §9.7).
- **Hugging Face Transformers integration**: `replace_attention()`
  swaps attention modules in-place while preserving pretrained
  weights elsewhere.
- **vLLM / FlashAttention / xFormers** interop helpers, each gated by
  an `is_*_available()` runtime check.

### Performance Notes

- Triton kernels are scaffolded but delegate to the PyTorch reference
  when CUDA + Triton are not both available. The first Triton-native
  release will follow in v0.2.0.
- On macOS (MPS) and CPU, expect throughput comparable to PyTorch SDPA
  plus the overhead of VQ precompute; adaptive refinement pays off for
  long sequences (≥1k tokens).

### Compatibility

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Optional: `transformers ≥ 4.40`, `triton ≥ 2.2`, `flash-attn ≥ 2.5`,
  `xformers ≥ 0.27`, `vllm ≥ 0.5`.

### Known Limitations

- Spec chapters 11-15 are not implemented in detail; the public API
  surfaces are honored, but kernel-internals, profiling-internals,
  visualization-rendering, and serialization-schema internals are
  left to vendor libraries (or future community extensions via the
  `avqa.registry` mechanism).
- Speculative decoding, FAISS, FP8/INT8 quantization, and per-batch
  dead-code resampling are documented in `docs/spec_gaps.md` as
  future work.

### Acknowledgements

This implementation is independent and is not affiliated with the
authors of the AVQ-Attention paper. See `README.md` for the full
disclaimer.