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
- **Drop-in `AVQAttention` nn.Module**; framework adapters for
  Hugging Face, vLLM, FlashAttention, and xFormers are user-supplied
  (see `src/avqa/integrations/README.md` for a scaffold).

### Performance Notes

- The PyTorch reference is the only execution path shipped; a Triton
  kernel is planned for v0.2.0.
- On macOS (MPS) and CPU, expect throughput comparable to PyTorch SDPA
  plus the overhead of VQ precompute; adaptive refinement pays off for
  long sequences (≥1k tokens).

### Compatibility

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Optional: `matplotlib`, `graphviz` (installed via `pip install -e ".[viz]"`).
  Framework integrations are user-supplied — see
  `src/avqa/integrations/README.md`.

### Known Limitations

- Spec chapters 11-15 are not implemented in detail; the public API
  surfaces are honored, but kernel-internals, profiling-internals,
  visualization-rendering, and serialization-schema internals are
  left to vendor libraries.
- Speculative decoding, FAISS, FP8/INT8 quantization, and per-batch
  dead-code resampling are deferred to future releases.
- Hugging Face, vLLM, FlashAttention, and xFormers adapters were
  removed in v0.1.0 (see CHANGELOG); ship them in sibling packages if
  you need them.

### Acknowledgements

This implementation is independent and is not affiliated with the
authors of the AVQ-Attention paper. See `README.md` for the full
disclaimer.