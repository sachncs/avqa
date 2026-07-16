# Changelog

All notable changes to AVQA are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial public release.
- Reference PyTorch implementation of Adaptive Vector Quantized
  Attention (AVQ-Attention) covering spec.md Chapters 2-10.
- Hierarchical codebook with mean-constrained parent-child structure
  (spec §7.9, §8.3).
- Two-stage hierarchical Euclidean vector quantizer with fused value
  aggregation (spec §8.4-§8.7).
- Online-softmax (FlashAttention-style) tile-based attention in pure
  PyTorch.
- Adaptive refinement: importance-based top-P parent selection,
  parent logit recovery from children, correcting attention with
  online-softmax state (spec §9.7, §7.12, §7.13).
- Probability / Weighted / Logit / Normalized merge strategies
  (spec §3.11).
- TopP / Threshold / Budget routers; Default + Adaptive schedulers.
- InMemory and Paged KV caches.
- TorchBackend (reference) + TritonBackend (CUDA-gated delegation).
- Profiler with stage timers, memory tracking, JSON export.
- JSON-only Visualizer (TreeNode, HeatmapData, TimelineEvent).
- Hugging Face Transformers integration: detect_compatible and
  replace_attention with HF-compatible wrapper.
- vLLM / FlashAttention / xFormers interop helpers (gated by
  availability checks).
- pytest-benchmark suite sweeping sequence lengths 64/128/256.
- 386 unit + integration + benchmark tests; ≥90% line coverage on
  src/avqa/.

### Notes

- Spec chapters 11-15 (Triton kernel internals, profiling internals,
  visualization internals, serialization schema, packaging) are out
  of scope of this release. The implementation honors the public API
  contracts these chapters imply; the internals are delegated to
  vendor libraries (PyTorch, Hugging Face, etc.) or to community
  extensions via the documented registries.
- This is an independent, community-driven implementation of AVQ-Attention.
  See `README.md` for the disclaimer.