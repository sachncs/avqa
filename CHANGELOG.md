# Changelog

All notable changes to AVQA are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2025-07-14

### Added

- Reference PyTorch implementation of Adaptive Vector Quantized
  Attention (AVQ-Attention).
- Hierarchical codebook with mean-constrained parent-child structure
  (spec §7.9, §8.3).
- Two-stage hierarchical Euclidean vector quantizer with fused value
  aggregation (spec §8.4–§8.7).
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
- Hugging Face Transformers integration: `detect_compatible` and
  `replace_attention` with HF-compatible wrapper that preserves
  pretrained weights.
- vLLM / FlashAttention / xFormers interop helpers (gated by
  availability checks); `AVQvLLMBackend` with real forward path.
- pytest-benchmark suite sweeping sequence lengths 64–2048 with
  output quality and SDPA numerical comparison tests.
- 429 unit + integration + reference + benchmark tests;
  ≥90% line coverage on `src/avqa/`.
- Hand-computed reference tests and invariant property tests
  (conservation, hierarchy, attention, count, assignment).
- Commitment (encoding) loss in `AVQAttention` (spec §8.9).
- Input validation in the forward pass: rejects non-rank-3 queries,
  mismatched key/value shapes (spec §6.12).
- Selective child attention: child logits computed only for selected
  parents (spec §9.8).
- `max_depth` config field on `CodebookConfig` (depth > 2 raises
  `ConfigurationError`; arbitrary depth planned for v0.2.0, spec §2.7).
- `setup.sh` (editable install + CUDA-only deps) and `cleanup.sh`
  (remove build artifacts, caches, bytecode).

### Fixed

- VQ attention denominator now includes the assignment count `n_a`
  weight from the state, correcting the unweighted denominator (spec §7.7).
- Correction parent value now receives raw aggregates `V̄_p` instead of
  weighted `A_p·V̄_p`, fixing the correction formula (spec §7.13).
- Output now uses the state reduction `Σ(A·V)/Σ(A)` from the full
  online-softmax state, not the child-only `merge_value` (spec §7.7, §7.14).
- `AdaptiveRefinement` class is now importable and fully functional.
- **M1**: Refinement output (the corrected state) is now actually used
  for the final attention result; previously the discarded correction
  meant adaptive refinement had no effect on output.
- **M2**: Correcting attention now correctly subtracts the parent
  contribution and adds the children via `OnlineSoftmaxState.replace()`
  with a numerically stable three-tile max; uses `parent_counts` and
  `child_counts` for empty-codeword scaling (spec §7.13, §9.12).
- **M3**: The supplied mask (e.g., causal) is now applied to parent
  logits in the AVQ path; previously the AVQ branch created an
  all-ones codeword mask.
- **M4**: KV cache is now looked up and concatenated with current K/V
  before running attention (previously only the current K/V participated);
  `InMemoryKVCache.state_dict()` now serializes actual tensor data.
- **M5**: Forward pass now validates dtype, embed_dim, and device match;
  codebook is moved to input device AND dtype.
- **M6**: Router, scheduler, and merge strategy are now selected from
  config; `to_dict()` recursively serializes nested sub-config dataclasses
  (JSON-safe); `from_dict()` rejects unknown fields.
- **M7**: Honest documentation of `TritonBackend` (fallback) and
  `AVQvLLMBackend` (paged attention deferred); `ProfilerReport.total_flops`
  added (spec §3.17).
- **L1**: Package docstring example now uses the correct nested-config
  API and rank-3 tensor layout.
- **L2**: `commitment_loss()` docstring no longer claims to return 0.0
  before a forward pass (it raises `RuntimeError`).
- **L3**: Removed duplicate `# noqa: PLR0915` directive.
- **L4**: Renamed misleading `Q` tuple variable to direct unpacking.
- **L5**: Removed dead `child_assign = torch.empty(...)` allocation.
- **L6**: Removed unused `Profiler.active` field.
- **L7**: `CacheEntry.positions` docstring documents paged-cache usage.
- **Naming**: All semi-private (`_`-prefixed) names have been renamed to
  public names (e.g., `_set_configured` → `set_configured`, `_require_positive`
  → `require_positive`, `_copy_hf_weights` → `copy_hf_weights`, etc.).
  No more `self._x` attributes either (`cache_key`/`cache_value`,
  `pages`, `items`, `last_result`, `last_keys`, `last_parent_assignments`,
  `active`, `session_start`, `module`, `num_kv_heads`, `head_size`).

### Notes

- Spec chapters 11–15 (Triton kernel internals, profiling internals,
  visualization internals, serialization schema, packaging) are out
  of scope. The implementation honors the public API contracts these
  chapters imply; internals are delegated to vendor libraries or
  community extensions via the documented registries.
- Distributed execution is deferred to a future release.
- This is an independent, community-driven implementation of
  AVQ-Attention. See `README.md` for the disclaimer.
