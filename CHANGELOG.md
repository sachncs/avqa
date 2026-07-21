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
  ≥85% line coverage on `src/avqa/`.
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

## Unreleased (governance refresh)

### Governance (added 2026-07-16)

- TASK-1.004: Black configuration + CI gate (`ce76d46`).
- Renamed `spec.md` → `SPEC.md` to match governance references (`9d02aa5`).
- Refreshed TODO.md ledger under the v2 numbering scheme (`c8b1f0a`).

### Configuration

- TASK-5.004: `AVQConfig.save_json` / `load_json` round-trip helpers (`442d1d0`).

### Specification

- Chapter 11 (Triton kernels) + Chapter 12 (adapter protocols) appended (`aa34dec`).

### Performance infrastructure

- EXP-0001: CPU AVQA-vs-SDPA baseline reproduction (`74bfd37`).
- Triton kernel package (vq, parent_attention, child_attention, correction) shipped (`bb660fd`).
- TritonBackend wired to SPEC §11 kernels with TorchBackend fallback (`1fede67`).
- GPU-gated Triton equivalence tests under `tests/integration/test_triton_kernels_gpu.py` (`cf48850`).

### Framework adapters

- TASK-12.001: HF weight copy handles biases and Identity projections (`155285a`).
- TASK-12.002: vLLM paged-attention adapter via PagedKVCache (`2131523`).
- TASK-12.003 / TASK-12.004: FlashAttention + xFormers numerical-equivalence tests (`ac6a841`).
- TASK-12.005: end-to-end AVQA + HF replacement integration test (`e7c818d`).

### Benchmarks

- EXP-0002: post-governance CPU baseline reproduction; AVQA at
  seq=1024 dropped from 22.215 ms (EXP-0001) to 19.618 ms (`b3e91a5`).

### Compliance

- SPEC_COMPLIANCE.md refresh — RTM with per-module coverage, v2 ledger, and 14-commit audit trail (`5af9e28`).
- OPTIMIZATIONS.md seed — OPT-0001 tracked as Proposed pending GPU acceptance (`d642e10`).
- PUBLICATION.md readiness score + outstanding-gap list (`f88c78a`).

## Unreleased (BCAR — first algorithmic contribution)

### Algorithmic extension (OPT-0003)

- **BCAR (Bias-Corrected Online Codebook Adaptation)** ships.
  Per-codeword EMA on inference-time K/V assignments, with the
  parent-child mean constraint (SPEC §7.9) preserved exactly via a
  post-step reprojection. Configured via
  `CodebookConfig(bcar_enabled=True, bcar_decay=0.99)`.
- SPEC §13 documents the algorithm, math, and acceptance plan.
- Four unit tests in `tests/unit/test_online_adaptation.py`
  validate mean preservation, empty-cell no-explode, decay API
  guard, and synthetic-stream convergence.
- EXP-0004 (CPU) closes 60.7 % of the static-to-oracle VQ-loss gap
  in 1024 streaming updates; ablation over decay recorded.
- Publication candidate `PUB-0001` staged; multi-seed statistical
  validation is the next gate on the CUDA-matrix runner.

## Unreleased (HVAQ — algorithmic contribution to the attention mechanism)

### Algorithmic contribution (OPT-0005)

- **HVAQ (Hopfield-VQ-Attention)** ships. This is the project's
  first **algorithmic contribution to the attention kernel itself**,
  not just an engineering wrapper. HVAQ generalises the paper's
  fixed-temperature softmax with a per-query temperature ``β_q``
  derived from the router's top-P attention-mass entropy.
- SPEC §16 documents the algorithm, the two schedules (HVAQ-ENT,
  HVAQ-LIN), and the two theorems (Equivalence and β-monotonicity).
- ``src/avqa/hopfield.py``: ``paper_beta``, ``per_query_beta``,
  ``hopfield_logits``, ``validate_adaptive``.
- ``src/avqa/config.py``: ``HopfieldConfig`` dataclass and
  ``BackendConfig.hopfield`` master switch.
- ``src/avqa/attention_module.py``: HVAQ block in ``forward_impl``,
  gated on ``backend.hopfield and hopfield.adaptive != "none"``.
- ``tests/unit/test_hopfield.py``: 31 SPEC §16 unit functions
  (33 collected with parametrized variants) covering the
  temperature schedules, HopfieldConfig validation, Theorem 16.1
  paper equivalence, learnable parameter gradient flow, and
  Theorem 16.2 downstream-consumer invariants
  (top-K index invariance under positive β).
- EXP-0006 captures the latency curve and output difference on a
  small synthetic task. The integration is gated off by default so
  every prior test continues to pass.

### Multi-pass refinement (disjoint-set re-routing)

- ``MultiPassRefiner`` now implements **disjoint-set multi-pass
  correction**: each pass corrects a different subset of parents.
  After pass *k*, already-refined parents are masked out, the router
  re-selects the top-P parents from the remaining set with budget
  decay, and fresh child logits are recomputed.  This guarantees
  converging residual norms rather than the divergent
  ``state_0 + k*(child - parent)`` of the naive approach.
- ``src/avqa/multipass.py``: ``query`` and ``child_keys`` optional
  params enable re-routing; falls back to single-pass with a
  warning when not provided.
- ``src/avqa/attention_module.py``: ``_refine_and_output`` now
  invokes ``MultiPassRefiner`` when ``refinement.passes > 1``.
- ``tests/unit/test_multipass.py``: 16 tests covering budget decay,
  re-routing, residual convergence, and fallback behaviour.

### torch.compile numerical equivalence

- ``tests/unit/test_attention_compile.py``: new
  ``TestCompileNumericalEquivalence`` class runs the compiled and
  eager forwards on CPU with shared weights and asserts
  ``torch.allclose`` within tolerance.  Dynamo tracing failures
  are caught and documented as skips (GPU runner provides the
  authoritative gate).

### Code quality

- Refactored ``AVQAttention.forward_impl`` into 10+ pipeline stage
  helpers (``_validate_inputs``, ``_sync_codebook_device``,
  ``_resolve_kv_cache``, ``_resolve_mask``, ``_run_naive``,
  ``_run_vq_precompute``, ``_compute_parent_logits``,
  ``_apply_hopfield``, ``_compute_online_softmax``,
  ``_compute_routing``, ``_compute_child_logits``,
  ``_refine_and_output``).
- Learnable HVAQ parameters: ``_parent_beta`` (``nn.Parameter [1,1,1,M0]``)
  and ``_alpha`` (``nn.Parameter [H]``) with gradient flow through
  ``hopfield_logits``.  Six new tests in ``tests/unit/test_hopfield.py``.
- Removed dead code: ``_STRATEGIES``, ``render_to_json``, unused
  ``json`` import, redundant asserts in ``cache.py``.
- Renamed public classes: ``HFAttentionWrapper``, ``VLLMSelector``.
- Replaced broad ``except Exception`` with specific exceptions
  (``OSError``, ``RuntimeError``) plus ``_logger.debug`` in
  ``backend.py``.
- Added Google-style docstrings to ``logging.py`` public functions.
- HF adapter: debug-level log when ``head_mask`` or ``past_key_value``
  are provided but not supported by AVQA.
- Deleted dead ``@parametrize([])`` test in
  ``tests/unit/test_attention_module.py``.

### CI/CD

- ``.github/workflows/ci.yml``: added build (``python -m build``)
  and ``pip-audit`` jobs.
- ``.github/workflows/release.yml``: new PyPI release workflow
  triggered on tag push.
- ``.pre-commit-config.yaml``: updated ruff to v0.11.6, mypy to
  v1.15.0, removed black hook.
- ``pyproject.toml``: ``dynamic = ["version"]``, removed
  ``[tool.black]`` section.
- ``Makefile``: removed black targets, fixed coverage help to
  ``>=85%``.

### Benchmarks

- EXP-0006 raw + summary: paper single-pass + hvaq entropy +
  hvaq linear latency comparison; ``benchmarks/raw/EXP-0006/`` is
  the canonical raw archive.

### Compliance

- REQ-3.50.004 (HVAQ integration) tracked in SPEC_COMPLIANCE.md.

### Publications

- PUB-0002 (HVAQ) candidate staged; multi-seed + downstream-quality
  validation is the next gate.
