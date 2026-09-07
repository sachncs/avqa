# Changelog

All notable changes to AVQA are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- **Visualizer import** (`from avqa import Visualizer`): the class was
  listed in `__all__` but never imported in `__init__.py`. Now exposed.
- **Integration tests collection**: `tests/integration/test_avqa_end_to_end.py`
  imported a non-existent `make_hf_attention_replacement` after the
  integrations package was removed; the file now exercises the still-
  shipped `AVQAttention` native path.
- **`WeightedMerge` double-counted `parent_probs`** — `parent_value` in
  the `MergeInputs` contract is already attention-weighted by the
  pipeline, so re-multiplying by `parent_probs` produced `A²·V̄`
  instead of `A·V̄`. Now matches `ProbabilityMerge` / `LogitMerge`.
- **`ProfilerReport.to_dict()` dropped `total_flops`** — silently omitted
  from JSON export despite being spec §3.17 M7. Now emitted.
- **NaN in `online_softmax_attention` and `apply_hopfield` on a fully-
  masked tile**: `exp(-inf - -inf) = NaN` poisoned downstream state. Now
  guarded with `nan_to_num`-style NaN-safe exp.
- **Causal mask was silently a no-op in the AVQ path** — `mask.any(dim=-1)`
  collapsed `[T_q, T_k] → [T_q, 1]`, so causal masks never excluded
  any codeword. Fixed: per-(b,h,q,p) **codeword mask** derived from
  the quantizer's `parent_assignments`. `resolve_mask` now also
  honours `[T_q, T_k_total]` (KV-cache aware) instead of just `[T, T]`.
- **`refine(..., state=...)` returned a rescaled-incorrectly state on
  multi-pass**: `OnlineSoftmaxState.replace()` recomputed its own
  max instead of using the caller's common scale, so the parent/
  child contributions were rescaled by an off-by-one factor once the
  state already had a higher max than the correction tile. Added an
  optional `m_anchor=...` argument so the caller can pass the common
  scale; `vectorized_correction` now uses it.
- **`ThresholdRouter` violated its own threshold** when fewer entries
  met the threshold than the requested budget — silently returned
  below-threshold indices. Now raises `RoutingError`.
- **`ThresholdRouter` vs `TopPRouter`**: tie-breaks now use stable
  sorting consistent with `TopPRouter`.
- **`Router.create("budget")` raised** despite the README, CHANGELOG,
  and spec advertising a "Budget" router — added `BudgetRouter`
  (strict-budget selector) and aliased `"budget"` in the factory.
- **Exception hierarchy**: ~30 sites raised plain `ValueError` instead
  of the documented `AVQAError` subclasses (`RoutingError`,
  `ConfigurationError`, `CodebookError`, `BackendError`, `ShapeError`,
  `MergeError`). Now properly typed.
- **`AVQConfig.from_dict()` skipped the outer `__post_init__`**
  validation — hand-crafted dicts with invalid `dropout`,
  `tolerance_atol`, or `head_dim` were silently accepted. Now
  `from_dict` calls `cls.__init__` which triggers `__post_init__`.
- **`AVQConfig.save_json()` only caught `OSError`, not `TypeError`** —
  `json.dumps` can raise `TypeError` on non-serializable values. Now
  both are converted into `ConfigurationError`.
- **`commitment_loss()` raised `RuntimeError`** before any forward
  pass — now raises `NotInitializedError` (AVQAError subclass).
- **`InMemoryKVCache` eviction produced a non-contiguous view** that
  could trigger hidden copies in downstream reshape paths. Now
  `.contiguous()` is called.
- **`AdaptiveScheduler` collapsed per-(B, H) entropy to a single
  scalar** (routing-decoupled, kept as noted; deferred for next minor).
- **PagedKVCache.full**: now `NotInitializedError` instead of
  `RuntimeError`.
- **`codebook.children` empty-cell EMA was overwriting `parents`** with
  a per-batch mean that didn't preserve the §7.9 invariant. Removed
  dead computation; parents are now derived only from
  `mean(children)` per step.
- **BCAR docstring `v_j` for children** was misleading — actual
  algorithm uses `k_j` (key space); reconciled with SPEC §13.2 and
  updated docstring.
- **HVAQ-ENT was using full-distribution entropy**, not the
  top-P entropy mandated by SPEC §16.2. Pipeline reordered so the
  router's selection drives the entropy computation; if no router
  selection is available, the previous behaviour is preserved.
- **`functional.attention` did not support `kv_cache`** for
  incremental decoding through the functional entry point. Now
  accepts `kv_cache=None` and forwards to the underlying module.

### Added

- **Direct `OnlineSoftmaxState.replace` tests** (4 cases): algebraic
  identity (no-op when removed==added), empty state, m_anchor path,
  difference from default-path rescale.
- **Direct `BudgetRouter` tests** (4 cases): exact-budget, tie-break
  by lower index, invalid budget, over-budget rejection.
- **`ThresholdRouter` over-budget raises** (2 cases).
- **AVQAError-subclass tests** for cache, scheduler, backend, seed,
  validation, hopfield, multipass, online_adaptation, refinement.
- **BCAR end-to-end tests** (`TestBCAREndToEnd`): mutation when
  enabled, no-mutation when disabled; SPEC §7.9 invariant preserved.
- **Streaming aggregate pinned to counts only** (the only
  VQ-intrinsic output). ponytail note records the keys-vs-values
  semantic mismatch.
- **per-comma type-coverage matrix tightened**: 458 tests collected
  total (was 442); coverage gate 90% (was 85%).

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
- TorchBackend (reference backend implementation).
- BCAR (Bias-Corrected Online Codebook Adaptation) — per-codeword
  inference-time EMA, configured via
  `CodebookConfig(bcar_enabled=True, bcar_decay=0.99)`.
- HVAQ (Hopfield-VQ-Attention) — per-query temperature schedules
  derived from router top-P entropy.
- Multi-pass refinement with disjoint-set re-routing (converging
  residuals, budget decay).
- Profiler with stage timers, memory tracking, JSON export.
- JSON-only Visualizer (TreeNode, HeatmapData, TimelineEvent).
- torch.compile opt-in for reduced Python overhead.
- 446 unit + reference + benchmark tests with ≥90% coverage on
  `src/avqa/`.
- Hand-computed reference tests and invariant property tests.
- Strict typing: mypy --strict clean across `src/avqa/`.
- Zero-warning lint: ruff clean across `src/avqa/` and `tests/`.
- Naming convention: no leading-underscore prefixes anywhere in
  `src/avqa/`, `tests/`, `benchmarks/`, `examples/`, `scripts/`.
  Class fields, methods, parameters, helper functions, helper
  variables, and dataclass field names are all public.

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
- **L1**: Package docstring example now uses the correct nested-config
  API and rank-3 tensor layout.
- **L2**: `commitment_loss()` docstring no longer claims to return 0.0
  before a forward pass (it raises `RuntimeError`).
- **L4**: Renamed misleading `Q` tuple variable to direct unpacking.
- **L5**: Removed dead `child_assign = torch.empty(...)` allocation.
- **L6**: Removed unused `Profiler.active` field.
- **L7**: `CacheEntry.positions` docstring documents paged-cache usage.

### Removed

- **`src/avqa/triton/` (Triton kernel package)**: removed because the
  Python wheels for triton are not published to PyPI for
  `platform_system != Linux`, the kernel module added significant
  build-toolchain coupling, and the only consumer (`TritonBackend`) had
  a single user. A future release can re-introduce a vendor kernel
  package as a separate distribution extra when a CUDA CI runner is
  available.
- **`TritonBackend`**: removed along with the triton kernels. The
  `Backend` factory now returns only `TorchBackend`.
- **`src/avqa/integrations/{flashattn,hf,vllm,xformers}.py`**: the
  upstream packages (`flash-attn`, `vllm`, `xformers`) have
  version-pinned dependencies that conflict with the AVQA core
  toolchain on non-CUDA hosts (notably the `clang -fopenmp` build
  flag fails on the macOS arm64 default). The `integrations` package
  is now a placeholder for users to re-introduce their own adapters.
- **`tests/integration/test_{triton_kernels,attention_interops,integrations}.py`**:
  removed along with the underlying modules.
- **`tests/unit/test_triton_kernels.py`**, **`tests/unit/test_vllm_adapter.py`**:
  removed.

### Notes

- Optional dependencies (`triton`, `flash-attn`, `xformers`, `vllm`)
  are no longer in `pyproject.toml`. The `dev` extra provides the test
  toolchain; the `viz` extra remains for matplotlib + graphviz.
- The integrations directory is now a single `__init__.py` placeholder.
- Distributed execution is deferred to a future release.
- This is an independent, community-driven implementation of
  AVQ-Attention. See `README.md` for the disclaimer.

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
- 33 SPEC §16 unit tests covering the temperature schedules,
  HopfieldConfig validation, Theorem 16.1 paper equivalence,
  learnable parameter gradient flow, and Theorem 16.2
  downstream-consumer invariants (top-K index invariance under
  positive β).

### Multi-pass refinement (disjoint-set re-routing)

- ``MultiPassRefiner`` now implements **disjoint-set multi-pass
  correction**: each pass corrects a different subset of parents.
  After pass *k*, already-refined parents are masked out, the router
  re-selects the top-P parents from the remaining set with budget
  decay, and fresh child logits are recomputed.  This guarantees
  converging residual norms rather than the divergent
  ``state_0 + k*(child - parent)`` of the naive approach.

### Code quality

- Refactored ``AVQAttention.forward_impl`` into 10+ pipeline stage
  helpers.
- Learnable HVAQ parameters with gradient flow through
  ``hopfield_logits``.  Six new tests in ``tests/unit/test_hopfield.py``.
- Removed dead code: ``_STRATEGIES``, ``render_to_json``, unused
  ``json`` import, redundant asserts in ``cache.py``.
- Renamed public classes: ``HFAttentionWrapper``, ``VLLMSelector``.
- Replaced broad ``except Exception`` with specific exceptions
  (``OSError``, ``RuntimeError``) plus ``_logger.debug`` in
  ``backend.py``.
- Added Google-style docstrings to ``logging.py`` public functions.
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
  ``>=90%``.

### Naming convention (enforced project-wide)

- No leading-underscore identifiers anywhere in the project
  (classes, methods, parameters, helper functions, dataclass fields,
  helper variables, properties). Class fields are public
  (`self.code_key` not `self._code_key`).
- All previously-underscored functions/methods/parameters renamed
  (e.g. `set_configured` instead of `set_configured` is already
  public; `require_positive` instead of `_require_positive`; etc.).

### Suppression removal (type-safety hardening)

- No `# type: ignore` comments remain anywhere in `src/`, `tests/`,
  `benchmarks/`, `examples/`, `scripts/`. Every previous suppression
  was replaced with a real fix: triton stubs at `stubs/triton/`
  (since removed with the package), explicit `Protocol` types for
  the kernel module surface, real try/except ImportError blocks
  for optional dependencies, and a proper `find_spec()` pattern.
- No `# noqa: ...` lint suppressions remain.
- No `cast()` calls remain.
- `Any` is no longer used as a substitute for an unknown type.
- ``object`` is no longer used as a substitute for an unknown type.
- `mypy --strict` passes on `src/avqa/` with zero errors.
