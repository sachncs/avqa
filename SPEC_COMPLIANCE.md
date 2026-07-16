# SPEC_COMPLIANCE.md

> **Project:** AVQA – Adaptive Vector Quantized Attention
>
> **Purpose:** Requirements Traceability Matrix (RTM)
>
> This document is the authoritative mapping between the engineering specification (`SPEC.md`) and the implementation.
>
> Every normative requirement defined in `SPEC.md` SHALL be traceable to:
>
> - implementation
> - tests
> - documentation
> - benchmarks (where applicable)
> - Git history
>
> No requirement is considered implemented unless it appears in this document.

---

# Compliance Philosophy

`SPEC.md` defines **what** the project shall do.
`SPEC_COMPLIANCE.md` proves that it actually does it.
Compliance is based on evidence, not implementation status.
Evidence consists of source code, tests, benchmarks, documentation, and review history.

---

# Live Requirement Index

The requirement index below maps each tracked requirement to its
evidence in the repository. IDs follow the v2 ledger: `REQ-<chapter>.<section>.<n>`.

## Implementation Coverage Summary (commit `e7c818d`)

| Metric | Value |
|--------|-------|
| Source modules (`src/avqa/`) | 24 (incl. `triton/`) |
| Tracked Triton kernel modules | 4 (vq, parent_attention, child_attention, correction) |
| Public pytest tests | 444 |
| Skipped tests (optional deps only) | 9 |
| Tracked SHALL statements fulfilled | (in-progress, see table below) |
| Coverage gate | 90% |

## Tracked Requirements (chapter by chapter)

| ID | Spec ref | Implementation | Tests | Status |
|----|----------|-----------------|-------|--------|
| REQ-3.6.001 | Configuration covers codebook size, branching factor, refinement budget, routing strategy, merge strategy, backend, execution mode, precision, cache | `src/avqa/config.py:53-227` | `tests/unit/test_config.py:208-258` | Verified |
| REQ-3.6.002 | Configuration immutable after construction | `src/avqa/config.py:54-226` (`frozen=True`) | `tests/unit/test_config.py:50-54` | Verified |
| REQ-3.6.003 | Configuration validation | `src/avqa/config.py:30-45, 84-95, 116-148, 165-169` | `tests/unit/test_config.py:33-98` | Verified |
| REQ-3.6.004 | Configuration serialization (`to_dict`/`from_dict`, JSON file I/O) | `src/avqa/config.py:329-461` | `tests/unit/test_config.py:208-258`, `test_save_json_*` (`442d1d0`) | Verified |
| REQ-3.7.001 | Vector quantizer assignment, count accumulation, training | `src/avqa/quantizer.py:34-263` | `tests/unit/test_quantizer.py` | Verified (M4 / `2073498`) |
| REQ-3.8.001 | Hierarchical codebook parent-child mean constraint, EMA | `src/avqa/codebook.py:70-326` | `tests/unit/test_codebook.py` | Verified (M3 / `5e4bea4`) |
| REQ-3.9.001 | Adaptive refinement with selectable policies | `src/avqa/refinement.py:1-320` | `tests/unit/test_refinement.py` | Verified (M7 / `e37d70c`) |
| REQ-3.10.001 | Routing subsystem (TopP, Threshold, Budget) | `src/avqa/routing.py:1-160` | `tests/unit/test_routing.py` | Verified (M5 / `c203ee2`) |
| REQ-3.11.001 | Merge strategies (probability, weighted, logit, normalized) | `src/avqa/merge.py` | `tests/unit/test_merge.py` | Verified (M6 / `de51f00`) |
| REQ-3.12.001 | Multiple execution backends (Torch + Triton, TritonBackend delegates to kernels when available) | `src/avqa/backend.py:32-326`, `src/avqa/triton/` | `tests/unit/test_backend.py`, `tests/unit/test_triton_kernels.py`, `tests/integration/test_triton_kernels_gpu.py` (`bb660fd`, `1fede67`) | Verified (CPU); CUDA gated on hardware |
| REQ-3.13.001 | Autoregressive decoding cache (InMemory + Paged) | `src/avqa/cache.py:1-320` | `tests/unit/test_cache.py`, `tests/unit/test_vllm_adapter.py` (`2131523`) | Verified |
| REQ-3.14.001 | Hugging Face integration | `src/avqa/integrations.py:36-320` | `tests/integration/test_integrations.py`, `tests/integration/test_avqa_end_to_end.py` (`e7c818d`) | Verified |
| REQ-3.15.001 | vLLM paged-attention adapter | `src/avqa/integrations.py:337-475` | `tests/unit/test_vllm_adapter.py` (`2131523`) | Verified |
| REQ-3.16.001 | FlashAttention / xFormers interop | `src/avqa/integrations.py:481-566` | `tests/integration/test_attention_interops_gpu.py` (`ac6a841`) | Verified (CPU); optional deps CUDA-gated |
| REQ-3.17.001 | Profiling (timing, memory, FLOPs, routing statistics) | `src/avqa/profiling.py` | `tests/unit/test_profiling_visualization.py` (M12) | Verified |
| REQ-3.18.001 | Visualization hooks (refinement tree, heatmap, timeline) | `src/avqa/visualization.py` | `tests/unit/test_profiling_visualization.py` (M12) | Verified |
| REQ-3.19.001 | Benchmark suite (AVQA vs SDPA reproduction) | `benchmarks/repro_cpu.py` (`74bfd37`, `b3e91a5`) | `tests/performance/test_benchmarks.py` | Verified |
| REQ-3.20.001 | Serialization of configurations and codebooks | `src/avqa/config.py:329-461`, `src/avqa/codebook.py:311-326` | round-trip tests | Verified |
| REQ-3.21.001 | Extension mechanism (registry, factories) | `src/avqa/registry.py`, `src/avqa/backend.py:328-355` | `tests/unit/test_registry.py` | Verified |
| REQ-3.22.001 | Custom exception hierarchy | `src/avqa/exceptions.py` | `tests/unit/test_exceptions.py` | Verified |
| REQ-3.23.001 | Logging via stdlib `logging` | `src/avqa/logging.py` | `tests/unit/test_logging.py` | Verified |
| REQ-3.24.001 | Public-API testing (unit + regression + edge cases + serialization + determinism) | every `tests/unit/*.py` | `PYTHONPATH=src pytest tests/unit` | Verified |
| REQ-3.25.001 | Functional acceptance (reference implementation, public APIs, tests, configuration) | full repo | All tests | Verified |
| REQ-3.50.001 | Online Codebook Adaptation (SPEC §13) — `online_codebook_adaptation` updates parents/children via EMA; preserves SPEC §7.9 mean invariant at every step | `src/avqa/online_adaptation.py` | `tests/unit/test_online_adaptation.py` (4 cases) | Verified (CPU); statistical validation on GPU runner pending |

### Per-Module Coverage

| Module | Coverage | Source |
|--------|---------:|--------|
| `src/avqa/attention.py` | 100 % | `de51f00` |
| `src/avqa/attention_module.py` | 100 % | `bce845f` |
| `src/avqa/backend.py` | 90 % | `1fede67` (+Triton wiring) |
| `src/avqa/cache.py` | 92 % | `3047d60` |
| `src/avqa/codebook.py` | 97 % | `5e4bea4` |
| `src/avqa/config.py` | 95 % | `ab35160` (+`442d1d0` JSON I/O) |
| `src/avqa/data.py` | 100 % | `4a3263b` |
| `src/avqa/exceptions.py` | 100 % | `cff1f21` |
| `src/avqa/functional.py` | 100 % | `9cb1f53` |
| `src/avqa/integrations.py` | 80 % | `0e0d397` + `155285a` + `2131523` |
| `src/avqa/logging.py` | 100 % | `2a12338` |
| `src/avqa/merge.py` | 100 % | `de51f00` |
| `src/avqa/profiling.py` | 87 % | `6a5b9d3` |
| `src/avqa/quantizer.py` | 95 % | `2073498` |
| `src/avqa/refinement.py` | 87 % | `e37d70c` |
| `src/avqa/registry.py` | 100 % | `e2ad754` |
| `src/avqa/routing.py` | 97 % | `c203ee2` |
| `src/avqa/scheduler.py` | 100 % | `e8b0112` |
| `src/avqa/utils/numerics.py` | 100 % | `e2c82d3` |
| `src/avqa/utils/seed.py` | 94 % | `b0ba7bb` |
| `src/avqa/utils/validation.py` | 95 % | `46eb394` |
| `src/avqa/visualization.py` | 96 % | `6a5b9d3` |
| `src/avqa/triton/` (Triton kernels) | CPU: 100 % of imports; CUDA: skipped | `bb660fd` |

Coverage gate: 90 % overall (matches `.github/workflows/ci.yml`).

## Audit Trail (v2 ledger commits)

| Date | Commit | Author | Reason |
|------|--------|--------|--------|
| 2026-07-16 | `ce76d46` | opencode | TASK-1.004: Black config + CI gate |
| 2026-07-16 | `9d02aa5` | opencode | spec.md → SPEC.md rename |
| 2026-07-16 | `c8b1f0a` | opencode | TODO ledger refresh v2 |
| 2026-07-16 | `442d1d0` | opencode | TASK-5.004: AVQConfig JSON I/O |
| 2026-07-16 | `74bfd37` | opencode | EXP-0001 CPU baseline |
| 2026-07-16 | `aa34dec` | opencode | SPEC Chapter 11 + 12 added |
| 2026-07-16 | `bb660fd` | opencode | Triton kernel package shipped |
| 2026-07-16 | `1fede67` | opencode | TritonBackend wired to kernels |
| 2026-07-16 | `cf48850` | opencode | GPU-gated Triton equivalence tests |
| 2026-07-16 | `155285a` | opencode | TASK-12.001 HF bias fix |
| 2026-07-16 | `2131523` | opencode | TASK-12.002 vLLM paged adapter |
| 2026-07-16 | `ac6a841` | opencode | TASK-12.003/12.004 FA/xFormers tests |
| 2026-07-16 | `e7c818d` | opencode | TASK-12.005 end-to-end tests |
| 2026-07-16 | `b3e91a5` | opencode | EXP-0002 CPU baseline post-governance |

## Continuous Compliance Goals

- TritonKernelRunner: deploy `tests/integration/test_triton_kernels_gpu.py` on a CUDA matrix (`runs-on: [self-hosted, gpu]`) so the FP32/BF16 numerical-equivalence assertions confirm SPEC §11.9 every release.
- Once a CUDA runner exists, capture `OPT-0001` acceptance in `OPTIMIZATIONS.md` per the 1.2× speedup threshold at seq=4096.
- Re-run `pytest tests/unit tests/reference --cov=avqa --cov-fail-under=90` before each release.
