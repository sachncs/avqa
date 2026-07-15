# Dependency Graph

This document describes the dependency graph of the AVQA subsystems.
Edges point from a higher-level subsystem to the subsystem it depends on.
No edge exists in the reverse direction. Implementation proceeds bottom-up.

## Legend

- Solid edge: required import.
- Dashed edge: optional import (e.g., registry hook, framework integration).

## High-level Subsystem Graph

```
                          ┌────────────────────────────┐
                          │   applications/user code    │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────┐
                          │   framework integrations    │
                          │  HF · vLLM · FA · xFormers  │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────┐
                          │      public API surface    │
                          │  AVQAttention · functional │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────┐
                          │   attention module/pipeline│
                          └────┬───────────────┬───────┘
                               │               │
                               ▼               ▼
                    ┌──────────────┐   ┌────────────────┐
                    │  refinement  │   │   backend base │
                    └──────┬───────┘   └────┬───────────┘
                           │               │
                           ▼               ▼
              ┌──────────┬───────┐   ┌─────────────────┐
              │ routing  │ merge │   │ torch_backend   │
              └─────┬────┴───┬───┘   │ triton_backend  │
                    │        │       └────────┬────────┘
                    ▼        ▼                │
              ┌────────┐  ┌─────────────┐     │
              │ quant. │  │  correction │     │
              └────┬───┘  │ online_soft │     │
                   │      └──────┬──────┘     │
                   ▼             │            │
              ┌──────────┐       │            │
              │ codebook │       │            │
              └────┬─────┘       │            │
                   │             │            │
                   ▼             ▼            ▼
              ┌──────────────────────────────────────┐
              │            foundations               │
              │  config · data · utils · exceptions  │
              │  logging · registry · scheduler      │
              │              cache                   │
              └──────────────────────────────────────┘
```

## Module-level Dependency Graph

```
exceptions ◀── logging ◀── utils ◀── registry ◀── data
                                              │
                                              ▼
                                          config
                                              │
                                              ▼
                                         codebook
                                              │
                                              ▼
                                         quantizer
                                              │
                                              ▼
                                          routing
                                              │
                                              ▼
                                            merge
                                              │
                                              ▼
                                     online_softmax
                                              │
                                              ▼
                                         correction
                                              │
                                              ▼
                                        refinement
                                              │
                                              ▼
                                         scheduler
                                              │
                                              ▼
                                       backend.base
                                       /          \
                              torch_backend   triton_backend
                                              │
                                              ▼
                                          cache
                                              │
                                              ▼
                                      attention.pipeline
                                              │
                                              ▼
                                       attention.module
                                              │
                                              ▼
                                          functional
                                              │
                                              ▼
                                profiling · visualization
                                              │
                                              ▼
                                         integrations
                                              │
                                              ▼
                                       benchmarks/examples
```

## File-level Cross-Dependencies

| Module | Direct Imports (algorithm core only) |
|--------|--------------------------------------|
| `avqa/__init__.py` | All public classes. |
| `avqa/exceptions.py` | stdlib only. |
| `avqa/logging.py` | `exceptions`, stdlib. |
| `avqa/registry.py` | `exceptions`, stdlib. |
| `avqa/utils/seed.py` | stdlib + torch. |
| `avqa/utils/validation.py` | `exceptions`, torch. |
| `avqa/utils/numerics.py` | torch. |
| `avqa/data/shapes.py` | stdlib. |
| `avqa/data/dtypes.py` | torch. |
| `avqa/data/devices.py` | torch. |
| `avqa/data/contracts.py` | `data.shapes`, `data.dtypes`, `data.devices`. |
| `avqa/config/avq.py` | All sub-configs, `exceptions`, `data.contracts`. |
| `avqa/config/codebook.py` | `data.shapes`. |
| `avqa/config/routing.py` | `exceptions`. |
| `avqa/config/refinement.py` | `exceptions`. |
| `avqa/config/merge.py` | `exceptions`. |
| `avqa/config/backend.py` | `exceptions`. |
| `avqa/config/cache.py` | `exceptions`. |
| `avqa/config/precision.py` | `data.dtypes`. |
| `avqa/config/execution.py` | `exceptions`. |
| `avqa/config/serialization.py` | All sub-configs. |
| `avqa/codebook/hierarchical.py` | `config.codebook`, `utils.validation`. |
| `avqa/codebook/init.py` | `codebook.hierarchical`. |
| `avqa/codebook/training.py` | `codebook.hierarchical`. |
| `avqa/quantizer/base.py` | `registry`, `codebook.hierarchical`. |
| `avqa/quantizer/euclidean.py` | `quantizer.base`. |
| `avqa/quantizer/hierarchical.py` | `quantizer.euclidean`, `codebook.hierarchical`. |
| `avqa/quantizer/ema.py` | `quantizer.hierarchical`. |
| `avqa/quantizer/stats.py` | `quantizer.hierarchical`. |
| `avqa/routing/base.py` | `registry`. |
| `avqa/routing/importance.py` | `routing.base`. |
| `avqa/routing/topp.py` | `routing.importance`. |
| `avqa/routing/threshold.py` | `routing.importance`. |
| `avqa/routing/budget.py` | `routing.importance`. |
| `avqa/merge/base.py` | `registry`. |
| `avqa/merge/probability.py` | `merge.base`. |
| `avqa/merge/weighted.py` | `merge.base`. |
| `avqa/merge/logit.py` | `merge.base`. |
| `avqa/merge/normalized.py` | `merge.base`. |
| `avqa/attention/online_softmax.py` | `utils.numerics`. |
| `avqa/attention/correction.py` | `attention.online_softmax`. |
| `avqa/refinement/expansion.py` | `codebook.hierarchical`, `quantizer.hierarchical`. |
| `avqa/refinement/adaptive.py` | `refinement.expansion`, `routing`, `merge`, `attention.correction`. |
| `avqa/scheduler/base.py` | `registry`. |
| `avqa/scheduler/default.py` | `scheduler.base`. |
| `avqa/scheduler/adaptive.py` | `scheduler.default`. |
| `avqa/cache/base.py` | `config.cache`. |
| `avqa/cache/in_memory.py` | `cache.base`. |
| `avqa/cache/paged.py` | `cache.base`. |
| `avqa/backend/base.py` | `config.backend`, `cache.base`. |
| `avqa/backend/torch_backend.py` | `backend.base`. |
| `avqa/backend/triton_backend.py` | `backend.base` (Triton optional). |
| `avqa/backend/factory.py` | `backend.base`, `registry`. |
| `avqa/attention/pipeline.py` | `quantizer`, `routing`, `refinement`, `merge`, `attention.online_softmax`, `attention.correction`, `backend.base`. |
| `avqa/attention/module.py` | `attention.pipeline`, `config.avq`. |
| `avqa/functional.py` | `attention.module`. |
| `avqa/profiling/base.py` | `registry`. |
| `avqa/profiling/metrics.py` | `profiling.base`. |
| `avqa/profiling/report.py` | `profiling.metrics`. |
| `avqa/visualization/base.py` | `registry`. |
| `avqa/visualization/tree.py` | `visualization.base`. |
| `avqa/visualization/heatmap.py` | `visualization.base`. |
| `avqa/visualization/timeline.py` | `visualization.base`. |
| `avqa/visualization/utilization.py` | `visualization.base`. |
| `avqa/integrations/huggingface.py` | `attention.module`. |
| `avqa/integrations/vllm.py` | `attention.module`, `cache.paged`. |
| `avqa/integrations/flash_attention.py` | `attention.module`. |
| `avqa/integrations/xformers.py` | `attention.module`. |

## Forbidden Imports

The following imports are forbidden in the algorithm core:

- `transformers` (Hugging Face) — use `avqa.integrations.huggingface` instead.
- `vllm` — use `avqa.integrations.vllm` instead.
- `flash_attn` — use `avqa.integrations.flash_attention` instead.
- `xformers` — use `avqa.integrations.xformers` instead.
- `triton` outside `avqa.backend.triton_backend` (Triton is allowed only inside
  the dedicated backend module).
- Any reverse import from `avqa.integrations` → algorithm core is fine; the
  reverse is not.

These rules are enforced by:

- Manual review per commit.
- Optional `import-linter` configuration (added in TASK-0164 if available).

## Critical Path

The longest dependency chain, used to estimate implementation depth:

```
exceptions → logging → registry → config → codebook → quantizer →
routing → merge → online_softmax → correction → refinement →
backend.base → torch_backend → cache → attention.pipeline →
attention.module → functional → profiling → visualization →
integrations → benchmarks → release
```

This chain defines the critical path. Tasks that branch off this chain
(parallel work) can be implemented concurrently once their branch's
dependencies are satisfied.

## Parallel Branch Opportunities

The following branches can be parallelized once the chain reaches the
indicated milestone:

- After M6: Profiling (M12) and Visualization (M12) can begin in parallel
  with Refinement (M7) and Backend (M8) work.
- After M8: Integrations (M13) can begin in parallel with Cache (M9),
  Attention module (M10), and Functional API (M11).
- After M11: Benchmarks (M14) and Examples (M14) can begin in parallel with
  Release prep (M15).

Single-developer execution does not benefit from this parallelism, but it
informs future contributor onboarding and review scheduling.
