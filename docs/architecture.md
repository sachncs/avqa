# AVQA Architecture

This document describes how the implementation maps to the spec's
layered architecture (spec §4) and the cross-references between the
files in `src/avqa/`.

## Layered View

```
applications / user code
        │
        ▼
public AVQA API (avqa.__init__, avqa.functional, avqa.attention_module)
        │
        ▼
core attention pipeline
   ├── avqa.backend (TorchBackend)
   ├── avqa.refinement
   ├── avqa.attention (online softmax state + correction)
   ├── avqa.merge
   ├── avqa.routing
   └── avqa.quantizer
        │
        ▼
   avqa.codebook
        │
        ▼
foundations
   ├── avqa.config (AVQConfig + sub-configs)
   ├── avqa.data (shape + dtype + device + contract)
   ├── avqa.utils (seed, validation, numerics)
   ├── avqa.exceptions
   ├── avqa.logging
   ├── avqa.cache (InMemory, Paged)
   ├── avqa.scheduler (Default, Adaptive)
   ├── avqa.profiling
   ├── avqa.visualization
   ├── avqa.online_adaptation (BCAR)
   ├── avqa.functional
   └── avqa.hopfield (HVAQ temperature schedules)
```

## Subsystem Mapping

| Subsystem | Source | Spec Reference |
|-----------|--------|----------------|
| AVQAttention | `src/avqa/attention_module.py` | §3.4, §5.6, §10.4 |
| Attention (state, correction) | `src/avqa/attention.py` | §7.12, §7.13, §7.14 |
| Backend (Torch) | `src/avqa/backend.py` | §3.12, §4.10, §5.9 |
| Cache | `src/avqa/cache.py` | §3.13, §3.15 |
| Codebook | `src/avqa/codebook.py` | §7.8, §7.9, §8.3-§8.10 |
| Config | `src/avqa/config.py` | §3.6, §5.8, §5.19 |
| Data (shapes, dtypes) | `src/avqa/data.py` | §6.5, §6.9, §6.10 |
| Exceptions | `src/avqa/exceptions.py` | §3.22, §5.13 |
| Functional API | `src/avqa/functional.py` | §3.5, §5.7 |
| HVAQ (Hopfield-VQ-Attention) | `src/avqa/hopfield.py` | §16, §10.7 |
| Logging | `src/avqa/logging.py` | §5.14 |
| Merge strategies | `src/avqa/merge.py` | §3.11, §9.9 |
| Online adaptation (BCAR) | `src/avqa/online_adaptation.py` | §13, §9.12 |
| Profiling | `src/avqa/profiling.py` | §3.17, §5.15 |
| Quantizer | `src/avqa/quantizer.py` | §7.5, §8.3-§8.7 |
| Refinement | `src/avqa/refinement.py` | §9.3, §9.7, §9.8 |
| Routing | `src/avqa/routing.py` | §3.10, §7.10, §9.6 |
| Scheduler | `src/avqa/scheduler.py` | §2.8, §4.7 |
| Utils (seed, validation, numerics) | `src/avqa/utils/*.py` | §3.2, §3.24, §6.12 |
| Visualization | `src/avqa/visualization.py` | §3.18, §5.16 |

## Dependency Rules

Per spec §4.6:

- **Core algorithm MUST NOT import** profiling or visualization modules.
- **Backends implement abstract execution interfaces** (see
  `avqa.backend.Backend`).
- **Profiling observes execution** without altering it.

The current code respects all of these rules. A simple check:

```bash
grep -rn "from avqa.profiling"    src/avqa/*.py     # should be empty
grep -rn "from avqa.visualization" src/avqa/*.py     # should be empty
```

## Extension Points

The core library is intentionally self-contained. Optional adapters for
external systems (Hugging Face Transformers, vLLM, FlashAttention,
xFormers) are not bundled — re-introduce them as separate distribution
extras if your deployment needs them. The internal structure makes this
straightforward: each `Backend` and `Router` subclass is a clean
extension point that the orchestrator can pick up at construction time.
