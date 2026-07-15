# Spec Compliance Matrix

This document tracks the implementation status of every requirement listed
in `docs/checklist.md`. It is regenerated after every milestone.

## Schema

| Column | Meaning |
|--------|---------|
| ID | Requirement ID from `docs/checklist.md`. |
| Section | Spec section reference. |
| Requirement | Brief requirement text. |
| Status | `pending` / `in-progress` / `implemented` / `tested` / `verified`. |
| Source Files | Source files that implement the requirement. |
| Tests | Test files that verify the requirement. |
| Notes | Free-form notes; gaps, deviations, references. |

## Status Definitions

- `pending`: TODO entry exists, not yet started.
- `in-progress`: TODO entry marked `[~]` in `TODO.md`.
- `implemented`: Source code exists; not yet covered by tests.
- `tested`: Source code + tests exist; tests pass locally.
- `verified`: Tests pass in CI; coverage and spec_compliance updated.

## Matrix

| ID | Section | Requirement | Status | Source Files | Tests | Notes |
|----|---------|-------------|--------|--------------|-------|-------|
| REQ-3.2.1 | §3.2 | Implement AVQ-Attention | pending | | | |
| REQ-3.2.2 | §3.2 | Reusable Python package | pending | | | |
| REQ-3.2.3 | §3.2 | Drop-in attention backend | pending | | | |
| REQ-3.2.4 | §3.2 | Inference and training | pending | | | |
| REQ-3.2.5 | §3.2 | CPU and GPU execution | pending | | | |
| REQ-3.2.6 | §3.2 | Pure PyTorch reference | pending | | | |
| REQ-3.2.7 | §3.2 | Optional Triton acceleration | pending | | | Gap G1. |
| REQ-3.2.8 | §3.2 | Mixed precision | pending | | | |
| REQ-3.2.9 | §3.2 | Deterministic mode | pending | | | |
| REQ-3.3.1 | §3.3 | Clean public API | pending | | | |
| REQ-3.3.2 | §3.3 | No internal modification | pending | | | |
| REQ-3.3.3 | §3.3 | Standard package manager install | pending | | | |
| REQ-3.3.4 | §3.3 | Internal modules hidden | pending | | | |
| REQ-3.4.1 | §3.4 | nn.Module inheritance | pending | | | |
| REQ-3.4.2 | §3.4 | Batched execution | pending | | | |
| REQ-3.4.3 | §3.4 | Multi-head attention | pending | | | |
| REQ-3.4.4 | §3.4 | Causal attention | pending | | | Gap G13. |
| REQ-3.4.5 | §3.4 | Bidirectional attention | pending | | | |
| REQ-3.4.6 | §3.4 | Masking | pending | | | |
| REQ-3.4.7 | §3.4 | Dropout | pending | | | |
| REQ-3.4.8 | §3.4 | Arbitrary sequence lengths | pending | | | |
| REQ-3.4.9 | §3.4 | Configurable embedding dim | pending | | | |
| REQ-3.4.10 | §3.4 | Configurable head count | pending | | | |
| REQ-3.4.11 | §3.4 | PyTorch-conventional API | pending | | | |
| REQ-3.5.1 | §3.5 | Functional API | pending | | | |
| REQ-3.5.2 | §3.5 | Stateless functional API | pending | | | |
| REQ-3.6.1 | §3.6 | Explicit configuration | pending | | | |
| REQ-3.6.2 | §3.6 | Configuration contents | pending | | | |
| REQ-3.6.3 | §3.6 | Configuration serialization | pending | | | |
| REQ-3.6.4 | §3.6 | Configuration validation | pending | | | |
| REQ-3.6.5 | §3.6 | Immutable configuration | pending | | | |
| REQ-3.7.1 | §3.7 | VectorQuantizer capabilities | pending | | | |
| REQ-3.7.2 | §3.7 | Optional VQ features | pending | | | Gap G11. |
| REQ-3.8.1 | §3.8 | Hierarchical codebook | pending | | | |
| REQ-3.8.2 | §3.8 | Codebook introspection | pending | | | |
| REQ-3.9.1 | §3.9 | Refinement stages | pending | | | |
| REQ-3.9.2 | §3.9 | Configurable selection | pending | | | |
| REQ-3.9.3 | §3.9 | Public refinement interface | pending | | | |
| REQ-3.10.1 | §3.10 | Routing as subsystem | pending | | | |
| REQ-3.10.2 | §3.10 | Routing does not compute attention | pending | | | |
| REQ-3.11.1 | §3.11 | Separate merge | pending | | | |
| REQ-3.11.2 | §3.11 | Merge strategy list | pending | | | |
| REQ-3.11.3 | §3.11 | Custom merge registration | pending | | | |
| REQ-3.12.1 | §3.12 | Multiple backends | pending | | | |
| REQ-3.12.2 | §3.12 | PyTorch and Triton backends | pending | | | Gap G1. |
| REQ-3.12.3 | §3.12 | Future backends addable | pending | | | |
| REQ-3.12.4 | §3.12 | Backend selection via config | pending | | | |
| REQ-3.12.5 | §3.12 | Automatic backend selection | pending | | | |
| REQ-3.13.1 | §3.13 | Autoregressive decoding | pending | | | |
| REQ-3.13.2 | §3.13 | Cache capabilities | pending | | | |
| REQ-3.13.3 | §3.13 | Stable cache interface | pending | | | |
| REQ-3.14.1 | §3.14 | Hugging Face support | pending | | | Gap G2. |
| REQ-3.14.2 | §3.14 | HF replacement helper | pending | | | Gap G2. |
| REQ-3.14.3 | §3.14 | HF preserves state | pending | | | Gap G2. |
| REQ-3.15.1 | §3.15 | vLLM integration | pending | | | Gap G2. |
| REQ-3.15.2 | §3.15 | vLLM features | pending | | | Gap G2. |
| REQ-3.15.3 | §3.15 | Framework isolation | pending | | | |
| REQ-3.16.1 | §3.16 | FlashAttention interop | pending | | | Gap G2. |
| REQ-3.16.2 | §3.16 | Backend selection order | pending | | | Gap G2. |
| REQ-3.16.3 | §3.16 | Numerical equivalence | pending | | | Gap G6. |
| REQ-3.17.1 | §3.17 | Profiling capabilities | pending | | | Gap G3. |
| REQ-3.17.2 | §3.17 | Profiling optional | pending | | | |
| REQ-3.18.1 | §3.18 | Visualization features | pending | | | Gap G4. |
| REQ-3.18.2 | §3.18 | Visualization isolation | pending | | | |
| REQ-3.19.1 | §3.19 | Benchmark suite | pending | | | |
| REQ-3.19.2 | §3.19 | Compare to baselines | pending | | | |
| REQ-3.19.3 | §3.19 | Reproducible benchmarks | pending | | | |
| REQ-3.20.1 | §3.20 | Serialization coverage | pending | | | |
| REQ-3.20.2 | §3.20 | Versioned serialization | pending | | | Gap G5. |
| REQ-3.21.1 | §3.21 | User-defined components | pending | | | |
| REQ-3.21.2 | §3.21 | Documented extension interfaces | pending | | | |
| REQ-3.22.1 | §3.22 | Custom exception types | pending | | | |
| REQ-3.22.2 | §3.22 | Error context | pending | | | |
| REQ-3.22.3 | §3.22 | Informative messages | pending | | | |
| REQ-3.23.1 | §3.23 | Public API documentation | pending | | | |
| REQ-3.24.1 | §3.24 | Automated test coverage | pending | | | |
| REQ-3.24.2 | §3.24 | Test categories | pending | | | |
| REQ-4.3.1 | §4.3 | No layer bypass | pending | | | |
| REQ-4.5.1 | §4.5 | Narrow subsystem interfaces | pending | | | |
| REQ-4.6.1 | §4.6 | No framework imports in core | pending | | | |
| REQ-4.6.2 | §4.6 | Integration dependency direction | pending | | | |
| REQ-4.6.3 | §4.6 | Backend abstract interface | pending | | | |
| REQ-4.6.4 | §4.6 | Algorithm has no backend logic | pending | | | |
| REQ-4.6.5 | §4.6 | Profiling does not modify | pending | | | |
| REQ-4.7.1 | §4.7 | Attention does not quantize | pending | | | |
| REQ-4.7.2 | §4.7 | Quantizer does not attend | pending | | | |
| REQ-4.7.3 | §4.7 | Codebook independent of routing | pending | | | |
| REQ-4.7.4 | §4.7 | Router does not expand | pending | | | |
| REQ-4.7.5 | §4.7 | Refinement does not select | pending | | | |
| REQ-4.7.6 | §4.7 | Merge does not route | pending | | | |
| REQ-4.7.7 | §4.7 | Scheduler backend-independent | pending | | | |
| REQ-4.8.1 | §4.8 | Immutable communication | pending | | | |
| REQ-4.10.1 | §4.10 | Common backend interface | pending | | | |
| REQ-4.10.2 | §4.10 | No hardware branching | pending | | | |
| REQ-4.11.1 | §4.11 | Adapters have no algorithm | pending | | | |
| REQ-4.12.1 | §4.12 | Extension interfaces | pending | | | |
| REQ-4.12.2 | §4.12 | Registration without modification | pending | | | |
| REQ-4.13.1 | §4.13 | Mode via config | pending | | | |
| REQ-4.14.1 | §4.14 | No circular deps | pending | | | |
| REQ-4.14.2 | §4.14 | No global state | pending | | | |
| REQ-4.14.3 | §4.14 | No framework logic in core | pending | | | |
| REQ-4.14.4 | §4.14 | No backend logic in math | pending | | | |
| REQ-4.14.5 | §4.14 | No mutable config | pending | | | |
| REQ-4.14.6 | §4.14 | No side effects in forward | pending | | | |
| REQ-4.14.7 | §4.14 | Backward compatibility | pending | | | |
| REQ-5.3.1 | §5.3 | Public namespace | pending | | | |
| REQ-5.4.1 | §5.4 | Direct import | pending | | | |
| REQ-5.4.2 | §5.4 | No private imports | pending | | | |
| REQ-5.5.1 | §5.5 | Public classes | pending | | | |
| REQ-5.6.1 | §5.6 | nn.Module inheritance | pending | | | |
| REQ-5.6.2 | §5.6 | Single config arg | pending | | | |
| REQ-5.7.1 | §5.7 | Stateless functional API | pending | | | |
| REQ-5.8.1 | §5.8 | Immutable dataclass config | pending | | | |
| REQ-5.8.2 | §5.8 | Config capabilities | pending | | | |
| REQ-5.8.3 | §5.8 | Config reuse | pending | | | |
| REQ-5.9.1 | §5.9 | Common backend interface | pending | | | |
| REQ-5.9.2 | §5.9 | Backend methods | pending | | | |
| REQ-5.10.1 | §5.10 | Registry mechanism | pending | | | |
| REQ-5.10.2 | §5.10 | Registry categories | pending | | | |
| REQ-5.10.3 | §5.10 | Registration without modification | pending | | | |
| REQ-5.12.1 | §5.12 | Public serialization | pending | | | |
| REQ-5.12.2 | §5.12 | Version metadata | pending | | | Gap G5. |
| REQ-5.13.1 | §5.13 | Exception hierarchy | pending | | | |
| REQ-5.13.2 | §5.13 | Documented exceptions | pending | | | |
| REQ-5.14.1 | §5.14 | logging integration | pending | | | |
| REQ-5.14.2 | §5.14 | User-controlled verbosity | pending | | | |
| REQ-5.14.3 | §5.14 | No stdout prints | pending | | | |
| REQ-5.15.1 | §5.15 | Optional profiling | pending | | | |
| REQ-5.15.2 | §5.15 | Profiling non-invasive | pending | | | |
| REQ-5.16.1 | §5.16 | Decoupled visualization | pending | | | |
| REQ-5.17.1 | §5.17 | Adapters for frameworks | pending | | | Gap G2. |
| REQ-5.17.2 | §5.17 | Adapter translation only | pending | | | |
| REQ-5.18.1 | §5.18 | Closed for modification | pending | | | |
| REQ-5.19.1 | §5.19 | Semantic versioning | pending | | | |
| REQ-5.19.2 | §5.19 | Minor backward compatible | pending | | | |
| REQ-5.19.3 | §5.19 | Patch only fixes | pending | | | |
| REQ-5.19.4 | §5.19 | Deprecation window | pending | | | |
| REQ-5.20.1 | §5.20 | Class name stability | pending | | | |
| REQ-5.20.2 | §5.20 | Function signature stability | pending | | | |
| REQ-5.20.3 | §5.20 | Config schema stability | pending | | | |
| REQ-5.20.4 | §5.20 | Behavior stability | pending | | | |
| REQ-6.3.1 | §6.3 | Symbol consistency | pending | | | |
| REQ-6.4.1 | §6.4 | Descriptive tensor names | pending | | | |
| REQ-6.5.1 | §6.5 | Q/K/V shape | pending | | | |
| REQ-6.5.2 | §6.5 | Codebook shape | pending | | | |
| REQ-6.5.3 | §6.5 | Assignment shape | pending | | | |
| REQ-6.5.4 | §6.5 | Routing scores shape | pending | | | |
| REQ-6.5.5 | §6.5 | Active codewords shape | pending | | | |
| REQ-6.5.6 | §6.5 | Refined codebook shape | pending | | | |
| REQ-6.5.7 | §6.5 | Refined attention shape | pending | | | |
| REQ-6.5.8 | §6.5 | Final attention shape | pending | | | |
| REQ-6.5.9 | §6.5 | Output shape | pending | | | |
| REQ-6.8.1 | §6.8 | Memory layout | pending | | | |
| REQ-6.9.1 | §6.9 | FP32/FP16/BF16 | pending | | | |
| REQ-6.9.2 | §6.9 | Optional dtypes | pending | | | Gap G10. |
| REQ-6.10.1 | §6.10 | Single device | pending | | | |
| REQ-6.10.2 | §6.10 | Explicit movement | pending | | | |
| REQ-6.10.3 | §6.10 | No implicit transfers | pending | | | |
| REQ-6.11.1 | §6.11 | Immutability during forward | pending | | | |
| REQ-6.11.2 | §6.11 | Localized mutation | pending | | | |
| REQ-6.12.1 | §6.12 | Validation | pending | | | |
| REQ-6.12.2 | §6.12 | Optional validation | pending | | | |
| REQ-6.13.1 | §6.13 | Q/K/V shape invariants | pending | | | |
| REQ-6.13.2 | §6.13 | Output shape invariant | pending | | | |
| REQ-6.13.3 | §6.13 | Exception on violation | pending | | | |
| REQ-6.15.1 | §6.15 | Immutable stage inputs | pending | | | |
| REQ-6.16.1 | §6.16 | Distributed semantics | pending | | | Gap G7. |
| REQ-6.17.1 | §6.17 | Serializable objects | pending | | | |
| REQ-6.17.2 | §6.17 | No transient serialization | pending | | | |
| REQ-6.18.1 | §6.18 | Tensor metadata | pending | | | |
| REQ-6.19.1 | §6.19 | Explicit contracts | pending | | | |
| REQ-6.19.2 | §6.19 | Function doc contracts | pending | | | |
| REQ-7.5.1 | §7.5 | Euclidean nearest assignment | pending | | | |
| REQ-7.5.2 | §7.5 | Keys-only quantization | pending | | | |
| REQ-7.7.1 | §7.7 | VQ attention formula | pending | | | |
| REQ-7.8.1 | §7.8 | Codebook size formula | pending | | | |
| REQ-7.8.2 | §7.8 | Children supplement | pending | | | |
| REQ-7.9.1 | §7.9 | Mean constraint | pending | | | |
| REQ-7.10.1 | §7.10 | Importance formula | pending | | | |
| REQ-7.10.2 | §7.10 | No auxiliary networks | pending | | | |
| REQ-7.11.1 | §7.11 | Refinement pipeline | pending | | | |
| REQ-7.11.2 | §7.11 | P-parents rule | pending | | | |
| REQ-7.12.1 | §7.12 | Parent logit recovery | pending | | | |
| REQ-7.12.2 | §7.12 | No extra matmul | pending | | | |
| REQ-7.13.1 | §7.13 | Correcting attention | pending | | | |
| REQ-7.14.1 | §7.14 | Online softmax | pending | | | |
| REQ-7.14.2 | §7.14 | No global recompute | pending | | | |
| REQ-7.15.1 | §7.15 | Empty max handling | pending | | | |
| REQ-7.15.2 | §7.15 | Empty contribution | pending | | | Gap G12. |
| REQ-7.16.1 | §7.16 | Complexity | pending | | | |
| REQ-7.16.2 | §7.16 | Linear in sequence | pending | | | |
| REQ-7.17.1 | §7.17 | Hierarchy invariant | pending | | | |
| REQ-7.17.2 | §7.17 | Assignment invariant | pending | | | |
| REQ-7.17.3 | §7.17 | Conservation invariant | pending | | | |
| REQ-7.17.4 | §7.17 | Count invariant | pending | | | |
| REQ-7.17.5 | §7.17 | Attention invariant | pending | | | |
| REQ-7.18.1 | §7.18 | Approximation source | pending | | | |
| REQ-7.19.1 | §7.19 | Complexity guarantee | pending | | | |
| REQ-7.19.2 | §7.19 | Parent reconstruction exactness | pending | | | |
| REQ-7.19.3 | §7.19 | Correcting attention semantics | pending | | | |
| REQ-7.19.4 | §7.19 | Online softmax stability | pending | | | |
| REQ-8.3.1 | §8.3 | Two-level hierarchy | pending | | | |
| REQ-8.3.2 | §8.3 | Mean constraint | pending | | | |
| REQ-8.4.1 | §8.4 | Receive K/V | pending | | | |
| REQ-8.4.2 | §8.4 | Parent assignment | pending | | | |
| REQ-8.4.3 | §8.4 | Child assignment | pending | | | |
| REQ-8.4.4 | §8.4 | Value accumulation | pending | | | |
| REQ-8.4.5 | §8.4 | Count accumulation | pending | | | |
| REQ-8.4.6 | §8.4 | Six outputs | pending | | | |
| REQ-8.5.1 | §8.5 | Two-stage assignment | pending | | | |
| REQ-8.5.2 | §8.5 | Reduced complexity | pending | | | |
| REQ-8.6.1 | §8.6 | Aggregate per codeword | pending | | | |
| REQ-8.6.2 | §8.6 | Direct inputs | pending | | | |
| REQ-8.6.3 | §8.6 | No per-key downstream | pending | | | |
| REQ-8.7.1 | §8.7 | Fused precompute | pending | | | |
| REQ-8.7.2 | §8.7 | Fused in optimized | pending | | | |
| REQ-8.8.1 | §8.8 | Stable tensor contracts | pending | | | |
| REQ-8.9.1 | §8.9 | EMA training | pending | | | |
| REQ-8.9.2 | §8.9 | Decay default | pending | | | |
| REQ-8.9.3 | §8.9 | Commitment default | pending | | | |
| REQ-8.9.4 | §8.9 | Repropagate parents | pending | | | |
| REQ-8.10.1 | §8.10 | Child initialization | pending | | | |
| REQ-8.10.2 | §8.10 | Init alternatives | pending | | | |
| REQ-8.11.1 | §8.11 | Reference EMA | pending | | | |
| REQ-8.11.2 | §8.11 | Optional enhancement | pending | | | Gap G9. |
| REQ-8.12.1 | §8.12 | Preprocess complexity | pending | | | |
| REQ-8.13.1 | §8.13 | Algorithmic separation | pending | | | |
| REQ-8.13.2 | §8.13 | Deterministic assignment | pending | | | |
| REQ-8.13.3 | §8.13 | Minimal copies | pending | | | |
| REQ-8.13.4 | §8.13 | Batched execution | pending | | | |
| REQ-8.13.5 | §8.13 | Three dtypes | pending | | | |
| REQ-8.13.6 | §8.13 | PyTorch + Triton | pending | | | |
| REQ-8.13.7 | §8.13 | Profiling statistics | pending | | | |
| REQ-8.13.8 | §8.13 | Extensibility | pending | | | |
| REQ-9.3.1 | §9.3 | Pipeline order | pending | | | |
| REQ-9.4.1 | §9.4 | Stage 1 parent attention | pending | | | |
| REQ-9.5.1 | §9.5 | Importance from attention | pending | | | |
| REQ-9.5.2 | §9.5 | Importance reuse | pending | | | |
| REQ-9.6.1 | §9.6 | Selection determinism | pending | | | Gap G14. |
| REQ-9.6.2 | §9.6 | Reference selection | pending | | | |
| REQ-9.7.1 | §9.7 | Expansion steps | pending | | | |
| REQ-9.7.2 | §9.7 | Unselected unchanged | pending | | | |
| REQ-9.8.1 | §9.8 | Recompute only children | pending | | | |
| REQ-9.8.2 | §9.8 | No parent recompute | pending | | | |
| REQ-9.9.1 | §9.9 | Replace not augment | pending | | | |
| REQ-9.10.1 | §9.10 | Parent logit reconstruction | pending | | | |
| REQ-9.10.2 | §9.10 | No matmul | pending | | | |
| REQ-9.11.1 | §9.11 | Running accumulators | pending | | | |
| REQ-9.11.2 | §9.11 | No global recompute | pending | | | |
| REQ-9.11.3 | §9.11 | Incremental correction | pending | | | |
| REQ-9.12.1 | §9.12 | Empty codeword rules | pending | | | Gap G12. |
| REQ-9.13.1 | §9.13 | Complexity | pending | | | |
| REQ-9.13.2 | §9.13 | Linear in sequence | pending | | | |
| REQ-9.14.1 | §9.14 | Numerical preservation | pending | | | |
| REQ-9.15.1 | §9.15 | Logical/backend separation | pending | | | |
| REQ-9.15.2 | §9.15 | Reuse intermediates | pending | | | |
| REQ-9.15.3 | §9.15 | No parent recompute | pending | | | |
| REQ-9.15.4 | §9.15 | Causal and non-causal | pending | | | Gap G13. |
| REQ-9.15.5 | §9.15 | Three dtypes | pending | | | |
| REQ-9.15.6 | §9.15 | Refinement stats | pending | | | |
| REQ-9.15.7 | §9.15 | Configurable budget | pending | | | |
| REQ-10.2.1 | §10.2 | Two phases | pending | | | |
| REQ-10.3.1 | §10.3 | Pipeline order | pending | | | |
| REQ-10.4.1 | §10.4 | Nine stages | pending | | | |
| REQ-10.4.2 | §10.4 | Backend ordering | pending | | | |
| REQ-10.5.1 | §10.5 | Pre-execution validation | pending | | | |
| REQ-10.5.2 | §10.5 | Engine verification | pending | | | |
| REQ-10.6.1 | §10.6 | Precompute before attention | pending | | | |
| REQ-10.6.2 | §10.6 | No dependency on uninit | pending | | | |
| REQ-10.7.1 | §10.7 | Parent attention outputs | pending | | | |
| REQ-10.8.1 | §10.8 | Importance immediately after | pending | | | |
| REQ-10.8.2 | §10.8 | No auxiliary networks | pending | | | |
| REQ-10.9.1 | §10.9 | Ordered list | pending | | | |
| REQ-10.9.2 | §10.9 | Selection properties | pending | | | |
| REQ-10.10.1 | §10.10 | Selective processing | pending | | | |
| REQ-10.11.1 | §10.11 | Correction steps | pending | | | |
| REQ-10.11.2 | §10.11 | No full matrix | pending | | | |
| REQ-10.12.1 | §10.12 | Output shape preserved | pending | | | |
| REQ-10.13.1 | §10.13 | Standard output projection | pending | | | |
| REQ-10.13.2 | §10.13 | No architectural change | pending | | | |
| REQ-10.14.1 | §10.14 | Ephemeral release | pending | | | |
| REQ-10.15.1 | §10.15 | Mode via config | pending | | | |
| REQ-10.16.1 | §10.16 | Error detection | pending | | | |
| REQ-10.16.2 | §10.16 | No partial output | pending | | | |
| REQ-10.19.1 | §10.19 | Ordering invariant | pending | | | |
| REQ-10.19.2 | §10.19 | Consistency invariant | pending | | | |
| REQ-10.19.3 | §10.19 | Isolation invariant | pending | | | |
| REQ-10.19.4 | §10.19 | Determinism invariant | pending | | | |
| REQ-10.19.5 | §10.19 | Normalization invariant | pending | | | |
| REQ-10.20.1 | §10.20 | Backend optimization freedom | pending | | | |
| REQ-10.21.1 | §10.21 | Linear memory | pending | | | |

## Summary

- **Total requirements:** 238
- **Implemented:** 0
- **Tested:** 0
- **Verified:** 0

Status counts will be updated as milestones complete.
