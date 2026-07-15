# Requirement Checklist

This checklist enumerates every SHALL, MUST, SHOULD, and MAY statement found
in `spec.md` (Chapters 1–10), plus the cross-chapter requirements referenced
in those chapters. Each requirement appears exactly once. Implementation
status is tracked in `docs/spec_compliance.md`; this file is the canonical
list of requirements.

**Source**: `spec.md` (3,960 lines), Chapters 2 through 10.

## Chapter 2 — Paper Review & Mathematical Foundations

This chapter is informational. It does not impose SHALL requirements but
motivates the architecture. No checklist entries.

## Chapter 3 — Functional Requirements

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-3.2.1 | §3.2 | Implement AVQ-Attention as described in the reference paper. |
| REQ-3.2.2 | §3.2 | Expose the algorithm through a reusable Python package. |
| REQ-3.2.3 | §3.2 | Operate as a drop-in attention backend. |
| REQ-3.2.4 | §3.2 | Support both inference and training. |
| REQ-3.2.5 | §3.2 | Support CPU and GPU execution. |
| REQ-3.2.6 | §3.2 | Provide a pure PyTorch reference implementation. |
| REQ-3.2.7 | §3.2 | Provide optional Triton acceleration. |
| REQ-3.2.8 | §3.2 | Support mixed precision execution. |
| REQ-3.2.9 | §3.2 | Maintain deterministic execution when deterministic mode is enabled. |
| REQ-3.3.1 | §3.3 | The package SHALL expose a clean public API. |
| REQ-3.3.2 | §3.3 | Users SHALL NOT be required to modify internal implementation code. |
| REQ-3.3.3 | §3.3 | The package SHALL support installation through standard Python package managers. |
| REQ-3.3.4 | §3.3 | Internal modules SHALL remain hidden unless explicitly documented. |
| REQ-3.4.1 | §3.4 | The primary attention module SHALL inherit from `torch.nn.Module`. |
| REQ-3.4.2 | §3.4 | The attention module SHALL support batched execution. |
| REQ-3.4.3 | §3.4 | The attention module SHALL support multi-head attention. |
| REQ-3.4.4 | §3.4 | The attention module SHALL support causal attention. |
| REQ-3.4.5 | §3.4 | The attention module SHALL support bidirectional attention. |
| REQ-3.4.6 | §3.4 | The attention module SHALL support masking. |
| REQ-3.4.7 | §3.4 | The attention module SHALL support dropout. |
| REQ-3.4.8 | §3.4 | The attention module SHALL support arbitrary sequence lengths. |
| REQ-3.4.9 | §3.4 | The attention module SHALL support configurable embedding dimensions. |
| REQ-3.4.10 | §3.4 | The attention module SHALL support configurable numbers of attention heads. |
| REQ-3.4.11 | §3.4 | The public interface SHOULD closely resemble existing PyTorch attention modules. |
| REQ-3.5.1 | §3.5 | The library SHALL provide a functional API. |
| REQ-3.5.2 | §3.5 | The functional interface SHALL remain stateless. |
| REQ-3.6.1 | §3.6 | All configurable behavior SHALL be controlled through explicit configuration objects. |
| REQ-3.6.2 | §3.6 | Configuration SHALL include at minimum: codebook size, branching factor, refinement budget, routing strategy, merge strategy, backend, execution mode, precision, cache configuration. |
| REQ-3.6.3 | §3.6 | Configuration SHALL support serialization. |
| REQ-3.6.4 | §3.6 | Configuration SHALL support validation. |
| REQ-3.6.5 | §3.6 | Configuration SHALL be immutable after construction unless explicitly documented. |
| REQ-3.7.1 | §3.7 | The implementation SHALL provide a vector quantizer capable of: assigning vectors to codewords, computing assignments in batches, updating codebooks during training, supporting inference without retraining, supporting configurable codebook sizes. |
| REQ-3.7.2 | §3.7 | Optional capabilities MAY include: EMA updates, FAISS acceleration, k-means initialization, custom distance metrics. |
| REQ-3.8.1 | §3.8 | The implementation SHALL provide a hierarchical codebook supporting: parent-child relationships, configurable tree depth, configurable branching factor, efficient traversal, serialization, statistics collection. |
| REQ-3.8.2 | §3.8 | The codebook SHALL expose sufficient information for routing, visualization, and debugging. |
| REQ-3.9.1 | §3.9 | Adaptive refinement SHALL support: coarse attention, active node selection, node expansion, refined attention, probability merging. |
| REQ-3.9.2 | §3.9 | Selection policies SHALL be configurable. |
| REQ-3.9.3 | §3.9 | The implementation SHALL expose a public interface for implementing additional refinement policies. |
| REQ-3.10.1 | §3.10 | Routing SHALL be implemented as an independent subsystem. |
| REQ-3.10.2 | §3.10 | Routing SHALL NOT perform attention computations directly. |
| REQ-3.11.1 | §3.11 | The implementation SHALL separate refinement from probability merging. |
| REQ-3.11.2 | §3.11 | Supported merge strategies SHOULD include: probability merge, weighted merge, logit merge, normalized merge. |
| REQ-3.11.3 | §3.11 | Users SHALL be able to register custom merge strategies. |
| REQ-3.12.1 | §3.12 | The implementation SHALL support multiple execution backends. |
| REQ-3.12.2 | §3.12 | Initially supported backends include: PyTorch, Triton. |
| REQ-3.12.3 | §3.12 | Future backends SHOULD be addable without modifying algorithmic code. |
| REQ-3.12.4 | §3.12 | Backend selection SHALL occur through configuration. |
| REQ-3.12.5 | §3.12 | Automatic backend selection MAY be supported. |
| REQ-3.13.1 | §3.13 | The implementation SHALL support autoregressive decoding. |
| REQ-3.13.2 | §3.13 | The cache SHALL support: incremental updates, efficient lookup, configurable storage, cache reset, serialization. |
| REQ-3.13.3 | §3.13 | The implementation SHALL expose a stable cache interface independent of any specific inference engine. |
| REQ-3.14.1 | §3.14 | The library SHALL support Hugging Face Transformers. |
| REQ-3.14.2 | §3.14 | Users SHOULD be able to replace compatible attention layers with AVQA using a documented helper function. |
| REQ-3.14.3 | §3.14 | The integration SHALL preserve: pretrained weights, model configuration, inference behavior, training compatibility. |
| REQ-3.15.1 | §3.15 | The library SHALL integrate with vLLM through documented extension points whenever possible. |
| REQ-3.15.2 | §3.15 | The integration SHALL support: paged attention, continuous batching, prefix caching, tensor parallelism where supported, speculative decoding where compatible. |
| REQ-3.15.3 | §3.15 | Framework-specific code SHALL remain isolated from the core algorithm. |
| REQ-3.16.1 | §3.16 | Where FlashAttention is available, the implementation SHALL support interoperability. |
| REQ-3.16.2 | §3.16 | The backend selection order SHOULD be configurable. |
| REQ-3.16.3 | §3.16 | Optimized kernels SHALL preserve numerical equivalence with the reference implementation within documented tolerances. |
| REQ-3.17.1 | §3.17 | The library SHALL provide profiling tools capable of measuring: execution time, memory usage, FLOPs, routing statistics, refinement statistics, codebook utilization, cache utilization. |
| REQ-3.17.2 | §3.17 | Profiling SHALL be optional. |
| REQ-3.18.1 | §3.18 | Visualization tools SHALL support: refinement trees, routing paths, attention heatmaps, codebook utilization, execution timelines. |
| REQ-3.18.2 | §3.18 | Visualization SHALL remain independent of the core algorithm. |
| REQ-3.19.1 | §3.19 | The library SHALL include a benchmark suite. |
| REQ-3.19.2 | §3.19 | Benchmarks SHALL compare AVQA against: PyTorch SDPA, FlashAttention, xFormers, other relevant baselines. |
| REQ-3.19.3 | §3.19 | Benchmark outputs SHALL be reproducible. |
| REQ-3.20.1 | §3.20 | The implementation SHALL support serialization of: configurations, codebooks, routing state where applicable, trained parameters. |
| REQ-3.20.2 | §3.20 | Serialization SHALL remain versioned to preserve backward compatibility. |
| REQ-3.21.1 | §3.21 | The architecture SHALL permit user-defined implementations of: quantizers, routing strategies, merge strategies, schedulers, codebooks, execution backends. |
| REQ-3.21.2 | §3.21 | Extension mechanisms SHALL rely on documented interfaces rather than internal implementation details. |
| REQ-3.22.1 | §3.22 | The implementation SHALL define custom exception types. |
| REQ-3.22.2 | §3.22 | Errors SHALL provide sufficient context for debugging. |
| REQ-3.22.3 | §3.22 | Recoverable errors SHOULD produce informative messages without exposing internal implementation details. |
| REQ-3.23.1 | §3.23 | Every public class, function, and configuration object SHALL include: purpose, arguments, return values, tensor shapes, supported dtypes, supported devices, usage examples, references to relevant sections of the paper where applicable. |
| REQ-3.24.1 | §3.24 | Every public interface SHALL be covered by automated tests. |
| REQ-3.24.2 | §3.24 | Tests SHALL include: correctness, gradients, serialization, numerical stability, mixed precision, distributed execution where supported, regression tests for reported defects. |

## Chapter 4 — System Architecture

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-4.3.1 | §4.3 | Each layer has clearly defined responsibilities and SHALL NOT bypass adjacent layers. |
| REQ-4.5.1 | §4.5 | Each subsystem SHALL expose a narrow, well-defined public interface. |
| REQ-4.6.1 | §4.6 | The algorithm layer SHALL NOT import: vLLM, Hugging Face, FlashAttention, xFormers. |
| REQ-4.6.2 | §4.6 | Integration modules MAY depend on the core library; reverse dependency is prohibited. |
| REQ-4.6.3 | §4.6 | Backends SHALL implement abstract execution interfaces. |
| REQ-4.6.4 | §4.6 | Algorithm code SHALL NOT contain backend-specific logic. |
| REQ-4.6.5 | §4.6 | Profiling SHALL observe execution. It SHALL NOT modify execution. |
| REQ-4.7.1 | §4.7 | Attention SHALL NOT perform quantization internally. |
| REQ-4.7.2 | §4.7 | The quantizer SHALL NOT compute attention. |
| REQ-4.7.3 | §4.7 | The codebook SHALL remain independent of routing policy. |
| REQ-4.7.4 | §4.7 | The router SHALL NOT expand nodes. |
| REQ-4.7.5 | §4.7 | Refinement SHALL NOT select routing candidates. |
| REQ-4.7.6 | §4.7 | Merge SHALL NOT modify routing decisions. |
| REQ-4.7.7 | §4.7 | Scheduling SHALL remain independent of backend implementation. |
| REQ-4.8.1 | §4.8 | Communication between subsystems SHALL occur through immutable tensors or immutable configuration objects wherever practical. |
| REQ-4.10.1 | §4.10 | Backend implementations SHALL satisfy a common interface. |
| REQ-4.10.2 | §4.10 | Algorithm code SHALL never branch on hardware details. |
| REQ-4.11.1 | §4.11 | Framework adapters SHALL contain no algorithmic logic. |
| REQ-4.12.1 | §4.12 | The architecture SHALL define explicit extension interfaces for: Quantizer, Codebook, Router, Scheduler, Merge Strategy, Backend, Profiler. |
| REQ-4.12.2 | §4.12 | New implementations SHALL be registerable without modifying existing source files. |
| REQ-4.13.1 | §4.13 | Execution mode SHALL be selected through configuration rather than conditional logic scattered throughout the codebase. |
| REQ-4.14.1 | §4.14 | No circular dependencies. |
| REQ-4.14.2 | §4.14 | No hidden global state. |
| REQ-4.14.3 | §4.14 | No framework-specific logic in the core algorithm. |
| REQ-4.14.4 | §4.14 | No backend-specific logic in mathematical components. |
| REQ-4.14.5 | §4.14 | No mutable configuration objects. |
| REQ-4.14.6 | §4.14 | No side effects during forward execution beyond documented cache updates. |
| REQ-4.14.7 | §4.14 | All public APIs must remain backward compatible within a major version. |

## Chapter 5 — Public API & Interface Specification

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-5.3.1 | §5.3 | The public namespace SHALL consist of the documented top-level modules. |
| REQ-5.4.1 | §5.4 | Public modules SHALL be importable directly. |
| REQ-5.4.2 | §5.4 | Users SHALL NOT import modules from private namespaces. |
| REQ-5.5.1 | §5.5 | The initial stable API SHALL include: AVQAttention, AVQConfig, VectorQuantizer, HierarchicalCodebook, Router, AdaptiveRefinement, Scheduler, KVCache, Backend, Profiler. |
| REQ-5.6.1 | §5.6 | `AVQAttention` SHALL inherit from `torch.nn.Module`. |
| REQ-5.6.2 | §5.6 | The constructor SHALL accept a single `AVQConfig` object rather than a large collection of keyword arguments. |
| REQ-5.7.1 | §5.7 | Functional APIs SHALL NOT retain internal state. |
| REQ-5.8.1 | §5.8 | All configuration SHALL be represented by immutable dataclasses. |
| REQ-5.8.2 | §5.8 | Configuration objects SHALL validate parameters, support serialization, support equality comparison, support versioning. |
| REQ-5.8.3 | §5.8 | Configurations SHALL be reusable across multiple modules. |
| REQ-5.9.1 | §5.9 | Execution backends SHALL implement a common interface. |
| REQ-5.9.2 | §5.9 | Every backend SHALL provide methods for: attention computation, quantization operations, refinement, merge, reductions, cache operations. |
| REQ-5.10.1 | §5.10 | AVQA SHALL include a registry mechanism for extensibility. |
| REQ-5.10.2 | §5.10 | The registry SHALL support: quantizers, routers, merge strategies, schedulers, backends, visualization plugins. |
| REQ-5.10.3 | §5.10 | Registration SHALL occur without modifying the core library. |
| REQ-5.12.1 | §5.12 | Every public object SHALL support serialization where applicable. |
| REQ-5.12.2 | §5.12 | Version metadata SHALL accompany serialized artifacts. |
| REQ-5.13.1 | §5.13 | The API SHALL expose a consistent exception hierarchy (AVQAError, ConfigurationError, BackendError, RoutingError, CodebookError). |
| REQ-5.13.2 | §5.13 | Public methods SHALL raise documented exceptions. |
| REQ-5.14.1 | §5.14 | The library SHALL integrate with Python's standard `logging` module. |
| REQ-5.14.2 | §5.14 | Users SHALL control verbosity through configuration. |
| REQ-5.14.3 | §5.14 | No library component SHALL print directly to stdout during normal operation. |
| REQ-5.15.1 | §5.15 | Profiling SHALL be optional and non-invasive. |
| REQ-5.15.2 | §5.15 | Profiling SHALL not alter algorithmic behavior. |
| REQ-5.16.1 | §5.16 | Visualization components SHALL be decoupled from the core library. |
| REQ-5.17.1 | §5.17 | Framework-specific integrations SHALL be accessed through dedicated adapters (HF, vLLM, xFormers). |
| REQ-5.17.2 | §5.17 | Adapters SHALL translate framework-specific abstractions into AVQA interfaces without embedding algorithmic logic. |
| REQ-5.18.1 | §5.18 | The core library SHALL remain closed to modification but open to extension. |
| REQ-5.19.1 | §5.19 | The public API SHALL follow Semantic Versioning. |
| REQ-5.19.2 | §5.19 | Minor releases SHALL remain backward compatible. |
| REQ-5.19.3 | §5.19 | Patch releases SHALL contain only bug fixes and documentation updates. |
| REQ-5.19.4 | §5.19 | Deprecated APIs SHALL remain available for at least one minor release before removal. |
| REQ-5.20.1 | §5.20 | Within a major version, public class names SHALL remain stable. |
| REQ-5.20.2 | §5.20 | Public function signatures SHALL remain stable. |
| REQ-5.20.3 | §5.20 | Configuration schemas SHALL remain backward compatible where practical. |
| REQ-5.20.4 | §5.20 | Documented behaviors SHALL not change unexpectedly. |

## Chapter 6 — Core Data Model & Tensor Specification

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-6.3.1 | §6.3 | Tensor notation symbols SHALL retain identical meanings across all documentation, code, tests, and benchmarks. |
| REQ-6.4.1 | §6.4 | Every tensor SHALL use descriptive names. |
| REQ-6.5.1 | §6.5 | Query, Key, Value SHALL have shape `[B, H, T, D]`. |
| REQ-6.5.2 | §6.5 | Codebook SHALL have shape `[H, C, D]`. |
| REQ-6.5.3 | §6.5 | Assignment matrix SHALL have shape `[B, H, T]`. |
| REQ-6.5.4 | §6.5 | Routing scores SHALL have shape `[B, H, C]`. |
| REQ-6.5.5 | §6.5 | Active codewords SHALL have shape `[B, H, K]`. |
| REQ-6.5.6 | §6.5 | Refined codebook SHALL have shape `[B, H, R, D]`. |
| REQ-6.5.7 | §6.5 | Refined attention scores SHALL have shape `[B, H, T, R]`. |
| REQ-6.5.8 | §6.5 | Final attention probabilities SHALL have shape `[B, H, T, C + R]`. |
| REQ-6.5.9 | §6.5 | Output SHALL have shape `[B, H, T, D]`. |
| REQ-6.8.1 | §6.8 | Tensors SHALL satisfy row-major layout, contiguous memory where practical, alignment suitable for vectorized GPU execution, compatibility with PyTorch contiguous tensors. |
| REQ-6.9.1 | §6.9 | Reference implementation SHALL support: float32, float16, bfloat16. |
| REQ-6.9.2 | §6.9 | Optional support: float64, FP8, INT8. |
| REQ-6.10.1 | §6.10 | Every tensor SHALL belong to exactly one device. |
| REQ-6.10.2 | §6.10 | Cross-device tensor movement SHALL occur explicitly. |
| REQ-6.10.3 | §6.10 | Implicit transfers are prohibited. |
| REQ-6.11.1 | §6.11 | Configuration, hierarchy topology, and public metadata SHALL be immutable during forward execution. |
| REQ-6.11.2 | §6.11 | Mutation SHALL be localized to the owning subsystem. |
| REQ-6.12.1 | §6.12 | Every public API SHALL validate: rank, dtype, device, shape compatibility, contiguity (where required). |
| REQ-6.12.2 | §6.12 | Validation MAY be disabled in optimized execution modes. |
| REQ-6.13.1 | §6.13 | Query, key, and value SHALL share batch size, head count, head dimension. |
| REQ-6.13.2 | §6.13 | Attention output SHALL preserve batch size, sequence length, embedding dimension. |
| REQ-6.13.3 | §6.13 | Violations SHALL raise documented exceptions. |
| REQ-6.15.1 | §6.15 | Each stage consumes immutable inputs and produces explicitly documented outputs. |
| REQ-6.16.1 | §6.16 | Future distributed implementations SHALL preserve the same logical tensor model. |
| REQ-6.17.1 | §6.17 | Serializable objects SHALL include: codebooks, configurations, learned parameters, cached statistics (optional). |
| REQ-6.17.2 | §6.17 | Transient tensors SHALL NOT be serialized. |
| REQ-6.18.1 | §6.18 | Every major tensor SHALL document: shape, rank, memory complexity, computational role, producer subsystem, consumer subsystem. |
| REQ-6.19.1 | §6.19 | Every subsystem SHALL publish explicit input and output contracts. |
| REQ-6.19.2 | §6.19 | For every public function, the documentation SHALL specify: input tensors, output tensors, shape, dtype, device, ownership, mutability, complexity. |

## Chapter 7 — Mathematical Specification

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-7.5.1 | §7.5 | Each key SHALL be assigned to its nearest codeword by Euclidean distance. |
| REQ-7.5.2 | §7.5 | Only keys SHALL be quantized. Queries and values SHALL remain unchanged. |
| REQ-7.7.1 | §7.7 | Attention SHALL be rewritten over codewords per §7.7 equation. |
| REQ-7.8.1 | §7.8 | Total codebook size SHALL equal M₀ × (1 + 𝒞). |
| REQ-7.8.2 | §7.8 | Children SHALL supplement rather than replace parents. |
| REQ-7.9.1 | §7.9 | Every parent codeword SHALL equal the arithmetic mean of its children. |
| REQ-7.10.1 | §7.10 | Importance SHALL be computed as w_j(I) = Σ A_ij · n_j / Z̄_i. |
| REQ-7.10.2 | §7.10 | Importance SHALL NOT require auxiliary importance networks. |
| REQ-7.11.1 | §7.11 | Adaptive refinement SHALL proceed: parent attention → importance → top-P selection → expansion → child attention → correction → output. |
| REQ-7.11.2 | §7.11 | Only P parents SHALL be refined. |
| REQ-7.12.1 | §7.12 | Parent logits SHALL be reconstructed as S_p = (1/𝒞) · Σ S_c. |
| REQ-7.12.2 | §7.12 | Reconstruction SHALL NOT require additional matrix multiplication. |
| REQ-7.13.1 | §7.13 | Correcting attention SHALL replace parent contributions with child contributions using ΔA = A_c - A_p. |
| REQ-7.14.1 | §7.14 | AVQ SHALL preserve FlashAttention's online-softmax algorithm. |
| REQ-7.14.2 | §7.14 | Global recomputation of softmax is prohibited. |
| REQ-7.15.1 | §7.15 | The running maximum SHALL ignore empty codewords. |
| REQ-7.15.2 | §7.15 | Empty codewords SHALL contribute neither values nor counts. |
| REQ-7.16.1 | §7.16 | Total complexity SHALL be 𝒪(N(M₀ + P𝒞)D). |
| REQ-7.16.2 | §7.16 | Complexity SHALL remain linear in sequence length for fixed hierarchy parameters. |
| REQ-7.17.1 | §7.17 | Hierarchy invariant: every parent equals the mean of its children. |
| REQ-7.17.2 | §7.17 | Assignment invariant: every key is assigned to exactly one parent; during refinement, to either parent or one of its children. |
| REQ-7.17.3 | §7.17 | Conservation invariant: Σ V̄_a = Σ V_j. |
| REQ-7.17.4 | §7.17 | Count invariant: Σ n_a = N. |
| REQ-7.17.5 | §7.17 | Attention invariant: result remains normalized after correction. |
| REQ-7.18.1 | §7.18 | AVQ SHALL introduce approximation exclusively through vector quantization. |
| REQ-7.19.1 | §7.19 | Computational complexity SHALL be linear in sequence length for fixed codebook parameters. |
| REQ-7.19.2 | §7.19 | Parent reconstruction SHALL be exact under the parent-child mean constraint. |
| REQ-7.19.3 | §7.19 | Correcting attention SHALL preserve intended refinement semantics without revisiting parent codewords. |
| REQ-7.19.4 | §7.19 | Online softmax SHALL maintain numerical stability while avoiding materialization of the full attention matrix. |

## Chapter 8 — Vector Quantization Engine

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-8.3.1 | §8.3 | AVQ SHALL replace the flat codebook with a two-level hierarchical codebook (parents and children). |
| REQ-8.3.2 | §8.3 | Parents SHALL be constrained to equal the arithmetic mean of their children. |
| REQ-8.4.1 | §8.4 | VQ Engine SHALL receive key and value tensors. |
| REQ-8.4.2 | §8.4 | VQ Engine SHALL assign each key to its nearest parent codeword. |
| REQ-8.4.3 | §8.4 | VQ Engine SHALL assign each key to the nearest child within that parent. |
| REQ-8.4.4 | §8.4 | VQ Engine SHALL accumulate value vectors. |
| REQ-8.4.5 | §8.4 | VQ Engine SHALL count assignments. |
| REQ-8.4.6 | §8.4 | VQ Engine SHALL produce: parent assignments, child assignments, parent value aggregates, child value aggregates, parent counts, child counts. |
| REQ-8.5.1 | §8.5 | Hierarchical assignment SHALL be two-stage (parent first, then child). |
| REQ-8.5.2 | §8.5 | Assignment complexity SHALL reduce from 𝒪(M_total) to 𝒪(M₀ + 𝒞) per key. |
| REQ-8.6.1 | §8.6 | For every codeword, the engine SHALL compute aggregated value vector and assignment count. |
| REQ-8.6.2 | §8.6 | Aggregated value and count SHALL be the direct inputs to the attention kernel. |
| REQ-8.6.3 | §8.6 | Individual keys SHALL NOT be required beyond aggregation for attention computation. |
| REQ-8.7.1 | §8.7 | Assignments and aggregation SHALL be fused into a single preprocessing kernel. |
| REQ-8.7.2 | §8.7 | The fused execution model SHALL be preserved in optimized backends. |
| REQ-8.8.1 | §8.8 | Tensor contracts SHALL remain stable across all supported execution backends. |
| REQ-8.9.1 | §8.9 | Codebook training SHALL follow EMA-based online k-means. |
| REQ-8.9.2 | §8.9 | Default EMA decay SHALL be 0.99. |
| REQ-8.9.3 | §8.9 | Default commitment loss weight SHALL be 0.25. |
| REQ-8.9.4 | §8.9 | After every update, parents SHALL be reprojected to satisfy the hierarchy constraint. |
| REQ-8.10.1 | §8.10 | Children SHALL be initialized near their parent using perturbation scale 0.1. |
| REQ-8.10.2 | §8.10 | Alternative initialization schemes MAY be implemented but SHALL preserve the parent-child mean constraint after projection. |
| REQ-8.11.1 | §8.11 | The reference backend SHALL implement the EMA approach for dead-code handling. |
| REQ-8.11.2 | §8.11 | Enhanced codebook maintenance MAY be exposed as optional training policies. |
| REQ-8.12.1 | §8.12 | Preprocessing complexity SHALL be 𝒪(N(M₀ + 𝒞)D). |
| REQ-8.13.1 | §8.13 | Algorithmic logic SHALL be separated from backend-specific kernels. |
| REQ-8.13.2 | §8.13 | Assignments SHALL be deterministic when deterministic execution is enabled. |
| REQ-8.13.3 | §8.13 | Tensor copies SHALL be minimized. |
| REQ-8.13.4 | §8.13 | Batched execution SHALL be supported across arbitrary batch sizes. |
| REQ-8.13.5 | §8.13 | FP32, FP16, and BF16 SHALL be supported. |
| REQ-8.13.6 | §8.13 | Both PyTorch reference and Triton implementation SHALL be provided. |
| REQ-8.13.7 | §8.13 | Profiling statistics SHALL include assignment counts, codebook utilization, aggregation time. |
| REQ-8.13.8 | §8.13 | Future quantization algorithms SHALL be registerable without modifying the core attention pipeline. |

## Chapter 9 — Adaptive Attention Algorithm

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-9.3.1 | §9.3 | The adaptive attention pipeline SHALL be: compressed representation → parent attention → importance → top-P selection → child expansion → child attention → correcting attention → final output. |
| REQ-9.4.1 | §9.4 | Stage 1 SHALL compute parent attention using online tiled attention (FlashAttention-style). |
| REQ-9.5.1 | §9.5 | Importance SHALL be derived from attention without auxiliary networks. |
| REQ-9.5.2 | §9.5 | Importance estimation SHALL reuse quantities already produced by parent attention. |
| REQ-9.6.1 | §9.6 | Top-P parent selection SHALL satisfy: deterministic ordering, stable tie handling, configurable budget, bounded computational cost. |
| REQ-9.6.2 | §9.6 | The reference implementation SHALL reproduce the selection procedure described in the paper. |
| REQ-9.7.1 | §9.7 | Selected parents SHALL be replaced by their children via: load child codewords, load child value aggregates, load child assignment counts, construct refined tile. |
| REQ-9.7.2 | §9.7 | Unselected parents SHALL remain unchanged. |
| REQ-9.8.1 | §9.8 | Attention SHALL be recomputed only for expanded children. |
| REQ-9.8.2 | §9.8 | No attention SHALL be recomputed for unselected parents. |
| REQ-9.9.1 | §9.9 | Correcting attention SHALL replace parent contributions with child contributions, NOT augment. |
| REQ-9.10.1 | §9.10 | Parent logits SHALL be reconstructed from child logits using the mean constraint. |
| REQ-9.10.2 | §9.10 | Reconstruction SHALL avoid recomputing parent-query matrix multiplications. |
| REQ-9.11.1 | §9.11 | Online softmax running max/numerator/denominator SHALL be maintained throughout refinement. |
| REQ-9.11.2 | §9.11 | Global recomputation of softmax is prohibited. |
| REQ-9.11.3 | §9.11 | Correcting attention SHALL update running accumulators incrementally. |
| REQ-9.12.1 | §9.12 | Empty codewords SHALL contribute zero values, zero counts, and SHALL be excluded from maximum-logit computation and normalization. |
| REQ-9.13.1 | §9.13 | Overall complexity SHALL be 𝒪(N(M₀ + P𝒞)D). |
| REQ-9.13.2 | §9.13 | Complexity SHALL remain linear in sequence length for fixed hierarchy parameters. |
| REQ-9.14.1 | §9.14 | Adaptive refinement SHALL preserve normalized attention probabilities, online-softmax stability, deterministic execution (when enabled), bounded floating-point error. |
| REQ-9.15.1 | §9.15 | The implementation SHALL separate logical refinement from backend execution. |
| REQ-9.15.2 | §9.15 | The implementation SHALL reuse intermediate computations. |
| REQ-9.15.3 | §9.15 | The implementation SHALL avoid recomputing parent attention. |
| REQ-9.15.4 | §9.15 | The implementation SHALL support batched, causal, and non-causal execution. |
| REQ-9.15.5 | §9.15 | The implementation SHALL support FP32, FP16, and BF16. |
| REQ-9.15.6 | §9.15 | The implementation SHALL expose refinement statistics for profiling. |
| REQ-9.15.7 | §9.15 | The implementation SHALL permit configurable refinement budgets. |

## Chapter 10 — Attention Execution Pipeline

| ID | Section | Requirement |
|----|---------|-------------|
| REQ-10.2.1 | §10.2 | A forward pass SHALL consist of two logical phases: VQ Precompute, Adaptive Attention Computation. |
| REQ-10.3.1 | §10.3 | The pipeline ordering SHALL be: projection → VQ precompute → parent attention → importance → selection → child attention → correction → reduction → output projection. |
| REQ-10.4.1 | §10.4 | The execution pipeline SHALL consist of: input, precompute, parent attention, importance, selection, child attention, correction, reduction, output. |
| REQ-10.4.2 | §10.4 | Every backend SHALL preserve this logical ordering. |
| REQ-10.5.1 | §10.5 | Input validation SHALL occur before execution begins. |
| REQ-10.5.2 | §10.5 | The engine SHALL verify: compatible tensor shapes, supported dtypes, compatible devices, valid refinement budget, initialized codebook. |
| REQ-10.6.1 | §10.6 | Precompute SHALL complete before attention computation begins. |
| REQ-10.6.2 | §10.6 | No attention computation SHALL depend on uninitialized aggregation statistics. |
| REQ-10.7.1 | §10.7 | Parent attention SHALL produce parent logits, parent probabilities, online softmax accumulators, importance statistics. |
| REQ-10.8.1 | §10.8 | Importance estimation SHALL execute immediately after parent attention. |
| REQ-10.8.2 | §10.8 | Implementation SHALL NOT introduce additional neural network modules or learned routing mechanisms. |
| REQ-10.9.1 | §10.9 | Selection SHALL produce an ordered refinement list consumed by the next stage. |
| REQ-10.9.2 | §10.9 | Selection SHALL produce: P highest-priority parents, deterministic tie resolution, no duplicate selections. |
| REQ-10.10.1 | §10.10 | Child attention SHALL process only the selected subset of parents. |
| REQ-10.11.1 | §10.11 | Correcting attention SHALL: remove coarse parent contribution, insert refined child contributions, update running softmax statistics, preserve normalization. |
| REQ-10.11.2 | §10.11 | Correcting attention SHALL execute without materializing the complete attention matrix. |
| REQ-10.12.1 | §10.12 | Reduction SHALL preserve the output tensor shape expected by standard Transformer attention modules. |
| REQ-10.13.1 | §10.13 | Output projection SHALL be identical to standard Transformer implementations. |
| REQ-10.13.2 | §10.13 | AVQA SHALL NOT alter the surrounding Transformer architecture. |
| REQ-10.14.1 | §10.14 | Ephemeral state SHALL be released at the end of the forward pass. |
| REQ-10.15.1 | §10.15 | Execution mode SHALL be selected through configuration. |
| REQ-10.16.1 | §10.16 | The engine SHALL detect and report: invalid tensor shapes, incompatible devices, unsupported dtypes, invalid refinement budgets, uninitialized codebooks, backend failures. |
| REQ-10.16.2 | §10.16 | Errors SHALL terminate execution before producing partial outputs. |
| REQ-10.19.1 | §10.19 | Ordering: Precompute SHALL complete before attention begins. |
| REQ-10.19.2 | §10.19 | Consistency: All attention computations SHALL use a single version of the codebook during a forward pass. |
| REQ-10.19.3 | §10.19 | Isolation: Execution SHALL NOT modify immutable configuration. |
| REQ-10.19.4 | §10.19 | Determinism: Reference mode SHALL produce deterministic outputs for identical inputs and seeds. |
| REQ-10.19.5 | §10.19 | Normalization: Final attention probabilities SHALL remain normalized after correcting attention. |
| REQ-10.20.1 | §10.20 | Backend implementations MAY fuse stages, reorder independent operations, optimize memory layouts, eliminate intermediate buffers, provided observable outputs remain equivalent, documented numerical tolerances are preserved, and pipeline invariants remain satisfied. |
| REQ-10.21.1 | §10.21 | Memory usage SHALL remain linear with respect to sequence length for fixed codebook parameters. |

## Total

- 0 entries (Chapter 2 — informational)
- 53 entries (Chapter 3)
- 28 entries (Chapter 4)
- 31 entries (Chapter 5)
- 25 entries (Chapter 6)
- 25 entries (Chapter 7)
- 26 entries (Chapter 8)
- 23 entries (Chapter 9)
- 27 entries (Chapter 10)

**Grand total: 238 normative requirements.**

Each requirement maps to at least one TODO entry in `TODO.md` and at least one
test in `tests/`. The mapping is maintained in `docs/spec_compliance.md`.
