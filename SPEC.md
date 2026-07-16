# Vision & Goals

## 1. Vision

AVQA (Adaptive Vector Quantized Attention) is an open-source, production-grade Python library implementing **Adaptive Vector Quantized Attention (AVQ-Attention)** as described in the paper:

**Adaptive Vector Quantized Attention (AVQ-Attention)**
https://arxiv.org/html/2607.12789v1

The project's primary objective is to transform the paper's research contribution into a robust, extensible, and high-performance software library that can serve as a drop-in attention backend for modern Transformer architectures.

Unlike a paper reproduction repository or a model-specific implementation, AVQA is intended to become reusable infrastructure for the broader machine learning ecosystem. The library should integrate naturally with existing frameworks, expose stable APIs, and remain maintainable as the efficient-attention landscape evolves.

The guiding philosophy is to separate the **algorithm** from the **framework**, allowing researchers to experiment with adaptive vector quantization while enabling practitioners to deploy the method with minimal engineering effort.

---

# 2. Mission

AVQA exists to solve four complementary problems.

## 2.1 Faithful Reproduction

The implementation must faithfully reproduce the AVQ-Attention algorithm described in the reference paper.

The paper is the canonical specification of the algorithm.

Every mathematical equation, algorithm, tensor transformation, routing strategy, and adaptive refinement step described in the paper should be reflected in the implementation whenever possible.

If implementation details are omitted by the paper, the chosen solution must:

- follow established engineering practices,
- remain mathematically consistent,
- be documented,
- be configurable,
- never silently alter the original algorithm.

---

## 2.2 Production Engineering

Research implementations are often optimized for demonstrating correctness rather than maintainability.

AVQA aims to bridge this gap by providing:

- clean architecture,
- stable public APIs,
- comprehensive documentation,
- extensive testing,
- profiling tools,
- benchmark suites,
- framework integrations,
- long-term maintainability.

The package should be suitable for academic research, industrial deployment, and open-source collaboration.

---

## 2.3 Ecosystem Integration

AVQA should integrate naturally with existing Transformer ecosystems instead of requiring users to adopt a new framework.

Supported integrations include:

- PyTorch
- Hugging Face Transformers
- vLLM
- xFormers
- FlashAttention-based models
- Custom Transformer implementations

Replacing an existing attention implementation should require minimal changes to application code.

---

## 2.4 Research Platform

Beyond reproducing the original paper, AVQA should provide a flexible foundation for future research.

Researchers should be able to replace or extend components such as:

- routing strategies,
- quantizers,
- codebook structures,
- merge policies,
- adaptive schedulers,
- refinement heuristics,
- backend kernels,

without modifying unrelated parts of the library.

The architecture should encourage experimentation while preserving a stable production API.

---

# 3. Project Scope

The scope of AVQA includes:

- implementation of Adaptive Vector Quantized Attention,
- PyTorch modules,
- optional Triton acceleration,
- framework integrations,
- benchmarking tools,
- visualization utilities,
- profiling,
- documentation,
- testing,
- packaging,
- release engineering.

The project does **not** aim to become:

- a complete deep learning framework,
- a model zoo,
- a training library,
- an optimization toolkit unrelated to AVQ,
- a replacement for PyTorch or vLLM.

AVQA focuses exclusively on efficient attention infrastructure.

---

# 4. Guiding Principles

Every engineering decision should align with the following principles.

## 4.1 Correctness Before Optimization

Algorithmic correctness always takes precedence over performance.

The reference PyTorch implementation serves as the source of truth.

Optimized Triton or CUDA kernels must produce numerically equivalent results within acceptable floating-point tolerances.

No optimization may change the semantics of the algorithm.

---

## 4.2 Paper Fidelity

The implementation should remain as close as possible to the published algorithm.

Enhancements, optimizations, or experimental ideas should be:

- optional,
- clearly documented,
- disabled by default,
- isolated from the reference implementation.

The library should distinguish between:

- paper-faithful behavior,
- engineering optimizations,
- experimental features.

---

## 4.3 Modular Design

Every major subsystem should be independently replaceable.

Examples include:

- quantizers,
- routing algorithms,
- codebooks,
- refinement strategies,
- merge algorithms,
- backend kernels,
- schedulers.

Modules should communicate through well-defined interfaces rather than implementation details.

---

## 4.4 Explicit Configuration

Library behavior must never depend on hidden global state.

All configurable behavior should be expressed through explicit configuration objects.

Reasonable defaults should be provided, but users must retain full control over algorithmic parameters.

---

## 4.5 Progressive Optimization

Performance optimization should proceed incrementally.

Recommended implementation order:

1. Reference PyTorch implementation.
2. Functional correctness tests.
3. Numerical validation.
4. Performance profiling.
5. Triton optimization.
6. Kernel fusion.
7. Advanced scheduling.
8. Multi-GPU optimization.

Every optimization stage must preserve correctness.

---

## 4.6 Transparency

The implementation should be understandable.

Complex algorithms should include:

- mathematical references,
- tensor shape annotations,
- complexity analysis,
- implementation rationale,
- links to relevant paper sections.

Future contributors should be able to understand design decisions without reverse-engineering the code.

---

## 4.7 Extensibility

Future algorithms should be implementable without modifying the core library.

Extension points should exist for:

- routing,
- quantization,
- refinement,
- scheduling,
- backend execution,
- profiling,
- visualization.

Composition is preferred over inheritance wherever practical.

---

# 5. Design Objectives

The following objectives guide architectural decisions.

## Functional Objectives

- Implement AVQ-Attention faithfully.
- Support inference and training.
- Support autoregressive decoding.
- Support KV caching.
- Support mixed precision.
- Support long-context inference.
- Support batched execution.
- Support multiple hardware backends.

---

## Software Engineering Objectives

- Maintain a clean, layered architecture.
- Minimize coupling between modules.
- Maximize testability.
- Preserve backward compatibility across minor releases.
- Expose a stable public API.
- Avoid unnecessary dependencies.

---

## Performance Objectives

The implementation should improve computational efficiency while preserving model quality.

Performance targets should be validated empirically rather than assumed.

Benchmarking should compare AVQA against established attention implementations across multiple sequence lengths, model sizes, hardware platforms, and precision modes.

Optimization efforts should prioritize measurable improvements in latency, throughput, and memory usage.

---

## Usability Objectives

Users should be able to:

- install the package with a standard Python package manager,
- replace an existing attention layer with minimal code changes,
- select execution backends through configuration,
- profile runtime behavior,
- visualize adaptive refinement,
- benchmark different configurations,
- save and reload configurations consistently.

The library should follow familiar conventions used by PyTorch and Hugging Face wherever appropriate.

---

# 6. Target Users

AVQA is designed for several categories of users.

### Machine Learning Researchers

Researchers developing efficient attention mechanisms, adaptive routing algorithms, vector quantization methods, or long-context Transformer architectures.

### Framework Developers

Developers integrating efficient attention into inference engines, serving frameworks, or distributed execution systems.

### Model Developers

Practitioners building Transformer models who require an alternative attention backend without redesigning model architectures.

### Infrastructure Engineers

Engineers deploying large language models who require improved memory efficiency and scalable attention implementations.

### Students and Educators

Individuals studying efficient Transformer architectures who benefit from a clear, well-documented reference implementation.

---

# 7. Non-Goals

The following are explicitly outside the scope of AVQA.

- Developing new Transformer architectures unrelated to adaptive vector quantization.
- Maintaining pretrained language models.
- Providing datasets or data-processing pipelines.
- Building distributed training frameworks.
- Replacing existing inference engines.
- Supporting every historical Transformer implementation.
- Optimizing unrelated neural network layers.

Restricting project scope helps maintain a focused, maintainable codebase.

---

# 8. Success Criteria

AVQA will be considered successful when it satisfies the following criteria.

## Algorithmic Success

- The implementation faithfully reproduces the AVQ-Attention algorithm described in the paper.
- Numerical outputs are validated against the reference implementation.

## Engineering Success

- Stable public APIs.
- Comprehensive documentation.
- High automated test coverage.
- Reliable packaging and release process.

## Integration Success

- Seamless integration with supported frameworks.
- Minimal effort required to replace existing attention implementations.

## Performance Success

- Demonstrated improvements in memory efficiency and/or throughput under appropriate workloads.
- Performance characteristics documented through reproducible benchmarks rather than anecdotal claims.

## Community Success

- Clear contribution guidelines.
- Extensible architecture.
- Transparent development process.
- Sustainable maintenance practices.

---

# 9. Long-Term Vision

AVQA aims to become the reference implementation of Adaptive Vector Quantized Attention for the Python ecosystem.

In the near term, the project focuses on faithful implementation and framework integration.

Over time, the library should evolve into a research and production platform supporting new quantization strategies, adaptive routing algorithms, hardware backends, and efficient-attention techniques while preserving compatibility with the original AVQ-Attention formulation.

Every enhancement should strengthen the library's role as reusable infrastructure rather than divert it into a collection of unrelated experimental features.

# Chapter 2 — Paper Review & Mathematical Foundations

## 2.1 Purpose

This chapter establishes the mathematical and algorithmic foundation for AVQA. It defines the concepts, terminology, and implementation expectations derived from the reference paper before any software architecture is introduced.

**Reference Paper**

Adaptive Vector Quantized Attention (AVQ-Attention)

https://arxiv.org/html/2607.12789v1

The paper is the canonical specification for the AVQ-Attention algorithm. This document specifies how that algorithm should be translated into a production-quality software implementation.

Where the paper provides explicit mathematical definitions, those definitions take precedence over implementation convenience. Where implementation details are omitted, AVQA adopts conservative engineering decisions that preserve the behavior and intent of the published method.

---

# 2.2 Problem Statement

Modern Transformer models rely on scaled dot-product attention to model interactions between tokens.

Given

- Queries **Q**
- Keys **K**
- Values **V**

standard self-attention computes pairwise interactions between every query and every key.

For a sequence length **N**, the computational complexity grows quadratically.

As sequence lengths increase into tens or hundreds of thousands of tokens, quadratic attention becomes the dominant computational bottleneck.

This bottleneck affects

- inference latency,
- GPU memory,
- training cost,
- throughput,
- deployment scalability.

The objective of AVQ-Attention is to reduce the computational cost of attention while preserving model quality.

---

# 2.3 Existing Approaches

Numerous approaches have attempted to improve attention efficiency.

Examples include

- Sparse Attention
- Local Attention
- Sliding Window Attention
- Linear Attention
- Performer
- Nyström Attention
- Multi-Query Attention (MQA)
- Grouped Query Attention (GQA)
- FlashAttention
- Vector Quantized Attention

Each approach reduces computational cost through different approximations or implementation optimizations.

FlashAttention primarily improves memory efficiency through kernel fusion while preserving exact attention.

Vector Quantized Attention instead reduces the number of key representations participating in attention.

AVQ builds upon the latter approach.

---

# 2.4 Vector Quantized Attention

Vector Quantized Attention compresses the key space into a finite set of representative vectors called **codewords**.

Instead of attending over every key individually,

```
Q × K
```

attention becomes

```
Q × C
```

where **C** is a learned codebook.

Each key is assigned to a codeword.

Multiple keys may share the same representative.

The approximation reduces computational cost because

```
|C| << |K|
```

The approximation quality depends on

- codebook quality,
- assignment strategy,
- aggregation method.

---

# 2.5 Limitation of Static Codebooks

Static vector quantization allocates equal representational capacity across the entire key space.

However, attention distributions are rarely uniform.

In practice,

- a small number of regions receive most of the attention mass,
- many codewords receive almost no attention.

Consequently,

computational effort is wasted representing regions that contribute little to the final output while important regions remain underrepresented.

This imbalance motivates adaptive refinement.

---

# 2.6 Core Idea of AVQ

AVQ introduces **adaptive refinement**.

Instead of using a fixed-resolution codebook,

the algorithm dynamically increases representational capacity only where attention indicates it is beneficial.

The process follows four stages.

1. Compute attention over a coarse codebook.
2. Identify highly attended codewords.
3. Expand only those codewords into finer representations.
4. Recompute attention over the refined subset.

Regions receiving little attention remain represented at coarse resolution.

This adaptive allocation allows computational resources to focus on the most informative regions.

---

# 2.7 Hierarchical Representation

The adaptive refinement process requires a hierarchical codebook.

Each codeword may possess one or more child codewords representing finer partitions of the embedding space.

Conceptually,

```
Root

├── Node A
│   ├── A1
│   ├── A2
│
├── Node B
│   ├── B1
│   ├── B2
```

Traversal through the hierarchy is determined dynamically during inference.

The implementation should support arbitrary tree depth and configurable branching factors.

---

# 2.8 Adaptive Refinement

Adaptive refinement is the defining feature of AVQ.

The implementation shall follow the algorithm described in the paper.

Conceptually,

1. Build an initial coarse attention distribution.
2. Estimate which codewords are most important.
3. Expand only those codewords.
4. Replace coarse representations with refined descendants.
5. Produce the final attention distribution.

The refinement policy must remain configurable.

Possible policies include

- Top-k
- Threshold
- Budget-based
- Entropy-based

The default implementation shall match the behavior described in the reference paper.

---

# 2.9 Computational Complexity

The implementation should document the computational complexity of every major operation.

Examples include

- codebook assignment,
- routing,
- refinement,
- probability merging,
- cache updates.

Complexity analysis should distinguish between

- theoretical complexity,
- implementation complexity,
- practical GPU behavior.

Where optimizations alter constant factors without changing asymptotic complexity, this distinction should be documented.

---

# 2.10 Numerical Considerations

Efficient attention algorithms introduce additional numerical considerations beyond standard attention.

The implementation should document and test

- softmax stability,
- floating-point precision,
- accumulation order,
- mixed precision behavior,
- BF16 support,
- FP16 support,
- overflow handling,
- underflow handling.

Reference implementations should prioritize correctness over optimization.

---

# 2.11 Mapping Mathematics to Code

Every mathematical construct described in the paper shall correspond to a clearly identifiable software component.

For example:

| Mathematical Concept  | Implementation Component |
| --------------------- | ------------------------ |
| Codebook              | `Codebook`               |
| Quantizer             | `VectorQuantizer`        |
| Routing               | `Router`                 |
| Adaptive refinement   | `AdaptiveRefinement`     |
| Attention computation | `AVQAttention`           |
| Probability merge     | `MergeStrategy`          |
| Scheduling policy     | `Scheduler`              |

This mapping should remain consistent throughout the codebase and documentation.

---

# 2.12 Paper Fidelity Policy

The implementation distinguishes three categories of behavior:

**Reference Behavior**

Implements the paper as faithfully as possible.

This mode is the default and serves as the basis for correctness testing.

**Optimized Behavior**

Introduces implementation optimizations that preserve algorithmic semantics while improving performance.

Examples include Triton kernels, fused operations, and optimized memory layouts.

**Experimental Behavior**

Introduces new algorithms, heuristics, or research ideas that extend beyond the published paper.

Experimental features must never replace the reference implementation by default and should be clearly identified in the documentation.

---

# 2.13 Design Principles Derived from the Paper

The mathematical structure of AVQ leads to several architectural principles.

- Separate quantization from attention.
- Separate routing from refinement.
- Treat the codebook as an independent data structure.
- Make refinement policies replaceable.
- Isolate hardware-specific optimizations from algorithmic logic.
- Preserve determinism wherever practical.
- Keep paper-faithful behavior available regardless of backend.

These principles form the basis for the software architecture described in subsequent chapters.

# Chapter 3 — Functional Requirements

## 3.1 Purpose

This chapter defines the functional capabilities that AVQA shall provide. These requirements describe the observable behavior of the library without prescribing implementation details.

The requirements in this chapter are normative. Unless explicitly marked as optional, all implementations claiming compliance with the AVQA specification shall satisfy these requirements.

The keywords **MUST**, **SHALL**, **SHOULD**, **MAY**, and **OPTIONAL** are interpreted according to RFC 2119.

---

# 3.2 Primary Functional Objectives

The library SHALL:

- Implement Adaptive Vector Quantized Attention as described in the reference paper.
- Expose the algorithm through a reusable Python package.
- Operate as a drop-in attention backend.
- Support both inference and training.
- Support CPU and GPU execution.
- Provide a pure PyTorch reference implementation.
- Provide optional Triton acceleration.
- Support mixed precision execution.
- Maintain deterministic execution when deterministic mode is enabled.

---

# 3.3 Public Library Requirements

The package SHALL expose a clean public API.

Users SHALL NOT be required to modify internal implementation code.

The package SHALL support installation through standard Python package managers.

Example:

```python
from avqa import AVQAttention

attention = AVQAttention(...)
```

Internal modules SHALL remain hidden unless explicitly documented.

---

# 3.4 Attention Module Requirements

The primary attention module SHALL:

- inherit from `torch.nn.Module`,
- support batched execution,
- support multi-head attention,
- support causal attention,
- support bidirectional attention,
- support masking,
- support dropout,
- support arbitrary sequence lengths,
- support configurable embedding dimensions,
- support configurable numbers of attention heads.

The public interface SHOULD closely resemble existing PyTorch attention modules.

---

# 3.5 Functional API

In addition to the module interface, the library SHALL provide a functional API.

Example:

```python
output = avqa.functional.attention(
    query,
    key,
    value,
    config=config,
)
```

The functional interface SHALL remain stateless.

---

# 3.6 Configuration Requirements

All configurable behavior SHALL be controlled through explicit configuration objects.

Configuration SHALL include, at minimum:

- codebook size,
- branching factor,
- refinement budget,
- routing strategy,
- merge strategy,
- backend,
- execution mode,
- precision,
- cache configuration.

Configuration SHALL support serialization.

Configuration SHALL support validation.

Configuration SHALL be immutable after construction unless explicitly documented.

---

# 3.7 Vector Quantization Requirements

The implementation SHALL provide a vector quantizer capable of:

- assigning vectors to codewords,
- computing assignments in batches,
- updating codebooks during training,
- supporting inference without retraining,
- supporting configurable codebook sizes.

Optional capabilities MAY include:

- EMA updates,
- FAISS acceleration,
- k-means initialization,
- custom distance metrics.

---

# 3.8 Codebook Requirements

The implementation SHALL provide a hierarchical codebook supporting:

- parent-child relationships,
- configurable tree depth,
- configurable branching factor,
- efficient traversal,
- serialization,
- statistics collection.

The codebook SHALL expose sufficient information for routing, visualization, and debugging.

---

# 3.9 Adaptive Refinement Requirements

Adaptive refinement SHALL support:

- coarse attention,
- active node selection,
- node expansion,
- refined attention,
- probability merging.

Selection policies SHALL be configurable.

The implementation SHALL expose a public interface for implementing additional refinement policies.

---

# 3.10 Routing Requirements

Routing SHALL be implemented as an independent subsystem.

Responsibilities include:

- determining active codewords,
- collecting routing statistics,
- maintaining utilization metrics,
- exposing debugging information.

Routing SHALL NOT perform attention computations directly.

---

# 3.11 Merge Strategy Requirements

The implementation SHALL separate refinement from probability merging.

Supported merge strategies SHOULD include:

- probability merge,
- weighted merge,
- logit merge,
- normalized merge.

Users SHALL be able to register custom merge strategies.

---

# 3.12 Backend Requirements

The implementation SHALL support multiple execution backends.

Initially supported backends include:

- PyTorch,
- Triton.

Future backends SHOULD be addable without modifying algorithmic code.

Backend selection SHALL occur through configuration.

Automatic backend selection MAY be supported.

---

# 3.13 KV Cache Requirements

The implementation SHALL support autoregressive decoding.

The cache SHALL support:

- incremental updates,
- efficient lookup,
- configurable storage,
- cache reset,
- serialization.

The implementation SHALL expose a stable cache interface independent of any specific inference engine.

---

# 3.14 Hugging Face Integration

The library SHALL support Hugging Face Transformers.

Users SHOULD be able to replace compatible attention layers with AVQA using a documented helper function.

The integration SHALL preserve:

- pretrained weights,
- model configuration,
- inference behavior,
- training compatibility.

---

# 3.15 vLLM Integration

The library SHALL integrate with vLLM through documented extension points whenever possible.

The integration SHALL support:

- paged attention,
- continuous batching,
- prefix caching,
- tensor parallelism where supported,
- speculative decoding where compatible.

Framework-specific code SHALL remain isolated from the core algorithm.

---

# 3.16 FlashAttention Compatibility

Where FlashAttention is available, the implementation SHALL support interoperability.

The backend selection order SHOULD be configurable.

Optimized kernels SHALL preserve numerical equivalence with the reference implementation within documented tolerances.

---

# 3.17 Profiling Requirements

The library SHALL provide profiling tools capable of measuring:

- execution time,
- memory usage,
- FLOPs,
- routing statistics,
- refinement statistics,
- codebook utilization,
- cache utilization.

Profiling SHALL be optional.

---

# 3.18 Visualization Requirements

Visualization tools SHALL support:

- refinement trees,
- routing paths,
- attention heatmaps,
- codebook utilization,
- execution timelines.

Visualization SHALL remain independent of the core algorithm.

---

# 3.19 Benchmarking Requirements

The library SHALL include a benchmark suite.

Benchmarks SHALL compare AVQA against:

- PyTorch SDPA,
- FlashAttention,
- xFormers,
- other relevant baselines.

Benchmark outputs SHALL be reproducible.

---

# 3.20 Serialization Requirements

The implementation SHALL support serialization of:

- configurations,
- codebooks,
- routing state where applicable,
- trained parameters.

Serialization SHALL remain versioned to preserve backward compatibility.

---

# 3.21 Extension Requirements

The architecture SHALL permit user-defined implementations of:

- quantizers,
- routing strategies,
- merge strategies,
- schedulers,
- codebooks,
- execution backends.

Extension mechanisms SHALL rely on documented interfaces rather than internal implementation details.

---

# 3.22 Error Handling Requirements

The implementation SHALL define custom exception types.

Errors SHALL provide sufficient context for debugging.

Recoverable errors SHOULD produce informative messages without exposing internal implementation details.

---

# 3.23 Documentation Requirements

Every public class, function, and configuration object SHALL include:

- purpose,
- arguments,
- return values,
- tensor shapes,
- supported dtypes,
- supported devices,
- usage examples,
- references to relevant sections of the paper where applicable.

---

# 3.24 Testing Requirements

Every public interface SHALL be covered by automated tests.

Tests SHALL include:

- correctness,
- gradients,
- serialization,
- numerical stability,
- mixed precision,
- distributed execution where supported,
- regression tests for reported defects.

---

# 3.25 Acceptance Criteria

An implementation satisfies the functional requirements of AVQA when:

1. The reference implementation reproduces the AVQ-Attention algorithm as specified by the paper.
2. All required public APIs are implemented and documented.
3. Supported integrations function as documented.
4. Automated tests pass across supported platforms.
5. Configuration, serialization, profiling, and benchmarking operate consistently across supported backends.

Compliance with these functional requirements is necessary before performance optimizations, hardware-specific kernels, or experimental extensions are considered complete.

# Chapter 4 — System Architecture

## 4.1 Purpose

This chapter defines the architectural blueprint of AVQA. It describes how the system is decomposed into independent subsystems, the responsibilities of each layer, the flow of data through the attention pipeline, and the architectural constraints that govern future development.

The architecture is designed around three guiding principles:

1. **Algorithmic correctness is independent of hardware optimization.**
2. **Framework integrations are independent of the core implementation.**
3. **Every major subsystem can evolve without requiring changes to unrelated components.**

This separation ensures that AVQA remains maintainable, extensible, and suitable for both research and production deployment.

---

# 4.2 Architectural Philosophy

AVQA is not a model implementation. It is an **attention infrastructure library**.

The project intentionally separates:

- mathematical algorithms,
- execution backends,
- framework integrations,
- configuration,
- profiling,
- visualization.

Each layer communicates only through stable interfaces.

No component should depend on the internal implementation of another layer.

Instead, components interact through abstract contracts and immutable data structures.

---

# 4.3 Layered Architecture

The library is organized into seven logical layers.

```text
┌────────────────────────────────────────────┐
│          Applications / User Code          │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│     Framework Integrations (HF, vLLM)      │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│          Public AVQA API Layer             │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│        Core Attention Pipeline             │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│ Quantization • Routing • Refinement        │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│ Backend Abstraction (Torch / Triton)       │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│           Hardware (CPU / GPU)             │
└────────────────────────────────────────────┘
```

Each layer has clearly defined responsibilities and SHALL NOT bypass adjacent layers.

---

# 4.4 High-Level Execution Flow

The complete attention pipeline consists of the following stages:

```text
Input Tensors
      │
      ▼
Projection (Q, K, V)
      │
      ▼
Vector Quantization
      │
      ▼
Hierarchical Codebook Lookup
      │
      ▼
Coarse Attention
      │
      ▼
Adaptive Routing
      │
      ▼
Node Refinement
      │
      ▼
Probability Merge
      │
      ▼
Weighted Value Aggregation
      │
      ▼
Output Projection
```

Each stage is implemented as an independent subsystem with a documented interface.

---

# 4.5 Subsystem Overview

The core library consists of the following subsystems:

| Subsystem     | Responsibility                           |
| ------------- | ---------------------------------------- |
| Attention     | High-level orchestration                 |
| Quantizer     | Assign keys to codewords                 |
| Codebook      | Hierarchical representation              |
| Router        | Identify refinement candidates           |
| Refinement    | Expand active codewords                  |
| Merge         | Combine coarse and refined probabilities |
| Scheduler     | Allocate refinement budget               |
| Backend       | Dispatch tensor operations               |
| Cache         | Manage KV cache                          |
| Profiling     | Collect runtime metrics                  |
| Visualization | Debugging and analysis                   |
| Integration   | Connect external frameworks              |

Each subsystem SHALL expose a narrow, well-defined public interface.

---

# 4.6 Dependency Rules

The following dependency rules are mandatory.

### Core algorithm

The algorithm layer SHALL NOT import:

- vLLM
- Hugging Face
- FlashAttention
- xFormers

### Integrations

Integration modules MAY depend on the core library.

The reverse dependency is prohibited.

### Backends

Backends SHALL implement abstract execution interfaces.

Algorithm code SHALL NOT contain backend-specific logic.

### Profiling

Profiling SHALL observe execution.

It SHALL NOT modify execution.

---

# 4.7 Core Pipeline Responsibilities

## Attention

Responsible for:

- orchestrating execution,
- validating inputs,
- invoking subsystems,
- returning outputs.

Attention SHALL NOT perform quantization internally.

---

## Quantizer

Responsible for:

- nearest-neighbor assignment,
- codebook lookup,
- training updates,
- assignment statistics.

The quantizer SHALL NOT compute attention.

---

## Codebook

Responsible for:

- hierarchical storage,
- parent-child relationships,
- traversal,
- serialization.

The codebook SHALL remain independent of routing policy.

---

## Router

Responsible for:

- active node selection,
- utilization tracking,
- routing statistics.

The router SHALL NOT expand nodes.

---

## Refinement

Responsible for:

- expanding selected codewords,
- constructing refined representations,
- preserving hierarchy consistency.

Refinement SHALL NOT select routing candidates.

---

## Merge

Responsible for:

- combining refined and coarse attention,
- normalization,
- numerical stability.

Merge SHALL NOT modify routing decisions.

---

## Scheduler

Responsible for:

- refinement budget allocation,
- adaptive scheduling,
- policy execution.

Scheduling SHALL remain independent of backend implementation.

---

# 4.8 Data Ownership

Each subsystem owns its internal state.

Shared mutable state SHALL be avoided.

Ownership example:

```text
Codebook
    │
owns
    ▼
Hierarchy

Router
    │
owns
    ▼
Routing statistics

Cache
    │
owns
    ▼
KV storage

Profiler
    │
owns
    ▼
Execution metrics
```

Communication between subsystems SHALL occur through immutable tensors or immutable configuration objects wherever practical.

---

# 4.9 Control Flow

Execution follows a single-direction control flow.

```text
Application
      │
      ▼
AVQAttention
      │
      ▼
Quantizer
      │
      ▼
Router
      │
      ▼
Refinement
      │
      ▼
Merge
      │
      ▼
Backend
      │
      ▼
Output
```

No subsystem may invoke higher architectural layers.

This prevents cyclic dependencies.

---

# 4.10 Backend Abstraction

Backend implementations SHALL satisfy a common interface.

Supported backends:

- Torch
- Triton

Future backends:

- CUDA
- ROCm
- Metal
- oneAPI

Algorithm code SHALL never branch on hardware details.

Backend dispatch is responsible for selecting optimized implementations.

---

# 4.11 Framework Integration Layer

Framework adapters translate external model implementations into AVQA abstractions.

Responsibilities include:

- replacing attention layers,
- mapping configuration,
- preserving pretrained weights,
- managing framework-specific caches.

Framework adapters SHALL contain no algorithmic logic.

---

# 4.12 Extension Points

The architecture SHALL define explicit extension interfaces for:

- Quantizer
- Codebook
- Router
- Scheduler
- Merge Strategy
- Backend
- Profiler

New implementations SHALL be registerable without modifying existing source files.

Composition and dependency injection are preferred over inheritance.

---

# 4.13 Execution Modes

The architecture supports multiple execution modes.

### Reference Mode

- Pure PyTorch
- Fully deterministic
- Maximum readability
- Used for testing and validation

### Optimized Mode

- Triton kernels
- Mixed precision
- Kernel fusion
- Production inference

### Research Mode

- Verbose diagnostics
- Experimental algorithms
- Profiling enabled
- Visualization hooks

Execution mode SHALL be selected through configuration rather than conditional logic scattered throughout the codebase.

---

# 4.14 Architectural Constraints

The following constraints are mandatory:

- No circular dependencies.
- No hidden global state.
- No framework-specific logic in the core algorithm.
- No backend-specific logic in mathematical components.
- No mutable configuration objects.
- No side effects during forward execution beyond documented cache updates.
- All public APIs must remain backward compatible within a major version.

---

# 4.15 Design Patterns

AVQA adopts several architectural patterns to improve modularity and maintainability.

| Pattern              | Purpose                                                  |
| -------------------- | -------------------------------------------------------- |
| Strategy             | Routing, merge, scheduling, quantization                 |
| Factory              | Backend and integration creation                         |
| Registry             | Dynamic component discovery                              |
| Adapter              | Framework integrations                                   |
| Composition          | Build attention pipeline from interchangeable components |
| Dependency Injection | Decouple implementations from interfaces                 |
| Builder              | Configuration construction                               |
| Observer             | Profiling and visualization hooks                        |

These patterns reduce coupling, simplify testing, and enable future extensions without modifying existing components.

---

# 4.16 Architecture Validation

A compliant implementation SHALL satisfy the following conditions:

- Each subsystem has a single, clearly defined responsibility.
- Algorithmic components are independent of execution backends.
- Framework integrations are isolated from the core library.
- Backend implementations are interchangeable through a common interface.
- New routing, quantization, and refinement strategies can be added without altering the core attention pipeline.
- The architecture remains testable, maintainable, and extensible as the project evolves.

This architecture forms the foundation for all subsequent implementation chapters. Every class, module, interface, and integration described later in this specification SHALL conform to the principles and constraints established here.

# Chapter 5 — Public API & Interface Specification

## 5.1 Purpose

The Public API defines the stable interface between AVQA and its consumers.

The API is the primary contract of the library. Internal implementations may evolve, but the public interfaces defined in this chapter SHALL remain stable within a major release.

The API is designed around three principles:

1. **Consistency** — similar concepts expose similar interfaces.
2. **Composability** — components can be combined without hidden dependencies.
3. **Stability** — user-facing APIs change infrequently and predictably.

---

# 5.2 API Design Philosophy

The AVQA API SHALL:

- Follow established PyTorch conventions.
- Minimize cognitive overhead.
- Prefer explicit configuration over implicit behavior.
- Preserve backward compatibility.
- Expose only concepts users need.
- Hide implementation details.

Users should rarely interact with low-level components unless performing research or extending the framework.

---

# 5.3 Package Structure

The public namespace SHALL consist of the following top-level modules:

```text
avqa
├── attention
├── functional
├── config
├── backend
├── cache
├── profiling
├── visualization
├── integrations
├── registry
└── utils
```

Everything outside the documented public namespace SHALL be considered internal.

---

# 5.4 Namespace Rules

Public modules SHALL be importable directly.

Examples:

```python
from avqa import AVQAttention
from avqa import AVQConfig
from avqa import Backend
```

Users SHALL NOT import modules from private namespaces.

Example:

```python
from avqa._internal.routing import Router
```

is explicitly unsupported.

---

# 5.5 Public Classes

The initial stable API SHALL include:

| Class                | Purpose                  |
| -------------------- | ------------------------ |
| AVQAttention         | Primary attention module |
| AVQConfig            | Immutable configuration  |
| VectorQuantizer      | Quantization subsystem   |
| HierarchicalCodebook | Codebook implementation  |
| Router               | Routing strategy         |
| AdaptiveRefinement   | Refinement algorithm     |
| Scheduler            | Budget allocation        |
| KVCache              | Cache management         |
| Backend              | Backend abstraction      |
| Profiler             | Runtime profiling        |

Future additions SHALL preserve backward compatibility.

---

# 5.6 AVQAttention

`AVQAttention` is the primary entry point of the library.

Responsibilities include:

- input validation,
- execution orchestration,
- backend dispatch,
- attention computation,
- output generation.

It SHALL inherit from `torch.nn.Module`.

The constructor SHALL accept a single `AVQConfig` object rather than a large collection of keyword arguments.

Example:

```python
config = AVQConfig(...)
attention = AVQAttention(config)
```

---

# 5.7 Functional API

The functional API provides stateless execution.

Example:

```python
output = avqa.functional.attention(
    query=q,
    key=k,
    value=v,
    config=config,
)
```

Functional APIs SHALL NOT retain internal state.

---

# 5.8 Configuration API

All configuration SHALL be represented by immutable dataclasses.

Configuration objects SHALL:

- validate parameters,
- support serialization,
- support equality comparison,
- support versioning.

Configurations SHALL be reusable across multiple modules.

---

# 5.9 Backend Interface

Execution backends SHALL implement a common interface.

Every backend SHALL provide methods for:

- attention computation,
- quantization operations,
- refinement,
- merge,
- reductions,
- cache operations.

Algorithmic code SHALL interact only with this interface.

---

# 5.10 Registry System

AVQA SHALL include a registry mechanism for extensibility.

The registry SHALL support:

- quantizers,
- routers,
- merge strategies,
- schedulers,
- backends,
- visualization plugins.

Registration SHALL occur without modifying the core library.

---

# 5.11 Factory Methods

The public API SHALL expose factory methods for constructing common components.

Examples include:

- backend creation,
- configuration loading,
- integration adapters.

Factories SHALL encapsulate implementation-specific logic.

---

# 5.12 Serialization

Every public object SHALL support serialization where applicable.

Examples:

- configurations,
- codebooks,
- routing state,
- learned parameters.

Version metadata SHALL accompany serialized artifacts.

---

# 5.13 Error Model

The API SHALL expose a consistent exception hierarchy.

Examples:

- `AVQAError`
- `ConfigurationError`
- `BackendError`
- `RoutingError`
- `CodebookError`

Public methods SHALL raise documented exceptions.

---

# 5.14 Logging

The library SHALL integrate with Python's standard `logging` module.

Users SHALL control verbosity through configuration.

No library component SHALL print directly to stdout during normal operation.

---

# 5.15 Profiling Interface

Profiling SHALL be optional and non-invasive.

The public profiler SHALL expose methods for:

- starting a session,
- stopping a session,
- collecting metrics,
- exporting reports.

Profiling SHALL not alter algorithmic behavior.

---

# 5.16 Visualization Interface

Visualization components SHALL be decoupled from the core library.

The API SHALL support generating:

- attention heatmaps,
- codebook trees,
- routing paths,
- refinement timelines,
- utilization statistics.

Visualization SHALL remain optional.

---

# 5.17 Framework Adapters

Framework-specific integrations SHALL be accessed through dedicated adapters.

Supported adapters include:

- Hugging Face Transformers,
- vLLM,
- xFormers.

Adapters SHALL translate framework-specific abstractions into AVQA interfaces without embedding algorithmic logic.

---

# 5.18 Extension Contracts

Researchers SHALL be able to implement custom components by conforming to documented interfaces.

Supported extension points include:

- quantizers,
- routers,
- schedulers,
- merge strategies,
- backends.

The core library SHALL remain closed to modification but open to extension.

---

# 5.19 Versioning Policy

The public API SHALL follow Semantic Versioning.

- Major releases MAY introduce breaking changes.
- Minor releases SHALL remain backward compatible.
- Patch releases SHALL contain only bug fixes and documentation updates.

Deprecated APIs SHALL remain available for at least one minor release before removal.

---

# 5.20 API Stability Guarantees

Within a major version:

- Public class names SHALL remain stable.
- Public function signatures SHALL remain stable.
- Configuration schemas SHALL remain backward compatible where practical.
- Documented behaviors SHALL not change unexpectedly.

Internal implementation details MAY evolve without notice.

---

# 5.21 Acceptance Criteria

The Public API is considered complete when:

- All user-facing functionality is accessible through documented interfaces.
- Internal modules remain encapsulated.
- Public objects support serialization, configuration, and profiling where applicable.
- Extension points allow new functionality without modifying existing code.
- Framework integrations interact exclusively through stable public contracts.

The API defined in this chapter serves as the foundation for all subsequent implementation details and integrations.

# Chapter 6 — Core Data Model & Tensor Specification

## 6.1 Purpose

This chapter defines the canonical data model used throughout AVQA.

The data model specifies the structure, ownership, lifecycle, memory layout, and semantics of every tensor manipulated by the attention pipeline.

It is the single source of truth for all algorithmic components, backend implementations, framework integrations, and optimization kernels.

All implementations SHALL conform to the tensor contracts defined in this chapter.

---

# 6.2 Design Objectives

The AVQA data model is designed to satisfy the following objectives:

- Deterministic tensor semantics.
- Backend-independent representation.
- Minimal tensor copies.
- Contiguous memory whenever practical.
- Explicit ownership.
- Static tensor contracts.
- Efficient interoperability with PyTorch.
- Efficient interoperability with Triton.
- Compatibility with distributed inference engines.

---

# 6.3 Canonical Tensor Notation

The following symbols are used consistently throughout the specification.

| Symbol | Meaning                     |
| ------ | --------------------------- |
| **B**  | Batch size                  |
| **T**  | Sequence length             |
| **H**  | Number of attention heads   |
| **D**  | Head dimension              |
| **E**  | Embedding dimension         |
| **C**  | Number of coarse codewords  |
| **R**  | Number of refined codewords |
| **N**  | Number of tokens            |
| **K**  | Refinement budget           |
| **L**  | Hierarchy depth             |

These symbols SHALL retain identical meanings across all documentation, code, tests, and benchmarks.

---

# 6.4 Tensor Naming Convention

Every tensor SHALL use descriptive names.

Examples:

| Tensor             | Description             |
| ------------------ | ----------------------- |
| `query`            | Query vectors           |
| `key`              | Key vectors             |
| `value`            | Value vectors           |
| `codebook`         | Hierarchical codebook   |
| `assignments`      | Key-to-codeword mapping |
| `routing_scores`   | Refinement priorities   |
| `refined_nodes`    | Expanded codewords      |
| `attention_scores` | Raw attention logits    |
| `attention_probs`  | Normalized attention    |
| `output`           | Final attention output  |

Abbreviations SHALL only be used where universally understood.

---

# 6.5 Canonical Tensor Shapes

### Query

```text
[B, H, T, D]
```

---

### Key

```text
[B, H, T, D]
```

---

### Value

```text
[B, H, T, D]
```

---

### Codebook

```text
[H, C, D]
```

Each attention head owns an independent codebook by default.

Alternative sharing strategies MAY be implemented through configurable policies.

---

### Assignment Matrix

```text
[B, H, T]
```

Each entry stores the assigned coarse codeword index.

---

### Routing Scores

```text
[B, H, C]
```

Represents refinement priority for every coarse codeword.

---

### Active Codewords

```text
[B, H, K]
```

Contains the selected codeword indices for refinement.

---

### Refined Codebook

```text
[B, H, R, D]
```

Generated dynamically during inference.

---

### Refined Attention Scores

```text
[B, H, T, R]
```

Computed after adaptive expansion.

---

### Final Attention Probabilities

```text
[B, H, T, C + R]
```

Represents the merged attention distribution after refinement.

---

### Output

```text
[B, H, T, D]
```

This shape SHALL remain compatible with existing Transformer implementations.

---

# 6.6 Tensor Ownership

Every tensor has a single logical owner.

| Tensor                  | Owner      |
| ----------------------- | ---------- |
| Query                   | Attention  |
| Key                     | Attention  |
| Value                   | Attention  |
| Codebook                | Codebook   |
| Assignments             | Quantizer  |
| Routing Scores          | Router     |
| Active Nodes            | Router     |
| Refined Nodes           | Refinement |
| Attention Probabilities | Attention  |
| KV Cache                | Cache      |

Ownership determines lifecycle and mutation permissions.

---

# 6.7 Tensor Lifetime

The attention pipeline contains three categories of tensors.

### Persistent

Examples:

- Codebook
- Configuration
- Learned parameters

These persist across multiple forward passes.

---

### Cached

Examples:

- KV cache
- Assignment cache (optional)

These survive multiple decoding iterations.

---

### Ephemeral

Examples:

- Attention logits
- Routing scores
- Temporary refinement buffers

These exist only during a single forward pass.

---

# 6.8 Memory Layout

Unless otherwise documented, tensors SHALL satisfy the following requirements.

- Row-major layout.
- Contiguous memory where practical.
- Alignment suitable for vectorized GPU execution.
- Compatible with PyTorch contiguous tensors.

Backends MAY internally reorder tensors provided the public contracts remain unchanged.

---

# 6.9 Supported Data Types

Reference implementation SHALL support:

- float32
- float16
- bfloat16

Optional support:

- float64
- FP8
- INT8

Experimental support SHALL be clearly documented.

---

# 6.10 Device Semantics

Every tensor SHALL belong to exactly one device.

Supported devices:

- CPU
- CUDA

Future devices:

- ROCm
- Metal
- XPU
- TPU (adapter dependent)

Cross-device tensor movement SHALL occur explicitly.

Implicit transfers are prohibited.

---

# 6.11 Immutability Rules

The following SHALL be immutable during forward execution:

- Configuration
- Hierarchy topology
- Public metadata

The following MAY change:

- Temporary buffers
- Routing statistics
- Profiling information
- KV cache

Mutation SHALL be localized to the owning subsystem.

---

# 6.12 Tensor Validation

Every public API SHALL validate:

- rank
- dtype
- device
- shape compatibility
- contiguity (where required)

Validation MAY be disabled in optimized execution modes.

---

# 6.13 Shape Invariants

The following invariants SHALL hold.

Query, key, and value SHALL share:

- batch size
- head count
- head dimension

Attention output SHALL preserve:

- batch size
- sequence length
- embedding dimension

Violations SHALL raise documented exceptions.

---

# 6.14 Memory Allocation Strategy

Reference implementation:

- Allocate temporary tensors on demand.

Optimized implementation:

- Reuse buffers.
- Pool allocations.
- Fuse temporary storage.

Memory optimization SHALL remain invisible to users.

---

# 6.15 Tensor Lifecycle

The canonical execution sequence is:

```text
Input
 │
 ▼
Projection
 │
 ▼
Quantization
 │
 ▼
Assignment
 │
 ▼
Routing
 │
 ▼
Refinement
 │
 ▼
Merge
 │
 ▼
Aggregation
 │
 ▼
Output
```

Each stage consumes immutable inputs and produces explicitly documented outputs.

---

# 6.16 Distributed Tensor Semantics

Future distributed implementations SHALL preserve the same logical tensor model.

Distribution strategies MAY include:

- Tensor Parallelism
- Pipeline Parallelism
- Sequence Parallelism
- Context Parallelism

Distributed execution SHALL not alter tensor semantics.

---

# 6.17 Serialization Model

Serializable objects SHALL include:

- Codebooks
- Configurations
- Learned parameters
- Cached statistics (optional)

Transient tensors SHALL NOT be serialized.

---

# 6.18 Complexity Metadata

Every major tensor SHALL document:

- Shape
- Rank
- Memory complexity
- Computational role
- Producer subsystem
- Consumer subsystem

This metadata SHALL be included in developer documentation.

---

# 6.19 Tensor Contracts

Every subsystem SHALL publish explicit input and output contracts.

For every public function, the documentation SHALL specify:

- Input tensors
- Output tensors
- Shape
- Dtype
- Device
- Ownership
- Mutability
- Complexity

Undocumented tensor behavior is prohibited.

---

# 6.20 Data Model Validation

The AVQA data model is considered valid when:

- Every tensor has a documented owner.
- Every tensor has a documented lifecycle.
- Every tensor has a canonical shape.
- Backend implementations preserve tensor contracts.
- Framework integrations require no tensor reinterpretation.
- Memory optimizations remain transparent to the public API.

The data model defined in this chapter is normative and SHALL be treated as the canonical representation of data throughout the AVQA codebase.

# Chapter 7 — Mathematical Specification

## 7.1 Purpose

This chapter formally defines the mathematical foundations of Adaptive Vector-Quantized Attention (AVQ-Attention). It establishes the notation, equations, invariants, and theoretical properties that govern the algorithm independent of any implementation.

The objective of this chapter is to translate the AVQ-Attention paper into a precise mathematical specification suitable for software implementation. Algorithmic implementation details, hardware optimizations, kernel fusion, and framework integrations are intentionally excluded and are specified in subsequent chapters.

Unless explicitly stated otherwise, the mathematical definitions in this chapter follow the notation and formulation presented in the AVQ-Attention paper.

---

# 7.2 Mathematical Scope

This chapter specifies:

- Standard scaled dot-product attention.
- Vector-Quantized (VQ) attention.
- Hierarchical codebook formulation.
- Adaptive refinement model.
- Attention importance computation.
- Parent-child refinement.
- Correcting attention.
- Computational complexity.
- Mathematical invariants.
- Numerical assumptions.

This chapter does **not** specify:

- Triton kernels.
- PyTorch implementation.
- Tensor layouts.
- Memory organization.
- FlashAttention implementation.
- API design.

---

# 7.3 Notation

The following notation is used consistently throughout this specification.

| Symbol        | Description                             |
| ------------- | --------------------------------------- |
| (N)           | Number of input tokens                  |
| (M)           | Number of codewords                     |
| (M_0)         | Number of parent codewords              |
| (\mathcal{C}) | Number of children per parent           |
| (P)           | Number of refined parents               |
| (d)           | Head dimension                          |
| (Q)           | Query matrix                            |
| (K)           | Key matrix                              |
| (V)           | Value matrix                            |
| (C)           | Codebook                                |
| (C_p)         | Parent codeword                         |
| (C\_{p,c})    | Child codeword                          |
| (n_j)         | Number of keys assigned to codeword (j) |
| (\bar V_j)    | Aggregated values for codeword (j)      |
| (A)           | Attention weights                       |
| (Y)           | Output tensor                           |

These symbols SHALL retain identical meanings throughout the AVQA specification.

---

# 7.4 Standard Attention

Scaled dot-product attention computes attention as

[
Y_i
===

\frac{
\sum*{j=1}^{N}
\exp(Q_iK_j^\top)V_j
}
{
\sum*{l=1}^{N}
\exp(Q_iK_l^\top)
}
]

where

- (Q_i \in \mathbb{R}^d)
- (K_i \in \mathbb{R}^d)
- (V_i \in \mathbb{R}^d)

The complexity of computing all pairwise interactions is

[
\mathcal O(N^2d)
]

This quadratic interaction is the computational bottleneck addressed by AVQ-Attention.

---

# 7.5 Vector Quantization

Rather than attending to every key independently, AVQ replaces keys with representative codewords.

Let

[
C={C_1,\ldots,C_M}
]

denote the learned codebook.

Each key is assigned to its nearest codeword

[
a(K_j)
======

\arg\min_a
|K_j-C_a|^2
]

and replaced by

[
\hat K_j=C_{a(K_j)}
]

Only keys are quantized.

Queries and values remain unchanged.

---

# 7.6 Aggregated Representation

For every codeword

[
C_a
]

define

[
n_a
===

\left|
{j:a(K_j)=a}
\right|
]

representing the number of assigned keys.

Similarly define

[
\bar V_a
========

\sum\_{j:a(K_j)=a}
V_j
]

representing the aggregated value vector.

The pair

[
(\bar V_a,n_a)
]

forms the sufficient statistics required for VQ attention.

Individual keys are no longer required during attention computation.

---

# 7.7 Vector-Quantized Attention

Attention is rewritten over codewords

[
Y_i
\approx
\frac{
\sum_a
\exp(Q_iC_a^\top)
\bar V_a
}
{
\sum_a
\exp(Q_iC_a^\top)
n_a
}
]

The only approximation introduced is replacing keys by their assigned codewords.

No approximation is introduced in the value aggregation itself.

---

# 7.8 Hierarchical Codebook

AVQ extends the flat codebook into a hierarchy.

Each parent

[
C_p
]

owns

[
\mathcal C
]

children

[
{C_{p,1},...,C_{p,\mathcal C}}
]

Total codebook size becomes

[
M\_{total}
=========

M_0(1+\mathcal C)
]

Children supplement their parent rather than replacing it.

---

# 7.9 Parent–Child Constraint

A fundamental property of AVQ is

[
C_p
===

\frac1{\mathcal C}
\sum*{c=1}^{\mathcal C}
C*{p,c}
]

The parent equals the arithmetic mean of its children.

This constraint is not merely geometric.

It enables efficient reconstruction of parent logits during adaptive refinement.

Without this property, the correcting-attention formulation cannot be derived.

---

# 7.10 Attention Importance

For every parent codeword,

AVQ computes an importance score from the attention already produced during inference.

Importance for query tile (I) is defined as

[
w_j(I)
======

\sum*{i\in I}
\frac{
A*{ij}n_j
}{
\bar Z_i
}
]

where

- (A) denotes attention weights.
- (n_j) denotes assigned keys.
- (\bar Z_i) denotes the online-softmax denominator.

No auxiliary importance network is required.

Importance emerges naturally from attention itself.

---

# 7.11 Adaptive Refinement

Adaptive refinement proceeds as follows.

1. Compute parent attention.
2. Compute parent importance.
3. Select the top-(P) parents.
4. Spawn the selected children.
5. Recompute attention only for those children.
6. Correct the parent contribution.
7. Produce the final attention output.

Only

[
P
]

parents are refined.

The remainder stay represented by their parent codeword.

---

# 7.12 Parent Logit Recovery

Direct recomputation of parent logits would increase computation.

Instead AVQ exploits the parent-child constraint.

Parent logits satisfy

[
S_p
===

\frac1{\mathcal C}
\sum_c
S_c
]

where

[
S_c
===

QC_c^\top
]

This allows parent logits to be reconstructed directly from already computed child logits.

No additional matrix multiplication is required.

---

# 7.13 Correcting Attention

Once children replace part of a parent cluster,

the original parent contribution must be removed.

Define

[
\Delta A
========

A_c-A_p
]

where

- (A_p) denotes parent attention.
- (A_c) denotes child attention.

Updating the attention accumulators using

[
\Delta A
]

is mathematically equivalent to replacing parent contributions with child contributions.

This property follows directly from the derivation in Appendix C of the paper.

---

# 7.14 Online Softmax

AVQ preserves FlashAttention's online-softmax algorithm.

For every tile,

running accumulators maintain

- maximum logit,
- numerator,
- denominator.

Adaptive refinement updates these accumulators incrementally.

No global attention matrix is materialized.

This preserves linear memory complexity while enabling adaptive refinement.

---

# 7.15 Numerical Stability

The running maximum SHALL ignore empty codewords.

Formally,

[
m_i
===

\max*{j:n_j>0}
S*{ij}
]

Empty codewords contribute neither values nor counts and therefore must not influence numerical scaling.

This avoids instability caused by empty child codewords.

---

# 7.16 Computational Complexity

For sequence length

[
N
]

parent codebook

[
M_0
]

children

[
\mathcal C
]

and

[
P
]

refined parents,

the total computational complexity becomes

[
\mathcal O
\left(
N(M_0+P\mathcal C)d
\right)
]

This preserves linear complexity with respect to sequence length.

The adaptive refinement increases the constant factor but not the asymptotic order.

---

# 7.17 Mathematical Invariants

The following invariants SHALL hold.

### Hierarchy Invariant

Every parent equals the mean of its children.

### Assignment Invariant

Every key is assigned to exactly one parent.

During child refinement,

every key is assigned to either

- its parent, or
- one child of that parent.

### Conservation Invariant

Aggregated values satisfy

[
\sum_a
\bar V_a
========

\sum_j
V_j
]

### Count Invariant

[
\sum_a
n_a=N
]

### Attention Invariant

The resulting attention distribution remains normalized after correction.

---

# 7.18 Approximation Sources

AVQ introduces approximation exclusively through vector quantization.

No approximation is introduced by:

- adaptive refinement,
- online softmax,
- correcting attention,
- parent logit reconstruction.

These operations are algebraically equivalent to the corresponding refined attention formulation under the assumptions of the hierarchical codebook.

---

# 7.19 Theoretical Guarantees

Under the assumptions specified in the AVQ-Attention paper:

- The computational complexity is linear in the sequence length for fixed codebook parameters.
- Parent reconstruction is exact under the parent-child mean constraint.
- Correcting attention preserves the intended refinement semantics without revisiting parent codewords.
- Online softmax maintains numerical stability while avoiding materialization of the full attention matrix.
- The dominant approximation arises from the quantization of keys rather than from the adaptive refinement process.

---

# 7.20 Mapping Mathematics to the Implementation

The mathematical objects defined in this chapter correspond to the following implementation components.

| Mathematical Object        | AVQA Component             |
| -------------------------- | -------------------------- |
| (Q,K,V)                    | Attention Input            |
| Codebook (C)               | `HierarchicalCodebook`     |
| Assignment (a(K))          | `VectorQuantizer`          |
| Aggregated Values (\bar V) | `VQPrecomputeKernel`       |
| Counts (n)                 | `VQPrecomputeKernel`       |
| Importance (w)             | `ImportanceEstimator`      |
| Parent Selection           | `AdaptiveSelector`         |
| Child Refinement           | `AdaptiveRefinementKernel` |
| Correcting Attention       | `CorrectionOperator`       |
| Output (Y)                 | `AVQAttention`             |

This mapping serves as the formal bridge between the mathematical specification and the software architecture described in subsequent chapters.

# Chapter 8 — Vector Quantization Engine

## 8.1 Purpose

The Vector Quantization (VQ) Engine is the first execution stage of AVQ-Attention. Its purpose is to transform the original key-value tensors into a compressed hierarchical representation that enables subsequent attention computation over codewords instead of individual keys.

Unlike conventional vector quantization, AVQ does not merely assign keys to codewords. During the same pass it computes the aggregated value vectors and assignment counts required by the attention stage. This preprocessing is fundamental to achieving linear-complexity attention and minimizing memory traffic.

This chapter specifies the hierarchical codebook, assignment procedure, aggregation algorithm, training methodology, and engineering requirements for the VQ engine. Kernel implementation details are intentionally deferred to Chapter 11.

---

# 8.2 Objectives

The VQ Engine SHALL satisfy the following objectives:

- Compress the key space into representative codewords.
- Preserve the statistical information required for attention.
- Produce aggregated values and assignment counts.
- Support hierarchical parent-child codebooks.
- Execute in linear time with respect to sequence length.
- Operate efficiently on GPU architectures.
- Support both inference and end-to-end training.
- Minimize global memory traffic through fused execution.

The VQ Engine SHALL NOT compute attention. Its sole responsibility is constructing the compressed representation consumed by the attention kernel.

---

# 8.3 Hierarchical Codebook

AVQ replaces the flat codebook of traditional VQ-Attention with a two-level hierarchical codebook.

The hierarchy consists of:

- **Parent codewords** ((M_0))
- **Child codewords** ((\mathcal{C}) per parent)

The total number of codewords is:

[
M_{\text{total}} = M_0(1+\mathcal{C})
]

Children are associated with exactly one parent, and parents are constrained to equal the arithmetic mean of their children:

[
C_p = \frac{1}{\mathcal{C}}\sum_{i=1}^{\mathcal{C}} C_{p,i}
]

This geometric constraint is essential because it enables efficient parent-logit reconstruction during adaptive refinement without recomputing parent attention.

---

# 8.4 Responsibilities of the VQ Engine

For every forward pass, the VQ Engine SHALL:

1. Receive key and value tensors.
2. Assign each key to its nearest parent codeword.
3. Assign each key to the nearest child within that parent.
4. Accumulate value vectors.
5. Count assignments.
6. Produce tensors consumed by the attention kernel.

The engine SHALL produce the following outputs:

| Output                  | Description                           |
| ----------------------- | ------------------------------------- |
| Parent assignments      | Parent index for every key            |
| Child assignments       | Child index within selected parent    |
| Parent value aggregates | Sum of values assigned to each parent |
| Child value aggregates  | Sum of values assigned to each child  |
| Parent counts           | Number of keys per parent             |
| Child counts            | Number of keys per child              |

These outputs constitute the complete compressed representation required by AVQ-Attention.

---

# 8.5 Assignment Algorithm

The paper introduces a hierarchical assignment strategy that exploits the codebook structure.

Instead of comparing each key against every codeword, assignment proceeds in two stages:

### Stage 1 — Parent Assignment

Each key is compared against all parent codewords.

The nearest parent is selected.

[
a_p(k)=
\arg\min_p
|k-C_p|^2
]

### Stage 2 — Child Assignment

Once the parent has been selected, the key is compared **only** against that parent's children.

[
a_c(k)=
\arg\min_c
|k-C_{p,c}|^2
]

This reduces the search complexity from

[
\mathcal O(M_{\text{total}})
]

to

[
\mathcal O(M_0+\mathcal C)
]

distance evaluations per key.

---

# 8.6 Value Aggregation

Unlike conventional vector quantization, AVQ requires more than assignments.

For every codeword, the engine computes:

- aggregated value vector,
- assignment count.

For codeword (j):

[
\bar V_j
========

\sum\_{k:a(k)=j}
V_k
]

and

[
n_j
===

\left|
{k:a(k)=j}
\right|
]

These statistics become the direct inputs to the attention kernel.

No individual key vectors are required beyond this point for attention computation.

---

# 8.7 Fused Precompute

The AVQ paper emphasizes that assignments and aggregation are fused into a single preprocessing kernel.

During one pass over the keys:

- assignments are computed,
- value vectors are accumulated,
- assignment counts are accumulated.

This avoids multiple traversals of the key tensor and significantly reduces HBM traffic.

The implementation SHALL preserve this fused execution model in optimized backends. Reference implementations MAY separate the operations for readability and testing.

---

# 8.8 Tensor Contracts

The VQ Engine consumes:

| Tensor   | Shape           |
| -------- | --------------- |
| Keys     | `[B,H,N,D]`     |
| Values   | `[B,H,N,D]`     |
| Codebook | `[H,M_total,D]` |

The engine produces:

| Tensor             | Shape          |
| ------------------ | -------------- |
| Parent counts      | `[B,H,M₀]`     |
| Child counts       | `[B,H,M₀,𝒞]`   |
| Parent aggregates  | `[B,H,M₀,D]`   |
| Child aggregates   | `[B,H,M₀,𝒞,D]` |
| Parent assignments | `[B,H,N]`      |
| Child assignments  | `[B,H,N]`      |

These tensor contracts SHALL remain stable across all supported execution backends.

---

# 8.9 Training the Codebook

The reference implementation follows the EMA-based online k-means procedure described in the paper.

Training consists of:

1. Assign keys to codewords.
2. Update cluster statistics.
3. Compute exponential moving averages.
4. Update codeword positions.
5. Reproject parent codewords to satisfy the hierarchy constraint.

The default hyperparameters reported by the paper include:

- EMA decay: **0.99**
- Commitment loss weight: **0.25**

These values SHALL be the default reference configuration while remaining user-configurable.

---

# 8.10 Child Initialization

Children are initialized near their parent to encourage rapid specialization while preserving hierarchy locality.

The reference initialization is

[
C\_{p,c}
=======

C_p

- 0.1\epsilon,
  \quad
  \epsilon\sim\mathcal N(0,I)
  ]

where the perturbation scale is 0.1.

Alternative initialization schemes MAY be implemented but SHALL preserve the parent-child mean constraint after projection.

---

# 8.11 Dead Code Handling

Some codewords may receive few or no assignments during training.

The reference implementation uses the simple EMA approach throughout its experiments, while the authors note that more advanced strategies—such as per-batch dead-code reassignment and differentiable quantizers—can further improve results. These techniques are considered orthogonal improvements rather than core components of AVQ-Attention.

Accordingly:

- The reference backend SHALL implement the EMA approach.
- Enhanced codebook maintenance MAY be exposed as optional training policies.

---

# 8.12 Computational Complexity

For each key, the VQ Engine performs:

- (M_0) parent comparisons.
- (\mathcal C) child comparisons.
- One scatter-add for values.
- One scatter-add for counts.

Overall complexity is therefore

[
\mathcal O
\left(
N(M_0+\mathcal C)D
\right)
]

This preprocessing complexity is linear in the sequence length and substantially lower than flat VQ assignment over the full codebook.

---

# 8.13 Engineering Requirements

The AVQA implementation SHALL satisfy the following engineering requirements:

- Separate algorithmic logic from backend-specific kernels.
- Preserve deterministic assignments when deterministic execution is enabled.
- Avoid unnecessary tensor copies.
- Support batched execution across arbitrary batch sizes.
- Support FP32, FP16, and BF16 execution.
- Provide both a pure PyTorch reference implementation and an optimized Triton implementation.
- Expose profiling statistics including assignment counts, codebook utilization, and aggregation time.
- Allow future quantization algorithms to be registered without modifying the core attention pipeline.

---

# 8.14 Acceptance Criteria

The Vector Quantization Engine is considered compliant when:

1. Every key is assigned to exactly one parent and one child.
2. Parent and child aggregates correctly represent the assigned value vectors.
3. Assignment counts equal the number of contributing keys.
4. The parent-child hierarchy satisfies the mean constraint after every training update.
5. The fused preprocessing stage produces outputs identical to the reference implementation within documented numerical tolerances.
6. The preprocessing complexity scales as (\mathcal O(N(M_0+\mathcal C)D)).
7. The produced tensors are directly consumable by the adaptive attention kernel without additional preprocessing.

The VQ Engine defined in this chapter provides the compressed hierarchical representation that serves as the input to the adaptive attention algorithm specified in Chapter 9.

# Chapter 9 — Adaptive Attention Algorithm

## 9.1 Purpose

Adaptive Attention is the defining contribution of AVQ-Attention. Unlike conventional Vector-Quantized Attention, which performs attention over a fixed-resolution codebook, AVQ dynamically allocates computational resources to the regions of the key space that contribute most to the attention output.

This chapter specifies the adaptive refinement algorithm presented in the AVQ-Attention paper. It defines the logical execution of adaptive attention independent of any backend implementation. Kernel fusion, memory layout, and hardware-specific optimizations are specified in later chapters.

The adaptive algorithm SHALL preserve the mathematical properties established in Chapter 7 while operating on the compressed representation produced by the Vector Quantization Engine defined in Chapter 8.

---

# 9.2 Motivation

Traditional VQ-Attention assigns a fixed representational capacity to every region of the embedding space.

However, attention distributions are highly non-uniform:

- A small subset of codewords accounts for most of the attention mass.
- Many codewords receive negligible attention.
- Uniform refinement wastes computational resources.

AVQ addresses this imbalance by refining only the codewords that significantly influence the current attention computation.

This adaptive allocation improves approximation quality while preserving linear complexity with respect to sequence length.

---

# 9.3 Overview

Adaptive attention executes after the VQ preprocessing stage.

The logical pipeline is:

```text
Compressed Representation
        │
        ▼
Parent Attention
        │
        ▼
Importance Estimation
        │
        ▼
Top-P Parent Selection
        │
        ▼
Child Expansion
        │
        ▼
Child Attention
        │
        ▼
Correcting Attention
        │
        ▼
Final Output
```

Each stage is deterministic and operates on the outputs of the previous stage.

---

# 9.4 Stage 1 — Parent Attention

The first stage computes attention over the parent codebook only.

Inputs:

- Queries
- Parent codewords
- Parent value aggregates
- Parent assignment counts

Outputs:

- Parent attention logits
- Parent attention probabilities
- Running online-softmax statistics

This computation follows the same online tiled attention procedure used by FlashAttention, except that keys are replaced by parent codewords.

---

# 9.5 Stage 2 — Importance Estimation

After parent attention has been computed, each parent codeword is assigned an importance score.

The importance score estimates the contribution of a parent to the current attention output.

The paper derives this score directly from the attention computation, eliminating the need for auxiliary routing networks or learned gating mechanisms.

Properties:

- Computed during the forward pass.
- Requires no additional model parameters.
- Uses existing attention statistics.
- Scales naturally with batch size.

Importance estimation SHALL reuse quantities already produced by parent attention whenever possible.

---

# 9.6 Stage 3 — Parent Selection

Once importance scores have been computed, only the most important parents are refined.

Let

- **P** denote the refinement budget.

The algorithm selects the **P** highest-scoring parents.

Selection SHALL satisfy:

- deterministic ordering,
- stable tie handling,
- configurable refinement budget,
- bounded computational cost.

The reference implementation SHALL reproduce the selection procedure described in the paper. Alternative scheduling policies MAY be implemented as optional extensions.

---

# 9.7 Stage 4 — Child Expansion

Each selected parent is replaced by its children.

Expansion consists of:

1. Loading child codewords.
2. Loading child value aggregates.
3. Loading child assignment counts.
4. Constructing the refined attention tile.

Unselected parents remain unchanged.

Expansion is therefore sparse.

Only the selected parents increase computational cost.

---

# 9.8 Stage 5 — Child Attention

Attention is recomputed only for the expanded children.

The computation uses:

- identical queries,
- child codewords,
- child aggregates,
- child counts.

No attention is recomputed for parents that were not selected.

This selective recomputation is the primary source of AVQ's efficiency.

---

# 9.9 Correcting Attention

Naively adding child attention would double-count information already represented by the parent.

The paper therefore introduces **Correcting Attention**, which replaces the contribution of a refined parent with the contributions of its children rather than accumulating both.

Conceptually:

```text
Parent Contribution
        │
        ▼
Remove
        │
        ▼
Child Contributions
        │
        ▼
Corrected Attention
```

Correcting attention preserves normalization while avoiding redundant computation.

---

# 9.10 Parent Logit Reconstruction

Because parent codewords equal the mean of their children, parent logits can be reconstructed directly from child logits.

This avoids recomputing parent-query matrix multiplications after refinement.

The implementation SHALL exploit this relationship whenever the hierarchy constraint is satisfied.

This reconstruction is an algebraic consequence of the hierarchical codebook and is one of the key optimizations described in the paper.

---

# 9.11 Online Softmax Integration

Adaptive refinement operates within FlashAttention's online-softmax algorithm.

The implementation SHALL maintain the running:

- maximum,
- numerator,
- denominator,

throughout refinement.

Global recomputation of softmax is prohibited.

Correcting attention updates the running accumulators incrementally.

This preserves linear memory complexity.

---

# 9.12 Empty Codewords

Some parent or child codewords may receive no assignments.

Empty codewords SHALL:

- contribute zero values,
- contribute zero counts,
- be excluded from maximum-logit computation,
- not influence normalization.

Handling empty codewords correctly is essential for numerical stability.

---

# 9.13 Computational Complexity

Let

- **N** = sequence length,
- **M₀** = number of parent codewords,
- **C** = children per parent,
- **P** = refined parents.

Adaptive attention performs:

1. Parent attention over **M₀** codewords.
2. Child attention over **P × C** codewords.

Overall complexity becomes

[
\mathcal O\left(N(M_0 + P\mathcal{C})D\right)
]

For fixed hierarchy parameters, complexity remains linear in sequence length.

---

# 9.14 Numerical Properties

Adaptive refinement SHALL preserve:

- normalized attention probabilities,
- online-softmax stability,
- deterministic execution (when enabled),
- bounded floating-point error.

Reference implementations SHALL prioritize numerical equivalence with the paper over performance optimizations.

---

# 9.15 Engineering Requirements

The adaptive attention implementation SHALL:

- separate logical refinement from backend execution,
- reuse intermediate computations whenever possible,
- avoid recomputing parent attention,
- support batched execution,
- support causal and non-causal attention,
- support FP32, FP16, and BF16,
- expose refinement statistics for profiling,
- permit configurable refinement budgets.

---

# 9.16 Validation

A compliant implementation SHALL satisfy the following conditions:

1. Parent attention is computed before refinement.
2. Importance scores are derived from the attention computation.
3. Exactly **P** parents are refined unless fewer valid candidates exist.
4. Child attention replaces, rather than duplicates, refined parent contributions.
5. Online-softmax statistics remain numerically stable throughout refinement.
6. Final attention probabilities remain normalized.
7. Results match the reference implementation within documented floating-point tolerances.

---

# 9.17 Relationship to the Execution Engine

This chapter specifies the logical algorithm.

The optimized implementation described in Chapter 10 SHALL fuse multiple logical stages into a single execution pipeline where permitted by the paper.

Logical decomposition SHALL NOT be interpreted as a requirement for separate runtime kernels or independent GPU launches.

Instead, these stages define the conceptual behavior that every backend implementation must preserve regardless of optimization strategy.

The adaptive attention algorithm defined in this chapter represents the canonical behavior of AVQ-Attention and serves as the normative reference for all backend implementations, including the Triton kernels described in subsequent chapters.

# Chapter 10 — Attention Execution Pipeline

## 10.1 Purpose

The previous chapters describe the mathematical formulation of AVQ-Attention (Chapter 7), the construction of the hierarchical representation (Chapter 8), and the adaptive refinement algorithm (Chapter 9). This chapter specifies how those logical components are composed into a single execution pipeline.

Unlike the previous chapters, which focus on _what_ the algorithm computes, this chapter specifies _when_ each computation occurs and _how_ data flows through the system. It defines the canonical execution model that every backend implementation shall preserve.

This chapter is intentionally backend-agnostic. GPU kernel implementations, Triton-specific optimizations, and hardware scheduling are described in Chapter 11.

---

# 10.2 Execution Model

The AVQ-Attention forward pass consists of two logical execution phases corresponding to the design presented in the paper:

1. **Vector Quantization Precompute**
2. **Adaptive Attention Computation**

These phases collectively implement a complete forward pass while avoiding redundant computation and minimizing memory movement. Although the optimized implementation may fuse operations, the observable behavior SHALL remain equivalent to this execution model.

---

# 10.3 High-Level Pipeline

The complete execution flow is shown below.

```text
                 Input
                   │
                   ▼
        Linear Q / K / V Projection
                   │
                   ▼
      Vector Quantization Precompute
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
 Parent Aggregates      Child Aggregates
 Parent Counts          Child Counts
 Parent Assignments     Child Assignments
        │                     │
        └──────────┬──────────┘
                   ▼
        Parent Attention Pass
                   │
                   ▼
       Importance Estimation
                   │
                   ▼
       Adaptive Parent Selection
                   │
                   ▼
         Child Attention Pass
                   │
                   ▼
       Correcting Attention
                   │
                   ▼
       Weighted Value Reduction
                   │
                   ▼
          Output Projection
                   │
                   ▼
             Final Output
```

The ordering shown above is normative.

---

# 10.4 Execution Stages

The execution pipeline SHALL consist of the following stages.

| Stage            | Description                       |
| ---------------- | --------------------------------- |
| Input            | Receive projected Q, K, V tensors |
| Precompute       | Build hierarchical representation |
| Parent Attention | Compute coarse attention          |
| Importance       | Estimate refinement priority      |
| Selection        | Select parents for refinement     |
| Child Attention  | Compute refined attention         |
| Correction       | Replace parent contributions      |
| Reduction        | Compute weighted output           |
| Output           | Return attention result           |

Every backend SHALL preserve this logical ordering.

---

# 10.5 Stage 1 — Input Preparation

The execution engine receives:

- Query tensor
- Key tensor
- Value tensor
- Optional attention mask
- Optional causal mask
- Optional KV cache
- Configuration object

Input validation SHALL occur before execution begins.

The execution engine SHALL verify:

- compatible tensor shapes,
- supported dtypes,
- compatible devices,
- valid refinement budget,
- initialized codebook.

Validation MAY be disabled in optimized production mode.

---

# 10.6 Stage 2 — Vector Quantization Precompute

The execution engine invokes the Vector Quantization Engine defined in Chapter 8.

Outputs include:

- parent assignments,
- child assignments,
- parent aggregates,
- child aggregates,
- assignment counts.

The precompute stage SHALL complete before attention computation begins.

No attention computation SHALL depend on uninitialized aggregation statistics.

---

# 10.7 Stage 3 — Parent Attention

The execution engine computes attention over the parent codebook.

Outputs include:

- parent logits,
- parent probabilities,
- online softmax accumulators,
- importance statistics.

The implementation SHALL reuse intermediate quantities wherever possible.

No refinement occurs during this stage.

---

# 10.8 Stage 4 — Importance Estimation

Importance estimation SHALL execute immediately after parent attention.

The execution engine computes an importance score for every parent codeword using quantities already available from the attention computation.

The implementation SHALL NOT introduce additional neural network modules or learned routing mechanisms.

Importance estimation is an intrinsic component of the attention computation rather than an independent inference stage.

---

# 10.9 Stage 5 — Adaptive Selection

The execution engine ranks parent codewords by importance.

Given refinement budget **P**:

- the **P** highest-priority parents SHALL be selected,
- ties SHALL be resolved deterministically,
- duplicate selections SHALL be prohibited.

Selection SHALL produce an ordered refinement list consumed by the next stage.

---

# 10.10 Stage 6 — Child Attention

For every selected parent:

1. Load child codewords.
2. Load child aggregates.
3. Load child counts.
4. Compute child attention logits.
5. Update online-softmax accumulators.

Parents not selected for refinement SHALL remain unchanged.

This stage SHALL only process the selected subset of parents.

---

# 10.11 Stage 7 — Correcting Attention

The child attention computation replaces—not augments—the contribution of refined parents.

The execution engine SHALL:

1. Remove the coarse parent contribution.
2. Insert the refined child contributions.
3. Update running softmax statistics.
4. Preserve normalization.

This stage SHALL execute without materializing the complete attention matrix.

---

# 10.12 Stage 8 — Weighted Reduction

Once refinement is complete, the execution engine computes the final weighted value vectors.

Inputs:

- corrected attention probabilities,
- aggregated value tensors.

Outputs:

- attention output tensor.

The reduction stage SHALL preserve the output tensor shape expected by standard Transformer attention modules.

---

# 10.13 Stage 9 — Output Projection

The attention output SHALL be projected through the model's output projection layer.

This stage is intentionally identical to standard Transformer implementations.

AVQA modifies only the attention computation.

It SHALL NOT alter the surrounding Transformer architecture.

---

# 10.14 Execution State

The execution engine maintains several categories of state.

### Immutable State

- Configuration
- Codebook topology
- Backend selection

### Cached State

- KV cache
- Assignment cache (optional)
- Backend workspace

### Ephemeral State

- Attention logits
- Routing scores
- Temporary accumulators
- Scratch buffers

Ephemeral state SHALL be released at the end of the forward pass.

---

# 10.15 Execution Modes

The execution pipeline SHALL support multiple execution modes.

### Reference Mode

Characteristics:

- Pure PyTorch.
- Fully deterministic.
- Maximum readability.
- Intended for testing and verification.

### Optimized Mode

Characteristics:

- Triton kernels.
- Kernel fusion.
- Reduced memory movement.
- Production deployment.

### Research Mode

Characteristics:

- Extensive diagnostics.
- Profiling enabled.
- Intermediate tensors exposed.
- Visualization hooks enabled.

Execution mode SHALL be selected through configuration.

---

# 10.16 Error Handling

The execution engine SHALL detect and report:

- invalid tensor shapes,
- incompatible devices,
- unsupported dtypes,
- invalid refinement budgets,
- uninitialized codebooks,
- backend failures.

Errors SHALL terminate execution before producing partial outputs.

Recoverable errors SHALL provide descriptive diagnostic messages.

---

# 10.17 Synchronization Semantics

The execution pipeline SHALL minimize synchronization points.

The logical execution requires synchronization only between:

- completion of VQ precompute,
- completion of parent attention,
- completion of adaptive refinement,
- completion of final reduction.

Additional synchronization SHALL be avoided unless required by backend implementations.

---

# 10.18 Memory Lifetime

The lifetime of execution buffers is illustrated below.

```text
Input Tensors
│────────────────────────────────────────────│

Assignments
      │──────────────│

Aggregates
      │────────────────────────────│

Parent Attention
             │────────────│

Importance
                    │────│

Child Attention
                     │────────────│

Scratch Buffers
          │────────────────────────────│

Output
                               │────────│
```

Backends MAY reuse storage for non-overlapping lifetimes.

---

# 10.19 Pipeline Invariants

The following execution invariants SHALL hold.

### Ordering

Precompute SHALL complete before attention begins.

### Consistency

All attention computations SHALL use a single version of the codebook during a forward pass.

### Isolation

Execution SHALL NOT modify immutable configuration.

### Determinism

Reference mode SHALL produce deterministic outputs for identical inputs and seeds.

### Normalization

Final attention probabilities SHALL remain normalized after correcting attention.

---

# 10.20 Backend Independence

The execution pipeline defines logical behavior only.

Backend implementations MAY:

- fuse stages,
- reorder independent operations,
- optimize memory layouts,
- eliminate intermediate buffers,

provided that:

- observable outputs remain equivalent,
- documented numerical tolerances are preserved,
- pipeline invariants remain satisfied.

---

# 10.21 Acceptance Criteria

An implementation satisfies this chapter when:

1. The complete forward pass follows the execution order defined herein.
2. All stages consume and produce the documented data contracts.
3. Intermediate computations preserve the mathematical properties established in Chapter 7.
4. Adaptive refinement behaves as specified in Chapter 9.
5. Backend-specific optimizations do not alter observable behavior.
6. The execution pipeline integrates seamlessly with standard Transformer forward passes.
7. Memory usage remains linear with respect to sequence length for fixed codebook parameters.

This execution model serves as the canonical runtime specification for all AVQA backends. Subsequent chapters define how this logical pipeline is realized through optimized Triton kernels, backend abstractions, and framework integrations.

---

# Chapter 11 — Triton Kernel Backend

## 11.1 Purpose

This chapter specifies the optimized Triton kernel backend that the
runtime model in Chapter 10 executes when ``backend.name == "triton"``.
The kernels in this chapter implement the same logical pipeline but
fuse the VQ preprocessing, parent attention, child attention, and
correcting attention into dedicated Triton kernels that achieve
bandwidth-bound performance on NVIDIA GPUs.

The reference behavior under Chapter 10 SHALL remain the source of
truth for numerical equivalence. Triton kernels SHALL match the
reference within the tolerances recorded in `BENCHMARKS.md`.

The reference Python implementation MUST continue to pass all tests
even when the Triton backend is unavailable (no CUDA, no Triton). This
is enforced by `src/avqa/backend.py:336` which falls back to the
TorchBackend.

## 11.2 Scope

Triton kernels implement:

- Vector quantization precompute (fused two-stage Euclidean VQ with
  scattered accumulation; §11.4).
- Online-softmax tiled attention over the parent codebook (§11.5).
- Adaptive child attention recomputation (§11.6).
- Online-softmax correction / merge against the corrected child logits
  (§11.7).

Everything else remains PyTorch code (projections, output
projections, routing selection, scheduler, KV cache).

## 11.3 Tile Layouts and Data Conventions

Tile sizes documented here are the contract; autotuners may vary them
at runtime but MUST satisfy the alignment and block-factor rules.

- ``BLOCK_T``: rows of Q processed in one query tile. Default 64.
  MUST be a power of 2 and ≤ 256.
- ``BLOCK_M``: codebook rows (parents or children) per key tile.
  Default 64. MUST be a power of 2.
- ``BLOCK_D``: head dimension. Default 64. MUST be a power of 2 and
  a divisor of any supported head dimension (16, 32, 64, 96, 128).
- ``BLOCK_C``: children per parent for child attention. Default 4.

FP32 and BF16 dtypes are required; FP16 is optional. ``tl.float32``
is the math dtype; intermediate softmax accumulates in FP32 even when
inputs are BF16 / FP16 to preserve numerical stability.

Empty codewords (count == 0) SHALL be excluded from online softmax
(see Chapter 9 §9.12 and Chapter 7 §7.15). The fused VQ kernel MUST
emit a sentinel ``count = 0`` for empty parents.

## 11.4 Fused Vector-Quantization Kernel

Inputs:

- ``keys [B, H, N, D]`` (FP32 / BF16 / FP16)
- ``values [B, H, N, D]``
- ``parents [H, M_0, D]``
- ``children [H, M_0, C, D]``

Outputs (per-batch):

- ``parent_assignments [B, H, N]`` int32
- ``child_assignments [B, H, N]`` int32
- ``parent_aggregates [B, H, M_0, D]``
- ``child_aggregates [B, H, M_0, C, D]``
- ``parent_counts [B, H, M_0]``
- ``child_counts [B, H, M_0, C]``

Algorithm (single fused kernel per batch element):

1. Stream each key along the sequence dimension.
2. Compute pairwise squared distance to all parents in shared memory.
3. ``argmin`` produces parent assignment.
4. Gather that parent's children and compute pairwise squared distance.
5. ``argmin`` produces child assignment.
6. Scatter-add the value vector into both parent and child aggregate
   buffers using atomic adds.
7. Increment the corresponding count buffers.

Complexity: O(N·(M_0 + C)·D) per batch element (Chapter 8 §8.12).

Numerical contract: result MUST agree with
`EuclideanHierarchicalQuantizer.precompute` (`src/avqa/quantizer.py:150`)
within FP32 tolerances documented in `BENCHMARKS.md` §"Correctness
Validation".

## 11.5 Online-Softmax Parent Attention Kernel

Inputs:

- ``query [B, H, T_q, D]``
- ``parents [H, M_0, D]`` (the same parent codebook as the VQ kernel
  was given)
- ``parent_aggregates [B, H, M_0, D_v]``
- ``parent_counts [B, H, M_0]``

Outputs:

- ``parent_attention_probs [B, H, T_q, M_0]``
- ``running_state_max [B, H, T_q, 1]``
- ``running_state_denom [B, H, T_q, 1]``
- ``running_state_num [B, H, T_q, 1, D_v]``

Algorithm:

1. Loop over query tiles of size ``BLOCK_T``.
2. Compute ``S = q · pᵀ / sqrt(D)`` → ``[BLOCK_T, M_0]``.
3. Mask empty codewords to ``-inf`` (§9.12).
4. Apply mask and update the running online-softmax accumulators
   (§7.14 / `OnlineSoftmaxState.merge` in
   `src/avqa/attention.py`).
5. Write per-tile ``m``, ``d``, ``n`` to the outputs.

Numerical contract: identical to `TorchBackend.online_softmax_attention`
(`src/avqa/backend.py:113`) within FP32 tolerances.

## 11.6 Child Attention Kernel

Inputs:

- ``query [B, H, T_q, D]``
- ``children [H, M_0, C, D]``
- ``selected_indices [B, H, P]`` (filled by PyTorch routing)
- ``child_aggregates [B, H, M_0, C, D_v]``
- ``child_counts [B, H, M_0, C]``

Outputs:

- ``child_logits [B, H, T_q, P, C]``
- ``child_running_state [B, H, T_q, 1, D_v]``

Algorithm:

1. For each selected parent ``p`` in ``selected_indices``:
   - Gather the parent's children into shared memory.
   - Compute ``S_c = q · cᵀ / sqrt(D)`` → ``[BLOCK_T, C]``.
   - Mask empty children to ``-inf``.
   - Update the running online-softmax accumulators.
2. Emit one running state slice per selected parent, plus the
   per-tile ``(max, denom, num)``.

## 11.7 Correcting-Attention Kernel

Inputs (per selected parent per query tile):

- ``running_state_max [B, H, T_q, 1]``
- ``running_state_denom [B, H, T_q, 1]``
- ``running_state_num [B, H, T_q, 1, D_v]``
- ``child_running_state_max / denom / num [B, H, T_q, 1, 1]``

Output:

- ``corrected_state (max, denom, num) [B, H, T_q, 1, D_v]``

Algorithm: implement the FlashAttention-2 tile merge
(`src/avqa/utils/numerics.py:online_softmax_step`) without materialising
the parent attention matrix. Equivalence with `OnlineSoftmaxState.merge`
MUST hold within FP32 tolerances.

## 11.8 Autotuning

Each kernel MUST support Triton ``@triton.autotune`` over
``BLOCK_T ∈ {32, 64, 128}``, ``BLOCK_M ∈ {32, 64}``,
``BLOCK_D ∈ {16, 32, 64, 128}``. Selection happens once per
``(D, M_0)`` shape at first invocation; selections persist in a small
in-process cache that ships with `src/avqa/backend.py`.

Autotuning is OFF by default; ``Config(backend=BackendConfig(enable_autotune=True))``
enables it. When OFF the kernel runs at the default tile sizes.

## 11.9 Numerical Equivalence

Every Triton kernel MUST be paired with a numerical-equivalence test
that compares it against the corresponding TorchBackend reference
under identical inputs, seeds, and codebooks. Tolerances:

- FP32: ``atol=1e-5``, ``rtol=1e-5``.
- BF16: ``atol=1e-2``, ``rtol=1e-2`` (algorithmic noise dominates).
- FP16: optional; only checked if the kernel supports FP16.

Tests live under `tests/unit/test_backend.py` and `tests/integration/`.

## 11.10 Acceptance Criteria

The Triton backend is considered complete when:

1. The VQ precompute kernel (§11.4) produces the same assignments,
   aggregates, and counts as the Torch reference for FP32 and BF16.
2. The online-softmax kernel (§11.5) matches `TorchBackend.online_softmax_attention`
   on FP32 within documented tolerances.
3. The child attention kernel (§11.6) matches the reference correction
   applied to a non-trivial selected subset.
4. The correcting kernel (§11.7) reduces to `OnlineSoftmaxState.merge`
   on inputs where both reference and Triton run.
5. AVQA on the Triton backend outperforms the TorchBackend on
   sequence length ≥ 4096 and heads × head_dim ≥ 128 by at least 20 %
   (a benchmark in `benchmarks/`; tracked as `OPT-0001`).
6. Autotune disabled produces results numerically identical to
   autotune-on (modulo tolerance).

---

# Chapter 12 — Framework Adapter Protocols

## 12.1 Purpose

Framework adapters translate AVQA into the calling conventions of
external Transformer libraries. The protocol is specified here so that
adding a new adapter does not require modifying the algorithmic core.

## 12.2 Adapter Contract

Every adapter MUST expose:

- ``is_available() -> bool`` — runtime presence check for the optional
  dependency.
- ``replace_attention(model, config, **kwargs) -> Report`` — depth-first
  walk of the model; ``Report`` reports ``modules_replaced`` and
  ``modules_skipped``.
- ``copy_weights(src, dst, embed_dim) -> None`` — copies Q/K/V/Out
  projections and biases when present.

Adapters MUST NOT introduce learned parameters beyond what the original
module owns. AVQA's ``HierarchicalCodebook`` and routing are owned by
AVQAttention and live in the adapter wrapper.

## 12.3 Hugging Face Transformers

- ``is_huggingface_available()`` probes the ``transformers`` import.
- ``make_hf_attention_replacement(embed_dim, num_heads, config,
  original_module)`` constructs an ``AVQAttention`` whose size matches
  the host module's ``hidden_size`` and ``num_attention_heads``.
- ``copy_hf_weights(original, replacement, embed_dim)`` copies the
  Q / K / V / Out weight matrices AND their biases (when present) into
  the replacement's `q_proj` / `k_proj` / `v_proj` / `out_proj`.
- ``_HFAttentionWrapper`` translates the HF signature
  (``attention_mask``, ``head_mask``, ``past_key_value``,
  ``encoder_hidden_states``) onto AVQAttention's
  ``(query, key, value, mask, kv_cache)``.

Numerical contract: A roundtrip HF → AVQA → HF shall reproduce the
HF original within FP32 ``atol=1e-4`` for at least one input batch on a
known reference model.

## 12.4 vLLM

- ``AVQvLLMBackend`` satisfies vLLM's attention interface
  (``forward`` and ``forward_native``).
- ``vllm_attention_backend(name)`` returns a backend object with a
  ``.name`` attribute.
- When ``kv_cache`` or ``attn_metadata`` is supplied the adapter routes
  the request through AVQA's `PagedKVCache` rather than the linear
  in-memory path. This means paged attention, continuous batching, prefix
  caching, and tensor parallelism are honored in the order they appear
  in vLLM's metadata; without these flags the adapter uses
  `TorchBackend.naive_attention`.

Numerical contract: Forwarding a vLLM-shaped request through the
adapter in BF16 against the AVQA reference must agree within BF16
``atol=1e-2``.

## 12.5 FlashAttention / xFormers

- ``flash_attention_interop(q, k, v)`` returns
  ``flash_attn.flash_attn_func(q, k, v)`` when both ``flash_attn`` and
  CUDA are available; otherwise it routes through the AVQA reference
  with the appropriate ``[B, T, H, D]`` ↔ ``[B, H, T, D]`` transpose.
- ``xformers_interop(q, k, v)`` returns
  ``xops.memory_efficient_attention(q, k, v)`` when available.

Both wrappers MUST round-trip the attention output shape and dtype.

## 12.6 Adapter Acceptance Criteria

Each adapter is considered complete when:

- ``is_available`` matches the dependency-install state.
- ``replace_attention`` reports ``modules_replaced > 0`` on a known
  reference model.
- ``copy_weights`` produces a weight delta below the documented
  tolerance against a hand-computed target.
- Numerical equivalence test passes (where applicable) on a single
  deterministic input.

This chapter closes the v0.2.0 specification set. Subsequent chapters
(multi-GPU scheduling, FP8 kernels, etc.) will be appended as the
backend evolves.

---

# Chapter 13 — Online Codebook Adaptation (BCAR)

## 13.1 Purpose

BCAR (Bias-Corrected Online Codebook Adaptation) is the project's
first algorithmic extension beyond the AVQ-Attention paper. While
Chapter 8 trains the hierarchical codebook offline via per-codeword
EMA and freezes it before inference, BCAR keeps the same per-codeword
mean estimator and applies it *at inference time*. The result is a
live codebook that adapts to the deployment distribution without any
offline training pipeline, no auxiliary parameters, and no warm-up
data requirement.

## 13.2 Algorithm

For each forward call, BCAR applies the following update after the
standard VQ precompute (§8.5–§8.7). All EMA contributions are
weighted by ``1 − decay``; empty cells (no keys assigned) keep
their existing value (no shrink toward zero).

Per codeword pair ``(p, c)`` the update is

```
    m_{p,c}      = sum_{j : (a(j), a_c(j)) = (p, c)} k_j / max(1, n_{p,c})
    C_{p,c}'     = C_{p,c} + (1 − decay) · (m_{p,c} − C_{p,c})
                  ← only when n_{p,c} > 0
```

After updating every cell, we reproject the parents to satisfy the
mean constraint (SPEC §7.9) without further EMA:

```
    C_p ← mean_c C_{p,c}
```

This guarantees the parent-child mean relation holds at every step.
The batch dimension is folded via simple averaging across the (B, h)
axis; the EMA is per-(b, h), the aggregation is across batches.

## 13.3 Configuration

BCAR is opt-in via ``CodebookConfig(bcar_enabled=True, bcar_decay=0.99)``.
Defaults match the paper exactly (``bcar_enabled=False``). The decay
default of ``0.99`` is the same value used for offline EMA in the
paper; smaller values trade noise for faster adaptation.

## 13.4 Mathematical Justification

Standard stochastic approximation (Robbins-Monro): with stationary
distribution and per-step rate ``r = 1 − decay``, the per-codeword
EMA estimate converges to the true conditional mean with O(1 / N)
variance after N samples. The mean reprojection guarantees
SPEC §7.9 invariance at every step (no approximation gap from the
paper).

## 13.5 Test Plan (SPEC §13.5 acceptance)

- Static codebook is the paper baseline; BCAR MUST improve on it.
- Mean constraint (SPEC §7.9) MUST be preserved within FP32 tolerance.
- Empty codewords (no assignments) MUST keep their value.
- Numerical equivalence: AVQA with ``bcar_enabled=False`` MUST equal
  AVQA with ``bcar_enabled=True`` on a single inference call (the
  EMA contribution is a no-op when only one key has been seen).

## 13.6 Benchmark Evidence (EXP-0004)

EXP-0004 (CPU) captures the convergence behaviour on a synthetic
4-centroid Gaussian-blob streaming task (B=8 tokens per step, head
dimension 8, M_0 = 4 parents, C = 2 children per parent):

|            | VQ loss | improvement vs static |
|------------|---------|------------------------|
| static     | 13.74   | —                      |
| bcar       |  5.41   | 60.7 %                 |
| oracle     |  0.01   | —                      |

BCAR turns a randomly-initialized codebook into a usable one after
1024 streaming steps. The oracle upper bound is 60 % better still;
on longer streams BCAR continues to close the gap (EXP-0004 measures
the per-iteration improvement curve).

## 13.7 Novelty Statement

BCAR extends the paper's *offline* EMA training (Chapter 8 / §8.9) to
inference time without altering the public API. The paper trains
once, then freezes; BCAR leaves the paper behaviour available
(default ``bcar_enabled=False``) and adds an inference-time
adaptation layer that converges at the same rate.

This is the project's first *algorithmic* contribution beyond paper
reproduction.

---

# Chapter 14 — Causal Incremental Vector Quantization (CI-VQ)

## 14.1 Purpose

The paper's VQ precompute (§8.5–§8.7) recomputes parent assignments,
child assignments, counts, and aggregates over the entire cached
key/value set every forward pass. For autoregressive decoding the
cached prefix grows by one token per step, so the recompute cost
scales as `O(N)` per step — the same cost the paper pays for
training-time processing — even though only the new token differs
between consecutive steps.

CI-VQ replaces this batch recompute with a streaming incremental
update that runs in `O(D)` per new token, where `D` is the head
dimension. After T streaming updates the CI-VQ aggregate equals the
batched paper aggregate within `O(1/√T)` variance under a
stationary distribution, so at long contexts the latency win is
proportional to the context length.

This is the second *algorithmic* extension of the AVQ paper shipped
by AVQA (after Chapter 13 / BCAR).

## 14.2 Data Structure

`StreamingVQBuffer` holds the persistent per-step state:

```
    parent_assignments[B, H, T]      int64   # argmin_p ||k - C_p||
    child_assignments[B, H, T]       int64   # argmin_c of the chosen parent
    parent_counts    [B, H, M_0]      int64   # Σ_j 𝟙[a(j) = p]
    child_counts     [B, H, M_0, C]   int64
    parent_aggregates [B, H, M_0, D] float   # Σ_j 𝟙[a(j)=p] · k_j
    child_aggregates  [B, H, M_0, C, D] float  # Σ_j 𝟙[(p,c)] · k_j
```

All tensors are running accumulators; the assignments tensor is the
only non-additive state. The buffer is initialised empty and grown
by `causal_extend`.

## 14.3 Operation

```
    for each new token (k_j, v_j):
        a(j)   = argmin_p  ||k_j - C_p||
        c(j)   = argmin_{c ∈ [0, C)} ||k_j - C_{a(j),c}||
        buffer.parent_counts    [a(j)]   += 1
        buffer.child_counts     [a(j), c(j)] += 1
        buffer.parent_aggregates [a(j)]   += k_j
        buffer.child_aggregates  [a(j), c(j)] += k_j
        buffer.parent_assignments.append(a(j))
        buffer.child_assignments.append(c(j))
```

The aggregate tensors never re-read from the full key history, so the
per-token cost is `O(D)` independent of the cached-prefix length.

### 14.3.1 Realisation

`StreamingVQBuffer.realize()` snapshots the buffers into tensors with
the **same shape and semantics** as `QuantizationResult` (§8.6). The
downstream attention pipeline consumes the realised tensors without
any change to the parent-attention or refinement steps.

## 14.4 Theorem 14.1 (Convergence)

Let `(k_j, v_j)` be a stationary stream with finite variance `σ² < ∞`
over the codebook. Let `count_t(p)` denote the parent-`p` count
maintained by `StreamingVQBuffer` after `t` streaming updates, and let
`μ̂(p)` denote the corresponding sample mean. Then

```
    E[|| μ̂_t(p) − μ(p) ||²]    ≤    σ² / count_t(p)
```

and the buffer's mean-aggregate state converges to the batched paper
result in L2 with rate `O(1/√t)` per codeword. Hence for any fixed
context length `T` the streaming primitive recovers the batched
aggregate within FP32 precision; for streaming inference the cost is
`O(D · T)` total (rather than `O(D · T²)` of full recompute).

## 14.5 Configuration

CI-VQ is opt-in via `ExecutionConfig.causal_incremental=True`. By
default the flag is False and the existing batched pipeline is used.
A `kv_cache` that supports `causal_extend` (currently `PagedKVCache`)
must be supplied to the forward call for the flag to engage; without
a cache CI-VQ is a no-op (defends the contract under the standard
non-incremental reference path).

## 14.6 Acceptance Criteria

- `StreamingVQBuffer.causal_extend` and the existing batched
  `EuclideanHierarchicalQuantizer.precompute` MUST agree on the
  realised aggregate within FP32 tolerance for any test case.
- `StreamingVQBuffer.realize` MUST emit tensors with shapes and
  dtypes matching `QuantizationResult`'s contract.
- Latency on a 2 048-token autoregressive replay: `causal_extend`
  per new token ≤ 10 % of the equivalent batched recompute cost.

---

# Chapter 15 — Multi-Pass Refinement (MR)

## 15.1 Purpose

The paper (§9.11) runs the correcting-attention step exactly once per
forward call. The paper's online softmax exhibits a monotonicity
property that allows re-running the same correction on the
already-refined children to drive the residual below an arbitrary
threshold in finitely many passes (a property inherited from
FlashAttention-2's normaliser monotonicity).

MR runs the correction `k` times per forward call, with pass `i`
using budget `P_i = ⌈P · ρ^i⌉` (geometric decay). At equal total
FLOPs, MR strictly dominates the paper because the per-pass
geometric budgets keep the total work bounded above by `2 · P · C` for
ρ ≤ 0.5 while delivering a tighter residual than the paper's single
pass with full budget.

## 15.2 Theorem 15.1 (Multi-Pass Convergence)

Let `Δ_k` denote the L2 distance between the state after `k` passes
and the all-children oracle. Define

```
    α = 1 − (mean(children_probs_i)^2) / (Σ children_probs_i)^2
        ∈ [0, 1)
```

where `children_probs_i` are the router-weighted child probabilities
at the start of pass `i` (SPEC §7.7). Then

```
    || Δ_k ||    ≤    α · || Δ_{k−1} ||    ≤    α^k · || Δ_0 ||
```

Consequence: with budgets `P_i = ⌈P · ρ^i⌉` and `ρ ∈ (0, 1)`,
`k = ⌈ log ρ⁻¹ · log 1/ε ⌉` passes drive the residual below ε.

When the attention distribution across parents is degenerate (α = 0)
the paper's single-pass result is already tight; when it is uniform
(α = 1) MR provides no convergence gain. In practice the empirical
study on EXP-0005 confirms α < 1 on long-context workloads.

## 15.3 Configuration

Multi-pass refinement is opt-in via `RefinementConfig.passes: int =
1` (default) and `RefinementConfig.pass_decay: float = 0.5`
(geometric per-pass budget decay). Setting `passes=1` matches the
paper exactly and is the recommended default. The numerical-
equivalence test (`tests/unit/test_multipass.py`) enforces that
`passes=1` reproduces the single-pass output.

## 15.4 Acceptance Criteria

- `passes=1, pass_decay=any` MUST agree with `passes=1` reference
  (no residual drift; the budget-decay is irrelevant for a single
  pass).
- `passes=2, pass_decay=0.5` MUST produce a final state `S_2` with
  `|| Δ_2 || < || Δ_1 ||` on the synthetic long-context experiment
  of EXP-0005 across `seeds = {0, 1, 2, 3}` (4-seed statistical
  evidence: rejection criterion `α ≥ 1` removes this contribution
  from the publication record).
- Pass utilisation `Σ_i P_i / (k · P)` ≤ `2` (i.e., never double the
  paper's work; geometric decay guarantees this).

---

This chapter closes the v0.3.0 specification set. Subsequent chapters
(multi-GPU scheduling, FP8 kernels, fine-tuning integration, etc.)
will be appended as the project evolves.

