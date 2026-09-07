# BENCHMARKS.md

> **Project:** AVQA – Adaptive Vector Quantized Attention
>
> **Purpose:** Benchmarking Protocol, Performance Validation, and Reproducibility Guide
>
> This document defines the canonical benchmark methodology for AVQA.
>
> Every benchmark, performance claim, optimization, and publication SHALL follow this protocol.
>
> Benchmark results produced outside this protocol SHALL NOT be considered authoritative.

---

# Benchmark Philosophy

The purpose of benchmarking is to evaluate:

- Correctness
- Performance
- Memory efficiency
- Scalability
- Numerical stability
- Reproducibility

Benchmarks SHALL measure real improvements rather than isolated micro-optimizations.

Performance improvements MUST NEVER compromise correctness.

---

# Benchmark Workflow

Every benchmark follows:

```text
Implementation
        │
        ▼
Correctness Validation
        │
        ▼
Regression Testing
        │
        ▼
Micro Benchmarks
        │
        ▼
Macro Benchmarks
        │
        ▼
Scalability Analysis
        │
        ▼
Statistical Analysis
        │
        ▼
Publication Report
```

Benchmarking SHALL begin only after correctness has been established.

---

# Benchmark Categories

## Functional

Verify:

- numerical equivalence
- deterministic execution
- serialization
- backward compatibility

---

## Performance

Measure:

- latency
- throughput
- tokens/sec
- sequences/sec

---

## Memory

Measure:

- peak memory
- average memory
- allocation count
- allocation size
- memory fragmentation

---

## GPU

Measure:

- SM occupancy
- HBM bandwidth
- L2 cache hit rate
- L1 cache hit rate
- register usage
- shared memory usage
- kernel launches
- synchronization overhead

---

## Scalability

Evaluate scaling with:

- sequence length
- batch size
- attention heads
- embedding dimension
- codebook size
- refinement budget

---

# Baselines

Every benchmark SHALL compare against:

- PyTorch SDPA
- FlashAttention
- FlashAttention-2
- FlashAttention-3 (if available)
- xFormers
- AVQA Reference Backend
- AVQA Triton Backend

New baselines MAY be added but existing baselines SHALL remain unless deprecated.

---

# Hardware Specification

Every benchmark MUST record:

CPU

- model
- cores
- threads

GPU

- model
- VRAM
- driver

Software

- Python
- PyTorch
- Triton
- CUDA
- cuDNN
- OS

Benchmark reports without environment information SHALL be considered incomplete.

---

# Benchmark Configuration

Record:

- precision
- batch size
- sequence length
- hidden dimension
- attention heads
- causal/non-causal
- dropout
- refinement budget
- codebook size

Configurations SHALL be stored as machine-readable files under:

```text
benchmarks/configs/
```

---

# Models

Benchmark using multiple representative models.

Minimum:

- Small
- Base
- Large

Preferred:

- 1B
- 3B
- 7B
- 13B
- 34B
- 70B

Document any deviations.

---

# Sequence Lengths

Minimum benchmark set:

- 512
- 1k
- 2k
- 4k
- 8k
- 16k
- 32k
- 64k
- 128k

Long-context benchmarks SHALL be included whenever supported by the model.

---

# Warm-Up

Every benchmark SHALL include warm-up iterations.

Minimum:

- 20 warm-up iterations

Warm-up measurements SHALL NOT be included in reported statistics.

---

# Repetitions

Each benchmark SHALL execute:

Minimum:

- 30 repetitions

Preferred:

- 100 repetitions

Single-run benchmarks are prohibited.

---

# Metrics

Every benchmark SHALL report:

Performance

- latency
- throughput
- tokens/sec

Memory

- peak memory
- average memory

Quality

- perplexity
- accuracy
- task-specific metrics

Compute

- FLOPs
- utilization
- bandwidth

---

# Statistical Analysis

For every metric report:

- mean
- median
- minimum
- maximum
- standard deviation
- 95% confidence interval

Where appropriate, report effect sizes and statistical significance.

---

# Correctness Validation

Before reporting performance:

Verify numerical equivalence with the reference implementation.

Tolerance SHALL be documented for each precision mode.

Performance benchmarks without correctness validation are invalid.

---

# Ablation Protocol

Every optimization SHALL include:

Baseline

↓

Partial implementation

↓

Complete implementation

Evaluate the contribution of each component independently.

---

# Profiling

Collect:

- PyTorch profiler
- Nsight Systems
- Nsight Compute (GPU kernels)
- Memory timeline

Archive profiler outputs under:

```text
benchmarks/profiling/
```

---

# Result Storage

Store:

```text
benchmarks/

    raw/
    processed/
    csv/
    json/
    plots/
    reports/
```

Raw data SHALL never be modified after collection.

Derived reports SHALL reference raw data.

---

# Visualization

Generate:

- latency curves
- throughput curves
- memory curves
- scaling plots
- roofline plots
- occupancy plots
- speedup plots
- Pareto frontiers

Every figure SHALL be reproducible from stored raw data.

---

# Reproducibility

Every benchmark SHALL be executable with:

```bash
python benchmarks/run.py \
    --config configs/...
```

No manual benchmark procedures are permitted.

---

# Benchmark Acceptance Criteria

A benchmark is considered valid only if:

- Environment is fully documented.
- Correctness verified.
- Warm-up completed.
- Required repetitions completed.
- Statistical analysis performed.
- Raw data archived.
- Figures reproducible.
- Configuration preserved.
- Scripts committed.
- Results independently reproducible.

---

# Performance Claims

No performance claim may be made unless supported by benchmark data produced under this protocol.

Marketing-style statements such as:

- "much faster"
- "dramatically better"
- "significantly improved"

are prohibited unless accompanied by quantitative evidence.

Every claim SHALL specify:

- baseline
- workload
- hardware
- metric
- confidence interval

---

# Continuous Benchmarking

The CI pipeline SHALL execute a representative benchmark suite on supported hardware.

Every pull request SHALL be checked for:

- performance regressions
- memory regressions
- correctness regressions

Thresholds SHALL be configurable.

Performance regressions beyond accepted tolerances SHALL block merging until reviewed.

---

# Definition of Done

Benchmarking is complete when:

- All benchmark suites execute successfully.
- Correctness is verified.
- Statistical analysis is complete.
- Raw data is archived.
- Reports are generated.
- Figures are reproducible.
- Results can be reproduced on equivalent hardware using the documented benchmark scripts.

This document defines the only accepted methodology for evaluating AVQA performance. Any published benchmark, optimization, or research result SHALL conform to this protocol.
