---
layout: default
title: AVQA
---

# AVQA — Adaptive Vector Quantized Attention

AVQA is a production-grade Python library implementing Adaptive Vector
Quantized Attention (AVQ-Attention) as a drop-in attention backend for
PyTorch-based Transformer architectures.

## Quick links

- [README](../README.md) — installation, quick-start, API reference
- [Architecture](../docs/architecture.md) — layered view of the package
- [Mathematical formulation](../docs/math.md) — spec derivation
- [API reference](../docs/api/README.md) — module-by-module contracts
- [Benchmarks](../BENCHMARKS.md) — performance methodology
- [Release notes](../RELEASE.md) — compatibility and known limitations

## Why AVQA?

- **Problem** — vanilla attention is O(N²); it dominates long-context
  compute and memory.
- **Solution** — AVQA quantizes keys into a hierarchical codebook, computes
  attention at the codebook level, then refines only the most-attended
  codewords.
- **Audience** — PyTorch users training or serving Transformer models
  with long sequences (≥1k tokens) where memory and FLOPs are the
  bottleneck.
- **Differentiator** — drop-in `nn.Module` plus algorithmic contributions
  (BCAR online codebook adaptation, HVAQ temperature schedules,
  multi-pass refinement with disjoint-set re-routing).

## Status

This is an independent, community-driven implementation of the paper
[AVQ-Attention: Adaptive Vector-Quantized Attention](https://arxiv.org/abs/2607.12789).
The author of this codebase is not an author of the paper and is not
affiliated with the paper's authors or their institutions.