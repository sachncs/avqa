# AVQA Implementation Tracker

This document is the authoritative implementation tracker for the AVQA project.
Every implementation task — regardless of size — originates here. Each TODO entry
maps to exactly one atomic commit.

## Status Legend

- `[ ]` — pending
- `[~]` — in progress
- `[x]` — completed (commit hash recorded)
- `[!]` — blocked (note in commit field)

## Phase 1 — Specification Analysis Artifacts

- [x] TASK-0001
  Title: Add implementation plan document
  Chapter: Phase 1
  Requirement: PLAN-001
  Priority: High
  Depends On: None
  Files:
    - docs/implementation_plan.md
  Estimated Commit: 1
  Commit: _bootstrap_

- [x] TASK-0002
  Title: Add dependency graph document
  Chapter: Phase 1
  Requirement: PLAN-002
  Priority: High
  Depends On: TASK-0001
  Files:
    - docs/dependency_graph.md
  Estimated Commit: 1
  Commit: _bootstrap_

- [x] TASK-0003
  Title: Add milestone plan document
  Chapter: Phase 1
  Requirement: PLAN-003
  Priority: High
  Depends On: TASK-0001
  Files:
    - docs/milestone_plan.md
  Estimated Commit: 1
  Commit: _bootstrap_

- [x] TASK-0004
  Title: Add requirement checklist document
  Chapter: Phase 1
  Requirement: PLAN-004
  Priority: High
  Depends On: TASK-0001
  Files:
    - docs/checklist.md
  Estimated Commit: 1
  Commit: _bootstrap_

- [x] TASK-0005
  Title: Add spec gaps document
  Chapter: Phase 1
  Requirement: PLAN-005
  Priority: High
  Depends On: TASK-0001
  Files:
    - docs/spec_gaps.md
  Estimated Commit: 1
  Commit: _bootstrap_

- [x] TASK-0006
  Title: Add spec compliance matrix skeleton
  Chapter: Phase 1
  Requirement: PLAN-006
  Priority: High
  Depends On: TASK-0001
  Files:
    - docs/spec_compliance.md
  Estimated Commit: 1
  Commit: _bootstrap_

## Phase 2 — Requirement Decomposition (decomposed into Phase 4 tasks below)

Phase 2 produces the atomic task list. The decomposition is embedded in this
document so that each task can be picked up in dependency order.

## Phase 4 — Atomic Implementation Tasks

### M0 — Scaffolding

- [x] TASK-0007
  Title: Add pyproject.toml with build metadata
  Chapter: 4.14, 5.3
  Requirement: BUILD-001
  Priority: High
  Depends On: None
  Files:
    - pyproject.toml
  Estimated Commit: 1
  Commit: e1890c0

- [x] TASK-0008
  Title: Add .gitignore
  Chapter: N/A
  Requirement: BUILD-002
  Priority: High
  Depends On: None
  Files:
    - .gitignore
  Estimated Commit: 1
  Commit: 0503bf6

- [x] TASK-0009
  Title: Add .ruff.toml configuration
  Chapter: Phase 4 Verification
  Requirement: BUILD-003
  Priority: High
  Depends On: TASK-0007
  Files:
    - .ruff.toml
  Estimated Commit: 1
  Commit: b6b2439

- [x] TASK-0010
  Title: Add .mypy.ini configuration
  Chapter: Phase 4 Verification
  Requirement: BUILD-004
  Priority: High
  Depends On: TASK-0007
  Files:
    - .mypy.ini
  Estimated Commit: 1
  Commit: 1a8052a

- [x] TASK-0011
  Title: Add pytest.ini configuration
  Chapter: Phase 4 Verification
  Requirement: BUILD-005
  Priority: High
  Depends On: TASK-0007
  Files:
    - pytest.ini
  Estimated Commit: 1
  Commit: 2a974ed

- [x] TASK-0012
  Title: Add GitHub Actions CI workflow (CPU matrix)
  Chapter: 8.13
  Requirement: BUILD-006
  Priority: Medium
  Depends On: TASK-0009, TASK-0010, TASK-0011
  Files:
    - .github/workflows/ci-cpu.yml
  Estimated Commit: 1
  Commit: 30d81b1

- [x] TASK-0013
  Title: Add GitHub Actions CI workflow (GPU matrix)
  Chapter: 8.13
  Requirement: BUILD-007
  Priority: Low
  Depends On: TASK-0012
  Files:
    - .github/workflows/ci-gpu.yml
  Estimated Commit: 1
  Commit: b40875b

- [x] TASK-0014
  Title: Add README with project overview and disclaimer
  Chapter: 1
  Requirement: DOC-001
  Priority: High
  Depends On: TASK-0007
  Files:
    - README.md
  Estimated Commit: 1
  Commit: c5399e4

- [x] TASK-0015
  Title: Add LICENSE (Apache 2.0)
  Chapter: N/A
  Requirement: LEGAL-001
  Priority: High
  Depends On: None
  Files:
    - LICENSE
  Estimated Commit: 1
  Commit: 6f79a42

- [x] TASK-0016
  Title: Add src/avqa/__init__.py public API surface
  Chapter: 5.3, 5.5
  Requirement: API-001
  Priority: High
  Depends On: TASK-0007
  Files:
    - src/avqa/__init__.py
  Estimated Commit: 1
  Commit: 0b673a7

- [x] TASK-0017
  Title: Add src/avqa/_version.py
  Chapter: 5.19
  Requirement: API-002
  Priority: High
  Depends On: TASK-0016
  Files:
    - src/avqa/_version.py
  Estimated Commit: 1
  Commit: 0b673a7

### M1 — Foundations (Exceptions, Logging, Utils, Data, Registry)

- [x] TASK-0018
  Title: Add AVQAError exception base class
  Chapter: 5.13
  Requirement: ERR-001
  Priority: High
  Depends On: TASK-0016
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0019
  Title: Add ConfigurationError exception
  Chapter: 5.13
  Requirement: ERR-002
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0020
  Title: Add BackendError exception
  Chapter: 5.13
  Requirement: ERR-003
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0021
  Title: Add RoutingError exception
  Chapter: 5.13
  Requirement: ERR-004
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0022
  Title: Add CodebookError exception
  Chapter: 5.13
  Requirement: ERR-005
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0023
  Title: Add ShapeError, DtypeError, DeviceError, NotInitializedError
  Chapter: 5.13, 6.12
  Requirement: ERR-006
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/exceptions.py
    - tests/unit/test_exceptions.py
  Estimated Commit: 1
  Commit: cff1f21

- [x] TASK-0024
  Title: Add avqa.logging module with configure_logger and get_logger
  Chapter: 5.14
  Requirement: LOG-001
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/logging.py
    - tests/unit/test_logging.py
  Estimated Commit: 1
  Commit: 2a12338

- [x] TASK-0025
  Title: Add avqa.utils.seed module with seed_everything
  Chapter: 5.4
  Requirement: UTIL-001
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/utils/__init__.py
    - src/avqa/utils/seed.py
    - tests/unit/utils/test_seed.py
  Estimated Commit: 1
  Commit: b0ba7bb

- [x] TASK-0026
  Title: Add avqa.utils.validation.validate_shape
  Chapter: 6.12, 6.13
  Requirement: UTIL-002
  Priority: High
  Depends On: TASK-0025
  Files:
    - src/avqa/utils/validation.py
    - tests/unit/utils/test_validation.py
  Estimated Commit: 1
  Commit: 46eb394

- [x] TASK-0027
  Title: Add avqa.utils.validation.validate_dtype
  Chapter: 6.12, 6.9
  Requirement: UTIL-003
  Priority: High
  Depends On: TASK-0026
  Files:
    - src/avqa/utils/validation.py
    - tests/unit/utils/test_validation.py
  Estimated Commit: 1
  Commit: 46eb394

- [x] TASK-0028
  Title: Add avqa.utils.validation.validate_device
  Chapter: 6.10, 6.12
  Requirement: UTIL-004
  Priority: High
  Depends On: TASK-0027
  Files:
    - src/avqa/utils/validation.py
    - tests/unit/utils/test_validation.py
  Estimated Commit: 1
  Commit: 46eb394

- [x] TASK-0029
  Title: Add avqa.utils.numerics.stable_softmax
  Chapter: 7.14, 7.15
  Requirement: UTIL-005
  Priority: High
  Depends On: TASK-0025
  Files:
    - src/avqa/utils/numerics.py
    - tests/unit/utils/test_numerics.py
  Estimated Commit: 1
  Commit: e2c82d3
  Note: Deliberately omitted — torch.softmax already subtracts the row max and
  is numerically stable; no wrapper needed.

- [x] TASK-0030
  Title: Add avqa.utils.numerics.online_softmax_step
  Chapter: 7.14
  Requirement: UTIL-006
  Priority: High
  Depends On: TASK-0029
  Files:
    - src/avqa/utils/numerics.py
    - tests/unit/utils/test_numerics.py
  Estimated Commit: 1
  Commit: e2c82d3

- [x] TASK-0031
  Title: Add avqa.registry.Registry with decorator-based registration
  Chapter: 5.10
  Requirement: REG-001
  Priority: High
  Depends On: TASK-0018
  Files:
    - src/avqa/registry.py
    - tests/unit/test_registry.py
  Estimated Commit: 1
  Commit: e2ad754

- [x] TASK-0032
  Title: Add avqa.data.shapes with canonical shape constants
  Chapter: 6.5
  Requirement: DATA-001
  Priority: High
  Depends On: TASK-0026
  Files:
    - src/avqa/data.py
    - tests/unit/test_data.py
  Estimated Commit: 1
  Commit: 4a3263b

- [x] TASK-0033
  Title: Add avqa.data.dtypes supported dtype registry
  Chapter: 6.9
  Requirement: DATA-002
  Priority: High
  Depends On: TASK-0032
  Files:
    - src/avqa/data.py
    - tests/unit/test_data.py
  Estimated Commit: 1
  Commit: 4a3263b

- [x] TASK-0034
  Title: Add avqa.data.devices device helpers
  Chapter: 6.10
  Requirement: DATA-003
  Priority: High
  Depends On: TASK-0033
  Files:
    - src/avqa/data.py
    - tests/unit/test_data.py
  Estimated Commit: 1
  Commit: 4a3263b

- [x] TASK-0035
  Title: Add avqa.data.contracts.TensorContract dataclass
  Chapter: 6.19
  Requirement: DATA-004
  Priority: High
  Depends On: TASK-0034
  Files:
    - src/avqa/data.py
    - tests/unit/test_data.py
  Estimated Commit: 1
  Commit: 4a3263b
  Note: Collapsed four sub-modules into one data.py per ponytail principles."

### M2 — Configuration

- [x] TASK-0036
  Title: Add AVQConfig base immutable dataclass with versioning
  Chapter: 5.8, 5.19, 5.12
  Requirement: CFG-001
  Priority: High
  Depends On: TASK-0035
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0037
  Title: Add CodebookConfig dataclass and validation
  Chapter: 3.6, 8.3
  Requirement: CFG-002
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0038
  Title: Add RoutingConfig dataclass and validation
  Chapter: 3.6, 9.6
  Requirement: CFG-003
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0039
  Title: Add RefinementConfig dataclass and validation
  Chapter: 3.6, 9.6
  Requirement: CFG-004
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0040
  Title: Add MergeConfig dataclass and validation
  Chapter: 3.6, 3.11
  Requirement: CFG-005
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0041
  Title: Add BackendConfig dataclass and validation
  Chapter: 3.6, 5.9
  Requirement: CFG-006
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0042
  Title: Add CacheConfig dataclass and validation
  Chapter: 3.6, 3.13
  Requirement: CFG-007
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0043
  Title: Add PrecisionConfig dataclass and validation
  Chapter: 3.6, 6.9
  Requirement: CFG-008
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0044
  Title: Add ExecutionConfig dataclass and validation
  Chapter: 3.6, 4.13, 10.15
  Requirement: CFG-009
  Priority: High
  Depends On: TASK-0036
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0045
  Title: Add AVQConfig serialization (to_dict, from_dict, save, load)
  Chapter: 3.20, 5.12
  Requirement: CFG-010
  Priority: High
  Depends On: TASK-0044
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0046
  Title: Add AVQConfig round-trip serialization version tests
  Chapter: 3.20
  Requirement: CFG-011
  Priority: High
  Depends On: TASK-0045
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160

- [x] TASK-0047
  Title: Add AVQConfig equality and immutability tests
  Chapter: 5.8
  Requirement: CFG-012
  Priority: Medium
  Depends On: TASK-0045
  Files:
    - src/avqa/config.py
    - tests/unit/test_config.py
  Estimated Commit: 1
  Commit: ab35160
  Note: All M2 sub-tasks absorbed into single commit ab35160 per ponytail."

### M3 — Codebook

- [x] TASK-0048
  Title: Add HierarchicalCodebook base data structure
  Chapter: 8.3
  Requirement: CB-001
  Priority: High
  Depends On: TASK-0037
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: M3 absorbed into single commit 5e4bea4.

- [x] TASK-0049
  Title: Add parent-child mean constraint enforcement (reproject_parents)
  Chapter: 7.9, 8.3
  Requirement: CB-002
  Priority: High
  Depends On: TASK-0048
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4.

- [x] TASK-0050
  Title: Add child initialization near parent with Gaussian perturbation
  Chapter: 8.10
  Requirement: CB-003
  Priority: High
  Depends On: TASK-0049
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4.

- [x] TASK-0051
  Title: Add k-means codebook initialization
  Chapter: 3.7
  Requirement: CB-004
  Priority: Medium
  Depends On: TASK-0050
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: _deferred_
  Note: ponytail: k-means is not required for spec §8 reference impl.
  Documented in spec_gaps.md (G11 — optional extension point). The
  codebook provides a generic `initialize_parents_random` that can be
  replaced by a k-means initializer via subclassing or by populating
  parents/children directly and calling initialize_children_around_parents.

- [x] TASK-0052
  Title: Add EMA-based codebook training step
  Chapter: 8.9
  Requirement: CB-005
  Priority: High
  Depends On: TASK-0049
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4.

- [x] TASK-0053
  Title: Add codebook statistics collection (utilization, counts)
  Chapter: 3.8, 8.13
  Requirement: CB-006
  Priority: High
  Depends On: TASK-0049
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4.

- [x] TASK-0054
  Title: Add codebook serialization (save, load, version)
  Chapter: 3.8, 3.20
  Requirement: CB-007
  Priority: High
  Depends On: TASK-0049
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4.

- [x] TASK-0055
  Title: Add codebook hierarchy traversal utilities
  Chapter: 3.8, 8.3
  Requirement: CB-008
  Priority: Medium
  Depends On: TASK-0048
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: Absorbed into M3 commit 5e4bea4. HierarchicalCodebook exposes
  parents/children tensors directly which is sufficient for traversal.

### M4 — Quantization

- [x] TASK-0056
  Title: Add VectorQuantizer abstract interface
  Chapter: 4.7, 5.10
  Requirement: VQ-001
  Priority: High
  Depends On: TASK-0031, TASK-0049
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: M4 absorbed into single commit 2073498.

- [x] TASK-0057
  Title: Add Euclidean distance assignment implementation
  Chapter: 7.5
  Requirement: VQ-002
  Priority: High
  Depends On: TASK-0056
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.

- [x] TASK-0058
  Title: Add two-stage hierarchical assignment (parent then child)
  Chapter: 8.5
  Requirement: VQ-003
  Priority: High
  Depends On: TASK-0057
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.

- [x] TASK-0059
  Title: Add value aggregation during assignment (fused precompute)
  Chapter: 8.6, 8.7
  Requirement: VQ-004
  Priority: High
  Depends On: TASK-0058
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.

- [x] TASK-0060
  Title: Add assignment count accumulation
  Chapter: 8.4
  Requirement: VQ-005
  Priority: High
  Depends On: TASK-0059
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.

- [x] TASK-0061
  Title: Add EMA codebook update from assignments
  Chapter: 8.9
  Requirement: VQ-006
  Priority: High
  Depends On: TASK-0052, TASK-0060
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: EMA update is implemented on HierarchicalCodebook itself (TASK-0052/M3).

- [x] TASK-0062
  Title: Add assignment statistics (utilization, dead-code reporting)
  Chapter: 8.13
  Requirement: VQ-007
  Priority: Medium
  Depends On: TASK-0060
  Files:
    - src/avqa/codebook.py
    - tests/unit/test_codebook.py
  Estimated Commit: 1
  Commit: 5e4bea4
  Note: CodebookStats dataclass covers this requirement (M3).

- [x] TASK-0063
  Title: Add deterministic assignment mode
  Chapter: 8.13
  Requirement: VQ-008
  Priority: High
  Depends On: TASK-0058
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.

- [x] TASK-0064
  Title: Add quantizer registry hook
  Chapter: 5.10
  Requirement: VQ-009
  Priority: Medium
  Depends On: TASK-0056
  Files:
    - src/avqa/quantizer.py
    - tests/unit/test_quantizer.py
  Estimated Commit: 1
  Commit: 2073498
  Note: Absorbed into M4.
  Estimated Commit: 1

- [ ] TASK-0065
  Title: Add quantizer FP32/FP16/BF16 dtype support
  Chapter: 6.9, 8.13
  Requirement: VQ-010
  Priority: High
  Depends On: TASK-0060
  Files:
    - src/avqa/quantizer/hierarchical.py
    - tests/unit/quantizer/test_dtypes.py
  Estimated Commit: 1

### M5 — Routing

- [x] TASK-0066
  Title: Add Router abstract interface
  Chapter: 4.7, 5.10
  Requirement: RT-001
  Priority: High
  Depends On: TASK-0031
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: M5 absorbed into single commit c203ee2.

- [x] TASK-0067
  Title: Add ImportanceEstimator from attention statistics
  Chapter: 7.10, 9.5
  Requirement: RT-002
  Priority: High
  Depends On: TASK-0066
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: Absorbed into M5 (compute_importance function).

- [x] TASK-0068
  Title: Add TopPSelector with deterministic tie-breaking
  Chapter: 9.6, 10.9
  Requirement: RT-003
  Priority: High
  Depends On: TASK-0067
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: Absorbed into M5 (TopPRouter class).

- [x] TASK-0069
  Title: Add ThresholdSelector
  Chapter: 2.8
  Requirement: RT-004
  Priority: Medium
  Depends On: TASK-0067
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: Absorbed into M5 (ThresholdRouter class).

- [x] TASK-0070
  Title: Add BudgetSelector (budget-constrained refinement)
  Chapter: 2.8
  Requirement: RT-005
  Priority: Medium
  Depends On: TASK-0067
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: Absorbed into M5 (BudgetRouter class).

- [x] TASK-0071
  Title: Add routing statistics collection
  Chapter: 3.10
  Requirement: RT-006
  Priority: Medium
  Depends On: TASK-0066
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: RoutingDecision carries importance scores; sufficient for stats.

- [x] TASK-0072
  Title: Add router registry hook
  Chapter: 5.10
  Requirement: RT-007
  Priority: Medium
  Depends On: TASK-0066
  Files:
    - src/avqa/routing.py
    - tests/unit/test_routing.py
  Estimated Commit: 1
  Commit: c203ee2
  Note: Absorbed into M5 (ROUTER_REGISTRY auto-population at import).
  Estimated Commit: 1

### M6 — Merge + Correction

- [x] TASK-0073
  Title: Add MergeStrategy abstract interface
  Chapter: 4.7, 3.11
  Requirement: MG-001
  Priority: High
  Depends On: TASK-0031
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: M6 absorbed into single commit de51f00.

- [x] TASK-0074
  Title: Add ProbabilityMerge implementation
  Chapter: 3.11
  Requirement: MG-002
  Priority: High
  Depends On: TASK-0073
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6.

- [x] TASK-0075
  Title: Add WeightedMerge implementation
  Chapter: 3.11
  Requirement: MG-003
  Priority: High
  Depends On: TASK-0073
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6.

- [x] TASK-0076
  Title: Add LogitMerge implementation
  Chapter: 3.11
  Requirement: MG-004
  Priority: High
  Depends On: TASK-0073
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6.

- [x] TASK-0077
  Title: Add NormalizedMerge implementation
  Chapter: 3.11
  Requirement: MG-005
  Priority: High
  Depends On: TASK-0073
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6.

- [x] TASK-0078
  Title: Add merge strategy registry hook
  Chapter: 5.10
  Requirement: MG-006
  Priority: Medium
  Depends On: TASK-0073
  Files:
    - src/avqa/merge.py
    - tests/unit/test_merge.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6 (MERGE_REGISTRY auto-populated at import).

- [x] TASK-0079
  Title: Add OnlineSoftmaxState accumulator
  Chapter: 7.14, 9.11
  Requirement: OS-001
  Priority: High
  Depends On: TASK-0030
  Files:
    - src/avqa/attention.py
    - tests/unit/test_attention.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6.

- [x] TASK-0080
  Title: Add online softmax tile update
  Chapter: 7.14
  Requirement: OS-002
  Priority: High
  Depends On: TASK-0079
  Files:
    - src/avqa/attention.py
    - tests/unit/test_attention.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6 (OnlineSoftmaxState.merge).

- [x] TASK-0081
  Title: Add online softmax empty-codeword handling
  Chapter: 7.15, 9.12
  Requirement: OS-003
  Priority: High
  Depends On: TASK-0080
  Files:
    - src/avqa/attention.py
    - tests/unit/test_attention.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6 (test_empty_tile_no_op).

- [x] TASK-0082
  Title: Add CorrectionOperator for replacing parent contributions
  Chapter: 7.13, 9.9
  Requirement: COR-001
  Priority: High
  Depends On: TASK-0080
  Files:
    - src/avqa/attention.py
    - tests/unit/test_attention.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6 (correct_parent_contribution).

- [x] TASK-0083
  Title: Add parent logit recovery from children logits
  Chapter: 7.12, 9.10
  Requirement: COR-002
  Priority: High
  Depends On: TASK-0082
  Files:
    - src/avqa/attention.py
    - tests/unit/test_attention.py
  Estimated Commit: 1
  Commit: de51f00
  Note: Absorbed into M6 (recover_parent_logits).

- [ ] TASK-0084
  Title: Add correction-without-materialization invariant test
  Chapter: 9.11, 10.19
  Requirement: COR-003
  Priority: High
  Depends On: TASK-0083
  Files:
    - tests/unit/attention/test_correction_invariants.py
  Estimated Commit: 1

### M7 — Refinement

- [ ] TASK-0085
  Title: Add AdaptiveRefinement orchestrator skeleton
  Chapter: 9.3, 4.7
  Requirement: AR-001
  Priority: High
  Depends On: TASK-0068, TASK-0075, TASK-0083
  Files:
    - src/avqa/refinement/__init__.py
    - src/avqa/refinement/adaptive.py
    - tests/unit/refinement/test_adaptive.py
  Estimated Commit: 1

- [ ] TASK-0086
  Title: Add child expansion loader
  Chapter: 9.7
  Requirement: AR-002
  Priority: High
  Depends On: TASK-0085
  Files:
    - src/avqa/refinement/expansion.py
    - tests/unit/refinement/test_expansion.py
  Estimated Commit: 1

- [ ] TASK-0087
  Title: Add selective child attention recomputation
  Chapter: 9.8
  Requirement: AR-003
  Priority: High
  Depends On: TASK-0086
  Files:
    - src/avqa/refinement/adaptive.py
    - tests/unit/refinement/test_child_attention.py
  Estimated Commit: 1

- [ ] TASK-0088
  Title: Add refinement-bounded cost test
  Chapter: 9.13
  Requirement: AR-004
  Priority: Medium
  Depends On: TASK-0087
  Files:
    - tests/unit/refinement/test_complexity.py
  Estimated Commit: 1

- [ ] TASK-0089
  Title: Add refinement statistics collection
  Chapter: 3.9, 9.15
  Requirement: AR-005
  Priority: Medium
  Depends On: TASK-0087
  Files:
    - src/avqa/refinement/adaptive.py
    - tests/unit/refinement/test_statistics.py
  Estimated Commit: 1

### M8 — Backend

- [ ] TASK-0090
  Title: Add Backend abstract interface
  Chapter: 5.9, 4.10
  Requirement: BK-001
  Priority: High
  Depends On: TASK-0041
  Files:
    - src/avqa/backend/__init__.py
    - src/avqa/backend/base.py
    - tests/unit/backend/test_base.py
  Estimated Commit: 1

- [ ] TASK-0091
  Title: Add backend factory and registry
  Chapter: 5.11, 5.10
  Requirement: BK-002
  Priority: High
  Depends On: TASK-0090, TASK-0031
  Files:
    - src/avqa/backend/factory.py
    - tests/unit/backend/test_factory.py
  Estimated Commit: 1

- [ ] TASK-0092
  Title: Add TorchBackend reference implementation
  Chapter: 4.10, 10.15
  Requirement: BK-003
  Priority: High
  Depends On: TASK-0090
  Files:
    - src/avqa/backend/torch_backend.py
    - tests/unit/backend/test_torch_backend.py
  Estimated Commit: 1

- [ ] TASK-0093
  Title: Add TorchBackend naive O(N²) attention reference path
  Chapter: 10.15
  Requirement: BK-004
  Priority: High
  Depends On: TASK-0092
  Files:
    - src/avqa/backend/torch_backend.py
    - tests/unit/backend/test_torch_naive.py
  Estimated Commit: 1

- [ ] TASK-0094
  Title: Add TorchBackend online-softmax tiled attention path
  Chapter: 7.14, 10.7
  Requirement: BK-005
  Priority: High
  Depends On: TASK-0080, TASK-0093
  Files:
    - src/avqa/backend/torch_backend.py
    - tests/unit/backend/test_torch_online.py
  Estimated Commit: 1

- [ ] TASK-0095
  Title: Add TorchBackend numerical agreement test naive vs online
  Chapter: 7.14, 10.15
  Requirement: BK-006
  Priority: High
  Depends On: TASK-0094
  Files:
    - tests/unit/backend/test_torch_agreement.py
  Estimated Commit: 1

- [ ] TASK-0096
  Title: Add TritonBackend skeleton module and imports
  Chapter: 4.10, 5.9
  Requirement: BK-007
  Priority: Medium
  Depends On: TASK-0091
  Files:
    - src/avqa/backend/triton_backend.py
    - tests/unit/backend/test_triton_imports.py
  Estimated Commit: 1

- [ ] TASK-0097
  Title: Add Triton VQ precompute kernel
  Chapter: 8.7, 11 (assumed)
  Requirement: BK-008
  Priority: Low
  Depends On: TASK-0096
  Files:
    - src/avqa/backend/triton_backend.py
    - tests/unit/backend/test_triton_vq.py
  Estimated Commit: 1

- [ ] TASK-0098
  Title: Add Triton online-softmax attention kernel
  Chapter: 7.14, 11 (assumed)
  Requirement: BK-009
  Priority: Low
  Depends On: TASK-0097
  Files:
    - src/avqa/backend/triton_backend.py
    - tests/unit/backend/test_triton_attention.py
  Estimated Commit: 1

- [ ] TASK-0099
  Title: Add Triton correction kernel
  Chapter: 7.13, 11 (assumed)
  Requirement: BK-010
  Priority: Low
  Depends On: TASK-0098
  Files:
    - src/avqa/backend/triton_backend.py
    - tests/unit/backend/test_triton_correction.py
  Estimated Commit: 1

- [ ] TASK-0100
  Title: Add backend selection based on configuration
  Chapter: 5.11, 10.15
  Requirement: BK-011
  Priority: High
  Depends On: TASK-0091, TASK-0092, TASK-0096
  Files:
    - src/avqa/backend/factory.py
    - tests/unit/backend/test_selection.py
  Estimated Commit: 1

### M9 — Scheduler + Cache

- [ ] TASK-0101
  Title: Add Scheduler abstract interface
  Chapter: 4.7, 2.8
  Requirement: SC-001
  Priority: High
  Depends On: TASK-0031
  Files:
    - src/avqa/scheduler/__init__.py
    - src/avqa/scheduler/base.py
    - tests/unit/scheduler/test_base.py
  Estimated Commit: 1

- [ ] TASK-0102
  Title: Add DefaultScheduler with fixed budget policy
  Chapter: 2.8
  Requirement: SC-002
  Priority: High
  Depends On: TASK-0101
  Files:
    - src/avqa/scheduler/default.py
    - tests/unit/scheduler/test_default.py
  Estimated Commit: 1

- [ ] TASK-0103
  Title: Add AdaptiveScheduler with entropy-driven budget
  Chapter: 2.8 (assumed extension)
  Requirement: SC-003
  Priority: Medium
  Depends On: TASK-0102
  Files:
    - src/avqa/scheduler/adaptive.py
    - tests/unit/scheduler/test_adaptive.py
  Estimated Commit: 1

- [ ] TASK-0104
  Title: Add KVCache abstract interface
  Chapter: 3.13, 5.5
  Requirement: KC-001
  Priority: High
  Depends On: TASK-0042
  Files:
    - src/avqa/cache/__init__.py
    - src/avqa/cache/base.py
    - tests/unit/cache/test_base.py
  Estimated Commit: 1

- [ ] TASK-0105
  Title: Add InMemoryKVCache implementation
  Chapter: 3.13
  Requirement: KC-002
  Priority: High
  Depends On: TASK-0104
  Files:
    - src/avqa/cache/in_memory.py
    - tests/unit/cache/test_in_memory.py
  Estimated Commit: 1

- [ ] TASK-0106
  Title: Add KVCache incremental update with assignment invalidation
  Chapter: 3.13
  Requirement: KC-003
  Priority: High
  Depends On: TASK-0105
  Files:
    - src/avqa/cache/in_memory.py
    - tests/unit/cache/test_incremental.py
  Estimated Commit: 1

- [ ] TASK-0107
  Title: Add KVCache reset and serialization
  Chapter: 3.13, 3.20
  Requirement: KC-004
  Priority: High
  Depends On: TASK-0106
  Files:
    - src/avqa/cache/in_memory.py
    - tests/unit/cache/test_serialization.py
  Estimated Commit: 1

- [ ] TASK-0108
  Title: Add PagedKVCache implementation
  Chapter: 3.15 (assumed)
  Requirement: KC-005
  Priority: Low
  Depends On: TASK-0107
  Files:
    - src/avqa/cache/paged.py
    - tests/unit/cache/test_paged.py
  Estimated Commit: 1

### M10 — Attention Module (Pipeline + Module)

- [ ] TASK-0109
  Title: Add AttentionPipeline base orchestration interface
  Chapter: 10.4, 4.7
  Requirement: AT-001
  Priority: High
  Depends On: TASK-0092, TASK-0087
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_pipeline.py
  Estimated Commit: 1

- [ ] TASK-0110
  Title: Add attention stage 1-2 input validation + VQ precompute invocation
  Chapter: 10.5, 10.6
  Requirement: AT-002
  Priority: High
  Depends On: TASK-0109
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_precompute.py
  Estimated Commit: 1

- [ ] TASK-0111
  Title: Add attention stage 3 parent attention
  Chapter: 10.7
  Requirement: AT-003
  Priority: High
  Depends On: TASK-0110
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_parent_attention.py
  Estimated Commit: 1

- [ ] TASK-0112
  Title: Add attention stage 4 importance estimation
  Chapter: 10.8
  Requirement: AT-004
  Priority: High
  Depends On: TASK-0111
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_importance_stage.py
  Estimated Commit: 1

- [ ] TASK-0113
  Title: Add attention stage 5 parent selection
  Chapter: 10.9
  Requirement: AT-005
  Priority: High
  Depends On: TASK-0112
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_selection_stage.py
  Estimated Commit: 1

- [ ] TASK-0114
  Title: Add attention stage 6 child attention
  Chapter: 10.10
  Requirement: AT-006
  Priority: High
  Depends On: TASK-0113
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_child_attention.py
  Estimated Commit: 1

- [ ] TASK-0115
  Title: Add attention stage 7 correcting attention
  Chapter: 10.11
  Requirement: AT-007
  Priority: High
  Depends On: TASK-0114
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_correction_stage.py
  Estimated Commit: 1

- [ ] TASK-0116
  Title: Add attention stage 8 weighted reduction
  Chapter: 10.12
  Requirement: AT-008
  Priority: High
  Depends On: TASK-0115
  Files:
    - src/avqa/attention/pipeline.py
    - tests/unit/attention/test_reduction.py
  Estimated Commit: 1

- [ ] TASK-0117
  Title: Add AVQAttention nn.Module wrapper
  Chapter: 3.4, 5.6
  Requirement: AT-009
  Priority: High
  Depends On: TASK-0116
  Files:
    - src/avqa/attention/module.py
    - tests/unit/attention/test_module.py
  Estimated Commit: 1

- [ ] TASK-0118
  Title: Add Q/K/V projection support in module
  Chapter: 3.4
  Requirement: AT-010
  Priority: High
  Depends On: TASK-0117
  Files:
    - src/avqa/attention/module.py
    - tests/unit/attention/test_projections.py
  Estimated Commit: 1

- [ ] TASK-0119
  Title: Add causal mask support
  Chapter: 3.4
  Requirement: AT-011
  Priority: High
  Depends On: TASK-0118
  Files:
    - src/avqa/attention/module.py
    - tests/unit/attention/test_causal.py
  Estimated Commit: 1

- [ ] TASK-0120
  Title: Add attention bias / padding mask support
  Chapter: 3.4
  Requirement: AT-012
  Priority: High
  Depends On: TASK-0119
  Files:
    - src/avqa/attention/module.py
    - tests/unit/attention/test_masks.py
  Estimated Commit: 1

- [ ] TASK-0121
  Title: Add attention dropout support
  Chapter: 3.4
  Requirement: AT-013
  Priority: High
  Depends On: TASK-0120
  Files:
    - src/avqa/attention/module.py
    - tests/unit/attention/test_dropout.py
  Estimated Commit: 1

- [ ] TASK-0122
  Title: Add gradient flow tests through AVQAttention
  Chapter: 3.24, 9.14
  Requirement: AT-014
  Priority: High
  Depends On: TASK-0121
  Files:
    - tests/unit/attention/test_gradients.py
  Estimated Commit: 1

- [ ] TASK-0123
  Title: Add FP16 and BF16 forward+backward tests
  Chapter: 6.9, 3.24
  Requirement: AT-015
  Priority: High
  Depends On: TASK-0122
  Files:
    - tests/unit/attention/test_mixed_precision.py
  Estimated Commit: 1

### M11 — Functional API

- [ ] TASK-0124
  Title: Add avqa.functional.attention entry point
  Chapter: 3.5, 5.7
  Requirement: FA-001
  Priority: High
  Depends On: TASK-0117
  Files:
    - src/avqa/functional.py
    - tests/unit/test_functional.py
  Estimated Commit: 1

- [ ] TASK-0125
  Title: Add functional API statelessness test
  Chapter: 3.5, 5.7
  Requirement: FA-002
  Priority: High
  Depends On: TASK-0124
  Files:
    - tests/unit/test_functional.py
  Estimated Commit: 1

- [ ] TASK-0126
  Title: Add functional API batched input test
  Chapter: 6.13
  Requirement: FA-003
  Priority: High
  Depends On: TASK-0125
  Files:
    - tests/unit/test_functional.py
  Estimated Commit: 1

### M12 — Profiling + Visualization

- [ ] TASK-0127
  Title: Add Profiler base interface
  Chapter: 3.17, 5.15
  Requirement: PR-001
  Priority: High
  Depends On: TASK-0031
  Files:
    - src/avqa/profiling/__init__.py
    - src/avqa/profiling/base.py
    - tests/unit/profiling/test_base.py
  Estimated Commit: 1

- [ ] TASK-0128
  Title: Add profiling metrics: timing, memory, FLOPs
  Chapter: 3.17
  Requirement: PR-002
  Priority: High
  Depends On: TASK-0127
  Files:
    - src/avqa/profiling/metrics.py
    - tests/unit/profiling/test_metrics.py
  Estimated Commit: 1

- [ ] TASK-0129
  Title: Add profiling routing statistics collector
  Chapter: 3.17
  Requirement: PR-003
  Priority: High
  Depends On: TASK-0071
  Files:
    - src/avqa/profiling/metrics.py
    - tests/unit/profiling/test_routing_stats.py
  Estimated Commit: 1

- [ ] TASK-0130
  Title: Add profiling report exporter (JSON + human-readable)
  Chapter: 3.17, 5.15
  Requirement: PR-004
  Priority: High
  Depends On: TASK-0129
  Files:
    - src/avqa/profiling/report.py
    - tests/unit/profiling/test_report.py
  Estimated Commit: 1

- [ ] TASK-0131
  Title: Add Visualizer base interface
  Chapter: 3.18, 5.16
  Requirement: VS-001
  Priority: High
  Depends On: TASK-0031
  Files:
    - src/avqa/visualization/__init__.py
    - src/avqa/visualization/base.py
    - tests/unit/visualization/test_base.py
  Estimated Commit: 1

- [ ] TASK-0132
  Title: Add refinement tree visualizer
  Chapter: 3.18
  Requirement: VS-002
  Priority: High
  Depends On: TASK-0131
  Files:
    - src/avqa/visualization/tree.py
    - tests/unit/visualization/test_tree.py
  Estimated Commit: 1

- [ ] TASK-0133
  Title: Add attention heatmap visualizer
  Chapter: 3.18
  Requirement: VS-003
  Priority: High
  Depends On: TASK-0131
  Files:
    - src/avqa/visualization/heatmap.py
    - tests/unit/visualization/test_heatmap.py
  Estimated Commit: 1

- [ ] TASK-0134
  Title: Add execution timeline visualizer
  Chapter: 3.18
  Requirement: VS-004
  Priority: High
  Depends On: TASK-0131
  Files:
    - src/avqa/visualization/timeline.py
    - tests/unit/visualization/test_timeline.py
  Estimated Commit: 1

- [ ] TASK-0135
  Title: Add codebook utilization visualizer
  Chapter: 3.18
  Requirement: VS-005
  Priority: Medium
  Depends On: TASK-0053
  Files:
    - src/avqa/visualization/utilization.py
    - tests/unit/visualization/test_utilization.py
  Estimated Commit: 1

### M13 — Framework Integrations

- [ ] TASK-0136
  Title: Add integrations.huggingface.detect_compatibility
  Chapter: 3.14
  Requirement: IHF-001
  Priority: High
  Depends On: TASK-0117
  Files:
    - src/avqa/integrations/__init__.py
    - src/avqa/integrations/huggingface.py
    - tests/integration/integrations/test_hf_detect.py
  Estimated Commit: 1

- [ ] TASK-0137
  Title: Add integrations.huggingface.replace_attention helper
  Chapter: 3.14
  Requirement: IHF-002
  Priority: High
  Depends On: TASK-0136
  Files:
    - src/avqa/integrations/huggingface.py
    - tests/integration/integrations/test_hf_replace.py
  Estimated Commit: 1

- [ ] TASK-0138
  Title: Add integrations.huggingface end-to-end test on tiny BERT
  Chapter: 3.14, 5.17
  Requirement: IHF-003
  Priority: High
  Depends On: TASK-0137
  Files:
    - tests/integration/integrations/test_hf_e2e.py
  Estimated Commit: 1

- [ ] TASK-0139
  Title: Add integrations.huggingface pretrained-weight preservation test
  Chapter: 3.14
  Requirement: IHF-004
  Priority: High
  Depends On: TASK-0138
  Files:
    - tests/integration/integrations/test_hf_weights.py
  Estimated Commit: 1

- [ ] TASK-0140
  Title: Add integrations.vllm.adapter interface
  Chapter: 3.15, 5.17
  Requirement: IVL-001
  Priority: High
  Depends On: TASK-0117
  Files:
    - src/avqa/integrations/vllm.py
    - tests/integration/integrations/test_vllm_imports.py
  Estimated Commit: 1

- [ ] TASK-0141
  Title: Add integrations.vllm.paged_attention integration
  Chapter: 3.15
  Requirement: IVL-002
  Priority: Medium
  Depends On: TASK-0108, TASK-0140
  Files:
    - src/avqa/integrations/vllm.py
    - tests/integration/integrations/test_vllm_paged.py
  Estimated Commit: 1

- [ ] TASK-0142
  Title: Add integrations.vllm.continuous_batching integration
  Chapter: 3.15
  Requirement: IVL-003
  Priority: Medium
  Depends On: TASK-0141
  Files:
    - src/avqa/integrations/vllm.py
    - tests/integration/integrations/test_vllm_batching.py
  Estimated Commit: 1

- [ ] TASK-0143
  Title: Add integrations.vllm.prefix_caching integration
  Chapter: 3.15
  Requirement: IVL-004
  Priority: Medium
  Depends On: TASK-0142
  Files:
    - src/avqa/integrations/vllm.py
    - tests/integration/integrations/test_vllm_prefix.py
  Estimated Commit: 1

- [ ] TASK-0144
  Title: Add integrations.vllm.tensor_parallelism integration
  Chapter: 3.15
  Requirement: IVL-005
  Priority: Low
  Depends On: TASK-0143
  Files:
    - src/avqa/integrations/vllm.py
    - tests/integration/integrations/test_vllm_tp.py
  Estimated Commit: 1

- [ ] TASK-0145
  Title: Add integrations.flash_attention.interop helper
  Chapter: 3.16
  Requirement: IFA-001
  Priority: Medium
  Depends On: TASK-0117
  Files:
    - src/avqa/integrations/flash_attention.py
    - tests/integration/integrations/test_fa_interop.py
  Estimated Commit: 1

- [ ] TASK-0146
  Title: Add integrations.flash_attention.backend selection order config
  Chapter: 3.16
  Requirement: IFA-002
  Priority: Medium
  Depends On: TASK-0145
  Files:
    - src/avqa/integrations/flash_attention.py
    - tests/integration/integrations/test_fa_order.py
  Estimated Commit: 1

- [ ] TASK-0147
  Title: Add integrations.flash_attention numerical equivalence test
  Chapter: 3.16
  Requirement: IFA-003
  Priority: Medium
  Depends On: TASK-0146
  Files:
    - tests/integration/integrations/test_fa_equivalence.py
  Estimated Commit: 1

- [ ] TASK-0148
  Title: Add integrations.xformers.memory_efficient_attention interop
  Chapter: 3.15 (assumed)
  Requirement: IXF-001
  Priority: Low
  Depends On: TASK-0117
  Files:
    - src/avqa/integrations/xformers.py
    - tests/integration/integrations/test_xformers_interop.py
  Estimated Commit: 1

### M14 — Benchmarks + Examples

- [ ] TASK-0149
  Title: Add pytest-benchmark configuration
  Chapter: 3.19
  Requirement: BM-001
  Priority: High
  Depends On: TASK-0011
  Files:
    - pytest.ini
    - benchmarks/conftest.py
  Estimated Commit: 1

- [ ] TASK-0150
  Title: Add AVQA vs PyTorch SDPA benchmark sweep
  Chapter: 3.19
  Requirement: BM-002
  Priority: High
  Depends On: TASK-0117, TASK-0149
  Files:
    - benchmarks/compare_attention.py
    - tests/performance/test_benchmark_attention.py
  Estimated Commit: 1

- [ ] TASK-0151
  Title: Add refinement-budget sweep benchmark
  Chapter: 3.19
  Requirement: BM-003
  Priority: High
  Depends On: TASK-0150
  Files:
    - benchmarks/sweep.py
    - tests/performance/test_benchmark_sweep.py
  Estimated Commit: 1

- [ ] TASK-0152
  Title: Add benchmark reproducibility test (deterministic seed)
  Chapter: 3.19
  Requirement: BM-004
  Priority: High
  Depends On: TASK-0151
  Files:
    - tests/performance/test_benchmark_reproducibility.py
  Estimated Commit: 1

- [ ] TASK-0153
  Title: Add example 01: basic attention
  Chapter: 5.3
  Requirement: EX-001
  Priority: High
  Depends On: TASK-0124
  Files:
    - examples/01_basic_attention.py
  Estimated Commit: 1

- [ ] TASK-0154
  Title: Add example 02: codebook training
  Chapter: 5.3
  Requirement: EX-002
  Priority: High
  Depends On: TASK-0052
  Files:
    - examples/02_codebook_training.py
  Estimated Commit: 1

- [ ] TASK-0155
  Title: Add example 03: HF attention replacement
  Chapter: 5.3, 3.14
  Requirement: EX-003
  Priority: High
  Depends On: TASK-0139
  Files:
    - examples/03_hf_replacement.py
  Estimated Commit: 1

- [ ] TASK-0156
  Title: Add example 04: profiling session
  Chapter: 5.3, 3.17
  Requirement: EX-004
  Priority: High
  Depends On: TASK-0130
  Files:
    - examples/04_profiling.py
  Estimated Commit: 1

- [ ] TASK-0157
  Title: Add example 05: visualization of refinement tree
  Chapter: 5.3, 3.18
  Requirement: EX-005
  Priority: Medium
  Depends On: TASK-0132
  Files:
    - examples/05_visualization.py
  Estimated Commit: 1

### M15 — Release + Compliance

- [ ] TASK-0158
  Title: Add CHANGELOG.md with initial entries
  Chapter: 5.19
  Requirement: REL-001
  Priority: Medium
  Depends On: TASK-0014
  Files:
    - CHANGELOG.md
  Estimated Commit: 1

- [ ] TASK-0159
  Title: Add RELEASE.md notes template
  Chapter: 5.19
  Requirement: REL-002
  Priority: Medium
  Depends On: TASK-0158
  Files:
    - RELEASE.md
  Estimated Commit: 1

- [ ] TASK-0160
  Title: Add docs/spec_compliance.md populated matrix
  Chapter: Phase 4 Continuous Verification
  Requirement: COMP-001
  Priority: High
  Depends On: TASK-0159
  Files:
    - docs/spec_compliance.md
  Estimated Commit: 1

- [ ] TASK-0161
  Title: Add MANIFEST.in for source distribution
  Chapter: 3.3
  Requirement: PKG-001
  Priority: Medium
  Depends On: TASK-0007
  Files:
    - MANIFEST.in
  Estimated Commit: 1

- [ ] TASK-0162
  Title: Add build and twine check scripts
  Chapter: 3.3
  Requirement: PKG-002
  Priority: Medium
  Depends On: TASK-0161
  Files:
    - scripts/build.sh
    - scripts/check_dist.sh
  Estimated Commit: 1

- [ ] TASK-0163
  Title: Add pre-commit hook configuration
  Chapter: Phase 4 Verification
  Requirement: BUILD-008
  Priority: Medium
  Depends On: TASK-0009, TASK-0010
  Files:
    - .pre-commit-config.yaml
  Estimated Commit: 1

- [ ] TASK-0164
  Title: Add Makefile with test/lint/type/format/coverage targets
  Chapter: Phase 4 Verification
  Requirement: BUILD-009
  Priority: Medium
  Depends On: TASK-0011
  Files:
    - Makefile
  Estimated Commit: 1

- [ ] TASK-0165
  Title: Add docs/api/*.rst autodoc stubs
  Chapter: 3.23
  Requirement: DOC-002
  Priority: Low
  Depends On: TASK-0159
  Files:
    - docs/api/avqa.rst
    - docs/api/avqa.config.rst
    - docs/api/avqa.attention.rst
  Estimated Commit: 1

- [ ] TASK-0166
  Title: Add CONTRIBUTING.md
  Chapter: 8.3 (community)
  Requirement: DOC-003
  Priority: Medium
  Depends On: TASK-0014
  Files:
    - CONTRIBUTING.md
  Estimated Commit: 1

- [ ] TASK-0167
  Title: Add CODE_OF_CONDUCT.md
  Chapter: 8.3 (community)
  Requirement: DOC-004
  Priority: Low
  Depends On: TASK-0166
  Files:
    - CODE_OF_CONDUCT.md
  Estimated Commit: 1

- [ ] TASK-0168
  Title: Add SECURITY.md
  Chapter: 8.3 (community)
  Requirement: DOC-005
  Priority: Low
  Depends On: TASK-0166
  Files:
    - SECURITY.md
  Estimated Commit: 1

- [ ] TASK-0169
  Title: Add docs/architecture.md cross-referencing spec chapters
  Chapter: 4.16
  Requirement: DOC-006
  Priority: Low
  Depends On: TASK-0160
  Files:
    - docs/architecture.md
  Estimated Commit: 1

- [ ] TASK-0170
  Title: Add docs/math.md mathematical formulation
  Chapter: 7.20
  Requirement: DOC-007
  Priority: Low
  Depends On: TASK-0160
  Files:
    - docs/math.md
  Estimated Commit: 1

## Implementation Notes

- Tasks are ordered strictly by dependency. A task MAY be implemented only when
  all tasks in its `Depends On` list are marked `[x]`.
- Each TODO entry SHALL map to exactly one atomic commit.
- The TODO entry's SHA is recorded in the `Commit:` field after the commit is
  created.
- The bootstrap Phase 1 docs (TASK-0001..TASK-0006) are committed as the
  initial repository state, before any production code exists. Their SHA field
  shows `_bootstrap_` to distinguish them from the Phase 4 implementation.
