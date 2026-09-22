# AVQA website design audit and redesign

Audit date: 2026-09-23. Scope: deployed homepage, its current source, and
repository-owned benchmark evidence. The current deployed experience was
inspected in a browser; source findings were checked against the repository.

## Findings

### Critical

- The EXP-0004 homepage callout says 60.7% after 1,024 updates, but the checked
  in raw experiment has static loss 15.8983 and BCAR loss 4.8974 at that step,
  a 69.2% reduction. The display also omits the measured residual and hides
  that the experiment is a tiny synthetic setup. Replace it with the values,
  metric, step, configuration, and limitation from the tracked artifact.
- The work-model rows combine asymptotic shorthand with qualitative estimates
  under a “comparison” heading. They are not measured benchmark results and
  could be read as evidence of lower AVQA cost. Separate cost-model context
  from empirical results and label it accordingly.
- The hero's tree illustration, pulsing paths, and “selected leaves” labels do
  not show the actual conceptual sequence (key grouping, coarse parent score,
  routing, child refinement). It is attractive decoration but poor technical
  communication. Replace it with an explicitly illustrative, static-first
  allocation-field diagram.

### Important

- The identity uses the familiar dark navy / violet / cyan gradient, blur,
  glow, pill, and glass-card vocabulary common to generic AI/SaaS templates.
  The mark's tree and overlaid A are too detailed at favicon scale and do not
  read as the allocation idea.
- The page repeats the same three-column card pattern across problem/features
  and layers large cards around the pipeline, code, and CTA. This makes the
  narrative feel vertically stacked and increases cognitive load.
- The “Built on” strip foregrounds internal engineering metrics (mypy and
  coverage) instead of helping a researcher decide whether the algorithm is
  relevant. Its claims are not linked to evidence.
- The architecture text overstates the backend/router contract and claims the
  core never imports profiling/visualization; present extension points and
  dependencies precisely, with a link to the source architecture.
- “Six pieces. One attention.” and feature copy call implementation details
  contributions of the paper without distinguishing paper method from this
  repository's additional implementation features.
- Type scale, all-caps labels, pill radii, shadows, and gradients lack a clear
  hierarchy. “Inter Display” is configured as a family but not loaded.
- The sticky capsule header and floating abstract badges compete with the
  content on narrow viewports. Mobile checks must include the 320–390px range,
  not just a browser's default responsive breakpoint.

### Polish

- The SVGs use duplicated hard-coded gradient definitions and dark-only
  colors, making them difficult to reuse and maintain as a coherent figure
  system.
- Footer, CTA, code blocks, and section headers use several unrelated border,
  radius, and button treatments.
- Long-form section copy would benefit from a narrower measure and a more
  deliberate editorial rhythm.

## Direction and system

- Concept: **allocation field** — a quiet computational grid where a small
  subset of regions is marked for higher-resolution work. It describes the
  method without pretending to visualize measured model attention.
- Palette: carbon-green canvas, warm mineral text, muted sage secondary text,
  restrained persimmon for route selection, and pale lichen for the secondary
  signal. Remove blue/purple and cyan glow as primary brand cues.
- Mark: compact orthogonal cell/path glyph that remains legible at 16px; the
  wordmark is typographic, not a miniature illustration.
- Type: loaded geometric sans for display/body, loaded mono for labels and
  code. Use sentence-case headlines; reserve uppercase mono for short metadata.
- Figures: shared 1px rules, square-corner geometry, labeled stages, and a
  legend. Color is reinforced by line weight, labels, and selection outlines.
- Motion: no continuously pulsing technical state. Use short state transitions
  only for navigation and explicit user-controlled examples; honor reduced
  motion. The first frame must explain the figure without animation.
- Grid: 12-column wide canvas, narrow prose measure, variable section layouts;
  use one restrained border radius for interactive controls and code only.

## New information architecture

1. Hero: project identity, literal one-sentence explanation, CPU alpha status,
   quick start, source, and a real conceptual routing visual.
2. Method: problem framing followed by a numbered coarse-to-fine explanation.
3. Implementation: separate paper method, repository APIs, and optional
   engineering capabilities.
4. Architecture: actual public extension points and ownership boundaries.
5. Evidence: verified CPU experiment values and protocol, with limits adjacent.
6. Paper and resources: independent implementation provenance, paper, docs,
   compatibility, and source.

The primary journey is: understand the approximation in under 15 seconds,
inspect its stages and limitation, judge the evidence, then reach a tested CPU
quick start. GPU support, CUDA/Triton equivalence, and performance remain
explicitly untested in a CUDA environment.

## QA record

- Production build screenshots were inspected at 390px and 1440px; tablet and
  320px layouts were checked in Chromium device emulation. At narrow widths,
  long code lines remain locally horizontally scrollable and do not expand the
  document canvas.
- Keyboard-visible focus uses a 2px high-contrast outline. The mobile navigation
  has an accessible name, expanded state, Escape-to-close, and focus restoration.
- Reduced-motion preference is respected by the page's remaining CSS
  transitions; content is static-first and is never hidden pending animation.
- `npm run typecheck`, `npm run lint`, `npm run format:check`,
  `npm run check:routes`, `npm run check:contrast`, and `npm run build` pass.
  The contrast script checks core brand token pairs; it is not a full automated
  WCAG audit. A manual assistive-technology audit remains recommended.
- No performance or superiority claim is inferred from the synthetic CPU result.
