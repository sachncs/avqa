# AVQA brand system

AVQA's visual idea is the **allocation field**: a regular computational grid in
which a small routed subset receives finer work. Use it to explain selective
resolution—not to imply that a diagram is measured model attention.

## Identity

The compact mark is an orthogonal path through a 4×4 field of cells. It
represents routing and selective refinement without an eye, brain, sparkle, or
network motif. At favicon size the grid is reduced to a single path and a few
selected cells. The wordmark is typographic; don't redraw the mark inside a
headline or use the mark as a decorative bullet.

## Assets

- `site/public/avqa-mark.svg` — primary dark-surface symbol and favicon-scale
  source.
- `site/public/avqa-wordmark.svg` — primary dark-surface lockup.
- `site/public/avqa-mark-light.svg` and `avqa-wordmark-light.svg` — color
  variants for light backgrounds.
- `site/public/avqa-mark-mono-{dark,light}.svg` and
  `avqa-wordmark-mono-{dark,light}.svg` — single-ink variants.
- `site/public/favicon.svg` — browser icon with the compact grid path.
- `site/public/og-image.svg` — editable source for the social preview.
- `site/public/og-image.png` — 1200×630 preview used by social metadata. Keep
  the conceptual illustration, title, alpha status, and CUDA boundary together.

Use the wordmark in navigation and project attribution; use the symbol alone
for compact app/browser contexts. Preserve clear space equal to one cell width.
Do not distort, rotate, add effects to, or place the dark lockup on a light
background. No external image dependencies are embedded in the social preview.
The deployed site uses a considered dark-only presentation. Light and
monochrome assets are for documents, talks, or other backgrounds—not an
unmaintained theme toggle on the site.

## Color tokens

| Role | Value | Use |
| --- | --- | --- |
| Carbon canvas | `#0D1510` | Page background |
| Raised forest | `#121C15` | Figure/code surfaces |
| Deep surface | `#080E0A` | Recessed code only |
| Mineral white | `#F5F3E9` | Primary text |
| Sage gray | `#B3BBA9` | Secondary content |
| Lichen | `#B9D692` | Focus/positive signal |
| Persimmon | `#E9784F` | Selected route / primary emphasis |
| Quiet line | `rgba(229,231,216,.13)` | Dividers, not card outlines |

Use persimmon only for selection and actionable emphasis. Lichen is a second
signal, not an ambient glow. Pair color with labels, shape, or line weight; no
important state may depend on color alone. `ink-500` is `#83907E`, selected to
retain WCAG AA contrast on the raised forest surface. Recheck contrast when
introducing a new surface or text color.

## Type and composition

- **Manrope** carries headlines and body copy. Headline tracking is tight, but
  body measure remains below roughly 68 characters per line.
- **IBM Plex Mono** is reserved for short technical labels, versions, values,
  and code. Avoid all-caps paragraphs and tiny mono body copy.
- **Spacing** uses a 4px base rhythm; page sections use 64–112px vertical
  intervals, and figure/row padding uses 16–24px.
- **Grid** is a 12-column canvas, capped at 1320px; long-form text stays narrow.
- **Geometry** is square by default. Use small radii only for controls and
  compact code interfaces; avoid pill navigation and floating card stacks.
- **Elevation** comes from contrast, rules, and negative space. No glow shadows,
  blurred orbs, or decorative noise.

## Figures, data, and motion

Diagrams share thin rules, right-angle flow, left-aligned mono stage labels,
and the same persimmon selected-state color. The site distinguishes conceptual
figures from data. Every quantitative figure names its metric, unit or
dimensionless status, setup, source, and limitations; use tabular numbers.

The site uses no entrance or scroll-reveal animation. Only user-triggered
control state transitions may animate; all transitions respect
`prefers-reduced-motion`. The first static frame and narrow-screen alternative
must communicate the full idea.
