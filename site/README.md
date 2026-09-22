name: AVQA Product Site

Product site and documentation experience for **AVQA — Adaptive Vector Quantized Attention**.

The deployed alpha site is available at `/avqa/`; the documentation experience
is available at `/avqa/docs/`. Both are built from the same Vite application so
links and release messaging stay synchronized.

## Stack

- **Vite 8** — build tool
- **React 18 + TypeScript** — UI
- **Tailwind CSS 3** — design system
- **Framer Motion** — motion
- **Lucide** — iconography

## Develop

```bash
npm ci
npm run dev      # http://localhost:5173/avqa/
```

## Build & preview

```bash
npm run build    # outputs to ./dist
npm run preview  # serves ./dist at http://localhost:4173/avqa/
```

## Deploy

The site is automatically deployed to GitHub Pages on every push to `main`
via `.github/workflows/pages.yml`.

Live URL: https://sachncs.github.io/avqa/

## Structure

```
src/
├── App.tsx                   # product landing page
├── DocsApp.tsx               # documentation experience at /avqa/docs/
├── main.tsx                  # bootstrap
├── components/
│   ├── Nav.tsx
│   ├── Hero.tsx
│   ├── TrustStrip.tsx
│   ├── WhySection.tsx
│   ├── SolutionSection.tsx
│   ├── FeaturesSection.tsx
│   ├── ArchitectureSection.tsx
│   ├── CodeSection.tsx
│   ├── CodeBlock.tsx
│   ├── BenchmarksSection.tsx
│   ├── CitationSection.tsx
│   ├── CtaSection.tsx
│   ├── Footer.tsx
│   ├── FadeIn.tsx
│   ├── visuals/
│   │   ├── HeroVisual.tsx
│   │   └── PipelineVisual.tsx
│   └── hooks/
├── hooks/
│   └── useScrollPosition.ts
└── lib/
    └── links.ts
public/
├── avqa-mark.svg
├── avqa-wordmark.svg
├── favicon.svg
└── .nojekyll
```

## Conventions

- `base: "/avqa/"` is set in `vite.config.ts` so assets resolve correctly on
  GitHub Pages under the project subpath.
- `_legacy/` holds the previous Jekyll landing files for archival.
- All copy is hand-written. No markdown is rendered.
