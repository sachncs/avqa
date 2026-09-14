name: AVQA Product Site

Premium product page for **AVQA — Adaptive Vector Quantized Attention**.

## Stack

- **Vite 5** — build tool
- **React 18 + TypeScript** — UI
- **Tailwind CSS 3** — design system
- **Framer Motion** — motion
- **Lucide** — iconography

## Develop

```bash
npm install
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
├── App.tsx                   # section composition
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
├── favicon.svg
├── logo.svg
└── .nojekyll
```

## Conventions

- `base: "/avqa/"` is set in `vite.config.ts` so assets resolve correctly on
  GitHub Pages under the project subpath.
- `_legacy/` holds the previous Jekyll landing files for archival.
- All copy is hand-written. No markdown is rendered.