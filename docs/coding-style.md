# Coding style

AVQA follows Google's published style guides where they fit Python, TypeScript,
and this repository's public contracts:

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)
- [Google Markdown Style Guide](https://google.github.io/styleguide/docguide/style.html)

The enforced Python conventions include 80-column formatting, Ruff formatting,
Google-convention docstrings, and strict mypy checks. Public and nontrivial
functions document their arguments, return values, errors, side effects, and
tensor shapes. Dunder methods and constructors are not required to repeat
documentation already present on their owning class; constructor parameters
belong in the class docstring.

The frontend uses Prettier at an 80-column print width and ESLint with rules
against explicit `any`, unsafe TypeScript suppression comments, and value
imports used only as types. Public UI behavior should be semantic, keyboard
reachable, labeled, and tested at mobile and desktop layouts.

Before submitting changes, run:

```bash
make lint
ruff format --check src/ tests/ examples/ benchmarks/ scripts/
make typecheck
make test
cd site
npm ci
npm run typecheck
npm run lint
npm run format:check
npm run check:routes
npm run build
```

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the contribution workflow and
[`architecture.md`](architecture.md) before adding components or extension
points.
