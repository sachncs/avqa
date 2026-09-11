# Contributing

Thank you for your interest in AVQA. This project is an **independent,
community-driven** implementation of the AVQ-Attention algorithm.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
By participating, you agree to its terms.

## Reporting bugs

Open a GitHub issue with:

- AVQA version (`python -c "import avqa; print(avqa.__version__)"`)
- PyTorch version
- Minimal reproduction snippet
- Full traceback (if applicable)

## Suggesting enhancements

Open a GitHub issue describing:

- The use case (what are you trying to do?)
- The proposed API (how should it look from a user perspective?)
- Links to relevant papers, RFCs, or prior art

## Pull requests

1. **Fork** the repo and create a feature branch.
2. **Pick** an issue (or open one for new ideas).
3. **Implement** following the existing patterns; keep the diff small and
   focused on one issue.
4. **Test** with `make test` (all 402 tests must pass) and `make lint`.
5. **Document** with Google-style docstrings and Google-style type
   annotations.
6. **Commit** with a Conventional Commit message.
7. **Open** the PR with a description referencing the issue.

## Code style

- PEP 8 (enforced via `ruff format`).
- Strict mypy on `src/avqa/` (enforced via `mypy`).
- Google-style docstrings on all public objects.
- Type annotations on every function signature.

## Adding new components

New components plug in via the `create()` factory pattern on the
existing abstract base class. To add a new quantizer, subclass
`VectorQuantizer` and register it inside `VectorQuantizer.create()`
in `src/avqa/quantizer.py`:

```python
from avqa.quantizer import VectorQuantizer

class MyQuantizer(VectorQuantizer):
    name = "my_quantizer"

    def quantize(self, keys, values, codebook):
        ...

# Register in VectorQuantizer.create():
VectorQuantizer.create("my_quantizer")  # -> MyQuantizer()
```

The same pattern applies to `Backend.create`, `Router.create`,
`MergeStrategy.create`, and `Scheduler.create` — add a `name`
attribute to the subclass and dispatch on it inside `create()`. New
components do not require modifying the abstract base class beyond
the dispatch line.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License 2.0 (see `LICENSE`).