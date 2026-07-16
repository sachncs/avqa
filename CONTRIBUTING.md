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
2. **Pick** an entry from `TODO.md` (or open an issue first for new ideas).
3. **Implement** following the existing patterns; keep the diff small and
   focused on one TODO entry.
4. **Test** with `make test` (all 402 tests must pass) and `make lint`.
5. **Document** with Google-style docstrings and Google-style type
   annotations.
6. **Commit** with a message of the form
   `TASK-NNNN: short summary` and update `TODO.md` with the commit SHA.
7. **Open** the PR with a description referencing the TODO entry.

## Code style

- PEP 8 (enforced via `ruff format`).
- Strict mypy on `src/avqa/` (enforced via `mypy`).
- Google-style docstrings on all public objects.
- Type annotations on every function signature.

## Adding new components

Each new component must implement the registry pattern documented in
`src/avqa/registry.py`. To add a new quantizer:

```python
from avqa.quantizer import VectorQuantizer
from avqa.registry import QUANTIZER_REGISTRY

@QUANTIZER_REGISTRY.register("my_quantizer")
class MyQuantizer(VectorQuantizer):
    def precompute(self, keys, values, codebook):
        ...
```

This avoids touching core library code and keeps the public API stable.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License 2.0 (see `LICENSE`).