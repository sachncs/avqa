# Contributing

Thank you for your interest in AVQA. This project is an **independent,
community-driven** public-alpha implementation of the AVQ-Attention algorithm.

Before opening a pull request, read the [architecture guide](docs/architecture.md),
the [benchmark protocol](BENCHMARKS.md), and the [release boundaries](RELEASE.md).

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
4. **Test** with `make test`, `make lint`, and `make typecheck`.
5. **Document** public and nontrivial Python APIs with Google-style docstrings.
6. **Commit** with a Conventional Commit message.
7. **Open** the PR with a description referencing the issue.

## Code style

- Follow the [AVQA coding-style guide](docs/coding-style.md), based on Google's
  Python, TypeScript, and Markdown guides.
- Run `make lint`, `ruff format --check src/ tests/ examples/ benchmarks/ scripts/`,
  and `make typecheck` before opening a pull request.
- Google-convention Python docstrings, strict mypy checks, and TypeScript
  restrictions for `any`, unsafe TypeScript suppression comments, and type-only
  imports are enforced in CI.
- Add type annotations to public Python functions and methods. Document
  arguments, return values, raised errors, side effects, and tensor shapes when
  they are not obvious from the signature.
- Test behavior and extension interfaces. Preserve source-file cohesion; avoid
  broad reformatting unrelated to the change.

## Adding new components

Strategies use abstract base classes and named factories. Routers, merge
strategies, and schedulers support runtime registration:

```python
import torch

from avqa.routing import Router, RoutingDecision


class MyRouter(Router):
    def select(self, importance: torch.Tensor, budget: int) -> RoutingDecision:
        ...


Router.register("my_router", MyRouter)
router = Router.create("my_router")
```

Do not edit the factory dispatch to add an implementation. See
[`docs/architecture.md`](docs/architecture.md) for extension boundaries.
Backends use `Backend.register`; quantizers and visualizers currently have a
closed built-in set. Do not claim an extension is supported until its public
contract and tests are in place.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License 2.0 (see `LICENSE`).
