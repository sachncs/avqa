# Public-alpha release checklist

## Evidence

- [ ] Version in `src/avqa/version.py`, package metadata, README, site, and release notes agrees.
- [ ] Unit, reference, integration, lint, and type checks pass.
- [ ] Coverage meets the configured 90% gate.
- [ ] Benchmark records include revision, environment, configuration, seed, and raw output.
- [ ] Unsupported integrations and performance limitations are documented.

## Artifacts

- [ ] `python -m build` succeeds.
- [ ] `twine check dist/*` succeeds.
- [ ] Source and wheel installation smoke tests pass in a clean environment.
- [ ] Changelog and release notes describe only shipped behavior.
- [ ] GitHub release assets and tag are generated from the intended commit.

## Product and OSS

- [ ] Landing page and documentation site build successfully.
- [ ] Docs links, examples, support, security, and contribution paths resolve.
- [ ] Public-alpha banner and compatibility matrix are current.
- [ ] No page claims production readiness, PyPI availability, vendor-kernel support,
      or universal benchmark superiority without evidence.
