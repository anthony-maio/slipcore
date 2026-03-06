# Contributing to Slipstream

Thanks for contributing.

## Development Setup

```bash
git clone https://github.com/anthony-maio/slipcore.git
cd slipcore
pip install -e ".[dev]"
```

## Required Checks

Run before opening a PR:

```bash
PYTHONPATH=src ruff check src/
PYTHONPATH=src mypy src/slipcore/
PYTHONPATH=src pytest tests -v
```

## Protocol Changes

If your PR changes protocol behavior, you must update in the same PR:

1. Spec (`spec/spec-00-invariants.md` and relevant conformance vectors)
2. Code (`src/slipcore/...`)
3. Tests (`tests/...`)
4. Paper source (`private/zenodo/slipstream-paper-v3.tex`)
5. Claim map (`docs/claim-map.md`)

## Extension Proposals (RFC-lite)

For new object tokens or extension lifecycle changes:

1. Open an issue with label `rfc`.
2. Include motivation, sample wires, migration impact, and security considerations.
3. Wait for maintainer approval before implementation.

## PR Expectations

- Small, focused changes.
- Backward-compatibility notes for breaking changes.
- Changelog entry for user-visible behavior changes.
- Updated docs and examples.

## License

By contributing, you agree that your contributions are licensed under Apache-2.0.
