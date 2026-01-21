# Contributing to Slipstream

Thanks for your interest in contributing to Slipstream! This guide explains how to set up the
project locally, run checks, and submit changes.

## Development setup

1. Clone the repository.
2. Create a virtual environment with Python 3.10+.
3. Install the project in editable mode with dev tooling:

```bash
pip install -e ".[dev]"
```

### Optional dependency groups

Slipstream ships optional features that pull in heavier dependencies. Install them as needed:

```bash
# Embedding-based quantization
pip install -e ".[embeddings]"

# UCR builder (corpus-based construction)
pip install -e ".[builder]"

# LLM-backed dataset generation
pip install -e ".[llm]"

# Everything
pip install -e ".[all]"
```

## Running tests and checks

```bash
pytest
```

We use Ruff for linting:

```bash
ruff check .
```

## Project structure

* `src/slipcore/` contains the core protocol, UCR definitions, and quantizer logic.
* `spec/` holds the Slipstream wire format specification.
* `examples/` includes usage demos and quick experiments.

## Pull requests

* Keep changes focused and include context in the PR description.
* Add tests where behavior changes.
* Update documentation if you add or change public APIs.

## Reporting issues

If you find a bug or have a feature request, please open an issue with clear reproduction steps
or a concrete proposal.
