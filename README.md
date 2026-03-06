# Slipstream

Semantic quantization for multi-agent AI coordination.

[![PyPI](https://img.shields.io/pypi/v/slipcore?color=blue)](https://pypi.org/project/slipcore/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/anthonym21/slipstream-tqt)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.18063451)

Slipstream encodes coordination messages as compact Force+Object intents:

```text
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

Typical JSON coordination messages (~40+ tokens) compress to ~6-8 wire tokens.

## Start Here

- [Quick adoption guide](docs/start-here.md)
- [SDK reference](docs/sdk-guide.md)
- [v3 -> v4 migration notes](docs/migration-v4.md)
- [Protocol invariants](spec/spec-00-invariants.md)

## Install

```bash
pip install slipcore
```

## Quick Start

```python
from slipcore import format_slip, parse_slip, quantize, render_human

# Direct formatting
wire = format_slip("alice", "bob", "Request", "Review", ["auth"])
assert wire == "SLIP v3 alice bob Request Review auth"

# Think -> Quantize -> Transmit helper
wire2 = quantize(
    "Please review the authentication code",
    src="dev",
    dst="reviewer",
)

msg = parse_slip(wire)
print(msg.force, msg.obj, msg.payload)
print(render_human(msg))
```

## Strict Fallback Rules (v4.0.0)

Fallback messages are now strict:

- `Fallback` messages must include a ref token.
- Ref token must be alphanumeric and 1-16 chars.
- Use `parse_slip_legacy()` only for explicit migration of old permissive wires.

```python
from slipcore import format_fallback, parse_slip_legacy

wire = format_fallback("qa", "planner", "ref7f3a")
legacy = parse_slip_legacy("SLIP v3 qa planner Fallback Generic")
assert legacy.fallback_ref == "reflegacy"
```

## Force Vocabulary (Closed, 12)

`Observe`, `Inform`, `Ask`, `Request`, `Propose`, `Commit`, `Eval`, `Meta`, `Accept`, `Reject`, `Error`, `Fallback`

## Ecosystem

- Model: [anthonym21/slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
- Dataset: [anthonym21/slipstream-tqt](https://huggingface.co/datasets/anthonym21/slipstream-tqt)
- A2A transport extension: [extensions/a2a-slipstream/v1](extensions/a2a-slipstream/v1)

## Governance

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Project Governance](GOVERNANCE.md)
- [Maintainers](MAINTAINERS.md)

## Citation

```bibtex
@misc{maio2025slipstream,
  title={Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination},
  author={Maio, Anthony},
  year={2025},
  url={https://github.com/anthony-maio/slipcore}
}
```

## License

Apache 2.0.