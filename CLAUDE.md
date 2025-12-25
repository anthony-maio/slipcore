# CLAUDE.md - Slipstream Development Guide

## Project Overview

**Slipstream (SLIP)** is a semantic quantization protocol for efficient multi-agent coordination. Unlike syntactic compression (minification), Slipstream achieves token efficiency by transmitting **pointers to concepts** rather than the concepts themselves.

Key innovation: The **Universal Concept Reference (UCR)** - a quantized semantic manifold that serves as a shared vocabulary for agent intents.

## Quick Commands

```bash
# Install in development mode
pip install -e .

# Run demo
python examples/slipstream_demo.py

# Generate finetuning dataset
python -m slipcore.finetune -n 500 -f sharegpt -o train.jsonl

# Test the library
python -c "from slipcore import slip, decode; print(slip('alice', 'bob', 'RequestReview'))"
```

## Architecture

### Core Modules (`src/slipcore/`)

- **`ucr.py`**: Universal Concept Reference - the semantic manifold
  - `UCRAnchor`: Named positions in the manifold
  - `UCR`: Registry of anchors with lookup methods
  - `Dimension`: Semantic axes (ACTION, POLARITY, DOMAIN, URGENCY)

- **`protocol.py`**: Token-aligned wire format
  - `SlipMessage`: The message dataclass
  - `encode()` / `decode()`: Wire format conversion
  - `slip()`: Quick message creation
  - `fallback()`: Unquantizable content handling

- **`quantizer.py`**: Think-Quantize-Transmit engine
  - `quantize()`: Map thought to nearest anchor
  - `think_quantize_transmit()`: Full TQT flow
  - `KeywordQuantizer`: Fast, no-dependency quantizer
  - `EmbeddingQuantizer`: Accurate, requires sentence-transformers

- **`extensions.py`**: Dynamic local anchors
  - `ExtensionManager`: Add/manage extension anchors
  - `FallbackTracker`: Track patterns for UCR evolution

- **`finetune.py`**: Training dataset generation
  - `generate_dataset()`: Create training data
  - Supports ShareGPT, Chat, Alpaca formats

### Wire Format

```
SLIP v1 <src> <dst> <anchor> [payload...]
```

Example: `SLIP v1 alice bob RequestReview auth_module`

Design principles:
- No special characters (avoids BPE fragmentation)
- Space-separated (clean tokenization)
- CamelCase anchors (often single tokens)

### UCR Structure

4-dimensional semantic manifold:
- **ACTION** (0-7): observe, inform, ask, request, propose, commit, evaluate, meta
- **POLARITY** (0-7): negative to positive valence
- **DOMAIN** (0-7): task, plan, observation, evaluation, control, resource, error, general
- **URGENCY** (0-7): background to critical

Address ranges:
- Core (0x0000-0x7FFF): Standard anchors, immutable per version
- Extension (0x8000-0xFFFF): Installation-specific, evolvable

## Key APIs

```python
from slipcore import slip, decode, quantize, think_quantize_transmit

# Create message directly
wire = slip("alice", "bob", "RequestReview")

# Think-Quantize-Transmit
wire = think_quantize_transmit(
    "Please review the auth code",
    src="dev", dst="reviewer"
)

# Decode
msg = decode(wire)
print(msg.anchor.canonical)

# Generate training data
from slipcore import generate_dataset
generate_dataset(Path("train.jsonl"), num_examples=500, format="sharegpt")
```

## File Structure

```
slipcore/
├── src/slipcore/
│   ├── __init__.py        # Public API
│   ├── ucr.py             # Semantic manifold
│   ├── protocol.py        # Wire format
│   ├── quantizer.py       # TQT engine
│   ├── extensions.py      # Local anchors
│   └── finetune.py        # Dataset generation
├── examples/
│   └── slipstream_demo.py # Full demo
├── spec/
│   └── slip-spec.md       # Protocol specification
├── .claude/
│   ├── skills/
│   │   ├── slipstream-protocol.md  # Protocol reference
│   │   └── slipstream-finetune.md  # Finetuning guide
│   └── commands/
│       ├── parse-slip.md
│       └── create-slip.md
└── pyproject.toml
```

## Claude Skills & Commands

### Skills
- **slipstream-protocol**: Complete protocol reference
- **slipstream-finetune**: Guide for finetuning with Unsloth

### Commands
- `/parse-slip <message>` - Parse and explain a SLIP message
- `/create-slip <description>` - Generate a SLIP message from natural language

## AAIF Integration

Slipstream is designed for the Linux Foundation Agentic AI ecosystem:

```
┌─────────────────────────────────────┐
│   Application (Agent Logic)        │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   MCP / A2A (Semantic Layer)        │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   Slipstream (Transport Layer)      │  ← 80% token reduction
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   Network                           │
└─────────────────────────────────────┘
```

## Contributing

1. Core UCR anchors are immutable within a version
2. Add extension anchors for domain-specific needs
3. Track fallback patterns to identify UCR gaps
4. Submit popular extensions for core promotion

## License

Apache 2.0
