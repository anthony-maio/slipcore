# CLAUDE.md - Slipstream v3 Development Guide

## Project Overview

**Slipstream (SLIP)** is a semantic quantization protocol for efficient multi-agent coordination. Unlike syntactic compression (minification), Slipstream achieves token efficiency by transmitting **factorized intents** (Force + Object) rather than verbose messages.

Key v3 innovation: **Factorized 2-token intents** replace ~46 flat anchors. `SLIP v3 src dst Request Plan` instead of `SLIP v1 src dst RequestPlan`. This reduces the classification problem from 46-way to 12-way + 30-way, making it learnable by small models.

The **Universal Concept Reference (UCR)** is a quantized semantic manifold that serves as a shared vocabulary for agent intents.

## Quick Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run demo
python examples/slipstream_demo.py

# Generate finetuning dataset
python -m slipcore.finetune -n 500 -f sharegpt_thought -o train.jsonl

# Migrate v2 data to v3
python scripts/migrate_v2_data.py data/slipstream-tqt.jsonl data/slipstream-tqt-v3.jsonl

# Test the library
python -c "from slipcore import format_slip, parse_slip; print(format_slip('alice', 'bob', 'Request', 'Review'))"
```

## Architecture

### Core Modules (`src/slipcore/`) - Zero external dependencies

- **`intent.py`**: Heart of v3 - factorized intent model
  - `ForceToken` enum (12 closed vocabulary)
  - `ObjectToken` frozen dataclass (30+ core objects with coords)
  - `Intent` combining Force + Object
  - `V2_TO_V3` migration dict for all 46 v2 mnemonics

- **`wire.py`**: Token-aligned wire format
  - `SlipMessage`: Frozen message dataclass
  - `format_slip()` / `parse_slip()`: Wire format conversion
  - `format_fallback()`: Pointer-based fallback
  - `validate_wire()`: Wire validation

- **`ucr.py`**: Universal Concept Reference - the semantic manifold
  - `UCRAnchor`: Named positions with Force/Object fields
  - `UCR`: Registry with content hashing
  - `AnchorState` enum: DRAFT -> PROPOSED -> APPROVED -> ACTIVE -> DEPRECATED
  - `create_base_ucr()`: 44 core anchors

- **`quantizer.py`**: Think-Quantize-Transmit engine (stdlib-only)
  - `KeywordQuantizer`: Two-stage classifier (Force then Object)
  - `quantize()`: Map thought to nearest intent
  - No ML dependencies - pure keyword matching

- **`events.py`**: Observability system
  - Pluggable sinks (register/emit/clear)
  - `FallbackStore`: Dict-backed store for raw text by ref

- **`render.py`**: Human adapter
  - `render_human()`: Human-readable format
  - `render_log_line()`: Structured log format

- **`extensions.py`**: Dynamic local anchors
  - `ExtensionManager`: Add/manage extensions with Force/Object
  - `FallbackTracker`: Track patterns for UCR evolution

- **`errors.py`**: Exception hierarchy
  - `SlipError`, `WireParseError`, `WireValidationError`, `UCRError`, etc.

- **`finetune.py`**: Template-based training dataset generation

- **`finetune_llm.py`**: LLM-enhanced dataset generation (requires httpx)

### ML Extras (`src/slipcore_ml/`) - Optional, requires `pip install slipcore[ml]`

- **`quantizer.py`**: `SemanticQuantizer` using sentence-transformers
- **`coords.py`**: `CoordsInferer` with prototype embeddings
- **`clustering.py`**: ML-based extension learning

### Wire Format

```
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

Example: `SLIP v3 alice bob Request Review auth`

Design principles:
- No special characters (avoids BPE fragmentation)
- Space-separated (clean tokenization)
- Factorized intents: Force (action verb) + Object (domain noun)
- All tokens alphanumeric
- Agent IDs: 1-20 alphanumeric chars
- Fallback uses pointer ref, never raw text on wire

### Force Tokens (12 closed vocabulary)

Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

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
from slipcore import (
    format_slip, parse_slip, format_fallback, validate_wire, SlipMessage,
    ForceToken, resolve_intent, from_v2_mnemonic,
    UCR, UCRAnchor, create_base_ucr,
    render_human, render_log_line,
    KeywordQuantizer, FallbackStore,
)

# Create message directly
wire = format_slip("alice", "bob", "Request", "Review")
# -> "SLIP v3 alice bob Request Review"

# Parse
msg = parse_slip(wire)
print(msg.force, msg.obj)  # Request Review

# Human-readable
print(render_human(msg))

# Keyword quantizer (stdlib-only)
q = KeywordQuantizer()
wire = q.quantize("Please review the auth code", "dev", "reviewer")
# -> "SLIP v3 dev reviewer Request Review"

# Generate training data
from slipcore.finetune import generate_dataset
from pathlib import Path
generate_dataset(Path("train.jsonl"), num_examples=500, format="sharegpt_thought")
```

## File Structure

```
slipcore/
├── src/slipcore/           # Core (zero dependencies)
│   ├── __init__.py         # Public API, v3.0.0a1
│   ├── intent.py           # ForceToken, ObjectToken, Intent, V2_TO_V3
│   ├── wire.py             # SlipMessage, format_slip, parse_slip
│   ├── ucr.py              # UCR registry, create_base_ucr
│   ├── quantizer.py        # KeywordQuantizer (stdlib-only)
│   ├── events.py           # Observability, FallbackStore
│   ├── render.py           # Human rendering
│   ├── extensions.py       # Extension manager
│   ├── errors.py           # Exception hierarchy
│   ├── finetune.py         # Template dataset generation
│   ├── finetune_llm.py     # LLM-enhanced dataset generation
│   └── py.typed            # PEP 561 marker
├── src/slipcore_ml/        # ML extras (optional)
│   ├── __init__.py
│   ├── quantizer.py        # SemanticQuantizer
│   ├── coords.py           # CoordsInferer
│   └── clustering.py       # ML extension learning
├── src/slipcore_tools/     # CLI tools (optional)
│   └── __init__.py
├── tests/
│   ├── test_wire.py
│   ├── test_intent.py
│   ├── test_ucr.py
│   ├── test_events.py
│   ├── test_render.py
│   ├── test_quantizer.py
│   ├── test_conformance.py
│   └── test_ml_quantizer.py
├── spec/
│   ├── spec-00-invariants.md
│   └── conformance/        # valid.jsonl, invalid.jsonl, roundtrip.jsonl
├── examples/
│   └── slipstream_demo.py
├── scripts/
│   └── migrate_v2_data.py
├── data/
│   ├── slipstream-tqt.jsonl      # v2 training data (2,283 examples)
│   └── slipstream-tqt-v3.jsonl   # Migrated v3 data
├── hf-space/
│   └── app.py              # Gradio demo
├── .claude/
│   ├── skills/
│   │   ├── slipstream-protocol.md
│   │   └── slipstream-finetune.md
│   └── commands/
│       ├── parse-slip.md
│       └── create-slip.md
└── pyproject.toml
```

## Claude Skills & Commands

### Skills
- **slipstream-protocol**: Complete v3 protocol reference
- **slipstream-finetune**: Guide for finetuning with Unsloth

### Commands
- `/parse-slip <message>` - Parse and explain a SLIP v3 message
- `/create-slip <description>` - Generate a SLIP v3 message from natural language

## Finetuning

### Dataset Generation

```bash
# Template-based (free, fast)
python -m slipcore.finetune -n 1000 -f sharegpt_thought -o train.jsonl

# LLM-enhanced (higher quality)
python -m slipcore.finetune_llm -n 1000 --provider gemini -o train_llm.jsonl

# Migrate v2 data
python scripts/migrate_v2_data.py data/slipstream-tqt.jsonl data/slipstream-tqt-v3.jsonl
```

### Dataset Formats

- `sharegpt_thought` (recommended): Includes THOUGHT reasoning chain
- `sharegpt_semantics`: Full THOUGHT + QUANTIZE + SLIP
- `sharegpt`: Direct instruction -> SLIP (no reasoning)

## Contributing

1. Core UCR anchors are immutable within a version
2. Force tokens are a closed vocabulary (12 members)
3. Object tokens can be extended via ExtensionManager
4. Zero external dependencies in core package
5. All wire tokens must be alphanumeric (BPE-safe)

## License

Apache 2.0
