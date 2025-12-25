# Slipstream - Semantic Quantization for Multi-Agent Coordination

**80% fewer tokens. Semantic meaning preserved. Built for the AAIF ecosystem.**

```
JSON (~45 tokens):
{"from": "alice", "to": "bob", "type": "request", "action": "review", "target": "auth_module"}

Slipstream (5 tokens):
SLIP v1 alice bob RequestReview auth_module
```

When running agent swarms at scale, **communication overhead is the bottleneck**. Slipstream solves this through *semantic quantization* - transmitting pointers to concepts rather than the concepts themselves.

## The Problem

Multi-agent systems waste **40-60% of compute on coordination**. BPE tokenizers fragment compressed formats (`|g=42` becomes 4 tokens), negating syntactic optimization.

## The Solution

Slipstream uses a **Universal Concept Reference (UCR)** - a shared semantic manifold where common agent intents have single-token names:

```python
from slipcore import slip, decode, think_quantize_transmit

# Direct message creation
wire = slip("alice", "bob", "RequestReview")
# -> "SLIP v1 alice bob RequestReview" (5 tokens)

# Think-Quantize-Transmit pattern
wire = think_quantize_transmit(
    "Please check the authentication code for security issues",
    src="dev", dst="reviewer"
)
# -> "SLIP v1 dev reviewer RequestReview" (5 tokens)

# Decode
msg = decode(wire)
print(msg.anchor.canonical)  # "Request review of work"
```

## Installation

```bash
pip install slipcore

# With embedding-based quantization (optional)
pip install slipcore[embeddings]
```

## Key Features

- **Token-aligned wire format** - No special characters that fragment in BPE
- **Semantic quantization** - Meaning over compression
- **UCR manifold** - 4-dimensional semantic space (ACTION, POLARITY, DOMAIN, URGENCY)
- **Extension layer** - Add domain-specific anchors (0x8000-0xFFFF range)
- **Finetuning support** - Train models to speak Slipstream natively

## Wire Format

```
SLIP v1 <src> <dst> <anchor> [payload...]
```

| Field | Description |
|-------|-------------|
| `SLIP v1` | Protocol marker |
| `<src>` | Source agent |
| `<dst>` | Destination agent |
| `<anchor>` | UCR semantic anchor |
| `[payload]` | Optional content |

## Core Anchors

| Category | Anchors |
|----------|---------|
| **Requests** | `RequestTask`, `RequestReview`, `RequestHelp`, `RequestPlan` |
| **Info** | `InformComplete`, `InformProgress`, `InformBlocked`, `InformStatus` |
| **Proposals** | `ProposePlan`, `ProposeChange`, `ProposeAlternative` |
| **Evaluation** | `EvalApprove`, `EvalReject`, `EvalNeedsWork` |
| **Meta** | `Accept`, `Reject`, `MetaAck`, `MetaHandoff`, `Fallback` |

## Finetuning

Train models to speak Slipstream natively:

```bash
# Generate training dataset (template-based, free)
python -m slipcore.finetune -n 1000 -f sharegpt -o train.jsonl

# Generate high-quality dataset (LLM-enhanced)
pip install slipcore[llm]
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY, TOGETHER_API_KEY
python -m slipcore.finetune_llm -n 1000 --provider anthropic -o train.jsonl
```

Recommended model: **GLM-4-9B-0414** (MIT licensed, optimized for agentic tasks)

See [.claude/skills/slipstream-finetune.md](.claude/skills/slipstream-finetune.md) for full guide.

## AAIF Integration

Slipstream is designed as the **transport layer** for the Linux Foundation Agentic AI ecosystem:

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
│   Slipstream (Transport Layer)      │  <- 80% token reduction
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   Network                           │
└─────────────────────────────────────┘
```

## Token Efficiency

| Format | Tokens | Savings |
|--------|--------|---------|
| JSON verbose | ~45 | baseline |
| JSON minimal | ~30 | 33% |
| **Slipstream** | **~5-8** | **80%+** |

## Project Structure

```
slipcore/
├── src/slipcore/
│   ├── ucr.py           # Universal Concept Reference
│   ├── protocol.py      # Wire format
│   ├── quantizer.py     # Think-Quantize-Transmit
│   ├── extensions.py    # Local anchor learning
│   ├── finetune.py      # Template dataset generation
│   └── finetune_llm.py  # LLM-enhanced dataset generation
├── examples/
│   └── slipstream_demo.py
└── spec/
    └── slip-spec.md
```

## Contributing

1. Core UCR anchors are immutable within a version
2. Add extension anchors for domain-specific needs
3. Track fallback patterns to identify UCR gaps
4. Submit popular extensions for core promotion

## License

Apache 2.0

---

**Stop paying the token tax.** Semantic quantization > syntactic compression.

```bash
pip install slipcore
```
