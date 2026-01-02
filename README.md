# Slipstream

### Semantic Quantization for Multi-Agent AI Communication

[![PyPI](https://img.shields.io/pypi/v/slipcore?color=blue)](https://pypi.org/project/slipcore/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/anthony-maio/slipstream-tqt)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.18063451)

---

**82% fewer tokens. Semantic meaning preserved. Built for the AAIF ecosystem.**

```
Before (45 tokens):
{"from": "alice", "to": "bob", "type": "request", "action": "review", "target": "auth_module"}

After (5 tokens):
SLIP v1 alice bob RequestReview auth_module
```

Multi-agent AI systems waste **40-60% of compute on coordination overhead**. At scale, that's **$180K-$2.5M/year** just for agents talking to each other.

Slipstream fixes this through *semantic quantization* - transmitting pointers to concepts rather than the concepts themselves.

---

## Quick Start

```bash
pip install slipcore
```

```python
from slipcore import slip, decode, think_quantize_transmit

# Create a message (5 tokens instead of 45)
wire = slip("alice", "bob", "RequestReview", ["auth_module"])
# -> "SLIP v1 alice bob RequestReview auth_module"

# Or let the quantizer map natural language
wire = think_quantize_transmit(
    "Please check the authentication code for security issues",
    src="dev", dst="reviewer"
)
# -> "SLIP v1 dev reviewer RequestReview"

# Decode
msg = decode(wire)
print(msg.anchor.canonical)  # "Request review of work"
```

---

## Why Slipstream?

### The Problem

BPE tokenizers **fragment compressed formats**, negating syntactic optimization:

```
Compressed: REQ/TSK|s=7|d=3|act=review
Expected:   8 tokens
Actual:     22 tokens (every | and = is a token!)
```

### The Solution

Slipstream uses a **Universal Concept Reference (UCR)** - a shared semantic manifold where common agent intents have single-token names that work across all LLM architectures.

| Format | Tokens | Annual Cost (50 agents) |
|--------|--------|------------------------|
| JSON verbose | ~45 | $180,000 |
| JSON minimal | ~30 | $120,000 |
| **Slipstream** | **~5-8** | **$32,000** |

---

## Wire Format

```
SLIP v1 <src> <dst> <anchor> [payload...]
```

- **No special characters** - avoids BPE fragmentation
- **Space-separated** - clean tokenization
- **CamelCase anchors** - single tokens in most tokenizers

### Core Anchors

| Category | Anchors |
|----------|---------|
| **Requests** | `RequestTask`, `RequestReview`, `RequestHelp`, `RequestPlan` |
| **Inform** | `InformComplete`, `InformProgress`, `InformBlocked`, `InformStatus` |
| **Propose** | `ProposePlan`, `ProposeChange`, `ProposeAlternative` |
| **Evaluate** | `EvalApprove`, `EvalReject`, `EvalNeedsWork` |
| **Meta** | `Accept`, `Reject`, `MetaAck`, `MetaHandoff`, `Fallback` |

---

## Finetuned Model

We provide a ready-to-use model trained on the Slipstream protocol:

| Format | Link | Use Case |
|--------|------|----------|
| LoRA Adapter | [slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b) | Merge with base |
| GGUF Q4 | [slipstream-glm-z1-9b-gguf](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-gguf) | Ollama / llama.cpp |
| Dataset | [slipstream-tqt](https://huggingface.co/datasets/anthony-maio/slipstream-tqt) | Train your own |

### Run with Ollama

```bash
ollama run anthony-maio/slipstream
```

### Train Your Own

```bash
# Generate training dataset
python -m slipcore.finetune_llm -n 1000 --provider gemini -o train.jsonl

# See notebooks/slipstream_finetune_colab.ipynb for full guide
```

---

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
│   Slipstream (Transport Layer)      │  ← 82% token reduction
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   Network                           │
└─────────────────────────────────────┘
```

---

## Resources

- **Paper**: [Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination](https://doi.org/10.5281/zenodo.18063451)
- **Model**: [HuggingFace](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/anthony-maio/slipstream-tqt)
- **Spec**: [spec/slip-spec.md](spec/slip-spec.md)

---

## Citation

```bibtex
@misc{maio2025slipstream,
  title={Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination},
  author={Maio, Anthony},
  year={2025},
  url={https://github.com/anthony-maio/slipcore}
}
```

---

## License

Apache 2.0

---

**Stop paying the token tax.**

```bash
pip install slipcore
```
