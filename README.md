# Slipstream

### Semantic Quantization for Multi-Agent AI Communication

[![PyPI](https://img.shields.io/pypi/v/slipcore?color=blue)](https://pypi.org/project/slipcore/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/anthony-maio/slipstream-tqt)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.18063451)

---

**82% fewer tokens. Factorized Force-Object intents. Built for the AAIF ecosystem.**

```
Before (45 tokens):
{"from": "alice", "to": "bob", "type": "request", "action": "review", "target": "auth_module"}

After (6 tokens):
SLIP v3 alice bob Request Review auth
```

Multi-agent AI systems waste **40-60% of compute on coordination overhead**. At scale, that's **$180K-$2.5M/year** just for agents talking to each other.

Slipstream fixes this through *semantic quantization* - transmitting factorized intents (Force + Object) rather than verbose messages.

**v3 Innovation:** Factorized 2-token intents replace ~46 flat anchors. `SLIP v3 src dst Request Plan` instead of `SLIP v1 src dst RequestPlan`. This reduces the classification problem from 46-way to 12-way + 30-way, making it learnable by small models.

---

## Quick Start

```bash
pip install slipcore
```

```python
from slipcore import format_slip, parse_slip, render_human, KeywordQuantizer

# Create a message (6 tokens instead of 45)
wire = format_slip("alice", "bob", "Request", "Review", ["auth"])
# -> "SLIP v3 alice bob Request Review auth"

# Or let the quantizer map natural language
q = KeywordQuantizer()
wire = q.quantize(
    "Please check the authentication code for security issues",
    src="dev", dst="reviewer"
)
# -> "SLIP v3 dev reviewer Request Review"

# Parse
msg = parse_slip(wire)
print(msg.force, msg.obj, msg.payload)
# Request Review ['auth']

# Human-readable
print(render_human(msg))
# [alice -> bob] Request Review: "Request review of work" (payload: auth)
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

Slipstream uses a **Universal Concept Reference (UCR)** - a shared semantic manifold where common agent intents have factorized names (Force + Object) that tokenize efficiently across all LLM architectures.

| Format | Tokens | Annual Cost (50 agents) |
|--------|--------|------------------------|
| JSON verbose | ~45 | $180,000 |
| JSON minimal | ~30 | $120,000 |
| **Slipstream v3** | **~6-8** | **$32,000** |

---

## Wire Format

```
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

- **Factorized intents** - Force (action verb) + Object (domain noun)
- **No special characters** - avoids BPE fragmentation
- **Space-separated** - clean tokenization
- **12 Force tokens** - closed vocabulary, easily learned
- **Zero core dependencies** - stdlib-only core package

### Force Tokens (12 closed vocabulary)

| Force | Description |
|-------|-------------|
| `Observe` | Passively notice state/change/error |
| `Inform` | Report information (status, completion, blockage) |
| `Ask` | Request information (clarification, status, permission) |
| `Request` | Ask for action (task, review, help, plan) |
| `Propose` | Suggest something (plan, change, alternative) |
| `Commit` | Commit to something (task, deadline, resource) |
| `Eval` | Evaluate work (approve, needs work) |
| `Meta` | Protocol-level (acknowledge, sync, handoff) |
| `Accept` | Accept a proposal/request |
| `Reject` | Decline a proposal/request |
| `Error` | Report system error |
| `Fallback` | Content too specific for standard tokens |

### Core Object Tokens

Task, Plan, Review, Help, Status, Complete, Blocked, Progress, State, Change, Error, Result, Clarify, Permission, Resource, Cancel, Priority, Alternative, Rollback, Deadline, Approve, NeedsWork, Ack, Sync, Handoff, Escalate, Abort, Condition, Defer, Timeout, Validation, Generic

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
# Generate v3 training dataset
python -m slipcore.finetune -n 1000 -f sharegpt_thought -o train.jsonl

# Or use LLM-enhanced generation
python -m slipcore.finetune_llm -n 1000 --provider gemini -o train.jsonl

# Migrate existing v2 data to v3
python scripts/migrate_v2_data.py data/slipstream-tqt.jsonl data/slipstream-tqt-v3.jsonl
```

---

## AAIF Integration

Slipstream is designed as the **transport layer** for the Linux Foundation Agentic AI ecosystem:

```
+-------------------------------------+
|   Application (Agent Logic)         |
+----------------+--------------------+
                 |
+----------------v--------------------+
|   MCP / A2A (Semantic Layer)        |
+----------------+--------------------+
                 |
+----------------v--------------------+
|   Slipstream (Transport Layer)      |  <- 82% token reduction
+----------------+--------------------+
                 |
+----------------v--------------------+
|   Network                           |
+-------------------------------------+
```

---

## Resources

- **Paper**: [Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination](https://doi.org/10.5281/zenodo.18063451)
- **Model**: [HuggingFace](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/anthony-maio/slipstream-tqt)
- **Spec**: [spec/spec-00-invariants.md](spec/spec-00-invariants.md)

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
