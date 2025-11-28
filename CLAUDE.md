# CLAUDE.md - SLIPCore Development Guide

## Project Overview

SLIPCore (Streamlined Intragent Protocol) is a compact, schema-driven protocol for agent-to-agent messaging in multi-agent LLM systems. It provides token-efficient communication between agents like planners, executors, critics, and coordinators.

## Quick Commands

```bash
# Install in development mode
pip install -e .

# Run examples
python examples/simple_roundtrip.py
python examples/coordinator_planner_demo.py

# Generate training dataset (1000 conversations = 4000 examples)
python -m slipcore.generate_dataset --num-conversations 1000

# LLM-enhanced generation (diverse, realistic goals via Ollama)
python -m slipcore.generate_dataset_llm --num-conversations 500 --model qwen2.5:14b

# Remote Ollama with larger model
python -m slipcore.generate_dataset_llm \
  --ollama-url http://192.168.86.35:11434 \
  --model qwen3:32b \
  --num-conversations 1000
```

## Architecture

### Core Protocol (`src/slipcore/protocol.py`)

- **Act enum**: Speech acts (OBSERVE, INFORM, ASK, REQUEST, PROPOSE, COMMIT, ACCEPT, REJECT, EVAL, ERROR, META)
- **FrameType enum**: Message frames (TASK, PLAN, OBSERVATION, EVALUATION, CONTROL)
- **Slot enum**: Typed fields (GOAL_ID, TASK_ID, PRIORITY, SCORE, STATUS, TAG, etc.)
- **SlipMessage dataclass**: The core message type
- **encode_message/decode_message**: nSLIP wire format encoding

### nSLIP Wire Format

Compact string format: `@a<act>|f<frame>|c<conv>|S<src>|d<dst>|T<turn>|<slots...>#`

Example: `@a3|f0|c1|S0|d1|T1|g1|k1|q2|t"refactor_auth"#`

- Header prefixes: `a`=act, `f`=frame, `c`=conv_id, `S`=src, `d`=dst, `T`=turn
- Slot prefixes: `g`=goal_id, `k`=task_id, `p`=parent_task, `r`=result_id, `q`=priority, `s`=score, `u`=status, `e`=error_code, `t`=tag
- Values: Base62 integers or quoted strings

### Dataset Generators

**Template-based** (`src/slipcore/generate_dataset.py`):
- Fast, deterministic
- Uses predefined goal templates

**LLM-enhanced** (`src/slipcore/generate_dataset_llm.py`):
- Uses Ollama to generate diverse, realistic goals
- Supports local or remote Ollama instances
- Better for training robust models

Both generate synthetic conversation flows:
1. Coordinator: human goal → REQUEST/TASK
2. Planner: REQUEST/TASK → PROPOSE/PLAN
3. Executor: PROPOSE/PLAN → INFORM/OBSERVATION
4. Critic: INFORM/OBSERVATION → EVAL/EVALUATION

Output: JSONL with `role`, `input`, `target` fields for LoRA/QLoRA training.

## Key Design Decisions

- **Base62 encoding** for compact integer representation
- **Uppercase S/T** for src/turn headers to avoid conflict with slot prefixes (s=score, t=tag)
- **IDs not blobs**: Large artifacts referenced by ID, stored externally
- **Framework agnostic**: Works with LangGraph, LangChain, or custom orchestrators

## File Structure

```
slipcore/
├── src/slipcore/
│   ├── __init__.py          # Public API exports
│   ├── protocol.py          # Core protocol implementation
│   └── generate_dataset.py  # Training data generator
├── examples/
│   ├── simple_roundtrip.py
│   └── coordinator_planner_demo.py
├── data/finetune/
│   ├── README.md
│   └── nslip_pairs.sample.jsonl
├── spec/
│   └── slip-spec.md         # Protocol specification
└── pyproject.toml
```

## Claude Skills & Commands

This repo includes Claude Code skills for working with nSLIP:

### Skill: nSLIP Protocol
Located at `.claude/skills/nslip-protocol.md` - provides complete protocol reference including:
- Message structure and encoding
- All acts, frames, and slots
- Base62 encoding rules
- Common message patterns

### Slash Commands
- `/parse-nslip <message>` - Parse and explain an nSLIP wire message
- `/create-nslip <description>` - Generate an nSLIP message from natural language

## Extending the Protocol

To add new slots:
1. Add to `Slot` enum in `protocol.py` with unique int value
2. Add single-char prefix to `_SLOT_PREFIX` dict
3. Update spec documentation

To add new acts/frames:
1. Add to respective enum
2. Update spec documentation
