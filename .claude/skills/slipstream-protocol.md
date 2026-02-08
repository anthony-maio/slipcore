---
name: slipstream-protocol
description: Slipstream Protocol v3 - Factorized Semantic Quantization for Multi-Agent Communication
---

# Slipstream Protocol v3 Reference

## Overview

Slipstream (SLIP) is a semantic quantization protocol for efficient multi-agent coordination. Instead of transmitting verbose natural language, agents communicate via compact factorized intents: **Force** (action verb) + **Object** (domain noun).

**Key benefit**: 80%+ token reduction vs JSON-wrapped messages.

## Wire Format

```
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

- `SLIP v3` - Protocol marker and version
- `<src>` - Source agent ID (1-20 alphanumeric chars)
- `<dst>` - Destination agent ID (1-20 alphanumeric chars)
- `<Force>` - Action verb from closed vocabulary (12 tokens)
- `<Object>` - Domain noun (30+ registered, extensible)
- `[payload...]` - Optional alphanumeric tokens

## Force Tokens (12 closed vocabulary)

| Force | Action Coord | Description |
|-------|-------------|-------------|
| `Observe` | 0 | Passively notice state/change/error |
| `Inform` | 1 | Report information |
| `Ask` | 2 | Request information |
| `Request` | 3 | Ask for action |
| `Propose` | 4 | Suggest something |
| `Commit` | 5 | Commit to something |
| `Eval` | 6 | Evaluate work |
| `Meta` | 7 | Protocol-level operations |
| `Accept` | 3 | Accept proposal/request |
| `Reject` | 3 | Decline proposal/request |
| `Error` | 6 | Report system error |
| `Fallback` | 7 | Unquantizable content |

## Core Object Tokens

| Category | Objects |
|----------|---------|
| **State** | `State`, `Change`, `Error`, `Result` |
| **Progress** | `Status`, `Complete`, `Blocked`, `Progress` |
| **Communication** | `Clarify`, `Permission`, `Resource` |
| **Work** | `Task`, `Plan`, `Review`, `Help` |
| **Control** | `Cancel`, `Priority`, `Alternative`, `Rollback`, `Deadline` |
| **Evaluation** | `Approve`, `NeedsWork` |
| **Protocol** | `Ack`, `Sync`, `Handoff`, `Escalate`, `Abort` |
| **Misc** | `Condition`, `Defer`, `Timeout`, `Validation`, `Generic` |

## Examples

```
# Request a code review
SLIP v3 alice bob Request Review auth

# Report task completion
SLIP v3 worker manager Inform Complete task42

# Propose a plan
SLIP v3 planner team Propose Plan auth

# Approve work
SLIP v3 reviewer author Eval Approve

# Fallback for complex content (pointer-based)
SLIP v3 devops sre Fallback Generic ref7f3a

# Accept a proposal
SLIP v3 pm dev Accept Generic

# Report error
SLIP v3 monitor ops Error Timeout api
```

## Python Usage

```python
from slipcore import (
    format_slip, parse_slip, format_fallback, validate_wire,
    render_human, render_log_line,
    ForceToken, resolve_intent, from_v2_mnemonic,
    create_base_ucr, KeywordQuantizer,
)

# Create message directly
wire = format_slip("alice", "bob", "Request", "Review")
# -> "SLIP v3 alice bob Request Review"

# With payload
wire = format_slip("planner", "exec", "Request", "Task", ["auth"])
# -> "SLIP v3 planner exec Request Task auth"

# Parse
msg = parse_slip(wire)
print(msg.force, msg.obj, msg.payload)

# Human-readable
print(render_human(msg))
# [planner -> exec] Request Task: "Request task execution" (payload: auth)

# Keyword quantizer (stdlib-only, no ML deps)
q = KeywordQuantizer()
wire = q.quantize("Please review the auth code", "dev", "reviewer")
# -> "SLIP v3 dev reviewer Request Review"

# Validate wire format
issues = validate_wire("SLIP v3 a b Request Plan")
# -> [] (empty = valid)
```

## UCR Semantic Manifold

Each anchor is a position in a 4-dimensional semantic space:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| ACTION | 0-7 | observe, inform, ask, request, propose, commit, evaluate, meta |
| POLARITY | 0-7 | negative to positive valence |
| DOMAIN | 0-7 | task, plan, observation, evaluation, control, resource, error, general |
| URGENCY | 0-7 | background to critical |

## Extension Layer

Core anchors: 0x0000-0x7FFF (standard, immutable)
Extension anchors: 0x8000-0xFFFF (installation-specific)

```python
from slipcore import ExtensionManager, create_base_ucr

ucr = create_base_ucr()
manager = ExtensionManager(ucr)
anchor = manager.add_extension(
    force="Request",
    obj="K8sScale",
    canonical="Request Kubernetes scaling",
)
```

## Design Principles

1. **No special characters** - Avoids BPE fragmentation
2. **Space-separated** - Clean tokenization
3. **Factorized intents** - Force + Object instead of flat anchors
4. **Pointer-based fallback** - Raw text never on wire
5. **Zero core dependencies** - stdlib-only core package
