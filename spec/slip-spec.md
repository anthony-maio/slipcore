# Slipstream Protocol Specification (v2.0)

## 1. Overview

**Slipstream (SLIP)** is a semantic quantization protocol for efficient multi-agent coordination. Unlike syntactic compression approaches (minification, base encoding), Slipstream achieves token efficiency by transmitting **pointers to concepts** rather than the concepts themselves.

### Key Innovation: Semantic Quantization

Traditional approaches compress the *syntax* of messages. Slipstream quantizes the *semantics*:

```
Traditional:  "Please review this code for security issues"  (~12 tokens)
              ↓ (minify)
              "REQ/REV|sec"  (~6 tokens, but BPE fragments to ~10)

Slipstream:   "Please review this code for security issues"  (~12 tokens)
              ↓ (semantic quantization)
              "SLIP v1 alice bob RequestReview"  (5 tokens)
```

### Goals

- **Token efficiency**: 70-80% reduction vs JSON-wrapped natural language
- **Model agnostic**: Works across GPT-4, Claude, Llama, etc.
- **BPE friendly**: No special characters that fragment in tokenizers
- **Evolvable**: Core standard + extension layer for local concepts

---

## 2. Universal Concept Reference (UCR)

The UCR is a **quantized semantic manifold** - a shared coordinate system for agent thoughts.

### 2.1 Semantic Dimensions

The manifold has 4 dimensions representing fundamental aspects of agent communication:

| Dimension | Description | Levels (0-7) |
|-----------|-------------|--------------|
| ACTION | What type of action | observe, inform, ask, request, propose, commit, evaluate, meta |
| POLARITY | Positive/negative valence | strongly negative → neutral → strongly positive |
| DOMAIN | Context/topic area | task, plan, observation, evaluation, control, resource, error, general |
| URGENCY | Priority level | background → normal → critical |

### 2.2 Anchors

Anchors are **named positions** in the manifold. Common agent intents get human-readable mnemonics:

```python
UCRAnchor(
    index=0x0032,
    mnemonic="RequestReview",      # Wire-format token
    canonical="Request review of work",  # Human description
    coords=(3, 4, 3, 3),           # (action, polarity, domain, urgency)
)
```

### 2.3 Address Ranges

- **Core UCR (0x0000-0x7FFF)**: Standard anchors, immutable per version
- **Extension UCR (0x8000-0xFFFF)**: Installation-specific, evolvable

---

## 3. Wire Format

### 3.1 Structure

```
SLIP <version> <src> <dst> <anchor> [payload...]
```

| Field | Description | Tokens |
|-------|-------------|--------|
| `SLIP` | Protocol marker | 1 |
| `<version>` | Protocol version (e.g., `v1`) | 1 |
| `<src>` | Source agent identifier | 1 |
| `<dst>` | Destination agent identifier | 1 |
| `<anchor>` | UCR mnemonic (e.g., `RequestReview`) | 1 |
| `[payload...]` | Optional unquantizable content | variable |

### 3.2 Design Principles

1. **No special characters**: Avoid `|`, `@`, `#`, `=` which fragment in BPE
2. **Space separation only**: Spaces are handled well by all tokenizers
3. **CamelCase mnemonics**: Compound words often tokenize as single units
4. **Natural words**: Use vocabulary that exists in standard tokenizers

### 3.3 Examples

```
# Simple coordination
SLIP v1 alice bob RequestReview

# With payload
SLIP v1 planner executor RequestTask auth refactor

# Fallback for unquantizable content
SLIP v1 devops sre Fallback check kubernetes pod logs

# With thread ID
SLIP v1 worker manager InformProgress thread42 milestone3
```

---

## 4. Think-Quantize-Transmit Pattern

The core workflow for semantic quantization:

### 4.1 Think
Agent formulates intent in natural language:
```
"I need someone to review this code for security vulnerabilities"
```

### 4.2 Quantize
Map thought to nearest UCR anchor:
```python
result = quantize(thought)
# → UCRAnchor(mnemonic="RequestReview", confidence=0.85)
```

### 4.3 Transmit
Send wire-format message:
```
SLIP v1 developer reviewer RequestReview
```

### 4.4 Reconstruct
Receiver looks up anchor and reconstructs intent:
```python
msg = decode(wire)
# → canonical: "Request review of work"
# → coords: (3, 4, 3, 3)
```

---

## 5. Fallback Mechanism

When quantization confidence is low, use natural language fallback:

```python
result = quantize("check kubernetes pods for OOMKilled events")
if result.confidence < threshold:
    wire = fallback(src, dst, original_text)
    # → SLIP v1 devops sre Fallback check kubernetes pods for OOMKilled events
```

Fallback messages are logged for UCR evolution analysis.

---

## 6. Extension Layer

### 6.1 Local Anchors

Installations can add domain-specific anchors in the 0x8000+ range:

```python
manager = ExtensionManager()
anchor = manager.add_extension(
    canonical="Request Kubernetes cluster scaling",
    mnemonic="RequestK8sScale",
)
# → index: 0x8000
```

### 6.2 UCR Evolution

1. **Track fallbacks**: Log patterns that don't quantize well
2. **Identify gaps**: Find frequently occurring fallback patterns
3. **Create extensions**: Add local anchors for common patterns
4. **Propose promotions**: Export extensions for core UCR consideration

---

## 7. Integration

### 7.1 AAIF Ecosystem

Slipstream operates at the **transport layer** beneath MCP and A2A:

```
┌─────────────────────────────────────┐
│   Application (Agent Logic)        │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   MCP / A2A (Semantic Layer)        │ ← Discovery, capabilities
└────────────────┬────────────────────┘
                 │ Natural language intent
┌────────────────▼────────────────────┐
│   Slipstream (Transport Layer)      │ ← Semantic quantization
│   Encode: Intent → UCR index        │
│   Wire: SLIP v1 A B Anchor          │
└────────────────┬────────────────────┘
                 │ 5 tokens (vs 50)
┌────────────────▼────────────────────┐
│   Network (HTTP, WebSocket, etc.)   │
└─────────────────────────────────────┘
```

### 7.2 Framework Compatibility

- **LangGraph / LangChain**: Message transformation layer
- **Custom orchestrators**: Encode/decode at system boundaries
- **Direct agent communication**: Native SLIP message passing

---

## 8. Versioning

### 8.1 UCR Versions

UCR follows semantic versioning:
- **Major**: Breaking changes to dimension structure
- **Minor**: New anchors added to core range
- **Patch**: Documentation, mnemonic clarifications

### 8.2 Protocol Versions

Wire format versions (v1, v2, ...) are backward compatible within major version.

---

## 9. Reference Implementation

See `slipcore` Python package:

```python
from slipcore import slip, decode, quantize, think_quantize_transmit

# Direct message creation
wire = slip("alice", "bob", "RequestReview")

# Full Think-Quantize-Transmit
wire = think_quantize_transmit(
    "Please review the auth code for security issues",
    src="dev",
    dst="reviewer"
)

# Decode
msg = decode(wire)
print(msg.anchor.canonical)  # "Request review of work"
```

---

## 10. Future Extensions

- **Hierarchical UCR**: Multi-level codebooks for domain specialization
- **Federated evolution**: Cross-installation UCR improvement
- **Binary format**: CBOR/protobuf encoding for high-frequency systems
- **Embedding quantizer**: sentence-transformers based matching
