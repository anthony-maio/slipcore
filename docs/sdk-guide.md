# Slipstream v3 SDK guide

Slipstream compresses multi-agent coordination messages from ~42 tokens (JSON) to ~7 tokens. It does this by transmitting pointers to shared concepts instead of the concepts themselves.

```
JSON:   {"from": "alice", "to": "bob", "type": "request", "action": "review", "target": "auth_module"}
         ~45 tokens

SLIP:   SLIP v3 alice bob Request Review auth
         7 tokens
```

The v3 wire format uses a factorized intent model. Instead of 46 flat CamelCase mnemonics (`RequestReview`), intents split into a Force token (12 closed action verbs) and an Object token (31+ extensible domain nouns): `Request` + `Review`.

This guide covers the `slipcore` Python SDK (v3.1.0).

---

## Table of contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Wire format reference](#3-wire-format-reference)
4. [The intent model](#4-the-intent-model)
5. [The UCR (Universal Concept Reference)](#5-the-ucr-universal-concept-reference)
6. [Quantization (Think-Quantize-Transmit)](#6-quantization-think-quantize-transmit)
7. [Fallback handling](#7-fallback-handling)
8. [Events and observability](#8-events-and-observability)
9. [Extensions](#9-extensions)
10. [Human rendering](#10-human-rendering)
11. [Error handling](#11-error-handling)
12. [Migrating from v2](#12-migrating-from-v2)
13. [Integration patterns](#13-integration-patterns)

---

## 1. Installation

```bash
pip install slipcore
```

Requires Python 3.10 or later. The core package has zero dependencies -- it uses only the Python standard library.

### Optional extras

| Extra | Install command | What it adds |
|-------|----------------|--------------|
| `ml` | `pip install slipcore[ml]` | numpy, sentence-transformers, scikit-learn |
| `a2a` | `pip install slipcore[a2a]` | httpx for A2A protocol integration |
| `dev` | `pip install slipcore[dev]` | pytest, mypy, ruff, hypothesis + all above |
| `all` | `pip install slipcore[all]` | ml + a2a |

For development from source:

```bash
git clone https://github.com/anthony-maio/slipcore.git
cd slipcore
pip install -e ".[dev]"
```

---

## 2. Quick start

```python
from slipcore import format_slip, parse_slip, render_human

# Format a wire message
wire = format_slip("alice", "bob", "Request", "Review", ["auth"])
print(wire)
# SLIP v3 alice bob Request Review auth

# Parse it back
msg = parse_slip(wire)
print(msg.force, msg.obj, msg.payload)
# Request Review ('auth',)

# Render for humans
print(render_human(msg))
# [alice -> bob] Request Review: "Request work review" (payload: auth)
```

That is the full read-write-render cycle. `format_slip` produces a wire string, `parse_slip` turns it back into a `SlipMessage` dataclass, and `render_human` makes it readable.

---

## 3. Wire format reference

### Grammar

Every SLIP v3 message follows this structure:

```
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

The formal grammar in ABNF:

```abnf
slip-message = "SLIP" SP "v3" SP agent-id SP agent-id SP force SP object
               0*20( SP payload-token )
agent-id     = 1*20( ALPHA / DIGIT )
force        = "Observe" / "Inform" / "Ask" / "Request" / "Propose" /
               "Commit" / "Eval" / "Meta" / "Accept" / "Reject" /
               "Error" / "Fallback"
object       = 1*30( ALPHA / DIGIT )
payload-token = 1*30( ALPHA / DIGIT )
```

### Why this format matters for tokenization

BPE tokenizers fragment special characters. A JSON message like `{"type": "request"}` costs far more tokens than the content warrants because `{`, `"`, `:`, and `}` each become separate tokens. SLIP v3 avoids this by using:

- Space-separated tokens (spaces are free in most tokenizers)
- Alphanumeric-only content (no punctuation to fragment)
- CamelCase identifiers (often a single BPE token each)

### SlipMessage dataclass

`parse_slip()` returns a frozen `SlipMessage`:

```python
@dataclass(frozen=True, slots=True)
class SlipMessage:
    version: str                          # Always "v3"
    src: str                              # Sender agent ID
    dst: str                              # Destination agent ID
    force: str                            # Force token (e.g., "Request")
    obj: str                              # Object token (e.g., "Review")
    payload: tuple[str, ...]              # Additional context tokens
    fallback_ref: Optional[str]           # Pointer ref when force is "Fallback"
```

Properties:

| Property | Type | Description |
|----------|------|-------------|
| `is_fallback` | `bool` | `True` when `force == "Fallback"` |
| `wire` | `str` | Reconstructed wire string |
| `token_count_estimate` | `int` | Number of space-separated tokens |

### format_slip()

```python
def format_slip(
    src: str,
    dst: str,
    force: str,
    obj: str,
    payload: Optional[list[str] | tuple[str, ...]] = None,
) -> str
```

Formats a validated SLIP v3 wire message. Raises `WireValidationError` if any token is invalid.

```python
from slipcore import format_slip

# Minimal message
wire = format_slip("planner", "exec", "Request", "Task")
# "SLIP v3 planner exec Request Task"

# With payload context
wire = format_slip("planner", "exec", "Request", "Task", ["auth", "refactor"])
# "SLIP v3 planner exec Request Task auth refactor"
```

Constraints enforced by `format_slip`:

- `src` and `dst`: 1-20 alphanumeric characters
- `force`: must be one of the 12 Force tokens
- `obj`: 1-30 alphanumeric characters
- Each payload token: 1-30 alphanumeric characters
- Maximum 20 payload tokens

### format_fallback()

```python
def format_fallback(src: str, dst: str, ref: str) -> str
```

Formats a fallback message with a pointer reference to locally stored text.

```python
from slipcore import format_fallback

wire = format_fallback("qa", "planner", "ref7f3a")
# "SLIP v3 qa planner Fallback Generic ref7f3a"
```

The `ref` must be 1-16 alphanumeric characters.

### parse_slip()

```python
def parse_slip(raw: str) -> SlipMessage
```

Parses a raw wire string into a `SlipMessage`. Raises `WireParseError` for structural issues and `WireValidationError` for constraint violations.

```python
from slipcore import parse_slip, WireParseError

msg = parse_slip("SLIP v3 planner exec Request Plan timeline")
print(msg.src)       # "planner"
print(msg.dst)       # "exec"
print(msg.force)     # "Request"
print(msg.obj)       # "Plan"
print(msg.payload)   # ("timeline",)

# Error handling
try:
    parse_slip("not a slip message")
except WireParseError as e:
    print(e)  # "Too short: need >= 6 tokens, got 5"
    print(e.raw)  # "not a slip message"
```

For fallback messages, the first payload token is extracted into `fallback_ref`. In `slipcore 3.1.0`, fallback messages without a ref are invalid and raise `WireValidationError`:

```python
msg = parse_slip("SLIP v3 qa planner Fallback Generic ref7f3a")
print(msg.is_fallback)    # True
print(msg.fallback_ref)   # "ref7f3a"
print(msg.payload)        # ()
```

### validate_wire()

```python
def validate_wire(raw: str) -> list[str]
```

Soft validation that returns a list of issues instead of raising exceptions. An empty list means the message is valid.

```python
from slipcore import validate_wire

issues = validate_wire("SLIP v3 alice bob Request Review")
print(issues)  # []

issues = validate_wire("SLIP v2 alice bob UnknownForce Review")
print(issues)  # ["Bad version: 'v2'", "Unknown force: 'UnknownForce'"]
```

Use `validate_wire` when you want to check messages without try/except, for example in middleware or logging pipelines.

---

## 4. The intent model

Slipstream v3 factors every agent intent into two tokens: a **Force** (what kind of speech act) and an **Object** (what domain concept it targets).

This reduces the classification problem from 46-way (v2 flat mnemonics) to 12-way (Force) + 31-way (Object), which is easier for both keyword classifiers and finetuned models to learn.

### Force tokens (closed set, 12 members)

Force tokens are an enum defined in `slipcore.intent.ForceToken`. This set is immutable within a major version.

| Force | Enum | Description | ACTION coord |
|-------|------|-------------|--------------|
| `Observe` | `ForceToken.OBSERVE` | Passively notice state, change, or error | 0 |
| `Inform` | `ForceToken.INFORM` | Report information outward | 1 |
| `Ask` | `ForceToken.ASK` | Request information from another agent | 2 |
| `Request` | `ForceToken.REQUEST` | Ask another agent to perform an action | 3 |
| `Propose` | `ForceToken.PROPOSE` | Suggest a plan, change, or alternative | 4 |
| `Commit` | `ForceToken.COMMIT` | Commit to a task, deadline, or resource | 5 |
| `Eval` | `ForceToken.EVAL` | Evaluate submitted work | 6 |
| `Meta` | `ForceToken.META` | Protocol-level operations (ack, sync, handoff) | 7 |
| `Accept` | `ForceToken.ACCEPT` | Accept a proposal or request | 5 |
| `Reject` | `ForceToken.REJECT` | Decline a proposal or request | 5 |
| `Error` | `ForceToken.ERROR` | Report a system error | 1 |
| `Fallback` | `ForceToken.FALLBACK` | Content too specific for standard vocabulary | 7 |

```python
from slipcore import ForceToken

# Iterate all forces
for f in ForceToken:
    print(f.value, f.action_coord)

# Access directly
print(ForceToken.REQUEST.value)  # "Request"
```

### Object tokens (extensible, 31 core members)

Object tokens are `ObjectToken` dataclasses registered in `CORE_OBJECTS`. Each carries semantic coordinates used for nearest-neighbor lookup in the UCR.

```python
@dataclass(frozen=True, slots=True)
class ObjectToken:
    mnemonic: str          # Wire token (e.g., "Plan")
    canonical: str         # Human description (e.g., "Plan creation")
    domain_coord: int      # Domain axis (0-7)
    polarity_coord: int    # Polarity axis (0-7)
    urgency_coord: int     # Urgency axis (0-7)
    is_core: bool          # True for core, False for extensions
```

Core objects grouped by typical Force usage:

| Group | Objects |
|-------|---------|
| Observation targets | `State`, `Change`, `Error` |
| Information types | `Result`, `Status`, `Complete`, `Blocked`, `Progress` |
| Questions | `Clarify`, `Permission`, `Resource` |
| Action requests | `Task`, `Plan`, `Review`, `Help`, `Cancel`, `Priority`, `Resource` |
| Proposals | `Plan`, `Change`, `Alternative`, `Rollback` |
| Commitments | `Task`, `Deadline`, `Resource` |
| Evaluations | `Approve`, `NeedsWork` |
| Protocol ops | `Ack`, `Sync`, `Handoff`, `Escalate`, `Abort` |
| Responses | `Condition`, `Defer` |
| Error types | `Timeout`, `Validation`, `Generic` |

```python
from slipcore import CORE_OBJECTS

# Look up an object
plan = CORE_OBJECTS["Plan"]
print(plan.canonical)       # "Plan creation"
print(plan.domain_coord)    # 1
print(plan.polarity_coord)  # 4
print(plan.urgency_coord)   # 4

# List all core objects
for name, obj in CORE_OBJECTS.items():
    print(f"{name}: {obj.canonical}")
```

Note: Force and Object are independent. Any Force can combine with any Object on the wire. The table above shows typical pairings, not constraints.

### resolve_intent()

```python
def resolve_intent(force_str: str, obj_str: str) -> Intent
```

Validates a Force-Object pair and returns a structured `Intent`.

```python
from slipcore import resolve_intent

intent = resolve_intent("Request", "Plan")
print(intent.force)          # ForceToken.REQUEST
print(intent.obj.mnemonic)   # "Plan"
print(intent.coords)         # (3, 4, 1, 4)  -- (action, polarity, domain, urgency)
print(intent.wire_tokens)    # ("Request", "Plan")
print(intent.mnemonic)       # "RequestPlan"
```

Raises `ValueError` if the Force or Object is unknown.

### from_v2_mnemonic()

```python
def from_v2_mnemonic(mnemonic: str) -> Intent
```

Converts a v2 flat mnemonic to a v3 `Intent`. See [section 12](#12-migrating-from-v2) for details.

---

## 5. The UCR (Universal Concept Reference)

The UCR is a registry of anchors -- named positions in a 4-dimensional semantic coordinate space. Each anchor maps a Force-Object pair to coordinates along four axes:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| ACTION | 0-7 | observe, inform, ask, request, propose, commit, evaluate, meta |
| POLARITY | 0-7 | negative (0) to positive (7) valence |
| DOMAIN | 0-7 | task, plan, observation, evaluation, control, resource, error, general |
| URGENCY | 0-7 | background (0) to critical (7) |

Address ranges:

- Core: `0x0000` - `0x7FFF` -- standard anchors, immutable at runtime
- Extension: `0x8000` - `0xFFFF` -- installation-specific, evolvable

### create_base_ucr()

```python
def create_base_ucr(authority_id: str = "slipcore-default") -> UCR
```

Returns a UCR instance populated with the 45 core anchors.

```python
from slipcore import create_base_ucr

ucr = create_base_ucr()
print(len(ucr))  # 45
```

### Looking up anchors

By Force-Object pair:

```python
anchor = ucr.get_by_force_obj("Request", "Plan")
print(anchor.index)      # 0x0031
print(anchor.canonical)  # "Request plan creation"
print(anchor.coords)     # (3, 4, 1, 4)
print(anchor.state)      # AnchorState.ACTIVE
```

By index:

```python
anchor = ucr.get_by_index(0x0031)
print(anchor.force, anchor.obj)  # "Request" "Plan"
```

Nearest-neighbor lookup (Manhattan distance on 4D coordinates):

```python
# Find the closest anchor to arbitrary coordinates
nearest = ucr.find_nearest((3, 4, 1, 4))
print(nearest.mnemonic)  # "RequestPlan"
```

`find_nearest` skips deprecated anchors and breaks ties by lowest index.

### Listing anchors

```python
# All core anchors
for anchor in ucr.core_anchors():
    print(f"0x{anchor.index:04X} {anchor.force} {anchor.obj}")

# All extension anchors
for anchor in ucr.extension_anchors():
    print(f"0x{anchor.index:04X} {anchor.force} {anchor.obj}")

# Iterate all
for anchor in ucr:
    print(anchor.mnemonic)
```

### Content hashing

```python
hash_str = ucr.content_hash()
print(hash_str)  # 16-character hex string, e.g., "a3f7b2c1d4e5f6a7"
```

The content hash is a truncated SHA-256 of all anchor data sorted by index. Use it to verify that two agents share the same UCR version.

### Save and load

```python
from pathlib import Path

# Save to JSON
ucr.save(Path("my_ucr.json"))

# Load from JSON
ucr2 = UCR.load(Path("my_ucr.json"))
```

The saved JSON includes the authority metadata and all anchors with their coordinates, states, and provenance.

### UCRAnchor fields

```python
@dataclass(slots=True)
class UCRAnchor:
    index: int                        # Unique address (0x0000-0xFFFF)
    force: str                        # Force token value
    obj: str                          # Object token mnemonic
    canonical: str                    # Human-readable description
    coords: tuple[int, int, int, int] # (action, polarity, domain, urgency)
    is_core: bool                     # True for core range
    state: AnchorState                # DRAFT, PROPOSED, APPROVED, ACTIVE, DEPRECATED
    created_by: str                   # Provenance (e.g., "core", "swarm")
    replaced_by: Optional[int]        # Index of replacement, if deprecated
```

### AnchorState lifecycle

```
DRAFT -> PROPOSED -> APPROVED -> ACTIVE -> DEPRECATED
```

Core anchors are always `ACTIVE`. Extension anchors start as `DRAFT`.

---

## 6. Quantization (Think-Quantize-Transmit)

Quantization maps natural language thoughts to Force-Object pairs. The core package includes a keyword-based quantizer with zero external dependencies.

### KeywordQuantizer

```python
from slipcore import KeywordQuantizer

q = KeywordQuantizer(fallback_threshold=0.2)
result = q.quantize("Please review the authentication code")
print(result.force)       # "Request"
print(result.obj)         # "Review"
print(result.confidence)  # 0.5 (example value, varies)
print(result.method)      # "keyword"
print(result.is_fallback) # False
```

The quantizer operates in two stages:

1. **Force classification**: Matches the input against keyword patterns for each Force token and picks the highest-scoring one.
2. **Object classification**: Matches against keyword patterns for each Object token.

If no Force pattern exceeds the `fallback_threshold` (default 0.2), the result is a fallback.

### QuantizeResult

```python
@dataclass
class QuantizeResult:
    force: str           # Matched Force token
    obj: str             # Matched Object token
    confidence: float    # Combined confidence (0.0 - 1.0)
    method: str          # "keyword" or "fallback"
    is_fallback: bool    # True if confidence was too low
```

### quantize() convenience function

```python
def quantize(thought: str, src: str, dst: str) -> str
```

Quantizes a thought and returns a complete wire message. Uses a module-level singleton `KeywordQuantizer`.

```python
from slipcore import quantize

wire = quantize("The deployment is finished", src="deploy", dst="lead")
print(wire)  # "SLIP v3 deploy lead Inform Complete"
```

If confidence is too low, it returns a fallback message and stores the raw text in the quantizer's `FallbackStore`:

```python
wire = quantize("xyzzy plugh", src="a", dst="b")
print(wire)  # "SLIP v3 a b Fallback Generic refDEADBEEF"
```

### think_quantize_transmit()

```python
def think_quantize_transmit(thought: str, src: str, dst: str) -> str
```

The full TQT flow. Functionally identical to `quantize()` -- takes a natural language thought and produces a wire-ready SLIP message.

```python
from slipcore import think_quantize_transmit

wire = think_quantize_transmit(
    "Please review the auth code",
    src="dev", dst="reviewer"
)
print(wire)  # "SLIP v3 dev reviewer Request Review"
```

### get_default_quantizer()

```python
def get_default_quantizer() -> KeywordQuantizer
```

Returns the module-level singleton quantizer, created on first call. The singleton persists across calls, so its `FallbackStore` accumulates stored text and its usage stats accumulate.

```python
from slipcore import get_default_quantizer

q = get_default_quantizer()
print(q.get_usage_stats())     # {"Request.Review": 3, "_fallback": 1, ...}
print(q.get_fallback_rate())   # 0.25
```

### Usage stats

The `KeywordQuantizer` tracks usage for monitoring:

```python
q = KeywordQuantizer()
q.quantize("review the code")
q.quantize("check the tests")
q.quantize("xyzzy")

print(q.get_usage_stats())
# {"Request.Review": 1, "Request.Task": 1, "_fallback": 1}

print(q.get_fallback_rate())
# 0.333...
```

---

## 7. Fallback handling

When a thought cannot be quantized with sufficient confidence, the system falls back to pointer-based references. The raw text stays local; only a short ref token goes on the wire.

### FallbackStore

```python
from slipcore import FallbackStore

store = FallbackStore()

# Store text, get a ref token
ref = store.store("This is a complex thought that doesn't map to any anchor")
print(ref)  # "ref<8 hex chars>" e.g., "ref7f3a1b2c"

# Retrieve later
text = store.retrieve(ref)
print(text)  # "This is a complex thought that doesn't map to any anchor"

# Check size
print(len(store))  # 1

# Clear
store.clear()
```

### Wire format for fallbacks

```python
from slipcore import format_fallback, parse_slip

wire = format_fallback("qa", "planner", "ref7f3a1b2c")
print(wire)  # "SLIP v3 qa planner Fallback Generic ref7f3a1b2c"

msg = parse_slip(wire)
print(msg.is_fallback)    # True
print(msg.fallback_ref)   # "ref7f3a1b2c"
```

### When fallback triggers

The `quantize()` function automatically handles fallbacks:

1. The `KeywordQuantizer` scores the input against all Force keyword patterns.
2. If the best score is below `fallback_threshold` (default 0.2), it returns `is_fallback=True`.
3. `quantize()` stores the raw text in the quantizer's `FallbackStore` and emits a `FallbackEmitted` event.
4. The wire message uses `Fallback Generic <ref>`.

The receiving agent can look up the ref in a shared store or request the full text out-of-band.

---

## 8. Events and observability

Slipcore emits structured events during quantization, wire formatting, and extension proposals. Events are dispatched to registered sink callbacks.

### Registering sinks

```python
from slipcore import register_sink, unregister_sink, clear_sinks

def my_logger(event):
    print(f"[{event.event_type}] {event}")

register_sink(my_logger)

# ... use slipcore normally, events flow to my_logger ...

unregister_sink(my_logger)

# Or remove all sinks
clear_sinks()
```

Sinks must not raise exceptions. If a sink raises, the exception is silently swallowed so it does not break the caller.

### Event types

| Event class | `event_type` | Emitted when | Key fields |
|-------------|-------------|--------------|------------|
| `WireSent` | `"wire.sent"` | A wire message is sent | `src`, `dst`, `force`, `obj`, `token_count`, `timestamp` |
| `WireReceived` | `"wire.received"` | A wire message is received | `src`, `dst`, `force`, `obj`, `timestamp` |
| `Quantized` | `"quantized"` | A thought is quantized | `input_hash`, `force`, `obj`, `confidence`, `mode`, `timestamp` |
| `FallbackEmitted` | `"fallback.emitted"` | Quantization falls back | `ref`, `reason`, `confidence`, `raw_text_hash`, `timestamp` |
| `ExtensionProposed` | `"extension.proposed"` | A new extension anchor is added | `force`, `obj`, `created_by`, `timestamp` |

All events are frozen dataclasses with `slots=True` and carry a `timestamp` field (from `time.time()`).

### Building a logger sink

```python
import json
from slipcore import register_sink, quantize, Quantized, FallbackEmitted

def structured_logger(event):
    if isinstance(event, Quantized):
        print(json.dumps({
            "type": event.event_type,
            "force": event.force,
            "obj": event.obj,
            "confidence": event.confidence,
            "mode": event.mode,
        }))
    elif isinstance(event, FallbackEmitted):
        print(json.dumps({
            "type": event.event_type,
            "ref": event.ref,
            "reason": event.reason,
            "confidence": event.confidence,
        }))

register_sink(structured_logger)

quantize("Please review the auth code", "dev", "reviewer")
# Prints: {"type": "quantized", "force": "Request", "obj": "Review", ...}
```

### Emitting custom events

```python
from slipcore import emit, WireSent

emit(WireSent(
    src="planner",
    dst="exec",
    force="Request",
    obj="Task",
    token_count=7,
))
```

---

## 9. Extensions

The extension layer lets installations add domain-specific anchors beyond the 45-anchor core. Extensions live in the address range `0x8000` - `0xFFFF`.

### ExtensionManager

```python
from slipcore import ExtensionManager

ext = ExtensionManager()  # Uses a fresh base UCR by default

# Add a domain-specific anchor
anchor = ext.add_extension(
    force="Request",
    obj="Deploy",
    canonical="Request deployment to environment",
    coords=(3, 4, 0, 5),
    created_by="devops_team",
)
print(f"0x{anchor.index:04X}")  # "0x8000"
print(anchor.state)              # AnchorState.DRAFT
print(anchor.is_core)            # False
```

You can also pass an existing UCR to the manager:

```python
from slipcore import create_base_ucr, ExtensionManager

ucr = create_base_ucr()
ext = ExtensionManager(ucr=ucr)
```

### FallbackTracker for gap analysis

The `FallbackTracker` records thoughts that failed quantization and extracts n-gram patterns to identify gaps in your vocabulary.

```python
ext.record_fallback(
    thought="deploy the service to staging",
    src="dev",
    dst="ops",
    nearest_force="Request",
    nearest_obj="Task",
    nearest_score=0.15,
)

ext.record_fallback(
    thought="deploy the service to production",
    src="dev",
    dst="ops",
)

# See what patterns keep failing
stats = ext.fallback_tracker.get_stats()
print(stats["total_events"])     # 2
print(stats["top_patterns"])     # [("deploy the", 2), ("the service", 2), ...]
```

### suggest_extensions()

```python
suggestions = ext.suggest_extensions(min_count=2)
print(suggestions)
# ["Agent intent: deploy the", "Agent intent: the service", ...]
```

Returns patterns that appear at least `min_count` times in fallback events. Use these to identify candidates for new extension anchors.

### export_extensions()

```python
from pathlib import Path

ext.export_extensions(Path("my_extensions.json"))
```

Exports all extension anchors and fallback statistics to a JSON file. Use this to share vocabulary customizations across installations.

### Extension stats

```python
stats = ext.get_stats()
print(stats)
# {
#     "core_anchors": 45,
#     "extension_anchors": 1,
#     "fallback_stats": {"total_events": 2, ...}
# }
```

---

## 10. Human rendering

Two rendering functions convert `SlipMessage` objects (or raw wire strings) into human-readable output.

### render_human()

```python
def render_human(msg: SlipMessage | str, ucr: Optional[UCR] = None) -> str
```

Produces a bracketed, readable summary. Accepts either a `SlipMessage` or a raw wire string.

```python
from slipcore import render_human

print(render_human("SLIP v3 planner exec Request Plan timeline"))
# [planner -> exec] Request Plan: "Request plan creation" (payload: timeline)

print(render_human("SLIP v3 qa planner Fallback Generic ref7f3a"))
# [qa -> planner] Fallback Generic: "Unquantizable - see ref" (fallback ref: ref7f3a)
```

If you have a custom UCR with extensions, pass it in:

```python
from slipcore import create_base_ucr, render_human

ucr = create_base_ucr()
print(render_human("SLIP v3 alice bob Inform Complete", ucr=ucr))
# [alice -> bob] Inform Complete: "Report task completion"
```

### render_log_line()

```python
def render_log_line(msg: SlipMessage | str, ucr: Optional[UCR] = None) -> str
```

Produces a pipe-delimited structured log line for governance audit trails.

```python
from slipcore import render_log_line

print(render_log_line("SLIP v3 planner exec Request Plan timeline"))
# planner->exec | Request Plan | Request plan creation | payload=timeline | ref=-

print(render_log_line("SLIP v3 qa planner Fallback Generic ref7f3a"))
# qa->planner | Fallback Generic | Unquantizable - see ref | payload=- | ref=ref7f3a
```

---

## 11. Error handling

All exceptions inherit from `SlipError`. Import them from `slipcore` or `slipcore.errors`.

### Exception hierarchy

```
SlipError
    WireParseError
    WireValidationError
    UCRError
        AnchorNotFoundError
        UCRVersionMismatchError
    PolicyViolationError
    MissingExtraError
```

### When each error is raised

| Exception | Raised by | Condition |
|-----------|-----------|-----------|
| `WireParseError` | `parse_slip()` | Structural issue: too few tokens, wrong prefix, wrong version |
| `WireValidationError` | `format_slip()`, `format_fallback()`, `parse_slip()` | Constraint violation: bad agent ID, unknown Force, token too long, too many payload tokens |
| `UCRError` | `UCR.add_anchor()`, `UCR.find_nearest()` | Duplicate index, duplicate Force-Object pair, no anchors, all deprecated |
| `AnchorNotFoundError` | UCR lookups | Anchor not in registry |
| `UCRVersionMismatchError` | Version checks | UCR hash mismatch between agents |
| `PolicyViolationError` | Policy enforcement | Disallowed Force-Object combination |
| `MissingExtraError` | Optional feature access | Required extra not installed |

### WireParseError details

`WireParseError` carries the raw input for debugging:

```python
from slipcore import parse_slip, WireParseError

try:
    parse_slip("SLIP v2 alice bob Request Plan")
except WireParseError as e:
    print(e)        # "Bad version: 'v2'"
    print(e.raw)    # "SLIP v2 alice bob Request Plan"
    print(e.position)  # -1 (not always set)
```

### MissingExtraError

Raised when code tries to use a feature that requires an optional extra:

```python
from slipcore import MissingExtraError

# This would be raised internally, e.g., by an embedding quantizer:
# MissingExtraError("sentence_transformers", "ml")
# -> "Optional dependency 'sentence_transformers' required.
#     Install with: pip install slipcore[ml]"
```

### Handling patterns

```python
from slipcore import (
    parse_slip, format_slip, validate_wire,
    SlipError, WireParseError, WireValidationError,
)

# Pattern 1: Validate before parse (soft check)
issues = validate_wire(raw_wire)
if issues:
    log.warning(f"Invalid wire: {issues}")
else:
    msg = parse_slip(raw_wire)

# Pattern 2: Try-except with specific types
try:
    msg = parse_slip(raw_wire)
except WireParseError:
    # Structural problem, message is not SLIP at all
    handle_non_slip(raw_wire)
except WireValidationError:
    # Structurally SLIP but violates constraints
    handle_invalid_slip(raw_wire)

# Pattern 3: Catch all slipcore errors
try:
    wire = format_slip(src, dst, force, obj, payload)
except SlipError as e:
    log.error(f"Slipcore error: {e}")
```

---

## 12. Migrating from v2

### Wire format differences

| Aspect | v2 | v3 |
|--------|----|----|
| Version token | `v1` | `v3` |
| Intent | Single mnemonic: `RequestReview` | Two tokens: `Request Review` |
| Grammar | `SLIP v1 <src> <dst> <mnemonic> [payload...]` | `SLIP v3 <src> <dst> <Force> <Object> [payload...]` |
| Minimum tokens | 5 | 6 |

### API changes

| v2 | v3 | Notes |
|----|-----|-------|
| `slip(src, dst, anchor)` | `format_slip(src, dst, force, obj)` | Two intent args instead of one |
| `decode(wire)` | `parse_slip(wire)` | Returns `SlipMessage` with `.force` and `.obj` |
| `msg.anchor` | `msg.force`, `msg.obj` | Factorized fields |
| `quantize(thought)` | `quantize(thought, src, dst)` | Returns wire string directly |

### from_v2_mnemonic()

Maps all 46 v2 flat mnemonics to v3 Force-Object pairs:

```python
from slipcore import from_v2_mnemonic

intent = from_v2_mnemonic("RequestReview")
print(intent.force.value)      # "Request"
print(intent.obj.mnemonic)     # "Review"
print(intent.wire_tokens)      # ("Request", "Review")

intent = from_v2_mnemonic("MetaAbort")
print(intent.force.value)      # "Meta"
print(intent.obj.mnemonic)     # "Abort"
```

The full mapping is available as `V2_TO_V3`:

```python
from slipcore import V2_TO_V3

for v2_mnemonic, (force, obj) in V2_TO_V3.items():
    print(f"{v2_mnemonic} -> {force} {obj}")
```

Selected mappings:

| v2 mnemonic | v3 Force | v3 Object |
|-------------|----------|-----------|
| `ObserveState` | `Observe` | `State` |
| `InformComplete` | `Inform` | `Complete` |
| `AskClarify` | `Ask` | `Clarify` |
| `RequestReview` | `Request` | `Review` |
| `ProposePlan` | `Propose` | `Plan` |
| `CommitDeadline` | `Commit` | `Deadline` |
| `EvalApprove` | `Eval` | `Approve` |
| `MetaHandoff` | `Meta` | `Handoff` |
| `Accept` | `Accept` | `Generic` |
| `Reject` | `Reject` | `Generic` |
| `AcceptWithCondition` | `Accept` | `Condition` |
| `ErrorTimeout` | `Error` | `Timeout` |
| `Fallback` | `Fallback` | `Generic` |

### Migration example

```python
# v2 code
# wire = slip("alice", "bob", "RequestReview")
# msg = decode(wire)
# print(msg.anchor.canonical)

# v3 equivalent
from slipcore import format_slip, parse_slip

wire = format_slip("alice", "bob", "Request", "Review")
msg = parse_slip(wire)
print(msg.force, msg.obj)  # "Request" "Review"
```

### Migrating data

If you have v2 training data or logs with flat mnemonics, use `from_v2_mnemonic()` in a conversion script:

```python
from slipcore import from_v2_mnemonic, format_slip

def convert_v2_wire(v2_wire: str) -> str:
    """Convert a v2 wire message to v3."""
    tokens = v2_wire.strip().split()
    # v2 format: SLIP v1 <src> <dst> <mnemonic> [payload...]
    src, dst, mnemonic = tokens[2], tokens[3], tokens[4]
    payload = tokens[5:] if len(tokens) > 5 else None

    intent = from_v2_mnemonic(mnemonic)
    force, obj = intent.wire_tokens
    return format_slip(src, dst, force, obj, payload)

# Example
v3_wire = convert_v2_wire("SLIP v1 alice bob RequestReview auth")
print(v3_wire)  # "SLIP v3 alice bob Request Review auth"
```

### Migrating permissive fallback wires (v3.1+)

`slipcore 3.1.0` enforces fallback refs strictly. For old logs that emitted
`SLIP v3 ... Fallback Generic` without a ref token, use the explicit legacy parser:

```python
from slipcore import parse_slip_legacy

msg = parse_slip_legacy("SLIP v3 qa planner Fallback Generic")
print(msg.fallback_ref)  # "reflegacy"
```

---

## 13. Integration patterns

### A2A integration

Slipstream can serve as the content encoding inside Google A2A (Agent-to-Agent) protocol messages. The SLIP wire text goes in the `TextPart` of an A2A message, keeping the LLM-visible content compact while the A2A JSON envelope handles routing and discovery.

See `extensions/a2a-slipstream/v1/` for the draft extension spec. Key points:

- Agents advertise SLIP support in their Agent Card `capabilities.extensions[]`
- SLIP wire text goes in `message.parts[].text`
- UCR version and hash go in `message.metadata` for compatibility verification
- Heavy payloads (diffs, files) go in separate A2A `FilePart` or `DataPart` entries

```python
# Construct an A2A-compatible message body using the A2A helper
from slipcore import format_slip

# The A2A adapter lives in extensions/a2a-slipstream/v1/
# Import it directly or copy a2a_slipstream.py into your project
from a2a_slipstream import build_slip_a2a_message

wire = format_slip("planner", "reviewer", "Request", "Review")
msg = build_slip_a2a_message(slip_wire=wire)

# Or construct manually (use the extension URI as the metadata key):
SLIP_EXT = "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1"

a2a_message = {
    "role": "user",
    "parts": [{"text": wire}],
    "extensions": [SLIP_EXT],
    "metadata": {
        SLIP_EXT: {
            "slipVersion": "v3",
            "ucrVersion": "3.0.0",
            "encoding": "force-object",
        }
    }
}
```

Note: For UCR hash verification, use the A2A helper `stable_ucr_hash()` from `a2a_slipstream.py`, not `UCR.content_hash()`. They produce different formats -- `content_hash()` is a truncated 16-char internal identifier, while the A2A spec requires a full `sha256:`-prefixed hash.

### MCP transport layer pattern

Slipstream sits below the MCP (Model Context Protocol) or A2A semantic layer:

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

The application layer decides *what* to communicate. MCP or A2A handles discovery, tool schemas, and task lifecycle. Slipstream handles *how* the intent is encoded on the wire so LLMs can read it in minimal tokens.

### Multi-agent system example

```python
from slipcore import (
    format_slip, parse_slip, render_human,
    quantize, register_sink, Quantized,
)

# Set up observability
def audit_log(event):
    if isinstance(event, Quantized):
        print(f"AUDIT: {event.force}.{event.obj} confidence={event.confidence:.2f}")

register_sink(audit_log)

# Agent 1: Planner sends a task request
wire = format_slip("planner", "coder", "Request", "Task", ["auth", "refactor"])
print(wire)
# SLIP v3 planner coder Request Task auth refactor

# Agent 2: Coder receives and processes
msg = parse_slip(wire)
if msg.force == "Request" and msg.obj == "Task":
    # Do the work...

    # Respond with completion
    response = format_slip("coder", "planner", "Inform", "Complete", ["auth"])
    print(response)
    # SLIP v3 coder planner Inform Complete auth

# Agent 3: Reviewer uses TQT from natural language
review_wire = quantize(
    "I've reviewed the auth changes and they look good",
    src="reviewer", dst="planner"
)
print(render_human(review_wire))
# [reviewer -> planner] Eval Approve: "Evaluation: approved"
```

### Dispatch pattern

Route messages by Force token for clean handler architecture:

```python
from slipcore import parse_slip, SlipMessage

handlers = {
    "Request": handle_request,
    "Inform": handle_inform,
    "Ask": handle_ask,
    "Eval": handle_eval,
    "Error": handle_error,
    "Meta": handle_meta,
}

def dispatch(wire: str):
    msg = parse_slip(wire)
    handler = handlers.get(msg.force)
    if handler:
        handler(msg)
    else:
        handle_unknown(msg)
```

---

## Appendix: complete core anchor table

All 45 anchors in the base UCR, sorted by index.

| Index | Force | Object | Canonical | Coords (A,P,D,U) |
|-------|-------|--------|-----------|-------------------|
| `0x0001` | Observe | State | Report current state | (0,4,2,3) |
| `0x0002` | Observe | Change | Report detected change | (0,4,2,4) |
| `0x0003` | Observe | Error | Report observed error | (0,2,6,6) |
| `0x0010` | Inform | Result | Share computed result | (1,5,2,3) |
| `0x0011` | Inform | Status | Provide status update | (1,4,0,3) |
| `0x0012` | Inform | Complete | Report task completion | (1,6,0,4) |
| `0x0013` | Inform | Blocked | Report being blocked | (1,2,0,5) |
| `0x0014` | Inform | Progress | Share progress update | (1,5,0,3) |
| `0x0020` | Ask | Clarify | Request clarification | (2,4,1,4) |
| `0x0021` | Ask | Status | Query current status | (2,4,0,3) |
| `0x0022` | Ask | Permission | Request permission | (2,4,4,4) |
| `0x0023` | Ask | Resource | Query resource availability | (2,4,5,3) |
| `0x0030` | Request | Task | Request task execution | (3,4,0,4) |
| `0x0031` | Request | Plan | Request plan creation | (3,4,1,4) |
| `0x0032` | Request | Review | Request work review | (3,4,3,3) |
| `0x0033` | Request | Help | Request assistance | (3,4,7,5) |
| `0x0034` | Request | Cancel | Request cancellation | (3,1,4,5) |
| `0x0035` | Request | Priority | Request priority change | (3,4,4,5) |
| `0x0036` | Request | Resource | Request resource allocation | (3,4,5,4) |
| `0x0040` | Propose | Plan | Propose a plan | (4,5,1,4) |
| `0x0041` | Propose | Change | Propose modification | (4,5,0,4) |
| `0x0042` | Propose | Alternative | Propose alternative | (4,5,1,4) |
| `0x0043` | Propose | Rollback | Propose reverting | (4,3,4,5) |
| `0x0050` | Commit | Task | Commit to task | (5,6,0,4) |
| `0x0051` | Commit | Deadline | Commit to deadline | (5,6,0,4) |
| `0x0052` | Commit | Resource | Commit resources | (5,6,5,4) |
| `0x0060` | Eval | Approve | Evaluation: approved | (6,7,3,4) |
| `0x0061` | Eval | Review | Evaluation: under review | (6,4,3,4) |
| `0x0062` | Eval | NeedsWork | Evaluation: needs revision | (6,3,3,4) |
| `0x0063` | Eval | Complete | Evaluation: work complete | (6,6,3,4) |
| `0x0070` | Meta | Ack | Acknowledge receipt | (7,5,4,2) |
| `0x0071` | Meta | Sync | Synchronization ping | (7,4,4,3) |
| `0x0072` | Meta | Handoff | Hand off responsibility | (7,4,4,4) |
| `0x0073` | Meta | Escalate | Escalate to authority | (7,3,4,6) |
| `0x0074` | Meta | Abort | Abort operation | (7,0,4,7) |
| `0x0080` | Accept | Generic | Accept proposal/request | (5,7,7,3) |
| `0x0081` | Reject | Generic | Reject proposal/request | (5,0,7,3) |
| `0x0082` | Accept | Condition | Conditional acceptance | (5,5,7,4) |
| `0x0083` | Meta | Defer | Defer decision | (5,4,7,2) |
| `0x0090` | Error | Generic | Generic error | (1,1,6,5) |
| `0x0091` | Error | Timeout | Operation timed out | (1,1,6,5) |
| `0x0092` | Error | Resource | Resource unavailable | (1,1,6,5) |
| `0x0093` | Error | Permission | Permission denied | (1,0,6,5) |
| `0x0094` | Error | Validation | Validation failed | (1,1,6,4) |
| `0x00FF` | Fallback | Generic | Unquantizable - see ref | (7,4,7,4) |
