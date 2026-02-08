# Parse Slipstream Message

Parse and explain the given Slipstream v3 wire format message: $ARGUMENTS

Use the slipcore Python library to parse the message and provide a human-readable explanation including:
1. Source and destination agents
2. The Force (action verb) and Object (domain noun)
3. Any payload content
4. Human-readable rendering

Run this Python code to parse:
```python
from slipcore import parse_slip, render_human, render_log_line, validate_wire

wire = "$ARGUMENTS"

# Validate first
issues = validate_wire(wire)
if issues:
    print(f"Validation issues: {issues}")

# Parse
msg = parse_slip(wire)

print(f"From: {msg.src} -> {msg.dst}")
print(f"Force: {msg.force}")
print(f"Object: {msg.obj}")
if msg.payload:
    print(f"Payload: {' '.join(msg.payload)}")
if msg.fallback_ref:
    print(f"Fallback Ref: {msg.fallback_ref}")
print(f"Human: {render_human(msg)}")
```

Example:
- Input: `SLIP v3 alice bob Request Review auth`
- Output: alice asks bob to review auth (Force=Request, Object=Review)
