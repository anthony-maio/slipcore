# Start Here (Python-First)

This guide is the fastest path to production adoption.

## 1. Install

```bash
pip install slipcore
```

## 2. Format and Parse

```python
from slipcore import format_slip, parse_slip

wire = format_slip("planner", "executor", "Request", "Task", ["auth"])
msg = parse_slip(wire)

assert msg.wire == wire
assert msg.force == "Request"
assert msg.obj == "Task"
```

## 3. Quantize Natural Language

```python
from slipcore import quantize

wire = quantize(
    "Please review the authentication module",
    src="dev",
    dst="reviewer",
)
print(wire)
```

## 4. Handle Fallback Correctly

`Fallback` requires a reference token on the wire.

```python
from slipcore import format_fallback

wire = format_fallback("qa", "planner", "ref7f3a")
```

Legacy migration only:

```python
from slipcore import parse_slip_legacy

msg = parse_slip_legacy("SLIP v3 qa planner Fallback Generic")
assert msg.fallback_ref == "reflegacy"
```

## 5. Conformance Checklist

- Use only alphanumeric tokens (`[A-Za-z0-9]+`).
- Keep `src`/`dst` between 1 and 20 chars.
- Keep payload tokens <= 30 chars, max 20 tokens.
- Keep Force in the closed 12-token vocabulary.

## Next Docs

- [SDK guide](sdk-guide.md)
- [v3 -> v4 migration](migration-v4.md)
- [Protocol invariants](../spec/spec-00-invariants.md)
- [A2A extension](../extensions/a2a-slipstream/v1/README.md)
