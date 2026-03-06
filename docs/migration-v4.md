# Migration to slipcore 4.0.0

`slipcore 4.0.0` keeps the wire token as `SLIP v3`, but tightens protocol enforcement.

## Breaking Changes

## 1. Strict fallback parsing

Old permissive behavior accepted missing fallback refs:

```text
SLIP v3 qa planner Fallback Generic
```

This is now invalid. Use:

```text
SLIP v3 qa planner Fallback Generic ref7f3a
```

Temporary migration helper:

```python
from slipcore import parse_slip_legacy
msg = parse_slip_legacy("SLIP v3 qa planner Fallback Generic")
```

## 2. Force vocabulary closure for extensions

`UCR.propose_extension()` now rejects unknown Force tokens.

Use a valid Force + new Object token, for example:

```python
from slipcore import create_base_ucr

ucr = create_base_ucr()
anchor = ucr.propose_extension(
    force="Request",
    obj="DeployContainer",
    canonical="Request container deployment",
    coords=(3, 4, 5, 4),
)
```

## 3. Core anchor immutability

Core range (`0x0000`-`0x7FFF`) is locked after base UCR creation and load.

## Upgrade Steps

1. Replace permissive fallback producers with `format_fallback()`.
2. Audit extension creation to ensure Force is one of the 12 canonical tokens.
3. Stop writing custom core-range anchors at runtime.
4. Run conformance tests from `spec/conformance/`.

## Validation Command

```bash
PYTHONPATH=src pytest tests -v
```
