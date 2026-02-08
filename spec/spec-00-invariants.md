# SLIP v3 Protocol Invariants

## Status: Normative

These invariants define correctness for any SLIP v3 implementation.
Conformance test vectors in `conformance/` verify these rules.

---

## 10 MUST Rules

1. **MUST start with `SLIP v3`**: Every wire message begins with the literal tokens `SLIP` and `v3`, space-separated.

2. **MUST contain exactly 6+ space-separated tokens**: `SLIP v3 <src> <dst> <force> <obj> [payload...]`. Messages with fewer than 6 tokens are invalid.

3. **MUST use closed Force vocabulary**: The `<force>` token MUST be one of exactly 12 values: `Observe`, `Inform`, `Ask`, `Request`, `Propose`, `Commit`, `Eval`, `Meta`, `Accept`, `Reject`, `Error`, `Fallback`.

4. **MUST use alphanumeric tokens only**: All tokens (src, dst, force, obj, payload) MUST match `[A-Za-z0-9]+`. No punctuation, no underscores, no special characters.

5. **MUST use agent IDs of 1-20 characters**: Both `<src>` and `<dst>` MUST be 1 to 20 alphanumeric characters.

6. **MUST use pointer reference for Fallback**: When `<force>` is `Fallback`, the payload MUST contain a reference token (e.g., `ref7f3a`). Raw natural language text MUST NOT appear on the wire.

7. **MUST keep payload tokens under 30 characters each**: Each payload token MUST be at most 30 alphanumeric characters.

8. **MUST limit payload to 20 tokens**: A wire message MUST NOT contain more than 20 payload tokens (tokens after position 6).

9. **MUST preserve token identity through roundtrip**: `parse(format(msg)) == msg` for all valid messages. The wire format is canonical.

10. **MUST treat core UCR anchors as immutable at runtime**: Core anchors (index < 0x8000) MUST NOT be added, removed, or modified after UCR initialization.

---

## 10 MUST NOT Rules

1. **MUST NOT use special characters in wire tokens**: No dots, hyphens, underscores, colons, brackets, or any non-alphanumeric characters in any wire token.

2. **MUST NOT transmit raw natural language on wire**: The wire format carries semantic pointers, not human text. Natural language belongs in out-of-band storage accessed via fallback refs.

3. **MUST NOT import non-stdlib modules in core**: The `slipcore` package (excluding `slipcore_ml` and `slipcore_tools`) MUST have zero external dependencies. Only Python stdlib imports are allowed.

4. **MUST NOT perform I/O at import time**: Importing `slipcore` MUST NOT read environment variables, touch the network, load files, configure logging, or create files.

5. **MUST NOT use mutable global state for UCR**: No module-level `_default_ucr` pattern. UCR instances are explicitly created and passed.

6. **MUST NOT accept v1/v2 wire format in v3 parser**: The v3 parser MUST reject messages with version tokens other than `v3`. Use `from_v2_mnemonic()` for explicit migration.

7. **MUST NOT allow Force vocabulary extension at runtime**: The 12 Force tokens are a closed set defined in code. Only Object tokens can be extended.

8. **MUST NOT swallow errors silently in core**: Parse errors, validation errors, and anchor-not-found errors MUST raise typed exceptions from `slipcore.errors`.

9. **MUST NOT require ML dependencies for basic operation**: `format_slip()`, `parse_slip()`, `validate_wire()`, and `render_human()` MUST work with zero external dependencies.

10. **MUST NOT use BPE-fragmenting patterns**: CamelCase tokens, space-separated fields, and alphanumeric-only content are chosen to minimize BPE token count. Implementations MUST NOT introduce patterns that fragment poorly (e.g., `snake_case`, JSON, special characters).
