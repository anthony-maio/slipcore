# Paper / Spec / Code Claim Map

This table is a release gate for normative changes.
Canonical tracked paper source: `docs/paper/slipstream-v3.1.md`.

| Paper Section | Protocol Claim | Spec Source | Code Source | Test Source |
|---|---|---|---|---|
| Protocol Specification (wire grammar) | `SLIP v3 <src> <dst> <Force> <Object> [payload...]` and token bounds | `spec/spec-00-invariants.md` rules 1-5, 7-8 | `src/slipcore/wire.py` (`format_slip`, `parse_slip`, `validate_wire`) | `tests/test_wire.py`, `tests/test_conformance.py` |
| Fallback Mechanism | `Fallback` uses pointer refs only, no raw text | `spec/spec-00-invariants.md` rule 6 | `src/slipcore/wire.py` (`format_fallback`, fallback checks) | `tests/test_wire.py`, `spec/conformance/invalid.jsonl` |
| Force Vocabulary Closure | Force set is closed and non-extensible | `spec/spec-00-invariants.md` rule 3 and MUST NOT 7 | `src/slipcore/intent.py`, `src/slipcore/ucr.py` | `tests/test_intent.py`, `tests/test_ucr.py` |
| Core Anchor Immutability | Core anchors are immutable at runtime | `spec/spec-00-invariants.md` rule 10 | `src/slipcore/ucr.py` (`allow_core_mutation` lock) | `tests/test_ucr.py` |
| Legacy Migration | v2/v3 legacy compatibility is explicit, not implicit | `spec/spec-00-invariants.md` MUST NOT 6 | `src/slipcore/intent.py` (`from_v2_mnemonic`), `src/slipcore/wire.py` (`parse_slip_legacy`) | `tests/test_intent.py`, `tests/test_wire.py` |

## Release Rule

Any normative change must update all five columns in the same release:

1. Paper text
2. Spec
3. Code
4. Tests
5. Changelog / release checklist
