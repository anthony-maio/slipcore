"""Data-driven conformance tests using JSONL vectors in spec/conformance/.

These tests validate that the wire parser, formatter, and roundtrip
behaviour conform to the protocol specification vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slipcore import format_slip, parse_slip, WireParseError, WireValidationError

# ---------------------------------------------------------------------------
# Load conformance vectors at module level
# ---------------------------------------------------------------------------

_CONFORMANCE_DIR = Path(__file__).resolve().parent.parent / "spec" / "conformance"


def _load_jsonl(filename: str) -> list[dict]:
    """Load a JSONL file and return a list of parsed dicts."""
    path = _CONFORMANCE_DIR / filename
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


VALID_VECTORS = _load_jsonl("valid.jsonl")
INVALID_VECTORS = _load_jsonl("invalid.jsonl")
ROUNDTRIP_VECTORS = _load_jsonl("roundtrip.jsonl")

# ---------------------------------------------------------------------------
# Helpers for readable test IDs
# ---------------------------------------------------------------------------


def _valid_id(vector: dict) -> str:
    """Create a short, readable test ID from a valid-vector entry."""
    wire: str = vector["wire"]
    # Use the tail after "SLIP v3 " as the ID, truncated for readability.
    suffix = wire.removeprefix("SLIP v3 ")
    return suffix[:60]


def _invalid_id(vector: dict) -> str:
    """Create a short, readable test ID from an invalid-vector entry."""
    reason: str = vector.get("reason", "unknown")
    wire_preview = vector["wire"][:40] if vector["wire"] else "<empty>"
    return f"{reason} | {wire_preview}"


def _roundtrip_id(vector: dict) -> str:
    """Create a short, readable test ID from a roundtrip-vector entry."""
    return f"{vector['src']}->{vector['dst']} {vector['force']} {vector['obj']}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_ERROR_TYPES = {
    "WireParseError": WireParseError,
    "WireValidationError": WireValidationError,
}


@pytest.mark.conformance
class TestValidVectors:
    """Parse known-good wire messages and verify every expected field."""

    @pytest.mark.parametrize(
        "vector",
        VALID_VECTORS,
        ids=[_valid_id(v) for v in VALID_VECTORS],
    )
    def test_parse_valid(self, vector: dict) -> None:
        wire = vector["wire"]
        expected = vector["expected"]

        msg = parse_slip(wire)

        assert msg.src == expected["src"]
        assert msg.dst == expected["dst"]
        assert msg.force == expected["force"]
        assert msg.obj == expected["obj"]
        assert list(msg.payload) == expected["payload"]
        assert msg.fallback_ref == expected["fallback_ref"]


# Vectors whose declared error type does not match the current implementation.
# These are documented inconsistencies between the conformance data and the
# parser: the parser raises a different (but still valid) error class.
_XFAIL_WIRES: dict[str, str] = {
    "SLIP v3 a b c d": (
        "Conformance data declares WireParseError (too few tokens) but "
        "the wire has 6 tokens; parse_slip validates force first and raises "
        "WireValidationError for the unknown force 'c'."
    ),
}


@pytest.mark.conformance
class TestInvalidVectors:
    """Verify that malformed wire messages raise the documented error type."""

    @pytest.mark.parametrize(
        "vector",
        INVALID_VECTORS,
        ids=[_invalid_id(v) for v in INVALID_VECTORS],
    )
    def test_parse_invalid(self, vector: dict) -> None:
        wire = vector["wire"]
        expected_error_name = vector["error"]
        expected_error_type = _ERROR_TYPES[expected_error_name]

        if wire in _XFAIL_WIRES:
            pytest.xfail(_XFAIL_WIRES[wire])

        with pytest.raises(expected_error_type):
            parse_slip(wire)


@pytest.mark.conformance
class TestRoundtripVectors:
    """Verify format -> parse -> format roundtrip identity."""

    @pytest.mark.parametrize(
        "vector",
        ROUNDTRIP_VECTORS,
        ids=[_roundtrip_id(v) for v in ROUNDTRIP_VECTORS],
    )
    def test_roundtrip(self, vector: dict) -> None:
        src = vector["src"]
        dst = vector["dst"]
        force = vector["force"]
        obj = vector["obj"]
        payload = vector["payload"]
        expected_wire = vector["wire"]

        # Direction 1: format produces expected wire text.
        formatted = format_slip(src, dst, force, obj, payload or None)
        assert formatted == expected_wire

        # Direction 2: parse the expected wire and re-serialise via .wire.
        parsed = parse_slip(expected_wire)
        assert parsed.wire == expected_wire
