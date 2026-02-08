"""Tests for slipcore.wire -- SLIP v3 wire format parse, format, validate."""

from __future__ import annotations

import pytest

from slipcore import (
    SlipMessage,
    WireParseError,
    WireValidationError,
    format_fallback,
    format_slip,
    parse_slip,
    validate_wire,
)
from slipcore.intent import FORCE_VALUES, ForceToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_FORCE_TOKENS: list[str] = [f.value for f in ForceToken]


# ===================================================================
# format_slip
# ===================================================================


class TestFormatSlip:
    """Tests for format_slip()."""

    def test_basic(self) -> None:
        wire = format_slip("planner", "exec", "Request", "Plan")
        assert wire == "SLIP v3 planner exec Request Plan"

    def test_with_payload(self) -> None:
        wire = format_slip("dev", "reviewer", "Request", "Review", ["auth", "module"])
        assert wire == "SLIP v3 dev reviewer Request Review auth module"

    @pytest.mark.parametrize("force", ALL_FORCE_TOKENS)
    def test_all_force_tokens(self, force: str) -> None:
        wire = format_slip("src", "dst", force, "Task")
        assert wire == f"SLIP v3 src dst {force} Task"

    def test_invalid_src_special_chars(self) -> None:
        with pytest.raises(WireValidationError, match="src invalid"):
            format_slip("bad agent!", "dst", "Request", "Task")

    def test_invalid_src_empty(self) -> None:
        with pytest.raises(WireValidationError, match="src invalid"):
            format_slip("", "dst", "Request", "Task")

    def test_invalid_dst_special_chars(self) -> None:
        with pytest.raises(WireValidationError, match="dst invalid"):
            format_slip("src", "bad-dst", "Request", "Task")

    def test_invalid_dst_too_long(self) -> None:
        with pytest.raises(WireValidationError, match="dst invalid"):
            format_slip("src", "a" * 21, "Request", "Task")

    def test_invalid_force(self) -> None:
        with pytest.raises(WireValidationError, match="Unknown force"):
            format_slip("src", "dst", "Bogus", "Task")

    def test_invalid_obj_special_chars(self) -> None:
        with pytest.raises(WireValidationError, match="Object invalid"):
            format_slip("src", "dst", "Request", "bad-obj!")

    def test_invalid_obj_too_long(self) -> None:
        with pytest.raises(WireValidationError, match="Object invalid"):
            format_slip("src", "dst", "Request", "x" * 31)

    def test_invalid_payload_token_special_chars(self) -> None:
        with pytest.raises(WireValidationError, match="Payload.*not alphanumeric"):
            format_slip("src", "dst", "Request", "Task", ["good", "bad token!"])

    def test_invalid_payload_token_too_long(self) -> None:
        with pytest.raises(WireValidationError, match="Payload.*too long"):
            format_slip("src", "dst", "Request", "Task", ["x" * 31])

    def test_payload_too_many_tokens(self) -> None:
        with pytest.raises(WireValidationError, match="Payload too long"):
            format_slip("src", "dst", "Request", "Task", ["tok"] * 21)

    def test_empty_payload_list(self) -> None:
        wire = format_slip("src", "dst", "Request", "Task", [])
        assert wire == "SLIP v3 src dst Request Task"


# ===================================================================
# format_fallback
# ===================================================================


class TestFormatFallback:
    """Tests for format_fallback()."""

    def test_basic(self) -> None:
        wire = format_fallback("qa", "planner", "ref7f3a")
        assert wire == "SLIP v3 qa planner Fallback Generic ref7f3a"

    def test_invalid_ref_special_chars(self) -> None:
        with pytest.raises(WireValidationError, match="Invalid fallback ref"):
            format_fallback("qa", "planner", "bad ref!")

    def test_invalid_ref_too_long(self) -> None:
        with pytest.raises(WireValidationError, match="Invalid fallback ref"):
            format_fallback("qa", "planner", "a" * 17)

    def test_invalid_ref_empty(self) -> None:
        with pytest.raises(WireValidationError, match="Invalid fallback ref"):
            format_fallback("qa", "planner", "")

    def test_invalid_src(self) -> None:
        with pytest.raises(WireValidationError, match="src invalid"):
            format_fallback("bad!", "planner", "ref1")

    def test_invalid_dst(self) -> None:
        with pytest.raises(WireValidationError, match="dst invalid"):
            format_fallback("qa", "bad!", "ref1")


# ===================================================================
# parse_slip
# ===================================================================


class TestParseSlip:
    """Tests for parse_slip()."""

    def test_basic(self) -> None:
        msg = parse_slip("SLIP v3 planner exec Request Plan")
        assert msg.version == "v3"
        assert msg.src == "planner"
        assert msg.dst == "exec"
        assert msg.force == "Request"
        assert msg.obj == "Plan"
        assert msg.payload == ()
        assert msg.fallback_ref is None

    def test_with_payload(self) -> None:
        msg = parse_slip("SLIP v3 dev reviewer Request Review auth module")
        assert msg.force == "Request"
        assert msg.obj == "Review"
        assert msg.payload == ("auth", "module")

    def test_fallback_with_ref(self) -> None:
        msg = parse_slip("SLIP v3 qa planner Fallback Generic ref7f3a")
        assert msg.is_fallback
        assert msg.force == "Fallback"
        assert msg.obj == "Generic"
        assert msg.fallback_ref == "ref7f3a"
        assert msg.payload == ()

    def test_fallback_with_ref_and_payload(self) -> None:
        msg = parse_slip("SLIP v3 qa planner Fallback Generic ref7f3a extra1")
        assert msg.is_fallback
        assert msg.fallback_ref == "ref7f3a"
        assert msg.payload == ("extra1",)

    def test_too_short(self) -> None:
        with pytest.raises(WireParseError, match="Too short"):
            parse_slip("SLIP v3 src dst Request")

    def test_wrong_prefix(self) -> None:
        with pytest.raises(WireParseError, match="Bad prefix"):
            parse_slip("XSLIP v3 src dst Request Task")

    def test_wrong_version(self) -> None:
        with pytest.raises(WireParseError, match="Bad version"):
            parse_slip("SLIP v1 src dst Request Task")

    def test_invalid_src_in_parse(self) -> None:
        with pytest.raises(WireValidationError, match="src invalid"):
            parse_slip("SLIP v3 bad!src dst Request Task")

    def test_invalid_force_in_parse(self) -> None:
        with pytest.raises(WireValidationError, match="Unknown force"):
            parse_slip("SLIP v3 src dst Bogus Task")

    def test_strips_whitespace(self) -> None:
        msg = parse_slip("  SLIP v3 src dst Request Task  ")
        assert msg.src == "src"
        assert msg.dst == "dst"


# ===================================================================
# validate_wire
# ===================================================================


class TestValidateWire:
    """Tests for validate_wire()."""

    def test_valid_message_returns_empty(self) -> None:
        issues = validate_wire("SLIP v3 planner exec Request Plan")
        assert issues == []

    def test_valid_with_payload_returns_empty(self) -> None:
        issues = validate_wire("SLIP v3 dev reviewer Request Review auth")
        assert issues == []

    def test_too_few_tokens(self) -> None:
        issues = validate_wire("SLIP v3 src")
        assert len(issues) == 1
        assert "Too few tokens" in issues[0]

    def test_bad_prefix(self) -> None:
        issues = validate_wire("NOPE v3 src dst Request Task")
        assert any("Bad prefix" in i for i in issues)

    def test_bad_version(self) -> None:
        issues = validate_wire("SLIP v1 src dst Request Task")
        assert any("Bad version" in i for i in issues)

    def test_bad_src(self) -> None:
        issues = validate_wire("SLIP v3 bad!src dst Request Task")
        assert any("Bad src" in i for i in issues)

    def test_bad_dst(self) -> None:
        issues = validate_wire("SLIP v3 src bad!dst Request Task")
        assert any("Bad dst" in i for i in issues)

    def test_unknown_force(self) -> None:
        issues = validate_wire("SLIP v3 src dst Bogus Task")
        assert any("Unknown force" in i for i in issues)

    def test_bad_obj(self) -> None:
        issues = validate_wire("SLIP v3 src dst Request bad-obj!")
        assert any("Bad obj" in i for i in issues)

    def test_bad_payload_token(self) -> None:
        issues = validate_wire("SLIP v3 src dst Request Task good bad!tok")
        assert any("Bad payload" in i for i in issues)

    def test_payload_token_too_long(self) -> None:
        long_tok = "x" * 31
        issues = validate_wire(f"SLIP v3 src dst Request Task {long_tok}")
        assert any("too long" in i for i in issues)

    def test_too_many_payload_tokens(self) -> None:
        tokens = " ".join(["tok"] * 21)
        issues = validate_wire(f"SLIP v3 src dst Request Task {tokens}")
        assert any("Too many payload" in i for i in issues)

    def test_multiple_issues_reported(self) -> None:
        issues = validate_wire("NOPE v1 bad! bad! Bogus bad!")
        assert len(issues) >= 4


# ===================================================================
# SlipMessage properties
# ===================================================================


class TestSlipMessage:
    """Tests for SlipMessage dataclass properties."""

    def test_wire_property(self) -> None:
        msg = SlipMessage(
            version="v3",
            src="alice",
            dst="bob",
            force="Request",
            obj="Task",
            payload=["auth"],
        )
        assert msg.wire == "SLIP v3 alice bob Request Task auth"

    def test_wire_property_no_payload(self) -> None:
        msg = SlipMessage(
            version="v3", src="alice", dst="bob", force="Request", obj="Task"
        )
        assert msg.wire == "SLIP v3 alice bob Request Task"

    def test_is_fallback_true(self) -> None:
        msg = SlipMessage(
            version="v3",
            src="qa",
            dst="planner",
            force="Fallback",
            obj="Generic",
            fallback_ref="ref1",
        )
        assert msg.is_fallback is True
        assert "ref1" in msg.wire

    def test_is_fallback_false(self) -> None:
        msg = SlipMessage(
            version="v3", src="a", dst="b", force="Request", obj="Task"
        )
        assert msg.is_fallback is False

    def test_token_count_estimate(self) -> None:
        msg = SlipMessage(
            version="v3",
            src="src",
            dst="dst",
            force="Request",
            obj="Task",
            payload=["p1", "p2"],
        )
        assert msg.token_count_estimate == 8  # SLIP v3 src dst Request Task p1 p2

    def test_frozen(self) -> None:
        msg = SlipMessage(
            version="v3", src="a", dst="b", force="Request", obj="Task"
        )
        with pytest.raises(AttributeError):
            msg.src = "other"  # type: ignore[misc]


# ===================================================================
# Roundtrip: format -> parse -> .wire == original
# ===================================================================


class TestRoundtrip:
    """Roundtrip: format_slip -> parse_slip -> .wire must equal original."""

    @pytest.mark.parametrize("force", ALL_FORCE_TOKENS)
    def test_roundtrip_all_forces_with_payload(self, force: str) -> None:
        if force == "Fallback":
            # Fallback uses format_fallback, tested separately
            original = format_fallback("src", "dst", "ref1")
        else:
            original = format_slip("src", "dst", force, "Task", ["data1"])
        msg = parse_slip(original)
        assert msg.wire == original

    def test_roundtrip_fallback(self) -> None:
        original = format_fallback("agent1", "agent2", "ref42abc")
        msg = parse_slip(original)
        assert msg.wire == original
        assert msg.is_fallback
        assert msg.fallback_ref == "ref42abc"

    def test_roundtrip_no_payload(self) -> None:
        original = format_slip("alpha", "beta", "Inform", "Status")
        msg = parse_slip(original)
        assert msg.wire == original

    def test_roundtrip_multi_payload(self) -> None:
        original = format_slip("a", "b", "Request", "Plan", ["x", "y", "z"])
        msg = parse_slip(original)
        assert msg.wire == original


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases for wire format handling."""

    def test_max_length_agent_ids(self) -> None:
        """Agent IDs up to 20 characters must be accepted."""
        long_id = "a" * 20
        wire = format_slip(long_id, long_id, "Request", "Task")
        msg = parse_slip(wire)
        assert msg.src == long_id
        assert msg.dst == long_id

    def test_agent_id_21_chars_rejected(self) -> None:
        with pytest.raises(WireValidationError, match="src invalid"):
            format_slip("a" * 21, "dst", "Request", "Task")

    def test_empty_payload_produces_no_trailing_space(self) -> None:
        wire = format_slip("src", "dst", "Request", "Task", [])
        assert not wire.endswith(" ")
        assert wire == "SLIP v3 src dst Request Task"

    def test_numeric_agent_ids(self) -> None:
        wire = format_slip("agent007", "agent42", "Observe", "State")
        msg = parse_slip(wire)
        assert msg.src == "agent007"
        assert msg.dst == "agent42"

    def test_single_char_agent_ids(self) -> None:
        wire = format_slip("a", "b", "Inform", "Result")
        msg = parse_slip(wire)
        assert msg.src == "a"
        assert msg.dst == "b"

    def test_payload_at_max_token_length(self) -> None:
        """A payload token at exactly 30 characters must be accepted."""
        tok = "x" * 30
        wire = format_slip("src", "dst", "Request", "Task", [tok])
        msg = parse_slip(wire)
        assert msg.payload == (tok,)

    def test_payload_at_max_count(self) -> None:
        """Exactly 20 payload tokens must be accepted."""
        tokens = [f"t{i}" for i in range(20)]
        wire = format_slip("src", "dst", "Request", "Task", tokens)
        msg = parse_slip(wire)
        assert len(msg.payload) == 20

    def test_validate_wire_on_valid_fallback(self) -> None:
        wire = format_fallback("qa", "planner", "ref1")
        issues = validate_wire(wire)
        # Fallback with "Fallback" force is in FORCE_VALUES, so it validates
        assert issues == []
