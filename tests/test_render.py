"""Tests for the render module (src/slipcore/render.py).

Covers render_human and render_log_line with SlipMessage objects,
raw wire strings, payloads, fallback refs, and explicit UCR arguments.
"""

from __future__ import annotations

import pytest

from slipcore import render_human, render_log_line, parse_slip, SlipMessage
from slipcore.ucr import create_base_ucr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_ucr():
    """Fresh base UCR instance for render tests."""
    return create_base_ucr()


@pytest.fixture
def basic_msg() -> SlipMessage:
    """A simple Request Review message."""
    return parse_slip("SLIP v3 dev reviewer Request Review")


@pytest.fixture
def payload_msg() -> SlipMessage:
    """A message with a payload."""
    return parse_slip("SLIP v3 lead architect Propose Plan redis caching")


@pytest.fixture
def fallback_msg() -> SlipMessage:
    """A Fallback message with a ref."""
    return parse_slip("SLIP v3 qa planner Fallback Generic ref7f3a")


# ---------------------------------------------------------------------------
# render_human
# ---------------------------------------------------------------------------


class TestRenderHuman:
    """Tests for render_human()."""

    def test_with_slip_message(self, basic_msg: SlipMessage) -> None:
        result = render_human(basic_msg)
        assert "[dev -> reviewer]" in result
        assert "Request Review:" in result

    def test_with_wire_string(self) -> None:
        result = render_human("SLIP v3 pm dev Meta Ack")
        assert "[pm -> dev]" in result
        assert "Meta Ack:" in result

    def test_with_payload(self, payload_msg: SlipMessage) -> None:
        result = render_human(payload_msg)
        assert "[lead -> architect]" in result
        assert "Propose Plan:" in result
        assert "(payload: redis caching)" in result

    def test_with_fallback_ref(self, fallback_msg: SlipMessage) -> None:
        result = render_human(fallback_msg)
        assert "[qa -> planner]" in result
        assert "Fallback Generic:" in result
        assert "(fallback ref: ref7f3a)" in result

    def test_with_explicit_ucr(self, basic_msg: SlipMessage, base_ucr) -> None:
        result = render_human(basic_msg, ucr=base_ucr)
        assert "[dev -> reviewer]" in result
        assert "Request Review:" in result
        # The canonical should come from the UCR lookup.
        assert '"Request work review"' in result

    def test_without_explicit_ucr(self, basic_msg: SlipMessage) -> None:
        # When ucr is None, render_human creates one internally.
        result = render_human(basic_msg, ucr=None)
        assert "[dev -> reviewer]" in result
        assert "Request Review:" in result

    def test_canonical_text_in_output(self) -> None:
        result = render_human("SLIP v3 a b Inform Complete")
        # The base UCR has canonical "Report task completion" for Inform Complete.
        assert '"Report task completion"' in result

    def test_no_payload_no_fallback(self, basic_msg: SlipMessage) -> None:
        result = render_human(basic_msg)
        # Should NOT contain payload or fallback markers.
        assert "(payload:" not in result
        assert "(fallback ref:" not in result


# ---------------------------------------------------------------------------
# render_log_line
# ---------------------------------------------------------------------------


class TestRenderLogLine:
    """Tests for render_log_line()."""

    def test_basic_format(self, basic_msg: SlipMessage) -> None:
        result = render_log_line(basic_msg)
        assert "dev->reviewer" in result
        assert "Request Review" in result
        # Payload should be "-" when empty.
        assert "payload=-" in result
        # No fallback ref.
        assert "ref=-" in result

    def test_with_wire_string(self) -> None:
        result = render_log_line("SLIP v3 pm dev Meta Ack")
        assert "pm->dev" in result
        assert "Meta Ack" in result

    def test_with_payload(self, payload_msg: SlipMessage) -> None:
        result = render_log_line(payload_msg)
        assert "payload=redis caching" in result

    def test_fallback_ref_in_output(self, fallback_msg: SlipMessage) -> None:
        result = render_log_line(fallback_msg)
        assert "ref=ref7f3a" in result

    def test_pipe_separated_structure(self, basic_msg: SlipMessage) -> None:
        result = render_log_line(basic_msg)
        parts = [p.strip() for p in result.split("|")]
        assert len(parts) == 5
        assert parts[0] == "dev->reviewer"
        assert parts[1] == "Request Review"

    def test_with_explicit_ucr(self, basic_msg: SlipMessage, base_ucr) -> None:
        result = render_log_line(basic_msg, ucr=base_ucr)
        # Canonical from UCR lookup should appear.
        assert "Request work review" in result

    def test_without_explicit_ucr(self, basic_msg: SlipMessage) -> None:
        result = render_log_line(basic_msg, ucr=None)
        # Should still work (internal UCR creation).
        assert "dev->reviewer" in result

    def test_unknown_force_obj_pair(self) -> None:
        # Build a message with a non-standard object that is not in the UCR.
        wire = "SLIP v3 a b Request ZzCustom"
        result = render_log_line(wire)
        # Should fall back to "unknown" for the canonical.
        assert "unknown" in result
