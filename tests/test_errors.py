"""Tests for slipcore.errors exception hierarchy."""

from __future__ import annotations

import pytest

from slipcore.errors import (
    AnchorNotFoundError,
    MissingExtraError,
    PolicyViolationError,
    SlipError,
    UCRError,
    UCRVersionMismatchError,
    WireParseError,
    WireValidationError,
)


class TestHierarchy:
    """All errors should inherit from SlipError."""

    @pytest.mark.parametrize("cls", [
        WireParseError,
        WireValidationError,
        UCRError,
        AnchorNotFoundError,
        UCRVersionMismatchError,
        PolicyViolationError,
        MissingExtraError,
    ])
    def test_inherits_from_slip_error(self, cls):
        assert issubclass(cls, SlipError)

    def test_ucr_errors_inherit_from_ucr_error(self):
        assert issubclass(AnchorNotFoundError, UCRError)
        assert issubclass(UCRVersionMismatchError, UCRError)


class TestWireParseError:
    def test_stores_raw_and_position(self):
        err = WireParseError("bad token", raw="SLIP v2 x", position=5)
        assert err.raw == "SLIP v2 x"
        assert err.position == 5
        assert "bad token" in str(err)

    def test_defaults(self):
        err = WireParseError("oops")
        assert err.raw == ""
        assert err.position == -1


class TestAnchorNotFoundError:
    def test_stores_string_key(self):
        err = AnchorNotFoundError("RequestReview")
        assert err.key == "RequestReview"
        assert "RequestReview" in str(err)

    def test_stores_int_key(self):
        err = AnchorNotFoundError(0x1234)
        assert err.key == 0x1234
        assert "4660" in str(err) or "0x1234" in str(err).lower()


class TestMissingExtraError:
    def test_formats_install_hint(self):
        err = MissingExtraError("sentence-transformers", "ml")
        msg = str(err)
        assert "sentence-transformers" in msg
        assert "pip install slipcore[ml]" in msg

    def test_is_catchable_as_slip_error(self):
        with pytest.raises(SlipError):
            raise MissingExtraError("numpy", "ml")


class TestSimpleErrors:
    def test_wire_validation_error(self):
        err = WireValidationError("invalid force token")
        assert "invalid force token" in str(err)

    def test_ucr_version_mismatch(self):
        err = UCRVersionMismatchError("hash mismatch")
        assert "hash mismatch" in str(err)

    def test_policy_violation(self):
        err = PolicyViolationError("Reject+Plan not allowed")
        assert "Reject+Plan" in str(err)
