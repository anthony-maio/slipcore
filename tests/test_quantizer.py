"""Tests for the keyword quantizer (src/slipcore/quantizer.py).

Covers KeywordQuantizer, QuantizeResult, the module-level quantize()
and think_quantize_transmit() convenience functions, FallbackStore
integration, and usage-stats tracking.
"""

from __future__ import annotations

import pytest

from slipcore.quantizer import (
    KeywordQuantizer,
    QuantizeResult,
    quantize,
    think_quantize_transmit,
)
from slipcore.events import FallbackStore


# ---------------------------------------------------------------------------
# KeywordQuantizer.quantize
# ---------------------------------------------------------------------------


class TestKeywordQuantizerQuantize:
    """Two-stage keyword classification tests."""

    def test_review_request(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("please review the auth code")
        assert result.force == "Request"
        assert result.obj == "Review"
        assert not result.is_fallback

    def test_inform_completion(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("I finished the task")
        assert result.force == "Inform"
        assert not result.is_fallback

    def test_unknown_text_fallback(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("xylophone quantum banana")
        assert result.is_fallback
        assert result.force == "Fallback"
        assert result.obj == "Generic"
        assert result.confidence == 0.0

    def test_ask_clarification(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("what do you mean by that? please clarify")
        assert result.force == "Ask"
        assert result.obj == "Clarify"

    def test_propose_alternative(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("I suggest we try an alternative approach instead")
        assert result.force == "Propose"
        assert result.obj == "Alternative"

    def test_meta_ack(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("acknowledged, got it")
        assert result.force == "Meta"
        assert result.obj == "Ack"

    def test_error_timeout(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("the operation timed out, timeout exceeded")
        assert result.force == "Error"

    def test_observe_state(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("I noticed the current state is degraded")
        assert result.force == "Observe"


# ---------------------------------------------------------------------------
# QuantizeResult structure
# ---------------------------------------------------------------------------


class TestQuantizeResult:
    """Verify QuantizeResult fields are populated correctly."""

    def test_fields_present(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("please review")
        assert isinstance(result, QuantizeResult)
        assert isinstance(result.force, str)
        assert isinstance(result.obj, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.method, str)
        assert isinstance(result.is_fallback, bool)

    def test_keyword_method(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("please help me with this")
        assert result.method == "keyword"

    def test_fallback_method(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("xylophone quantum banana")
        assert result.method == "fallback"

    def test_is_fallback_for_low_confidence(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("xylophone quantum banana")
        assert result.is_fallback is True
        assert result.confidence == 0.0

    def test_is_not_fallback_for_good_match(self) -> None:
        kq = KeywordQuantizer()
        result = kq.quantize("please execute the task immediately")
        assert result.is_fallback is False
        assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Module-level quantize() function
# ---------------------------------------------------------------------------


class TestQuantizeFunction:
    """Test the convenience quantize() that returns a wire string."""

    def test_returns_slip_v3_wire(self) -> None:
        wire = quantize("please review the code", src="dev", dst="reviewer")
        assert wire.startswith("SLIP v3 ")
        assert "dev" in wire
        assert "reviewer" in wire

    def test_known_input_produces_force_obj(self) -> None:
        wire = quantize("please review the auth code", src="dev", dst="qa")
        tokens = wire.split()
        assert tokens[0] == "SLIP"
        assert tokens[1] == "v3"
        assert tokens[2] == "dev"
        assert tokens[3] == "qa"
        # Force and Object are in tokens[4] and tokens[5].
        assert tokens[4] == "Request"
        assert tokens[5] == "Review"

    def test_fallback_produces_fallback_wire(self) -> None:
        wire = quantize("xylophone quantum banana", src="a", dst="b")
        assert "Fallback" in wire
        assert "Generic" in wire
        # The wire should include a ref token.
        tokens = wire.split()
        assert len(tokens) == 7  # SLIP v3 a b Fallback Generic <ref>
        assert tokens[6].startswith("ref")


# ---------------------------------------------------------------------------
# think_quantize_transmit()
# ---------------------------------------------------------------------------


class TestThinkQuantizeTransmit:
    """Test the full TQT pipeline."""

    def test_returns_wire_format(self) -> None:
        wire = think_quantize_transmit(
            "Please review the auth code",
            src="dev",
            dst="reviewer",
        )
        assert wire.startswith("SLIP v3 ")
        assert "dev" in wire
        assert "reviewer" in wire

    def test_known_input(self) -> None:
        wire = think_quantize_transmit(
            "Please review the auth code",
            src="dev",
            dst="reviewer",
        )
        assert "Request" in wire
        assert "Review" in wire

    def test_fallback_path(self) -> None:
        wire = think_quantize_transmit(
            "xylophone quantum banana",
            src="x",
            dst="y",
        )
        assert "Fallback" in wire


# ---------------------------------------------------------------------------
# FallbackStore integration
# ---------------------------------------------------------------------------


class TestFallbackStoreIntegration:
    """Verify that the quantizer stores fallback text and produces refs."""

    def test_fallback_stores_text(self) -> None:
        store = FallbackStore()
        ref = store.store("some unquantizable thought")
        assert store.retrieve(ref) == "some unquantizable thought"

    def test_ref_is_alphanumeric(self) -> None:
        store = FallbackStore()
        ref = store.store("anything")
        assert ref.isalnum(), f"ref {ref!r} is not alphanumeric"

    def test_ref_starts_with_ref(self) -> None:
        store = FallbackStore()
        ref = store.store("anything")
        assert ref.startswith("ref")

    def test_store_length_increases(self) -> None:
        store = FallbackStore()
        assert len(store) == 0
        store.store("first")
        assert len(store) == 1
        store.store("second")
        assert len(store) == 2

    def test_quantizer_fallback_store(self) -> None:
        kq = KeywordQuantizer()
        # Trigger a fallback.
        kq.quantize("xylophone quantum banana")
        # The quantizer's fallback store is separate from module-level quantize(),
        # but it should still exist.
        assert isinstance(kq.fallback_store, FallbackStore)

    def test_clear(self) -> None:
        store = FallbackStore()
        store.store("a")
        store.store("b")
        assert len(store) == 2
        store.clear()
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Usage stats tracking
# ---------------------------------------------------------------------------


class TestUsageStats:
    """Verify that KeywordQuantizer tracks usage statistics."""

    def test_empty_stats_initially(self) -> None:
        kq = KeywordQuantizer()
        assert kq.get_usage_stats() == {}

    def test_stats_after_quantize(self) -> None:
        kq = KeywordQuantizer()
        kq.quantize("please review the code")
        stats = kq.get_usage_stats()
        assert len(stats) > 0
        # At least one key should have a count of 1.
        assert any(v == 1 for v in stats.values())

    def test_fallback_stats(self) -> None:
        kq = KeywordQuantizer()
        kq.quantize("xylophone quantum banana")
        stats = kq.get_usage_stats()
        assert stats.get("_fallback", 0) == 1

    def test_fallback_rate_zero_initially(self) -> None:
        kq = KeywordQuantizer()
        assert kq.get_fallback_rate() == 0.0

    def test_fallback_rate_after_mixed(self) -> None:
        kq = KeywordQuantizer()
        kq.quantize("please review this")
        kq.quantize("xylophone quantum banana")
        rate = kq.get_fallback_rate()
        assert 0.0 < rate < 1.0
        assert rate == pytest.approx(0.5)

    def test_stats_accumulate(self) -> None:
        kq = KeywordQuantizer()
        kq.quantize("please review the code")
        kq.quantize("please review the tests")
        stats = kq.get_usage_stats()
        total = sum(stats.values())
        assert total == 2
