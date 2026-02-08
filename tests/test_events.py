"""Tests for the observability events module."""

from __future__ import annotations

import pytest

from slipcore.events import (
    ExtensionProposed,
    FallbackEmitted,
    FallbackStore,
    Quantized,
    WireReceived,
    WireSent,
    clear_sinks,
    emit,
    register_sink,
    unregister_sink,
)


@pytest.fixture(autouse=True)
def _clean_sinks() -> None:
    """Ensure a clean sink registry for every test."""
    clear_sinks()


# ---------------------------------------------------------------------------
# Sink registration and emission
# ---------------------------------------------------------------------------


class TestSinkRegistration:
    """Tests for register_sink, unregister_sink, and emit."""

    def test_register_sink_receives_events(self) -> None:
        """A registered sink receives emitted events."""
        received: list[object] = []
        register_sink(received.append)

        event = WireSent(src="a", dst="b", force="Request", obj="Plan")
        emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_unregister_sink_stops_delivery(self) -> None:
        """After unregistering, the sink no longer receives events."""
        received: list[object] = []
        register_sink(received.append)

        emit(WireSent(src="a", dst="b"))
        assert len(received) == 1

        unregister_sink(received.append)

        emit(WireSent(src="c", dst="d"))
        assert len(received) == 1  # no new event

    def test_unregister_unknown_sink_is_noop(self) -> None:
        """Unregistering a callback that was never registered does not raise."""
        unregister_sink(lambda e: None)  # should not raise

    def test_multiple_sinks(self) -> None:
        """Multiple registered sinks all receive the same event."""
        a: list[object] = []
        b: list[object] = []
        register_sink(a.append)
        register_sink(b.append)

        event = WireSent()
        emit(event)

        assert len(a) == 1
        assert len(b) == 1
        assert a[0] is event
        assert b[0] is event


class TestEmit:
    """Tests for the emit function behaviour."""

    def test_emit_calls_all_sinks(self) -> None:
        """emit fans out to every registered sink."""
        call_count = 0

        def counter(event: object) -> None:
            nonlocal call_count
            call_count += 1

        register_sink(counter)
        register_sink(counter)

        emit(WireSent())
        assert call_count == 2

    def test_sink_exception_is_swallowed(self) -> None:
        """A failing sink does not prevent other sinks from running."""
        received: list[object] = []

        def bad_sink(event: object) -> None:
            raise RuntimeError("boom")

        register_sink(bad_sink)
        register_sink(received.append)

        emit(WireSent())

        # The good sink still received the event
        assert len(received) == 1

    def test_emit_no_sinks(self) -> None:
        """Emitting with no sinks registered does not raise."""
        emit(WireSent())  # should not raise


class TestClearSinks:
    """Tests for clear_sinks."""

    def test_clear_removes_all_sinks(self) -> None:
        """clear_sinks removes every registered sink."""
        received: list[object] = []
        register_sink(received.append)
        register_sink(received.append)

        clear_sinks()
        emit(WireSent())

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


class TestEventDataclasses:
    """Tests for the frozen event dataclass constructors."""

    def test_wire_sent_defaults(self) -> None:
        """WireSent can be constructed with all defaults."""
        e = WireSent()
        assert e.event_type == "wire.sent"
        assert e.src == ""
        assert e.dst == ""
        assert e.force == ""
        assert e.obj == ""
        assert e.token_count == 0
        assert isinstance(e.timestamp, float)
        assert e.timestamp > 0

    def test_wire_sent_custom(self) -> None:
        """WireSent accepts custom field values."""
        e = WireSent(src="alice", dst="bob", force="Request", obj="Plan", token_count=5)
        assert e.src == "alice"
        assert e.dst == "bob"
        assert e.force == "Request"
        assert e.obj == "Plan"
        assert e.token_count == 5

    def test_wire_received_defaults(self) -> None:
        """WireReceived can be constructed with all defaults."""
        e = WireReceived()
        assert e.event_type == "wire.received"
        assert e.src == ""
        assert e.dst == ""
        assert e.force == ""
        assert e.obj == ""
        assert isinstance(e.timestamp, float)

    def test_quantized_defaults(self) -> None:
        """Quantized can be constructed with all defaults."""
        e = Quantized()
        assert e.event_type == "quantized"
        assert e.input_hash == ""
        assert e.force == ""
        assert e.obj == ""
        assert e.confidence == 0.0
        assert e.mode == ""
        assert isinstance(e.timestamp, float)

    def test_fallback_emitted_defaults(self) -> None:
        """FallbackEmitted can be constructed with all defaults."""
        e = FallbackEmitted()
        assert e.event_type == "fallback.emitted"
        assert e.ref == ""
        assert e.reason == ""
        assert e.confidence == 0.0
        assert e.raw_text_hash == ""
        assert isinstance(e.timestamp, float)

    def test_extension_proposed_defaults(self) -> None:
        """ExtensionProposed can be constructed with all defaults."""
        e = ExtensionProposed()
        assert e.event_type == "extension.proposed"
        assert e.force == ""
        assert e.obj == ""
        assert e.created_by == ""
        assert isinstance(e.timestamp, float)

    def test_extension_proposed_custom(self) -> None:
        """ExtensionProposed accepts custom values."""
        e = ExtensionProposed(force="Deploy", obj="Container", created_by="ci")
        assert e.force == "Deploy"
        assert e.obj == "Container"
        assert e.created_by == "ci"

    def test_events_are_frozen(self) -> None:
        """Event dataclasses are frozen (immutable)."""
        e = WireSent()
        with pytest.raises(AttributeError):
            e.src = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FallbackStore
# ---------------------------------------------------------------------------


class TestFallbackStore:
    """Tests for the FallbackStore key-value store."""

    def test_store_returns_ref_starting_with_ref(self) -> None:
        """store() returns a reference string beginning with 'ref'."""
        store = FallbackStore()
        ref = store.store("some fallback text")
        assert ref.startswith("ref")

    def test_store_and_retrieve(self) -> None:
        """Stored text is retrievable by its ref."""
        store = FallbackStore()
        text = "This thought could not be quantized"
        ref = store.store(text)
        assert store.retrieve(ref) == text

    def test_retrieve_unknown_ref_returns_none(self) -> None:
        """Retrieving a ref that was never stored returns None."""
        store = FallbackStore()
        assert store.retrieve("ref_nonexistent") is None

    def test_multiple_stores(self) -> None:
        """Multiple stores produce unique refs with correct retrieval."""
        store = FallbackStore()
        ref1 = store.store("text one")
        ref2 = store.store("text two")
        assert ref1 != ref2
        assert store.retrieve(ref1) == "text one"
        assert store.retrieve(ref2) == "text two"

    def test_len(self) -> None:
        """__len__ reflects the number of stored entries."""
        store = FallbackStore()
        assert len(store) == 0
        store.store("a")
        assert len(store) == 1
        store.store("b")
        assert len(store) == 2

    def test_clear(self) -> None:
        """clear() removes all stored entries."""
        store = FallbackStore()
        ref = store.store("to be cleared")
        assert len(store) == 1

        store.clear()

        assert len(store) == 0
        assert store.retrieve(ref) is None

    def test_store_empty_string(self) -> None:
        """Storing an empty string works correctly."""
        store = FallbackStore()
        ref = store.store("")
        assert store.retrieve(ref) == ""
        assert len(store) == 1
