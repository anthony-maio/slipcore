"""Tests for LangGraph adapter helpers."""

from __future__ import annotations

import pytest

from slipcore import (
    LangGraphSlipstreamAdapter,
    make_decode_node,
    make_encode_node,
    make_force_object_router,
    make_force_router,
    parse_slip,
)


class TestLangGraphAdapter:
    """End-to-end tests for encode/decode and routing helpers."""

    def test_encode_thought_quantized(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        wire, result = adapter.encode_thought(
            "Please review the authentication module",
            src="planner",
            dst="reviewer",
        )
        msg = parse_slip(wire)
        assert msg.force == "Request"
        assert msg.obj == "Review"
        assert result.is_fallback is False

    def test_encode_thought_fallback(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        thought = "blorf zyxv qqqq"
        wire, result = adapter.encode_thought(thought, src="a", dst="b")
        msg = parse_slip(wire)
        assert msg.is_fallback is True
        assert msg.fallback_ref is not None
        assert result.is_fallback is True
        assert adapter.fallback_store.retrieve(msg.fallback_ref) == thought

    def test_encode_state(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        update = adapter.encode_state(
            {
                "thought": "Please review the auth module",
                "src": "planner",
                "dst": "reviewer",
                "payload": ["auth"],
            }
        )
        assert update["slip_wire"] == "SLIP v3 planner reviewer Request Review auth"
        assert update["slip_force"] == "Request"
        assert update["slip_obj"] == "Review"
        assert update["slip_is_fallback"] is False
        assert update["slip_payload"] == ("auth",)

    def test_encode_state_missing_key(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        with pytest.raises(KeyError):
            adapter.encode_state({"src": "planner", "dst": "reviewer"})

    def test_decode_state_quantized(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        update = adapter.decode_state({"slip_wire": "SLIP v3 planner reviewer Inform Status"})
        assert update["slip_force"] == "Inform"
        assert update["slip_obj"] == "Status"
        assert update["slip_payload"] == ()

    def test_decode_state_fallback_resolves_text(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        ref = adapter.fallback_store.store("inspect kubernetes events")
        wire = f"SLIP v3 ops sre Fallback Generic {ref}"
        update = adapter.decode_state({"slip_wire": wire})
        assert update["slip_force"] == "Fallback"
        assert update["slip_fallback_ref"] == ref
        assert update["slip_fallback_text"] == "inspect kubernetes events"

    def test_route_helpers(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        state = {"slip_wire": "SLIP v3 planner reviewer Request Plan"}
        assert adapter.route_by_force(state) == "Request"
        assert adapter.route_by_force_object(state) == "Request:Plan"


class TestLangGraphNodeFactories:
    """Factory helpers for LangGraph node/route callables."""

    def test_make_encode_node(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        node = make_encode_node(adapter)
        update = node(
            {"thought": "Please review auth", "src": "planner", "dst": "reviewer"}
        )
        assert update["slip_force"] == "Request"
        assert update["slip_obj"] == "Review"

    def test_make_decode_node(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        node = make_decode_node(adapter)
        update = node({"slip_wire": "SLIP v3 planner reviewer Inform Status"})
        assert update["slip_force"] == "Inform"
        assert update["slip_obj"] == "Status"

    def test_make_force_router(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        route = make_force_router(adapter)
        assert route({"slip_wire": "SLIP v3 a b Request Task"}) == "Request"

    def test_make_force_object_router(self) -> None:
        adapter = LangGraphSlipstreamAdapter()
        route = make_force_object_router(adapter)
        assert route({"slip_wire": "SLIP v3 a b Request Task"}) == "Request:Task"
