"""LangGraph adapter helpers for Slipstream transport.

This module is dependency-free by design: it does not import LangGraph.
Use these helpers inside LangGraph node functions and routers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .events import FallbackStore
from .quantizer import KeywordQuantizer, QuantizeResult
from .wire import SlipMessage, format_fallback, format_slip, parse_slip

StateLike = Mapping[str, Any]
NodeUpdate = dict[str, Any]
NodeFn = Callable[[dict[str, Any]], NodeUpdate]
RouteFn = Callable[[dict[str, Any]], str]


def _require_str(state: StateLike, key: str) -> str:
    if key not in state:
        raise KeyError(f"Missing required state key: {key!r}")
    value = state[key]
    if not isinstance(value, str):
        raise TypeError(f"State key {key!r} must be str, got {type(value).__name__}")
    return value


def _optional_payload(state: StateLike, key: str) -> list[str] | None:
    if key not in state:
        return None

    value = state[key]
    if value is None:
        return None

    if isinstance(value, tuple):
        payload = list(value)
    elif isinstance(value, list):
        payload = value
    else:
        raise TypeError(
            "State key "
            f"{key!r} must be list[str] | tuple[str, ...] | None, "
            f"got {type(value).__name__}"
        )

    for idx, token in enumerate(payload):
        if not isinstance(token, str):
            raise TypeError(f"State key {key!r}[{idx}] must be str")

    return list(payload)


@dataclass(frozen=True, slots=True)
class DecodedSlip:
    """Parsed Slipstream wire with optional fallback text resolution."""

    wire: str
    message: SlipMessage
    fallback_text: str | None = None


class LangGraphSlipstreamAdapter:
    """Bridge LangGraph state dictionaries to Slipstream wire transport.

    This adapter is intended for orchestrator-level integration:
    - encode natural-language node output into `SLIP v3 ...` wire text
    - decode incoming wire into parsed fields for routing
    - resolve fallback refs through a shared store
    """

    def __init__(
        self,
        quantizer: KeywordQuantizer | None = None,
        fallback_store: FallbackStore | None = None,
    ) -> None:
        self.quantizer = quantizer or KeywordQuantizer()
        if fallback_store is None:
            self.fallback_store = self.quantizer.fallback_store
        else:
            self.fallback_store = fallback_store
            self.quantizer.fallback_store = fallback_store

    def encode_thought(
        self,
        thought: str,
        src: str,
        dst: str,
        payload: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[str, QuantizeResult]:
        """Quantize natural language into Slipstream wire text."""
        result = self.quantizer.quantize(thought)
        if result.is_fallback:
            ref = self.fallback_store.store(thought)
            return format_fallback(src, dst, ref), result
        return format_slip(src, dst, result.force, result.obj, payload), result

    def encode_state(
        self,
        state: StateLike,
        *,
        thought_key: str = "thought",
        src_key: str = "src",
        dst_key: str = "dst",
        payload_key: str = "payload",
        wire_key: str = "slip_wire",
        force_key: str = "slip_force",
        object_key: str = "slip_obj",
        confidence_key: str = "slip_confidence",
        fallback_flag_key: str = "slip_is_fallback",
        fallback_ref_key: str = "slip_fallback_ref",
    ) -> NodeUpdate:
        """Read LangGraph state keys and emit Slipstream transport fields."""
        thought = _require_str(state, thought_key)
        src = _require_str(state, src_key)
        dst = _require_str(state, dst_key)
        payload = _optional_payload(state, payload_key)

        wire, result = self.encode_thought(thought, src, dst, payload)
        parsed = parse_slip(wire)

        update: NodeUpdate = {
            wire_key: wire,
            force_key: parsed.force,
            object_key: parsed.obj,
            confidence_key: result.confidence,
            fallback_flag_key: parsed.is_fallback,
        }
        if parsed.fallback_ref is not None:
            update[fallback_ref_key] = parsed.fallback_ref
        if parsed.payload:
            update["slip_payload"] = parsed.payload

        return update

    def decode_wire(self, wire: str) -> DecodedSlip:
        """Parse wire and resolve fallback text when present."""
        message = parse_slip(wire)
        if message.fallback_ref is None:
            return DecodedSlip(wire=wire, message=message, fallback_text=None)

        return DecodedSlip(
            wire=wire,
            message=message,
            fallback_text=self.fallback_store.retrieve(message.fallback_ref),
        )

    def decode_state(
        self,
        state: StateLike,
        *,
        wire_key: str = "slip_wire",
        message_key: str = "slip_message",
        force_key: str = "slip_force",
        object_key: str = "slip_obj",
        payload_key: str = "slip_payload",
        fallback_ref_key: str = "slip_fallback_ref",
        fallback_text_key: str = "slip_fallback_text",
    ) -> NodeUpdate:
        """Decode `slip_wire` from state into structured routing fields."""
        wire = _require_str(state, wire_key)
        decoded = self.decode_wire(wire)

        update: NodeUpdate = {
            message_key: decoded.message,
            force_key: decoded.message.force,
            object_key: decoded.message.obj,
            payload_key: decoded.message.payload,
        }
        if decoded.message.fallback_ref is not None:
            update[fallback_ref_key] = decoded.message.fallback_ref
        if decoded.fallback_text is not None:
            update[fallback_text_key] = decoded.fallback_text
        return update

    def route_by_force(self, state: StateLike, *, wire_key: str = "slip_wire") -> str:
        """Return `Force` for conditional edges."""
        wire = _require_str(state, wire_key)
        return parse_slip(wire).force

    def route_by_force_object(self, state: StateLike, *, wire_key: str = "slip_wire") -> str:
        """Return `Force:Object` for fine-grained conditional edges."""
        wire = _require_str(state, wire_key)
        msg = parse_slip(wire)
        return f"{msg.force}:{msg.obj}"


def make_encode_node(
    adapter: LangGraphSlipstreamAdapter,
    *,
    thought_key: str = "thought",
    src_key: str = "src",
    dst_key: str = "dst",
    payload_key: str = "payload",
    wire_key: str = "slip_wire",
) -> NodeFn:
    """Build a LangGraph node function that writes Slipstream wire to state."""

    def _node(state: dict[str, Any]) -> NodeUpdate:
        return adapter.encode_state(
            state,
            thought_key=thought_key,
            src_key=src_key,
            dst_key=dst_key,
            payload_key=payload_key,
            wire_key=wire_key,
        )

    return _node


def make_decode_node(
    adapter: LangGraphSlipstreamAdapter,
    *,
    wire_key: str = "slip_wire",
) -> NodeFn:
    """Build a LangGraph node function that decodes Slipstream wire from state."""

    def _node(state: dict[str, Any]) -> NodeUpdate:
        return adapter.decode_state(state, wire_key=wire_key)

    return _node


def make_force_router(
    adapter: LangGraphSlipstreamAdapter,
    *,
    wire_key: str = "slip_wire",
) -> RouteFn:
    """Build a router returning Force token for LangGraph conditional edges."""

    def _route(state: dict[str, Any]) -> str:
        return adapter.route_by_force(state, wire_key=wire_key)

    return _route


def make_force_object_router(
    adapter: LangGraphSlipstreamAdapter,
    *,
    wire_key: str = "slip_wire",
) -> RouteFn:
    """Build a router returning `Force:Object` for conditional edges."""

    def _route(state: dict[str, Any]) -> str:
        return adapter.route_by_force_object(state, wire_key=wire_key)

    return _route
