"""
Slipstream Protocol v2 - Semantic Quantization

Token-aligned wire format for efficient multi-agent communication.
Avoids special characters that fragment in BPE tokenizers.

Wire format:
    SLIP <version> <src> <dst> <anchor> [payload...]

Examples:
    SLIP v1 alice bob RequestReview
    SLIP v1 planner coordinator ProposePlan auth refactor
    SLIP v1 critic executor EvalNeedsWork security validation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from .ucr import UCR, UCRAnchor, get_default_ucr


# Protocol marker - single token in most BPE vocabularies
PROTOCOL_MARKER = "SLIP"
DEFAULT_VERSION = "v1"


@dataclass
class SlipMessage:
    """
    A Slipstream message.

    Attributes:
        src: Source agent identifier (string, should be single BPE token)
        dst: Destination agent identifier
        anchor: The UCR anchor (semantic intent)
        payload: Optional payload tokens (unquantizable content)
        version: Protocol version
        thread_id: Optional conversation thread identifier
    """
    src: str
    dst: str
    anchor: UCRAnchor
    payload: list[str] = field(default_factory=list)
    version: str = DEFAULT_VERSION
    thread_id: Optional[str] = None

    def __repr__(self) -> str:
        payload_str = f" payload={self.payload}" if self.payload else ""
        thread_str = f" thread={self.thread_id}" if self.thread_id else ""
        return (
            f"<SlipMessage {self.src}->{self.dst} "
            f"{self.anchor.mnemonic}{payload_str}{thread_str}>"
        )


def encode(msg: SlipMessage) -> str:
    """
    Encode a SlipMessage to wire format.

    Returns a space-separated string optimized for BPE tokenization.
    No special characters - just words and spaces.

    Examples:
        "SLIP v1 alice bob RequestReview"
        "SLIP v1 planner coord ProposePlan auth_refactor high_priority"
    """
    parts = [PROTOCOL_MARKER, msg.version, msg.src, msg.dst, msg.anchor.mnemonic]

    # Add thread_id if present (prefixed with 'thread' for clarity)
    if msg.thread_id:
        parts.append(f"thread{msg.thread_id}")

    # Add payload tokens
    parts.extend(msg.payload)

    return " ".join(parts)


def decode(wire: str, ucr: Optional[UCR] = None) -> SlipMessage:
    """
    Decode a wire-format string to SlipMessage.

    Args:
        wire: The wire-format string
        ucr: UCR instance for anchor lookup (uses default if not provided)

    Raises:
        ValueError: If the message format is invalid or anchor unknown

    Examples:
        decode("SLIP v1 alice bob RequestReview")
        decode("SLIP v1 planner coord ProposePlan auth_refactor")
    """
    if ucr is None:
        ucr = get_default_ucr()

    tokens = wire.strip().split()

    if len(tokens) < 5:
        raise ValueError(
            f"Invalid SLIP message: need at least 5 tokens "
            f"(SLIP version src dst anchor), got {len(tokens)}"
        )

    marker, version, src, dst, mnemonic = tokens[:5]
    rest = tokens[5:]

    if marker != PROTOCOL_MARKER:
        raise ValueError(f"Invalid protocol marker: expected '{PROTOCOL_MARKER}', got '{marker}'")

    # Look up anchor
    anchor = ucr.get_by_mnemonic(mnemonic)
    if anchor is None:
        raise ValueError(f"Unknown anchor mnemonic: '{mnemonic}'")

    # Parse remaining tokens for thread_id and payload
    thread_id = None
    payload = []

    for token in rest:
        if token.startswith("thread") and thread_id is None:
            thread_id = token[6:]  # Strip "thread" prefix
        else:
            payload.append(token)

    return SlipMessage(
        src=src,
        dst=dst,
        anchor=anchor,
        payload=payload,
        version=version,
        thread_id=thread_id,
    )


def is_valid_agent_name(name: str) -> bool:
    """
    Check if a name is suitable for use as an agent identifier.
    Should be a single BPE token (alphanumeric, no special chars).
    """
    return name.isalnum() and len(name) > 0


# ============ Convenience Functions ============

def slip(
    src: str,
    dst: str,
    mnemonic: str,
    payload: Optional[list[str]] = None,
    thread_id: Optional[str] = None,
    ucr: Optional[UCR] = None,
) -> str:
    """
    Quick helper to create and encode a SLIP message.

    Args:
        src: Source agent
        dst: Destination agent
        mnemonic: UCR anchor mnemonic (e.g., "RequestPlanReview")
        payload: Optional payload tokens
        thread_id: Optional thread identifier
        ucr: UCR instance (uses default if not provided)

    Returns:
        Wire-format string ready to transmit

    Example:
        >>> slip("alice", "bob", "RequestReview", ["auth_module"])
        "SLIP v1 alice bob RequestReview auth_module"
    """
    if ucr is None:
        ucr = get_default_ucr()

    anchor = ucr.get_by_mnemonic(mnemonic)
    if anchor is None:
        raise ValueError(f"Unknown anchor: '{mnemonic}'")

    msg = SlipMessage(
        src=src,
        dst=dst,
        anchor=anchor,
        payload=payload or [],
        thread_id=thread_id,
    )
    return encode(msg)


def fallback(src: str, dst: str, natural_language: str, ucr: Optional[UCR] = None) -> str:
    """
    Create a fallback message for unquantizable content.

    When an agent's intent doesn't match any UCR anchor well,
    use this to send natural language with a Fallback marker.

    Example:
        >>> fallback("alice", "bob", "check the auth logs for timing anomalies")
        "SLIP v1 alice bob Fallback check the auth logs for timing anomalies"
    """
    if ucr is None:
        ucr = get_default_ucr()

    anchor = ucr.get_by_mnemonic("Fallback")
    if anchor is None:
        raise ValueError("UCR missing required 'Fallback' anchor")

    # Split natural language into tokens for payload
    payload = natural_language.split()

    msg = SlipMessage(src=src, dst=dst, anchor=anchor, payload=payload)
    return encode(msg)


# ============ Message Builders ============

class MessageBuilder:
    """
    Fluent builder for constructing SLIP messages.

    Example:
        msg = (MessageBuilder()
            .from_agent("planner")
            .to_agent("executor")
            .intent("RequestTask")
            .with_payload("implement", "auth", "module")
            .in_thread("task42")
            .build())
    """

    def __init__(self, ucr: Optional[UCR] = None):
        self._ucr = ucr or get_default_ucr()
        self._src: Optional[str] = None
        self._dst: Optional[str] = None
        self._mnemonic: Optional[str] = None
        self._payload: list[str] = []
        self._thread_id: Optional[str] = None

    def from_agent(self, src: str) -> "MessageBuilder":
        self._src = src
        return self

    def to_agent(self, dst: str) -> "MessageBuilder":
        self._dst = dst
        return self

    def intent(self, mnemonic: str) -> "MessageBuilder":
        self._mnemonic = mnemonic
        return self

    def with_payload(self, *tokens: str) -> "MessageBuilder":
        self._payload.extend(tokens)
        return self

    def in_thread(self, thread_id: str) -> "MessageBuilder":
        self._thread_id = thread_id
        return self

    def build(self) -> SlipMessage:
        if not all([self._src, self._dst, self._mnemonic]):
            raise ValueError("Must specify src, dst, and intent")

        anchor = self._ucr.get_by_mnemonic(self._mnemonic)
        if anchor is None:
            raise ValueError(f"Unknown anchor: '{self._mnemonic}'")

        return SlipMessage(
            src=self._src,
            dst=self._dst,
            anchor=anchor,
            payload=self._payload,
            thread_id=self._thread_id,
        )

    def encode(self) -> str:
        return encode(self.build())


# ============ Smoke Test ============

if __name__ == "__main__":
    # Basic encode/decode roundtrip
    print("=== Slipstream Protocol v2 ===\n")

    # Simple message
    wire1 = slip("alice", "bob", "RequestReview")
    print(f"Wire: {wire1}")
    msg1 = decode(wire1)
    print(f"Decoded: {msg1}")
    print(f"Canonical: {msg1.anchor.canonical}\n")

    # Message with payload
    wire2 = slip("planner", "executor", "RequestTask", ["auth", "refactor"])
    print(f"Wire: {wire2}")
    msg2 = decode(wire2)
    print(f"Decoded: {msg2}\n")

    # Fallback for unquantizable content
    wire3 = fallback("critic", "team", "check for SQL injection in the login handler")
    print(f"Fallback wire: {wire3}")
    msg3 = decode(wire3)
    print(f"Decoded: {msg3}")
    print(f"Payload: {' '.join(msg3.payload)}\n")

    # Token count comparison
    json_equiv = '{"from": "alice", "to": "bob", "type": "request", "action": "plan_review"}'
    print(f"JSON equivalent ({len(json_equiv)} chars): {json_equiv}")
    print(f"SLIP wire ({len(wire1)} chars): {wire1}")
    print(f"\nApproximate token savings: ~70-80%")
