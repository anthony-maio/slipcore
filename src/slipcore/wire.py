"""SLIP v3 wire format: parse, format, validate.

Wire grammar (token-friendly, BPE-safe):
    slip-message = "SLIP" SP "v3" SP src SP dst SP force SP object *(SP payload)
    src/dst = 1*20(ALPHA / DIGIT)
    force = ForceToken value
    object = ObjectToken mnemonic
    payload = 1*30(ALPHA / DIGIT)

Fallback variant:
    "SLIP" SP "v3" SP src SP dst SP "Fallback" SP "Generic" SP ref
    ref = 1*16(ALPHA / DIGIT)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .errors import WireParseError, WireValidationError
from .intent import FORCE_VALUES

WIRE_VERSION = "v3"
WIRE_PREFIX = "SLIP"
MAX_AGENT_ID_LEN = 20
MAX_PAYLOAD_TOKEN_LEN = 30
MAX_PAYLOAD_TOKENS = 20
MAX_FALLBACK_REF_LEN = 16

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+$")
_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9]{1,20}$")


@dataclass(frozen=True, slots=True)
class SlipMessage:
    """A parsed and validated SLIP v3 wire message."""

    version: str
    src: str
    dst: str
    force: str
    obj: str
    payload: tuple[str, ...] = field(default_factory=tuple)
    fallback_ref: Optional[str] = None

    @property
    def is_fallback(self) -> bool:
        return self.force == "Fallback"

    @property
    def wire(self) -> str:
        parts = [WIRE_PREFIX, self.version, self.src, self.dst, self.force, self.obj]
        if self.is_fallback and self.fallback_ref:
            parts.append(self.fallback_ref)
        if self.payload:
            parts.extend(self.payload)
        return " ".join(parts)

    @property
    def token_count_estimate(self) -> int:
        return len(self.wire.split())


def format_slip(
    src: str,
    dst: str,
    force: str,
    obj: str,
    payload: Optional[list[str] | tuple[str, ...]] = None,
) -> str:
    """Format a SLIP v3 wire message.

    Example:
        >>> format_slip("planner", "exec", "Request", "Plan", ["auth"])
        'SLIP v3 planner exec Request Plan auth'
    """
    _validate_agent_id(src, "src")
    _validate_agent_id(dst, "dst")
    _validate_force(force)
    _validate_object(obj)

    if force == "Fallback":
        if not payload:
            raise WireValidationError("Fallback messages require a ref token")
        _validate_fallback_ref(payload[0])

    parts = [WIRE_PREFIX, WIRE_VERSION, src, dst, force, obj]

    if payload:
        for i, token in enumerate(payload):
            _validate_payload_token(token, i)
        if len(payload) > MAX_PAYLOAD_TOKENS:
            raise WireValidationError(f"Payload too long: {len(payload)} > {MAX_PAYLOAD_TOKENS}")
        parts.extend(payload)

    return " ".join(parts)


def format_fallback(src: str, dst: str, ref: str) -> str:
    """Format a Fallback wire message with pointer reference.

    Example:
        >>> format_fallback("qa", "planner", "ref7f3a")
        'SLIP v3 qa planner Fallback Generic ref7f3a'
    """
    _validate_agent_id(src, "src")
    _validate_agent_id(dst, "dst")
    _validate_fallback_ref(ref)
    return " ".join([WIRE_PREFIX, WIRE_VERSION, src, dst, "Fallback", "Generic", ref])


def parse_slip(raw: str) -> SlipMessage:
    """Parse a SLIP v3 wire message.

    Example:
        >>> msg = parse_slip("SLIP v3 planner exec Request Plan timeline")
        >>> msg.force
        'Request'
    """
    raw = raw.strip()
    tokens = raw.split()

    if len(tokens) < 6:
        raise WireParseError(f"Too short: need >= 6 tokens, got {len(tokens)}", raw=raw)

    prefix, version, src, dst, force, obj = tokens[:6]
    payload = tokens[6:]

    if prefix != WIRE_PREFIX:
        raise WireParseError(f"Bad prefix: {prefix!r}", raw=raw)
    if version != WIRE_VERSION:
        raise WireParseError(f"Bad version: {version!r}", raw=raw)

    _validate_agent_id(src, "src")
    _validate_agent_id(dst, "dst")
    _validate_force(force)
    _validate_object(obj)

    for i, tok in enumerate(payload):
        _validate_payload_token(tok, i)
    if len(payload) > MAX_PAYLOAD_TOKENS:
        raise WireValidationError(
            f"Too many payload tokens: {len(payload)} > {MAX_PAYLOAD_TOKENS}"
        )

    fallback_ref: Optional[str] = None
    if force == "Fallback":
        if not payload:
            raise WireValidationError("Fallback messages require a ref token")
        fallback_ref = payload[0]
        _validate_fallback_ref(fallback_ref)
        payload = list(payload[1:])
    else:
        payload = list(payload)

    return SlipMessage(
        version=version,
        src=src,
        dst=dst,
        force=force,
        obj=obj,
        payload=tuple(payload),
        fallback_ref=fallback_ref,
    )


def validate_wire(raw: str) -> list[str]:
    """Validate wire message, return list of issues (empty = valid)."""
    issues: list[str] = []
    tokens = raw.strip().split()

    if len(tokens) < 6:
        issues.append(f"Too few tokens: {len(tokens)} < 6")
        return issues

    prefix, version, src, dst, force, obj = tokens[:6]
    payload_tokens = tokens[6:]

    if prefix != WIRE_PREFIX:
        issues.append(f"Bad prefix: {prefix!r}")
    if version != WIRE_VERSION:
        issues.append(f"Bad version: {version!r}")
    if not _AGENT_ID_RE.match(src):
        issues.append(f"Bad src: {src!r}")
    if not _AGENT_ID_RE.match(dst):
        issues.append(f"Bad dst: {dst!r}")
    if force not in FORCE_VALUES:
        issues.append(f"Unknown force: {force!r}")
    if not _SAFE_TOKEN_RE.match(obj):
        issues.append(f"Bad obj: {obj!r}")
    elif len(obj) > MAX_PAYLOAD_TOKEN_LEN:
        issues.append(f"Obj too long: {len(obj)}")

    for i, tok in enumerate(payload_tokens):
        if not _SAFE_TOKEN_RE.match(tok):
            issues.append(f"Bad payload[{i}]: {tok!r}")
        elif len(tok) > MAX_PAYLOAD_TOKEN_LEN:
            issues.append(f"Payload[{i}] too long: {len(tok)}")

    if len(payload_tokens) > MAX_PAYLOAD_TOKENS:
        issues.append(f"Too many payload tokens: {len(payload_tokens)} > {MAX_PAYLOAD_TOKENS}")

    if force == "Fallback":
        if not payload_tokens:
            issues.append("Fallback requires a ref token")
        else:
            ref = payload_tokens[0]
            if len(ref) > MAX_FALLBACK_REF_LEN:
                issues.append(f"Fallback ref too long: {len(ref)} > {MAX_FALLBACK_REF_LEN}")

    return issues


def parse_slip_legacy(raw: str, default_fallback_ref: str = "reflegacy") -> SlipMessage:
    """Parse SLIP wire and repair known legacy fallback wires.

    Legacy v3 messages that omit the fallback ref can be parsed by
    supplying a deterministic default ref token.
    """
    try:
        return parse_slip(raw)
    except WireValidationError:
        tokens = raw.strip().split()
        is_legacy_fallback = (
            len(tokens) == 6
            and tokens[:2] == [WIRE_PREFIX, WIRE_VERSION]
            and tokens[4] == "Fallback"
        )
        if is_legacy_fallback:
            _validate_fallback_ref(default_fallback_ref)
            repaired = " ".join(tokens + [default_fallback_ref])
            return parse_slip(repaired)
        raise


def _validate_agent_id(agent_id: str, field_name: str) -> None:
    if not _AGENT_ID_RE.match(agent_id):
        raise WireValidationError(f"{field_name} invalid: {agent_id!r}")


def _validate_force(force: str) -> None:
    if force not in FORCE_VALUES:
        raise WireValidationError(f"Unknown force: {force!r}")


def _validate_object(obj: str) -> None:
    if not _SAFE_TOKEN_RE.match(obj) or len(obj) > MAX_PAYLOAD_TOKEN_LEN:
        raise WireValidationError(f"Object invalid: {obj!r}")


def _validate_payload_token(token: str, index: int) -> None:
    if not _SAFE_TOKEN_RE.match(token):
        raise WireValidationError(f"Payload[{index}] not alphanumeric: {token!r}")
    if len(token) > MAX_PAYLOAD_TOKEN_LEN:
        raise WireValidationError(f"Payload[{index}] too long: {len(token)}")


def _validate_fallback_ref(ref: str) -> None:
    if not _SAFE_TOKEN_RE.match(ref) or len(ref) > MAX_FALLBACK_REF_LEN:
        raise WireValidationError(f"Invalid fallback ref: {ref!r}")
