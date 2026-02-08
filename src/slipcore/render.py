"""Human adapter: deterministic renderer for SLIP wire messages."""

from __future__ import annotations

from typing import Optional

from .ucr import UCR, create_base_ucr
from .wire import SlipMessage, parse_slip


def render_human(msg: SlipMessage | str, ucr: Optional[UCR] = None) -> str:
    """Render a SLIP message as human-readable text.

    Example:
        >>> render_human("SLIP v3 planner exec Request Plan timeline")
        '[planner -> exec] Request Plan: "Request plan creation" (payload: timeline)'
    """
    if isinstance(msg, str):
        msg = parse_slip(msg)

    if ucr is None:
        ucr = create_base_ucr()

    anchor = ucr.get_by_force_obj(msg.force, msg.obj)
    canonical = anchor.canonical if anchor else f"{msg.force} {msg.obj}"

    parts = [f"[{msg.src} -> {msg.dst}]"]
    parts.append(f"{msg.force} {msg.obj}:")
    parts.append(f'"{canonical}"')

    if msg.is_fallback and msg.fallback_ref:
        parts.append(f"(fallback ref: {msg.fallback_ref})")
    elif msg.payload:
        parts.append(f"(payload: {' '.join(msg.payload)})")

    return " ".join(parts)


def render_log_line(msg: SlipMessage | str, ucr: Optional[UCR] = None) -> str:
    """Render structured log line for governance audit."""
    if isinstance(msg, str):
        msg = parse_slip(msg)

    if ucr is None:
        ucr = create_base_ucr()

    anchor = ucr.get_by_force_obj(msg.force, msg.obj)
    canonical = anchor.canonical if anchor else "unknown"
    payload_str = " ".join(msg.payload) if msg.payload else "-"
    ref_str = f"ref:{msg.fallback_ref}" if msg.fallback_ref else "-"

    return (
        f"{msg.src}->{msg.dst} | {msg.force} {msg.obj} | "
        f"{canonical} | payload={payload_str} | ref={ref_str}"
    )
