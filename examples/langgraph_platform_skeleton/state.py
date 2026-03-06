"""Shared LangGraph state schema for Slipstream transport."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # App-level intent fields
    thought: str
    src: str
    dst: str
    payload: list[str]

    # Slipstream transport fields
    slip_wire: str
    slip_force: str
    slip_obj: str
    slip_payload: tuple[str, ...]
    slip_confidence: float
    slip_is_fallback: bool
    slip_fallback_ref: str
    slip_fallback_text: str
