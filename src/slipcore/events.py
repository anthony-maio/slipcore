"""Observability events emitted out-of-band by the library."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

EventCallback = Callable[[Any], None]
_sinks: list[EventCallback] = []


def register_sink(callback: EventCallback) -> None:
    """Register an event sink callback."""
    _sinks.append(callback)


def unregister_sink(callback: EventCallback) -> None:
    """Unregister an event sink callback."""
    try:
        _sinks.remove(callback)
    except ValueError:
        pass


def emit(event: Any) -> None:
    """Emit an event to all registered sinks."""
    for sink in _sinks:
        try:
            sink(event)
        except Exception:
            pass  # Sinks must not break the caller


def clear_sinks() -> None:
    """Remove all registered sinks."""
    _sinks.clear()


@dataclass(frozen=True, slots=True)
class WireSent:
    event_type: str = "wire.sent"
    src: str = ""
    dst: str = ""
    force: str = ""
    obj: str = ""
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class WireReceived:
    event_type: str = "wire.received"
    src: str = ""
    dst: str = ""
    force: str = ""
    obj: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Quantized:
    event_type: str = "quantized"
    input_hash: str = ""
    force: str = ""
    obj: str = ""
    confidence: float = 0.0
    mode: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class FallbackEmitted:
    event_type: str = "fallback.emitted"
    ref: str = ""
    reason: str = ""
    confidence: float = 0.0
    raw_text_hash: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ExtensionProposed:
    event_type: str = "extension.proposed"
    force: str = ""
    obj: str = ""
    created_by: str = ""
    timestamp: float = field(default_factory=time.time)


class FallbackStore:
    """Dict-backed store for fallback raw text, keyed by ref.

    When a thought can't be quantized, the raw text is stored here
    and only a short ref token goes on the wire.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def store(self, text: str) -> str:
        """Store text and return a ref token."""
        ref = "ref" + uuid.uuid4().hex[:8]
        self._store[ref] = text
        return ref

    def retrieve(self, ref: str) -> Optional[str]:
        """Retrieve raw text by ref."""
        return self._store.get(ref)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
