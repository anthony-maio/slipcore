"""Pluggable fallback backend for distributed LangGraph deployments."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol


class FallbackBackend(Protocol):
    """Backend contract for fallback ref storage."""

    def put(self, ref: str, text: str) -> None: ...

    def get(self, ref: str) -> str | None: ...


@dataclass
class InMemoryFallbackBackend:
    """Local backend for development/testing."""

    data: dict[str, str] = field(default_factory=dict)

    def put(self, ref: str, text: str) -> None:
        self.data[ref] = text

    def get(self, ref: str) -> str | None:
        return self.data.get(ref)


class BackendFallbackStore:
    """Fallback store adapter matching slipcore FallbackStore interface."""

    def __init__(self, backend: FallbackBackend) -> None:
        self.backend = backend

    def store(self, text: str) -> str:
        ref = "ref" + uuid.uuid4().hex[:8]
        self.backend.put(ref, text)
        return ref

    def retrieve(self, ref: str) -> str | None:
        return self.backend.get(ref)
