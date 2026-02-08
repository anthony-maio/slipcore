"""Slipcore exception hierarchy."""

from __future__ import annotations


class SlipError(Exception):
    """Base exception for all slipcore errors."""


class WireParseError(SlipError):
    """Raised when wire text cannot be parsed."""

    def __init__(self, message: str, raw: str = "", position: int = -1) -> None:
        self.raw = raw
        self.position = position
        super().__init__(message)


class WireValidationError(SlipError):
    """Raised when a parsed message fails validation."""


class UCRError(SlipError):
    """Base for UCR-related errors."""


class AnchorNotFoundError(UCRError):
    """Raised when an anchor is not in the registry."""

    def __init__(self, key: str | int) -> None:
        self.key = key
        super().__init__(f"Anchor not found: {key!r}")


class UCRVersionMismatchError(UCRError):
    """Raised when UCR version/hash doesn't match."""


class PolicyViolationError(SlipError):
    """Raised when a Force-Object combination violates policy."""


class MissingExtraError(SlipError):
    """Raised when optional dependencies are not installed."""

    def __init__(self, module: str, extra: str) -> None:
        super().__init__(
            f"Optional dependency '{module}' required. "
            f"Install with: pip install slipcore[{extra}]"
        )
