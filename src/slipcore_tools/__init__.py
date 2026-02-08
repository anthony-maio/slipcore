"""Slipcore CLI and evaluation tools (requires slipcore[tools]).

This package provides CLI commands and evaluation utilities.
Install with: pip install slipcore[tools]
"""

from __future__ import annotations


def _check_deps() -> None:
    """Verify tools dependencies are installed."""
    try:
        import click  # noqa: F401
    except ImportError:
        from slipcore.errors import MissingExtraError
        raise MissingExtraError("click", "tools")


_check_deps()
