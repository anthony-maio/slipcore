"""Slipcore ML extras (requires slipcore[ml]).

This package provides embedding-based quantization and ML-powered extension
learning. Install with: pip install slipcore[ml]
"""

from __future__ import annotations


def _check_deps() -> None:
    """Verify ML dependencies are installed."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        from slipcore.errors import MissingExtraError
        raise MissingExtraError("numpy", "ml")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        from slipcore.errors import MissingExtraError
        raise MissingExtraError("sentence-transformers", "ml")


_check_deps()
