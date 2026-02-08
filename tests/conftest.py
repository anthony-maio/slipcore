"""Shared test fixtures for slipcore tests."""

from __future__ import annotations

import pytest

from slipcore.ucr import UCR, UCRAuthority, create_base_ucr


@pytest.fixture
def base_ucr() -> UCR:
    """Create a fresh base UCR for testing."""
    return create_base_ucr()


@pytest.fixture
def authority() -> UCRAuthority:
    """Create a test UCR authority."""
    return UCRAuthority(authority_id="test", ucr_version="3.0.0")
