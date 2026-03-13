"""Set up tests."""

from pathlib import Path

import pytest


@pytest.fixture
def test_dir() -> Path:
    """Get test dir."""
    return Path(__file__).parent / "test_data"
