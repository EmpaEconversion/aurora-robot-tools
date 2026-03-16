"""Set up tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aurora_robot_tools import config


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add option to ignore gxipy module in tests."""
    parser.addoption("--mock-camera", action="store_true", default=False)


def pytest_configure(config: pytest.Config) -> None:
    """Ignore gxipy module, which needs drivers installed."""
    if config.getoption("--mock-camera", default=False):
        sys.modules["gxipy"] = MagicMock()


@pytest.fixture
def test_dir() -> Path:
    """Get test dir."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def override_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set all folders to inside tmp_path, cleaned up after every test."""
    monkeypatch.setattr(config, "DATABASE_FILEPATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_BACKUP_DIR", tmp_path / "backup")
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path / "inputs")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "IMAGE_DIR", tmp_path / "images")

    # create dirs that your code expects to exist
    (tmp_path / "backup").mkdir()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "images").mkdir()
