# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import os
import logging
import pytest
from pathlib import Path
import spectrochempy as scp
from spectrochempy.application import app
from unittest.mock import patch
import subprocess


def test_app_initialization():
    """Test basic application initialization"""
    assert app.name == "SpectroChemPy"
    assert app.running is True  # Should be True after import
    assert isinstance(app.config_dir, Path)
    assert app.config_dir.exists()


def test_version_info():
    """Test version information"""
    assert len(scp.version.split(".")) >= 3  # Should have at least major.minor.patch
    assert isinstance(scp.copyright, str)
    assert "LCS" in scp.copyright


def test_logging_levels():
    """Test logging level settings"""
    # Test default level
    initial_level = scp.get_loglevel()

    # Test setting different levels
    scp.set_loglevel("DEBUG")
    assert scp.get_loglevel() == logging.DEBUG

    scp.set_loglevel("INFO")
    assert scp.get_loglevel() == logging.INFO

    scp.set_loglevel("WARNING")
    assert scp.get_loglevel() == logging.WARNING

    # Reset to initial level
    scp.set_loglevel(initial_level)


def test_preferences():
    """Test preferences handling"""
    # Test preferences exist
    assert app.preferences is not None

    # Test plot preferences
    assert app.plot_preferences is not None

    # Test resetting preferences
    app.reset_preferences()
    assert app.preferences is not None


def test_config_directory():
    """Test config directory handling"""
    # Test with environment variable
    test_config_path = Path.home() / ".spectrochempy_test"
    os.environ["SCP_CONFIG_HOME"] = str(test_config_path)

    if test_config_path.exists():
        config_dir = app.config_dir
        assert config_dir.exists()
        assert str(test_config_path) in str(config_dir)

    # Clean up
    if "SCP_CONFIG_HOME" in os.environ:
        del os.environ["SCP_CONFIG_HOME"]


def test_datadir():
    """Test datadir functionality"""
    assert app.preferences.datadir is not None
    assert app.preferences.datadir.exists()


@pytest.mark.parametrize(
    "level,expected",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
    ],
)
def test_log_levels(level, expected):
    """Test different logging levels"""
    scp.set_loglevel(level)
    assert scp.get_loglevel() == expected


def test_get_release_date_with_git():
    """Test release date retrieval with git available"""
    from spectrochempy.application.application import _get_release_date

    date = _get_release_date()
    assert isinstance(date, str)
    assert date != "unknown"


def test_get_release_date_without_git():
    """Test release date retrieval when git is not available"""
    from spectrochempy.application.application import _get_release_date

    def mock_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    with patch("subprocess.run", side_effect=mock_run):
        date = _get_release_date()
        assert date == "unknown"


def test_get_release_date_with_error():
    """Test release date retrieval when git command fails"""
    from spectrochempy.application.application import _get_release_date

    def mock_run(*args, **kwargs):
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        result.check = False
        return result

    with patch("subprocess.run", side_effect=mock_run):
        date = _get_release_date()
        assert isinstance(date, str)


if __name__ == "__main__":
    pytest.main([__file__])
