from pathlib import Path

import pytest

from spectrochempy.utils.packages import get_pkg_path


class TestGetPkgPath:
    def test_get_directory_path(self):
        """Test retrieving a directory path from a package."""
        # Test with built-in package
        path = get_pkg_path("", "spectrochempy.utils")
        assert isinstance(path, Path)
        assert path.is_dir()
        assert path.name == "utils"

    def test_get_parent_when_not_dir(self):
        """Test getting parent directory when target is not a directory."""
        # Test with this test file
        path = get_pkg_path("test_packages.py", "spectrochempy.utils")
        assert isinstance(path, Path)
        # Should return the directory containing the module
        assert path.name == "utils"

    def test_relative_path(self):
        """Test with a relative path."""
        path = get_pkg_path("../utils", "spectrochempy.core")
        assert isinstance(path, Path)
        assert path.name == "utils"

    @pytest.mark.parametrize(
        "data_name, expected_suffix",
        [
            ("data/stylesheets", "stylesheets"),
            ("file.txt", ""),  # Will return parent since file.txt doesn't exist
            (".", "spectrochempy"),  # Current directory
        ],
    )
    def test_various_paths(self, data_name, expected_suffix):
        """Test with various path formats."""
        path = get_pkg_path(data_name, "spectrochempy")
        assert isinstance(path, Path)
        if expected_suffix:
            assert path.name == expected_suffix
